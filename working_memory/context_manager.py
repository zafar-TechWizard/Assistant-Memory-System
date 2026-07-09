"""
SOFi Working Context Manager — Thread-Safe Disk Persistence

Responsibilities:
  - Provide a thread-safe load() / save() interface for the working_context.json file.
  - Use an atomic write (write-to-temp-then-rename) so a crash mid-write never
    leaves a corrupt file.
  - NOT responsible for in-memory caching — that lives in WorkingMemory._state.
  - NOT polled.  WorkingMemory writes here periodically for crash recovery only.

v2 additions (working_context orchestrator):
  - load_full()     — returns the full schema-v2 document, backfilling any
                      missing keys from defaults so readers never KeyError.
  - save_section()  — atomically updates one named section and increments the
                      global sequence counter.  Other sections are untouched.
  - append_event()  — appends one record to the event ring buffer and trims
                      if it exceeds ring_size.
  - save_capabilities() — convenience wrapper for the capabilities section.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from memory.observability import observer

# Minimum structure guaranteed to exist in every loaded context (legacy compat)
_DEFAULTS: Dict[str, Any] = {
    "active_entities": {},
    "current_entities": [],
    "memories": [],
}


class WorkingContextManager:
    """
    Handles reading and writing the working_context.json file.

    Thread-safe: a single threading.Lock serialises all I/O so concurrent
    executor threads (persist + restore) do not race.

    Atomic writes: data is first written to a .tmp sibling file, then renamed
    over the real file.  On POSIX this rename is atomic; on Windows it's a
    best-effort replace (os.replace) which is also safe against partial writes.

    Section writes (save_section) load the current document, replace only the
    named section, increment _meta.sequence, and write the whole file back
    atomically — so no section can be left in a half-written state.
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._lock = Lock()
        self._ensure_file_exists()

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy API (backward compatible — used by working_mem crash recovery)
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """
        Load working context from disk.

        Returns:
            Dict with at least the keys from _DEFAULTS.  Any missing keys are
            filled in with empty defaults so callers never KeyError.
        """
        with self._lock:
            data = self._read_raw()

        # Back-fill any missing legacy top-level keys
        for key, default_val in _DEFAULTS.items():
            if key not in data:
                data[key] = type(default_val)()

        return data

    def save(self, data: Dict[str, Any]) -> None:
        """
        Atomically save working context to disk (full document replace).

        Args:
            data: The full working context dict to persist.
        """
        with self._lock:
            self._write_raw(data)

    # ──────────────────────────────────────────────────────────────────────────
    # Schema v2 API
    # ──────────────────────────────────────────────────────────────────────────

    def load_full(self) -> Dict[str, Any]:
        """
        Load the full schema-v2 document, backfilling any missing keys.

        Safe to call from multiple threads — the lock is held only for the
        file read, not for the backfill step.

        Returns:
            Complete working context dict with all schema sections present.
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()

        return backfill(data)

    def save_section(
        self,
        section: str,
        section_data: Any,
        updated_by: str = "",
    ) -> None:
        """
        Atomically update one top-level section and increment the global
        sequence counter.  All other sections are left unchanged.

        Pattern: read → mutate section → increment _meta → write.
        The lock is held for the entire read-modify-write cycle so no other
        thread can interleave a write in between.

        Args:
            section:      Top-level key in the document (e.g. "user", "sofi").
            section_data: New value for that key.
            updated_by:   Component name for the _meta.updated_by audit field.
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()
            data = backfill(data)

            data[section] = section_data

            # Update _meta
            meta = data.setdefault("_meta", {})
            meta["sequence"]   = int(meta.get("sequence", 0)) + 1
            meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["updated_by"] = updated_by or section

            # Mirror legacy fields so crash-recovery code still works
            if section == "memory":
                data["active_entities"]  = section_data.get("active_entities", {})
                data["current_entities"] = section_data.get("current_entities", [])
                # Flat memories = must_know items (for backward compat)
                tiers = section_data.get("tiers", {})
                data["memories"] = tiers.get("must_know", {}).get("items", [])

            self._write_raw(data)

    def append_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        seq: Optional[int] = None,
    ) -> int:
        """
        Append one event to the ring buffer and trim to ring_size.

        The event gets the next global sequence number (from _meta.sequence + 1)
        unless `seq` is explicitly provided (useful for testing).

        Returns the sequence number assigned to the event.
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()
            data = backfill(data)

            meta = data.setdefault("_meta", {})
            new_seq = int(meta.get("sequence", 0)) + 1
            event_seq = seq if seq is not None else new_seq

            record = {
                "seq":  event_seq,
                "type": event_type,
                "at":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **payload,
            }

            events = data.setdefault("events", {})
            ring_size = int(events.get("ring_size", 50))
            items: List[Dict] = events.setdefault("items", [])
            items.append(record)

            # Trim oldest entries if over ring_size
            if len(items) > ring_size:
                events["items"] = items[-ring_size:]

            meta["sequence"]   = new_seq
            meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["updated_by"] = f"event:{event_type}"

            self._write_raw(data)

        return event_seq

    def save_capabilities(
        self,
        caps: Dict[str, Dict[str, Any]],
        updated_by: str = "self_model",
    ) -> None:
        """
        Convenience wrapper — update the capabilities section.

        Args:
            caps: Dict mapping capability_name →
                  {installed: bool, available: bool, last_failure?: str}
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()
            data = backfill(data)

            cap_section = data.setdefault("capabilities", {})
            cap_section["_v"]     = int(cap_section.get("_v", 0)) + 1
            cap_section["_owner"] = updated_by
            cap_section["_updated_at"] = (
                datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
            # Merge new capability entries (don't blow away _v/_owner)
            for cap_name, cap_data in caps.items():
                cap_section[cap_name] = cap_data

            meta = data.setdefault("_meta", {})
            meta["sequence"]   = int(meta.get("sequence", 0)) + 1
            meta["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            meta["updated_by"] = updated_by

            self._write_raw(data)

    def mark_last_event_read(self, last_seq: int) -> None:
        """
        Update events.last_read_seq so Brain tracks which events are new.
        Called by Brain after draining the event ring at turn start.
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()
            data = backfill(data)
            data.setdefault("events", {})["last_read_seq"] = last_seq
            self._write_raw(data)

    def get_unread_events(self) -> List[Dict[str, Any]]:
        """
        Return all event records with seq > events.last_read_seq.
        Does NOT update last_read_seq — call mark_last_event_read() after processing.
        """
        from memory.working_memory.wc_schema import backfill

        with self._lock:
            data = self._read_raw()

        data = backfill(data)
        events = data.get("events", {})
        last_read = int(events.get("last_read_seq", 0))
        return [e for e in events.get("items", []) if e.get("seq", 0) > last_read]

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _read_raw(self) -> Dict[str, Any]:
        """Read the JSON file. Caller must hold self._lock."""
        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            return dict(_DEFAULTS)

    def _write_raw(self, data: Dict[str, Any]) -> None:
        """Write the JSON file atomically. Caller must hold self._lock.

        On Windows, OneDrive can lock the .tmp file during its sync scan,
        making os.replace fail with PermissionError. We retry up to 3 times
        with a short back-off, then fall back to a direct (non-atomic) write
        so the caller is never silently blocked.
        """
        import time

        tmp_path = self.file_path.with_suffix(".tmp")
        payload = json.dumps(data, indent=2, ensure_ascii=False, default=str)

        # Write the temp file (may itself be briefly locked on first attempt).
        for attempt in range(3):
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                break
            except PermissionError:
                if attempt == 2:
                    # Temp write failed; fall back to writing the target directly.
                    try:
                        self.file_path.write_text(payload, encoding="utf-8")
                    except Exception:
                        pass
                    return
                time.sleep(0.05 * (attempt + 1))

        # Atomic rename — retry on Windows lock contention.
        for attempt in range(3):
            try:
                os.replace(tmp_path, self.file_path)
                return
            except PermissionError:
                if attempt == 2:
                    # Rename failed; tmp is written correctly, copy content directly.
                    try:
                        self.file_path.write_text(payload, encoding="utf-8")
                        tmp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return
                time.sleep(0.05 * (attempt + 1))

    def _ensure_file_exists(self) -> None:
        """Create the file with the full default schema if it doesn't exist yet."""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            from memory.working_memory.wc_schema import make_default_context
            self.save(make_default_context())
