"""
SOFi Working Context Manager — Thread-Safe Disk Persistence

Responsibilities:
  - Provide a thread-safe load() / save() interface for the working_context.json file.
  - Use an atomic write (write-to-temp-then-rename) so a crash mid-write never
    leaves a corrupt file.
  - NOT responsible for in-memory caching — that lives in WorkingMemory._state.
  - NOT polled.  WorkingMemory writes here periodically for crash recovery only.
"""

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from utils.logger import UniversalLogger

logger = UniversalLogger.get_logger("working_memory")

# Minimum structure guaranteed to exist in every loaded context
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
    """

    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self._lock = Lock()
        self._ensure_file_exists()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> Dict[str, Any]:
        """
        Load working context from disk.

        Returns:
            Dict with at least the keys from _DEFAULTS.  Any missing keys are
            filled in with empty defaults so callers never KeyError.

        Raises:
            FileNotFoundError: if the file genuinely does not exist yet.
            json.JSONDecodeError: if the file is corrupt.
        """
        with self._lock:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)

        # Back-fill any missing top-level keys
        for key, default_val in _DEFAULTS.items():
            if key not in data:
                data[key] = type(default_val)()

        logger.debug(
            f"[ctx_mgr] Loaded — "
            f"{len(data.get('active_entities', {}))} active entities, "
            f"{len(data.get('memories', []))} memories"
        )
        return data

    def save(self, data: Dict[str, Any]) -> None:
        """
        Atomically save working context to disk.

        Writes to a .tmp file first, then renames over the real file to
        guarantee the real file is never left in a partial state.

        Args:
            data: The full working context dict to persist.
        """
        tmp_path = self.file_path.with_suffix(".tmp")

        with self._lock:
            # Write to temp file
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

            # Atomic replace (safe on both POSIX and Windows)
            os.replace(tmp_path, self.file_path)

        logger.debug(
            f"[ctx_mgr] Saved — "
            f"{len(data.get('active_entities', {}))} active entities, "
            f"{len(data.get('memories', []))} memories"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _ensure_file_exists(self) -> None:
        """Create the file with empty defaults if it doesn't exist yet."""
        if not self.file_path.exists():
            logger.info(f"[ctx_mgr] Creating new context file: {self.file_path}")
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            # Use save() so we get atomic write from day one
            self.save(dict(_DEFAULTS))
