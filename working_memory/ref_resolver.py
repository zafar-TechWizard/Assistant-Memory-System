"""
memory/working_memory/ref_resolver.py — Working Context $ref Resolver

Single utility for all file I/O that follows a $ref pointer in the
working context document.  NO component outside this module should
read or write sidecar files directly — all file access goes through here.

Design contract:
  - Every $ref string in working_context.json is a RELATIVE path.
  - RefResolver resolves it against the $refs registry in the document.
  - Reads are cached in memory (LRU, max_cache entries) for the duration
    of a single turn — the cache is invalidated on every write.
  - All writes are atomic (tmp + rename) — a crash mid-write never leaves
    a corrupt sidecar file.
  - Errors are always surfaced — never silently swallowed.

Typical usage:
  resolver = RefResolver(context_dir=Path("BRAIN/memory/data"))

  # Read context tier memories
  items = resolver.read(doc, "memory.tiers.context.$ref")

  # Write agent output and get a $ref string back
  ref = resolver.write(content, "tasks/task_001_output.txt")
  doc["tasks"]["pending_delivery"][0]["$content"] = ref
"""

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class RefResolver:
    """
    Resolves and manages $ref pointers in the working context document.

    Thread-safe: all cache access and file I/O are serialised under a
    single Lock.  This matches the single-writer model — one section
    owner writes at a time.
    """

    def __init__(
        self,
        context_dir: Path,
        max_cache: int = 32,
    ) -> None:
        """
        Args:
            context_dir: Directory that contains working_context.json.
                         Sidecar files are resolved relative to this.
            max_cache:   Maximum number of ref reads to cache per turn.
        """
        self._context_dir = Path(context_dir)
        self._lock = threading.Lock()
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._max_cache = max_cache

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def read(self, ref: str, default: Any = None) -> Any:
        """
        Read the content referenced by a $ref string.

        Args:
            ref:     Relative path stored in the $ref field
                     (e.g. "memory/tiers/context_tier.json").
            default: Returned if the file does not exist yet.

        Returns:
            Parsed JSON (list or dict) for .json files,
            raw string for .txt / .md files.
            `default` if the ref file does not exist.
        """
        if not ref:
            return default

        with self._lock:
            if ref in self._cache:
                # Move to end (LRU)
                self._cache.move_to_end(ref)
                return self._cache[ref]

        abs_path = self._resolve(ref)
        if not abs_path.exists():
            return default

        try:
            content = self._read_file(abs_path)
            with self._lock:
                self._cache[ref] = content
                if len(self._cache) > self._max_cache:
                    self._cache.popitem(last=False)
            return content
        except Exception as exc:
            raise IOError(f"RefResolver.read: failed to read '{ref}': {exc}") from exc

    def write(
        self,
        content: Any,
        ref: str,
    ) -> str:
        """
        Write content to a sidecar file atomically and return the $ref string.

        Args:
            content: Data to write. Lists/dicts → JSON. Strings → raw text.
            ref:     Relative path for the sidecar file
                     (e.g. "tasks/task_001_output.txt").

        Returns:
            The same `ref` string — caller stores this in the JSON.
        """
        abs_path = self._resolve(ref)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                if isinstance(content, (dict, list)):
                    json.dump(content, fh, indent=2, ensure_ascii=False, default=str)
                else:
                    fh.write(str(content))
            os.replace(tmp, abs_path)
        except Exception as exc:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
            raise IOError(f"RefResolver.write: failed to write '{ref}': {exc}") from exc

        # Invalidate cache entry on write
        with self._lock:
            self._cache.pop(ref, None)

        return ref

    def append_jsonl(self, record: Dict[str, Any], ref: str) -> str:
        """
        Append a single JSON record as a line to a .jsonl sidecar file.
        Used for append-only streams like agent iteration traces.

        Args:
            record: Dict to serialise as one JSON line.
            ref:    Relative path for the .jsonl file.

        Returns:
            The same `ref` string.
        """
        abs_path = self._resolve(ref)
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(abs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            raise IOError(
                f"RefResolver.append_jsonl: failed to append to '{ref}': {exc}"
            ) from exc

        with self._lock:
            self._cache.pop(ref, None)

        return ref

    def exists(self, ref: str) -> bool:
        """Return True if the sidecar file for `ref` exists on disk."""
        if not ref:
            return False
        return self._resolve(ref).exists()

    def delete(self, ref: str) -> bool:
        """
        Delete a sidecar file.  Returns True if deleted, False if not found.
        Silently ignores missing files.
        """
        if not ref:
            return False
        abs_path = self._resolve(ref)
        try:
            abs_path.unlink()
            with self._lock:
                self._cache.pop(ref, None)
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            raise IOError(
                f"RefResolver.delete: failed to delete '{ref}': {exc}"
            ) from exc

    def invalidate_cache(self) -> None:
        """Clear all cached reads.  Call at the start of each turn."""
        with self._lock:
            self._cache.clear()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _resolve(self, ref: str) -> Path:
        """
        Convert a relative $ref string to an absolute Path.
        Raises ValueError for suspicious paths (e.g. absolute or traversal).
        """
        if os.path.isabs(ref):
            raise ValueError(f"$ref must be a relative path, got: '{ref}'")
        resolved = (self._context_dir / ref).resolve()
        # Safety: resolved path must stay within context_dir
        try:
            resolved.relative_to(self._context_dir.resolve())
        except ValueError:
            raise ValueError(
                f"$ref '{ref}' resolves outside context_dir — path traversal denied"
            )
        return resolved

    def _read_file(self, abs_path: Path) -> Any:
        """Read a file and parse based on extension."""
        with open(abs_path, "r", encoding="utf-8") as fh:
            if abs_path.suffix == ".json":
                return json.load(fh)
            if abs_path.suffix == ".jsonl":
                return [json.loads(line) for line in fh if line.strip()]
            return fh.read()

    def ref_path(self, ref: str) -> Path:
        """Expose the resolved absolute Path for a ref (for callers that need the path)."""
        return self._resolve(ref)
