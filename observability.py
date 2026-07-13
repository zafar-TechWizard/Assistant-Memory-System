"""
Memory System — Observability

Two independent modes, both off by default. The singleton `observer` is the
single entry point — every component imports it and calls its methods.

  log mode    → diagnostic events (errors, warnings, lifecycle) for debugging
  review mode → structured per-query traces for behavioural analysis

When both modes are disabled, every observer call is a fast no-op (single
boolean check, no I/O). The system runs silent in production.

Usage:
    from memory.observability import observer

    # Configuration (called once by MemoryManager.setup):
    observer.configure(log=True, review=False)

    # Log mode:
    observer.error("BM25 failed", exception=exc)
    observer.warning("GLiNER not installed")
    observer.info("MemoryManager ready")

    # Review mode (correlated automatically via thread-local trace):
    observer.start_trace(role, content, wc_summary)
    observer.stage("entity_extraction", {"entities": [...]}, ms=12.4)
    observer.stage("intent_classification", {"intent": "emotional"}, ms=0.3)
    observer.end_trace(outcome={...}, wc_summary_after=...)
"""

import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# LOG WRITER (synchronous — errors must persist before any crash)
# ═══════════════════════════════════════════════════════════════════════════════

class _LogWriter:
    """Appends events to a daily rolling .log file. Thread-safe via lock."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, level: str, msg: str, fields: Dict[str, Any]) -> None:
        now = datetime.now()
        path = self.log_dir / f"{now.date().isoformat()}.log"

        line_parts = [
            now.isoformat(timespec="milliseconds"),
            level.upper().ljust(7),
            msg,
        ]
        if fields:
            # Render extra fields as compact key=value
            extras = " ".join(
                f"{k}={self._safe_repr(v)}" for k, v in fields.items()
                if k not in ("exception", "traceback")
            )
            if extras:
                line_parts.append(extras)

        line = " | ".join(line_parts)

        exc = fields.get("exception")
        tb = fields.get("traceback")
        if exc is not None:
            line += f"\n  exception: {type(exc).__name__}: {exc}"
        if tb:
            line += f"\n{tb}"

        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                # Never crash on a log write. Swallow silently.
                pass

    @staticmethod
    def _safe_repr(value: Any) -> str:
        try:
            s = repr(value)
            return s if len(s) < 200 else s[:197] + "..."
        except Exception:
            return "<unrepresentable>"


# ═══════════════════════════════════════════════════════════════════════════════
# REVIEW WRITER (asynchronous — never blocks the conversation loop)
# ═══════════════════════════════════════════════════════════════════════════════

class _ReviewWriter:
    """
    Writes per-trace JSON files on a dedicated background thread.

    One executor with one worker — preserves write order and bounds resource
    use. Submitted writes are fire-and-forget; if the queue can't keep up
    that's a signal you're producing traces faster than you can analyze them.
    """

    def __init__(self, review_dir: Path):
        self.review_dir = review_dir
        self.review_dir.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MemReview"
        )

    def submit(self, trace: Dict[str, Any]) -> None:
        try:
            self._executor.submit(self._write, trace)
        except RuntimeError:
            # Executor already shut down; drop the trace silently
            pass

    def _write(self, trace: Dict[str, Any]) -> None:
        try:
            ts_raw = trace.get("timestamp") or datetime.now().isoformat()
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", ""))
            except Exception:
                ts = datetime.now()

            day_dir = self.review_dir / ts.date().isoformat()
            day_dir.mkdir(parents=True, exist_ok=True)

            tid = str(trace.get("trace_id", uuid.uuid4()))[:8]
            filename = f"{ts.strftime('%H_%M_%S')}_{tid}.json"
            path = day_dir / filename

            with open(path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, default=self._json_default)
        except Exception:
            # Review writes never crash the caller
            pass

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    @staticmethod
    def _json_default(obj: Any) -> Any:
        # Pydantic / enum / datetime fallbacks
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "value"):
            return obj.value
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return repr(obj)


# ═══════════════════════════════════════════════════════════════════════════════
# OBSERVER (singleton, module-level instance below)
# ═══════════════════════════════════════════════════════════════════════════════

class Observer:
    """
    Two-mode observability facade. Module-level singleton: `observer`.

    Trace correlation uses thread-local storage. Each thread maintains its own
    "current trace" — components don't need to pass trace IDs through their
    call chains. The reactive_processing thread starts a trace, every stage
    along the way appends to it, the same thread ends the trace and writes.
    """

    def __init__(self) -> None:
        self._log_enabled = False
        self._review_enabled = False
        self._log_writer: Optional[_LogWriter] = None
        self._review_writer: Optional[_ReviewWriter] = None
        self._tls = threading.local()

    # ─── Configuration ─────────────────────────────────────────────────────────

    def configure(
        self,
        log: bool = False,
        review: bool = False,
        data_dir: Optional[Path] = None,
    ) -> None:
        """
        Called once by MemoryManager.setup. Idempotent — safe to call again.

        Args:
            log:      enable diagnostic logging to <BRAIN/memory/data>/logs/YYYY-MM-DD.log
            review:   enable per-query traces to <BRAIN/memory/data>/reviews/observe/YYYY-MM-DD/
            data_dir: override base data directory. If None, uses memory config:
                      <project>/BRAIN/memory/data/
        """
        if data_dir is None:
            # Resolve at call-time from central config so we always land at BRAIN/memory/data
            from memory.config import config as _mem_cfg
            data_dir = _mem_cfg.data_dir

        if log and self._log_writer is None:
            self._log_writer = _LogWriter(data_dir / "logs")
        if review and self._review_writer is None:
            self._review_writer = _ReviewWriter(data_dir / "reviews" / "observe")

        self._log_enabled = log
        self._review_enabled = review

    def shutdown(self) -> None:
        """Flush review writer cleanly. Called by MemoryManager.shutdown."""
        if self._review_writer is not None:
            self._review_writer.shutdown()

    @property
    def log_enabled(self) -> bool:
        return self._log_enabled

    @property
    def review_enabled(self) -> bool:
        return self._review_enabled

    # ─── Log mode API ──────────────────────────────────────────────────────────

    def error(
        self,
        msg: str,
        exception: Optional[BaseException] = None,
        **fields: Any,
    ) -> None:
        if not self._log_enabled or self._log_writer is None:
            return
        if exception is not None:
            fields["exception"] = exception
            fields["traceback"] = "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
        self._log_writer.write("error", msg, fields)

    def warning(self, msg: str, **fields: Any) -> None:
        if not self._log_enabled or self._log_writer is None:
            return
        self._log_writer.write("warning", msg, fields)

    def info(self, msg: str, **fields: Any) -> None:
        if not self._log_enabled or self._log_writer is None:
            return
        self._log_writer.write("info", msg, fields)

    # ─── Review mode API ───────────────────────────────────────────────────────

    def start_trace(
        self,
        role: str,
        content: str,
        working_context_before: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Start a new trace on this thread. Returns trace_id, or None if disabled."""
        if not self._review_enabled or self._review_writer is None:
            return None

        trace = {
            "trace_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "input": {"role": role, "content": content},
            "working_context_before": working_context_before or {},
            "stages": [],
        }
        self._tls.current_trace = trace
        return trace["trace_id"]

    def stage(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        ms: Optional[float] = None,
    ) -> None:
        """Append a stage event to the current thread's trace."""
        if not self._review_enabled:
            return
        trace = getattr(self._tls, "current_trace", None)
        if trace is None:
            return

        event: Dict[str, Any] = {"stage": name}
        if ms is not None:
            event["ms"] = round(ms, 3)
        if data:
            event.update(data)
        trace["stages"].append(event)

    def end_trace(
        self,
        outcome: Optional[Dict[str, Any]] = None,
        working_context_after: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finalize and submit the current thread's trace for writing."""
        if not self._review_enabled or self._review_writer is None:
            return
        trace = getattr(self._tls, "current_trace", None)
        if trace is None:
            return

        trace["working_context_after"] = working_context_after or {}
        trace["outcome"] = outcome or {}

        # Reset thread-local before submission so any error in write doesn't
        # leave a partial trace bound to this thread
        self._tls.current_trace = None
        self._review_writer.submit(trace)


# Module-level singleton ── import this everywhere ──────────────────────────────
observer = Observer()


# Helper to build a compact working-context summary for trace inclusion.
# Kept here so every component generates the same shape.
def summarize_working_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a small, review-friendly summary of working memory state.
    Includes counts, ID/name lists, and key metadata — not full memory bodies.
    """
    if not state:
        return {}

    tiered = state.get("tiered_memories") or {}
    must_know = tiered.get("must_know") or []
    context_mem = tiered.get("context") or []
    associations = tiered.get("associations") or []

    def _ids(mems):
        return [str(m.get("id") or m.get("root_id") or "?")[:12] for m in mems[:10]]

    return {
        "active_entities":  list((state.get("active_entities") or {}).keys()),
        "current_entities": list(state.get("current_entities") or []),
        "tiered_counts": {
            "must_know":    len(must_know),
            "context":      len(context_mem),
            "associations": len(associations),
        },
        "tiered_ids": {
            "must_know":    _ids(must_know),
            "context":      _ids(context_mem),
            "associations": _ids(associations),
        },
        "retrieval_meta":     state.get("retrieval_meta") or {},
        "emotional_baseline": state.get("emotional_baseline") or {},
    }
