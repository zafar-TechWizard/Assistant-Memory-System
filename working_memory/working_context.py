"""
Working Context — Single Source of Truth for All Cognitive State

The Working Context is a live, multi-writer state document that represents
everything the assistant knows, feels, is doing, and is responsible for.

Four Pillars
============
  1. MemoryState      — Tiered LTM memories + recent conversation turns
  2. AssistantState   — The assistant's own state, self-awareness, env awareness
  3. UserState        — Who the user is, current state, what they need right now
  4. AgenticWorkspace — Active tasks, sub-agent outputs, reminders, proactive flags

Thread Safety
=============
  WorkingContextManager uses a single RLock for memory/assistant/user section updates.
  AgenticWorkspace has its own internal RLock so sub-agent writes never block
  the main working-memory thread and vice versa.

Writers
=======
  working_mem.py        → updates MemoryState after every router call
  WorkspaceWatcher      → reads AgenticWorkspace; marks items HANDLED
  sub-agents            → write to AgenticWorkspace via workspace.add_item()
  background timer      → updates AssistantState.current_datetime every 60s
  [future] EmotionalModule → writes AssistantState.emotional_tone + UserState.current_emotional_state
"""

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from memory.config import config
from memory.observability import observer


# =============================================================================
# PILLAR 1 — MEMORY STATE
# =============================================================================

@dataclass
class ConversationTurn:
    """A single message turn from the conversation log."""
    role: str            # "user" | "assistant" | "system"
    content: str
    timestamp: datetime
    entities: List[str] = field(default_factory=list)  # extracted from this turn


@dataclass
class RetrievalMeta:
    """Audit record of what the MemoryRouter did on the last retrieval."""
    intent: str          = "unknown"
    confidence: float    = 0.0
    signals_fired: List[str] = field(default_factory=list)
    latency_ms: float    = 0.0


@dataclass
class MemoryState:
    """
    Everything the assistant knows from memory at this moment.

    must_know    — directly answers the current query (coverage-verified)
    context      — relevant background
    associations — graph neighbours, loosely related
    recent_turns — last N conversation turns from ConversationLogger
    """
    must_know:    List[Dict] = field(default_factory=list)
    context:      List[Dict] = field(default_factory=list)
    associations: List[Dict] = field(default_factory=list)
    recent_turns: List[ConversationTurn] = field(default_factory=list)
    retrieval_meta: RetrievalMeta = field(default_factory=RetrievalMeta)
    emotional_baseline: Dict = field(default_factory=dict)

    def flat_memories(self) -> List[Dict]:
        """All memories ordered by tier: must_know first."""
        return self.must_know + self.context + self.associations


# =============================================================================
# PILLAR 2 — ASSISTANT STATE
# =============================================================================

@dataclass
class AssistantState:
    """
    The assistant's own cognitive and emotional state.

    Static fields (set at boot, rarely change):
        name, persona_version

    Dynamic fields (updated per turn or on a timer):
        emotional_tone, energy_level, current_mode,
        last_action, current_focus,
        current_datetime, timezone, time_of_day
    """
    # Identity (static) — name comes from config.assistant_name at boot
    name: str            = "assistant"
    persona_version: str = "1.0"

    # Emotional state (updated by EmotionalModule — placeholder until then)
    emotional_tone: str  = "neutral"        # calm | warm | concerned | excited | focused
    energy_level: str    = "normal"         # low | normal | high
    current_mode: str    = "conversational" # conversational | task-focused | empathetic | analytical

    # Self-awareness (updated after each response)
    last_action: str     = ""  # brief description of what the assistant did last turn
    current_focus: str   = ""  # what she's attending to right now

    # Environmental awareness (updated by background timer every 60s)
    current_datetime: str = ""   # ISO 8601
    timezone: str         = ""
    time_of_day: str      = ""   # morning | afternoon | evening | night

    # Response analysis (updated post-response by ResponseAnalyzer — fire-and-forget)
    last_topics_discussed: List[str] = field(default_factory=list)
    last_commitments:      List[str] = field(default_factory=list)
    last_questions_asked:  List[str] = field(default_factory=list)


# =============================================================================
# PILLAR 3 — USER STATE
# =============================================================================

@dataclass
class UserState:
    """
    Everything the assistant knows about the user right now.

    Profile fields (semi-static, loaded at boot from config / long-term memory):
        user_id, name, preferences

    Current-state fields (dynamic, updated per turn):
        current_emotional_state, emotional_intensity, current_focus,
        current_need, engagement_level, last_message_sentiment,
        mentioned_entities
    """
    # Profile (semi-static)
    user_id: str              = ""
    name: str                 = ""
    preferences: Dict         = field(default_factory=dict)

    # Current conversational state (dynamic)
    current_emotional_state: str  = "unknown"   # neutral | stressed | excited | sad | ...
    emotional_intensity: float    = 0.0         # 0.0 – 1.0
    current_focus: str            = ""          # what the user is working on / asking about
    current_need: str             = ""          # practical | emotional | informational
    engagement_level: str         = "normal"    # disengaged | normal | highly-engaged
    last_message_sentiment: float = 0.0         # −1.0 (negative) → +1.0 (positive)

    # Entities currently in focus (updated by EntityExtractor after each turn)
    mentioned_entities: List[str] = field(default_factory=list)


# =============================================================================
# PILLAR 4 — AGENTIC WORKSPACE
# =============================================================================

class WorkspaceItemType(str, Enum):
    TASK             = "task"
    REMINDER         = "reminder"
    ALARM            = "alarm"
    SUB_AGENT_OUTPUT = "sub_agent_output"
    ALERT            = "alert"


class WorkspaceItemStatus(str, Enum):
    PENDING     = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    BLOCKED     = "blocked"
    HANDLED     = "handled"   # proactive notification was fired, item cleared


class NotifyPriority(str, Enum):
    URGENT = "urgent"   # interrupt immediately — fire watcher regardless of conversation state
    NORMAL = "normal"   # wait for a conversation gap (N seconds of inactivity)
    LOW    = "low"      # surface at next user-initiated turn (queued injection)


@dataclass
class WorkspaceItem:
    """
    A single item in the Agentic Workspace.

    Sub-agents create these to communicate progress, completion, failures,
    reminders, and alerts back to the assistant and the user.
    """
    id: str                        = field(default_factory=lambda: str(uuid.uuid4()))
    type: WorkspaceItemType        = WorkspaceItemType.TASK
    title: str                     = ""
    description: str               = ""
    status: WorkspaceItemStatus    = WorkspaceItemStatus.PENDING
    progress: float                = 0.0       # 0.0 – 1.0
    notify: bool                   = False     # True = proactive surfacing needed
    notify_priority: NotifyPriority = NotifyPriority.NORMAL
    due_at: Optional[datetime]     = None      # for reminders + alarms
    created_at: datetime           = field(default_factory=datetime.now)
    updated_at: datetime           = field(default_factory=datetime.now)
    source_agent: str              = ""        # which sub-agent created this
    metadata: Dict[str, Any]       = field(default_factory=dict)


# =============================================================================
# SERIALIZATION HELPERS  (module-level so they can be used in AgenticWorkspace)
# =============================================================================

def _item_to_dict(item: "WorkspaceItem") -> Dict[str, Any]:
    """Serialise a WorkspaceItem to a JSON-safe dict."""
    return {
        "id":              item.id,
        "type":            item.type.value,
        "title":           item.title,
        "description":     item.description,
        "status":          item.status.value,
        "progress":        item.progress,
        "notify":          item.notify,
        "notify_priority": item.notify_priority.value,
        "due_at":          item.due_at.isoformat() if item.due_at else None,
        "created_at":      item.created_at.isoformat(),
        "updated_at":      item.updated_at.isoformat(),
        "source_agent":    item.source_agent,
        "metadata":        item.metadata,
    }


def _dict_to_item(d: Dict[str, Any]) -> "WorkspaceItem":
    """Reconstruct a WorkspaceItem from a dict (inverse of _item_to_dict)."""
    def _dt(val: Any) -> Optional[datetime]:
        return datetime.fromisoformat(val) if val else None

    return WorkspaceItem(
        id              = d.get("id") or str(uuid.uuid4()),
        type            = WorkspaceItemType(d.get("type", "task")),
        title           = d.get("title", ""),
        description     = d.get("description", ""),
        status          = WorkspaceItemStatus(d.get("status", "pending")),
        progress        = float(d.get("progress", 0.0)),
        notify          = bool(d.get("notify", False)),
        notify_priority = NotifyPriority(d.get("notify_priority", "normal")),
        due_at          = _dt(d.get("due_at")),
        created_at      = _dt(d.get("created_at")) or datetime.now(),
        updated_at      = _dt(d.get("updated_at")) or datetime.now(),
        source_agent    = d.get("source_agent", ""),
        metadata        = d.get("metadata") or {},
    )


# =============================================================================
# PILLAR 4 — AGENTIC WORKSPACE  (disk-backed)
# =============================================================================

class AgenticWorkspace:
    """
    Thread-safe collection of WorkspaceItems with optional disk persistence.

    Multiple sub-agents can write concurrently. The WorkspaceWatcher reads
    concurrently. All write operations are protected by a single internal RLock.

    Persistence
    -----------
    Pass persist_path to the constructor to enable disk persistence.
    Items are saved automatically after every mutation (atomic write via
    tmp-file + rename).  On startup, items from the previous session are
    restored — only PENDING and IN_PROGRESS items are reloaded so completed
    tasks don't clutter the workspace after a restart.

    A write failure never crashes a mutation — it is logged as a warning
    and the in-memory state remains authoritative.
    """

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._lock: threading.RLock     = threading.RLock()
        self._items: List[WorkspaceItem] = []
        self._persist_path: Optional[Path] = persist_path

        if persist_path is not None:
            self._load_from_disk()

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_item(self, item: WorkspaceItem) -> str:
        """Add a new item. Returns its id."""
        with self._lock:
            self._items.append(item)
        self._persist()
        return item.id

    def update_item(self, item_id: str, **kwargs) -> bool:
        """
        Update fields on an existing item by id.
        Always updates `updated_at` automatically.
        Returns True if found and updated, False if not found.
        """
        found = False
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    for key, value in kwargs.items():
                        if hasattr(item, key):
                            setattr(item, key, value)
                        else:
                            observer.warning("workspace unknown field", field=key)
                    item.updated_at = datetime.now()
                    found = True
                    break
        if found:
            self._persist()
        return found

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by id. Returns True if removed."""
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i.id != item_id]
            removed = len(self._items) < before
        if removed:
            self._persist()
        return removed

    # ── Read (return copies to avoid external mutation) ───────────────────────

    def get_all(self) -> List[WorkspaceItem]:
        with self._lock:
            return list(self._items)

    def get_active_tasks(self) -> List[WorkspaceItem]:
        """Tasks that are currently pending or in-progress."""
        with self._lock:
            return [
                i for i in self._items
                if i.type == WorkspaceItemType.TASK
                and i.status in (WorkspaceItemStatus.PENDING, WorkspaceItemStatus.IN_PROGRESS)
            ]

    def get_pending_notifications(self) -> List[WorkspaceItem]:
        """Items with notify=True that have not yet been handled."""
        with self._lock:
            return [
                i for i in self._items
                if i.notify and i.status != WorkspaceItemStatus.HANDLED
            ]

    def get_due_reminders(self, within_seconds: int = 60) -> List[WorkspaceItem]:
        """Reminders/alarms whose due_at is within the next N seconds."""
        now = datetime.now()
        with self._lock:
            return [
                i for i in self._items
                if i.type in (WorkspaceItemType.REMINDER, WorkspaceItemType.ALARM)
                and i.due_at is not None
                and i.status not in (WorkspaceItemStatus.HANDLED, WorkspaceItemStatus.COMPLETED)
                and (i.due_at - now).total_seconds() <= within_seconds
            ]

    def get_by_id(self, item_id: str) -> Optional[WorkspaceItem]:
        """Return the WorkspaceItem with the given id, or None if not found."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    return item
            return None

    def get_by_type(self, item_type: WorkspaceItemType) -> List[WorkspaceItem]:
        with self._lock:
            return [i for i in self._items if i.type == item_type]

    def snapshot(self) -> List[Dict]:
        """Return a serialisable snapshot of all items (no lock held by caller)."""
        with self._lock:
            return [_item_to_dict(i) for i in self._items]

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """
        Write all items to disk atomically (tmp + rename).

        Never raises — a disk failure is logged and the in-memory state
        remains authoritative.  Called after every successful mutation.
        """
        if self._persist_path is None:
            return
        try:
            data = {
                "_saved_at": datetime.now().isoformat(),
                "items":     self.snapshot(),
            }
            tmp = self._persist_path.with_suffix(".json.tmp")
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            tmp.replace(self._persist_path)
        except Exception as exc:
            observer.warning(
                "AgenticWorkspace persist failed",
                path=str(self._persist_path),
                error=str(exc),
            )

    def _load_from_disk(self) -> None:
        """
        Restore workspace items from the previous session.

        Only PENDING and IN_PROGRESS items are restored — completed, failed,
        and handled items are not reloaded so the workspace starts clean.
        A corrupt or missing file is silently ignored (fresh start).
        """
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            _active_statuses = {
                WorkspaceItemStatus.PENDING.value,
                WorkspaceItemStatus.IN_PROGRESS.value,
                WorkspaceItemStatus.BLOCKED.value,
            }
            loaded: List[WorkspaceItem] = []
            for d in data.get("items", []):
                if d.get("status") not in _active_statuses:
                    continue
                try:
                    loaded.append(_dict_to_item(d))
                except Exception as item_exc:
                    observer.warning(
                        "AgenticWorkspace item deserialization failed",
                        item_id=d.get("id"),
                        error=str(item_exc),
                    )

            with self._lock:
                self._items = loaded

            if loaded:
                observer.info(
                    "AgenticWorkspace restored from disk",
                    count=len(loaded),
                    path=str(self._persist_path),
                )
        except Exception as exc:
            observer.warning(
                "AgenticWorkspace load failed — starting fresh",
                path=str(self._persist_path),
                error=str(exc),
            )


# =============================================================================
# WORKING CONTEXT  (the assembled document)
# =============================================================================

@dataclass
class WorkingContext:
    """
    The assistant's complete cognitive state at a single point in time.
    This is what every system reads from. It is assembled by WorkingContextManager.
    """
    memory:    MemoryState
    assistant: AssistantState
    user:      UserState
    workspace: AgenticWorkspace
    snapshot_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# WORKING CONTEXT MANAGER
# =============================================================================

class WorkingContextManager:
    """
    Central state manager for the Working Context.

    Maintains the live state of all four pillars and provides thread-safe
    update methods for each system that writes to it.

    Writers:
      working_mem.py      → update_memory()
      entity extractor    → update_user_state(mentioned_entities=...)
      background timer    → update_assistant_state(current_datetime=...)
      user_state_inferencer → update_user_state(...)
      response_analyzer   → update_assistant_response_state(...)
      sub-agents          → workspace.add_item() / workspace.update_item()

    Disk persistence (Phase 4):
      update_assistant_state() and update_user_state() fire-and-forget their
      sections to working_context.json via DiskContextManager.save_section().
      The executor (1 worker) serialises writes without blocking callers.
    """

    def __init__(self, disk_ctx=None) -> None:
        """
        Args:
            disk_ctx: Optional DiskContextManager instance. If None, one is
                      created automatically using config.context_file_path.
                      Pass one explicitly to share the instance with WorkingMemory.
        """
        self._lock = threading.RLock()

        # The JSON section key for the assistant pillar — comes from config.assistant_name
        # so it matches what wc_schema.make_default_context() wrote to disk.
        self._assistant_section_key: str = config.assistant_name

        # Initialise all four pillars
        self._memory    = MemoryState()
        self._assistant = AssistantState(
            name=config.assistant_name,
        )
        self._user      = UserState(
            user_id=config.user_id,
            name=config.user_id,
        )

        # AgenticWorkspace — disk-backed so task state survives session restarts.
        # Placed in the same data directory as the working context file.
        _tasks_path = config.context_file_path.parent / "agentic_tasks.json"
        self._workspace = AgenticWorkspace(persist_path=_tasks_path)

        # Disk persistence: use injected DiskContextManager or create one.
        # Import lazily to avoid circular import (context_manager → working_context).
        if disk_ctx is not None:
            self._disk: Any = disk_ctx
        else:
            from memory.working_memory.context_manager import WorkingContextManager as _DiskCM
            self._disk = _DiskCM(config.context_file_path)

        # 1-worker executor: serialises async section writes without blocking callers.
        self._persist_exec = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="MemoryWCPersist"
        )

        # Kick off the background datetime timer immediately
        self._start_datetime_timer()

        observer.info("WorkingContextManager initialised")

    # ── Memory section update ─────────────────────────────────────────────────

    def update_memory(
        self,
        must_know: Optional[List[Dict]]          = None,
        context:   Optional[List[Dict]]          = None,
        associations: Optional[List[Dict]]       = None,
        recent_turns: Optional[List[ConversationTurn]] = None,
        retrieval_meta: Optional[Dict]           = None,
        emotional_baseline: Optional[Dict]       = None,
    ) -> None:
        """
        Update the MemoryState pillar.
        Called by working_mem.py after every MemoryRouter.route() call.
        Any argument left as None keeps the existing value.
        """
        with self._lock:
            if must_know    is not None: self._memory.must_know    = must_know
            if context      is not None: self._memory.context      = context
            if associations is not None: self._memory.associations = associations
            if recent_turns is not None: self._memory.recent_turns = recent_turns
            if emotional_baseline is not None:
                self._memory.emotional_baseline = emotional_baseline
            if retrieval_meta is not None:
                self._memory.retrieval_meta = RetrievalMeta(
                    intent=retrieval_meta.get("intent", "unknown"),
                    confidence=retrieval_meta.get("confidence", 0.0),
                    signals_fired=retrieval_meta.get("signals_fired", []),
                    latency_ms=retrieval_meta.get("latency_ms", 0.0),
                )

    # ── Assistant section update ──────────────────────────────────────────────

    def update_assistant_state(self, **kwargs) -> None:
        """
        Update any field(s) on AssistantState and fire-and-forget disk persist.
        e.g. update_assistant_state(emotional_tone="concerned", current_mode="empathetic")
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._assistant, key):
                    setattr(self._assistant, key, value)
                else:
                    observer.warning("unknown AssistantState field", field=key)
            assistant_snapshot = self._build_assistant_section()

        self._persist_exec.submit(
            self._disk.save_section,
            self._assistant_section_key,
            assistant_snapshot,
            "assistant_state",
        )

    # ── Assistant response-analysis update ───────────────────────────────────

    def update_assistant_response_state(
        self,
        last_topics_discussed: Optional[List[str]] = None,
        last_commitments:      Optional[List[str]] = None,
        last_questions_asked:  Optional[List[str]] = None,
    ) -> None:
        """
        Update the three post-response analysis fields on AssistantState and persist.
        Called fire-and-forget after each assistant turn.
        """
        with self._lock:
            if last_topics_discussed is not None:
                self._assistant.last_topics_discussed = last_topics_discussed
            if last_commitments is not None:
                self._assistant.last_commitments = last_commitments
            if last_questions_asked is not None:
                self._assistant.last_questions_asked = last_questions_asked
            assistant_snapshot = self._build_assistant_section()

        self._persist_exec.submit(
            self._disk.save_section,
            self._assistant_section_key,
            assistant_snapshot,
            "response_analyzer",
        )

    # ── User section update ───────────────────────────────────────────────────

    def update_user_state(self, **kwargs) -> None:
        """
        Update any field(s) on UserState and fire-and-forget disk persist.
        e.g. update_user_state(mentioned_entities=["Alice", "project"])
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._user, key):
                    setattr(self._user, key, value)
                else:
                    observer.warning("unknown UserState field", field=key)
            user_snapshot = self._build_user_section()

        self._persist_exec.submit(
            self._disk.save_section, "user", user_snapshot, "user_state_inferencer"
        )

    # ── Workspace access (direct reference — has its own lock) ────────────────

    @property
    def workspace(self) -> AgenticWorkspace:
        """Direct reference to AgenticWorkspace. Thread-safe via its own lock."""
        return self._workspace

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> WorkingContext:
        """
        Return a thread-safe point-in-time snapshot of the full Working Context.
        All callers get a deep-enough copy that they cannot mutate live state.
        """
        with self._lock:
            from copy import deepcopy
            return WorkingContext(
                memory=deepcopy(self._memory),
                assistant=deepcopy(self._assistant),
                user=deepcopy(self._user),
                workspace=self._workspace,   # AgenticWorkspace has its own lock
                snapshot_at=datetime.now(),
            )

    # ── Convenience getters ───────────────────────────────────────────────────

    def get_memory(self) -> MemoryState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._memory)

    def get_assistant_state(self) -> AssistantState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._assistant)

    def get_user_state(self) -> UserState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._user)

    # ── Section serialisers (convert in-memory dataclasses → schema-v2 dicts) ──

    def _build_assistant_section(self) -> Dict[str, Any]:
        """Build the schema-v2 assistant section from live AssistantState. Lock must be held."""
        s = self._assistant
        return {
            "_v":               0,   # incremented by save_section
            "_owner":           "response_analyzer",
            "_updated_at":      "",  # set by save_section
            "mode":             s.current_mode,
            "energy_level":     s.energy_level,
            "emotional_tone":   s.emotional_tone,
            "current_focus":    s.current_focus,
            "last_action":      s.last_action,
            "last_topics":      list(s.last_topics_discussed),
            "last_commitments": list(s.last_commitments),
            "last_questions":   list(s.last_questions_asked),
            "current_datetime": s.current_datetime,
            "timezone":         s.timezone,
            "time_of_day":      s.time_of_day,
        }

    def _build_user_section(self) -> Dict[str, Any]:
        """Build the schema-v2 'user' section from live UserState. Lock must be held."""
        u = self._user
        return {
            "_v":                  0,   # incremented by save_section
            "_owner":              "user_state_inferencer",
            "_updated_at":         "",  # set by save_section
            "emotional_state":     u.current_emotional_state,
            "emotional_intensity": u.emotional_intensity,
            "need":                u.current_need,
            "engagement_level":    u.engagement_level,
            "current_focus":       u.current_focus,
            "mentioned_entities":  list(u.mentioned_entities),
            "sentiment":           u.last_message_sentiment,
        }

    def save_capabilities(self, caps: Dict[str, Any], updated_by: str = "self_model") -> None:
        """
        Persist the capabilities section to working_context.json (fire-and-forget).
        `caps` maps capability_name → {installed: bool, available: bool, ...}
        """
        self._persist_exec.submit(
            self._disk.save_capabilities, caps, updated_by
        )

    def append_wc_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Append one event to the working_context event ring (fire-and-forget).
        Use EventType constants from wc_schema for event_type strings.
        """
        self._persist_exec.submit(
            self._disk.append_event, event_type, payload
        )

    def shutdown(self) -> None:
        """Flush pending disk writes, stop the datetime timer, shut down executor."""
        self._timer_stop.set()
        try:
            self._persist_exec.shutdown(wait=True, cancel_futures=False)
        except Exception as exc:
            observer.warning("WorkingContextManager shutdown failed", error=str(exc))

    # ── Background: environmental datetime timer ──────────────────────────────

    def _start_datetime_timer(self) -> None:
        """
        Background daemon thread that refreshes AssistantState environmental
        awareness (current_datetime, timezone, time_of_day) every 60 seconds.
        """
        import threading as _threading
        self._timer_stop = _threading.Event()
        t = _threading.Thread(
            target=self._datetime_loop,
            name="wc-datetime-timer",
            daemon=True,
        )
        t.start()

    def _datetime_loop(self) -> None:
        while not self._timer_stop.wait(60):
            try:
                self._refresh_datetime()
            except Exception:
                pass

    def _refresh_datetime(self) -> None:
        now = datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            tod = "morning"
        elif 12 <= hour < 17:
            tod = "afternoon"
        elif 17 <= hour < 21:
            tod = "evening"
        else:
            tod = "night"

        import time as _time
        tz_name = _time.tzname[0]

        self.update_assistant_state(
            current_datetime=now.isoformat(timespec="seconds"),
            timezone=tz_name,
            time_of_day=tod,
        )


