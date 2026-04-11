"""
SOFi Working Context — Single Source of Truth for All Cognitive State

The Working Context is a live, multi-writer state document that represents
everything SOFi knows, feels, is doing, and is responsible for at any moment.

Four Pillars
============
  1. MemoryState      — Tiered LTM memories + recent conversation turns
  2. SofiState        — SOFi's own emotional state, self-awareness, env awareness
  3. UserState        — Who the user is, current state, what they need right now
  4. AgenticWorkspace — Active tasks, sub-agent outputs, reminders, proactive flags

Thread Safety
=============
  WorkingContextManager uses a single RLock for memory/sofi/user section updates.
  AgenticWorkspace has its own internal RLock so sub-agent writes never block
  the main working-memory thread and vice versa.

Writers
=======
  working_mem.py        → updates MemoryState after every router call
  WorkspaceWatcher      → reads AgenticWorkspace; marks items HANDLED
  sub-agents            → write to AgenticWorkspace via workspace.add_item()
  background timer      → updates SofiState.current_datetime every 60s
  [future] EmotionalModule → writes SofiState.emotional_tone + UserState.current_emotional_state
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from memory.config import config
from utils.logger import UniversalLogger

logger = UniversalLogger.get_logger("working_context")


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
    Everything SOFi knows from memory at this moment.

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
# PILLAR 2 — SOFI STATE
# =============================================================================

@dataclass
class SofiState:
    """
    SOFi's own cognitive and emotional state.

    Static fields (set at boot, rarely change):
        name, persona_version

    Dynamic fields (updated per turn or on a timer):
        emotional_tone, energy_level, current_mode,
        last_action, current_focus,
        current_datetime, timezone, time_of_day
    """
    # Identity (static)
    name: str            = "SOFi"
    persona_version: str = "1.0"

    # Emotional state (updated by EmotionalModule — placeholder until then)
    emotional_tone: str  = "neutral"        # calm | warm | concerned | excited | focused
    energy_level: str    = "normal"         # low | normal | high
    current_mode: str    = "conversational" # conversational | task-focused | empathetic | analytical

    # Self-awareness (updated after each SOFi response)
    last_action: str     = ""  # brief description of what she did last turn
    current_focus: str   = ""  # what she's attending to right now

    # Environmental awareness (updated by background timer every 60s)
    current_datetime: str = ""   # ISO 8601
    timezone: str         = ""
    time_of_day: str      = ""   # morning | afternoon | evening | night


# =============================================================================
# PILLAR 3 — USER STATE
# =============================================================================

@dataclass
class UserState:
    """
    Everything SOFi knows about the user right now.

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
    reminders, and alerts back to SOFi and the user.
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


class AgenticWorkspace:
    """
    Thread-safe collection of WorkspaceItems.

    Multiple sub-agents can write concurrently. The WorkspaceWatcher reads
    concurrently. All operations are protected by a single internal RLock.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._items: List[WorkspaceItem] = []

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_item(self, item: WorkspaceItem) -> str:
        """Add a new item. Returns its id."""
        with self._lock:
            self._items.append(item)
            logger.info(
                f"[workspace] + {item.type.value} '{item.title}' "
                f"notify={item.notify} priority={item.notify_priority.value}"
            )
            return item.id

    def update_item(self, item_id: str, **kwargs) -> bool:
        """
        Update fields on an existing item by id.
        Always updates `updated_at` automatically.
        Returns True if found and updated, False if not found.
        """
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    for key, value in kwargs.items():
                        if hasattr(item, key):
                            setattr(item, key, value)
                        else:
                            logger.warning(f"[workspace] Unknown field '{key}' on WorkspaceItem")
                    item.updated_at = datetime.now()
                    logger.debug(f"[workspace] Updated '{item.title}': {kwargs}")
                    return True
            logger.warning(f"[workspace] update_item: id '{item_id}' not found")
            return False

    def remove_item(self, item_id: str) -> bool:
        """Remove an item by id. Returns True if removed."""
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i.id != item_id]
            removed = len(self._items) < before
            if removed:
                logger.debug(f"[workspace] Removed item '{item_id}'")
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

    def get_by_type(self, item_type: WorkspaceItemType) -> List[WorkspaceItem]:
        with self._lock:
            return [i for i in self._items if i.type == item_type]

    def snapshot(self) -> List[Dict]:
        """Return a serialisable snapshot of all items (no lock held by caller)."""
        with self._lock:
            return [
                {
                    "id":           i.id,
                    "type":         i.type.value,
                    "title":        i.title,
                    "description":  i.description,
                    "status":       i.status.value,
                    "progress":     i.progress,
                    "notify":       i.notify,
                    "notify_priority": i.notify_priority.value,
                    "due_at":       i.due_at.isoformat() if i.due_at else None,
                    "source_agent": i.source_agent,
                    "updated_at":   i.updated_at.isoformat(),
                    "metadata":     i.metadata,
                }
                for i in self._items
            ]


# =============================================================================
# WORKING CONTEXT  (the assembled document)
# =============================================================================

@dataclass
class WorkingContext:
    """
    SOFi's complete cognitive state at a single point in time.
    This is what every system reads from. It is assembled by WorkingContextManager.
    """
    memory:    MemoryState
    sofi:      SofiState
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
      background timer    → update_sofi_state(current_datetime=...)
      [future] emotional  → update_sofi_state(emotional_tone=...)
                            update_user_state(current_emotional_state=...)
      sub-agents          → workspace.add_item() / workspace.update_item()
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        # Initialise all four pillars
        self._memory    = MemoryState()
        self._sofi      = SofiState(
            name=config.user_id,   # placeholder — SOFi's name comes from persona config
        )
        self._user      = UserState(
            user_id=config.user_id,
            name=config.user_id,
        )
        self._workspace = AgenticWorkspace()   # has its own internal lock

        # Kick off the background datetime timer immediately
        self._start_datetime_timer()

        logger.info("WorkingContextManager initialised")

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
        logger.debug(
            f"[wc] Memory updated: must={len(must_know or [])} "
            f"ctx={len(context or [])} assoc={len(associations or [])}"
        )

    # ── SOFi section update ───────────────────────────────────────────────────

    def update_sofi_state(self, **kwargs) -> None:
        """
        Update any field(s) on SofiState.
        e.g. update_sofi_state(emotional_tone="concerned", current_mode="empathetic")
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._sofi, key):
                    setattr(self._sofi, key, value)
                else:
                    logger.warning(f"[wc] Unknown SofiState field: '{key}'")

    # ── User section update ───────────────────────────────────────────────────

    def update_user_state(self, **kwargs) -> None:
        """
        Update any field(s) on UserState.
        e.g. update_user_state(mentioned_entities=["Alice", "project"])
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._user, key):
                    setattr(self._user, key, value)
                else:
                    logger.warning(f"[wc] Unknown UserState field: '{key}'")

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
                sofi=deepcopy(self._sofi),
                user=deepcopy(self._user),
                workspace=self._workspace,   # AgenticWorkspace has its own lock
                snapshot_at=datetime.now(),
            )

    # ── Convenience getters ───────────────────────────────────────────────────

    def get_memory(self) -> MemoryState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._memory)

    def get_sofi_state(self) -> SofiState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._sofi)

    def get_user_state(self) -> UserState:
        with self._lock:
            from copy import deepcopy
            return deepcopy(self._user)

    # ── Background: environmental datetime timer ──────────────────────────────

    def _start_datetime_timer(self) -> None:
        """
        Background daemon thread that refreshes SofiState environmental
        awareness (current_datetime, timezone, time_of_day) every 60 seconds.
        """
        import threading as _threading
        t = _threading.Thread(
            target=self._datetime_loop,
            name="wc-datetime-timer",
            daemon=True,
        )
        t.start()

    def _datetime_loop(self) -> None:
        import time as _time
        while True:
            self._refresh_datetime()
            _time.sleep(60)

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

        self.update_sofi_state(
            current_datetime=now.isoformat(timespec="seconds"),
            timezone=tz_name,
            time_of_day=tod,
        )
