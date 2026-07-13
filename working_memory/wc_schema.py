"""
memory/working_memory/wc_schema.py — Working Context Schema

Defines the canonical shape of working_context.json as Python TypedDicts
and a factory function that returns a properly-defaulted empty document.

Design rules (pre-set, not dynamic):
  INLINE  — small, hot, read every turn; always stored directly in the JSON.
  REF     — large or append-heavy; stored in a sidecar file, pointer in JSON.

This module has zero dependencies on other memory components.  It is the
single authoritative source for what working_context.json looks like.

Inline/Ref rules
================
  INLINE:
    _meta, session,
    user (all fields),
    <assistant_name> section (all fields — key is config.assistant_name),
    memory.intent / confidence / retrieval_ms / active_entities / current_entities,
    memory.tiers.must_know.items  (always small — 3-5 items max),
    memory.tiers.context.count + assoc.count  (counts only),
    tasks.active   (summaries — no trace),
    tasks.pending_delivery  (summaries — NO full content),
    tasks.delivered (id + title + delivered_at only),
    tasks.failed  (id + exit_reason + error summary),
    capabilities (all tool flags),
    events  (fixed-size ring buffer, bounded),
    notifications  (pending + reminders),
    $refs  (path strings only, tiny)

  $REF (pointer to sidecar file):
    memory.tiers.context.items
    memory.tiers.associations.items
    tasks.active[n].$trace
    tasks.pending_delivery[n].$content
    tasks.delivered[n].$content
    tasks.failed[n].$partial
"""

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Schema version — bump whenever a field is added/removed/renamed.
# Readers must check this so they can backfill missing keys safely.
# ---------------------------------------------------------------------------
SCHEMA_VERSION: str = "2.0"

# ---------------------------------------------------------------------------
# Default values — used by make_default_context() and backfill logic.
# ---------------------------------------------------------------------------

def make_default_context(
    session_id: str = "",
    prev_session_id: str = "",
    assistant_section_key: str = "assistant",
) -> Dict[str, Any]:
    """
    Return a fully-defaulted, empty working context document.

    All sections are present with their correct shapes so callers never
    need to guard against missing keys.

    Args:
        assistant_section_key: Top-level JSON key for the assistant state section.
            Comes from config.assistant_name (e.g. "assistant", "aria", "nova").
            Defaults to "assistant" — the generic name for open-source use.
    """
    doc: Dict[str, Any] = {
        # ── SCHEMA HEADER ─────────────────────────────────────────────────────
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "session_id":     session_id,
            "sequence":       0,        # incremented on every atomic section write
            "updated_at":     "",       # ISO 8601 of last write
            "updated_by":     "",       # which component wrote last
        },

        # ── SESSION ───────────────────────────────────────────────────────────
        "session": {
            "id":              session_id,
            "started_at":      "",
            "prev_session_id": prev_session_id,
            "turn_count":      0,
            "boot_state":      "cold",  # cold | warm | recovered
        },

        # ── USER STATE (all inline) ───────────────────────────────────────────
        "user": {
            "_v":                  0,        # section version (optimistic concurrency)
            "_owner":              "user_state_inferencer",
            "_updated_at":         "",
            "emotional_state":     "unknown",
            "emotional_intensity": 0.0,
            "need":                "",
            "engagement_level":    "normal",
            "current_focus":       "",
            "mentioned_entities":  [],
            "sentiment":           0.0,
        },

        # ── MEMORY CONTEXT ────────────────────────────────────────────────────
        # active_entities / current_entities / must_know.items → INLINE
        # context.items / associations.items                   → $REF
        "memory": {
            "_v":                  0,
            "_owner":              "working_mem",
            "_updated_at":         "",
            "intent":              "unknown",
            "intent_confidence":   0.0,
            "retrieval_ms":        0.0,
            "signals_fired":       [],
            "active_entities":     {},    # {entity: expiry_ms}
            "current_entities":    [],
            "emotional_baseline":  {},
            "tiers": {
                "must_know": {
                    "count": 0,
                    "items": [],          # INLINE — always small
                },
                "context": {
                    "count": 0,
                    "$ref":  "",          # REF → memory/context_tier.json
                },
                "associations": {
                    "count": 0,
                    "$ref":  "",          # REF → memory/assoc_tier.json
                },
            },
        },

        # ── TASKS ─────────────────────────────────────────────────────────────
        # Replaces AgenticWorkspace + TaskManager as a single lifecycle tracker.
        # Summaries inline; full content via $ref.
        "tasks": {
            "_v": 0,
            "active": [],           # TaskActiveSummary[]
            "pending_delivery": [], # TaskDeliverySummary[]
            "delivered": [],        # TaskDeliveredRecord[]
            "failed": [],           # TaskFailedRecord[]
        },

        # ── CAPABILITIES ──────────────────────────────────────────────────────
        # Single authoritative map of what the assistant can do right now.
        # Replaces both persona baseline + SelfModel dual description.
        "capabilities": {
            "_v":     0,
            "_owner": "self_model",
            # Populated at runtime by tool_registry.sync_with_self_model().
            # Shape per entry: {installed: bool, available: bool, last_failure?: str}
        },

        # ── EVENT RING ────────────────────────────────────────────────────────
        # Fixed-size ring buffer. All components append; Brain drains each turn.
        # Zero-poll coordination — no callbacks, no polling loops.
        "events": {
            "ring_size":     50,
            "last_read_seq": 0,     # Brain updates this after draining
            "items":         [],    # EventRecord[]
        },

        # ── NOTIFICATIONS ─────────────────────────────────────────────────────
        # Replaces AgenticWorkspace notification polling.
        "notifications": {
            "pending":   [],   # NotificationRecord[]
            "reminders": [],   # ReminderRecord[]
        },

        # ── NAMED POINTER REGISTRY ────────────────────────────────────────────
        # All $ref strings in the document resolve relative to these roots.
        "$refs": {
            "memory_tiers": "memory/tiers/",
            "task_outputs":  "tasks/",
            "task_traces":   "tasks/traces/",
            "conv_log":      "",    # set at session start
        },

        # ── LEGACY (kept for crash-recovery backward compat) ──────────────────
        # working_mem still reads these on startup; removed in a future schema bump.
        "active_entities":  {},
        "current_entities": [],
        "memories":         [],
    }

    # ── ASSISTANT STATE (all inline) — key is caller-supplied ─────────────────
    # Must be set after the dict literal since the key is a runtime value.
    doc[assistant_section_key] = {
        "_v":              0,
        "_owner":          "response_analyzer",
        "_updated_at":     "",
        "mode":            "conversational",
        "energy_level":    "normal",
        "emotional_tone":  "neutral",
        "current_focus":   "",
        "last_action":     "",
        "last_topics":     [],
        "last_commitments":[],
        "last_questions":  [],
        "current_datetime":"",
        "timezone":        "",
        "time_of_day":     "",
    }

    return doc


# ---------------------------------------------------------------------------
# Per-section inline rules (informational — used by writers to decide
# what to store directly vs. write to a sidecar file).
# ---------------------------------------------------------------------------
INLINE_FIELDS: frozenset = frozenset({
    "_meta",
    "session",
    "user",
    "assistant",   # generic default; actual key matches config.assistant_name
    "memory._meta",
    "memory.intent",
    "memory.intent_confidence",
    "memory.retrieval_ms",
    "memory.signals_fired",
    "memory.active_entities",
    "memory.current_entities",
    "memory.emotional_baseline",
    "memory.tiers.must_know.items",
    "memory.tiers.context.count",
    "memory.tiers.associations.count",
    "tasks",                    # task summaries (not content)
    "capabilities",
    "events",
    "notifications",
    "$refs",
})

REF_FIELDS: frozenset = frozenset({
    "memory.tiers.context.items",
    "memory.tiers.associations.items",
    "tasks.active.trace",
    "tasks.pending_delivery.content",
    "tasks.delivered.content",
    "tasks.failed.partial_content",
})


# ---------------------------------------------------------------------------
# Event type constants — used when appending to the event ring.
# ---------------------------------------------------------------------------
class EventType:
    TASK_COMPLETED   = "task_completed"
    TASK_FAILED      = "task_failed"
    TASK_PROGRESS    = "task_progress"
    AGENT_SPAWNED    = "agent_spawned"
    TOOL_OFFLINE     = "tool_offline"
    TOOL_ONLINE      = "tool_online"
    MODE_CHANGE      = "mode_change"
    SESSION_START    = "session_start"
    MEMORY_RETRIEVED = "memory_retrieved"
    FILE_CONFLICT    = "file_conflict"
    CAPABILITY_CHANGE = "capability_change"


# ---------------------------------------------------------------------------
# Schema backfill — ensure an old document has all new keys.
# Called by DiskContextManager.load_full() when reading an older file.
# ---------------------------------------------------------------------------

def backfill(doc: Dict[str, Any], assistant_section_key: str = "assistant") -> Dict[str, Any]:
    """
    Merge missing top-level keys from make_default_context() into doc.
    Existing values are never overwritten — only gaps are filled.
    Returns the same dict (mutated in place for efficiency).
    """
    defaults = make_default_context(assistant_section_key=assistant_section_key)
    for key, default_val in defaults.items():
        if key not in doc:
            doc[key] = default_val
        elif isinstance(default_val, dict) and isinstance(doc[key], dict):
            # One level deep — fill missing sub-keys only
            for sub_key, sub_val in default_val.items():
                if sub_key not in doc[key]:
                    doc[key][sub_key] = sub_val
    return doc
