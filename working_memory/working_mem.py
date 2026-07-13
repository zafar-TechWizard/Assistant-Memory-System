"""
SOFi Working Memory (L1) — In-Memory, Thread-Safe Cognitive Workspace

Architecture
============
  State lives entirely in RAM (_state dict, protected by threading.RLock).
  Disk is used ONLY for crash recovery (write periodically, never read polled).

  Lifecycle of a single turn:
    1. memory_manager.observe()  →  run_in_executor(reactive_processing)
       ├─ entity extraction      (~5-20 ms, spaCy warm)
       ├─ update active entities  (in-memory, ~0 ms)
       ├─ LTM retrieval for NEW entities  (async bridge → Neo4j ~50-150 ms)
       ├─ merge memories          (in-memory)
       ├─ log conversation        (async, fire-and-forget)
       └─ persist state to disk   (async, fire-and-forget)

    2. memory_manager.get_context()  →  working_memory.get_working_context()
       └─ threading.Event.wait(timeout)  →  read snapshot under RLock

  Total observed latency budget:
    Entity extraction + Neo4j retrieval = ~70-170 ms  (well under 300 ms)

Key Design Decisions
====================
- NO disk polling.  get_working_context() waits on a threading.Event, not a
  file read loop.
- Async bridge via asyncio.run_coroutine_threadsafe() for Neo4j calls from the
  background thread without running a second event loop.
- Entity expiry:  active entities older than entity_expiry_minutes are pruned
  from the whiteboard on every get_working_context() call.
- Any non-AMBIENT intent triggers LTM retrieval — including EMOTIONAL queries
  with no named entities ("I'm stressed"). AMBIENT intent is the only bypass.
  Known entities get familiarity-gate treatment (0ms reuse if already loaded).
- The memory list is capped at max_total_memories to prevent unbounded growth.
"""

import asyncio
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from memory.config import config
# Renamed to avoid collision with the new centralized WorkingContextManager
from memory.working_memory.context_manager import WorkingContextManager as DiskContextManager
from memory.working_memory.ref_resolver import RefResolver
from memory.working_memory.working_context import (
    WorkingContextManager,
    ConversationTurn,
)
from memory.processing.conversationLogger import ConversationLogger
from memory.processing.entity_extractor import EntityExtractor
from memory.long_term.memory_router import RoutedMemories
from memory.observability import observer, summarize_working_context


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _ts_ms() -> int:
    """Current Unix time in milliseconds."""
    return int(time.time() * 1000)


# ===========================================================================
# WorkingMemory
# ===========================================================================

class WorkingMemory:
    """
    SOFi's active cognitive workspace.

    This is the 'whiteboard' that SOFi reads when generating a response.
    It always reflects the freshest known state:
      - Which entities are currently active (and when they expire)
      - Which entities were in the very last message
      - A list of relevant long-term memories surfaced by those entities
    """

    def __init__(
        self,
        user_id: str,
        retrieval_engine=None,
        memory_router=None,
        context_manager: Optional[WorkingContextManager] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        context_file: Optional[Path] = None,
    ):
        """
        Args:
            user_id:          The user this memory belongs to.
            retrieval_engine: MemoryRetrievalEngine instance (fallback if no router).
            memory_router:    MemoryRouter instance (preferred — enables tiered recall).
            context_manager:  WorkingContextManager — the central state document.
            event_loop:       The running asyncio event loop for the async bridge.
            context_file:     Optional override for the crash-recovery file path.
        """
        self.user_id          = user_id
        self.retrieval_engine = retrieval_engine
        self._router          = memory_router
        self._ctx_mgr         = context_manager   # central Working Context
        self._loop            = event_loop

        # ── Configuration ────────────────────────────────────────────────────
        self._entity_expiry_ms: int = config.entity_expiry_minutes * 60 * 1000
        self._timeout_s: float = config.context_retrieval_timeout_ms / 1000
        self._max_per_entity: int = config.max_memories_per_entity
        self._max_total: int = config.max_total_memories
        self._auto_cleanup: bool = config.enable_auto_cleanup

        # ── In-Memory State (the whiteboard) ─────────────────────────────────
        self._state_lock = threading.RLock()
        self._state: Dict[str, Any] = {
            "active_entities":  {},   # entity → expiry_ms
            "current_entities": [],   # entities in the most recent message
            # ── Tiered memories (filled by MemoryRouter) ──
            "memories": [],           # flat list — must_know first (backward compat)
            "tiered_memories": {
                "must_know":    [],   # directly answers the current query
                "context":      [],   # relevant background
                "associations": [],   # graph neighbours, loosely related
            },
            # ── Retrieval metadata ────────────────────────────────────────────
            "retrieval_meta": {
                "intent":       None,
                "confidence":   0.0,
                "signals_fired": [],
                "latency_ms":   0.0,
            },
            "emotional_baseline": {},
        }

        # ── Background-Processing Signal ─────────────────────────────────────
        # Set when idle/done, clear when reactive_processing is running.
        self._processing_done = threading.Event()
        self._processing_done.set()  # nothing in-flight at startup

        # ── Persistence (crash recovery only — NO polling) ────────────────────
        cfg_file = context_file or config.context_file_path
        cfg_file.parent.mkdir(parents=True, exist_ok=True)
        # Reuse the passed-in WorkingContextManager's own disk writer instead of
        # creating a second, independent DiskContextManager pointed at the same
        # file. Both objects do read-whole-file → mutate-one-key → write-whole-
        # file; two separate Lock instances serializing separately (rather than
        # one shared lock) meant _persist_state() here and
        # update_sofi_state()/update_user_state() over in WorkingContextManager
        # could race and silently clobber each other's just-written section.
        # Falls back to a private instance only if no context_manager (or one
        # without a `_disk` attribute) was supplied, e.g. in isolated tests.
        _shared_disk = getattr(context_manager, "_disk", None)
        self._disk_ctx = _shared_disk if _shared_disk is not None else DiskContextManager(cfg_file)
        # RefResolver handles all sidecar $ref I/O (context/assoc tiers, traces)
        self._ref_resolver = RefResolver(context_dir=cfg_file.parent)
        self._restore_from_disk()

        # ── Supporting Components ─────────────────────────────────────────────
        self.entity_extractor = EntityExtractor(strict_spacy=False)
        self.conversation_logger = ConversationLogger(user_id=self.user_id)

        # ── Thread Pool ───────────────────────────────────────────────────────
        # 3 workers: LTM retrieval + logging + disk persistence run concurrently
        self._executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="SOFiWM",
        )

        observer.info(
            "WorkingMemory ready",
            user=user_id,
            entity_expiry_min=config.entity_expiry_minutes,
            context_timeout_ms=config.context_retrieval_timeout_ms,
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def reactive_processing(self, role: str, content: str) -> None:
        """
        Process one incoming message.

        Designed to run inside a ThreadPoolExecutor (called via run_in_executor
        from memory_manager.observe).  Never call from the async event loop
        directly — it will block it.

        Steps:
          1. Extract entities   (~5-20 ms)
          2. Update active entities, identify NEW ones   (~0 ms)
          3. Fetch LTM memories for NEW entities via async bridge (~50-150 ms)
          4. Merge memories, prune stale ones   (~0 ms)
          5. Fire-and-forget: log conversation, persist state

        Args:
            role:    "user" | "assistant" | "system"
            content: Raw message text.
        """
        self._processing_done.clear()
        t0 = time.perf_counter()

        # Snapshot state for review (if enabled) — cheap when disabled
        wc_before: Dict[str, Any] = {}
        if observer.review_enabled:
            with self._state_lock:
                wc_before = summarize_working_context(self._state)
            observer.start_trace(role, content, wc_before)

        try:
            # ── 1. Entity extraction (two passes) ─────────────────────────────
            t_ext = time.perf_counter()

            # 1a) Current-only entities — drive active_entities tracking/expiry
            current_only: Set[str] = set(
                self.entity_extractor.extract_entities(content)
            )

            # 1b) Sliding-window entities — combine last 3 turns + current for
            #     pronoun resolution and topic anchoring.
            try:
                recent_raw = self.conversation_logger.get_conversation_history(max_turns=3)
                recent_texts = [t.get("content", "") for t in (recent_raw or [])][-3:]
            except Exception:
                recent_texts = []

            if recent_texts and hasattr(self.entity_extractor, "extract_entities_with_context"):
                context_entities: Set[str] = set(
                    self.entity_extractor.extract_entities_with_context(
                        current_message=content,
                        recent_messages=recent_texts,
                    )
                )
            else:
                context_entities = set(current_only)

            # ── 2. Active entity propagation ──────────────────────────────────
            # If current message has NO entities (e.g. "I'm stressed"), inherit
            # the 3 most-recently-active entities so retrieval still has an anchor.
            with self._state_lock:
                # Sort by expiry desc — most recently refreshed first
                _active_sorted = sorted(
                    self._state["active_entities"].items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
                active_keys = {e for e, _ in _active_sorted[:3]}

            if not current_only and not context_entities and active_keys:
                retrieval_entities = active_keys
                anchor_source = "active_propagation"
            else:
                retrieval_entities = context_entities or current_only
                anchor_source = "sliding_window" if context_entities else "current_only"

            # ── 3. Update active_entities with CURRENT message entities only ──
            with self._state_lock:
                self._state["current_entities"] = list(current_only)
                new_entities = self._update_active_entities(current_only)

            entities = retrieval_entities   # alias used downstream

            observer.stage("entity_extraction", {
                "current_only":       sorted(current_only),
                "context_entities":   sorted(context_entities),
                "retrieval_entities": sorted(retrieval_entities),
                "new_entities":       sorted(new_entities),
                "anchor_source":      anchor_source,
                "recent_turns_used":  len(recent_texts),
            }, ms=(time.perf_counter() - t_ext) * 1000)

            # ── 4. Intent classification ──────────────────────────────────────
            t_intent = time.perf_counter()
            _ir = self._router.classify(content, list(entities)) if self._router else None
            _non_ambient = _ir is not None and _ir.primary_intent.value != "ambient"

            if _ir is not None:
                observer.stage("intent_classification", {
                    "primary_intent":  _ir.primary_intent.value,
                    "confidence":      _ir.confidence,
                    "signals_fired":   _ir.signals_fired,
                    "temporal_window": str(_ir.temporal_window) if _ir.temporal_window else None,
                }, ms=(time.perf_counter() - t_intent) * 1000)

            # ── 5. LTM retrieval — router or fallback ─────────────────────────
            routed = None  # captured so we can update context_manager outside the branch
            if _non_ambient and self._loop:
                entities_list = list(entities)
                _intent_val = _ir.primary_intent.value

                # RF-Mem familiarity gate: if every entity is already warm AND
                # has loaded memories, skip LTM entirely (0ms, no DB call).
                fam_hit = bool(entities_list) and self._is_familiarity_hit(
                    entities_list, _intent_val
                )
                observer.stage("familiarity_gate", {
                    "hit":      fam_hit,
                    "entities": entities_list,
                    "reason":   "warm+loaded" if fam_hit else (
                        "always_miss_for_emotional_temporal"
                        if _intent_val in ("emotional", "temporal")
                        else "cold_or_unloaded"
                    ),
                })

                if fam_hit:
                    pass   # reuse loaded memories from previous turn
                else:
                    t_route = time.perf_counter()
                    routed = self._fetch_via_router(content, entities)
                    route_ms = (time.perf_counter() - t_route) * 1000
                    if routed is not None:
                        flat_new = routed.must_know + routed.context + routed.associations
                        # Tag memories so familiarity gate works on the next turn
                        if entities_list:
                            for idx, m in enumerate(flat_new):
                                if not m.get("_trigger_entity"):
                                    m["_trigger_entity"] = entities_list[
                                        idx % len(entities_list)
                                    ]
                        with self._state_lock:
                            self._state["tiered_memories"]["must_know"]    = routed.must_know
                            self._state["tiered_memories"]["context"]      = routed.context
                            self._state["tiered_memories"]["associations"] = routed.associations
                            self._state["memories"] = self._merge_memories(
                                self._state["memories"], flat_new
                            )
                            self._state["retrieval_meta"] = {
                                "intent":        routed.intent.value,
                                "confidence":    routed.confidence,
                                "signals_fired": routed.signals_fired,
                                "latency_ms":    routed.latency_ms,
                            }
                            self._state["emotional_baseline"] = routed.emotional_baseline

                        observer.stage("router_result", {
                            "intent":          routed.intent.value,
                            "must_know_count": len(routed.must_know),
                            "context_count":   len(routed.context),
                            "assoc_count":     len(routed.associations),
                            "router_latency_ms": routed.latency_ms,
                            "must_know_ids":   [str(m.get("id", ""))[:12] for m in routed.must_know],
                        }, ms=route_ms)
                    elif new_entities:
                        # Router returned None (timeout/error) — fallback to direct retrieval
                        fetched = self._fetch_from_longterm(new_entities)
                        observer.stage("router_fallback", {
                            "reason": "router_none",
                            "memories_fetched": len(fetched),
                        })
                        with self._state_lock:
                            self._state["memories"] = self._merge_memories(
                                self._state["memories"], fetched
                            )

            elif new_entities and self._loop:
                # No router configured — direct retrieval for new entities only
                fetched = self._fetch_from_longterm(new_entities)
                observer.stage("direct_retrieval", {
                    "memories_fetched": len(fetched),
                })
                with self._state_lock:
                    self._state["memories"] = self._merge_memories(
                        self._state["memories"], fetched
                    )

            # ── 6. Log conversation SYNCHRONOUSLY before reading back ─────────
            # Logging used to be fire-and-forget at the very end, which raced
            # with the recent_turns read below — the current turn often hadn't
            # been written to disk yet. Logging here (after retrieval, before
            # the WorkingContext update) is the simplest reliable fix.
            self._safe_log(role, content)

            # ── 7. ALWAYS push to WorkingContextManager (regardless of intent path) ──
            # This is what populates WorkingContext.memory.recent_turns. Used to
            # live inside the `if routed is not None:` branch, which meant
            # AMBIENT / familiarity-hit / router-timeout turns never refreshed
            # the conversation history surfaced to the brain layer.
            if self._ctx_mgr:
                try:
                    raw_turns = self.conversation_logger.get_conversation_history(
                        max_turns=config.working_context_recent_turns * 2
                    )
                    recent_turns = [
                        ConversationTurn(
                            role=t["role"],
                            content=t["content"],
                            timestamp=datetime.fromisoformat(
                                t["timestamp"].replace("Z", "")
                            ),
                        )
                        for t in (raw_turns or [])[-config.working_context_recent_turns:]
                    ]
                except Exception as exc:
                    observer.warning("recent_turns read failed", error=str(exc))
                    recent_turns = []

                update_kwargs: Dict[str, Any] = {"recent_turns": recent_turns}
                if routed is not None:
                    update_kwargs.update({
                        "must_know":          routed.must_know,
                        "context":            routed.context,
                        "associations":       routed.associations,
                        "retrieval_meta": {
                            "intent":        routed.intent.value,
                            "confidence":    routed.confidence,
                            "signals_fired": routed.signals_fired,
                            "latency_ms":    routed.latency_ms,
                        },
                        "emotional_baseline": routed.emotional_baseline,
                    })
                self._ctx_mgr.update_memory(**update_kwargs)

                focus_entities = list(entities) if entities else list(new_entities)
                self._ctx_mgr.update_user_state(
                    mentioned_entities=focus_entities,
                    current_focus=", ".join(focus_entities[:3]) if focus_entities else "",
                )

            # ── 8. Persist state in background (disk write, not on hot path) ──
            self._executor.submit(self._persist_state)

            elapsed = (time.perf_counter() - t0) * 1000

            # End review trace with full outcome
            if observer.review_enabled:
                with self._state_lock:
                    wc_after = summarize_working_context(self._state)
                observer.end_trace(
                    outcome={"total_ms": round(elapsed, 3)},
                    working_context_after=wc_after,
                )

        except Exception as exc:
            observer.error("reactive_processing failed", exception=exc, role=role)
            if observer.review_enabled:
                observer.stage("error", {
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                observer.end_trace(
                    outcome={"failed": True, "error": str(exc)},
                    working_context_after={},
                )

        finally:
            # Always signal completion so get_working_context() unblocks
            self._processing_done.set()

    def get_working_context(self, role: str = "", message: str = "") -> Dict[str, Any]:
        """
        Return a snapshot of the current working context (the whiteboard).

        Waits up to `context_retrieval_timeout_ms` for any in-flight
        reactive_processing to finish, then returns immediately.

        Returns:
            {
                "active_entities":  {entity_name: expiry_ms, ...},
                "current_entities": [str, ...],
                "memories":         [flat list — must_know first],
                "tiered_memories":  {"must_know": [...], "context": [...], "associations": [...]},
                "retrieval_meta":   {intent, confidence, signals_fired, latency_ms},
                "emotional_baseline": {...}
            }
        """
        completed = self._processing_done.wait(timeout=self._timeout_s)
        if not completed:
            observer.warning(
                "get_working_context timeout — returning partial",
                timeout_ms=int(self._timeout_s * 1000),
            )

        with self._state_lock:
            if self._auto_cleanup:
                self._prune_expired_entities()

            return {
                "active_entities":    dict(self._state["active_entities"]),
                "current_entities":   list(self._state["current_entities"]),
                "memories":           list(self._state["memories"]),
                "tiered_memories":    {
                    "must_know":    list(self._state["tiered_memories"]["must_know"]),
                    "context":      list(self._state["tiered_memories"]["context"]),
                    "associations": list(self._state["tiered_memories"]["associations"]),
                },
                "retrieval_meta":     dict(self._state["retrieval_meta"]),
                "emotional_baseline": dict(self._state["emotional_baseline"]),
            }

    def shutdown(self) -> None:
        """
        Flush state to disk and shut down the thread pool.
        Call from memory_manager.shutdown() (sync context is fine).
        """
        try:
            self._persist_state()
        except Exception as e:
            observer.warning("shutdown persist failed", error=str(e))

        self._executor.shutdown(wait=True, cancel_futures=False)
        observer.info("WorkingMemory shutdown complete")

    # =========================================================================
    # ENTITY MANAGEMENT
    # =========================================================================

    def _update_active_entities(self, entities: Set[str]) -> Set[str]:
        """
        Refresh expiry for known entities; add new ones.

        Returns the set of BRAND-NEW entities (those not already active).
        Only new entities trigger LTM retrieval — refreshed ones are assumed
        to already have their memories in the whiteboard.

        Called with _state_lock held.
        """
        now_ms = _ts_ms()
        new_expiry = now_ms + self._entity_expiry_ms
        new_entities: Set[str] = set()

        for entity in entities:
            if entity in self._state["active_entities"]:
                self._state["active_entities"][entity] = new_expiry
            else:
                self._state["active_entities"][entity] = new_expiry
                new_entities.add(entity)

        return new_entities

    def _prune_expired_entities(self) -> None:
        """
        Remove entities past their expiry time and drop their orphaned memories.
        Called with _state_lock held.
        """
        now_ms = _ts_ms()
        expired = [
            e for e, exp in self._state["active_entities"].items()
            if now_ms > exp
        ]
        if not expired:
            return

        for entity in expired:
            del self._state["active_entities"][entity]

        # Drop memories whose trigger entity is no longer active
        active_set = set(self._state["active_entities"].keys())
        self._state["memories"] = [
            m for m in self._state["memories"]
            # Keep if no trigger entity tagged, or if trigger is still active
            if not m.get("_trigger_entity")
            or m["_trigger_entity"] in active_set
        ]

    # =========================================================================
    # LONG-TERM MEMORY RETRIEVAL — Router Bridge + Legacy Fallback
    # =========================================================================

    def _fetch_via_router(
        self,
        message: str,
        entities: Set[str],
    ) -> Optional[RoutedMemories]:
        """
        Bridge the async MemoryRouter into this synchronous background thread.

        Uses asyncio.run_coroutine_threadsafe() to schedule router.route() on
        the main event loop without blocking it.  Waits up to 80% of the
        configured timeout budget so Phase A + coverage backup both fit.

        Returns:
            RoutedMemories on success, None if router unavailable or timed out.
        """
        if not self._router or not self._loop:
            return None

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._router.route(message, list(entities)),
                self._loop,
            )
            return future.result(timeout=self._timeout_s * 0.80)
        except TimeoutError:
            observer.warning(
                "router timeout",
                timeout_s=round(self._timeout_s * 0.80, 2),
            )
            return None
        except Exception as exc:
            observer.error("router call failed", exception=exc)
            return None

    def _fetch_from_longterm(self, entities: Set[str]) -> List[Dict[str, Any]]:
        """
        Retrieve memories from Neo4j for the given entities.

        Bridges the async retrieval engine into this synchronous thread via
        asyncio.run_coroutine_threadsafe(), scheduling work on the main event
        loop without blocking it.

        ┌─────────────────────────────────────────────────────────────────┐
        │  NOTE — Routing Intelligence (deferred)                          │
        │                                                                   │
        │  Currently uses get_memories_by_topic() as the default strategy. │
        │  The routing layer (deciding WHICH retrieval method to call and  │
        │  WHEN to call it based on intent) will be implemented in a later │
        │  phase. All retrieval methods exist in MemoryRetrievalEngine and │
        │  are ready to be called.                                          │
        └─────────────────────────────────────────────────────────────────┘

        Args:
            entities: Set of entity names to look up.

        Returns:
            Flat list of memory dicts from Neo4j (possibly empty on error).
        """
        if not self._loop or not self.retrieval_engine:
            return []

        all_memories: List[Dict[str, Any]] = []

        # Spread the time budget across entities
        per_entity_timeout = max(
            0.10,
            min(0.20, self._timeout_s * 0.6 / max(len(entities), 1)),
        )

        for entity in entities:
            try:
                # Schedule the coroutine on the main event loop
                future = asyncio.run_coroutine_threadsafe(
                    self.retrieval_engine.get_memories_by_topic(
                        topic=entity,
                        limit=self._max_per_entity,
                    ),
                    self._loop,
                )
                memories: List[Dict] = future.result(timeout=per_entity_timeout)

                # Tag each memory so we can prune it when the entity expires
                for m in memories:
                    m["_trigger_entity"] = entity

                all_memories.extend(memories)

            except TimeoutError:
                observer.warning(
                    "ltm fetch timeout",
                    entity=entity,
                    timeout_ms=int(per_entity_timeout * 1000),
                )
            except Exception as exc:
                observer.error("ltm fetch failed", exception=exc, entity=entity)

        return all_memories

    # =========================================================================
    # MEMORY MANAGEMENT
    # =========================================================================

    def _merge_memories(
        self,
        existing: List[Dict[str, Any]],
        new_memories: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Merge new memories into existing, deduplicating by memory ID/content.
        Caps the total list at max_total_memories (keeps most-recently added).
        """
        seen: Set[str] = set()
        merged: List[Dict[str, Any]] = []

        for m in existing:
            key = str(m.get("id") or m.get("content", ""))
            if key not in seen:
                seen.add(key)
                merged.append(m)

        for m in new_memories:
            key = str(m.get("id") or m.get("content", ""))
            if key not in seen:
                seen.add(key)
                merged.append(m)

        # Hard cap
        if len(merged) > self._max_total:
            merged = merged[-self._max_total:]

        return merged

    # =========================================================================
    # PERSISTENCE  (crash recovery only)
    # =========================================================================

    def _restore_from_disk(self) -> None:
        """Load previously persisted state on startup (one-time, at init)."""
        try:
            persisted = self._disk_ctx.load()
            now_ms = _ts_ms()

            # Parse active entities, discarding already-expired ones
            active: Dict[str, int] = {}
            for entity, expiry in persisted.get("active_entities", {}).items():
                # Handle both old dict format and new int format
                if isinstance(expiry, dict):
                    expiry = expiry.get("expiry_time", 0)
                    if expiry < 10_000_000_000:   # stored as seconds, not ms
                        expiry = int(expiry * 1000)
                if isinstance(expiry, (int, float)) and int(expiry) > now_ms:
                    active[entity] = int(expiry)

            with self._state_lock:
                self._state["active_entities"] = active
                self._state["current_entities"] = persisted.get(
                    "current_entities", []
                )
                self._state["memories"] = persisted.get("memories", [])

            observer.info(
                "state restored from disk",
                active_entities=len(active),
                memories=len(self._state["memories"]),
            )
        except FileNotFoundError:
            pass   # fresh start
        except Exception as exc:
            observer.warning("state restore failed", error=str(exc))

    def _persist_state(self) -> None:
        """
        Snapshot current state to the full schema-v2 memory section.

        Runs in executor thread (fire-and-forget from reactive_processing).
        Large tiers (context, associations) go to sidecar files via RefResolver;
        small/hot data (active_entities, current_entities, must_know) is inline.
        """
        try:
            with self._state_lock:
                active_entities  = dict(self._state["active_entities"])
                current_entities = list(self._state["current_entities"])
                must_know        = list(self._state["tiered_memories"]["must_know"])
                context_items    = list(self._state["tiered_memories"]["context"])
                assoc_items      = list(self._state["tiered_memories"]["associations"])
                retrieval_meta   = dict(self._state["retrieval_meta"])
                emotional_base   = dict(self._state["emotional_baseline"])

            # Write large tiers to sidecar files (always, even if empty —
            # so the $ref is always a valid pointer and readers never KeyError).
            context_ref = self._ref_resolver.write(
                context_items, "memory/tiers/context_tier.json"
            )
            assoc_ref = self._ref_resolver.write(
                assoc_items, "memory/tiers/assoc_tier.json"
            )

            memory_section = {
                "_v":                0,   # incremented by save_section
                "_owner":            "working_mem",
                "_updated_at":       "",  # set by save_section
                "intent":            retrieval_meta.get("intent") or "unknown",
                "intent_confidence": retrieval_meta.get("confidence", 0.0),
                "retrieval_ms":      retrieval_meta.get("latency_ms", 0.0),
                "signals_fired":     retrieval_meta.get("signals_fired", []),
                "active_entities":   active_entities,
                "current_entities":  current_entities,
                "emotional_baseline":emotional_base,
                "tiers": {
                    "must_know": {
                        "count": len(must_know),
                        "items": must_know,       # INLINE — always small
                    },
                    "context": {
                        "count": len(context_items),
                        "$ref":  context_ref,     # → memory/tiers/context_tier.json
                    },
                    "associations": {
                        "count": len(assoc_items),
                        "$ref":  assoc_ref,       # → memory/tiers/assoc_tier.json
                    },
                },
            }

            self._disk_ctx.save_section("memory", memory_section, updated_by="working_mem")

        except Exception as exc:
            observer.warning("disk persist failed", error=str(exc))

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _is_familiarity_hit(self, entities: List[str], intent_value: str) -> bool:
        """
        RF-Mem familiarity gate: returns True if every requested entity is already
        warm in active_entities AND at least one loaded memory is tagged with it.

        When True, the caller can skip LTM retrieval entirely (0ms, no DB call).
        Always returns False for emotional and temporal intents — those need fresh
        retrieval regardless of entity warmth.
        """
        if intent_value in ("emotional", "temporal"):
            return False
        if not entities:
            return False
        with self._state_lock:
            active = self._state["active_entities"]
            loaded = {m.get("_trigger_entity") for m in self._state["memories"]}
            return all(e in active and e in loaded for e in entities)

    def _safe_log(self, role: str, content: str) -> None:
        """Log conversation turn with full error suppression."""
        try:
            self.conversation_logger.log_message(role, content)
        except Exception as exc:
            observer.warning("conversation log write failed", error=str(exc))

    def __del__(self) -> None:
        """Best-effort cleanup on GC — do not rely on this for correctness."""
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
