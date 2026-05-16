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

from utils.logger import UniversalLogger
from memory.config import config
# Renamed to avoid collision with the new centralized WorkingContextManager
from memory.working_memory.context_manager import WorkingContextManager as DiskContextManager
from memory.working_memory.working_context import (
    WorkingContextManager,
    ConversationTurn,
)
from memory.processing.conversationLogger import ConversationLogger
from memory.processing.entity_extractor import EntityExtractor
from memory.long_term.memory_router import RoutedMemories

logger = UniversalLogger.get_logger("working_memory")


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
        self._disk_ctx = DiskContextManager(cfg_file)  # disk persistence only
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

        logger.info(
            f"WorkingMemory ready | user={user_id} | "
            f"entity_expiry={config.entity_expiry_minutes}min | "
            f"context_timeout={config.context_retrieval_timeout_ms}ms"
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

        try:
            logger.info(f"[reactive] {role} message ({len(content)} chars)")

            # ── 1. Entity extraction (two passes) ─────────────────────────────
            # 1a) Current-only entities — drive active_entities tracking/expiry
            current_only: Set[str] = set(
                self.entity_extractor.extract_entities(content)
            )

            # 1b) Sliding-window entities — combine last 3 turns + current for
            #     pronoun resolution and topic anchoring. "she" in the current
            #     message resolves to "Sarah" if Sarah appeared 2 turns ago.
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
            # currently warm entities so retrieval still has an anchor.
            with self._state_lock:
                active_keys = set(self._state["active_entities"].keys())

            if not current_only and not context_entities and active_keys:
                logger.debug(
                    f"[reactive] No entities extracted — inheriting active: {active_keys}"
                )
                retrieval_entities = active_keys
            else:
                # Prefer sliding-window entities for richer retrieval anchoring;
                # falls back to current-only if window is empty.
                retrieval_entities = context_entities or current_only

            # ── 3. Update active_entities with CURRENT message entities only ──
            # Only entities truly in the current message refresh expiry —
            # sliding-window entities are anchors, not "currently active".
            with self._state_lock:
                self._state["current_entities"] = list(current_only)
                new_entities = self._update_active_entities(current_only)

            entities = retrieval_entities   # alias used downstream
            logger.debug(
                f"[reactive] current={current_only} context={context_entities} "
                f"retrieval={retrieval_entities} new={new_entities}"
            )

            # ── 3. LTM retrieval — router or fallback ─────────────────────────
            # Classify intent first (~0ms). Trigger LTM on any non-AMBIENT intent,
            # not only when new entities appear — EMOTIONAL/TEMPORAL queries like
            # "I'm stressed" have no new entities but still need long-term retrieval.
            _ir = self._router.classify(content, list(entities)) if self._router else None
            _non_ambient = _ir is not None and _ir.primary_intent.value != "ambient"

            if _non_ambient and self._loop:
                entities_list = list(entities)
                _intent_val = _ir.primary_intent.value

                # RF-Mem familiarity gate: if every entity is already warm AND
                # has loaded memories, skip LTM entirely (0ms, no DB call).
                if entities_list and self._is_familiarity_hit(entities_list, _intent_val):
                    logger.info(
                        f"[reactive] Familiarity hit — reusing loaded memories "
                        f"{entities_list}"
                    )
                else:
                    routed = self._fetch_via_router(content, entities)
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

                        # ── Push into WorkingContextManager (central state doc) ──
                        if self._ctx_mgr:
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
                                for t in raw_turns[-config.working_context_recent_turns:]
                            ]
                            self._ctx_mgr.update_memory(
                                must_know=routed.must_know,
                                context=routed.context,
                                associations=routed.associations,
                                recent_turns=recent_turns,
                                retrieval_meta={
                                    "intent":        routed.intent.value,
                                    "confidence":    routed.confidence,
                                    "signals_fired": routed.signals_fired,
                                    "latency_ms":    routed.latency_ms,
                                },
                                emotional_baseline=routed.emotional_baseline,
                            )
                            focus_entities = list(entities) if entities else list(new_entities)
                            self._ctx_mgr.update_user_state(
                                mentioned_entities=focus_entities,
                                current_focus=", ".join(focus_entities[:3]),
                            )

                        logger.info(
                            f"[reactive] Router: {routed.intent.value} "
                            f"must_know={len(routed.must_know)} "
                            f"context={len(routed.context)} "
                            f"assoc={len(routed.associations)} "
                            f"in {routed.latency_ms:.1f}ms"
                        )
                    elif new_entities:
                        # Router returned None (timeout/error) — fallback to direct retrieval
                        fetched = self._fetch_from_longterm(new_entities)
                        logger.info(f"[reactive] Fallback: {len(fetched)} memories retrieved")
                        with self._state_lock:
                            self._state["memories"] = self._merge_memories(
                                self._state["memories"], fetched
                            )

            elif new_entities and self._loop:
                # No router configured — direct retrieval for new entities only
                fetched = self._fetch_from_longterm(new_entities)
                logger.info(f"[reactive] Direct retrieval: {len(fetched)} memories")
                with self._state_lock:
                    self._state["memories"] = self._merge_memories(
                        self._state["memories"], fetched
                    )

            # ── 4. Non-critical, fire-and-forget ─────────────────────────────
            self._executor.submit(self._safe_log, role, content)
            self._executor.submit(self._persist_state)

            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(f"[reactive] Done in {elapsed:.1f} ms")

        except Exception as exc:
            logger.error(f"[reactive] Unexpected error: {exc}", exc_info=True)

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
            logger.warning(
                f"[context] Background processing timed out after "
                f"{self._timeout_s * 1000:.0f} ms — returning partial context"
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
        logger.info("[shutdown] Flushing working memory to disk…")
        try:
            self._persist_state()
        except Exception as e:
            logger.warning(f"[shutdown] Persist failed: {e}")

        self._executor.shutdown(wait=True, cancel_futures=False)
        logger.info("[shutdown] WorkingMemory stopped.")

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
                logger.debug(f"[entities] Refreshed '{entity}'")
            else:
                self._state["active_entities"][entity] = new_expiry
                new_entities.add(entity)
                logger.debug(f"[entities] New → '{entity}'")

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
            logger.info(f"[entities] Expired: '{entity}'")

        # Drop memories whose trigger entity is no longer active
        active_set = set(self._state["active_entities"].keys())
        before = len(self._state["memories"])
        self._state["memories"] = [
            m for m in self._state["memories"]
            # Keep if no trigger entity tagged, or if trigger is still active
            if not m.get("_trigger_entity")
            or m["_trigger_entity"] in active_set
        ]
        pruned = before - len(self._state["memories"])
        if pruned:
            logger.debug(f"[entities] Pruned {pruned} stale memories")

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
            logger.warning(
                f"[wm] Router timed out after {self._timeout_s * 0.80:.2f}s"
            )
            return None
        except Exception as exc:
            logger.warning(f"[wm] Router failed — falling back to direct retrieval: {exc}")
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
                logger.debug(f"[ltm] '{entity}' → {len(memories)} memories")

            except TimeoutError:
                logger.warning(
                    f"[ltm] Timed out retrieving '{entity}' "
                    f"({per_entity_timeout * 1000:.0f} ms limit)"
                )
            except Exception as exc:
                logger.warning(f"[ltm] Failed for '{entity}': {exc}")

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
            logger.debug(f"[memories] Capped at {self._max_total}")

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

            logger.info(
                f"[recovery] Restored {len(active)} active entities and "
                f"{len(self._state['memories'])} memories from disk"
            )
        except FileNotFoundError:
            logger.info("[recovery] No persisted state found — starting fresh")
        except Exception as exc:
            logger.warning(f"[recovery] Could not restore state: {exc} — fresh start")

    def _persist_state(self) -> None:
        """Snapshot current state to disk (runs in executor thread)."""
        try:
            with self._state_lock:
                snapshot = {
                    "active_entities": dict(self._state["active_entities"]),
                    "current_entities": list(self._state["current_entities"]),
                    "memories": list(self._state["memories"]),
                    "_persisted_at": datetime.now().isoformat(),
                }
            self._disk_ctx.save(snapshot)
        except Exception as exc:
            logger.warning(f"[persist] Disk write failed: {exc}")

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
            logger.warning(f"[log] Conversation logging failed: {exc}")

    def __del__(self) -> None:
        """Best-effort cleanup on GC — do not rely on this for correctness."""
        try:
            if hasattr(self, "_executor"):
                self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
