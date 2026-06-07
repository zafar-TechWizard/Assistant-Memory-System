"""
SOFi Memory Router — Zero-LLM Intent Classification & Tiered Retrieval Dispatch

Architecture Overview
=====================
  IntentClassifier  : ~0ms multi-signal regex scorer → 5 intents + co-intent detection
  MemoryRouter      : orchestrates tiered dispatch, coverage check, re-ranking, packaging
  CoverageChecker   : pure-Python post-retrieval analysis for missed signals
  TemporalParser    : relative natural-language time → (start, end) datetime window

Retrieval Cascade (per message)
=================================
  Phase A  [PARALLEL via asyncio.gather]:
    Tier 1 — Primary dispatch (intent-routed: bm25_search / spreading_activation /
             emotional / temporal). Semantic search is NOT used on the hot path —
             embeddings stay at consolidation-time only.
    Tier 2 — Budget-fill (runs CONCURRENTLY with Tier 1, not after it)

  Phase B  [SEQUENTIAL, only if needed]:
    Coverage check — pure Python, ~0ms — detects any missed entity / time signals
    Backup queries  — targeted, parallel — promotes results to must_know tier

  Phase C  [pure Python, ~0ms]:
    Normalize → deduplicate → composite-rank → split into must_know/context/associations

  Phase D  [fire-and-forget via asyncio.create_task]:
    Hebbian reinforcement — co-retrieved memories get their graph edges strengthened
    emotional baseline fetch (for EMOTIONAL intent)

Latency budget:
   Phase A: ~50-80ms (parallel, wall-clock = max of both)
   Phase B: 0ms (no miss) or 20-40ms (targeted Cypher, no vector)
   Phase C: ~0ms
   Phase D: 0ms impact (fire-and-forget)
   TOTAL:   50-120ms  ← safe inside 300ms budget

All methods on MemoryRetrievalEngine are used — mapped to tiers:
  Tier 1  : get_memories_with_connected_nodes, bm25_search,
            get_emotionally_significant_memories, get_recent_memories,
            get_memories_by_time_range
  Tier 2  : get_memories_with_weighted_relevance, get_memories_by_topic,
            get_memories_by_multiple_topics, get_most_connected_memories
  Backup  : get_memories_by_topic (entity miss),
            get_memories_by_time_range (temporal miss)
  Post-A  : get_emotions_summary (EMOTIONAL baseline)
  F&F     : reinforce_memory_connection (Hebbian)
  Special : get_strongly_connected_memories, get_memory_cluster,
            get_connected_nodes_only, get_memories_grouped_by_topics,
            get_memories_by_emotion, get_memory_statistics
            (triggered by direct caller when needed — not in normal message path)
"""

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from memory.long_term import reranker as _reranker
from memory.observability import observer

# Optional: dateparser for named day-of-week parsing ("last Tuesday")
try:
    import dateparser as _dateparser
    _HAS_DATEPARSER = True
except ImportError:
    _HAS_DATEPARSER = False


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class Intent(str, Enum):
    ENTITY    = "entity"     # "Who is Alice?" / "Tell me about X"
    FACTUAL   = "factual"    # "What was the restaurant?" / semantic Q&A
    EMOTIONAL = "emotional"  # "I'm stressed today" / emotionally-charged
    TEMPORAL  = "temporal"   # "Last Tuesday we..." / time-anchored recall
    AMBIENT   = "ambient"    # Casual chat / continuation — no LTM retrieval needed


@dataclass
class IntentResult:
    primary_intent: Intent
    confidence: float                                       # 0.0 – 1.0
    co_intents: List[Intent] = field(default_factory=list) # secondary intents above threshold
    signals_fired: List[str] = field(default_factory=list) # audit trail
    temporal_window: Optional[Tuple[datetime, datetime]] = None
    primary_entities: List[str] = field(default_factory=list)


@dataclass
class CoverageReport:
    """Result of the post-retrieval coverage analysis."""
    missed: List[Tuple[str, Any]] = field(default_factory=list)
    # Each entry: ("entity", "Alice") or ("temporal", (start_dt, end_dt))

    @property
    def has_misses(self) -> bool:
        return len(self.missed) > 0


@dataclass
class RoutedMemories:
    """
    Fully-packaged result of one route() cycle.

    must_know    — directly answers the current query; coverage-verified
    context      — relevant background; high composite score
    associations — graph neighbours; loosely related; lowest priority
    emotional_baseline — emotion distribution summary (EMOTIONAL intent only)
    """
    intent: Intent
    confidence: float
    must_know: List[Dict]    = field(default_factory=list)
    context: List[Dict]      = field(default_factory=list)
    associations: List[Dict] = field(default_factory=list)
    emotional_baseline: Dict = field(default_factory=dict)
    signals_fired: List[str] = field(default_factory=list)
    latency_ms: float        = 0.0

    def flat_memories(self) -> List[Dict]:
        """Flat ordered list: must_know first, then context, then associations."""
        return self.must_know + self.context + self.associations


# =============================================================================
# TEMPORAL PARSER
# =============================================================================

_DAY_NAMES = [
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
]

_DATEPARSER_SETTINGS = {
    "PREFER_DATES_FROM": "past",
    "RETURN_AS_TIMEZONE_AWARE": False,
}


def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_end(dt: datetime) -> datetime:
    return dt.replace(hour=23, minute=59, second=59, microsecond=0)


def parse_temporal_window(text: str) -> Optional[Tuple[datetime, datetime]]:
    """
    Parse natural-language time expressions into a (start, end) datetime window.

    Strategy
    --------
    1. Simple arithmetic patterns → pure datetime math (no library needed, always reliable)
    2. Named day-of-week patterns → dateparser if available, otherwise arithmetic fallback

    Returns None when no temporal signal is detected.
    """
    msg = text.lower()
    now = datetime.now()

    # ── "yesterday" ───────────────────────────────────────────────────────────
    if re.search(r'\byesterday\b', msg):
        d = now - timedelta(days=1)
        return (_day_start(d), _day_end(d))

    # ── "N days ago" ─────────────────────────────────────────────────────────
    m = re.search(r'(\d+)\s+days?\s+ago', msg)
    if m:
        d = now - timedelta(days=int(m.group(1)))
        return (_day_start(d), _day_end(d))

    # ── "N weeks ago" ────────────────────────────────────────────────────────
    m = re.search(r'(\d+)\s+weeks?\s+ago', msg)
    if m:
        anchor = now - timedelta(weeks=int(m.group(1)))
        # ± 3 day window around that week's midpoint
        return (_day_start(anchor - timedelta(days=3)), _day_end(anchor + timedelta(days=3)))

    # ── "N months ago" ────────────────────────────────────────────────────────
    m = re.search(r'(\d+)\s+months?\s+ago', msg)
    if m:
        anchor = now - timedelta(days=int(m.group(1)) * 30)
        return (_day_start(anchor - timedelta(days=7)), _day_end(anchor + timedelta(days=7)))

    # ── "last week" / "past week" ─────────────────────────────────────────────
    if re.search(r'\b(last|past)\s+week\b', msg):
        return (_day_start(now - timedelta(days=7)), _day_end(now - timedelta(days=1)))

    # ── "last month" / "past month" ───────────────────────────────────────────
    if re.search(r'\b(last|past)\s+month\b', msg):
        return (_day_start(now - timedelta(days=30)), _day_end(now - timedelta(days=1)))

    # ── "earlier today" / "this morning" ─────────────────────────────────────
    if re.search(r'\b(earlier today|this morning|this afternoon)\b', msg):
        return (_day_start(now), _day_end(now))

    # ── "recently" / "the other day" ─────────────────────────────────────────
    if re.search(r'\b(recently|the other day)\b', msg):
        return (_day_start(now - timedelta(days=2)), _day_end(now))

    # ── Named day of week: "last Tuesday", "on Monday", "that Thursday" ───────
    for i, day in enumerate(_DAY_NAMES):
        if re.search(rf'\b(last|on|that)\s+{day}\b', msg):
            if _HAS_DATEPARSER:
                try:
                    parsed = _dateparser.parse(f"last {day}", settings=_DATEPARSER_SETTINGS)
                    if parsed:
                        return (_day_start(parsed), _day_end(parsed))
                except Exception:
                    pass
            # Arithmetic fallback — find most-recent occurrence of that weekday
            days_back = (now.weekday() - i) % 7
            if days_back == 0:
                days_back = 7   # "last X" means the previous week, not today
            d = now - timedelta(days=days_back)
            return (_day_start(d), _day_end(d))

    return None


# Pre-compiled quick presence check (no expensive parse)
_TEMPORAL_QUICK_RE = re.compile(
    r'\b(yesterday|last|past|ago|recently|the other day|earlier today|this morning|'
    r'this afternoon|tomorrow|today|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday|'
    r'\d+\s+(?:days?|weeks?|months?))\b',
    re.IGNORECASE,
)


def has_temporal_signal(text: str) -> bool:
    return bool(_TEMPORAL_QUICK_RE.search(text))


# =============================================================================
# INTENT CLASSIFIER
# =============================================================================

# Pre-compiled at import time — zero runtime compilation overhead
_ENTITY_PHRASE_RE = re.compile(
    r'\b(who is|tell me about|what do (you|i) know about|remind me about|'
    r'do you remember|what happened (with|to)|describe|explain|'
    r'talk to me about|what can you tell me|have i (told|mentioned)|'
    r'did i (say|tell|mention))\b',
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(
    r'\b(what (was|is|were|happened|do|did|can)|where (is|was|did)|'
    r'when (did|was|is)|which|how (did|do|was|were|to|can)|why (did|do|am|is))\b',
    re.IGNORECASE,
)
_EMOTION_RE = re.compile(
    r'\b(stressed|stress|anxious|anxiety|happy|happiness|sad|sadness|'
    r'excited|worried|worry|frustrated|frustration|proud|depressed|'
    r'angry|anger|fear|scared|nervous|overwhelmed|feel|feeling|felt|emotion)\b',
    re.IGNORECASE,
)
_TEMPORAL_SIGNAL_RE = re.compile(
    r'\b(yesterday|last|past|ago|recently|the other day|earlier today|'
    r'\d+\s+(?:days?|weeks?|months?)|monday|tuesday|wednesday|thursday|'
    r'friday|saturday|sunday)\b',
    re.IGNORECASE,
)
_CASUAL_RE = re.compile(
    r'^(ok|okay|sure|yeah|yes|no|nope|thanks|thank you|got it|'
    r'alright|cool|nice|great|fine|understood|noted|hmm|mm|uh|ah)[\.,!?]?\s*$',
    re.IGNORECASE,
)

# Signal weights (tuned empirically)
_W_ENTITY_PHRASE  = 0.70
_W_ENTITY_ONE     = 0.30
_W_ENTITY_MULTI   = 0.50
_W_SHORT_ENTITY   = 0.55   # single entity + ≤ 4 words (e.g., "Alice?")
_W_QUESTION       = 0.50
_W_EMOTION        = 0.70
_W_TEMPORAL       = 0.70
_W_CASUAL         = 0.65

# Both intents above this threshold → run both primaries concurrently
_CO_FIRE_THRESHOLD = 0.50


class IntentClassifier:
    """
    Maps (message, entities) to an IntentResult in ~0ms using pre-compiled
    regex and entity-list scoring.

    No model. No API call. No blocking.
    """

    def classify(self, message: str, entities: List[str]) -> IntentResult:
        msg_l = message.lower().strip()
        n_words = len(message.split())
        n_entities = len(entities)

        scores: Dict[Intent, float] = {
            Intent.ENTITY:    0.0,
            Intent.FACTUAL:   0.0,
            Intent.EMOTIONAL: 0.0,
            Intent.TEMPORAL:  0.0,
            Intent.AMBIENT:   0.0,
        }
        signals: List[str] = []

        # ── Fast-path: one-word casual reply ──────────────────────────────────
        if _CASUAL_RE.match(msg_l):
            scores[Intent.AMBIENT] = _W_CASUAL
            signals.append("casual_reply")
            return self._build(scores, signals, message, entities)

        # ── Fast-path: very short + no entities + no special signal ───────────
        if n_words < 4 and n_entities == 0:
            if not _QUESTION_RE.search(msg_l) and not _EMOTION_RE.search(msg_l):
                scores[Intent.AMBIENT] += 0.45
                signals.append("short_no_entity")

        # ── Signal 1: Entity-reference phrases (strong ENTITY push) ───────────
        if _ENTITY_PHRASE_RE.search(msg_l):
            scores[Intent.ENTITY] += _W_ENTITY_PHRASE
            signals.append("entity_phrase")

        # ── Signal 2: Entity count ─────────────────────────────────────────────
        if n_entities == 1:
            scores[Intent.ENTITY] += _W_ENTITY_ONE
            signals.append("entity_count_1")
            if n_words <= 4:
                scores[Intent.ENTITY] += _W_SHORT_ENTITY
                signals.append("short_single_entity")
        elif n_entities >= 2:
            scores[Intent.ENTITY] += _W_ENTITY_MULTI
            signals.append("entity_count_multi")

        # ── Signal 3: Question words → FACTUAL (entity boosts it toward ENTITY)
        if _QUESTION_RE.search(msg_l):
            scores[Intent.FACTUAL] += _W_QUESTION
            signals.append("question_word")
            if n_entities > 0:
                # Question is *about* an entity → shift slightly toward ENTITY
                scores[Intent.ENTITY] += 0.20
                signals.append("entity_question_blend")

        # ── Signal 4: Emotion keywords ─────────────────────────────────────────
        if _EMOTION_RE.search(msg_l):
            scores[Intent.EMOTIONAL] += _W_EMOTION
            signals.append("emotion_keyword")

        # ── Signal 5: Temporal references ────────────────────────────────────
        if _TEMPORAL_SIGNAL_RE.search(msg_l):
            scores[Intent.TEMPORAL] += _W_TEMPORAL
            signals.append("temporal_ref")

        return self._build(scores, signals, message, entities)

    def _build(
        self,
        scores: Dict[Intent, float],
        signals: List[str],
        message: str,
        entities: List[str],
    ) -> IntentResult:
        winner = max(scores, key=lambda k: scores[k])
        winner_score = scores[winner]

        # If NO signal fired anywhere, default to AMBIENT (safe bypass) rather
        # than letting dict-ordering pick ENTITY arbitrarily. Prevents the
        # "Bobby" / "what do I know about X" → entity with confidence 0 case.
        if winner_score <= 0.0:
            winner = Intent.AMBIENT
            winner_score = 0.3
            signals.append("no_signals_default_ambient")

        # Secondary intents that also fired above the co-fire threshold
        co_intents = [
            intent for intent, score in scores.items()
            if intent != winner and score >= _CO_FIRE_THRESHOLD
        ]

        # Only parse temporal window if a temporal signal actually fired
        temporal_window = None
        if scores[Intent.TEMPORAL] > 0 or winner == Intent.TEMPORAL:
            temporal_window = parse_temporal_window(message)

        return IntentResult(
            primary_intent=winner,
            confidence=min(winner_score, 1.0),
            co_intents=co_intents,
            signals_fired=signals,
            temporal_window=temporal_window,
            primary_entities=entities,
        )


# =============================================================================
# COVERAGE CHECKER
# =============================================================================

class CoverageChecker:
    """
    Zero-cost post-retrieval analysis: runs on already-fetched dicts, no DB calls.
    Identifies which key signals (entities, time window) were NOT addressed by
    Phase A results so targeted backup queries can fill the gaps.
    """

    def check(
        self,
        results: List[Dict],
        entities: List[str],
        temporal_window: Optional[Tuple[datetime, datetime]],
    ) -> CoverageReport:
        missed: List[Tuple[str, Any]] = []

        # ── Entity coverage ───────────────────────────────────────────────────
        for entity in entities:
            el = entity.lower()
            found = any(
                el in str(m.get("content",      m.get("root_content", ""))).lower()
                or el in str(m.get("person_name",  "")).lower()
                or el in str(m.get("participants", "")).lower()
                or el in str(m.get("concept",      "")).lower()
                for m in results
            )
            if not found:
                missed.append(("entity", entity))

        # ── Temporal coverage ─────────────────────────────────────────────────
        if temporal_window:
            start_ts, end_ts = temporal_window
            found = False
            for m in results:
                raw_ts = m.get("timestamp") or m.get("root_timestamp")
                if not raw_ts:
                    continue
                try:
                    if _HAS_DATEPARSER:
                        dt = _dateparser.parse(str(raw_ts), settings=_DATEPARSER_SETTINGS)
                    else:
                        dt = datetime.fromisoformat(str(raw_ts).replace("Z", ""))
                    if dt and start_ts <= dt.replace(tzinfo=None) <= end_ts:
                        found = True
                        break
                except Exception:
                    continue
            if not found:
                missed.append(("temporal", temporal_window))

        return CoverageReport(missed=missed)


# =============================================================================
# MEMORY ROUTER
# =============================================================================

class MemoryRouter:
    """
    Orchestrates the full retrieval pipeline for every incoming message.

    Call route() from a background thread (via asyncio.run_coroutine_threadsafe)
    or directly from an async context.
    """

    def __init__(self, engine) -> None:
        """
        Args:
            engine: Connected MemoryRetrievalEngine instance.
        """
        self._engine     = engine
        self._classifier = IntentClassifier()
        self._coverage   = CoverageChecker()
        # Surfaced in retrieval_meta after each _package call so callers / review
        # traces can see if the confidence filter had to relax below the strict default.
        self._last_confidence_threshold: float = 0.4
        observer.info("MemoryRouter ready")

    # -------------------------------------------------------------------------
    # Public: classify only (no retrieval)
    # -------------------------------------------------------------------------

    def classify(self, message: str, entities: List[str]) -> IntentResult:
        return self._classifier.classify(message, entities)

    # -------------------------------------------------------------------------
    # Public: full pipeline
    # -------------------------------------------------------------------------

    async def route(
        self,
        message: str,
        entities: List[str],
        budget: Optional[int] = None,
    ) -> RoutedMemories:
        """
        Full pipeline: classify → dispatch (parallel) → coverage → package → reinforce.

        Args:
            message:  Raw message text.
            entities: Already-extracted entity strings (no re-parsing).
            budget:   Override auto-calculated memory budget (None = auto).

        Returns:
            RoutedMemories with must_know / context / associations tiers.
        """
        t0 = time.perf_counter()

        # ── 1. Intent classification (~0ms) ───────────────────────────────────
        ir = self._classifier.classify(message, entities)

        # ── AMBIENT fast-path: skip LTM entirely ──────────────────────────────
        if ir.primary_intent == Intent.AMBIENT and not ir.co_intents:
            return RoutedMemories(
                intent=Intent.AMBIENT,
                confidence=ir.confidence,
                signals_fired=ir.signals_fired,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        # ── 2. Dynamic budget ─────────────────────────────────────────────────
        eff_budget = budget or self._budget(ir, entities)

        # ── 2b. Fire emotional baseline early (independent of Phase A/B) ──────
        # The aggregation only needs `entities`; nothing about it depends on
        # Phase A results. Kick it off now so it overlaps with everything else.
        emotional_baseline_task: Optional[asyncio.Task] = None
        if ir.primary_intent == Intent.EMOTIONAL:
            emotional_baseline_task = asyncio.create_task(
                self._engine.get_emotions_summary(
                    topic=entities[0] if entities else None
                )
            )

        # ── 3. Phase A: Tier 1 + Tier 2 CONCURRENTLY ─────────────────────────
        tier1_res, tier2_res = await asyncio.gather(
            self._tier1(ir, message, entities, eff_budget),
            self._tier2(ir, entities, eff_budget),
            return_exceptions=True,
        )
        if isinstance(tier1_res, Exception):
            observer.warning("router tier1 failed", error=str(tier1_res))
            tier1_res = []
        if isinstance(tier2_res, Exception):
            observer.warning("router tier2 failed", error=str(tier2_res))
            tier2_res = []

        phase_a: List[Dict] = list(tier1_res) + list(tier2_res)

        # ── 4. Phase B: Coverage check + targeted backup ──────────────────────
        report = self._coverage.check(phase_a, entities, ir.temporal_window)
        backup: List[Dict] = []
        if report.has_misses:
            raw_backup = await self._backup(report, entities)
            backup = [dict(m, _coverage_source=True) for m in raw_backup]

        # ── 4b. Collect emotional baseline result (already running) ───────────
        emotional_baseline: Dict = {}
        if emotional_baseline_task is not None:
            try:
                emotional_baseline = await emotional_baseline_task
            except Exception as exc:
                observer.warning("emotional baseline failed", error=str(exc))

        # ── 5. Phase C: Normalize → deduplicate → rank → package ─────────────
        all_results = phase_a + backup
        must_know, context, associations = self._package(
            all_results, backup, eff_budget,
            entities=entities, message=message, intent=ir.primary_intent.value,
        )

        # ── 6. Phase D: Fire-and-forget side-effects ───────────────────────────
        surfaced = must_know + context + associations
        if all_results:
            asyncio.create_task(self._reinforce(all_results))
        if surfaced:
            asyncio.create_task(self._update_access_stats(surfaced))

        ms = (time.perf_counter() - t0) * 1000

        # Surface the confidence threshold that survived to retrieval as a signal —
        # makes it visible in review traces and retrieval_meta. A value below 0.4
        # means the system had to relax because too few high-confidence memories
        # were available (graph is sparse for this query).
        signals = list(ir.signals_fired)
        if self._last_confidence_threshold < 0.4:
            signals.append(f"confidence_relaxed_to_{self._last_confidence_threshold:.1f}")

        return RoutedMemories(
            intent=ir.primary_intent,
            confidence=ir.confidence,
            must_know=must_know,
            context=context,
            associations=associations,
            emotional_baseline=emotional_baseline,
            signals_fired=signals,
            latency_ms=ms,
        )

    # =========================================================================
    # BUDGET
    # =========================================================================

    def _budget(self, ir: IntentResult, entities: List[str]) -> int:
        base = {
            Intent.ENTITY:    15,
            Intent.FACTUAL:   10,
            Intent.EMOTIONAL: 10,
            Intent.TEMPORAL:  12,
            Intent.AMBIENT:   0,
        }[ir.primary_intent]

        # Extra slots per additional entity (cap to avoid runaway)
        base += min((len(entities) - 1), 4) * 4

        # Bonus when a temporal window narrowed the scope (needs date breadth)
        if ir.temporal_window:
            base += 5

        # Penalty for low confidence (don't over-fetch on uncertain intent)
        if ir.confidence < 0.4:
            base = max(5, base // 2)

        return base

    # =========================================================================
    # TIER 1 — Primary dispatch
    # =========================================================================

    async def _tier1(
        self,
        ir: IntentResult,
        message: str,
        entities: List[str],
        budget: int,
    ) -> List[Dict]:
        """
        Dispatch primary intent + any co-intents, all concurrently.
        Co-intent primaries get half the budget to avoid over-fetching.
        """
        primary_coro = self._primary(ir.primary_intent, ir, message, entities, budget)

        if not ir.co_intents:
            return await primary_coro

        co_coros = [
            self._primary(ci, ir, message, entities, max(3, budget // 2))
            for ci in ir.co_intents
        ]
        results_list = await asyncio.gather(
            primary_coro, *co_coros, return_exceptions=True
        )
        merged: List[Dict] = []
        for r in results_list:
            if not isinstance(r, Exception) and r:
                merged.extend(r)
        return merged

    async def _primary(
        self,
        intent: Intent,
        ir: IntentResult,
        message: str,
        entities: List[str],
        budget: int,
    ) -> List[Dict]:
        """Map a single intent to its primary retrieval method."""
        eng  = self._engine
        ent0 = entities[0] if entities else None
        half = max(3, budget // 2)

        if intent == Intent.ENTITY and entities:
            results = await eng._spreading_activation(
                entities=entities, intent=intent.value, budget=budget,
            )
            return self._tag_tier_hints(results)

        if intent == Intent.FACTUAL:
            # Semantic search is intentionally NOT used on the hot path —
            # embedding compute stays at consolidation time only. BM25 covers
            # this intent at zero embedding cost. The content_vector field is
            # still populated on every node for offline analysis later.
            if entities:
                return await eng.bm25_search(query_terms=entities, limit=budget)
            # No entities + no semantic fallback. Coverage check + Tier 2 may
            # still find something. If real usage shows this gap matters, add
            # a keyword-based BM25 fallback then — with actual failure cases.
            return []

        if intent == Intent.EMOTIONAL:
            # Two concurrent queries — mirrors human recency primacy:
            # 1) Recent window (last 7 days, lower threshold 0.3) — ordered by recency.
            #    "What just happened that was intense?" surfaces first.
            # 2) All-time significant (threshold 0.6) — ordered by intensity.
            #    Historically significant emotional memories as background.
            recent_task  = eng.get_recent_emotional_memories(
                days=7, topic=ent0, min_intensity=0.3, limit=budget,
            )
            intense_task = eng.get_emotionally_significant_memories(
                topic=ent0, min_emotional_intensity=0.6, limit=budget,
            )
            recent_res, intense_res = await asyncio.gather(
                recent_task, intense_task, return_exceptions=True
            )
            merged: List[Dict] = []
            if not isinstance(recent_res, Exception):
                for m in recent_res:
                    m["_recency_boosted"] = True   # heat score gives extra weight
                merged.extend(recent_res)
            if not isinstance(intense_res, Exception):
                merged.extend(intense_res)
            return merged

        if intent == Intent.TEMPORAL:
            if ir.temporal_window:
                start, end = ir.temporal_window
                return await eng.get_memories_by_time_range(
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    topic=ent0,
                    limit=budget,
                )
            return await eng.get_recent_memories(topic=ent0, days=7, limit=budget)

        return []  # AMBIENT

    # =========================================================================
    # TIER 2 — Budget fill (CONCURRENT with Tier 1)
    # =========================================================================

    async def _tier2(
        self,
        ir: IntentResult,
        entities: List[str],
        budget: int,
    ) -> List[Dict]:
        """
        Supplemental retrieval that fills remaining budget.
        Runs in PARALLEL with Tier 1 via asyncio.gather — NOT after it.
        Uses different methods from Tier 1 to maximise coverage diversity.
        """
        if ir.primary_intent == Intent.AMBIENT:
            return []

        eng  = self._engine
        ent0 = entities[0] if entities else None
        half = max(3, budget // 2)

        # ENTITY intent: BM25 runs concurrently with spreading activation from T1
        if ir.primary_intent == Intent.ENTITY and entities:
            return await eng.bm25_search(query_terms=entities, limit=half)

        # Multiple entities: BM25 OR-match across all of them
        if len(entities) > 1:
            return await eng.bm25_search(query_terms=entities, limit=half)

        # Single entity: BM25 keyword search
        if ent0:
            return await eng.bm25_search(query_terms=[ent0], limit=half)

        # No entities: use memory hubs (most connected)
        return await eng.get_most_connected_memories(limit=max(3, half))

    # =========================================================================
    # PHASE B — Coverage backup
    # =========================================================================

    async def _backup(
        self,
        report: CoverageReport,
        entities: List[str],
    ) -> List[Dict]:
        """
        Targeted, parallel backup queries for each missed signal.
        get_memories_by_topic / get_memories_by_time_range are fast keyword-only
        Cypher operations (~20-35ms) — much faster than vector search.
        """
        coros = []
        ent0 = entities[0] if entities else None

        for miss_type, miss_val in report.missed:
            if miss_type == "entity":
                coros.append(
                    self._engine.get_memories_by_topic(topic=miss_val, limit=3)
                )
            elif miss_type == "temporal":
                start, end = miss_val
                coros.append(
                    self._engine.get_memories_by_time_range(
                        start_date=start.isoformat(),
                        end_date=end.isoformat(),
                        topic=ent0,
                        limit=5,
                    )
                )

        if not coros:
            return []

        raw = await asyncio.gather(*coros, return_exceptions=True)
        out: List[Dict] = []
        for r in raw:
            if not isinstance(r, Exception) and r:
                out.extend(r)
        return out

    # =========================================================================
    # PHASE C — Normalize, deduplicate, rank, package
    # =========================================================================

    def _normalize_connected(self, raw: List[Dict]) -> List[Dict]:
        """
        Flatten get_memories_with_connected_nodes results into a uniform format.

        Root nodes get a high base score and _tier_hint='must_know'.
        Connected nodes get a lower base score and _tier_hint='associations'.
        This keeps downstream packaging format-agnostic.
        """
        flat: List[Dict] = []
        for r in raw:
            if "root_id" not in r:
                flat.append(r)
                continue
            # Root node
            root: Dict[str, Any] = {
                "id":        r.get("root_id"),
                "content":   r.get("root_content"),
                "type":      r.get("root_type"),
                "timestamp": r.get("root_timestamp"),
                "score":     0.80,
                "_tier_hint": "must_know",
            }
            flat.append(root)
            # Connected nodes
            for cm in r.get("connected_memories", []):
                if cm and cm.get("id"):
                    connected = dict(cm)
                    connected.setdefault("score", 0.40)
                    connected["_tier_hint"] = "associations"
                    flat.append(connected)
        return flat

    # Temporal decay rate per intent.
    # Higher lambda = steeper decay = recent memories dominate more strongly.
    #   emotional 0.15 → 7 days ago scores exp(-1.05) ≈ 0.35× of today
    #   temporal  0.10 → 7 days ago scores exp(-0.70) ≈ 0.50× of today
    #   entity    0.05 → 7 days ago scores exp(-0.35) ≈ 0.70× of today  (default)
    #   factual   0.02 → 7 days ago scores exp(-0.14) ≈ 0.87× of today  (facts don't age)
    _INTENT_DECAY: Dict[str, float] = {
        "emotional": 0.15,
        "temporal":  0.10,
        "entity":    0.05,
        "factual":   0.02,
        "ambient":   0.05,
    }

    # Dynamic confidence filter — starts strict, relaxes when too few survive.
    # Default 0.4 cuts noisy low-confidence consolidation output. If fewer than
    # _MIN_RESULTS_AFTER_FILTER memories pass, step through progressively lower
    # thresholds until either enough survive or we hit the floor.
    # Coverage-backup memories ALWAYS pass (they were specifically retrieved
    # to fill known gaps — confidence filter shouldn't second-guess them).
    _CONFIDENCE_STEPS: List[float] = [0.4, 0.3, 0.2, 0.1, 0.0]
    _MIN_RESULTS_AFTER_FILTER: int = 3

    @staticmethod
    def _memory_confidence(m: Dict) -> float:
        """Confidence value with safe defaults. Missing field → 1.0 (treat as high-conf)."""
        c = m.get("confidence")
        if c is None:
            return 1.0
        try:
            return float(c)
        except (TypeError, ValueError):
            return 1.0

    def _apply_confidence_filter(
        self, memories: List[Dict]
    ) -> Tuple[List[Dict], float]:
        """
        Dynamic confidence-based filter with auto-relaxation.

        Algorithm:
          1. Coverage-backup memories pass unconditionally (always relevant).
          2. For each threshold in _CONFIDENCE_STEPS (starting at 0.4):
               filter regular candidates by confidence >= threshold.
               If coverage + filtered >= _MIN_RESULTS_AFTER_FILTER → return.
          3. If the strictest threshold yields too few AND the loosest still
             gives too few, return everything (graph is genuinely sparse;
             refusing memories at this point hurts more than it helps).

        Returns (filtered_memories, threshold_actually_used). The threshold is
        surfaced in retrieval_meta so the review trace shows whether the system
        had to relax — a signal that the graph is undersupplied for that query.
        """
        if not memories:
            return [], 0.0

        # Coverage memories always pass — they were targeted retrievals
        coverage = [m for m in memories if m.get("_coverage_source")]
        candidates = [m for m in memories if not m.get("_coverage_source")]

        # Try progressively looser thresholds
        for threshold in self._CONFIDENCE_STEPS:
            filtered = [m for m in candidates
                        if self._memory_confidence(m) >= threshold]
            if len(coverage) + len(filtered) >= self._MIN_RESULTS_AFTER_FILTER:
                return coverage + filtered, threshold

        # Floor — return everything. Better to show low-confidence memories than nothing.
        return memories, 0.0

    def _heat_score(self, memory: Dict, entities: List[str], intent: str = "entity") -> float:
        """
        ACT-R-inspired composite score with intent-aware temporal decay.

          base          = log(1+access_count) × exp(-λ×days_since)   λ varies by intent
          recency_bonus = up to +0.5 for memories ≤3 days old         (human primacy of recency)
          recency_tag   = +0.3 if _recency_boosted flag set           (came from recent window scan)
          assoc         = 0.3 × edge_strength
          activation    = 0.4 × activation_score
          bm25          = 0.1 × min(bm25_score, 5)
          semantic      = 0.15 × vector_score
          emotional     = 0.2 × abs(emotional_tone)

        Rationale for intent-aware decay:
          Emotional queries: yesterday's argument matters more than a historical one.
          Factual queries: when Python was released doesn't get less true over time.
        """
        access_count = int(memory.get("access_count") or 1)
        raw_ts = (
            memory.get("last_accessed")
            or memory.get("timestamp")
            or memory.get("root_timestamp")
        )
        days_since = 0
        if raw_ts:
            try:
                if _HAS_DATEPARSER:
                    dt = _dateparser.parse(str(raw_ts), settings=_DATEPARSER_SETTINGS)
                else:
                    dt = datetime.fromisoformat(str(raw_ts).replace("Z", ""))
                if dt:
                    days_since = max(0, (datetime.now() - dt.replace(tzinfo=None)).days)
            except Exception:
                pass

        decay_lambda = self._INTENT_DECAY.get(intent, 0.05)
        base = math.log(1 + access_count) * math.exp(-decay_lambda * days_since)

        # Recency bonus: memories ≤3 days old get an additive boost.
        # At 0 days: +0.5. At 1 day: +0.33. At 2 days: +0.17. At 3+ days: 0.
        # Reflects human working memory — very recent events are immediately available.
        recency_bonus = max(0.0, (1.0 - days_since / 3.0) * 0.5) if days_since <= 3 else 0.0

        # Extra boost for memories explicitly returned from the recent-window scan
        if memory.get("_recency_boosted"):
            recency_bonus += 0.3

        edge_strength = min(float(memory.get("edge_strength") or 0.0), 1.0)
        assoc_boost   = 0.3 * edge_strength

        activation = min(float(memory.get("activation_score") or 0.0), 1.0) * 0.4
        bm25       = min(float(memory.get("bm25_score") or 0.0), 5.0) * 0.1
        semantic   = float(memory.get("score") or 0.0) * 0.15
        emotional  = abs(float(memory.get("emotional_tone") or 0.0)) * 0.2

        return base + recency_bonus + assoc_boost + activation + bm25 + semantic + emotional

    def _package(
        self,
        all_results: List[Dict],
        backup: List[Dict],
        budget: int,
        entities: Optional[List[str]] = None,
        message: str = "",
        intent: str = "entity",
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        1. Dynamic confidence filter (auto-relaxes if too few memories survive)
        2. Score every survivor with ACT-R heat score
        3. Deduplicate by id
        4. Sort descending
        5. Cross-encoder rerank top-50
        6. Split tiers:
             _recency_boosted  → must_know  (recent window scan results)
             _coverage_source  → must_know  (coverage-verified)
             _tier_hint=assoc  → associations
             top 1/3 of rest   → must_know
             next 1/3          → context
             bottom 1/3        → associations
        7. Hard-cap each tier
        """
        ents = entities or []
        backup_ids: Set[str] = {
            str(m.get("id", "")) for m in backup if m.get("id")
        }

        # ── 0. Dynamic confidence filter ─────────────────────────────────────
        # Strict 0.4 by default. Auto-relax (0.3 → 0.2 → 0.1 → 0.0) if the strict
        # cut leaves us with too few memories. The threshold actually used is
        # exposed via self._last_confidence_threshold so route() can record it
        # in retrieval_meta — making it visible in review traces.
        all_results, self._last_confidence_threshold = self._apply_confidence_filter(all_results)

        seen:   Set[str]                 = set()
        scored: List[Tuple[float, Dict]] = []

        for m in all_results:
            mid = str(m.get("id") or m.get("root_id") or id(m))
            if mid in seen:
                continue
            seen.add(mid)

            score = self._heat_score(m, ents, intent=intent)
            # Coverage-verified memories always surface above un-retrieved candidates
            if mid in backup_ids or m.get("_coverage_source"):
                score = max(score, 5.0)
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [m for _, m in scored]

        # Cross-encoder rerank the top-50 for semantic precision (~10-15ms)
        if message and ordered:
            try:
                top50    = ordered[:50]
                rest_tail = ordered[50:]
                ordered  = _reranker.rerank(message, top50, top_n=len(top50)) + rest_tail
            except Exception as exc:
                observer.warning("reranker skipped", error=str(exc))

        # Separate by tier hint / coverage source / recency (O(1) ID-set checks)
        def _mid(m: Dict) -> str:
            return str(m.get("id") or m.get("root_id") or id(m))

        # Recency-boosted: came from recent-window scan → always must_know
        recency_ids: Set[str] = {
            _mid(m) for m in ordered if m.get("_recency_boosted")
        }
        coverage_ids: Set[str] = {
            _mid(m) for m in ordered
            if (m.get("_coverage_source") or _mid(m) in backup_ids)
            and _mid(m) not in recency_ids
        }
        priority_ids = recency_ids | coverage_ids
        assoc_ids: Set[str] = {
            _mid(m) for m in ordered
            if m.get("_tier_hint") == "associations"
            and _mid(m) not in priority_ids
        }

        recency_mems  = [m for m in ordered if _mid(m) in recency_ids]
        coverage_mems = [m for m in ordered if _mid(m) in coverage_ids]
        assoc_hinted  = [m for m in ordered if _mid(m) in assoc_ids]
        rest = [
            m for m in ordered
            if _mid(m) not in priority_ids and _mid(m) not in assoc_ids
        ]

        third = max(2, len(rest) // 3)
        # Recency memories always lead must_know — most mentally available first
        must_know    = recency_mems + coverage_mems + rest[:third]
        context      = rest[third : third * 2]
        associations = assoc_hinted + rest[third * 2:]

        # Hard caps to protect token budget
        cap_must  = max(3,  budget // 3)
        cap_ctx   = max(5,  budget // 2)
        cap_assoc = max(5,  budget // 3)

        return must_know[:cap_must], context[:cap_ctx], associations[:cap_assoc]

    @staticmethod
    def _tag_tier_hints(memories: List[Dict]) -> List[Dict]:
        """
        Assign _tier_hint to spreading activation results based on graph distance.
        Distance 1 (direct neighbour of seed) → context.
        Distance 2 (two hops away) → associations.
        Nodes with no distance field default to context.
        """
        for m in memories:
            if m.get("_tier_hint"):
                continue
            m["_tier_hint"] = "associations" if m.get("distance", 1) >= 2 else "context"
        return memories

    @staticmethod
    def _dedup(memories: List[Dict], seen: Optional[Set[str]] = None) -> List[Dict]:
        seen = seen or set()
        out  = []
        for m in memories:
            mid = str(m.get("id") or m.get("root_id") or id(m))
            if mid not in seen:
                seen.add(mid)
                out.append(m)
        return out

    # =========================================================================
    # PHASE D — Hebbian reinforcement (fire-and-forget)
    # =========================================================================

    async def _reinforce(self, memories: List[Dict]) -> None:
        """
        Strengthen graph edges between co-retrieved memories.
        "Neurons that fire together, wire together."

        Capped at top-5 to bound the number of Neo4j calls (O(N²) pairs).
        Each call boosts edge strength by 0.02 — small per-retrieval, cumulative.
        Called via asyncio.create_task → zero impact on response latency.
        """
        ids = [
            str(m.get("id") or m.get("root_id", ""))
            for m in memories
            if m.get("id") or m.get("root_id")
        ][:5]

        if len(ids) < 2:
            return

        tasks = [
            self._engine.reinforce_memory_connection(
                from_memory_id=ids[i],
                to_memory_id=ids[j],
                strength_boost=0.02,
            )
            for i in range(len(ids))
            for j in range(i + 1, len(ids))
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _update_access_stats(self, memories: List[Dict]) -> None:
        """
        Increment access_count and stamp last_accessed on every surfaced memory.
        ACT-R frequency dimension — without this, heat scores ignore how often
        a memory has been recalled, making frequently-useful memories rank
        identically to memories that have never been retrieved.

        Fire-and-forget via asyncio.create_task → zero latency impact.
        One batched UNWIND query, not N individual queries.
        """
        ids = [
            str(m.get("id") or m.get("root_id", ""))
            for m in memories
            if m.get("id") or m.get("root_id")
        ]
        if not ids:
            return

        query = """
        UNWIND $ids AS node_id
        MATCH (n {id: node_id})
        SET n.access_count  = coalesce(n.access_count, 0) + 1,
            n.last_accessed = $now
        """
        try:
            await self._engine.client.execute_query(
                query,
                parameters={"ids": ids, "now": datetime.now().isoformat()},
            )
        except Exception as exc:
            observer.warning("access_count update failed", error=str(exc))
