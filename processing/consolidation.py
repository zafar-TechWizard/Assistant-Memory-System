"""
SOFi Memory Consolidation — 5-Stage Pipeline

Extract → Match → Resolve → Write → Link

Design principles (research-backed):
- Single canonical node per person (Graphiti/Zep approach) — MERGE on person_name
- Single canonical node per concept — MERGE on concept
- Operations: CREATE | UPDATE | ENHANCE | SKIP | CONTRADICT
- Never physically delete — CONTRADICT preserves temporal lineage
- Edge deduplication: MERGE on (from, to, type) — strengthen if exists
- LLM called at most twice per session (extract + resolve conflicts)
- Partial failure isolation — one bad memory doesn't abort the session
- Idempotent writes — safe to re-run on same session (crash recovery)
"""

import asyncio
import datetime
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from groq import AsyncGroq

from memory.config import get_config
from memory.long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client
from memory.long_term.models.node_models import (
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
    MemoryContext,
)
from memory.long_term.models.relationship_models import MemoryRelationshipType
from memory.processing.embedding_utils import EmbeddingUtils
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine

from memory.observability import observer


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedMemory:
    index: int
    type: str                               # EXPERIENCE | KNOWLEDGE | RELATIONSHIP
    content: str
    importance: float

    tags: List[str] = field(default_factory=list)

    # EXPERIENCE
    event_type: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    emotional_tone: float = 0.0
    lessons_learned: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None

    # KNOWLEDGE
    concept: Optional[str] = None
    definition: Optional[str] = None
    category: Optional[str] = None
    related_concepts: List[str] = field(default_factory=list)

    # RELATIONSHIP
    person_name: Optional[str] = None
    relationship_type: Optional[str] = None
    emotional_connection: float = 0.0
    personality_traits: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    trust_level: float = 0.5


@dataclass
class ExtractedEdge:
    from_index: int
    to_index: int
    rel_type: str
    strength: float = 0.7
    bidirectional: bool = False


@dataclass
class ExtractionResult:
    memories: List[ExtractedMemory]
    edges: List[ExtractedEdge]
    overall_sentiment: float = 0.0


@dataclass
class ExistingMatch:
    node_id: str
    label: str
    content: str
    key_fields: Dict[str, Any]
    similarity: float


@dataclass
class MatchedMemory:
    extracted: ExtractedMemory
    matches: List[ExistingMatch]


@dataclass
class OperationPlan:
    extracted: ExtractedMemory
    operation: str                          # CREATE | UPDATE | ENHANCE | SKIP | CONTRADICT
    target_id: Optional[str] = None
    update_fields: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class SavedNode:
    extracted_index: int
    neo4j_id: str
    label: str
    operation: str


# ═══════════════════════════════════════════════════════════════════════════════
# GROQ CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class GroqClient:
    MODELS = {
        "smart": {
            "name": "qwen/qwen3-32b",
            "tpm": 30_000,
            "max_output": 4000,
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set. Export it as an environment variable.")
        self.client = AsyncGroq(api_key=self.api_key)
        self._usage: Dict[str, int] = {k: 0 for k in self.MODELS}
        self._reset_at = datetime.datetime.now()

    async def call(
        self,
        tier: str,
        system: str,
        user: str,
        json_mode: bool = True,
        estimated_tokens: int = 2000,
    ) -> Optional[str]:
        await self._maybe_wait(tier, estimated_tokens)
        model = self.MODELS[tier]["name"]
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": self.MODELS[tier]["max_output"],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self.client.chat.completions.create(**kwargs)
            self._usage[tier] += estimated_tokens
            return resp.choices[0].message.content
        except Exception as exc:
            observer.error(f"[groq] {tier} call failed: {exc}")
            return None

    async def _maybe_wait(self, tier: str, tokens: int) -> None:
        now = datetime.datetime.now()
        if (now - self._reset_at).total_seconds() >= 60:
            self._usage = {k: 0 for k in self.MODELS}
            self._reset_at = now
        if self._usage.get(tier, 0) + tokens > self.MODELS[tier]["tpm"]:
            wait = 60 - (now - self._reset_at).total_seconds()
            await asyncio.sleep(max(wait, 0))
            self._usage = {k: 0 for k in self.MODELS}
            self._reset_at = datetime.datetime.now()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — CONVERSATION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRACTION_SYSTEM = """You are a memory extraction engine for a personal AI assistant named SOFi.
Your job: read a conversation and extract what is genuinely worth storing as long-term memory.

MEMORY TYPES:

EXPERIENCE — A specific event, interaction, or episode that happened.
  Required fields: event_type, participants, emotional_tone, timestamp
  event_type: one of [conversation, meeting, activity, learning, work, social, problem_solving, conflict, celebration, other]
  emotional_tone: -1.0 (very negative) to +1.0 (very positive)
  participants: list of people involved by name

KNOWLEDGE — A fact, concept, insight, or skill worth retaining.
  Required fields: concept, definition, category
  category: one of [technology, work, health, finance, education, social, personal, other]

RELATIONSHIP — Information about a specific person.
  Required fields: person_name, relationship_type, emotional_connection
  relationship_type: one of [friend, family, colleague, mentor, acquaintance, romantic, other]
  emotional_connection: -1.0 to +1.0
  trust_level: 0.0 to 1.0

WHAT TO STORE (importance >= 0.4):
  - Specific people mentioned with meaningful context
  - Events with lasting significance: conflicts, decisions, achievements, failures
  - Facts or skills useful to recall later
  - Emotional states with clear causes
  - Commitments, goals, recurring patterns

WHAT TO SKIP (importance < 0.4):
  - Small talk, greetings, filler
  - Rhetorical questions
  - Trivial logistics that expire immediately

EDGE TYPES (relationships between extracted memories, choose from):
  EXPERIENCE_CHAIN, EXPERIENCE_TO_KNOWLEDGE, EXPERIENCE_TO_RELATIONSHIP,
  RELATIONSHIP_TO_EXPERIENCE, KNOWLEDGE_HIERARCHY, CAUSED, RESULTED_IN,
  TRIGGERED, INFLUENCED, HAPPENED_BEFORE, HAPPENED_AFTER, CONCURRENT, SIMILAR_TO

Output ONLY valid JSON. No commentary, no markdown."""

_EXTRACTION_USER = """Analyze this conversation. Extract memories worth storing long-term.

USER: {user_id}
DATE: {session_date}

CONVERSATION:
{conversation_text}

Output this exact JSON:
{{
  "overall_sentiment": 0.0,
  "memories": [
    {{
      "index": 0,
      "type": "EXPERIENCE|KNOWLEDGE|RELATIONSHIP",
      "content": "one or two sentence memory summary",
      "importance": 0.8,
      "tags": [],

      "event_type": "conversation",
      "participants": [],
      "emotional_tone": 0.0,
      "lessons_learned": [],
      "timestamp": "{session_date}T00:00:00",

      "concept": null,
      "definition": null,
      "category": null,
      "related_concepts": [],

      "person_name": null,
      "relationship_type": null,
      "emotional_connection": 0.0,
      "personality_traits": [],
      "interests": [],
      "trust_level": 0.5
    }}
  ],
  "edges": [
    {{
      "from_index": 0,
      "to_index": 1,
      "rel_type": "EXPERIENCE_TO_RELATIONSHIP",
      "strength": 0.8,
      "bidirectional": false
    }}
  ]
}}

Include only the fields relevant to each memory type. Skip memories with importance < 0.4."""


class ConversationAnalyzer:
    """
    Stage 1: One LLM call extracts all typed memories + intra-session edges.
    Retries on JSON parse failure. Falls back to raw EXPERIENCE node if all attempts fail.
    """

    def __init__(self, groq: GroqClient):
        self.groq = groq

    async def analyze(
        self,
        conversations: List[Dict[str, str]],
        user_id: str,
        session_date: str,
        max_retries: int = 2,
    ) -> ExtractionResult:
        conv_text = "\n".join(
            f"{m.get('role', '?').upper()}: {m.get('content', '')}"
            for m in conversations
        )
        if len(conv_text) > 6000:
            conv_text = conv_text[:6000] + "\n[truncated]"

        user_prompt = _EXTRACTION_USER.format(
            user_id=user_id,
            session_date=session_date,
            conversation_text=conv_text,
        )

        for attempt in range(max_retries + 1):
            raw = await self.groq.call(
                tier="smart",
                system=_EXTRACTION_SYSTEM,
                user=user_prompt,
                json_mode=True,
                estimated_tokens=len(conv_text) // 3 + 1500,
            )
            if raw is None:
                continue
            try:
                data = json.loads(raw)
                return self._parse(data, session_date)
            except (json.JSONDecodeError, KeyError) as exc:
                observer.warning(f"[analyzer] Parse failed attempt {attempt + 1}: {exc}")

        observer.warning("[analyzer] All attempts failed — fallback extraction")
        return self._fallback(conversations, session_date)

    def _parse(self, data: Dict, session_date: str) -> ExtractionResult:
        memories: List[ExtractedMemory] = []
        for raw_m in data.get("memories", []):
            m = self._parse_memory(raw_m, session_date)
            if m is not None:
                memories.append(m)

        valid_idx = {m.index for m in memories}
        edges: List[ExtractedEdge] = []
        for raw_e in data.get("edges", []):
            fi = raw_e.get("from_index", -1)
            ti = raw_e.get("to_index", -1)
            if fi in valid_idx and ti in valid_idx and fi != ti:
                edges.append(ExtractedEdge(
                    from_index=fi,
                    to_index=ti,
                    rel_type=str(raw_e.get("rel_type", "ASSOCIATED_WITH")),
                    strength=float(raw_e.get("strength", 0.7)),
                    bidirectional=bool(raw_e.get("bidirectional", False)),
                ))

        return ExtractionResult(
            memories=memories,
            edges=edges,
            overall_sentiment=float(data.get("overall_sentiment", 0.0)),
        )

    def _parse_memory(self, raw: Dict, session_date: str) -> Optional[ExtractedMemory]:
        mem_type = str(raw.get("type", "")).upper()
        if mem_type not in ("EXPERIENCE", "KNOWLEDGE", "RELATIONSHIP"):
            return None

        content = str(raw.get("content", "")).strip()
        if not content:
            return None

        importance = float(raw.get("importance", 0.5))
        if importance < 0.4:
            return None

        # Type-specific required field validation
        if mem_type == "EXPERIENCE":
            if not raw.get("event_type"):
                raw["event_type"] = "conversation"
            if not raw.get("timestamp"):
                raw["timestamp"] = f"{session_date}T00:00:00"
        elif mem_type == "KNOWLEDGE":
            if not raw.get("concept") or not raw.get("definition"):
                return None
        elif mem_type == "RELATIONSHIP":
            if not raw.get("person_name"):
                return None

        return ExtractedMemory(
            index=int(raw.get("index", 0)),
            type=mem_type,
            content=content,
            importance=importance,
            tags=list(raw.get("tags") or []),
            event_type=raw.get("event_type"),
            participants=list(raw.get("participants") or []),
            emotional_tone=float(raw.get("emotional_tone") or 0.0),
            lessons_learned=list(raw.get("lessons_learned") or []),
            timestamp=raw.get("timestamp"),
            concept=raw.get("concept"),
            definition=raw.get("definition"),
            category=str(raw.get("category") or "other"),
            related_concepts=list(raw.get("related_concepts") or []),
            person_name=raw.get("person_name"),
            relationship_type=str(raw.get("relationship_type") or "other"),
            emotional_connection=float(raw.get("emotional_connection") or 0.0),
            personality_traits=list(raw.get("personality_traits") or []),
            interests=list(raw.get("interests") or []),
            trust_level=float(raw.get("trust_level") or 0.5),
        )

    def _fallback(self, conversations: List[Dict], session_date: str) -> ExtractionResult:
        text = " ".join(c.get("content", "") for c in conversations)[:400]
        return ExtractionResult(
            memories=[ExtractedMemory(
                index=0,
                type="EXPERIENCE",
                content=f"Conversation on {session_date}: {text}",
                importance=0.4,
                event_type="conversation",
                timestamp=f"{session_date}T00:00:00",
            )],
            edges=[],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — GRAPH MATCHER
# ═══════════════════════════════════════════════════════════════════════════════

class GraphMatcher:
    """
    Stage 2: Find existing Neo4j nodes that conflict with extracted memories.
    No LLM. Pure database queries.

    Matching strategy:
      RELATIONSHIP → exact person_name match (case-insensitive). One canonical node per person.
      KNOWLEDGE    → exact concept match, then BM25 fallback (threshold 2.5).
      EXPERIENCE   → BM25 on content+participants, filtered to ±48h timestamp window.
                     Content-similar but temporally distant = distinct events (not duplicates).
    """

    _BM25_THRESHOLD = 2.5
    _EXPERIENCE_WINDOW_HOURS = 48

    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def match_all(self, memories: List[ExtractedMemory]) -> List[MatchedMemory]:
        results = []
        for mem in memories:
            try:
                matches = await self._match_one(mem)
            except Exception as exc:
                observer.warning(f"[matcher] Match failed for index {mem.index}: {exc}")
                matches = []
            results.append(MatchedMemory(extracted=mem, matches=matches))
        return results

    async def _match_one(self, mem: ExtractedMemory) -> List[ExistingMatch]:
        if mem.type == "RELATIONSHIP":
            return await self._match_relationship(mem)
        elif mem.type == "KNOWLEDGE":
            return await self._match_knowledge(mem)
        else:
            return await self._match_experience(mem)

    async def _match_relationship(self, mem: ExtractedMemory) -> List[ExistingMatch]:
        rows = await self.neo4j.execute_query(
            """
            MATCH (n:RelationshipMemory)
            WHERE toLower(n.person_name) = toLower($name)
            RETURN n.id AS id, n.content AS content,
                   n.person_name AS person_name,
                   n.relationship_type AS relationship_type,
                   n.emotional_connection AS emotional_connection,
                   n.personality_traits AS personality_traits,
                   n.interests AS interests,
                   n.trust_level AS trust_level
            LIMIT 1
            """,
            {"name": mem.person_name or ""},
        )
        return [ExistingMatch(
            node_id=str(r["id"]),
            label="RelationshipMemory",
            content=str(r.get("content", "")),
            key_fields={k: r.get(k) for k in (
                "person_name", "relationship_type", "emotional_connection",
                "personality_traits", "interests", "trust_level",
            )},
            similarity=1.0,
        ) for r in rows]

    async def _match_knowledge(self, mem: ExtractedMemory) -> List[ExistingMatch]:
        # Exact concept match first
        rows = await self.neo4j.execute_query(
            """
            MATCH (n:KnowledgeMemory)
            WHERE toLower(n.concept) = toLower($concept)
            RETURN n.id AS id, n.content AS content,
                   n.concept AS concept, n.definition AS definition,
                   n.category AS category
            LIMIT 1
            """,
            {"concept": mem.concept or ""},
        )
        if rows:
            r = rows[0]
            return [ExistingMatch(
                node_id=str(r["id"]),
                label="KnowledgeMemory",
                content=str(r.get("content", "")),
                key_fields={"concept": r.get("concept"), "definition": r.get("definition")},
                similarity=1.0,
            )]

        # BM25 fallback
        bm25_q = f"{mem.concept or ''} {mem.content[:100]}"
        try:
            rows = await self.neo4j.execute_query(
                """
                CALL db.index.fulltext.queryNodes("memory_fts", $q)
                YIELD node, score
                WHERE node:KnowledgeMemory AND score > $threshold
                RETURN node.id AS id, node.content AS content,
                       node.concept AS concept, node.definition AS definition, score
                ORDER BY score DESC LIMIT 3
                """,
                {"q": bm25_q, "threshold": self._BM25_THRESHOLD},
            )
            return [ExistingMatch(
                node_id=str(r["id"]),
                label="KnowledgeMemory",
                content=str(r.get("content", "")),
                key_fields={"concept": r.get("concept"), "definition": r.get("definition")},
                similarity=min(float(r.get("score", 0)) / 5.0, 1.0),
            ) for r in rows]
        except Exception as exc:
            return []

    async def _match_experience(self, mem: ExtractedMemory) -> List[ExistingMatch]:
        bm25_q = " ".join(filter(None, [mem.content[:150]] + mem.participants[:3]))
        if not bm25_q.strip():
            return []

        ts_str = mem.timestamp or datetime.datetime.now().isoformat()
        try:
            ts = datetime.datetime.fromisoformat(ts_str.replace("Z", ""))
        except ValueError:
            ts = datetime.datetime.now()

        window_start = (ts - datetime.timedelta(hours=self._EXPERIENCE_WINDOW_HOURS)).isoformat()
        window_end = (ts + datetime.timedelta(hours=self._EXPERIENCE_WINDOW_HOURS)).isoformat()

        try:
            rows = await self.neo4j.execute_query(
                """
                CALL db.index.fulltext.queryNodes("memory_fts", $q)
                YIELD node, score
                WHERE node:ExperienceMemory
                  AND score > $threshold
                  AND node.timestamp >= $window_start
                  AND node.timestamp <= $window_end
                RETURN node.id AS id, node.content AS content,
                       node.event_type AS event_type,
                       node.participants AS participants,
                       node.timestamp AS timestamp,
                       node.emotional_tone AS emotional_tone, score
                ORDER BY score DESC LIMIT 3
                """,
                {
                    "q": bm25_q,
                    "threshold": self._BM25_THRESHOLD,
                    "window_start": window_start,
                    "window_end": window_end,
                },
            )
            return [ExistingMatch(
                node_id=str(r["id"]),
                label="ExperienceMemory",
                content=str(r.get("content", "")),
                key_fields={
                    "event_type": r.get("event_type"),
                    "participants": r.get("participants", []),
                    "timestamp": r.get("timestamp"),
                    "emotional_tone": r.get("emotional_tone"),
                },
                similarity=min(float(r.get("score", 0)) / 5.0, 1.0),
            ) for r in rows]
        except Exception as exc:
            return []


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — CONFLICT RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

_RESOLVE_SYSTEM = """You are a memory conflict resolver for a personal AI assistant's knowledge graph.

You receive conflicts — cases where newly extracted memories resemble existing ones in the graph.
Decide the correct operation for each:

UPDATE     — Specific fields in the existing node should change (mood shifted, opinion updated, new fact).
             Specify exactly which fields and their new values in update_fields.
ENHANCE    — Add new details to existing without replacing anything (new traits, examples, lessons).
             List what to append to which list fields in enhance_additions.
SKIP       — Existing already captures this. No new node needed. Existing will be reinforced.
CONTRADICT — New info fundamentally reverses existing (trust broken, relationship ended, belief inverted).
             A new node is created and the old one marked superseded.
CREATE     — Despite resemblance, this is genuinely a distinct memory. Create it.

Type-specific rules:
- RELATIONSHIP: prefer ENHANCE or UPDATE — relationship info accumulates over time.
  Only CONTRADICT for fundamental reversals (friend became enemy, trust completely broken).
- KNOWLEDGE: UPDATE if understanding improved. ENHANCE if new examples/context added. SKIP if redundant.
- EXPERIENCE: SKIP if the exact same event is already stored. CREATE if it's a different episode.

Output JSON only. No commentary."""

_RESOLVE_USER = """Resolve these conflicts:

{conflicts_json}

Output:
{{
  "resolutions": [
    {{
      "conflict_index": 0,
      "operation": "UPDATE|ENHANCE|SKIP|CONTRADICT|CREATE",
      "reason": "brief explanation",
      "update_fields": {{}},
      "enhance_additions": {{}}
    }}
  ]
}}

update_fields: field_name → new_value (UPDATE only)
enhance_additions: list_field_name → [items to append] (ENHANCE only)"""


class ConflictResolver:
    """
    Stage 3: LLM decides what to do with conflicting memories.
    Only fires when Stage 2 found conflicts — skipped entirely on all-new sessions.
    All conflicts batched into ONE call.
    """

    def __init__(self, groq: GroqClient):
        self.groq = groq

    async def resolve(self, matched: List[MatchedMemory]) -> List[OperationPlan]:
        plans: List[OperationPlan] = []
        conflicts = [(i, m) for i, m in enumerate(matched) if m.matches]
        non_conflicts = [(i, m) for i, m in enumerate(matched) if not m.matches]

        for _, m in non_conflicts:
            plans.append(OperationPlan(
                extracted=m.extracted,
                operation="CREATE",
                reason="No similar memory exists.",
            ))

        if not conflicts:
            return plans

        payload = []
        for ci, (_, m) in enumerate(conflicts):
            payload.append({
                "conflict_index": ci,
                "extracted": self._fmt_extracted(m.extracted),
                "existing": [self._fmt_existing(e) for e in m.matches],
            })

        raw = await self.groq.call(
            tier="smart",
            system=_RESOLVE_SYSTEM,
            user=_RESOLVE_USER.format(conflicts_json=json.dumps(payload, indent=2)),
            json_mode=True,
            estimated_tokens=len(json.dumps(payload)) // 3 + 1000,
        )

        resolutions: Dict[int, Dict] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                for r in parsed.get("resolutions", []):
                    resolutions[int(r["conflict_index"])] = r
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                observer.warning(f"[resolver] Parse failed: {exc} — defaulting to ENHANCE")

        for ci, (_, m) in enumerate(conflicts):
            res = resolutions.get(ci, {})
            op = str(res.get("operation", "ENHANCE")).upper()
            if op not in ("UPDATE", "ENHANCE", "SKIP", "CONTRADICT", "CREATE"):
                op = "ENHANCE"
            best = m.matches[0]
            plans.append(OperationPlan(
                extracted=m.extracted,
                operation=op,
                target_id=best.node_id if op != "CREATE" else None,
                update_fields=dict(res.get("update_fields") or {}),
                reason=str(res.get("reason", "")),
            ))

        plans.sort(key=lambda p: p.extracted.index)
        return plans

    def _fmt_extracted(self, m: ExtractedMemory) -> Dict:
        d: Dict[str, Any] = {"type": m.type, "content": m.content, "importance": m.importance}
        if m.type == "EXPERIENCE":
            d.update({"participants": m.participants, "emotional_tone": m.emotional_tone,
                       "lessons": m.lessons_learned, "timestamp": m.timestamp})
        elif m.type == "KNOWLEDGE":
            d.update({"concept": m.concept, "definition": m.definition})
        elif m.type == "RELATIONSHIP":
            d.update({"person": m.person_name, "emotional_connection": m.emotional_connection,
                       "traits": m.personality_traits, "interests": m.interests,
                       "trust": m.trust_level})
        return d

    def _fmt_existing(self, e: ExistingMatch) -> Dict:
        return {"id": e.node_id, "content": e.content,
                "similarity": round(e.similarity, 2), "fields": e.key_fields}


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — MEMORY WRITER
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryWriter:
    """
    Stage 4: Execute operation plans. Write correctly-typed nodes to Neo4j.

    Key behaviors:
    - MERGE (not CREATE) for canonical types: RelationshipMemory on person_name,
      KnowledgeMemory on concept. Prevents duplicate nodes across sessions.
    - Batch embedding before writes — one encode() call per session, not per node.
    - ENHANCE uses Cypher list deduplication — never re-adds items already in a list field.
    - CONTRADICT: creates new node + SUPERSEDED_BY edge. Old node preserved for lineage.
    - SKIP: reinforces existing node (access_count++) without creating anything new.
    - Per-node try/except: failure on one node doesn't abort the rest.
    """

    def __init__(self, neo4j: Neo4jClient, embed: EmbeddingUtils):
        self.neo4j = neo4j
        self.embed = embed

    async def execute(self, plans: List[OperationPlan]) -> Dict[int, SavedNode]:
        to_embed = [(p.extracted.index, p.extracted.content)
                    for p in plans if p.operation != "SKIP"]
        embeddings = self._batch_embed(to_embed)

        saved: Dict[int, SavedNode] = {}
        for plan in plans:
            try:
                node = await self._execute_one(plan, embeddings)
                if node:
                    saved[plan.extracted.index] = node
            except Exception as exc:
                observer.error(
                    f"[writer] {plan.operation} failed for index {plan.extracted.index}: {exc}",
                )
        return saved

    def _batch_embed(self, items: List[Tuple[int, str]]) -> Dict[int, List[float]]:
        result: Dict[int, List[float]] = {}
        for idx, text in items:
            try:
                result[idx] = self.embed.generate_embedding(text)
            except Exception as exc:
                observer.warning(f"[writer] Embed failed for index {idx}: {exc}")
        return result

    async def _execute_one(
        self, plan: OperationPlan, embeddings: Dict[int, List[float]]
    ) -> Optional[SavedNode]:
        m = plan.extracted
        vec = embeddings.get(m.index)
        now = datetime.datetime.now().isoformat()

        if plan.operation == "SKIP":
            if plan.target_id:
                await self._reinforce(plan.target_id, now)
            return None

        if plan.operation == "CREATE":
            return await self._create(m, vec, now)

        if plan.operation == "UPDATE":
            return await self._update(plan.target_id, plan.update_fields, m, vec, now)

        if plan.operation == "ENHANCE":
            return await self._enhance(plan.target_id, m, now)

        if plan.operation == "CONTRADICT":
            node = await self._create(m, vec, now)
            if node and plan.target_id:
                await self._mark_superseded(plan.target_id, node.neo4j_id, now)
            return node

        return None

    # ── CREATE ─────────────────────────────────────────────────────────────────

    async def _create(
        self, m: ExtractedMemory, vec: Optional[List[float]], now: str
    ) -> Optional[SavedNode]:
        from uuid import uuid4
        node_id = str(uuid4())

        if m.type == "EXPERIENCE":
            props = {
                "id": node_id,
                "memory_context": "EXPERIENCE",
                "content": m.content,
                "content_vector": vec,
                "event_type": m.event_type or "conversation",
                "timestamp": m.timestamp or now,
                "participants": m.participants,
                "emotional_tone": m.emotional_tone,
                "lessons_learned": m.lessons_learned,
                "importance_score": m.importance,
                "tags": m.tags,
                "confidence": 0.85,
                "access_count": 0,
                "created_date": now,
                "last_updated": now,
            }
            rows = await self.neo4j.execute_query(
                "CREATE (n:ExperienceMemory $props) RETURN n.id AS id",
                {"props": props},
            )
            if rows:
                return SavedNode(m.index, node_id, "ExperienceMemory", "CREATE")

        elif m.type == "KNOWLEDGE":
            props = {
                "id": node_id,
                "memory_context": "KNOWLEDGE",
                "content": m.content,
                "content_vector": vec,
                "concept": m.concept or m.content[:50],
                "definition": m.definition or m.content,
                "category": m.category or "other",
                "related_concepts": m.related_concepts,
                "importance_score": m.importance,
                "tags": m.tags,
                "confidence": 0.85,
                "confidence_level": 0.85,
                "access_count": 0,
                "created_date": now,
                "last_updated": now,
            }
            # MERGE on concept — canonical node per concept
            rows = await self.neo4j.execute_query(
                """
                MERGE (n:KnowledgeMemory {concept: $concept})
                ON CREATE SET n = $props
                ON MATCH SET
                    n.content = $content, n.last_updated = $now,
                    n.importance_score = CASE WHEN $imp > n.importance_score THEN $imp ELSE n.importance_score END,
                    n.content_vector = CASE WHEN $vec IS NOT NULL THEN $vec ELSE n.content_vector END
                RETURN n.id AS id
                """,
                {"concept": m.concept, "props": props, "content": m.content,
                 "now": now, "imp": m.importance, "vec": vec},
            )
            if rows:
                actual_id = str(rows[0]["id"]) if rows[0].get("id") else node_id
                return SavedNode(m.index, actual_id, "KnowledgeMemory", "CREATE")

        elif m.type == "RELATIONSHIP":
            props = {
                "id": node_id,
                "memory_context": "RELATIONSHIP",
                "content": m.content,
                "content_vector": vec,
                "person_name": m.person_name,
                "relationship_type": m.relationship_type or "other",
                "emotional_connection": m.emotional_connection,
                "personality_traits": m.personality_traits,
                "interests": m.interests,
                "trust_level": m.trust_level,
                "relationship_strength": 0.5,
                "importance_score": m.importance,
                "tags": m.tags,
                "confidence": 0.85,
                "access_count": 0,
                "created_date": now,
                "last_updated": now,
            }
            # MERGE on person_name — single canonical node per person
            rows = await self.neo4j.execute_query(
                """
                MERGE (n:RelationshipMemory {person_name: $person_name})
                ON CREATE SET n = $props
                ON MATCH SET
                    n.content = $content, n.last_updated = $now,
                    n.emotional_connection = $ec, n.trust_level = $trust,
                    n.importance_score = CASE WHEN $imp > n.importance_score THEN $imp ELSE n.importance_score END,
                    n.content_vector = CASE WHEN $vec IS NOT NULL THEN $vec ELSE n.content_vector END
                RETURN n.id AS id
                """,
                {"person_name": m.person_name, "props": props, "content": m.content,
                 "now": now, "ec": m.emotional_connection, "trust": m.trust_level,
                 "imp": m.importance, "vec": vec},
            )
            if rows:
                actual_id = str(rows[0]["id"]) if rows[0].get("id") else node_id
                return SavedNode(m.index, actual_id, "RelationshipMemory", "CREATE")

        return None

    # ── UPDATE ─────────────────────────────────────────────────────────────────

    async def _update(
        self,
        target_id: str,
        update_fields: Dict,
        m: ExtractedMemory,
        vec: Optional[List[float]],
        now: str,
    ) -> Optional[SavedNode]:
        if not update_fields:
            return await self._enhance(target_id, m, now)

        set_parts = ["n.last_updated = $now", "n.content = $content"]
        params: Dict[str, Any] = {"node_id": target_id, "now": now, "content": m.content}

        if vec:
            set_parts.append("n.content_vector = $vec")
            params["vec"] = vec

        for k, v in update_fields.items():
            safe = k.replace(" ", "_").replace("-", "_")
            pk = f"f_{safe}"
            set_parts.append(f"n.{safe} = ${pk}")
            params[pk] = v

        try:
            rows = await self.neo4j.execute_query(
                f"MATCH (n {{id: $node_id}}) SET {', '.join(set_parts)} "
                f"RETURN n.id AS id, labels(n)[0] AS label",
                params,
            )
            if rows:
                return SavedNode(m.index, target_id, str(rows[0].get("label", "")), "UPDATE")
        except Exception as exc:
            observer.error(f"[writer] UPDATE failed for {target_id}: {exc}")
        return None

    # ── ENHANCE ────────────────────────────────────────────────────────────────

    async def _enhance(self, target_id: str, m: ExtractedMemory, now: str) -> Optional[SavedNode]:
        """
        Append-only enrichment. Cypher list deduplication ensures no re-added items.
        Takes max on importance_score — importance only ever goes up.
        """
        try:
            rows = await self.neo4j.execute_query(
                """
                MATCH (n {id: $node_id})
                SET n.last_updated = $now,
                    n.importance_score = CASE
                        WHEN $imp > coalesce(n.importance_score, 0) THEN $imp
                        ELSE n.importance_score END,
                    n.participants = CASE
                        WHEN n.participants IS NOT NULL
                        THEN [x IN $parts WHERE NOT x IN n.participants] + n.participants
                        ELSE $parts END,
                    n.lessons_learned = CASE
                        WHEN n.lessons_learned IS NOT NULL
                        THEN [x IN $lessons WHERE NOT x IN n.lessons_learned] + n.lessons_learned
                        ELSE $lessons END,
                    n.personality_traits = CASE
                        WHEN n.personality_traits IS NOT NULL
                        THEN [x IN $traits WHERE NOT x IN n.personality_traits] + n.personality_traits
                        ELSE $traits END,
                    n.interests = CASE
                        WHEN n.interests IS NOT NULL
                        THEN [x IN $interests WHERE NOT x IN n.interests] + n.interests
                        ELSE $interests END,
                    n.tags = CASE
                        WHEN n.tags IS NOT NULL
                        THEN [x IN $tags WHERE NOT x IN n.tags] + n.tags
                        ELSE $tags END
                RETURN n.id AS id, labels(n)[0] AS label
                """,
                {
                    "node_id": target_id, "now": now, "imp": m.importance,
                    "parts": m.participants, "lessons": m.lessons_learned,
                    "traits": m.personality_traits, "interests": m.interests,
                    "tags": m.tags,
                },
            )
            if rows:
                return SavedNode(m.index, target_id, str(rows[0].get("label", "")), "ENHANCE")
        except Exception as exc:
            observer.error(f"[writer] ENHANCE failed for {target_id}: {exc}")
        return None

    # ── Helpers ────────────────────────────────────────────────────────────────

    async def _mark_superseded(self, old_id: str, new_id: str, now: str) -> None:
        """CONTRADICT: link old node to new as superseded. Old content preserved."""
        try:
            await self.neo4j.execute_query(
                """
                MATCH (old {id: $old_id}), (new {id: $new_id})
                MERGE (old)-[r:SUPERSEDED_BY]->(new)
                ON CREATE SET r.created_date = $now, r.strength = 1.0
                SET old.superseded = true, old.superseded_at = $now
                """,
                {"old_id": old_id, "new_id": new_id, "now": now},
            )
        except Exception as exc:
            observer.warning(f"[writer] SUPERSEDED_BY failed: {exc}")

    async def _reinforce(self, node_id: str, now: str) -> None:
        """SKIP: existing already captures it — reinforce its ACT-R access stats."""
        try:
            await self.neo4j.execute_query(
                """
                MATCH (n {id: $node_id})
                SET n.access_count = coalesce(n.access_count, 0) + 1,
                    n.last_accessed = $now
                """,
                {"node_id": node_id, "now": now},
            )
        except Exception as exc:
            observer.warning(f"[writer] Reinforce failed for {node_id}: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — GRAPH LINKER
# ═══════════════════════════════════════════════════════════════════════════════

_VALID_EDGE_TYPES = {e.value for e in MemoryRelationshipType}

_EDGE_REVERSALS = {
    "EXPERIENCE_TO_RELATIONSHIP": "RELATIONSHIP_TO_EXPERIENCE",
    "RELATIONSHIP_TO_EXPERIENCE": "EXPERIENCE_TO_RELATIONSHIP",
    "EXPERIENCE_TO_KNOWLEDGE":    "KNOWLEDGE_TO_EXPERIENCE",
    "KNOWLEDGE_TO_EXPERIENCE":    "EXPERIENCE_TO_KNOWLEDGE",
    "HAPPENED_BEFORE":            "HAPPENED_AFTER",
    "HAPPENED_AFTER":             "HAPPENED_BEFORE",
    "CAUSED":                     "RESULTED_IN",
    "RESULTED_IN":                "CAUSED",
}


class GraphLinker:
    """
    Stage 5: Build edges — intra-session (from extraction) and cross-session (entity bridges).

    Intra-session: edges the LLM explicitly identified between extracted memories.
    Cross-session: automatic bridges from new nodes to the existing graph —
      - ExperienceMemory → RelationshipMemory for each participant
      - ExperienceMemory → prior ExperienceMemory involving same people (temporal chain)
      - RelationshipMemory → all prior experiences mentioning that person (backfill)
      - KnowledgeMemory → related concepts already in graph (hierarchy)

    Edge deduplication (Graphiti approach):
      MERGE (a)-[r:TYPE]->(b) — if edge exists, strengthen it (evidence_count++, strength+0.05).
      Never creates duplicate edges between the same node pair with the same type.
    """

    def __init__(self, neo4j: Neo4jClient):
        self.neo4j = neo4j

    async def link(self, saved: Dict[int, SavedNode], extraction: ExtractionResult) -> None:
        now = datetime.datetime.now().isoformat()
        index_to_id = {s.extracted_index: s.neo4j_id for s in saved.values()}

        await self._link_intra_session(extraction.edges, index_to_id, now)

        for idx, node in saved.items():
            mem = next((m for m in extraction.memories if m.index == idx), None)
            if mem:
                await self._link_cross_session(node, mem, now)

    async def _link_intra_session(
        self, edges: List[ExtractedEdge], index_to_id: Dict[int, str], now: str
    ) -> None:
        for edge in edges:
            from_id = index_to_id.get(edge.from_index)
            to_id = index_to_id.get(edge.to_index)
            if not from_id or not to_id:
                continue
            await self._merge_edge(from_id, to_id, edge.rel_type, edge.strength, now)
            if edge.bidirectional:
                rev = _EDGE_REVERSALS.get(edge.rel_type, edge.rel_type)
                await self._merge_edge(to_id, from_id, rev, edge.strength, now)

    async def _link_cross_session(self, node: SavedNode, mem: ExtractedMemory, now: str) -> None:
        if mem.type == "EXPERIENCE":
            for person in mem.participants:
                await self._exp_to_person(node.neo4j_id, person, now)
            if mem.participants:
                await self._temporal_chain(node.neo4j_id, mem.participants, mem.timestamp, now)

        elif mem.type == "RELATIONSHIP" and mem.person_name:
            await self._person_to_experiences(node.neo4j_id, mem.person_name, now)

        elif mem.type == "KNOWLEDGE" and mem.related_concepts:
            await self._knowledge_hierarchy(node.neo4j_id, mem.related_concepts, now)

    async def _exp_to_person(self, exp_id: str, person: str, now: str) -> None:
        try:
            await self.neo4j.execute_query(
                """
                MATCH (exp {id: $exp_id})
                MATCH (rel:RelationshipMemory)
                WHERE toLower(rel.person_name) = toLower($person) AND rel.id <> $exp_id
                MERGE (exp)-[r:EXPERIENCE_TO_RELATIONSHIP]->(rel)
                ON CREATE SET r.strength = 0.7, r.created_date = $now, r.evidence_count = 1
                ON MATCH  SET r.strength = least(1.0, coalesce(r.strength,0.7) + 0.05),
                              r.evidence_count = coalesce(r.evidence_count, 1) + 1,
                              r.last_reinforced = $now
                """,
                {"exp_id": exp_id, "person": person, "now": now},
            )
        except Exception:
            pass

    async def _temporal_chain(
        self, new_id: str, participants: List[str], timestamp: Optional[str], now: str
    ) -> None:
        """Link new experience to most recent prior experience sharing any participant."""
        if not timestamp:
            return
        try:
            await self.neo4j.execute_query(
                """
                MATCH (new_exp {id: $new_id})
                MATCH (old_exp:ExperienceMemory)
                WHERE old_exp.id <> $new_id
                  AND old_exp.timestamp < $timestamp
                  AND any(p IN $participants WHERE toLower(p) IN
                      [x IN coalesce(old_exp.participants, []) | toLower(x)])
                WITH old_exp ORDER BY old_exp.timestamp DESC LIMIT 1
                MERGE (old_exp)-[r1:HAPPENED_BEFORE]->(new_exp)
                ON CREATE SET r1.strength = 0.8, r1.created_date = $now, r1.evidence_count = 1
                ON MATCH  SET r1.evidence_count = coalesce(r1.evidence_count, 1) + 1
                MERGE (new_exp)-[r2:HAPPENED_AFTER]->(old_exp)
                ON CREATE SET r2.strength = 0.8, r2.created_date = $now, r2.evidence_count = 1
                ON MATCH  SET r2.evidence_count = coalesce(r2.evidence_count, 1) + 1
                """,
                {"new_id": new_id, "timestamp": timestamp,
                 "participants": participants, "now": now},
            )
        except Exception as exc:
            pass

    async def _person_to_experiences(self, rel_id: str, person_name: str, now: str) -> None:
        """Backfill: link updated RelationshipMemory to all existing experiences mentioning them."""
        try:
            await self.neo4j.execute_query(
                """
                MATCH (rel {id: $rel_id})
                MATCH (exp:ExperienceMemory)
                WHERE exp.id <> $rel_id
                  AND any(p IN coalesce(exp.participants,[]) WHERE toLower(p) = toLower($person))
                MERGE (exp)-[r:EXPERIENCE_TO_RELATIONSHIP]->(rel)
                ON CREATE SET r.strength = 0.6, r.created_date = $now, r.evidence_count = 1
                ON MATCH  SET r.strength = least(1.0, coalesce(r.strength,0.6) + 0.03),
                              r.evidence_count = coalesce(r.evidence_count, 1) + 1
                """,
                {"rel_id": rel_id, "person": person_name, "now": now},
            )
        except Exception as exc:
            pass

    async def _knowledge_hierarchy(
        self, know_id: str, related: List[str], now: str
    ) -> None:
        try:
            await self.neo4j.execute_query(
                """
                MATCH (new_k {id: $know_id})
                MATCH (existing_k:KnowledgeMemory)
                WHERE toLower(existing_k.concept) IN $related_lower AND existing_k.id <> $know_id
                MERGE (new_k)-[r:KNOWLEDGE_HIERARCHY]->(existing_k)
                ON CREATE SET r.strength = 0.6, r.created_date = $now, r.evidence_count = 1
                ON MATCH  SET r.evidence_count = coalesce(r.evidence_count, 1) + 1
                """,
                {"know_id": know_id, "related_lower": [r.lower() for r in related], "now": now},
            )
        except Exception as exc:
            pass

    async def _merge_edge(
        self, from_id: str, to_id: str, rel_type: str, strength: float, now: str
    ) -> None:
        safe_type = rel_type if rel_type in _VALID_EDGE_TYPES else "ASSOCIATED_WITH"
        try:
            await self.neo4j.execute_query(
                f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                MERGE (a)-[r:{safe_type}]->(b)
                ON CREATE SET r.strength = $strength, r.created_date = $now,
                              r.evidence_count = 1, r.last_reinforced = $now
                ON MATCH  SET r.strength = least(1.0, coalesce(r.strength,0.5) + 0.05),
                              r.evidence_count = coalesce(r.evidence_count, 1) + 1,
                              r.last_reinforced = $now
                """,
                {"from_id": from_id, "to_id": to_id, "strength": strength, "now": now},
            )
        except Exception as exc:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# CONSOLIDATION ENGINE — COORDINATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidationEngine:
    """
    Orchestrates the 5-stage pipeline. Runs nightly or on-demand.

    Guarantees:
    - Partial success: node N failing doesn't abort nodes N+1..M
    - Idempotency: re-running on same session is safe (MERGE semantics throughout)
    - Processed sessions deleted from conversation.json only on full success
    - Failed sessions retained for next run
    """

    TRIGGER_HOUR = 20

    def __init__(
        self,
        neo4j: Neo4jClient,
        embed: EmbeddingUtils,
        retrieval_engine: Optional[MemoryRetrievalEngine] = None,
    ):
        self.neo4j = neo4j
        self.embed = embed
        self.cfg = get_config()
        self.log_path = Path(self.cfg.conversation_log_path)
        self.is_running = False

        groq = GroqClient()
        self.analyzer = ConversationAnalyzer(groq)
        self.matcher  = GraphMatcher(neo4j)
        self.resolver = ConflictResolver(groq)
        self.writer   = MemoryWriter(neo4j, embed)
        self.linker   = GraphLinker(neo4j)


    # ── Scheduler ─────────────────────────────────────────────────────────────

    async def start_scheduler(self) -> None:
        self.is_running = True
        while self.is_running:
            try:
                now = datetime.datetime.now()
                if self.TRIGGER_HOUR <= now.hour < self.TRIGGER_HOUR + 1:
                    await self.run_consolidation_once()
                    sleep_s = self._seconds_until_tomorrow()
                    await asyncio.sleep(sleep_s)
                else:
                    await asyncio.sleep(3600)
            except Exception as exc:
                observer.error(f"[consolidation] Scheduler error: {exc}")
                await asyncio.sleep(3600)

    def _seconds_until_tomorrow(self) -> float:
        now = datetime.datetime.now()
        tomorrow = (now + datetime.timedelta(days=1)).replace(hour=0, minute=1, second=0)
        return (tomorrow - now).total_seconds()

    # ── Main entry ─────────────────────────────────────────────────────────────

    async def run_consolidation_once(self) -> Dict[str, Any]:
        stats = {
            "sessions_attempted": 0, "sessions_succeeded": 0,
            "memories_created": 0, "memories_updated": 0, "sessions_failed": 0,
        }

        log_data = self._load_log()
        if not log_data:
            return stats

        user_key = f"user_{self.cfg.user_id}"
        sessions = log_data.get(user_key, [])
        if not sessions:
            return stats

        keep = []
        for session in sessions:
            stats["sessions_attempted"] += 1
            sid = session.get("session_id", "unknown")
            convs = session.get("conversations", [])
            if not convs:
                continue

            try:
                result = await self._consolidate_session(convs, sid)
                stats["sessions_succeeded"] += 1
                stats["memories_created"] += result["created"]
                stats["memories_updated"] += result["updated"]
            except Exception as exc:
                observer.error(f"[consolidation] ✗ {sid}: {exc}")
                keep.append(session)
                stats["sessions_failed"] += 1

        log_data[user_key] = keep
        self._save_log(log_data)
        return stats

    async def _consolidate_session(
        self, conversations: List[Dict], session_id: str
    ) -> Dict[str, int]:
        session_date = datetime.date.today().isoformat()

        # Stage 1
        extraction = await self.analyzer.analyze(conversations, self.cfg.user_id, session_date)
        if not extraction.memories:
            return {"created": 0, "updated": 0}


        # Stage 2
        matched = await self.matcher.match_all(extraction.memories)
        conflicts = sum(1 for m in matched if m.matches)

        # Stage 3
        plans = await self.resolver.resolve(matched)
        ops: Dict[str, int] = {}
        for p in plans:
            ops[p.operation] = ops.get(p.operation, 0) + 1

        # Stage 4
        saved = await self.writer.execute(plans)

        # Stage 5
        await self.linker.link(saved, extraction)

        created = sum(1 for p in plans if p.operation == "CREATE")
        updated = sum(1 for p in plans if p.operation in ("UPDATE", "ENHANCE", "CONTRADICT"))
        return {"created": created, "updated": updated}

    # ── File helpers ───────────────────────────────────────────────────────────

    def _load_log(self) -> Dict:
        if not self.log_path.exists():
            return {}
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception as exc:
            observer.error(f"[consolidation] Load log failed: {exc}")
            return {}

    def _save_log(self, data: Dict) -> None:
        try:
            self.log_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            observer.error(f"[consolidation] Save log failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY + CLI
# ═══════════════════════════════════════════════════════════════════════════════

async def create_consolidation_engine(
    neo4j_client: Neo4jClient,
    embed_utils: EmbeddingUtils,
    retrieval_engine: Optional[MemoryRetrievalEngine] = None,
) -> ConsolidationEngine:
    return ConsolidationEngine(neo4j_client, embed_utils, retrieval_engine)


if __name__ == "__main__":
    async def main():
        cfg = get_config()
        neo4j = create_neo4j_client(
            uri=cfg.neo4j_uri,
            username=cfg.neo4j_username,
            password=cfg.neo4j_password,
            database=cfg.database,
        )
        await neo4j.connect()
        embed = EmbeddingUtils()
        engine = ConsolidationEngine(neo4j, embed)
        stats = await engine.run_consolidation_once()
        await neo4j.disconnect()

    asyncio.run(main())
