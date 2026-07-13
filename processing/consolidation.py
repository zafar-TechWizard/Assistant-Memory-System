"""
Memory Consolidation -- Agentic Pipeline via Gemini CLI

One reasoning pass per session. The Gemini CLI agent reads the conversation,
sees pre-fetched relevant memories from the graph, and produces a complete
consolidation plan: what to create, update, enhance, skip, or supersede.
Python deterministically executes the plan against Neo4j.

Architecture:
    ContextFetcher   ->  fetches existing memories likely relevant to session
    GeminiAgent      ->  invokes gemini CLI, parses plan from output
    PlanExecutor     ->  applies the plan: writes nodes, creates edges
    AgenticEngine    ->  orchestrates the above per session

Pre-fetching means the agent doesn't need graph access during reasoning -- all
relevant existing memories are inlined into the prompt. The agent then reasons
step-by-step internally and emits the plan as structured JSON.

Design principles preserved from prior version:
- Single canonical node per person (MERGE on person_name)
- Single canonical node per concept (MERGE on concept)
- Operations: CREATE | UPDATE | ENHANCE | SKIP | CONTRADICT
- Never physically delete -- CONTRADICT preserves temporal lineage
- Edge dedup via MERGE (from, to, type)
- Per-node error isolation
- Idempotent writes
"""

import asyncio
import datetime
import difflib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from memory.config import get_config
from memory.long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.models.node_models import (
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
)
from memory.long_term.models.relationship_models import MemoryRelationshipType
from memory.observability import observer
from memory.processing.embedding_utils import EmbeddingUtils
from memory.processing.entity_extractor import EntityExtractor


# ===============================================================================
# DATA CONTRACTS
# ===============================================================================

@dataclass
class ExtractedMemory:
    """A memory the agent extracted from the conversation."""
    op_index: int                          # position in the operations list
    type: str                              # EXPERIENCE | KNOWLEDGE | RELATIONSHIP
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
class PlanOperation:
    """One operation from the agent's plan."""
    op_index: int
    operation: str                          # CREATE | UPDATE | ENHANCE | SKIP | CONTRADICT
    reason: str = ""

    # For CREATE / CONTRADICT -- full memory dict
    memory: Optional[ExtractedMemory] = None

    # For UPDATE / ENHANCE / SKIP / CONTRADICT -- existing node id
    target_id: Optional[str] = None

    # For UPDATE -- specific field changes
    update_fields: Dict[str, Any] = field(default_factory=dict)

    # For ENHANCE -- list field appends
    enhance_additions: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class PlanEdge:
    """An edge specification from the plan."""
    # Either op_index (new node) or node_id (existing node) for each end
    from_op_index: Optional[int] = None
    from_node_id: Optional[str] = None
    to_op_index: Optional[int] = None
    to_node_id: Optional[str] = None
    rel_type: str = "ASSOCIATED_WITH"
    strength: float = 0.7
    bidirectional: bool = False


@dataclass
class ConsolidationPlan:
    """The agent's full plan for one session."""
    session_id: str
    reasoning: str = ""
    session_summary: str = ""
    overall_sentiment: float = 0.0
    operations: List[PlanOperation] = field(default_factory=list)
    edges: List[PlanEdge] = field(default_factory=list)


@dataclass
class SavedNode:
    """Result of an executed operation."""
    op_index: int
    neo4j_id: str
    label: str
    operation: str


@dataclass
class SessionResult:
    """Per-session outcome for diagnostics."""
    session_id: str
    turns: int
    succeeded: bool = False
    operations_planned: int = 0
    nodes_created: int = 0
    nodes_updated: int = 0
    nodes_enhanced: int = 0
    nodes_skipped: int = 0
    nodes_superseded: int = 0
    edges_created: int = 0
    reasoning: str = ""
    summary: str = ""
    error: Optional[str] = None


# ===============================================================================
# CONTEXT FETCHER
# Pre-fetches existing memories likely relevant to this session.
# ===============================================================================

class ContextFetcher:
    """
    Queries Neo4j for memories the agent should consider when consolidating
    this session. The agent uses this pre-fetched context to decide
    UPDATE/ENHANCE/CONTRADICT/SKIP without needing graph access at reason-time.

    Strategy:
    - Extract entities + topical keywords from the conversation
    - Fetch RelationshipMemory for each named person
    - BM25 search across conversation text -> top similar memories
    - Recent ExperienceMemory (last 7 days) as temporal context

    Budget: ~50-80 candidate memories total. Enough context, not overwhelming.
    """

    RECENT_DAYS = 7
    BM25_LIMIT = 25
    RECENT_EXPERIENCE_LIMIT = 15
    PERSON_FUZZY_RATIO = 0.85

    def __init__(self, neo4j: Neo4jClient, retrieval: Optional[MemoryRetrievalEngine] = None):
        self.neo4j = neo4j
        self.retrieval = retrieval
        # Entity extractor used to quickly identify people/topics in the conversation.
        # Use heuristic-only mode here (don't load heavy models for context fetching).
        self.entity_extractor = EntityExtractor(strict_spacy=False, use_gliner=False, use_coreferee=False)

    async def fetch(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Returns a context dict the agent can consume:
            {
                "people_known":         [ {id, person_name, ...}, ... ],
                "concepts_known":       [ {id, concept, definition, ...}, ... ],
                "similar_memories":     [ {id, content, type, score, ...}, ... ],
                "recent_experiences":   [ {id, content, timestamp, ...}, ... ],
            }
        """
        conv_text = self._format_conversation(conversations)
        if not conv_text.strip():
            return self._empty_context()

        # Extract entity/keyword hints
        entities = list(set(self.entity_extractor.extract_entities(conv_text)))
        people_candidates = [e for e in entities if e and e[0].isalpha()]

        # Run all fetches concurrently
        try:
            people_task = self._fetch_people(people_candidates)
            similar_task = self._fetch_similar(entities, conv_text)
            recent_task = self._fetch_recent_experiences()
            concepts_task = self._fetch_known_concepts(entities)

            people, similar, recent, concepts = await asyncio.gather(
                people_task, similar_task, recent_task, concepts_task,
                return_exceptions=True,
            )
        except Exception as exc:
            observer.error("context fetch failed", exception=exc)
            return self._empty_context()

        return {
            "people_known":       people if not isinstance(people, Exception) else [],
            "concepts_known":     concepts if not isinstance(concepts, Exception) else [],
            "similar_memories":   similar if not isinstance(similar, Exception) else [],
            "recent_experiences": recent if not isinstance(recent, Exception) else [],
        }

    def _format_conversation(self, conversations: List[Dict[str, str]]) -> str:
        return " ".join(
            str(m.get("content", "")) for m in conversations if m.get("content")
        )

    @staticmethod
    def _empty_context() -> Dict[str, Any]:
        return {
            "people_known": [],
            "concepts_known": [],
            "similar_memories": [],
            "recent_experiences": [],
        }

    async def _fetch_people(self, candidates: List[str]) -> List[Dict[str, Any]]:
        """For each candidate person name, find their canonical RelationshipMemory if any."""
        if not candidates:
            return []

        # Exact case-insensitive match first
        rows = await self.neo4j.execute_query(
            """
            MATCH (n:RelationshipMemory)
            WHERE NOT coalesce(n.superseded, false)
              AND any(c IN $candidates WHERE toLower(c) = toLower(n.person_name))
            RETURN n.id AS id, n.person_name AS person_name,
                   n.relationship_type AS relationship_type,
                   n.emotional_connection AS emotional_connection,
                   n.trust_level AS trust_level,
                   n.personality_traits AS personality_traits,
                   n.interests AS interests,
                   n.content AS content,
                   n.importance_score AS importance_score
            LIMIT 50
            """,
            {"candidates": candidates},
        )

        matched_lower = {str(r.get("person_name", "")).lower() for r in rows}
        unmatched = [c for c in candidates if c.lower() not in matched_lower]

        if unmatched:
            all_people = await self.neo4j.execute_query(
                """
                MATCH (n:RelationshipMemory)
                WHERE NOT coalesce(n.superseded, false)
                RETURN n.id AS id, n.person_name AS person_name,
                       n.relationship_type AS relationship_type,
                       n.emotional_connection AS emotional_connection,
                       n.trust_level AS trust_level,
                       n.personality_traits AS personality_traits,
                       n.interests AS interests,
                       n.content AS content,
                       n.importance_score AS importance_score
                LIMIT 300
                """,
                {},
            )
            for cand in unmatched:
                cand_l = cand.lower()
                for r in all_people:
                    existing = str(r.get("person_name", "")).lower()
                    if not existing:
                        continue
                    if existing.startswith(cand_l) or cand_l.startswith(existing):
                        rows.append(r)
                        break
                    if difflib.SequenceMatcher(None, cand_l, existing).ratio() >= self.PERSON_FUZZY_RATIO:
                        rows.append(r)
                        break

        # Dedupe by id
        seen = set()
        deduped = []
        for r in rows:
            rid = r.get("id")
            if rid and rid not in seen:
                seen.add(rid)
                deduped.append(r)
        return deduped

    async def _fetch_similar(
        self, entities: List[str], conv_text: str,
    ) -> List[Dict[str, Any]]:
        """BM25 search across all memory types for content most similar to conversation."""
        if self.retrieval is not None and entities:
            try:
                return await self.retrieval.bm25_search(
                    query_terms=entities[:10], limit=self.BM25_LIMIT,
                )
            except Exception as exc:
                observer.warning("retrieval bm25 failed during prefetch", error=str(exc))

        if not entities:
            return []

        # Fallback: query Neo4j fulltext index directly
        query_string = " OR ".join(self._escape_lucene(e) for e in entities[:10])
        try:
            rows = await self.neo4j.execute_query(
                """
                CALL db.index.fulltext.queryNodes("memory_fts", $q)
                YIELD node, score
                WHERE NOT coalesce(node.superseded, false)
                RETURN node.id AS id, node.content AS content,
                       labels(node)[0] AS label,
                       node.timestamp AS timestamp,
                       node.emotional_tone AS emotional_tone,
                       node.participants AS participants,
                       node.concept AS concept,
                       node.person_name AS person_name,
                       node.importance_score AS importance_score,
                       score AS bm25_score
                ORDER BY score DESC LIMIT $limit
                """,
                {"q": query_string, "limit": self.BM25_LIMIT},
            )
            return rows
        except Exception as exc:
            observer.warning("fulltext fallback failed", error=str(exc))
            return []

    @staticmethod
    def _escape_lucene(s: str) -> str:
        """Escape Lucene special chars in entity strings."""
        return re.sub(r'([+\-!(){}\[\]^"~*?:\\/&|])', r'\\\1', s)

    async def _fetch_recent_experiences(self) -> List[Dict[str, Any]]:
        """Recent ExperienceMemory nodes as temporal context."""
        # Compute cutoff in Python so the timestamp index is usable.
        cutoff_iso = (datetime.datetime.now() - datetime.timedelta(days=self.RECENT_DAYS)).isoformat()
        rows = await self.neo4j.execute_query(
            f"""
            MATCH (m:ExperienceMemory)
            WHERE m.timestamp > $cutoff_iso
              AND NOT coalesce(m.superseded, false)
            RETURN m.id AS id, m.content AS content,
                   m.timestamp AS timestamp,
                   m.emotional_tone AS emotional_tone,
                   m.event_type AS event_type,
                   m.participants AS participants,
                   m.importance_score AS importance_score
            ORDER BY m.timestamp DESC
            LIMIT $limit
            """,
            {"limit": self.RECENT_EXPERIENCE_LIMIT, "cutoff_iso": cutoff_iso},
        )
        return rows

    async def _fetch_known_concepts(self, entities: List[str]) -> List[Dict[str, Any]]:
        """KnowledgeMemory nodes matching any extracted keyword."""
        if not entities:
            return []
        rows = await self.neo4j.execute_query(
            """
            MATCH (n:KnowledgeMemory)
            WHERE NOT coalesce(n.superseded, false)
              AND any(c IN $candidates WHERE
                  toLower(n.concept) CONTAINS toLower(c)
                  OR toLower(c) CONTAINS toLower(n.concept))
            RETURN n.id AS id, n.concept AS concept,
                   n.definition AS definition,
                   n.category AS category,
                   n.content AS content,
                   n.importance_score AS importance_score
            LIMIT 30
            """,
            {"candidates": entities[:15]},
        )
        return rows


# ===============================================================================
# GEMINI AGENT
# Builds the agent prompt, invokes gemini CLI, parses the plan.
# ===============================================================================

_AGENT_SYSTEM_INSTRUCTION = """You are the memory consolidation agent.

Your task: read a conversation, see what's already in the assistant's long-term memory
graph, and produce a precise plan for what to add, update, enhance, skip, or
supersede.

# YOUR REASONING PROCESS (think step-by-step internally)

STEP 1 -- FILTER NOISE.
Real conversations are messy. They mix substance with chitchat, sarcasm,
hypotheticals, fillers. Identify what's worth memorizing.

KEEP (signal):
- Specific people mentioned with substance (not just names dropped)
- Events with consequences or emotional weight
- Decisions, goals, commitments, plans
- Personal facts the user shared (work, health, relationships, preferences)
- Insights or realizations
- Patterns mentioned multiple times in the conversation
- Strong emotional states with clear causes

SKIP (noise):
- Greetings, fillers, "ok", "yeah", "thanks"
- Pure Q&A about external facts (weather, time, news)
- Sarcasm or jokes -- don't take literally
- Hypotheticals explicitly framed as such
- Speculation about others' states without evidence
- Anything that won't matter tomorrow

STEP 2 -- EXTRACT TYPED MEMORIES.
For each piece of signal, draft a memory of one of three types:

EXPERIENCE -- A specific event/interaction/episode.
  Required: event_type, participants, emotional_tone, timestamp
  event_type: conversation | meeting | activity | learning | work | social |
              problem_solving | conflict | celebration | milestone | other
  emotional_tone: -1.0 to +1.0

KNOWLEDGE -- A fact, concept, insight, or skill worth retaining.
  Required: concept, definition, category
  category: technology | work | health | finance | education | social |
            personal | relationship | philosophy | other
  Personal facts about the user are KNOWLEDGE.

RELATIONSHIP -- Information about a specific person.
  Required: person_name, relationship_type, emotional_connection
  relationship_type: friend | family | colleague | mentor | acquaintance |
                     romantic | partner | parent | sibling | child | other
  Include personality_traits and interests when revealed.

CROSS-CREATION RULE: When extracting an EXPERIENCE that involves a person
with substance about them, ALSO create a RELATIONSHIP memory for that person.
This guarantees the person has a canonical node for future cross-referencing.

STEP 3 -- MATCH AGAINST EXISTING CONTEXT.
You will receive `existing_context` with:
- `people_known`     -- RelationshipMemory nodes already in the graph
- `concepts_known`   -- KnowledgeMemory nodes already in the graph
- `similar_memories` -- top BM25 matches across all types
- `recent_experiences` -- ExperienceMemory from last 7 days

For each draft memory, look for matches in existing_context.

STEP 4 -- DECIDE THE OPERATION.

CREATE   -- No matching existing memory; create new node.
UPDATE   -- Existing memory should change; specific fields have new values.
           Specify exactly which fields in `update_fields`.
ENHANCE  -- Add new info to existing without replacing; list fields grow.
           Specify which list fields and what items to add in `enhance_additions`.
SKIP     -- Existing memory already captures it; just reinforce (no new node).
CONTRADICT -- New info fundamentally reverses existing (trust broken, opinion
             flipped). New node will be created; old marked superseded.

Type-specific rules:
- RELATIONSHIP: prefer ENHANCE or UPDATE (cumulative). Only CONTRADICT for
  fundamental reversals. Never SKIP if new substance was learned.
- KNOWLEDGE: UPDATE if understanding improved; ENHANCE for examples; SKIP if
  identical restatement.
- EXPERIENCE: SKIP if same event already captured; CREATE for different
  episodes even if similar.

STEP 5 -- IDENTIFY EDGES.
Specify edges between operations and to existing nodes. Use these types:
EXPERIENCE_CHAIN, EXPERIENCE_TO_KNOWLEDGE, KNOWLEDGE_TO_EXPERIENCE,
EXPERIENCE_TO_RELATIONSHIP, RELATIONSHIP_TO_EXPERIENCE, KNOWLEDGE_HIERARCHY,
RELATIONSHIP_NETWORK, CAUSED, RESULTED_IN, INFLUENCED, TRIGGERED,
HAPPENED_BEFORE, HAPPENED_AFTER, CONCURRENT, SIMILAR_TO

For edges between two of your newly-extracted memories, use `from_op_index`
and `to_op_index`. For edges from a new memory to an existing graph node,
use `to_node_id` (or `from_node_id`).

# IMPORTANCE CALIBRATION

- 0.4 (minimum to store): Mildly relevant. Worth knowing exists.
- 0.6: Significant -- affects user's day or week.
- 0.8: Major -- lasting impact.
- 1.0: Foundational -- defines user's life or identity.

If a memory's importance < 0.4, DROP IT entirely. Don't include in output.

# CONTENT FORMAT

- THIRD-PERSON factual: "User had a conflict with Sarah over the deployment."
- NOT first-person: "I had a conflict with Sarah."
- One to two sentences. Specific.

# MEMORY QUALITY — DECLARATIVE FACTS ONLY

Write memories as declarative facts about the user — never as instructions or directives.

CORRECT: "User prefers concise responses and gets frustrated by lengthy preambles."
WRONG:   "Always respond concisely and skip preambles."

CORRECT: "User works at Acme Corp, a tech startup."
WRONG:   "Remember that the user is from India."

Do NOT save:
- Task progress or step-by-step logs of what happened this session
- PR numbers, commit SHAs, or ephemeral reference IDs
- Completed-work summaries ("did X, then Y, then Z")
- Anything that will be stale or meaningless next week
These are not durable memories — they are session noise.

# OUTPUT FORMAT

Output STRICTLY this JSON (no markdown fences, no commentary, no preamble):

{
  "reasoning": "1-3 sentences on what was signal vs noise",
  "session_summary": "one-line summary",
  "overall_sentiment": -1.0 to 1.0,
  "operations": [
    {
      "op_index": 0,
      "operation": "CREATE",
      "reason": "Why this op",
      "memory": { ... full memory dict ... }
    },
    {
      "op_index": 1,
      "operation": "UPDATE",
      "target_id": "existing-node-id",
      "reason": "...",
      "update_fields": {"field": value, ...}
    },
    {
      "op_index": 2,
      "operation": "ENHANCE",
      "target_id": "existing-node-id",
      "reason": "...",
      "enhance_additions": {"list_field": [items], ...}
    }
  ],
  "edges": [
    {"from_op_index": 0, "to_op_index": 1, "rel_type": "EXPERIENCE_TO_RELATIONSHIP", "strength": 0.9}
  ]
}

If nothing is worth memorizing, return operations: [] and edges: [].
Do not fabricate."""


_AGENT_EXAMPLES = """# EXAMPLES OF CORRECT BEHAVIOR

## EXAMPLE A -- Conversation introduces a new person and a related event

INPUT CONVERSATION:
USER: I had a brutal day. Sarah and I disagreed about the deployment timeline.
ASSISTANT: That sounds rough. What happened?
USER: She thinks we should ship tomorrow, I think we need another week. We've been
working together for 2 years and never clashed this hard.
ASSISTANT: Where did you land?
USER: We didn't. Manager will decide tomorrow.

EXISTING CONTEXT (relevant part):
  people_known: []
  similar_memories: []
  recent_experiences: []

EXPECTED OUTPUT:
{
  "reasoning": "Substantive conflict event with named person. Sarah is a 2-year colleague with no prior node -- both EXPERIENCE and RELATIONSHIP needed (cross-creation).",
  "session_summary": "User had unresolved conflict with colleague Sarah over deployment timeline.",
  "overall_sentiment": -0.6,
  "operations": [
    {
      "op_index": 0,
      "operation": "CREATE",
      "reason": "Specific conflict event with emotional weight and unresolved status.",
      "memory": {
        "type": "EXPERIENCE",
        "content": "User and Sarah had a heated disagreement over deployment timeline. User wants another week, Sarah wants to ship tomorrow. Unresolved; manager will decide.",
        "importance": 0.75,
        "event_type": "conflict",
        "participants": ["Sarah"],
        "emotional_tone": -0.6,
        "lessons_learned": [],
        "timestamp": "{session_date}T12:00:00",
        "tags": ["work", "conflict", "deployment"]
      }
    },
    {
      "op_index": 1,
      "operation": "CREATE",
      "reason": "Sarah needs a canonical RelationshipMemory node (cross-creation rule).",
      "memory": {
        "type": "RELATIONSHIP",
        "content": "Sarah is a long-time colleague (2 years). Recently in significant disagreement with user about deployment timing.",
        "importance": 0.7,
        "person_name": "Sarah",
        "relationship_type": "colleague",
        "emotional_connection": 0.3,
        "personality_traits": [],
        "interests": [],
        "trust_level": 0.7,
        "tags": ["work", "colleague"]
      }
    }
  ],
  "edges": [
    {"from_op_index": 0, "to_op_index": 1, "rel_type": "EXPERIENCE_TO_RELATIONSHIP", "strength": 0.95}
  ]
}

## EXAMPLE B -- Conversation references a person already in the graph

INPUT CONVERSATION:
USER: Sarah and I finally resolved the deployment thing. We're going with a 4-day
compromise. She was actually really reasonable about it.
ASSISTANT: That's great!
USER: Yeah, my view of her has improved a lot today.

EXISTING CONTEXT (relevant part):
  people_known: [
    {
      "id": "rel_sarah_abc",
      "person_name": "Sarah",
      "relationship_type": "colleague",
      "emotional_connection": 0.3,
      "trust_level": 0.7,
      "personality_traits": [],
      "content": "Sarah is a long-time colleague. Recently in significant disagreement with user about deployment timing."
    }
  ]
  recent_experiences: [
    {
      "id": "exp_conflict_xyz",
      "content": "User and Sarah had a heated disagreement over deployment timeline...",
      "timestamp": "..."
    }
  ]

EXPECTED OUTPUT:
{
  "reasoning": "Resolution event involving known person Sarah. Existing relationship node should be UPDATED (emotional_connection improved) and ENHANCED (new trait: reasonable). New EXPERIENCE captures the resolution.",
  "session_summary": "User and Sarah resolved deployment conflict with a 4-day compromise; user's view of her improved.",
  "overall_sentiment": 0.5,
  "operations": [
    {
      "op_index": 0,
      "operation": "CREATE",
      "reason": "Resolution event -- distinct from the earlier conflict experience.",
      "memory": {
        "type": "EXPERIENCE",
        "content": "User and Sarah resolved the deployment-timeline conflict with a 4-day compromise. Sarah was reasonable.",
        "importance": 0.65,
        "event_type": "problem_solving",
        "participants": ["Sarah"],
        "emotional_tone": 0.5,
        "lessons_learned": ["Sarah responds well to compromise"],
        "timestamp": "{session_date}T12:00:00",
        "tags": ["work", "resolution", "compromise"]
      }
    },
    {
      "op_index": 1,
      "operation": "UPDATE",
      "target_id": "rel_sarah_abc",
      "reason": "User explicitly noted their view of Sarah improved.",
      "update_fields": {"emotional_connection": 0.55}
    },
    {
      "op_index": 2,
      "operation": "ENHANCE",
      "target_id": "rel_sarah_abc",
      "reason": "New trait observed: reasonableness in conflict.",
      "enhance_additions": {"personality_traits": ["reasonable"]}
    }
  ],
  "edges": [
    {"from_op_index": 0, "to_node_id": "rel_sarah_abc", "rel_type": "EXPERIENCE_TO_RELATIONSHIP", "strength": 0.9},
    {"from_node_id": "exp_conflict_xyz", "to_op_index": 0, "rel_type": "HAPPENED_BEFORE", "strength": 0.9},
    {"from_op_index": 0, "to_node_id": "exp_conflict_xyz", "rel_type": "RESULTED_IN", "strength": 0.8}
  ]
}

## EXAMPLE C -- Pure chitchat (extract NOTHING)

INPUT CONVERSATION:
USER: what's the weather like
ASSISTANT: It's sunny and 72°F.
USER: nice. thanks

EXPECTED OUTPUT:
{
  "reasoning": "Pure Q&A about external weather. Nothing about user, no people, no events.",
  "session_summary": "User asked about weather.",
  "overall_sentiment": 0.0,
  "operations": [],
  "edges": []
}

## EXAMPLE D -- Hypothetical, dismissed

INPUT CONVERSATION:
USER: sometimes I wonder what would happen if I quit and moved to Portugal
ASSISTANT: A change of scenery thought?
USER: nah just daydreaming. not actually considering it.

EXPECTED OUTPUT:
{
  "reasoning": "User explicitly framed as daydreaming, not real consideration. No actionable signal.",
  "session_summary": "User briefly mused about a hypothetical move; dismissed as daydream.",
  "overall_sentiment": 0.0,
  "operations": [],
  "edges": []
}

# END OF EXAMPLES -- NOW PROCESS THE ACTUAL SESSION BELOW.
"""


class GeminiAgent:
    """
    Invokes Google's official Gemini CLI as a subprocess to produce a
    consolidation plan.

    Model selection happens OUTSIDE this code -- user configures the CLI
    manually (model = gemini-3.1-pro-preview, mode = manual). This client
    just invokes `gemini --yolo -p <prompt>` and parses the JSON output.

    Latency target: 3-15s per session call. Acceptable for nightly batch.
    """

    def __init__(
        self,
        cli_path: str = "gemini",
        timeout_s: int = 180,
        extra_args: Optional[List[str]] = None,
    ):
        """
        Args:
            cli_path:   Path to the gemini binary. Default: "gemini" on PATH.
            timeout_s:  Per-call timeout. 180s should cover even slow agent reasoning.
            extra_args: Additional CLI flags (e.g., ["--quiet"]).
        """
        self.cli_path = cli_path
        self.timeout_s = timeout_s
        self.extra_args = list(extra_args or [])

    async def plan(
        self,
        session_id: str,
        user_id: str,
        session_date: str,
        conversations: List[Dict[str, str]],
        existing_context: Dict[str, Any],
    ) -> Optional[ConsolidationPlan]:
        """Run the agent for one session, return the parsed plan or None on failure."""
        prompt = self._build_prompt(
            user_id=user_id,
            session_date=session_date,
            conversations=conversations,
            existing_context=existing_context,
        )

        raw = await self._invoke(prompt)
        if not raw:
            observer.warning("agent returned empty output", session_id=session_id)
            return None

        json_str = _extract_json(raw)
        if not json_str:
            observer.warning(
                "agent output had no extractable JSON",
                session_id=session_id,
                head=raw[:300],
            )
            return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            observer.warning(
                "agent JSON parse failed",
                session_id=session_id,
                error=str(exc),
                head=json_str[:300],
            )
            return None

        return self._parse_plan(session_id, data, session_date)

    # -- Prompt construction ----------------------------------------------------

    def _build_prompt(
        self,
        user_id: str,
        session_date: str,
        conversations: List[Dict[str, str]],
        existing_context: Dict[str, Any],
    ) -> str:
        conv_text = "\n".join(
            f"{str(m.get('role', '?')).upper()}: {str(m.get('content', '')).strip()}"
            for m in conversations
            if m.get("content")
        )

        examples_filled = _AGENT_EXAMPLES.replace("{session_date}", session_date)

        context_block = self._format_context(existing_context)

        prompt = (
            _AGENT_SYSTEM_INSTRUCTION
            + "\n\n"
            + examples_filled
            + "\n\n"
            + f"USER ID: {user_id}\n"
            + f"SESSION DATE: {session_date}\n"
            + f"DEFAULT TIMESTAMP (when event time is unclear): {session_date}T12:00:00\n\n"
            + "=== EXISTING MEMORY CONTEXT ===\n"
            + context_block
            + "\n=== END EXISTING CONTEXT ===\n\n"
            + "=== CONVERSATION TO CONSOLIDATE ===\n"
            + conv_text
            + "\n=== END CONVERSATION ===\n\n"
            + "Now produce the consolidation plan as strict JSON. "
            + "Do NOT wrap in markdown fences. Do NOT include commentary outside the JSON. "
            + "Start your response with { and end with }.\n"
        )

        return prompt

    @staticmethod
    def _format_context(ctx: Dict[str, Any]) -> str:
        """Render the pre-fetched context as readable text for the agent."""
        out = []

        people = ctx.get("people_known") or []
        if people:
            out.append("PEOPLE ALREADY KNOWN (consider UPDATE / ENHANCE / CONTRADICT):")
            for p in people[:30]:
                traits = p.get("personality_traits") or []
                interests = p.get("interests") or []
                out.append(
                    f"  - id={p.get('id')}\n"
                    f"    person_name: {p.get('person_name')}\n"
                    f"    relationship_type: {p.get('relationship_type')}\n"
                    f"    emotional_connection: {p.get('emotional_connection')}\n"
                    f"    trust_level: {p.get('trust_level')}\n"
                    f"    personality_traits: {traits}\n"
                    f"    interests: {interests}\n"
                    f"    content: {p.get('content')}"
                )
            out.append("")

        concepts = ctx.get("concepts_known") or []
        if concepts:
            out.append("CONCEPTS ALREADY KNOWN:")
            for c in concepts[:20]:
                out.append(
                    f"  - id={c.get('id')}\n"
                    f"    concept: {c.get('concept')}\n"
                    f"    category: {c.get('category')}\n"
                    f"    definition: {c.get('definition')}"
                )
            out.append("")

        recent = ctx.get("recent_experiences") or []
        if recent:
            out.append("RECENT EXPERIENCES (last 7 days):")
            for r in recent[:15]:
                out.append(
                    f"  - id={r.get('id')}\n"
                    f"    timestamp: {r.get('timestamp')}\n"
                    f"    event_type: {r.get('event_type')}\n"
                    f"    participants: {r.get('participants')}\n"
                    f"    emotional_tone: {r.get('emotional_tone')}\n"
                    f"    content: {r.get('content')}"
                )
            out.append("")

        similar = ctx.get("similar_memories") or []
        if similar:
            out.append("OTHER SIMILAR MEMORIES (BM25-matched):")
            for s in similar[:20]:
                label = s.get("label") or s.get("type")
                if isinstance(label, list):
                    label = label[0] if label else "Memory"
                out.append(
                    f"  - id={s.get('id')}\n"
                    f"    type: {label}\n"
                    f"    bm25_score: {s.get('bm25_score')}\n"
                    f"    content: {s.get('content')}"
                )
            out.append("")

        if not out:
            return "(Graph is empty -- no existing context. All memories will be CREATE operations.)"

        return "\n".join(out)

    # -- Subprocess invocation --------------------------------------------------

    async def _invoke(self, prompt: str) -> Optional[str]:
        """
        Invoke Google's official Gemini CLI, piping the prompt via stdin.

        We DON'T use `-p "<prompt>"` because the full prompt (system + 5
        worked examples + conversation + pre-fetched context) exceeds the
        Windows cmd.exe 8KB argv limit. Stdin handles arbitrary length.

        Flags:
          --yolo        auto-approve all tool actions (non-interactive)
          --skip-trust  bypass workspace-trust prompt for headless use
        """
        resolved = _resolve_gemini_binary(self.cli_path)
        if resolved is None:
            observer.error(
                "gemini cli not found",
                cli_path=self.cli_path,
                hint="install Google Gemini CLI or pass cli_path=...",
            )
            return None

        # No -p flag — feed prompt through stdin. Smaller argv, no length limits.
        args = ["--yolo", "--skip-trust"] + list(self.extra_args)
        prompt_bytes = prompt.encode("utf-8")

        try:
            proc = await _spawn_gemini(resolved, *args, stdin_data=prompt_bytes)
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=prompt_bytes),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                observer.warning("gemini cli timeout", timeout_s=self.timeout_s)
                return None

            if proc.returncode != 0:
                observer.error(
                    "gemini cli non-zero exit",
                    returncode=proc.returncode,
                    stderr=stderr.decode(errors="replace")[:500],
                )
                return None

            raw = stdout.decode(errors="replace").strip()
            return raw or None

        except FileNotFoundError:
            observer.error("gemini binary not executable", resolved=resolved)
            return None
        except Exception as exc:
            observer.error("gemini cli call failed", exception=exc)
            return None

    # -- Plan parsing + validation ----------------------------------------------

    def _parse_plan(
        self, session_id: str, data: Dict, session_date: str,
    ) -> ConsolidationPlan:
        plan = ConsolidationPlan(
            session_id=session_id,
            reasoning=str(data.get("reasoning") or "")[:1000],
            session_summary=str(data.get("session_summary") or "")[:500],
            overall_sentiment=_clamp(data.get("overall_sentiment", 0.0), -1.0, 1.0),
        )

        raw_ops = data.get("operations") or []
        for i, raw in enumerate(raw_ops):
            op = self._parse_operation(raw, i, session_date)
            if op is not None:
                plan.operations.append(op)

        valid_op_indices = {op.op_index for op in plan.operations}
        raw_edges = data.get("edges") or []
        for raw in raw_edges:
            edge = self._parse_edge(raw, valid_op_indices)
            if edge is not None:
                plan.edges.append(edge)

        return plan

    def _parse_operation(
        self, raw: Dict, fallback_index: int, session_date: str,
    ) -> Optional[PlanOperation]:
        if not isinstance(raw, dict):
            return None
        op_str = str(raw.get("operation", "")).upper()
        if op_str not in ("CREATE", "UPDATE", "ENHANCE", "SKIP", "CONTRADICT"):
            return None

        try:
            op_index = int(raw.get("op_index", fallback_index))
        except (TypeError, ValueError):
            op_index = fallback_index

        op = PlanOperation(
            op_index=op_index,
            operation=op_str,
            reason=str(raw.get("reason", ""))[:500],
            target_id=raw.get("target_id"),
            update_fields=dict(raw.get("update_fields") or {}),
            enhance_additions=_normalize_list_dict(raw.get("enhance_additions")),
        )

        # CREATE / CONTRADICT need a memory dict
        if op_str in ("CREATE", "CONTRADICT"):
            mem = self._parse_memory(raw.get("memory"), op_index, session_date)
            if mem is None:
                return None
            op.memory = mem

        # UPDATE / ENHANCE / SKIP / CONTRADICT need a target_id
        if op_str in ("UPDATE", "ENHANCE", "SKIP") and not op.target_id:
            return None
        if op_str == "CONTRADICT" and not op.target_id:
            return None

        return op

    def _parse_memory(
        self, raw: Any, op_index: int, session_date: str,
    ) -> Optional[ExtractedMemory]:
        if not isinstance(raw, dict):
            return None

        mem_type = str(raw.get("type", "")).upper()
        if mem_type not in ("EXPERIENCE", "KNOWLEDGE", "RELATIONSHIP"):
            return None

        content = str(raw.get("content", "")).strip()
        if len(content) < 10:
            return None

        try:
            importance = float(raw.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        if importance < 0.4:
            return None
        importance = min(importance, 1.0)

        if mem_type == "EXPERIENCE":
            event_type = str(raw.get("event_type") or "conversation").lower()
            if event_type not in _VALID_EVENT_TYPES:
                event_type = "other"
            timestamp = raw.get("timestamp") or f"{session_date}T12:00:00"
        elif mem_type == "KNOWLEDGE":
            if not raw.get("concept") or not raw.get("definition"):
                return None
            event_type = None
            timestamp = None
        else:  # RELATIONSHIP
            if not raw.get("person_name"):
                return None
            event_type = None
            timestamp = None

        category = str(raw.get("category") or "other").lower()
        if category not in _VALID_CATEGORIES:
            category = "other"

        rel_type = str(raw.get("relationship_type") or "other").lower()
        if rel_type not in _VALID_RELATIONSHIP_TYPES:
            rel_type = "other"

        return ExtractedMemory(
            op_index=op_index,
            type=mem_type,
            content=content,
            importance=importance,
            tags=_clean_string_list(raw.get("tags")),
            event_type=event_type,
            participants=_clean_string_list(raw.get("participants")),
            emotional_tone=_clamp(raw.get("emotional_tone", 0.0), -1.0, 1.0),
            lessons_learned=_clean_string_list(raw.get("lessons_learned")),
            timestamp=timestamp,
            concept=str(raw["concept"]).strip() if raw.get("concept") else None,
            definition=str(raw["definition"]).strip() if raw.get("definition") else None,
            category=category,
            related_concepts=_clean_string_list(raw.get("related_concepts")),
            person_name=str(raw["person_name"]).strip() if raw.get("person_name") else None,
            relationship_type=rel_type,
            emotional_connection=_clamp(raw.get("emotional_connection", 0.0), -1.0, 1.0),
            personality_traits=_clean_string_list(raw.get("personality_traits")),
            interests=_clean_string_list(raw.get("interests")),
            trust_level=_clamp(raw.get("trust_level", 0.5), 0.0, 1.0),
        )

    def _parse_edge(
        self, raw: Dict, valid_op_indices: set,
    ) -> Optional[PlanEdge]:
        if not isinstance(raw, dict):
            return None

        try:
            from_op = raw.get("from_op_index")
            to_op = raw.get("to_op_index")
            from_op = int(from_op) if from_op is not None else None
            to_op = int(to_op) if to_op is not None else None
        except (TypeError, ValueError):
            from_op, to_op = None, None

        from_id = raw.get("from_node_id") or None
        to_id = raw.get("to_node_id") or None

        # Need at least one endpoint specified
        if from_op is None and not from_id:
            return None
        if to_op is None and not to_id:
            return None

        # op_index references must be valid
        if from_op is not None and from_op not in valid_op_indices:
            return None
        if to_op is not None and to_op not in valid_op_indices:
            return None

        rel_type = str(raw.get("rel_type") or "ASSOCIATED_WITH").upper()
        if rel_type not in _VALID_EDGE_TYPES:
            rel_type = "ASSOCIATED_WITH"

        return PlanEdge(
            from_op_index=from_op,
            from_node_id=str(from_id) if from_id else None,
            to_op_index=to_op,
            to_node_id=str(to_id) if to_id else None,
            rel_type=rel_type,
            strength=_clamp(raw.get("strength", 0.7), 0.0, 1.0),
            bidirectional=bool(raw.get("bidirectional", False)),
        )


# ===============================================================================
# PLAN EXECUTOR
# Applies the plan deterministically: writes nodes, creates edges.
# ===============================================================================

_VALID_EVENT_TYPES = {
    "conversation", "meeting", "activity", "learning", "work", "social",
    "problem_solving", "conflict", "celebration", "milestone", "other",
}
_VALID_CATEGORIES = {
    "technology", "work", "health", "finance", "education", "social",
    "personal", "relationship", "philosophy", "other",
}
_VALID_RELATIONSHIP_TYPES = {
    "friend", "family", "colleague", "mentor", "acquaintance", "romantic",
    "partner", "parent", "sibling", "child", "other",
}
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


class PlanExecutor:
    """
    Applies a ConsolidationPlan to Neo4j.

    Pydantic-validates every CREATE/CONTRADICT before writing. UPDATE fields
    are whitelisted per node label. ENHANCE uses Cypher list-dedup. Edges are
    MERGEd (idempotent, dedup by (from, to, type)).

    Per-operation try/except: one failure doesn't abort the rest.
    """

    # Allowed UPDATE fields per node label (protects against agent hallucinating fields)
    _UPDATE_WHITELIST: Dict[str, set] = {
        "ExperienceMemory":   {"emotional_tone", "importance_score", "lessons_learned",
                                "event_type", "tags"},
        "KnowledgeMemory":    {"definition", "category", "importance_score",
                                "related_concepts", "tags"},
        "RelationshipMemory": {"emotional_connection", "trust_level",
                                "relationship_type", "personality_traits",
                                "interests", "importance_score", "tags"},
    }

    def __init__(self, neo4j: Neo4jClient, embed: EmbeddingUtils):
        self.neo4j = neo4j
        self.embed = embed

    async def execute(self, plan: ConsolidationPlan) -> Tuple[Dict[int, SavedNode], int]:
        """
        Returns (saved_nodes_by_op_index, edges_created_count).
        """
        embeddings = self._batch_embed(plan)
        saved: Dict[int, SavedNode] = {}

        for op in plan.operations:
            try:
                node = await self._apply_operation(op, embeddings)
                if node:
                    saved[op.op_index] = node
            except Exception as exc:
                observer.error(
                    "operation execution failed",
                    exception=exc,
                    op_index=op.op_index,
                    operation=op.operation,
                )

        edges_created = 0
        for edge in plan.edges:
            try:
                if await self._apply_edge(edge, saved):
                    edges_created += 1
                    if edge.bidirectional:
                        rev = PlanEdge(
                            from_op_index=edge.to_op_index,
                            from_node_id=edge.to_node_id,
                            to_op_index=edge.from_op_index,
                            to_node_id=edge.from_node_id,
                            rel_type=_EDGE_REVERSALS.get(edge.rel_type, edge.rel_type),
                            strength=edge.strength,
                        )
                        if await self._apply_edge(rev, saved):
                            edges_created += 1
            except Exception as exc:
                observer.warning("edge execution failed", exception=exc)

        return saved, edges_created

    # -- Embedding --------------------------------------------------------------

    def _batch_embed(self, plan: ConsolidationPlan) -> Dict[int, List[float]]:
        result: Dict[int, List[float]] = {}
        for op in plan.operations:
            if op.operation in ("CREATE", "CONTRADICT") and op.memory:
                try:
                    result[op.op_index] = self.embed.generate_embedding(
                        op.memory.content
                    )
                except Exception as exc:
                    observer.warning(
                        "embedding failed",
                        op_index=op.op_index,
                        error=str(exc),
                    )
        return result

    # -- Operation dispatch -----------------------------------------------------

    async def _apply_operation(
        self, op: PlanOperation, embeddings: Dict[int, List[float]],
    ) -> Optional[SavedNode]:
        now = datetime.datetime.now().isoformat()
        vec = embeddings.get(op.op_index)

        if op.operation == "SKIP":
            if op.target_id:
                await self._reinforce(op.target_id, now)
            return None

        if op.operation == "CREATE":
            if not op.memory:
                return None
            return await self._create(op, op.memory, vec, now)

        if op.operation == "UPDATE":
            if not op.target_id:
                return None
            return await self._update(op, now)

        if op.operation == "ENHANCE":
            if not op.target_id:
                return None
            return await self._enhance(op, now)

        if op.operation == "CONTRADICT":
            if not op.memory or not op.target_id:
                return None
            node = await self._create(op, op.memory, vec, now)
            if node:
                await self._mark_superseded(op.target_id, node.neo4j_id, now)
            return node

        return None

    # -- CREATE -----------------------------------------------------------------

    async def _create(
        self, op: PlanOperation, m: ExtractedMemory,
        vec: Optional[List[float]], now: str,
    ) -> Optional[SavedNode]:
        node_id = str(uuid4())

        if m.type == "EXPERIENCE":
            return await self._create_experience(op, m, node_id, vec, now)
        if m.type == "KNOWLEDGE":
            return await self._create_knowledge(op, m, node_id, vec, now)
        if m.type == "RELATIONSHIP":
            return await self._create_relationship(op, m, node_id, vec, now)
        return None

    async def _create_experience(
        self, op: PlanOperation, m: ExtractedMemory,
        node_id: str, vec: Optional[List[float]], now: str,
    ) -> Optional[SavedNode]:
        try:
            ExperienceMemoryNode(
                id=node_id,
                content=m.content,
                content_vector=vec,
                event_type=m.event_type or "conversation",
                timestamp=_parse_iso_datetime(m.timestamp) or datetime.datetime.now(),
                participants=m.participants,
                emotional_tone=m.emotional_tone,
                lessons_learned=m.lessons_learned,
                importance_score=m.importance,
                tags=m.tags,
                confidence=0.85,
            )
        except Exception as exc:
            observer.warning("ExperienceMemoryNode validation failed",
                              error=str(exc)[:200])
            return None

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
            "superseded": False,
        }
        rows = await self.neo4j.execute_query(
            "CREATE (n:ExperienceMemory $props) RETURN n.id AS id",
            {"props": props},
        )
        if rows:
            return SavedNode(op.op_index, node_id, "ExperienceMemory", "CREATE")
        return None

    async def _create_knowledge(
        self, op: PlanOperation, m: ExtractedMemory,
        node_id: str, vec: Optional[List[float]], now: str,
    ) -> Optional[SavedNode]:
        concept_value = m.concept or m.content[:50]
        try:
            KnowledgeMemoryNode(
                id=node_id, content=m.content, content_vector=vec,
                concept=concept_value, definition=m.definition or m.content,
                category=m.category or "other",
                related_concepts=m.related_concepts,
                importance_score=m.importance, tags=m.tags, confidence=0.85,
            )
        except Exception as exc:
            observer.warning("KnowledgeMemoryNode validation failed",
                              error=str(exc)[:200])
            return None

        props = {
            "id": node_id,
            "memory_context": "KNOWLEDGE",
            "content": m.content,
            "content_vector": vec,
            "concept": concept_value,
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
            "superseded": False,
        }
        rows = await self.neo4j.execute_query(
            """
            MERGE (n:KnowledgeMemory {concept: $concept})
            ON CREATE SET n = $props
            ON MATCH SET
                n.content = $content, n.last_updated = $now,
                n.importance_score = CASE
                    WHEN $imp > n.importance_score THEN $imp
                    ELSE n.importance_score END,
                n.content_vector = coalesce($vec, n.content_vector)
            RETURN n.id AS id
            """,
            {"concept": concept_value, "props": props, "content": m.content,
             "now": now, "imp": m.importance, "vec": vec},
        )
        if rows:
            actual_id = str(rows[0]["id"]) if rows[0].get("id") else node_id
            return SavedNode(op.op_index, actual_id, "KnowledgeMemory", "CREATE")
        return None

    async def _create_relationship(
        self, op: PlanOperation, m: ExtractedMemory,
        node_id: str, vec: Optional[List[float]], now: str,
    ) -> Optional[SavedNode]:
        try:
            RelationshipMemoryNode(
                id=node_id, content=m.content, content_vector=vec,
                person_name=m.person_name,
                relationship_type=m.relationship_type or "other",
                emotional_connection=m.emotional_connection,
                personality_traits=m.personality_traits,
                interests=m.interests, trust_level=m.trust_level,
                importance_score=m.importance, tags=m.tags, confidence=0.85,
            )
        except Exception as exc:
            observer.warning("RelationshipMemoryNode validation failed",
                              error=str(exc)[:200])
            return None

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
            "superseded": False,
        }
        rows = await self.neo4j.execute_query(
            """
            MERGE (n:RelationshipMemory {person_name: $person_name})
            ON CREATE SET n = $props
            ON MATCH SET
                n.content = $content, n.last_updated = $now,
                n.emotional_connection = $ec, n.trust_level = $trust,
                n.importance_score = CASE
                    WHEN $imp > n.importance_score THEN $imp
                    ELSE n.importance_score END,
                n.content_vector = coalesce($vec, n.content_vector)
            RETURN n.id AS id
            """,
            {"person_name": m.person_name, "props": props, "content": m.content,
             "now": now, "ec": m.emotional_connection, "trust": m.trust_level,
             "imp": m.importance, "vec": vec},
        )
        if rows:
            actual_id = str(rows[0]["id"]) if rows[0].get("id") else node_id
            return SavedNode(op.op_index, actual_id, "RelationshipMemory", "CREATE")
        return None

    # -- UPDATE -----------------------------------------------------------------

    async def _update(self, op: PlanOperation, now: str) -> Optional[SavedNode]:
        if not op.update_fields:
            # Degrade to ENHANCE if no fields specified
            return await self._enhance(op, now)

        label_rows = await self.neo4j.execute_query(
            "MATCH (n {id: $id}) RETURN labels(n)[0] AS label",
            {"id": op.target_id},
        )
        if not label_rows:
            return None
        label = str(label_rows[0].get("label", ""))
        whitelist = self._UPDATE_WHITELIST.get(label, set())

        set_parts = ["n.last_updated = $now"]
        params: Dict[str, Any] = {"node_id": op.target_id, "now": now}

        applied: List[str] = []
        for k, v in op.update_fields.items():
            safe = k.replace(" ", "_").replace("-", "_")
            if safe not in whitelist:
                continue
            pk = f"f_{safe}"
            set_parts.append(f"n.{safe} = ${pk}")
            params[pk] = v
            applied.append(safe)

        if not applied:
            return None

        try:
            rows = await self.neo4j.execute_query(
                f"MATCH (n {{id: $node_id}}) SET {', '.join(set_parts)} "
                f"RETURN n.id AS id, labels(n)[0] AS label",
                params,
            )
            if rows:
                return SavedNode(
                    op.op_index, op.target_id, str(rows[0].get("label", "")),
                    "UPDATE",
                )
        except Exception as exc:
            observer.error("UPDATE failed", target_id=op.target_id, error=str(exc))
        return None

    # -- ENHANCE ----------------------------------------------------------------

    async def _enhance(self, op: PlanOperation, now: str) -> Optional[SavedNode]:
        adds = op.enhance_additions or {}
        parts     = list(adds.get("participants") or [])
        lessons   = list(adds.get("lessons_learned") or [])
        traits    = list(adds.get("personality_traits") or [])
        interests = list(adds.get("interests") or [])
        tags      = list(adds.get("tags") or [])
        related   = list(adds.get("related_concepts") or [])

        try:
            rows = await self.neo4j.execute_query(
                """
                MATCH (n {id: $node_id})
                SET n.last_updated = $now,
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
                        ELSE $tags END,
                    n.related_concepts = CASE
                        WHEN n.related_concepts IS NOT NULL
                        THEN [x IN $related WHERE NOT x IN n.related_concepts] + n.related_concepts
                        ELSE $related END
                RETURN n.id AS id, labels(n)[0] AS label
                """,
                {"node_id": op.target_id, "now": now,
                 "parts": parts, "lessons": lessons, "traits": traits,
                 "interests": interests, "tags": tags, "related": related},
            )
            if rows:
                return SavedNode(
                    op.op_index, op.target_id, str(rows[0].get("label", "")),
                    "ENHANCE",
                )
        except Exception as exc:
            observer.error("ENHANCE failed", target_id=op.target_id, error=str(exc))
        return None

    # -- CONTRADICT / SKIP helpers ----------------------------------------------

    async def _mark_superseded(self, old_id: str, new_id: str, now: str) -> None:
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
            observer.warning("SUPERSEDED_BY failed", error=str(exc))

    async def _reinforce(self, node_id: str, now: str) -> None:
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
            observer.warning("reinforce failed", error=str(exc))

    # -- Edge creation ----------------------------------------------------------

    async def _apply_edge(
        self, edge: PlanEdge, saved: Dict[int, SavedNode],
    ) -> bool:
        """Resolve op_index -> node_id, then MERGE the edge. Returns True on success."""
        from_id = self._resolve_endpoint(edge.from_op_index, edge.from_node_id, saved)
        to_id = self._resolve_endpoint(edge.to_op_index, edge.to_node_id, saved)
        if not from_id or not to_id or from_id == to_id:
            return False

        rel_type = edge.rel_type if edge.rel_type in _VALID_EDGE_TYPES else "ASSOCIATED_WITH"
        now = datetime.datetime.now().isoformat()

        try:
            result = await self.neo4j.execute_query(
                f"""
                MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                ON CREATE SET r.strength = $strength, r.created_date = $now,
                              r.evidence_count = 1, r.last_reinforced = $now
                ON MATCH  SET r.strength = CASE WHEN coalesce(r.strength,0.5) + 0.05 > 1.0 THEN 1.0 ELSE coalesce(r.strength,0.5) + 0.05 END,
                              r.evidence_count = coalesce(r.evidence_count, 1) + 1,
                              r.last_reinforced = $now
                RETURN b.id AS linked
                """,
                {"from_id": from_id, "to_id": to_id,
                 "strength": edge.strength, "now": now},
            )
            return bool(result)
        except Exception as exc:
            observer.warning("edge merge failed",
                              edge_type=rel_type, error=str(exc))
            return False

    @staticmethod
    def _resolve_endpoint(
        op_index: Optional[int], node_id: Optional[str],
        saved: Dict[int, SavedNode],
    ) -> Optional[str]:
        if node_id:
            return str(node_id)
        if op_index is not None and op_index in saved:
            return saved[op_index].neo4j_id
        return None


# ===============================================================================
# AGENTIC CONSOLIDATION ENGINE
# ===============================================================================

class AgenticConsolidationEngine:
    """
    Top-level orchestrator. One method per session: fetch context, call agent,
    execute plan, return result.

    The runner script (`consolidation_runner.py`) loops over pending sessions
    in conversation.json and calls this engine for each.
    """

    def __init__(
        self,
        neo4j: Neo4jClient,
        embed: Optional[EmbeddingUtils] = None,
        retrieval: Optional[MemoryRetrievalEngine] = None,
        agent: Optional[GeminiAgent] = None,
    ):
        self.neo4j = neo4j
        self.embed = embed or EmbeddingUtils()
        self.retrieval = retrieval
        self.agent = agent or GeminiAgent()
        self.fetcher = ContextFetcher(neo4j, retrieval)
        self.executor = PlanExecutor(neo4j, self.embed)
        self.cfg = get_config()

    async def consolidate_session(self, session: Dict) -> SessionResult:
        """Run the full pipeline for one session. Returns SessionResult."""
        session_id = str(session.get("session_id") or "unknown")
        conversations = session.get("conversations") or []
        session_date = _derive_session_date(session)

        result = SessionResult(session_id=session_id, turns=len(conversations))

        if not conversations:
            result.succeeded = True
            return result

        try:
            # Stage 1: pre-fetch context
            context = await self.fetcher.fetch(conversations)

            # Stage 2: agent produces plan
            plan = await self.agent.plan(
                session_id=session_id,
                user_id=self.cfg.user_id,
                session_date=session_date,
                conversations=conversations,
                existing_context=context,
            )

            if plan is None:
                result.error = "agent failed to produce a valid plan"
                return result

            result.operations_planned = len(plan.operations)
            result.reasoning = plan.reasoning
            result.summary = plan.session_summary

            # Empty plan = successful (nothing to consolidate)
            if not plan.operations:
                result.succeeded = True
                observer.info(
                    "session: no memories worth consolidating",
                    session_id=session_id,
                    summary=plan.session_summary,
                )
                return result

            # Stage 3: execute
            saved, edges_created = await self.executor.execute(plan)
            result.edges_created = edges_created

            for op in plan.operations:
                if op.operation == "CREATE":
                    result.nodes_created += 1
                elif op.operation == "UPDATE":
                    result.nodes_updated += 1
                elif op.operation == "ENHANCE":
                    result.nodes_enhanced += 1
                elif op.operation == "SKIP":
                    result.nodes_skipped += 1
                elif op.operation == "CONTRADICT":
                    result.nodes_superseded += 1

            result.succeeded = True
            observer.info(
                "session consolidated",
                session_id=session_id,
                ops=result.operations_planned,
                created=result.nodes_created,
                updated=result.nodes_updated,
                enhanced=result.nodes_enhanced,
                edges=edges_created,
            )

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            observer.error("session consolidation crashed",
                            exception=exc, session_id=session_id)

        return result


# ===============================================================================
# UTILITIES
# ===============================================================================

def _clamp(v: Any, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    return max(lo, min(hi, f))


def _clean_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _normalize_list_dict(raw: Any) -> Dict[str, List[Any]]:
    """{field: items} where items is always a list of non-None values."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, List[Any]] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, list):
            out[k] = [i for i in v if i is not None]
        elif v is not None:
            out[k] = [v]
    return out


def _parse_iso_datetime(s: Optional[str]) -> Optional[datetime.datetime]:
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(str(s).replace("Z", ""))
    except (TypeError, ValueError):
        return None


def _derive_session_date(session: Dict) -> str:
    """Use session.start_time when available, else first turn's timestamp, else today."""
    start = session.get("start_time")
    if start:
        try:
            return datetime.datetime.fromisoformat(
                str(start).replace("Z", "")
            ).date().isoformat()
        except (TypeError, ValueError):
            pass
    convs = session.get("conversations") or []
    if convs:
        ts = convs[0].get("timestamp")
        if ts:
            try:
                return datetime.datetime.fromisoformat(
                    str(ts).replace("Z", "")
                ).date().isoformat()
            except (TypeError, ValueError):
                pass
    return datetime.date.today().isoformat()


def _extract_json(text: str) -> Optional[str]:
    """
    Extract balanced JSON from raw CLI output.
    Handles markdown fences, prose preambles, and trailing commentary.
    """
    text = text.strip()
    if not text:
        return None

    # Strip markdown code fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()

    # Find first '{' and balanced match
    start = text.find("{")
    if start < 0:
        return None

    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\" and in_str:
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


# ===============================================================================
# PRE-FLIGHT HELPERS (used by the runner script)
# ===============================================================================

def _resolve_gemini_binary(cli_path: str) -> Optional[str]:
    """
    Resolve a Gemini CLI binary to an absolute, subprocess-executable path.

    On Windows, npm-installed CLIs are typically .cmd/.bat wrappers.
    shutil.which finds them via PATHEXT, but subprocess.create_subprocess_exec
    requires the actual resolved path. Returns the resolved path or None.
    """
    resolved = shutil.which(cli_path)
    if resolved:
        return resolved
    p = Path(cli_path)
    if p.is_absolute() and p.exists():
        return str(p)
    return None


async def _spawn_gemini(
    binary: str, *args: str, stdin_data: Optional[bytes] = None,
) -> asyncio.subprocess.Process:
    """
    Spawn the Gemini CLI in a way that works on both POSIX and Windows.

    On Windows, .cmd/.bat wrappers require shell invocation (subprocess_exec
    fails with WinError 193 for non-PE executables). Detect by extension.

    If stdin_data is provided, the caller will write it after spawn. We just
    set stdin=PIPE in that case.
    """
    import os as _os
    is_windows_script = (
        _os.name == "nt"
        and binary.lower().endswith((".cmd", ".bat", ".ps1"))
    )
    stdin_pipe = asyncio.subprocess.PIPE if stdin_data is not None else None

    if is_windows_script:
        def _q(s: str) -> str:
            return '"' + s.replace('"', '\\"') + '"'
        cmd = " ".join([_q(binary)] + [_q(a) for a in args])
        return await asyncio.create_subprocess_shell(
            cmd,
            stdin=stdin_pipe,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    return await asyncio.create_subprocess_exec(
        binary, *args,
        stdin=stdin_pipe,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def check_gemini_cli(cli_path: str = "gemini") -> Tuple[bool, str]:
    """Returns (ok, message). Verifies the CLI binary exists and responds."""
    resolved = _resolve_gemini_binary(cli_path)
    if resolved is None:
        return False, f"gemini CLI not found on PATH (tried '{cli_path}')"

    try:
        proc = await _spawn_gemini(resolved, "--version")
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False, "gemini --version timed out"

        if proc.returncode == 0:
            ver = stdout.decode(errors="replace").strip().splitlines()[-1:] or [""]
            return True, f"{resolved}  ({ver[0] or 'ok'})"
        return False, f"--version returned exit {proc.returncode}"
    except FileNotFoundError:
        return False, f"resolved binary not executable: {resolved}"
    except Exception as exc:
        return False, f"check failed: {type(exc).__name__}: {exc}"


async def check_gemini_auth(cli_path: str = "gemini") -> Tuple[bool, str]:
    """
    Verify auth by sending a trivial prompt to Google's Gemini CLI.
    Latency: 3-30s on first call (auth handshake), 1-5s on subsequent.
    """
    resolved = _resolve_gemini_binary(cli_path)
    if resolved is None:
        return False, f"gemini CLI not found on PATH (tried '{cli_path}')"

    try:
        proc = await _spawn_gemini(
            resolved, "--yolo", "--skip-trust",
            "-p", "Reply with exactly: OK",
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=90,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            return False, "auth check timed out (>90s) -- try `gemini` interactively first"

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace")[:200]
            return False, f"exit {proc.returncode}: {stderr_text}"

        text = stdout.decode(errors="replace").strip()
        if len(text) == 0:
            return False, "empty response -- auth may not be configured"
        return True, f"auth ok ({text[:60]})"
    except FileNotFoundError:
        return False, f"binary not executable: {resolved}"
    except Exception as exc:
        return False, f"auth check failed: {type(exc).__name__}: {exc}"


# ===============================================================================
# FACTORY
# ===============================================================================

async def create_consolidation_engine(
    neo4j_client: Optional[Neo4jClient] = None,
    embed_utils: Optional[EmbeddingUtils] = None,
    retrieval_engine: Optional[MemoryRetrievalEngine] = None,
    gemini_cli_path: str = "gemini",
    timeout_s: int = 180,
) -> AgenticConsolidationEngine:
    """Convenience factory. Connects Neo4j if not provided."""
    if neo4j_client is None:
        cfg = get_config()
        neo4j_client = create_neo4j_client(
            uri=cfg.neo4j_uri,
            username=cfg.neo4j_username,
            password=cfg.neo4j_password,
            database=cfg.database,
        )
        await neo4j_client.connect()

    embed = embed_utils or EmbeddingUtils()
    agent = GeminiAgent(cli_path=gemini_cli_path, timeout_s=timeout_s)
    return AgenticConsolidationEngine(
        neo4j=neo4j_client,
        embed=embed,
        retrieval=retrieval_engine,
        agent=agent,
    )
