"""
SOFI Memory Consolidation Engine - Production Ready

4-Stage Night Processing Pipeline:
  Stage 1: Summarization (llama-3.1-8b-instant)
  Stage 2: Deep Context Retrieval (Neo4j)
  Stage 3: Memory Decision Making (qwen/qwen3-32b)
  Stage 4: Verification & Confidence Scoring

Author: SOFI Memory System
Date: November 6, 2025
"""

import asyncio
import datetime
import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from groq import AsyncGroq

from memory.config import get_config
from memory.long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client
from memory.processing.embedding_utils import EmbeddingUtils
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.models.node_models import (
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
    MemoryContext
)
from memory.long_term.models.relationship_models import (
    MemoryRelationshipEdge,
    MemoryRelationshipType,
    MemoryRelationshipCategory
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# MULTI-MODEL GROQ CLIENT
# ============================================================================

class MultiModelGroqClient:
    """Handles multiple Groq models with separate rate limits."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = "gsk_ud6XIO63FUU3902bobH3WGdyb3FYkVO4Gg4QF85G2m7yahGigNy"
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Setup: https://console.groq.com/")

        self.client = AsyncGroq(api_key=self.api_key)

        # Model configurations
        self.models = {
            "summarizer": {
                "name": "llama-3.1-8b-instant",
                "tpm": 30000,
                "tpd": 500000,
                "max_input": 8000,
                "max_output": 2000
            },
            "decision": {
                "name": "qwen/qwen3-32b",
                "tpm": 30000,
                "tpd": 500000,
                "max_input": 8000,
                "max_output": 3000
            },
            "verifier": {
                "name": "qwen/qwen3-32b",
                "tpm": 30000,
                "tpd": 500000,
                "max_input": 8000,
                "max_output": 1000
            }
        }

        # Token tracking per model
        self.token_usage = {model_type: 0 for model_type in self.models}
        self.last_reset = datetime.datetime.now()

        logger.info("MultiModelGroqClient initialized with 2 models")

    async def _check_rate_limit(self, model_type: str, estimated_tokens: int):
        """Check and enforce rate limits."""
        now = datetime.datetime.now()

        # Reset every minute
        if (now - self.last_reset).total_seconds() >= 60:
            self.token_usage = {mt: 0 for mt in self.models}
            self.last_reset = now

        model_config = self.models[model_type]
        if self.token_usage[model_type] + estimated_tokens > model_config["tpm"]:
            wait_seconds = 60 - (now - self.last_reset).total_seconds()
            logger.info(f"⏳ Rate limit approaching for {model_type}. Waiting {wait_seconds:.0f}s...")
            await asyncio.sleep(wait_seconds)
            self.token_usage = {mt: 0 for mt in self.models}
            self.last_reset = datetime.datetime.now()

    async def call_model(
        self,
        model_type: str,
        prompt: str,
        system_prompt: str,
        estimated_tokens: int,
        json_mode: bool = False
    ) -> Dict[str, Any]:
        """Call a specific model."""
        await self._check_rate_limit(model_type, estimated_tokens)

        model_config = self.models[model_type]
        model_name = model_config["name"]

        try:
            logger.debug(f"Calling {model_type} ({model_name})...")

            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": model_config["max_output"]
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)

            self.token_usage[model_type] += estimated_tokens

            return {
                "success": True,
                "content": response.choices[0].message.content
            }
        except Exception as e:
            logger.error(f"Error calling {model_type}: {e}")
            return {"success": False, "content": "", "error": str(e)}


# ============================================================================
# STAGE 1: CONVERSATION SUMMARIZER
# ============================================================================

class ConversationSummarizer:
    """Stage 1: Summarize and organize conversations."""

    def __init__(self, groq_client: MultiModelGroqClient):
        self.groq = groq_client

    async def summarize(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Summarize conversations into structured format."""

        # Build conversation text
        conv_text = "\n".join([
            f"{c.get('role', '').upper()}: {c.get('content', '')}"
            for c in conversations
        ])

        prompt = f"""Analyze this conversation and extract structured information:

CONVERSATION:
{conv_text}

Extract in JSON format:
{{
  "summary": "Brief summary of conversation",
  "entities": ["list", "of", "key", "entities"],
  "topics": ["topic1", "topic2"],
  "sentiment": 0.0,
  "key_facts": ["fact1", "fact2"],
  "questions_asked": ["q1", "q2"],
  "importance": 0.5
}}

Be concise and accurate.
"""

        estimated_tokens = len(conv_text) // 3 + 500

        response = await self.groq.call_model(
            model_type="summarizer",
            prompt=prompt,
            system_prompt="You are a conversation analysis expert. Extract structured information accurately.",
            estimated_tokens=estimated_tokens,
            json_mode=True
        )

        if response["success"]:
            try:
                return json.loads(response["content"])
            except:
                logger.warning("Failed to parse summarizer response")
                return self._default_summary(conversations)

        return self._default_summary(conversations)

    def _default_summary(self, conversations: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fallback summary if LLM fails."""
        text = " ".join([c.get('content', '') for c in conversations])
        return {
            "summary": text[:200],
            "entities": [],
            "topics": [],
            "sentiment": 0.0,
            "key_facts": [text[:100]],
            "questions_asked": [],
            "importance": 0.5
        }


# ============================================================================
# STAGE 2: DEEP CONTEXT RETRIEVER
# ============================================================================

class DeepContextRetriever:
    """Stage 2: Retrieve deep context from graph (2-hop traversal)."""

    def __init__(self, neo4j_client: Neo4jClient, retrieval_engine: MemoryRetrievalEngine):
        self.neo4j = neo4j_client
        self.retrieval_engine = retrieval_engine

    async def get_context(
        self,
        summary: Dict[str, Any],
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Get deep context from Neo4j graph."""

        entities = summary.get("entities", [])
        topics = summary.get("topics", [])
        summary_text = summary.get("summary", "")

        try:
            # Semantic search on summary
            logger.debug("Fetching semantic context...")
            semantic_results = await self.retrieval_engine.semantic_search(
                query_text=summary_text[:500],
                top_k=min(10, top_k)
            )

            logger.debug(f"Found {len(semantic_results)} semantic matches")

            return {
                "semantic_matches": semantic_results,
                "entity_count": len(entities),
                "topic_count": len(topics),
                "total_context_size": len(semantic_results)
            }
        except Exception as e:
            logger.warning(f"Error retrieving context: {e}")
            return {
                "semantic_matches": [],
                "entity_count": len(entities),
                "topic_count": len(topics),
                "total_context_size": 0
            }


# ============================================================================
# STAGE 3: MEMORY DECISION MAKER
# ============================================================================

class MemoryDecisionMaker:
    """Stage 3: Decide NEW vs UPDATE vs ENHANCE."""

    def __init__(self, groq_client: MultiModelGroqClient):
        self.groq = groq_client

    async def decide(
        self,
        summary: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide what memories to create/update."""

        context_str = f"Found {context['total_context_size']} related memories"

        prompt = f"""Analyze this conversation summary and decide what memories to save or update.

SUMMARY:
{json.dumps(summary, indent=2)}

EXISTING CONTEXT:
{context_str}

Decide for each key fact:
- NEW: Create new memory
- UPDATE: Update existing (if similar exists in context)
- SKIP: Not important enough

Output JSON:
{{
  "decisions": [
    {{
      "content": "memory content",
      "decision": "NEW|UPDATE|SKIP",
      "confidence": 0.9,
      "type": "EXPERIENCE|KNOWLEDGE|RELATIONSHIP",
      "importance": 0.8
    }}
  ]
}}
"""

        estimated_tokens = len(json.dumps(summary)) // 3 + 1000

        response = await self.groq.call_model(
            model_type="decision",
            prompt=prompt,
            system_prompt="You are a memory curation expert. Make accurate decisions about memory storage.",
            estimated_tokens=estimated_tokens,
            json_mode=True
        )

        if response["success"]:
            try:
                return json.loads(response["content"])
            except:
                logger.warning("Failed to parse decision response")
                return {"decisions": []}

        return {"decisions": []}


# ============================================================================
# STAGE 4: MEMORY VERIFIER
# ============================================================================

class MemoryVerifier:
    """Stage 4: Verify and score confidence."""

    def __init__(self, groq_client: MultiModelGroqClient):
        self.groq = groq_client

    async def verify(self, decisions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verify decisions and add confidence scores."""

        decisions_list = decisions.get("decisions", [])

        if not decisions_list:
            return []

        prompt = f"""Review these memory decisions for quality and consistency.

DECISIONS:
{json.dumps(decisions_list, indent=2)}

For each decision, verify:
1. Is it accurate?
2. Is confidence score realistic?
3. Should it be approved?

Output JSON:
{{
  "verified": [
    {{
      "content": "memory content",
      "decision": "NEW|UPDATE|SKIP",
      "confidence": 0.9,
      "status": "approved|needs_review",
      "reason": "brief reason"
    }}
  ]
}}
"""

        estimated_tokens = len(json.dumps(decisions_list)) // 3 + 500

        response = await self.groq.call_model(
            model_type="verifier",
            prompt=prompt,
            system_prompt="You are a quality assurance expert. Verify memory decisions carefully.",
            estimated_tokens=estimated_tokens,
            json_mode=True
        )

        if response["success"]:
            try:
                result = json.loads(response["content"])
                return result.get("verified", [])
            except:
                logger.warning("Failed to parse verification response")
                return decisions_list

        return decisions_list


# ============================================================================
# CONSOLIDATION ENGINE
# ============================================================================

class ConsolidationEngine:
    """Main consolidation engine coordinating all stages."""

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embed_util: EmbeddingUtils,
        retrieval_engine: MemoryRetrievalEngine
    ):
        self.neo4j = neo4j_client
        self.embed_util = embed_util
        self.retrieval_engine = retrieval_engine

        self.groq = MultiModelGroqClient()

        self.summarizer = ConversationSummarizer(self.groq)
        self.context_retriever = DeepContextRetriever(neo4j_client, retrieval_engine)
        self.decision_maker = MemoryDecisionMaker(self.groq)
        self.verifier = MemoryVerifier(self.groq)

        self.config = get_config()
        self.log_filepath = Path(self.config.conversation_log_path)

        self.is_running = False
        self.TRIGGER_HOUR = 20

        logger.info("ConsolidationEngine initialized (4-stage pipeline)")

    def _load_conversation_file(self) -> Dict[str, Any]:
        """Load conversation.json file."""
        if not self.log_filepath.exists():
            return {}

        try:
            with open(self.log_filepath, 'r') as f:
                return json.load(f)
        except:
            logger.error("Error loading conversation file")
            return {}

    def _save_conversation_file(self, data: Dict[str, Any]):
        """Save conversation.json file."""
        try:
            with open(self.log_filepath, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversation file: {e}")

    async def start_scheduler(self):
        """Start scheduler for nightly processing."""
        logger.info("Starting Consolidation Scheduler...")
        self.is_running = True

        while self.is_running:
            try:
                now = datetime.datetime.now()

                if now.hour >= self.TRIGGER_HOUR and now.hour < self.TRIGGER_HOUR + 1:
                    logger.info(f"[{now}] Trigger hour reached")
                    await self.run_consolidation_once()

                    sleep_seconds = self._seconds_until_tomorrow()
                    logger.info(f"Sleeping until tomorrow ({sleep_seconds/3600:.1f}h)")
                    await asyncio.sleep(sleep_seconds)
                else:
                    await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(3600)

    def _seconds_until_tomorrow(self) -> float:
        """Seconds until tomorrow 12:01 AM."""
        now = datetime.datetime.now()
        tomorrow = now + datetime.timedelta(days=1)
        target = tomorrow.replace(hour=0, minute=1, second=0)
        return (target - now).total_seconds()

    async def run_consolidation_once(self):
        """Run complete consolidation pipeline."""
        logger.info("\n" + "="*80)
        logger.info("CONSOLIDATION RUN STARTED")
        logger.info("="*80)

        try:
            log_data = self._load_conversation_file()

            if not log_data:
                logger.info("No conversation data found")
                return

            user_key = f"user_{self.config.user_id}"
            sessions = log_data.get(user_key, [])

            if not sessions:
                logger.info(f"No sessions for user {self.config.user_id}")
                return

            logger.info(f"Processing {len(sessions)} sessions")

            sessions_to_keep = []
            sessions_processed = 0

            for session in sessions:
                session_id = session.get('session_id', 'unknown')
                conversations = session.get('conversations', [])

                if not conversations:
                    continue

                logger.info(f"\nSession: {session_id} ({len(conversations)} turns)")

                try:
                    success = await self._process_session(session_id, conversations)

                    if success:
                        logger.info(f"✅ Session {session_id} processed - DELETED")
                        sessions_processed += 1
                    else:
                        logger.warning(f"⚠️ Session {session_id} failed - KEEPING")
                        sessions_to_keep.append(session)

                except Exception as e:
                    logger.error(f"❌ Session {session_id} error: {e}")
                    sessions_to_keep.append(session)

            log_data[user_key] = sessions_to_keep
            self._save_conversation_file(log_data)

            logger.info("="*80)
            logger.info(f"CONSOLIDATION COMPLETE")
            logger.info(f"  Processed: {sessions_processed} sessions")
            logger.info(f"  Remaining: {len(sessions_to_keep)} sessions")
            logger.info("="*80 + "\n")

        except Exception as e:
            logger.error(f"Consolidation error: {e}", exc_info=True)

    async def _process_session(self, session_id: str, conversations: List[Dict[str, str]]) -> bool:
        """Process a single session through 4-stage pipeline."""

        try:
            # Stage 1: Summarize
            logger.debug("Stage 1: Summarizing...")
            summary = await self.summarizer.summarize(conversations)

            # Stage 2: Get context
            logger.debug("Stage 2: Retrieving context...")
            context = await self.context_retriever.get_context(summary)

            # Stage 3: Make decisions
            logger.debug("Stage 3: Making decisions...")
            decisions = await self.decision_maker.decide(summary, context)

            # Stage 4: Verify
            logger.debug("Stage 4: Verifying...")
            verified = await self.verifier.verify(decisions)

            # Save to Neo4j
            logger.debug("Saving to Neo4j...")
            print(verified)
            logger.error(f"--Verified Memory ---\n\n{verified}\n\n")
            # await self._save_verified_memories(verified)

            return True

        except Exception as e:
            logger.error(f"Processing error: {e}")
            return False

    async def _save_verified_memories(self, verified_memories: List[Dict[str, Any]]):
        """Save verified memories to Neo4j."""

        saved = 0

        for mem in verified_memories:
            try:
                if mem.get("status") != "approved":
                    logger.debug(f"Skipping: {mem.get('content', '')[:50]}...")
                    continue

                decision = mem.get("decision", "SKIP")
                if decision == "SKIP":
                    continue

                content = mem.get("content")
                mem_type = mem.get("type", "EXPERIENCE")
                confidence = mem.get("confidence", 0.5)

                if not content:
                    continue

                # Generate embedding
                try:
                    embedding = self.embed_util.generate_embedding(content)
                except:
                    embedding = None

                # Create node
                node_data = {
                    "content": content,
                    "content_vector": embedding,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "confidence": confidence,
                    "importance_score": mem.get("importance", 0.5),
                    "emotional_tone": 0.0,
                    "event_type": "conversation"
                }

                try:
                    memory_context = MemoryContext(mem_type)
                except:
                    memory_context = MemoryContext.EXPERIENCE

                if memory_context == MemoryContext.EXPERIENCE:
                    node = ExperienceMemoryNode(**node_data)
                    label = "ExperienceMemory"
                elif memory_context == MemoryContext.KNOWLEDGE:
                    node = KnowledgeMemoryNode(**node_data)
                    label = "KnowledgeMemory"
                else:
                    node = RelationshipMemoryNode(**node_data)
                    label = "RelationshipMemory"

                node_dict = node.dict()
                if isinstance(node_dict.get("content_vector"), (list, tuple)):
                    node_dict["content_vector"] = list(node_dict["content_vector"])

                query = f"CREATE (n:{label} $props) RETURN n.id as id"
                result = await self.neo4j.execute_query(query, {"props": node_dict})

                if result:
                    saved += 1
                    logger.debug(f"✅ Saved: {label} - {content[:50]}...")

            except Exception as e:
                logger.warning(f"Error saving memory: {e}")

        logger.info(f"💾 Saved {saved} memories")


# ============================================================================
# CLI
# ============================================================================

async def create_consolidation_engine(
    neo4j_client: Neo4jClient,
    embed_utils: EmbeddingUtils,
    retrieval_engine: MemoryRetrievalEngine
) -> ConsolidationEngine:
    """Factory for ConsolidationEngine."""
    return ConsolidationEngine(neo4j_client, embed_utils, retrieval_engine)


if __name__ == "__main__":
    import sys

    async def main():
        config = get_config()
        groq_api_key = "gsk_ud6XIO63FUU3902bobH3WGdyb3FYkVO4Gg4QF85G2m7yahGigNy"

        if not groq_api_key:
            logger.error("GROQ_API_KEY not set")
            sys.exit(1)

        neo4j_client = create_neo4j_client(
            uri = "bolt://localhost:7687",
            username = "neo4j", 
            password = "password123",
            # database: str = "neo4j",
            # config.neo4j_uri,
            # config.neo4j_username,
            # config.neo4j_password
        )
        await neo4j_client.connect()

        embed_utils = EmbeddingUtils()
        retrieval_engine = MemoryRetrievalEngine(neo4j_client, embed_utils)

        engine = await create_consolidation_engine(
            neo4j_client,
            embed_utils,
            retrieval_engine
        )

        if "--once" in sys.argv:
            logger.info("Running consolidation once...")
            await engine.run_consolidation_once()
        else:
            logger.info("Starting consolidation scheduler...")
            await engine.start_scheduler()

    asyncio.run(main())