"""
SOFI Memory Retrieval Engine - Complete Implementation

This module provides a comprehensive retrieval interface for SOFI's graph memory system.
It offers multiple retrieval patterns to fetch memories in any manner needed.

Features:
- Basic retrieval (single/multiple topics)
- Graph traversal (connected nodes, clusters)
- Semantic search (vector similarity)
- Grouped retrieval (organized by topics)
- Time-based retrieval (recent, date ranges)
- Advanced retrieval (hubs, emotions, weighted relevance)
- Relationship strength filtering
- Connection reinforcement
"""

import math
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from memory.long_term.infrastructure.neo4j_client import Neo4jClient
from memory.processing.embedding_utils import EmbeddingUtils


logger = logging.getLogger(__name__)

# Filter appended to every retrieval WHERE clause.
# Nodes marked superseded=true during consolidation (CONTRADICT operation) must
# never surface in retrieval — the newer node supersedes them.
# coalesce handles nodes that predate the field (treat absence as not superseded).
_NOT_SUPERSEDED_m    = "AND NOT coalesce(m.superseded, false)"
_NOT_SUPERSEDED_node = "AND NOT coalesce(node.superseded, false)"
_NOT_SUPERSEDED_conn = "AND NOT coalesce(connected.superseded, false)"


# ============================================================================
# ENUMS
# ============================================================================

class MemoryType(Enum):
    """Memory node types in the graph"""
    EXPERIENCE = "ExperienceMemory"
    KNOWLEDGE = "KnowledgeMemory"
    RELATIONSHIP = "RelationshipMemory"
    ALL = "All"


class EmotionType(str, Enum):
    """Predefined emotion types for filtering"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    HAPPY = "happy"
    EXCITED = "excited"
    PROUD = "proud"
    SAD = "sad"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    STRESSED = "stressed"
    NEUTRAL = "neutral"


# ============================================================================
# SPREADING ACTIVATION CONSTANTS
# ============================================================================

# Baseline conductivity for each typed relationship.
# Higher = energy transfers more readily across that edge type.
BASE_WEIGHTS: Dict[str, float] = {
    "CAUSED":                   0.90,
    "RESULTED_IN":              0.90,
    "TRIGGERED":                0.85,
    "EXPERIENCE_CHAIN":         0.80,
    "WITHIN_CONTEXT":           0.80,
    "KNOWLEDGE_HIERARCHY":      0.70,
    "EXPERIENCE_TO_KNOWLEDGE":  0.65,
    "KNOWLEDGE_TO_EXPERIENCE":  0.65,
    "CROSS_CONTEXT":            0.60,
    "TEMPORAL":                 0.50,
    "HAPPENED_BEFORE":          0.50,
    "HAPPENED_AFTER":           0.50,
    "CONCURRENT":               0.45,
    "SIMILAR_TO":               0.15,  # deliberately low — prevents topic drift
    "ASSOCIATED_WITH":          0.10,
}

# Edge types traversed per intent — only the conductivity paths relevant to that intent.
# Using Intent.value strings (e.g. "entity") to avoid a circular import with memory_router.
INTENT_EDGE_MAP: Dict[str, List[str]] = {
    "entity": [
        "EXPERIENCE_TO_RELATIONSHIP", "RELATIONSHIP_TO_EXPERIENCE",
        "RELATIONSHIP_NETWORK", "EXPERIENCE_CHAIN",
    ],
    "factual": [
        "EXPERIENCE_TO_KNOWLEDGE", "KNOWLEDGE_TO_EXPERIENCE",
        "KNOWLEDGE_HIERARCHY", "SIMILAR_TO",
    ],
    "emotional": [
        "INFLUENCED", "TRIGGERED", "CAUSED",
        "EXPERIENCE_CHAIN",
    ],
    "temporal": [
        "HAPPENED_BEFORE", "HAPPENED_AFTER", "CONCURRENT", "DURING",
        "EXPERIENCE_CHAIN",
    ],
}

# Per-intent boosts applied on top of BASE_WEIGHTS for specific edge types.
INTENT_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "entity":    {"RELATIONSHIP_NETWORK": 1.5,  "EXPERIENCE_TO_RELATIONSHIP": 1.5},
    "factual":   {"EXPERIENCE_TO_KNOWLEDGE": 1.4, "KNOWLEDGE_HIERARCHY": 1.4},
    "emotional": {"INFLUENCED": 1.5, "TRIGGERED": 1.4, "CAUSED": 1.3},
    "temporal":  {"HAPPENED_BEFORE": 1.8, "HAPPENED_AFTER": 1.8, "CONCURRENT": 1.6},
}


# ============================================================================
# MAIN RETRIEVAL ENGINE CLASS
# ============================================================================

class MemoryRetrievalEngine:
    """
    Comprehensive retrieval engine for SOFI's graph memory system.
    
    This class provides 20+ retrieval methods organized into categories:
    - Basic retrieval (topics, keywords)
    - Graph traversal (connected nodes, clusters)
    - Semantic search (vector similarity)
    - Grouped retrieval (organized by topics)
    - Time-based retrieval (recent, date ranges)
    - Advanced retrieval (hubs, emotions, strength-based)
    - Utility methods (statistics, reinforcement)
    """
    
    def __init__(
        self,
        neo4j_client: Neo4jClient,
        embedding_utils: Optional[EmbeddingUtils] = None
    ):
        """
        Initialize the retrieval engine.
        
        Args:
            neo4j_client: Connected Neo4j client instance
            embedding_utils: EmbeddingUtils instance for semantic search (optional)
        """
        self.client = neo4j_client
        self.embed_util = embedding_utils or EmbeddingUtils()
        logger.info("MemoryRetrievalEngine initialized successfully")
    
    # ========================================================================
    # BASIC RETRIEVAL METHODS
    # ========================================================================
    
    async def get_memories_by_topic(
        self,
        topic: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all memories about a single topic/person.
        
        Args:
            topic: The topic or person name to search for
            memory_types: Filter by memory types (default: all types)
            limit: Maximum number of results
            
        Returns:
            List of memory dictionaries with id, content, type, timestamp
            
        Example:
            ```python
            memories = await engine.get_memories_by_topic("father")
            memories = await engine.get_memories_by_topic(
                "project",
                memory_types=[MemoryType.EXPERIENCE]
            )
            ```
        """
        # Build type filter if specified
        type_filter = self._build_type_filter(memory_types)
        
        query = f"""
        MATCH (m)
        WHERE (toLower(coalesce(m.content, "")) CONTAINS toLower($topic)
           OR toLower(coalesce(m.person_name, "")) = toLower($topic)
           OR toLower(coalesce(m.concept, "")) CONTAINS toLower($topic)
           OR any(p IN coalesce(m.participants, []) WHERE toLower(p) = toLower($topic)))
        {_NOT_SUPERSEDED_m}
        {type_filter}
        RETURN
            m.id as id,
            labels(m) as type,
            m.content as content,
            m.timestamp as timestamp,
            m.emotional_tone as emotional_tone,
            m.participants as participants,
            m.concept as concept,
            m.person_name as person_name
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(
            query,
            parameters={'topic': topic, 'limit': limit}
        )
        
        logger.info(f"Retrieved {len(results)} memories for topic '{topic}'")
        return results
    
    async def get_memories_by_multiple_topics(
        self,
        topics: List[str],
        match_all: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories about multiple topics/people.
        
        Args:
            topics: List of topics/people to search for
            match_all: If True, returns memories matching ALL topics (AND logic)
                      If False, returns memories matching ANY topic (OR logic)
            limit: Maximum number of results
            
        Returns:
            List of memory dictionaries
            
        Example:
            ```python
            # Get memories about father OR girlfriend OR project
            memories = await engine.get_memories_by_multiple_topics(
                ["father", "girlfriend", "project"]
            )
            
            # Get memories mentioning BOTH father AND project
            memories = await engine.get_memories_by_multiple_topics(
                ["father", "project"],
                match_all=True
            )
            ```
        """
        # Build WHERE conditions based on AND/OR logic
        where_conditions, parameters = self._build_multi_topic_conditions(
            topics, match_all
        )
        parameters['limit'] = limit
        
        query = f"""
        MATCH (m)
        WHERE {where_conditions}
        RETURN 
            m.id as id,
            labels(m) as type,
            m.content as content,
            m.timestamp as timestamp,
            m.emotional_tone as emotional_tone,
            m.participants as participants
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} memories for {len(topics)} topics")
        return results
    
    async def bm25_search(
        self,
        query_terms: List[str],
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        BM25 keyword search via Neo4j Lucene full-text index (~10-20ms).
        No embedding generation — uses a pre-built index on content, participants,
        concept, person_name, and tags fields.

        Args:
            query_terms: Entity names or keywords to search for.
            limit:        Maximum number of results.

        Returns:
            List of memory dicts with a bm25_score field added.
        """
        if not query_terms:
            return []

        query_string = " OR ".join(self._escape_lucene(t) for t in query_terms)

        query = f"""
        CALL db.index.fulltext.queryNodes("memory_fts", $query_string)
        YIELD node, score
        WHERE {_NOT_SUPERSEDED_node.lstrip("AND ")}
        RETURN
            node.id            AS id,
            node.content       AS content,
            labels(node)       AS type,
            node.timestamp     AS timestamp,
            node.emotional_tone AS emotional_tone,
            node.participants  AS participants,
            node.concept       AS concept,
            node.person_name   AS person_name,
            score              AS bm25_score
        ORDER BY score DESC
        LIMIT $limit
        """

        results = await self.client.execute_query(
            query,
            parameters={"query_string": query_string, "limit": limit},
        )
        logger.info(f"BM25 '{query_string}' → {len(results)} results")
        return results

    # ========================================================================
    # GRAPH TRAVERSAL METHODS
    # ========================================================================

    async def get_memories_with_connected_nodes(
        self,
        topic: str,
        max_hops: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories about a topic AND all connected memories.
        
        This follows graph relationships to find related memories,
        mimicking human associative memory recall.
        
        Args:
            topic: The topic/person to search for
            max_hops: Maximum relationship hops to traverse (1-3 recommended)
            limit: Maximum number of root memories to return
            
        Returns:
            List of dictionaries with root_id, root_content, connected_memories
            
        Example:
            ```python
            results = await engine.get_memories_with_connected_nodes(
                "father",
                max_hops=2
            )
            
            for item in results:
                print(f"Main: {item['root_content']}")
                print(f"Connected: {len(item['connected_memories'])}")
            ```
        """
        query = f"""
        MATCH (root)
        WHERE toLower(coalesce(root.content, "")) CONTAINS toLower($topic)
           OR toLower(coalesce(root.person_name, "")) = toLower($topic)
           OR toLower(coalesce(root.concept, "")) CONTAINS toLower($topic)
           OR any(p IN coalesce(root.participants, []) WHERE toLower(p) = toLower($topic))
        
        OPTIONAL MATCH path = (root)-[r:MEMORY_RELATIONSHIP*1..{max_hops}]-(connected)
        
        RETURN 
            root.id as root_id,
            root.content as root_content,
            labels(root) as root_type,
            root.timestamp as root_timestamp,
            COLLECT(DISTINCT {{
                id: connected.id,
                content: connected.content,
                type: labels(connected),
                timestamp: connected.timestamp,
                distance: length(path)
            }}) as connected_memories
        ORDER BY root.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(
            query,
            parameters={'topic': topic, 'max_hops': max_hops, 'limit': limit}
        )
        
        logger.info(
            f"Retrieved {len(results)} memories with connected nodes "
            f"for topic '{topic}'"
        )
        return results
    
    async def get_connected_nodes_only(
        self,
        memory_id: str,
        max_hops: int = 2,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get ONLY the connected nodes for a specific memory.
        
        Useful for expanding context around a known memory.
        
        Args:
            memory_id: ID of the memory to find connections for
            max_hops: Maximum relationship hops to traverse
            limit: Maximum number of connected nodes to return
            
        Returns:
            List of connected memory dictionaries
            
        Example:
            ```python
            connected = await engine.get_connected_nodes_only(
                "exp_12345",
                max_hops=2
            )
            ```
        """
        query = f"""
        MATCH (root {{id: $memory_id}})
        MATCH path = (root)-[r:MEMORY_RELATIONSHIP*1..{max_hops}]-(connected)
        RETURN DISTINCT
            connected.id as id,
            connected.content as content,
            labels(connected) as type,
            connected.timestamp as timestamp,
            length(path) as distance,
            [rel in relationships(path) | type(rel)] as relationship_path
        ORDER BY distance, connected.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(
            query,
            parameters={'memory_id': memory_id, 'max_hops': max_hops, 'limit': limit}
        )
        
        logger.info(
            f"Retrieved {len(results)} connected nodes for memory '{memory_id}'"
        )
        return results
    
    async def get_memory_cluster(
        self,
        starting_memory_id: str,
        max_hops: int = 3,
        min_connections: int = 2
    ) -> Dict[str, Any]:
        """
        Get a cluster of highly interconnected memories around a starting memory.
        
        This finds "memory communities" - groups of memories that reference each other.
        
        Args:
            starting_memory_id: The memory to start clustering from
            max_hops: Maximum distance to explore
            min_connections: Minimum connections a node must have to be included
            
        Returns:
            Dictionary with starting_memory, cluster_size, and members list
            
        Example:
            ```python
            cluster = await engine.get_memory_cluster(
                "exp_12345",
                max_hops=3,
                min_connections=2
            )
            print(f"Cluster size: {cluster['cluster_size']}")
            ```
        """
        query = f"""
        MATCH (start {{id: $memory_id}})
        MATCH path = (start)-[r:MEMORY_RELATIONSHIP*1..{max_hops}]-(member)
        WITH member, count(path) as connection_count
        WHERE connection_count >= $min_connections
        RETURN 
            member.id as id,
            member.content as content,
            labels(member) as type,
            member.timestamp as timestamp,
            connection_count
        ORDER BY connection_count DESC
        """
        
        results = await self.client.execute_query(
            query,
            parameters={
                'memory_id': starting_memory_id,
                'max_hops': max_hops,
                'min_connections': min_connections
            }
        )
        
        cluster = {
            'starting_memory': starting_memory_id,
            'cluster_size': len(results),
            'members': results
        }
        
        logger.info(
            f"Found cluster of {len(results)} memories "
            f"around '{starting_memory_id}'"
        )
        return cluster

    async def _spreading_activation(
        self,
        entities: List[str],
        intent: str,
        budget: int,
    ) -> List[Dict[str, Any]]:
        """
        Typed graph traversal with activation decay scoring.

        Traversal steps:
          1. Resolve entity strings to seed node IDs (no embedding).
          2. Expand 2 hops using only the edge types for the given intent.
          3. Score each reachable node: edge_weight × intent_multiplier
             × temporal_decay × path_boost (soft lateral inhibition).
          4. Return top-budget nodes sorted by activation score.

        Args:
            entities: Entity strings extracted from the message.
            intent:   Intent.value string — selects which edge types to traverse.
            budget:   Maximum nodes to return after scoring.
        """
        seed_ids = await self._resolve_seed_nodes(entities)
        if not seed_ids:
            return []

        edge_types = INTENT_EDGE_MAP.get(intent, list(BASE_WEIGHTS.keys()))
        rel_filter = "|".join(edge_types)
        raw_limit  = min(budget * 3, 150)

        # r is a list of relationships in a variable-length path.
        # r[0] is the first (closest-to-seed) relationship — its type and strength
        # are most representative of why this node was reached.
        query = f"""
        MATCH (seed) WHERE seed.id IN $seed_ids
        MATCH (seed)-[r:{rel_filter}*1..2]-(connected)
        WHERE connected.id <> seed.id
          AND NOT coalesce(connected.superseded, false)
        WITH connected,
             collect(DISTINCT type(r[0]))           AS rel_types,
             collect(coalesce(r[0].strength, 0.5))  AS strengths,
             count(DISTINCT seed)                   AS path_count,
             min(size(r))                           AS distance
        RETURN
            connected.id             AS id,
            connected.content        AS content,
            labels(connected)        AS type,
            connected.timestamp      AS timestamp,
            connected.emotional_tone AS emotional_tone,
            connected.access_count   AS access_count,
            connected.last_accessed  AS last_accessed,
            rel_types[0]             AS rel_type,
            strengths[0]             AS edge_strength,
            distance,
            path_count
        ORDER BY path_count DESC, distance ASC
        LIMIT $limit
        """

        rows = await self.client.execute_query(
            query,
            parameters={"seed_ids": seed_ids, "limit": raw_limit},
        )

        if not rows:
            return []

        intent_mults = INTENT_MULTIPLIERS.get(intent, {})
        scored: List[tuple] = []

        for node in rows:
            rel_type  = node.get("rel_type") or ""
            edge_w    = BASE_WEIGHTS.get(rel_type, 0.5)
            mult      = intent_mults.get(rel_type, 1.0)

            days_old = 0
            raw_ts   = node.get("timestamp")
            if raw_ts:
                try:
                    dt = datetime.fromisoformat(str(raw_ts).replace("Z", ""))
                    days_old = max(0, (datetime.now() - dt.replace(tzinfo=None)).days)
                except Exception:
                    pass
            decay = math.exp(-0.01 * days_old)

            # Soft lateral inhibition: nodes reachable from multiple seeds are
            # more likely to be genuinely relevant — boost without hard-filtering.
            path_count = int(node.get("path_count") or 1)
            path_boost = 1.0 + 0.25 * min(path_count - 1, 3)

            activation = edge_w * mult * decay * path_boost
            scored.append((activation, {**node, "activation_score": round(activation, 4)}))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for _, m in scored[:budget]]

        logger.info(
            f"Spreading activation ({intent}): {len(entities)} entities → "
            f"{len(rows)} candidates → {len(top)} returned"
        )
        return top

    # ========================================================================
    # SEMANTIC SEARCH METHODS
    # ========================================================================
    
    async def semantic_search(
        self,
        query_text: str,
        top_k: int = 10,
        include_connected: bool = False,
        max_hops: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search memories by semantic meaning (not just keywords).

        Uses vector embeddings to find memories with similar meaning,
        even if they use different words.

        Args:
            query_text: Natural language query
            top_k: Number of most similar memories to return
            include_connected: Whether to also return connected memories
            max_hops: If include_connected=True, how many hops to traverse

        Returns:
            List of memory dictionaries with similarity scores

        Example:
            ```python
            # Will find "father", "dad", "parent", etc.
            memories = await engine.semantic_search(
                "family advice",
                top_k=5
            )
            ```
        """
        import asyncio as _asyncio

        # ── CRITICAL: run the synchronous, CPU-bound embedding in a thread ────
        # Without this, generate_embedding() blocks the event loop, preventing
        # every other coroutine inside asyncio.gather() from making progress.
        # run_in_executor() offloads to the default ThreadPoolExecutor so the
        # event loop stays free for concurrent Neo4j queries.
        loop = _asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(
            None,                              # default executor
            self.embed_util.generate_embedding,
            query_text,
        )

        if include_connected:
            query = f"""
            CALL db.index.vector.queryNodes(
                'memory_vector_index',
                $top_k,
                $query_vector
            )
            YIELD node, score

            OPTIONAL MATCH (node)-[r:MEMORY_RELATIONSHIP*1..{max_hops}]-(connected)

            RETURN
                node.id as id,
                node.content as content,
                labels(node) as type,
                node.timestamp as timestamp,
                score,
                COLLECT(DISTINCT {{
                    id: connected.id,
                    content: connected.content,
                    type: labels(connected)
                }}) as connected_memories
            ORDER BY score DESC
            """
            parameters = {
                'top_k': top_k,
                'query_vector': query_vector,
                'max_hops': max_hops
            }
        else:
            query = """
            CALL db.index.vector.queryNodes(
                'memory_vector_index',
                $top_k,
                $query_vector
            )
            YIELD node, score
            RETURN
                node.id as id,
                node.content as content,
                labels(node) as type,
                node.timestamp as timestamp,
                score
            ORDER BY score DESC
            """
            parameters = {'top_k': top_k, 'query_vector': query_vector}

        results = await self.client.execute_query(query, parameters)
        logger.info(
            f"Semantic search for '{query_text}' returned {len(results)} results"
        )
        return results


    
    # ========================================================================
    # GROUPED RETRIEVAL METHODS
    # ========================================================================
    
    async def get_memories_grouped_by_topics(
        self,
        topics: List[str],
        include_connected: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve memories organized by topic.
        
        Returns a dictionary where each topic has its own list of memories
        and optionally connected nodes.
        
        Args:
            topics: List of topics to retrieve and group
            include_connected: Whether to include connected memories
            
        Returns:
            Dict mapping topic -> {count, memories, connected_memories}
            
        Example:
            ```python
            grouped = await engine.get_memories_grouped_by_topics(
                ["father", "girlfriend", "project"]
            )
            
            print(f"Father: {grouped['father']['count']} memories")
            ```
        """
        if include_connected:
            query = """
            WITH $topics as topics
            UNWIND topics as topic
            
            MATCH (m)
            WHERE m.content CONTAINS topic
               OR m.person_name = topic
               OR topic IN m.participants
            
            OPTIONAL MATCH (m)-[r:MEMORY_RELATIONSHIP]-(connected)
            
            RETURN 
                topic,
                COUNT(DISTINCT m) as memory_count,
                COLLECT(DISTINCT {
                    id: m.id,
                    content: m.content,
                    type: labels(m),
                    timestamp: m.timestamp
                }) as memories,
                COLLECT(DISTINCT {
                    id: connected.id,
                    content: connected.content,
                    type: labels(connected)
                }) as connected_memories
            """
        else:
            query = """
            WITH $topics as topics
            UNWIND topics as topic
            
            MATCH (m)
            WHERE m.content CONTAINS topic
               OR m.person_name = topic
               OR topic IN m.participants
            
            RETURN 
                topic,
                COUNT(DISTINCT m) as memory_count,
                COLLECT(DISTINCT {
                    id: m.id,
                    content: m.content,
                    type: labels(m),
                    timestamp: m.timestamp
                }) as memories
            """
        
        results = await self.client.execute_query(
            query,
            parameters={'topics': topics}
        )
        
        # Convert to dictionary format
        grouped = {}
        for row in results:
            grouped[row['topic']] = {
                'count': row['memory_count'],
                'memories': row['memories']
            }
            if include_connected:
                grouped[row['topic']]['connected_memories'] = row.get(
                    'connected_memories', []
                )
        
        logger.info(f"Retrieved memories grouped by {len(topics)} topics")
        return grouped
    
    # ========================================================================
    # TIME-BASED RETRIEVAL METHODS
    # ========================================================================
    
    async def get_recent_memories(
        self,
        topic: Optional[str] = None,
        days: int = 7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get recent memories, optionally filtered by topic.
        
        Args:
            topic: Optional topic filter
            days: Number of days back to search
            limit: Maximum results
            
        Returns:
            List of recent memories
            
        Example:
            ```python
            # All memories from last 7 days
            recent = await engine.get_recent_memories(days=7)
            
            # Project memories from last 3 days
            recent = await engine.get_recent_memories(
                topic="project",
                days=3
            )
            ```
        """
        topic_filter = self._build_topic_filter(topic)
        parameters = {'cutoff_days': days, 'limit': limit}
        
        if topic:
            parameters['topic'] = topic
        
        query = f"""
        MATCH (m)
        WHERE m.timestamp > datetime() - duration({{days: $cutoff_days}})
        {_NOT_SUPERSEDED_m}
        {topic_filter}
        RETURN
            m.id as id,
            m.content as content,
            labels(m) as type,
            m.timestamp as timestamp,
            m.emotional_tone as emotional_tone
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} recent memories")
        return results
    
    async def get_memories_by_time_range(
        self,
        start_date: str,
        end_date: str,
        topic: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get memories within a specific time range.
        
        Args:
            start_date: Start date (ISO format: "2025-10-01T00:00:00")
            end_date: End date (ISO format)
            topic: Optional topic filter
            limit: Maximum results
            
        Returns:
            List of memories in time range
            
        Example:
            ```python
            memories = await engine.get_memories_by_time_range(
                "2025-10-01T00:00:00",
                "2025-10-31T23:59:59",
                topic="project"
            )
            ```
        """
        topic_filter = self._build_topic_filter(topic)
        parameters = {'start': start_date, 'end': end_date, 'limit': limit}
        
        if topic:
            parameters['topic'] = topic
        
        query = f"""
        MATCH (m)
        WHERE datetime(m.timestamp) >= datetime($start)
          AND datetime(m.timestamp) <= datetime($end)
        {_NOT_SUPERSEDED_m}
        {topic_filter}
        RETURN
            m.id as id,
            m.content as content,
            labels(m) as type,
            m.timestamp as timestamp
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} memories in time range")
        return results
    
    # ========================================================================
    # ADVANCED RETRIEVAL METHODS
    # ========================================================================
    
    async def get_most_connected_memories(
        self,
        topic: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find "memory hubs" - memories with the most connections.
        
        These are often important/central memories that link many concepts.
        
        Args:
            topic: Optional topic filter
            limit: Number of hubs to return
            
        Returns:
            List of highly connected memories with connection counts
            
        Example:
            ```python
            hubs = await engine.get_most_connected_memories(limit=5)
            ```
        """
        topic_filter = self._build_topic_filter(topic, prefix="WHERE")
        parameters = {'limit': limit}
        
        if topic:
            parameters['topic'] = topic
        
        query = f"""
        MATCH (m)-[r:MEMORY_RELATIONSHIP]-(connected)
        {topic_filter}
        RETURN 
            m.id as id,
            m.content as content,
            labels(m) as type,
            m.timestamp as timestamp,
            count(r) as connection_count
        ORDER BY connection_count DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} most connected memories")
        return results
    
    async def get_emotionally_significant_memories(
        self,
        topic: Optional[str] = None,
        min_emotional_intensity: float = 0.7,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get memories with high emotional significance.
        
        Args:
            topic: Optional topic filter
            min_emotional_intensity: Minimum emotional tone (0-1)
            limit: Maximum results
            
        Returns:
            List of emotionally significant memories
            
        Example:
            ```python
            emotional = await engine.get_emotionally_significant_memories(
                topic="father",
                min_emotional_intensity=0.7
            )
            ```
        """
        topic_filter = self._build_topic_filter(topic, prefix="AND")
        parameters = {'min_emotion': min_emotional_intensity, 'limit': limit}
        
        if topic:
            parameters['topic'] = topic
        
        query = f"""
        MATCH (m)
        WHERE abs(coalesce(m.emotional_tone, 0.0)) >= $min_emotion
        {_NOT_SUPERSEDED_m}
        {topic_filter}
        RETURN
            m.id as id,
            m.content as content,
            labels(m) as type,
            m.timestamp as timestamp,
            m.emotional_tone as emotional_tone
        ORDER BY abs(m.emotional_tone) DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} emotionally significant memories")
        return results

    async def get_recent_emotional_memories(
        self,
        days: int = 7,
        topic: Optional[str] = None,
        min_intensity: float = 0.3,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Recency-first emotional scan: memories from the last N days ordered by
        recency then intensity.

        This mirrors human memory primacy — when someone says "I'm stressed,"
        a close friend immediately filters recent events, not all-time records.
        Lower intensity threshold (0.3 vs 0.5) because recent mild stress is
        more relevant than ancient intense trauma.

        ORDER BY: timestamp DESC (recency), then abs(emotional_tone) DESC (intensity)
        """
        topic_filter = self._build_topic_filter(topic, prefix="AND")
        parameters = {
            "min_emotion": min_intensity,
            "days": days,
            "limit": limit,
        }
        if topic:
            parameters["topic"] = topic

        query = f"""
        MATCH (m)
        WHERE m.timestamp >= datetime() - duration({{days: $days}})
          AND abs(coalesce(m.emotional_tone, 0.0)) >= $min_emotion
        {_NOT_SUPERSEDED_m}
        {topic_filter}
        RETURN
            m.id              AS id,
            m.content         AS content,
            labels(m)         AS type,
            m.timestamp       AS timestamp,
            m.emotional_tone  AS emotional_tone,
            m.participants    AS participants
        ORDER BY m.timestamp DESC, abs(m.emotional_tone) DESC
        LIMIT $limit
        """
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} recent emotional memories (last {days}d)")
        return results

    async def get_memories_by_emotion(
        self,
        emotion: str,
        topic: Optional[str] = None,
        min_intensity: float = 0.5,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories filtered by specific emotion type.
        
        Supports: positive, negative, happy, excited, proud, sad,
                 angry, anxious, stressed, neutral
        
        Args:
            emotion: The emotion to search for
            topic: Optional topic filter
            min_intensity: Minimum emotional intensity (0-1)
            limit: Maximum results
            
        Returns:
            List of memories matching the emotion
            
        Example:
            ```python
            happy = await engine.get_memories_by_emotion("happy")
            sad = await engine.get_memories_by_emotion(
                "sad",
                topic="father",
                min_intensity=0.7
            )
            ```
        """
        # Get emotion filter from predefined ranges
        emotion_filter = self._get_emotion_filter(emotion, min_intensity)
        topic_filter = self._build_topic_filter(topic, prefix="AND")
        
        parameters = {
            'min_intensity': min_intensity,
            'limit': limit,
            'emotion': emotion
        }
        
        if topic:
            parameters['topic'] = topic
        
        query = f"""
        MATCH (m)
        WHERE {emotion_filter}
        {topic_filter}
        RETURN 
            m.id as id,
            m.content as content,
            labels(m) as type,
            m.timestamp as timestamp,
            m.emotional_tone as emotional_tone,
            m.emotion_label as emotion_label,
            m.participants as participants
        ORDER BY abs(m.emotional_tone) DESC, m.timestamp DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(query, parameters)
        logger.info(f"Retrieved {len(results)} memories with emotion '{emotion}'")
        return results
    
    async def get_emotions_summary(
        self,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a summary of emotions across memories.
        
        Args:
            topic: Optional topic filter
            
        Returns:
            Dictionary with emotion distribution and statistics
            
        Example:
            ```python
            summary = await engine.get_emotions_summary(topic="father")
            # Returns: positive_count, negative_count, neutral_count,
            #          avg_emotional_tone, most_emotional_memory
            ```
        """
        topic_filter = self._build_topic_filter(topic, prefix="WHERE")
        parameters = {}
        
        if topic:
            parameters['topic'] = topic
        
        # Get emotion counts
        query = f"""
        MATCH (m)
        {topic_filter}
        RETURN 
            count(CASE WHEN coalesce(m.emotional_tone, 0.0) >= 0.5 THEN 1 END) as positive_count,
            count(CASE WHEN coalesce(m.emotional_tone, 0.0) <= -0.5 THEN 1 END) as negative_count,
            count(CASE WHEN coalesce(m.emotional_tone, 0.0) > -0.5 AND coalesce(m.emotional_tone, 0.0) < 0.5 THEN 1 END) as neutral_count,
            avg(coalesce(m.emotional_tone, 0.0)) as avg_emotional_tone,
            max(abs(coalesce(m.emotional_tone, 0.0))) as max_intensity
        """
        
        results = await self.client.execute_query(query, parameters)
        
        # Get most emotional memory
        query_most = f"""
        MATCH (m)
        {topic_filter}
        RETURN 
            m.id as id,
            m.content as content,
            m.emotional_tone as emotional_tone
        ORDER BY abs(m.emotional_tone) DESC
        LIMIT 1
        """
        
        most_emotional = await self.client.execute_query(query_most, parameters)
        
        summary = results[0] if results else {}
        summary['most_emotional_memory'] = most_emotional[0] if most_emotional else None
        
        logger.info("Retrieved emotions summary")
        return summary
    
    # ========================================================================
    # RELATIONSHIP STRENGTH METHODS
    # ========================================================================
    
    async def get_strongly_connected_memories(
        self,
        memory_id: str,
        min_strength: float = 0.7,
        max_hops: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get only STRONGLY connected memories (above strength threshold).
        
        Filters connections by minimum relationship strength and calculates
        cumulative strength across multi-hop paths.
        
        Args:
            memory_id: Starting memory ID
            min_strength: Minimum relationship strength (0.0-1.0)
            max_hops: Maximum traversal depth
            limit: Maximum results
            
        Returns:
            List of strongly connected memories with strength scores
            
        Example:
            ```python
            strong = await engine.get_strongly_connected_memories(
                "exp_12345",
                min_strength=0.7,
                max_hops=2
            )
            # Only returns connections with strength >= 0.7
            ```
        """
        query = f"""
        MATCH (root {{id: $memory_id}})
        MATCH path = (root)-[r:MEMORY_RELATIONSHIP*1..{max_hops}]-(connected)
        
        WITH connected, 
             relationships(path) as rels,
             reduce(s = 1.0, rel in rels | s * rel.strength) as path_strength
        
        WHERE path_strength >= $min_strength
        
        RETURN DISTINCT
            connected.id as id,
            connected.content as content,
            labels(connected) as type,
            path_strength as connection_strength,
            length(path) as distance
        ORDER BY path_strength DESC, distance ASC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(
            query,
            parameters={
                'memory_id': memory_id,
                'max_hops': max_hops,
                'min_strength': min_strength,
                'limit': limit
            }
        )
        
        logger.info(
            f"Retrieved {len(results)} strongly connected memories "
            f"(min_strength={min_strength})"
        )
        return results
    
    async def get_memories_with_weighted_relevance(
        self,
        topic: str,
        include_weak_connections: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get memories with weighted relevance score.
        
        Calculates relevance as: strength × importance × confidence
        Prioritizes memories that are strongly connected AND important.
        
        Args:
            topic: Topic to search for
            include_weak_connections: Include weak connections (strength < 0.5)
            limit: Maximum results
            
        Returns:
            List of memories sorted by weighted relevance
            
        Example:
            ```python
            memories = await engine.get_memories_with_weighted_relevance(
                "project",
                include_weak_connections=False
            )
            # Results sorted by: strength × importance × confidence
            ```
        """
        strength_filter = "" if include_weak_connections else "WHERE r.strength >= 0.5"
        
        query = f"""
        MATCH (root)
        WHERE toLower(coalesce(root.content, "")) CONTAINS toLower($topic)
           OR toLower(coalesce(root.person_name, "")) = toLower($topic)
           OR toLower(coalesce(root.concept, "")) CONTAINS toLower($topic)
           OR any(p IN coalesce(root.participants, []) WHERE toLower(p) = toLower($topic))
        
        MATCH (root)-[r:MEMORY_RELATIONSHIP]-(connected)
        {strength_filter}
        
        WITH 
            connected,
            coalesce(r.strength, 0.0) as connection_strength,
            coalesce(connected.importance_score, 0.5) as importance,
            (coalesce(r.strength, 0.0) * coalesce(connected.importance_score, 0.5) * coalesce(r.confidence, 1.0)) as weighted_relevance
        
        RETURN 
            connected.id as id,
            connected.content as content,
            labels(connected) as type,
            connection_strength,
            importance,
            weighted_relevance
        ORDER BY weighted_relevance DESC
        LIMIT $limit
        """
        
        results = await self.client.execute_query(
            query,
            parameters={'topic': topic, 'limit': limit}
        )
        
        logger.info(
            f"Retrieved {len(results)} memories with weighted relevance "
            f"for '{topic}'"
        )
        return results
    
    async def reinforce_memory_connection(
        self,
        from_memory_id: str,
        to_memory_id: str,
        strength_boost: float = 0.05
    ) -> Dict[str, Any]:
        """
        Strengthen a connection between two memories.
        
        This mimics how human memory works: frequently accessed connections
        become stronger over time.
        
        Args:
            from_memory_id: Source memory ID
            to_memory_id: Target memory ID
            strength_boost: Amount to increase strength (default: 0.05)
            
        Returns:
            Updated relationship with new strength
            
        Example:
            ```python
            updated = await engine.reinforce_memory_connection(
                "exp_001",
                "know_123",
                strength_boost=0.05
            )
            # strength: 0.7 → 0.75
            # evidence_count: 3 → 4
            ```
        """
        query = """
        MATCH (m1 {id: $from_id})-[r:MEMORY_RELATIONSHIP]-(m2 {id: $to_id})
        SET 
            r.strength = CASE 
                WHEN r.strength + $boost <= 1.0 THEN r.strength + $boost
                ELSE 1.0 
            END,
            r.evidence_count = r.evidence_count + 1,
            r.last_reinforced = datetime()
        RETURN 
            r.strength as new_strength,
            r.evidence_count as evidence_count,
            r.confidence as confidence
        """
        
        results = await self.client.execute_query(
            query,
            parameters={
                'from_id': from_memory_id,
                'to_id': to_memory_id,
                'boost': strength_boost
            }
        )
        
        if results:
            logger.info(
                f"Reinforced connection: {from_memory_id} → {to_memory_id} "
                f"(new strength: {results[0]['new_strength']:.2f})"
            )
        
        return results[0] if results else {}
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    async def ensure_fulltext_index(self) -> None:
        """
        Create the BM25 Lucene full-text index if it does not already exist.
        Safe to call on every startup — the IF NOT EXISTS clause makes it idempotent.
        Must be called before bm25_search() can be used.
        """
        query = """
        CREATE FULLTEXT INDEX memory_fts IF NOT EXISTS
        FOR (n:ExperienceMemory|KnowledgeMemory|RelationshipMemory)
        ON EACH [n.content, n.participants, n.concept, n.person_name, n.tags]
        """
        try:
            await self.client.execute_query(query)
            logger.info("Full-text index 'memory_fts' ready")
        except Exception as exc:
            logger.warning(f"Full-text index creation returned: {exc}")

    async def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the memory graph.
        
        Returns:
            Dictionary with node counts, relationship counts, timestamp
            
        Example:
            ```python
            stats = await engine.get_memory_statistics()
            print(f"Total relationships: {stats['total_relationships']}")
            ```
        """
        # Get node counts by type
        query = """
        MATCH (n)
        RETURN labels(n) as type, count(n) as count
        ORDER BY count DESC
        """
        node_counts = await self.client.execute_query(query)
        
        # Get relationship count
        query = """
        MATCH ()-[r:MEMORY_RELATIONSHIP]->()
        RETURN count(r) as relationship_count
        """
        rel_result = await self.client.execute_query(query)
        relationship_count = rel_result[0]['relationship_count'] if rel_result else 0
        
        return {
            'node_counts': node_counts,
            'total_relationships': relationship_count,
            'timestamp': datetime.now().isoformat()
        }
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _build_type_filter(
        self,
        memory_types: Optional[List[MemoryType]]
    ) -> str:
        """Build Cypher filter clause for memory types."""
        if not memory_types or MemoryType.ALL in memory_types:
            return ""
        
        labels = [mt.value for mt in memory_types]
        return f"AND ({' OR '.join([f'm:{label}' for label in labels])})"
    
    def _build_topic_filter(
        self,
        topic: Optional[str],
        prefix: str = "AND"
    ) -> str:
        """Build Cypher filter clause for topic."""
        if not topic:
            return ""
        
        return f"""
        {prefix} (toLower(coalesce(m.content, "")) CONTAINS toLower($topic)
            OR toLower(coalesce(m.person_name, "")) = toLower($topic)
            OR toLower(coalesce(m.concept, "")) CONTAINS toLower($topic)
            OR any(p IN coalesce(m.participants, []) WHERE toLower(p) = toLower($topic)))
        """
    
    def _build_multi_topic_conditions(
        self,
        topics: List[str],
        match_all: bool
    ) -> tuple[str, Dict[str, str]]:
        """Build WHERE conditions and parameters for multiple topics."""
        logic_operator = " AND " if match_all else " OR "
        
        conditions = logic_operator.join([
            f"(toLower(coalesce(m.content, \"\")) CONTAINS toLower($topic{i}) OR "
            f"toLower(coalesce(m.person_name, \"\")) = toLower($topic{i}) OR "
            f"toLower(coalesce(m.concept, \"\")) CONTAINS toLower($topic{i}) OR "
            f"any(p IN coalesce(m.participants, []) WHERE toLower(p) = toLower($topic{i})))"
            for i in range(len(topics))
        ])
        
        parameters = {f'topic{i}': topic for i, topic in enumerate(topics)}
        
        return conditions, parameters
    
    def _get_emotion_filter(
        self,
        emotion: str,
        min_intensity: float
    ) -> str:
        """Get Cypher filter for specific emotion type."""
        emotion_filters = {
            "positive": "coalesce(m.emotional_tone, 0.0) >= $min_intensity",
            "negative": "coalesce(m.emotional_tone, 0.0) <= -$min_intensity",
            "happy": "coalesce(m.emotional_tone, 0.0) >= 0.7",
            "excited": "coalesce(m.emotional_tone, 0.0) >= 0.8",
            "proud": "coalesce(m.emotional_tone, 0.0) >= 0.6",
            "sad": "coalesce(m.emotional_tone, 0.0) <= -0.6",
            "angry": "coalesce(m.emotional_tone, 0.0) <= -0.7",
            "anxious": "coalesce(m.emotional_tone, 0.0) <= -0.5",
            "stressed": "coalesce(m.emotional_tone, 0.0) <= -0.5",
            "neutral": "coalesce(m.emotional_tone, 0.0) > -0.3 AND coalesce(m.emotional_tone, 0.0) < 0.3"
        }
        
        return emotion_filters.get(
            emotion.lower(),
            "toLower(coalesce(m.emotion_label, \"\")) = toLower($emotion)"
        )

    async def _resolve_seed_nodes(self, entities: List[str]) -> List[str]:
        """
        Resolve entity strings to Neo4j node IDs via direct property match.
        No embedding — matches person_name, concept, or participants list.
        Returns up to 3 matching node IDs per entity.
        """
        if not entities:
            return []

        seed_ids: List[str] = []
        for entity in entities:
            query = """
            MATCH (seed)
            WHERE toLower(coalesce(seed.person_name, "")) = toLower($entity)
               OR toLower(coalesce(seed.concept, ""))     = toLower($entity)
               OR any(p IN coalesce(seed.participants, [])
                      WHERE toLower(p) = toLower($entity))
            RETURN seed.id AS id
            LIMIT 3
            """
            rows = await self.client.execute_query(query, parameters={"entity": entity})
            seed_ids.extend(r["id"] for r in rows if r.get("id"))

        logger.debug(f"Seed resolution: {entities} → {len(seed_ids)} node(s)")
        return seed_ids

    @staticmethod
    def _escape_lucene(term: str) -> str:
        """Escape Lucene special characters in a search term."""
        special = set('+-&|!(){}[]^"~*?:\\/')
        return "".join(f"\\{c}" if c in special else c for c in term)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_memory_retrieval_engine(
    neo4j_client: Neo4jClient,
    embedding_utils: Optional[EmbeddingUtils] = None
) -> MemoryRetrievalEngine:
    """
    Factory function to create a MemoryRetrievalEngine instance.
    
    Args:
        neo4j_client: Connected Neo4j client
        embedding_utils: Optional EmbeddingUtils instance
        
    Returns:
        Initialized MemoryRetrievalEngine
        
    Example:
        ```python
        from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
        from memory.processing.embedding_utils import EmbeddingUtils
        
        client = create_neo4j_client("bolt://localhost:7687", "neo4j", "password")
        await client.connect()
        
        engine = create_memory_retrieval_engine(client, EmbeddingUtils())
        ```
    """
    return MemoryRetrievalEngine(neo4j_client, embedding_utils)


