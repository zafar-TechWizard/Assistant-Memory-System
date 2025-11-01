# SOFI Long-Term Memory System

> A sophisticated graph-based memory system for AI assistants that mimics human memory recall and association.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Core Components](#core-components)
- [Memory Retrieval Engine](#memory-retrieval-engine)
  - [Basic Retrieval](#basic-retrieval)
  - [Graph Traversal](#graph-traversal)
  - [Semantic Search](#semantic-search)
  - [Grouped Retrieval](#grouped-retrieval)
  - [Time-Based Retrieval](#time-based-retrieval)
  - [Advanced Retrieval](#advanced-retrieval)
  - [Relationship Strength](#relationship-strength)
  - [Memory Management](#memory-management)
  - [Utility Methods](#utility-methods)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

SOFI's Long-Term Memory System is a production-ready, graph-based memory architecture that stores and retrieves memories using Neo4j. It mimics human memory by:

- **Associative Recall**: Following connections between related memories
- **Emotional Context**: Storing and filtering by emotional significance
- **Adaptive Strength**: Reinforcing frequently used connections
- **Semantic Understanding**: Finding memories by meaning, not just keywords
- **Temporal Awareness**: Tracking when memories were created and accessed

### Key Features

✅ **18+ Retrieval Methods** - From basic keyword search to complex graph traversal  
✅ **Emotion-Aware** - Filter memories by emotional tone and significance  
✅ **Relationship Strength** - Prioritize strongly connected memories  
✅ **Graph Traversal** - Follow memory associations 1-3 hops deep  
✅ **Semantic Search** - Find memories by meaning using vector embeddings  
✅ **Time-Based Queries** - Retrieve recent memories or specific date ranges  
✅ **Self-Optimizing** - Connections strengthen with repeated use  
✅ **Production-Ready** - Async/await, error handling, logging, type hints

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    SOFI APPLICATION                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Short-Term Memory (conversation.json)             │  │
│  │           ↓ consolidation.py                       │  │
│  │  Long-Term Memory (Neo4j Graph)                    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Memory Retrieval Engine                           │  │
│  │  • get_memories_by_topic()                         │  │
│  │  • get_memories_with_connected_nodes()             │  │
│  │  • semantic_search()                               │  │
│  │  • get_memories_by_emotion()                       │  │
│  │  • get_strongly_connected_memories()               │  │
│  │  • ... 13+ more methods                            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
                          ↓
        ┌───────────────────────────────────┐
        │   Neo4j Graph Database            │
        │                                   │
        │  Nodes:                           │
        │  • ExperienceMemory               │
        │  • KnowledgeMemory                │
        │  • RelationshipMemory             │
        │                                   │
        │  Relationships:                   │
        │  • MEMORY_RELATIONSHIP            │
        │    - strength (0.0-1.0)           │
        │    - confidence (0.0-1.0)         │
        │    - evidence_count               │
        └───────────────────────────────────┘
```
---

## 🔧 Core Components

### 1. `node_models.py` - Memory Node Models

Defines three types of memory nodes using Pydantic for validation:

#### **ExperienceMemoryNode**
Stores personal experiences and events.

```python
{
    "id": "exp_12345",
    "content": "Had dinner with father, discussed career plans",
    "timestamp": "2025-11-01T18:30:00",
    "emotional_tone": 0.8,           # Positive emotion
    "personal_impact": 0.9,          # High impact
    "social_significance": 0.7,
    "participants": ["Father", "User"],
    "location": "Home",
    "importance_score": 0.85
}
```

**Key Fields:**
- `content` - The actual experience description
- `emotional_tone` - Emotional valence (-1.0 to 1.0)
- `personal_impact` - How much it affected the user (0.0 to 1.0)
- `participants` - People involved in the experience

#### **KnowledgeMemoryNode**
Stores learned facts, concepts, and skills.

```python
{
    "id": "know_789",
    "content": "Python uses duck typing",
    "concept": "Python programming",
    "category": "programming",
    "confidence": 0.95,
    "source": "university course",
    "importance_score": 0.7
}
```

**Key Fields:**
- `concept` - Main concept/topic of the knowledge
- `confidence` - How confident the system is (0.0 to 1.0)
- `category` - Classification (e.g., "programming", "health")

#### **RelationshipMemoryNode**
Stores information about people and relationships.

```python
{
    "id": "rel_456",
    "content": "Father is supportive and gives career advice",
    "person_name": "Father",
    "emotional_connection": 0.9,     # Strong positive bond
    "trust_level": 0.95,
    "intimacy_level": 0.8,
    "relationship_strength": 0.85,
    "interaction_frequency": "weekly"
}
```

**Key Fields:**
- `person_name` - Name of the person
- `emotional_connection` - Emotional bond strength (-1.0 to 1.0)
- `trust_level` - Trust in the person (0.0 to 1.0)
- `relationship_strength` - Overall relationship quality

---

### 2. `relationship_models.py` - Memory Relationships

Defines how memories connect to each other in the graph.

#### **MemoryRelationshipEdge**

```python
{
    "from_memory_id": "exp_001",
    "to_memory_id": "know_123",
    "relationship_type": "EXPERIENCE_TO_KNOWLEDGE",
    "strength": 0.9,                 # Strong connection
    "confidence": 0.95,              # High confidence
    "evidence_count": 3,             # Observed 3 times
    "last_reinforced": "2025-11-01T15:00:00"
}
```

**Key Fields:**
- `strength` - Connection strength (0.0 to 1.0)
- `confidence` - Confidence in the connection
- `evidence_count` - How many times the connection was observed
- `last_reinforced` - When the connection was last strengthened

**Relationship Types:**
- `EXPERIENCE_TO_KNOWLEDGE` - Experience led to learning
- `EXPERIENCE_TO_RELATIONSHIP` - Experience involved a person
- `KNOWLEDGE_TO_EXPERIENCE` - Applied knowledge in experience
- `ASSOCIATED_WITH` - General association

---

### 3. `neo4j_client.py` - Database Client

Manages connection to Neo4j and executes Cypher queries.

```python
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client

# Create and connect
client = create_neo4j_client(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password"
)
await client.connect()

# Execute custom queries
result = await client.execute_query(
    query="MATCH (m:ExperienceMemory) RETURN m.content as content LIMIT 5",
    parameters={}
)

# Close connection
await client.disconnect()
```

**Key Methods:**
- `connect()` - Establish connection to Neo4j
- `execute_query(query, parameters)` - Execute Cypher query
- `disconnect()` - Close connection
- `verify_connectivity()` - Test connection

---

### 4. `embedding_utils.py` - Semantic Embeddings

Generates vector embeddings for semantic search.

```python
from memory.processing.embedding_utils import EmbeddingUtils

embed_util = EmbeddingUtils()

# Generate embedding for text
vector = embed_util.generate_embedding("career advice from father")
# Returns: [0.123, -0.456, 0.789, ...] (768-dimensional vector)

# Used internally by semantic_search()
```

---

## 🚀 Memory Retrieval Engine

The heart of the system. Provides 18+ methods to retrieve memories in any manner needed.

### Initialization

```python
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.processing.embedding_utils import EmbeddingUtils
from memory.processing.memory_retrieval_engine import create_memory_retrieval_engine

# Initialize
client = create_neo4j_client("bolt://localhost:7687", "neo4j", "password")
await client.connect()

engine = create_memory_retrieval_engine(client, EmbeddingUtils())
```

---

## 📖 Retrieval Methods

### Basic Retrieval

#### 1. `get_memories_by_topic()`

Retrieve all memories about a single topic or person.

**Parameters:**
- `topic` (str) - Topic or person name to search for
- `memory_types` (Optional[List[MemoryType]]) - Filter by memory types
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of memory dictionaries

**Example:**

```python
# Get all memories about "father"
memories = await engine.get_memories_by_topic("father")

for memory in memories:
    print(f"[{memory['type']}] {memory['content']}")
    print(f"  Timestamp: {memory['timestamp']}")
    print(f"  Emotion: {memory['emotional_tone']}")

# Output:
# [ExperienceMemory] Had dinner with father, discussed career
#   Timestamp: 2025-11-01T18:30:00
#   Emotion: 0.8
# [RelationshipMemory] Father is supportive and wise
#   Timestamp: 2025-10-28T15:00:00
#   Emotion: 0.9
```

**Filter by memory type:**

```python
from memory_retrieval_engine import MemoryType

# Get only experiences about "project"
experiences = await engine.get_memories_by_topic(
    "project",
    memory_types=[MemoryType.EXPERIENCE]
)

# Get only knowledge about "Python"
knowledge = await engine.get_memories_by_topic(
    "Python",
    memory_types=[MemoryType.KNOWLEDGE]
)
```

---

#### 2. `get_memories_by_multiple_topics()`

Retrieve memories about multiple topics/people at once.

**Parameters:**
- `topics` (List[str]) - List of topics to search for
- `match_all` (bool) - AND logic (True) or OR logic (False, default)
- `limit` (int) - Maximum results (default: 100)

**Returns:** List of memory dictionaries

**Example:**

```python
# OR logic: Get memories about father OR girlfriend OR project
memories = await engine.get_memories_by_multiple_topics(
    ["father", "girlfriend", "project"]
)

print(f"Found {len(memories)} memories about any of these topics")
# Output: Found 47 memories about any of these topics


# AND logic: Get memories mentioning BOTH father AND project
memories = await engine.get_memories_by_multiple_topics(
    ["father", "project"],
    match_all=True
)

print(f"Found {len(memories)} memories about both topics")
# Output: Found 5 memories about both topics
```

---

### Graph Traversal

#### 3. `get_memories_with_connected_nodes()`

**Most Powerful Method** - Retrieves memories about a topic AND all connected memories.

This follows graph relationships to find related memories, mimicking human associative memory recall.

**Parameters:**
- `topic` (str) - Topic to search for
- `max_hops` (int) - Maximum relationship hops (1-3 recommended, default: 2)
- `limit` (int) - Maximum root memories (default: 50)

**Returns:** List of dictionaries with root memory and connected memories

**What are "hops"?**
- **1 hop** = Directly connected memories
- **2 hops** = Memories connected to those memories (friend of a friend)
- **3 hops** = Even more distant connections

**Example:**

```python
# Get memories about "father" + everything connected
results = await engine.get_memories_with_connected_nodes(
    "father",
    max_hops=2
)

for item in results:
    print(f"\n📌 Main Memory: {item['root_content']}")
    print(f"   Type: {item['root_type']}")
    print(f"   Connected: {len(item['connected_memories'])} related memories")

    # Show connected memories
    for conn in item['connected_memories'][:3]:  # Show first 3
        if conn['id']:
            print(f"   → {conn['content'][:80]}... (distance: {conn['distance']})")

# Output:
# 📌 Main Memory: Father gave me career advice yesterday
#    Type: ['ExperienceMemory']
#    Connected: 8 related memories
#    → Career planning is important for long-term success... (distance: 1)
#    → Father has 30 years of engineering experience... (distance: 1)
#    → Applied for software engineering job... (distance: 2)
```

**Use Cases:**
- **max_hops=1**: Fast, focused context (10-50ms)
- **max_hops=2**: Balanced context (50-200ms) - **Recommended default**
- **max_hops=3**: Comprehensive context (200-500ms) - May return 100+ memories

---

#### 4. `get_connected_nodes_only()`

Get ONLY the connected memories for a specific memory (excludes the root memory).

**Parameters:**
- `memory_id` (str) - ID of the memory to find connections for
- `max_hops` (int) - Maximum traversal depth (default: 2)
- `limit` (int) - Maximum connected nodes (default: 100)

**Returns:** List of connected memory dictionaries

**Example:**

```python
# Get all memories connected to a specific experience
connected = await engine.get_connected_nodes_only(
    "exp_12345",
    max_hops=2
)

print(f"Found {len(connected)} connected memories")

for node in connected[:5]:
    print(f"- [{node['type']}] {node['content'][:80]}...")
    print(f"  Distance: {node['distance']} hops")
    print(f"  Path: {' → '.join(node['relationship_path'])}")

# Output:
# Found 23 connected memories
# - [KnowledgeMemory] Career planning requires goal-setting...
#   Distance: 1 hops
#   Path: EXPERIENCE_TO_KNOWLEDGE
# - [ExperienceMemory] Applied for new job at tech company...
#   Distance: 2 hops
#   Path: EXPERIENCE_TO_KNOWLEDGE → KNOWLEDGE_TO_EXPERIENCE
```

---

#### 5. `get_memory_cluster()`

Find "memory communities" - groups of highly interconnected memories.

**Parameters:**
- `starting_memory_id` (str) - Memory to start clustering from
- `max_hops` (int) - Maximum distance to explore (default: 3)
- `min_connections` (int) - Minimum connections required (default: 2)

**Returns:** Dictionary with cluster statistics and members

**Example:**

```python
# Find cluster around a memory
cluster = await engine.get_memory_cluster(
    "exp_12345",
    max_hops=3,
    min_connections=2
)

print(f"Cluster size: {cluster['cluster_size']} memories")
print(f"Starting from: {cluster['starting_memory']}")

for member in cluster['members'][:5]:
    print(f"- {member['content'][:80]}...")
    print(f"  Connections: {member['connection_count']}")

# Output:
# Cluster size: 15 memories
# Starting from: exp_12345
# - Career planning is essential for success...
#   Connections: 7
# - Father has extensive engineering background...
#   Connections: 5
```

---

### Semantic Search

#### 6. `semantic_search()`

Search memories by **meaning**, not just keywords. Uses vector embeddings to find semantically similar memories.

**Parameters:**
- `query_text` (str) - Natural language query
- `top_k` (int) - Number of results (default: 10)
- `include_connected` (bool) - Include connected memories (default: False)
- `max_hops` (int) - Hops for connected memories (default: 1)

**Returns:** List of memories with similarity scores

**Example:**

```python
# Search by meaning
memories = await engine.semantic_search(
    "family advice and relationships",
    top_k=5
)

print("Semantically similar memories:")
for memory in memories:
    print(f"\n- {memory['content']}")
    print(f"  Similarity: {memory['score']:.2f}")
    print(f"  Type: {memory['type']}")

# Output:
# Semantically similar memories:
# - Father gave me career advice yesterday
#   Similarity: 0.92
#   Type: ['ExperienceMemory']
# - Mother always supports my decisions
#   Similarity: 0.87
#   Type: ['RelationshipMemory']
# - Brother helped me with job interview
#   Similarity: 0.83
#   Type: ['ExperienceMemory']
```

**Why it's powerful:**
- Query: "family advice" finds "father", "mother", "parent", "dad", etc.
- Query: "happy moments" finds positive experiences even without the word "happy"
- Query: "stress at work" finds related memories about "pressure", "deadline", "anxiety"

**With connected memories:**

```python
# Include connected context
memories = await engine.semantic_search(
    "career guidance",
    top_k=3,
    include_connected=True,
    max_hops=1
)

for memory in memories:
    print(f"\n- {memory['content']} (similarity: {memory['score']:.2f})")
    print(f"  Connected: {len(memory['connected_memories'])} memories")
```

---

### Grouped Retrieval

#### 7. `get_memories_grouped_by_topics()`

Retrieve memories organized by topic into separate groups.

**Parameters:**
- `topics` (List[str]) - Topics to retrieve and group
- `include_connected` (bool) - Include connected memories (default: False)

**Returns:** Dictionary mapping topic → {count, memories, connected_memories}

**Example:**

```python
# Group memories by topics
grouped = await engine.get_memories_grouped_by_topics(
    ["father", "girlfriend", "project", "colleague"]
)

# Access each topic's memories
for topic, data in grouped.items():
    print(f"\n{topic.upper()}:")
    print(f"  Count: {data['count']} memories")

    # Show first 3 memories
    for memory in data['memories'][:3]:
        print(f"  - {memory['content'][:80]}...")

# Output:
# FATHER:
#   Count: 15 memories
#   - Father gave career advice yesterday...
#   - Father has 30 years of engineering experience...
#   - Had dinner with father, discussed future plans...
#
# GIRLFRIEND:
#   Count: 23 memories
#   - Girlfriend supported my job decision...
#   - Anniversary dinner at Italian restaurant...
```

**Access specific topic:**

```python
# Get just father memories
father_memories = grouped['father']['memories']
print(f"Found {len(father_memories)} memories about father")

for memory in father_memories:
    print(f"- [{memory['timestamp']}] {memory['content']}")
```

**With connected memories:**

```python
grouped = await engine.get_memories_grouped_by_topics(
    ["father", "project"],
    include_connected=True
)

for topic, data in grouped.items():
    print(f"\n{topic}:")
    print(f"  Direct memories: {data['count']}")
    print(f"  Connected memories: {len(data['connected_memories'])}")
```

---

### Time-Based Retrieval

#### 8. `get_recent_memories()`

Get recent memories from the last N days.

**Parameters:**
- `topic` (Optional[str]) - Optional topic filter
- `days` (int) - Number of days back (default: 7)
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of recent memories

**Example:**

```python
# Get all memories from last 7 days
recent = await engine.get_recent_memories(days=7)

print(f"Found {len(recent)} memories from last week:")
for memory in recent:
    print(f"- [{memory['timestamp']}] {memory['content'][:80]}...")

# Output:
# Found 34 memories from last week:
# - [2025-11-01T18:30:00] Had dinner with father...
# - [2025-10-31T14:00:00] Completed project milestone...


# Get project-related memories from last 3 days
recent = await engine.get_recent_memories(
    topic="project",
    days=3
)

print(f"Project memories from last 3 days: {len(recent)}")
```

---

#### 9. `get_memories_by_time_range()`

Get memories within a specific date range.

**Parameters:**
- `start_date` (str) - Start date (ISO format: "2025-10-01T00:00:00")
- `end_date` (str) - End date (ISO format)
- `topic` (Optional[str]) - Optional topic filter
- `limit` (int) - Maximum results (default: 100)

**Returns:** List of memories in time range

**Example:**

```python
# Get all memories from October 2025
memories = await engine.get_memories_by_time_range(
    start_date="2025-10-01T00:00:00",
    end_date="2025-10-31T23:59:59"
)

print(f"October memories: {len(memories)}")


# Get project memories from specific week
memories = await engine.get_memories_by_time_range(
    start_date="2025-10-20T00:00:00",
    end_date="2025-10-27T23:59:59",
    topic="project"
)

for memory in memories:
    print(f"- [{memory['timestamp']}] {memory['content'][:80]}...")
```

---

### Advanced Retrieval

#### 10. `get_most_connected_memories()`

Find "memory hubs" - memories with the most connections. These are often important/central concepts.

**Parameters:**
- `topic` (Optional[str]) - Optional topic filter
- `limit` (int) - Number of hubs to return (default: 10)

**Returns:** List of highly connected memories with connection counts

**Example:**

```python
# Find most connected memories overall
hubs = await engine.get_most_connected_memories(limit=5)

print("Memory hubs (most connected):")
for hub in hubs:
    print(f"\n- {hub['content'][:80]}...")
    print(f"  Connections: {hub['connection_count']}")
    print(f"  Type: {hub['type']}")

# Output:
# Memory hubs (most connected):
# - Career planning is essential for success...
#   Connections: 12
#   Type: ['KnowledgeMemory']
# - Father is supportive and wise...
#   Connections: 9
#   Type: ['RelationshipMemory']


# Find most connected memories about "project"
project_hubs = await engine.get_most_connected_memories(
    topic="project",
    limit=5
)
```

---

#### 11. `get_emotionally_significant_memories()`

Get memories with high emotional intensity.

**Parameters:**
- `topic` (Optional[str]) - Optional topic filter
- `min_emotional_intensity` (float) - Minimum emotion (0.0-1.0, default: 0.7)
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of emotionally significant memories

**Example:**

```python
# Get all highly emotional memories
emotional = await engine.get_emotionally_significant_memories(
    min_emotional_intensity=0.7
)

print(f"Found {len(emotional)} emotionally significant memories:")
for memory in emotional:
    emotion = "😊 Positive" if memory['emotional_tone'] > 0 else "😔 Negative"
    print(f"- {emotion} ({memory['emotional_tone']:.2f}) {memory['content'][:80]}...")

# Output:
# Found 18 emotionally significant memories:
# - 😊 Positive (0.9) Got promoted to senior engineer...
# - 😊 Positive (0.85) Anniversary dinner with girlfriend...
# - 😔 Negative (-0.8) Failed important project deadline...


# Get emotional memories about specific topic
father_emotional = await engine.get_emotionally_significant_memories(
    topic="father",
    min_emotional_intensity=0.7
)
```

---

#### 12. `get_memories_by_emotion()`

Retrieve memories filtered by specific emotion type.

**Supported Emotions:**
- Positive: `"positive"`, `"happy"`, `"excited"`, `"proud"`
- Negative: `"negative"`, `"sad"`, `"angry"`, `"anxious"`, `"stressed"`
- Neutral: `"neutral"`

**Parameters:**
- `emotion` (str) - Emotion to search for
- `topic` (Optional[str]) - Optional topic filter
- `min_intensity` (float) - Minimum intensity (0.0-1.0, default: 0.5)
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of memories matching the emotion

**Example:**

```python
# Get all happy memories
happy = await engine.get_memories_by_emotion("happy")

print(f"Found {len(happy)} happy memories:")
for memory in happy:
    print(f"😊 {memory['content'][:80]}...")
    print(f"   Intensity: {memory['emotional_tone']}")

# Output:
# Found 12 happy memories:
# 😊 Got promoted to senior engineer position...
#    Intensity: 0.9
# 😊 Anniversary celebration with girlfriend...
#    Intensity: 0.85


# Get sad memories about specific topic
sad_father = await engine.get_memories_by_emotion(
    emotion="sad",
    topic="father",
    min_intensity=0.6
)

print(f"Found {len(sad_father)} sad memories about father")


# Get stressed memories about project
stressed = await engine.get_memories_by_emotion(
    emotion="stressed",
    topic="project"
)

for memory in stressed:
    print(f"😰 {memory['content'][:80]}...")
```

---

#### 13. `get_emotions_summary()`

Get statistics about emotions across memories.

**Parameters:**
- `topic` (Optional[str]) - Optional topic filter

**Returns:** Dictionary with emotion distribution and statistics

**Example:**

```python
# Get overall emotion summary
summary = await engine.get_emotions_summary()

print("Overall emotions:")
print(f"  Positive: {summary['positive_count']} memories")
print(f"  Negative: {summary['negative_count']} memories")
print(f"  Neutral: {summary['neutral_count']} memories")
print(f"  Average tone: {summary['avg_emotional_tone']:.2f}")
print(f"  Max intensity: {summary['max_intensity']:.2f}")

if summary['most_emotional_memory']:
    print(f"\nMost emotional memory:")
    print(f"  {summary['most_emotional_memory']['content'][:80]}...")
    print(f"  Tone: {summary['most_emotional_memory']['emotional_tone']}")

# Output:
# Overall emotions:
#   Positive: 45 memories
#   Negative: 12 memories
#   Neutral: 8 memories
#   Average tone: 0.32
#   Max intensity: 0.95
#
# Most emotional memory:
#   Got promoted after 2 years of hard work...
#   Tone: 0.95


# Get emotions summary for specific topic
father_emotions = await engine.get_emotions_summary(topic="father")

print(f"\nFather emotions:")
print(f"  Positive: {father_emotions['positive_count']}")
print(f"  Negative: {father_emotions['negative_count']}")
print(f"  Overall tone: {father_emotions['avg_emotional_tone']:.2f}")


# Compare emotions across topics
topics = ["father", "girlfriend", "project", "colleague"]

for topic in topics:
    summary = await engine.get_emotions_summary(topic=topic)
    print(f"\n{topic}: {summary['avg_emotional_tone']:.2f} "
          f"(+{summary['positive_count']} / -{summary['negative_count']})")

# Output:
# father: 0.65 (+12 / -2)
# girlfriend: 0.72 (+18 / -3)
# project: -0.15 (+8 / -15)
# colleague: 0.42 (+10 / -5)
```

---

### Relationship Strength

#### 14. `get_strongly_connected_memories()`

Get ONLY strongly connected memories (filters by minimum relationship strength).

**Parameters:**
- `memory_id` (str) - Starting memory ID
- `min_strength` (float) - Minimum strength (0.0-1.0, default: 0.7)
- `max_hops` (int) - Maximum traversal depth (default: 2)
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of strongly connected memories with strength scores

**Example:**

```python
# Get only strong connections (strength >= 0.7)
strong = await engine.get_strongly_connected_memories(
    memory_id="exp_12345",
    min_strength=0.7,
    max_hops=2
)

print(f"Found {len(strong)} strongly connected memories:")
for memory in strong:
    print(f"\n- {memory['content'][:80]}...")
    print(f"  Connection strength: {memory['connection_strength']:.2f}")
    print(f"  Distance: {memory['distance']} hops")

# Output:
# Found 8 strongly connected memories:
# - Career planning requires clear goals...
#   Connection strength: 0.95
#   Distance: 1 hops
# - Applied for senior engineer position...
#   Connection strength: 0.72  (0.9 × 0.8 = 0.72 cumulative)
#   Distance: 2 hops
```

**How it works:**
- 1 hop: Direct connection strength (e.g., 0.9)
- 2 hops: Multiplied strengths (e.g., 0.9 × 0.8 = 0.72)
- 3 hops: Triple multiplication (e.g., 0.9 × 0.8 × 0.7 = 0.504)

**Use cases:**
- Filter out weak/spurious connections
- Focus on the most relevant related memories
- Improve retrieval quality by ignoring noise

---

#### 15. `get_memories_with_weighted_relevance()`

Get memories sorted by weighted relevance (strength × importance × confidence).

**Parameters:**
- `topic` (str) - Topic to search for
- `include_weak_connections` (bool) - Include weak connections (default: False)
- `limit` (int) - Maximum results (default: 50)

**Returns:** List of memories sorted by weighted relevance

**Example:**

```python
# Get most relevant memories about "project"
memories = await engine.get_memories_with_weighted_relevance(
    "project",
    include_weak_connections=False
)

print("Most relevant project memories:")
for memory in memories[:5]:
    print(f"\n- {memory['content'][:80]}...")
    print(f"  Connection strength: {memory['connection_strength']:.2f}")
    print(f"  Importance: {memory['importance']:.2f}")
    print(f"  Weighted relevance: {memory['weighted_relevance']:.3f}")

# Output:
# Most relevant project memories:
# - Project deadline is next Friday...
#   Connection strength: 0.95
#   Importance: 0.90
#   Weighted relevance: 0.812  (0.95 × 0.90 × 0.95 = 0.812)
# - Python best practices for clean code...
#   Connection strength: 0.80
#   Importance: 0.70
#   Weighted relevance: 0.532
```

**Why it's useful:**
- Prioritizes memories that are BOTH strongly connected AND important
- Filters out weak connections by default
- Natural ranking for context retrieval

---

### Memory Management

#### 16. `reinforce_memory_connection()`

Strengthen a connection between two memories when it's used.

**This mimics human memory:** Frequently accessed connections become stronger over time.

**Parameters:**
- `from_memory_id` (str) - Source memory ID
- `to_memory_id` (str) - Target memory ID
- `strength_boost` (float) - Amount to increase (default: 0.05)

**Returns:** Updated relationship with new strength

**Example:**

```python
# User asks: "What did my father say about my project?"

# Step 1: Retrieve memories
memories = await engine.get_memories_by_topic("father")

# Step 2: Use memories in conversation
response = generate_response_with_memories(memories)

# Step 3: Reinforce the connection that was useful
updated = await engine.reinforce_memory_connection(
    from_memory_id="exp_001",    # Father memory
    to_memory_id="know_123",     # Project advice
    strength_boost=0.05
)

print(f"Connection reinforced:")
print(f"  New strength: {updated['new_strength']:.2f}")
print(f"  Evidence count: {updated['evidence_count']}")

# Output:
# Connection reinforced:
#   New strength: 0.75  (was 0.70, now 0.75)
#   Evidence count: 4    (was 3, now 4)
```

**Over time:**
```
First use:  strength=0.60, evidence=1
After 3 uses: strength=0.75, evidence=4
After 10 uses: strength=1.00, evidence=11  (maxed out)
```

**When to use:**
- After successfully using a memory in conversation
- When a connection proves useful
- To prioritize frequently accessed relationships

**Optional feature:**
You don't have to use this! Memory works fine without reinforcement. But it makes the system more adaptive and human-like.

---

### Utility Methods

#### 17. `get_memory_statistics()`

Get statistics about the memory graph.

**Parameters:** None

**Returns:** Dictionary with node counts and relationship counts

**Example:**

```python
# Get graph statistics
stats = await engine.get_memory_statistics()

print("Memory Graph Statistics:")
print(f"  Timestamp: {stats['timestamp']}")
print(f"\nNode counts:")
for node_type in stats['node_counts']:
    print(f"  {node_type['type']}: {node_type['count']} nodes")

print(f"\nTotal relationships: {stats['total_relationships']}")

# Output:
# Memory Graph Statistics:
#   Timestamp: 2025-11-01T15:30:00
#
# Node counts:
#   ['ExperienceMemory']: 45 nodes
#   ['KnowledgeMemory']: 32 nodes
#   ['RelationshipMemory']: 18 nodes
#
# Total relationships: 127
```

**Use cases:**
- Monitor memory growth over time
- Debug memory storage
- Display statistics in dashboard

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install neo4j pydantic openai numpy
```

### 2. Start Neo4j

```bash
# Using Docker
docker run -d \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/your_password \
    neo4j:5.0

# Or download from https://neo4j.com/download/
```

### 3. Basic Usage

```python
import asyncio
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.processing.memory_retrieval_engine import create_memory_retrieval_engine

async def main():
    # Connect to Neo4j
    client = create_neo4j_client(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="your_password"
    )
    await client.connect()

    # Create retrieval engine
    engine = create_memory_retrieval_engine(client)

    # Retrieve memories
    memories = await engine.get_memories_by_topic("father")

    print(f"Found {len(memories)} memories")
    for memory in memories[:3]:
        print(f"- {memory['content']}")

    # Get statistics
    stats = await engine.get_memory_statistics()
    print(f"\nTotal nodes: {sum(n['count'] for n in stats['node_counts'])}")
    print(f"Total relationships: {stats['total_relationships']}")

    # Cleanup
    await client.disconnect()

# Run
asyncio.run(main())
```

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/your-repo/sofi-memory.git
cd sofi-memory

# Install dependencies
pip install -r requirements.txt

# Configure Neo4j connection
cp config.example.py config.py
# Edit config.py with your Neo4j credentials
```

**requirements.txt:**
```
neo4j>=5.0.0
pydantic>=2.0.0
openai>=1.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
```

---

## ⚙️ Configuration

Create a `config.py` file:

```python
# config.py

# Neo4j Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"

# OpenAI Configuration (for embeddings)
OPENAI_API_KEY = "your_openai_key"
EMBEDDING_MODEL = "text-embedding-3-small"

# Memory Configuration
MAX_MEMORY_AGE_DAYS = 365  # Delete memories older than 1 year
DEFAULT_IMPORTANCE_THRESHOLD = 0.3  # Minimum importance to store
DEFAULT_RETRIEVAL_LIMIT = 50
DEFAULT_MAX_HOPS = 2
```

---

## 💡 Examples

### Example 1: Building Context for Conversation

```python
async def get_conversation_context(user_query: str) -> str:
    """Get relevant context for user query"""

    # Extract topic from query (simplified)
    topic = extract_topic(user_query)

    # Get memories with connected nodes
    results = await engine.get_memories_with_connected_nodes(
        topic,
        max_hops=2,
        limit=10
    )

    # Build context string
    context = []
    for item in results:
        context.append(f"Main: {item['root_content']}")
        for conn in item['connected_memories'][:3]:
            if conn['id']:
                context.append(f"  Related: {conn['content']}")

    return "\n".join(context)


# Usage
user_query = "What did my father say about my career?"
context = await get_conversation_context(user_query)
response = generate_ai_response(user_query, context)
```

### Example 2: Emotion-Based Memory Recall

```python
async def get_emotional_context(emotion: str, topic: str = None):
    """Get emotional memories for empathetic responses"""

    # Get emotional memories
    memories = await engine.get_memories_by_emotion(
        emotion=emotion,
        topic=topic,
        min_intensity=0.6
    )

    # Get emotion summary
    summary = await engine.get_emotions_summary(topic=topic)

    return {
        'emotional_memories': memories,
        'sentiment': summary['avg_emotional_tone'],
        'positive_count': summary['positive_count'],
        'negative_count': summary['negative_count']
    }


# Usage
context = await get_emotional_context("sad", topic="project")
print(f"User sentiment about project: {context['sentiment']:.2f}")
```

### Example 3: Adaptive Learning System

```python
async def learn_from_interaction(
    user_query: str,
    retrieved_memories: List[Dict],
    was_helpful: bool
):
    """Reinforce useful connections"""

    if not was_helpful:
        return

    # Reinforce connections between memories that were used together
    for i, mem1 in enumerate(retrieved_memories):
        for mem2 in retrieved_memories[i+1:]:
            await engine.reinforce_memory_connection(
                mem1['id'],
                mem2['id'],
                strength_boost=0.05
            )

    print(f"Reinforced {len(retrieved_memories)} memory connections")


# Usage
memories = await engine.get_memories_by_topic("career")
response = generate_response(user_query, memories)
user_satisfaction = get_user_feedback()

await learn_from_interaction(user_query, memories, user_satisfaction)
```

---

## 🎯 Best Practices

### 1. Choose the Right Retrieval Method

| Use Case | Method | Why |
|----------|--------|-----|
| Simple keyword search | `get_memories_by_topic()` | Fast, straightforward |
| Rich context needed | `get_memories_with_connected_nodes()` | Associative recall |
| Meaning-based search | `semantic_search()` | Finds similar concepts |
| Multiple topics | `get_memories_by_multiple_topics()` | Efficient batching |
| Empathetic responses | `get_memories_by_emotion()` | Emotional context |
| Quality filtering | `get_strongly_connected_memories()` | Removes noise |

### 2. Optimize Performance

```python
# ✅ Good: Use appropriate limits
memories = await engine.get_memories_by_topic("father", limit=20)

# ❌ Bad: Retrieving too many
memories = await engine.get_memories_by_topic("father", limit=1000)


# ✅ Good: Use max_hops=2 for balanced performance
results = await engine.get_memories_with_connected_nodes(
    "project",
    max_hops=2
)

# ❌ Bad: max_hops=3 can be slow
results = await engine.get_memories_with_connected_nodes(
    "project",
    max_hops=3  # Only use if you need comprehensive context
)
```

### 3. Handle Empty Results

```python
memories = await engine.get_memories_by_topic("unknown_topic")

if not memories:
    print("No memories found. Try a different query.")
    # Fall back to semantic search
    memories = await engine.semantic_search("unknown_topic", top_k=5)
```

### 4. Use Reinforcement Wisely

```python
# ✅ Good: Reinforce after successful use
if user_found_response_helpful:
    await engine.reinforce_memory_connection(mem1_id, mem2_id)

# ❌ Bad: Reinforcing without validation
# Always reinforce blindly - creates noise
```

### 5. Monitor Memory Growth

```python
# Check statistics periodically
stats = await engine.get_memory_statistics()
total_nodes = sum(n['count'] for n in stats['node_counts'])

if total_nodes > 10000:
    print("Warning: Large memory graph. Consider cleanup.")
```

---

## ⚡ Performance

### Query Performance

| Method | Typical Latency | Notes |
|--------|----------------|-------|
| `get_memories_by_topic()` | 10-30ms | Direct node match |
| `get_memories_with_connected_nodes(max_hops=1)` | 20-50ms | One hop traversal |
| `get_memories_with_connected_nodes(max_hops=2)` | 50-200ms | **Recommended** |
| `get_memories_with_connected_nodes(max_hops=3)` | 200-500ms | Use sparingly |
| `semantic_search()` | 100-300ms | Vector similarity |
| `get_strongly_connected_memories()` | 50-150ms | Filtered traversal |

### Optimization Tips

1. **Use indexes** - Neo4j automatically creates indexes on `id` fields
2. **Limit results** - Always set reasonable limits (50-100)
3. **Cache frequently used queries** - Implement application-level caching
4. **Batch operations** - Use `get_memories_by_multiple_topics()` instead of multiple single calls
5. **Connection pooling** - Reuse Neo4j client connections

---

## 🔧 Troubleshooting

### Problem: No memories returned

```python
# Check if memories exist
stats = await engine.get_memory_statistics()
print(f"Total nodes: {sum(n['count'] for n in stats['node_counts'])}")

# Try broader search
memories = await engine.semantic_search(your_query, top_k=10)
```

### Problem: Slow queries

```python
# Reduce max_hops
results = await engine.get_memories_with_connected_nodes(
    topic,
    max_hops=1  # Instead of 2 or 3
)

# Reduce limit
memories = await engine.get_memories_by_topic(topic, limit=20)
```

### Problem: Connection refused

```python
# Check Neo4j is running
docker ps | grep neo4j

# Verify connection
client = create_neo4j_client(uri, username, password)
try:
    await client.connect()
    await client.verify_connectivity()
    print("✅ Connected to Neo4j")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

### Problem: Out of memory errors

```python
# Check memory usage
stats = await engine.get_memory_statistics()
print(f"Total relationships: {stats['total_relationships']}")

# If too large, implement cleanup
# (Future feature: memory decay/cleanup)
```

---

## 📚 Additional Resources

- [Neo4j Documentation](https://neo4j.com/docs/)
- [Cypher Query Language](https://neo4j.com/docs/cypher-manual/)
- [Graph Database Concepts](https://neo4j.com/developer/graph-database/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

**Built with ❤️ by the SOFI Developer [Zafar](https://github.com/zafar-TechWizard)**

*Making AI assistants more human, one memory at a time.*
