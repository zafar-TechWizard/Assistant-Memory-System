# SOFI Memory System - Layer 2 Foundation

## Overview

This is the foundational Layer 2 infrastructure for SOFI's human-like memory system. Layer 2 serves as SOFI's **declarative memory** - the persistent knowledge store that organizes information in interconnected graphs mimicking human hippocampal-neocortical memory architecture.

## What We've Built

### 🏗️ Infrastructure Components

#### 1. **Neo4j Database Client** (`layer2/infrastructure/neo4j_client.py`)
- **Robust connection management** with connection pooling and retry logic
- **Async/await support** for high performance
- **Automatic retry logic** for transient failures
- **Transaction management** with proper error handling
- **Performance monitoring** and query optimization
- **Health checks** and connection validation

#### 2. **Entity Models** (`layer2/models/entity_models.py`)
- **PersonNode**: People mentioned in conversations with personality traits, relationships, and interaction history
- **EventNode**: Events and occurrences with participants, timing, and emotional significance
- **ConceptNode**: Concepts, ideas, and knowledge with learning progress and understanding levels
- **PreferenceNode**: User preferences with strength, context conditions, and evolution tracking
- **LocationNode**: Geographic locations with significance and user relationships

#### 3. **Relationship Models** (`layer2/models/relationship_models.py`)
- **Temporal Relationships**: HAPPENED_BEFORE, HAPPENED_AFTER, CONCURRENT, DURING
- **Causal Relationships**: CAUSED, RESULTED_IN, INFLUENCED, TRIGGERED
- **Social Relationships**: KNOWS, RELATED_TO, WORKS_WITH, FRIENDS_WITH, FAMILY_OF, etc.
- **Semantic Relationships**: SIMILAR_TO, OPPOSITE_OF, PART_OF, CATEGORY_OF, etc.
- **Emotional Relationships**: LIKES, DISLIKES, LOVES, FEARS, EXCITED_ABOUT, etc.
- **Spatial Relationships**: LOCATED_AT, NEAR, INSIDE, ADJACENT_TO, etc.

#### 4. **Configuration Management** (`config.py`)
- **Environment-based configuration** with Pydantic settings
- **Database connection settings** for Neo4j, Redis, and ChromaDB
- **Performance targets** and thresholds
- **Feature flags** for enabling/disabling components
- **Logging configuration**

#### 5. **Setup and Testing** (`setup.py`, `test_layer2_infrastructure.py`)
- **Automated setup** of database schema and constraints
- **Environment validation** and health checks
- **Comprehensive testing** of all components
- **Performance validation** and monitoring

## Architecture Highlights

### 🧠 Human-like Memory Design
- **Graph-based knowledge representation** for complex relationship modeling
- **Entity-relationship structure** mimicking human memory organization
- **Temporal, causal, social, semantic, emotional, and spatial relationships**
- **Confidence scoring** and importance weighting
- **Memory consolidation** and forgetting mechanisms (to be implemented)

### ⚡ Performance Optimized
- **Sub-second response times** for memory retrieval
- **Connection pooling** and async operations
- **Query optimization** with proper indexing
- **Retry logic** and graceful error handling
- **Health monitoring** and performance tracking

### 🔧 Production Ready
- **Comprehensive error handling** and logging
- **Configuration management** with environment variables
- **Health checks** and monitoring
- **Database schema** with constraints and indexes
- **Type safety** with Pydantic models

## File Structure

```
memory/
├── __init__.py                 # Main package initialization
├── config.py                   # Configuration management
├── setup.py                    # Setup and health checks
├── test_layer2_infrastructure.py  # Comprehensive tests
├── README.md                   # This file
└── layer2/                     # Layer 2: Long-Term Knowledge Graph
    ├── __init__.py
    ├── infrastructure/         # Database and infrastructure
    │   ├── __init__.py
    │   └── neo4j_client.py     # Neo4j database client
    └── models/                 # Data models
        ├── __init__.py
        ├── entity_models.py    # Entity node models
        └── relationship_models.py  # Relationship models
```

## Getting Started

### 1. Install Dependencies
```bash
pip install neo4j>=5.15.0
```

### 2. Set Up Neo4j Database
```bash
# Start Neo4j (using Docker)
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15

# Or install locally and start the service
```

### 3. Configure Environment
```bash
# Set environment variables (optional - defaults are provided)
export SOFI_MEMORY_NEO4J_URI="bolt://localhost:7687"
export SOFI_MEMORY_NEO4J_USERNAME="neo4j"
export SOFI_MEMORY_NEO4J_PASSWORD="password"
export SOFI_MEMORY_NEO4J_DATABASE="neo4j"
```

### 4. Run Setup
```bash
cd memory
python setup.py
```

### 5. Run Tests
```bash
python test_layer2_infrastructure.py
```

## Usage Examples

### Creating Entities
```python
from memory.layer2.models.entity_models import PersonNode, EventNode

# Create a person
person = PersonNode(
    name="John Doe",
    relationship_to_user="friend",
    personality_traits=["helpful", "tech-savvy"],
    relationship_strength=0.8
)

# Create an event
event = EventNode(
    name="Team Meeting",
    event_type="meeting",
    participants=["John", "Jane"],
    emotional_valence=0.3
)
```

### Creating Relationships
```python
from memory.layer2.models.relationship_models import SocialRelationship, RelationshipType

# Create a social relationship
relationship = SocialRelationship(
    from_entity_id=person.id,
    from_entity_type="Person",
    to_entity_id=person.id,
    to_entity_type="Person",
    relationship_type=RelationshipType.FRIENDS_WITH,
    strength=0.9
)
```

### Database Operations
```python
from memory.layer2.infrastructure.neo4j_client import create_neo4j_client

# Create client
client = create_neo4j_client()

# Connect and execute queries
await client.connect()
result = await client.execute_query(
    "MATCH (p:Person) RETURN p.name as name LIMIT 5"
)
```

## Performance Targets

- **Layer 2 Single-hop Queries**: <50ms
- **Layer 2 Multi-hop Queries**: <200ms
- **Database Storage**: Support 1M+ entities and 10M+ relationships
- **Update Throughput**: Process 1000+ memory updates per minute
- **Graph Traversal**: Handle 6-degree separation queries efficiently

## Next Steps

This foundational Layer 2 infrastructure is ready for the next phase of development:

1. **Entity Extraction Engine** - LLM-powered entity extraction from conversations
2. **Relationship Inference System** - Graph neural networks for relationship discovery
3. **Memory Consolidation Engine** - Background processing for memory organization
4. **Knowledge Graph Query Engine** - Efficient semantic and associative searches

## Vision Alignment

This implementation aligns perfectly with SOFI's vision of becoming a truly intelligent, emotionally aware companion with human-like memory capabilities:

- ✅ **Proactive**: Foundation for anticipating user needs
- ✅ **Affectionate**: Emotional relationship modeling
- ✅ **Evolving**: Continuous learning through relationship updates
- ✅ **Real-time**: Optimized for instant responses
- ✅ **Human-like**: Graph-based associative memory structure

The foundation is solid, modular, and ready for the next phase of development! 🚀
