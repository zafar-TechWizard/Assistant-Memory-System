"""
Layer 2: Memory Context System (Hybrid Memory Architecture)

This layer serves as SOFI's hybrid memory system using Memory Contexts instead of rigid
entity types. Memories are organized by context - how they're used and when they're
relevant - creating a more human-like and flexible memory system.

Key Components:
- Memory Context Models: Experience, Knowledge, Relationship, and Current contexts
- Cross-Context Relationships: Connections between different memory contexts
- Context-Aware Query Engine: Efficient retrieval based on memory context
- Memory Consolidation Engine: Background processing for context organization
"""

from .infrastructure.neo4j_client import Neo4jClient
from .models.node_models import (
    MemoryContext,
    BaseMemoryNode,
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
    CurrentMemoryNode
)
from .models.relationship_models import (
    MemoryRelationshipType,
    MemoryRelationshipCategory,
    MemoryRelationshipEdge
)

__all__ = [
    "Neo4jClient",
    "MemoryContext",
    "BaseMemoryNode",
    "ExperienceMemoryNode",
    "KnowledgeMemoryNode",
    "RelationshipMemoryNode",
    "CurrentMemoryNode",
    "MemoryRelationshipType",
    "MemoryRelationshipCategory",
    "MemoryRelationshipEdge"
]
