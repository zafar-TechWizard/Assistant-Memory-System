"""
Layer 2 Memory Context Models

This module contains Pydantic models for the hybrid memory system using Memory Contexts.
Instead of rigid entity types, memories are organized by context - how they're used and
when they're relevant. This creates a more human-like and flexible memory system.

Memory Contexts:
- ExperienceMemoryNode: What happened to me + what I learned from it
- KnowledgeMemoryNode: What I know + how I use it
- RelationshipMemoryNode: Who I know + how I feel + how we interact
- CurrentMemoryNode: What I'm thinking about now + recent relevant experiences

Relationship Types:
- MemoryRelationshipEdge: Base relationship model between memory contexts
- MemoryRelationshipType: Enumeration of memory relationship types
"""

from .node_models import (
    MemoryContext,
    BaseMemoryNode,
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
    CurrentMemoryNode
)

from .relationship_models import (
    MemoryRelationshipType,
    MemoryRelationshipCategory,
    MemoryRelationshipEdge,
    MEMORY_RELATIONSHIP_TYPE_MAPPING
)

__all__ = [
    "MemoryContext",
    "BaseMemoryNode",
    "ExperienceMemoryNode",
    "KnowledgeMemoryNode",
    "RelationshipMemoryNode",
    "CurrentMemoryNode",
    "MemoryRelationshipType",
    "MemoryRelationshipCategory",
    "MemoryRelationshipEdge",
    "MEMORY_RELATIONSHIP_TYPE_MAPPING"
]
