"""
SOFI Memory System - Hybrid Memory Context Architecture

This package implements SOFI's hybrid memory system using Memory Contexts:
- Layer 1: Context Builder & Manager (Working Memory) - Real-time conversation management
- Layer 2: Memory Context System (Hybrid Memory) - Context-based memory organization
- Integration: Unified Memory Manager - Inter-layer communication and SOFI core integration

The system uses Memory Contexts instead of rigid entity types:
- Experience Context: What happened to me + what I learned from it
- Knowledge Context: What I know + how I use it
- Relationship Context: Who I know + how I feel + how we interact
- Current Context: What I'm thinking about now + recent relevant experiences

The system is designed to provide:
- Proactive: Anticipate user needs and initiate relevant conversations
- Affectionate: Remember emotional contexts and personal preferences
- Evolving: Continuously improve understanding through interactions
- Real-time: Provide instant contextual responses during conversations
- Human-like: Organize and retrieve memories using contextual patterns
"""

__version__ = "0.2.0"
__author__ = "SOFI Development Team"

# Core memory system components
from .long_term import (
    Neo4jClient,
    MemoryContext,
    BaseMemoryNode,
    ExperienceMemoryNode,
    KnowledgeMemoryNode,
    RelationshipMemoryNode,
    CurrentMemoryNode,
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
