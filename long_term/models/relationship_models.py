"""
Memory Context Relationship Models for SOFI Memory System

This module defines models for relationships between memories in different contexts.
Relationships capture the complex interconnections between experiences, knowledge,
relationships, and current focus that make SOFI's memory truly human-like and contextual.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, validator, root_validator
from .node_models import MemoryContext


class MemoryRelationshipType(str, Enum):
    """Enumeration of relationship types between memory contexts"""
    # Cross-context relationships
    EXPERIENCE_TO_KNOWLEDGE = "EXPERIENCE_TO_KNOWLEDGE"  # Experience led to knowledge
    KNOWLEDGE_TO_EXPERIENCE = "KNOWLEDGE_TO_EXPERIENCE"  # Knowledge applied in experience
    EXPERIENCE_TO_RELATIONSHIP = "EXPERIENCE_TO_RELATIONSHIP"  # Experience involved person
    RELATIONSHIP_TO_EXPERIENCE = "RELATIONSHIP_TO_EXPERIENCE"  # Person was in experience
    KNOWLEDGE_TO_RELATIONSHIP = "KNOWLEDGE_TO_RELATIONSHIP"  # Knowledge about person
    RELATIONSHIP_TO_KNOWLEDGE = "RELATIONSHIP_TO_KNOWLEDGE"  # Person has knowledge
    CURRENT_TO_EXPERIENCE = "CURRENT_TO_EXPERIENCE"  # Current focus on experience
    EXPERIENCE_TO_CURRENT = "EXPERIENCE_TO_CURRENT"  # Experience relevant to current
    CURRENT_TO_KNOWLEDGE = "CURRENT_TO_KNOWLEDGE"  # Current focus on knowledge
    KNOWLEDGE_TO_CURRENT = "KNOWLEDGE_TO_CURRENT"  # Knowledge relevant to current
    CURRENT_TO_RELATIONSHIP = "CURRENT_TO_RELATIONSHIP"  # Current focus on person
    RELATIONSHIP_TO_CURRENT = "RELATIONSHIP_TO_CURRENT"  # Person relevant to current
    
    # Within-context relationships
    EXPERIENCE_CHAIN = "EXPERIENCE_CHAIN"  # One experience led to another
    KNOWLEDGE_HIERARCHY = "KNOWLEDGE_HIERARCHY"  # Knowledge builds on other knowledge
    RELATIONSHIP_NETWORK = "RELATIONSHIP_NETWORK"  # People know each other
    CURRENT_SEQUENCE = "CURRENT_SEQUENCE"  # Current focus transitions
    
    # Temporal relationships
    HAPPENED_BEFORE = "HAPPENED_BEFORE"
    HAPPENED_AFTER = "HAPPENED_AFTER"
    CONCURRENT = "CONCURRENT"
    DURING = "DURING"
    
    # Causal relationships
    CAUSED = "CAUSED"
    RESULTED_IN = "RESULTED_IN"
    INFLUENCED = "INFLUENCED"
    TRIGGERED = "TRIGGERED"
    
    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    OPPOSITE_OF = "OPPOSITE_OF"
    RELATED_TO = "RELATED_TO"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


class MemoryRelationshipCategory(str, Enum):
    """Categories of memory relationships for organization and querying"""
    CROSS_CONTEXT = "CROSS_CONTEXT"  # Relationships between different memory contexts
    WITHIN_CONTEXT = "WITHIN_CONTEXT"  # Relationships within the same memory context
    TEMPORAL = "TEMPORAL"  # Time-based relationships
    CAUSAL = "CAUSAL"  # Cause and effect relationships
    SIMILARITY = "SIMILARITY"  # Similarity and association relationships


class MemoryRelationshipEdge(BaseModel):
    """
    Base model for all relationships between memories in different contexts.
    
    Represents the connections between memories with properties like strength,
    confidence, and contextual information.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the relationship")
    relationship_type: MemoryRelationshipType = Field(description="Type of memory relationship")
    category: MemoryRelationshipCategory = Field(description="Category of memory relationship")
    
    # Memory references
    from_memory_id: UUID = Field(description="ID of the source memory")
    from_memory_context: MemoryContext = Field(description="Context of the source memory")
    to_memory_id: UUID = Field(description="ID of the target memory")
    to_memory_context: MemoryContext = Field(description="Context of the target memory")
    
    # Relationship properties
    strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Strength of the relationship (0-1)")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in the relationship (0-1)")
    bidirectional: bool = Field(default=False, description="Whether the relationship is bidirectional")
    
    # Contextual information
    context_relevance: Dict[MemoryContext, float] = Field(
        default_factory=dict, 
        description="How relevant this relationship is to each memory context (0-1)"
    )
    
    # Temporal information
    created_date: datetime = Field(default_factory=datetime.now, description="When relationship was created")
    last_reinforced: datetime = Field(default_factory=datetime.now, description="When relationship was last reinforced")
    valid_from: Optional[datetime] = Field(None, description="When relationship became valid")
    valid_until: Optional[datetime] = Field(None, description="When relationship expires")
    
    # Evidence and metadata
    context: Optional[str] = Field(None, description="Context in which relationship was established")
    source_conversation: Optional[str] = Field(None, description="Conversation where relationship was identified")
    evidence_count: int = Field(default=1, description="Number of times relationship has been observed")
    last_evidence: Optional[datetime] = Field(None, description="Last time relationship was observed")
    
    # Additional properties
    properties: Dict[str, Any] = Field(default_factory=dict, description="Additional relationship properties")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
    
    @validator('strength')
    def validate_strength(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Strength must be between 0.0 and 1.0')
        return v
    
    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return v
    
    @root_validator
    def update_reinforcement_timestamp(cls, values):
        """Update last_reinforced timestamp when relationship is modified"""
        if 'last_reinforced' in values:
            values['last_reinforced'] = datetime.now()
        return values


# Memory relationship type mappings for easy access
MEMORY_RELATIONSHIP_TYPE_MAPPING = {
    MemoryRelationshipCategory.CROSS_CONTEXT: [
        MemoryRelationshipType.EXPERIENCE_TO_KNOWLEDGE,
        MemoryRelationshipType.KNOWLEDGE_TO_EXPERIENCE,
        MemoryRelationshipType.EXPERIENCE_TO_RELATIONSHIP,
        MemoryRelationshipType.RELATIONSHIP_TO_EXPERIENCE,
        MemoryRelationshipType.KNOWLEDGE_TO_RELATIONSHIP,
        MemoryRelationshipType.RELATIONSHIP_TO_KNOWLEDGE,
        MemoryRelationshipType.CURRENT_TO_EXPERIENCE,
        MemoryRelationshipType.EXPERIENCE_TO_CURRENT,
        MemoryRelationshipType.CURRENT_TO_KNOWLEDGE,
        MemoryRelationshipType.KNOWLEDGE_TO_CURRENT,
        MemoryRelationshipType.CURRENT_TO_RELATIONSHIP,
        MemoryRelationshipType.RELATIONSHIP_TO_CURRENT
    ],
    MemoryRelationshipCategory.WITHIN_CONTEXT: [
        MemoryRelationshipType.EXPERIENCE_CHAIN,
        MemoryRelationshipType.KNOWLEDGE_HIERARCHY,
        MemoryRelationshipType.RELATIONSHIP_NETWORK,
        MemoryRelationshipType.CURRENT_SEQUENCE
    ],
    MemoryRelationshipCategory.TEMPORAL: [
        MemoryRelationshipType.HAPPENED_BEFORE,
        MemoryRelationshipType.HAPPENED_AFTER,
        MemoryRelationshipType.CONCURRENT,
        MemoryRelationshipType.DURING
    ],
    MemoryRelationshipCategory.CAUSAL: [
        MemoryRelationshipType.CAUSED,
        MemoryRelationshipType.RESULTED_IN,
        MemoryRelationshipType.INFLUENCED,
        MemoryRelationshipType.TRIGGERED
    ],
    MemoryRelationshipCategory.SIMILARITY: [
        MemoryRelationshipType.SIMILAR_TO,
        MemoryRelationshipType.OPPOSITE_OF,
        MemoryRelationshipType.RELATED_TO,
        MemoryRelationshipType.ASSOCIATED_WITH
    ]
}

# Alias for backward compatibility
RelationshipTypes = MemoryRelationshipType
