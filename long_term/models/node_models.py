"""
Memory Context Models for SOFI Memory System

This module defines Pydantic models for the hybrid memory system using Memory Contexts.
Instead of rigid entity types, memories are organized by context - how they're used and
when they're relevant. This creates a more human-like and flexible memory system.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from uuid import uuid4, UUID

from pydantic import BaseModel, Field, validator, root_validator
import logging # Import logger for validation warnings

logger = logging.getLogger(__name__)


class MemoryContext(str, Enum):
    """Memory contexts - how memories are organized and retrieved"""
    EXPERIENCE = "EXPERIENCE"  # What happened to me + what I learned from it
    KNOWLEDGE = "KNOWLEDGE"    # What I know + how I use it
    RELATIONSHIP = "RELATIONSHIP"  # Who I know + how I feel + how we interact
    # REFACTOR: Removed 'CURRENT' context to solve Gap #5


class BaseMemoryNode(BaseModel):
    """
    Base class for all memory nodes in the knowledge graph.
    
    Provides common fields and validation for all memory contexts.
    """
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the memory")
    memory_context: MemoryContext = Field(description="Context this memory belongs to")
    content: str = Field(description="The actual memory content")
    
    # --- NEW: Add content_vector field ---
    # This field is required for the vector index.
    content_vector: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    
    description: Optional[str] = Field(None, description="Detailed description of the memory")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Importance score (0-1)")
    emotional_significance: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional significance (-1 to 1)")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0, description="Confidence in memory extraction (0-1)")
    created_date: datetime = Field(default_factory=datetime.now, description="When memory was created")
    last_updated: datetime = Field(default_factory=datetime.now, description="When memory was last updated")
    last_accessed: Optional[datetime] = Field(None, description="When memory was last accessed")
    access_count: int = Field(default=0, description="Number of times memory has been accessed")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and search")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Context-specific relevance
    context_relevance: Dict[MemoryContext, float] = Field(
        default_factory=dict, 
        description="How relevant this memory is to each context (0-1)"
    )
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }
        # Allow extra fields (like content_vector) to be pruned if not defined in a child
        extra = "ignore" 
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v.strip()
    
    @root_validator
    def update_timestamp_on_change(cls, values):
        """Update last_updated timestamp when memory is modified"""
        if 'last_updated' in values:
            values['last_updated'] = datetime.now()
        return values


class ExperienceMemoryNode(BaseMemoryNode):
    """
    Experience Context Memory: What happened to me + what I learned from it
    
    Stores specific events, experiences, and the knowledge gained from them.
    This combines episodic memory (what happened) with semantic learning (what I learned).
    """
    memory_context: MemoryContext = Field(default=MemoryContext.EXPERIENCE, description="Memory context")
    
    # Experience Details
    event_type: str = Field(description="Type of experience (meeting, conversation, activity, etc.)")
    timestamp: datetime = Field(description="When the experience occurred")
    participants: List[str] = Field(default_factory=list, description="People involved in the experience")
    location: Optional[str] = Field(None, description="Where the experience took place")
    
    # Learning and Insights
    lessons_learned: List[str] = Field(default_factory=list, description="What was learned from this experience")
    insights_gained: List[str] = Field(default_factory=list, description="Insights or realizations from the experience")
    skills_practiced: List[str] = Field(default_factory=list, description="Skills that were practiced or developed")
    
    # Emotional and Social Context
    emotional_tone: float = Field(default=0.0, ge=-1.0, le=1.0, description="Overall emotional tone of the experience")
    social_significance: float = Field(default=0.5, ge=0.0, le=1.0, description="How socially significant this experience was")
    personal_impact: float = Field(default=0.5, ge=0.0, le=1.0, description="Personal impact of this experience")
    
    # Context Relevance (how relevant this experience is to other contexts)
    knowledge_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to knowledge context")
    relationship_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to relationship context")
    # REFACTOR: Removed 'current_relevance'
    
    @validator('event_type')
    def validate_event_type(cls, v):
        common_types = [
            'meeting', 'conversation', 'activity', 'learning', 'work', 'social',
            'travel', 'celebration', 'problem_solving', 'collaboration', 'other'
        ]
        if v.lower() not in common_types:
            logger.warning(f"Uncommon event type: {v}")
        return v


class KnowledgeMemoryNode(BaseMemoryNode):
    """
    Knowledge Context Memory: What I know + how I use it
    
    Stores general knowledge, facts, concepts, and how they're applied.
    This combines semantic memory (what I know) with procedural knowledge (how I use it).
    """
    memory_context: MemoryContext = Field(default=MemoryContext.KNOWLEDGE, description="Memory context")
    
    # Knowledge Details
    concept: str = Field(description="The main concept or knowledge area")
    definition: str = Field(description="Definition or explanation of the concept")
    category: str = Field(description="Category of knowledge (technology, science, art, etc.)")
    
    # Application and Usage
    how_to_use: List[str] = Field(default_factory=list, description="How this knowledge is applied or used")
    practical_examples: List[str] = Field(default_factory=list, description="Practical examples of this knowledge")
    use_cases: List[str] = Field(default_factory=list, description="When and where this knowledge is useful")
    
    # Understanding and Mastery
    understanding_level: float = Field(default=0.0, ge=0.0, le=1.0, description="How well this knowledge is understood")
    confidence_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this knowledge")
    mastery_level: float = Field(default=0.0, ge=0.0, le=1.0, description="How well this knowledge is mastered")
    
    # Related Knowledge
    prerequisites: List[str] = Field(default_factory=list, description="Knowledge needed to understand this")
    related_concepts: List[str] = Field(default_factory=list, description="Related concepts and knowledge")
    applications: List[str] = Field(default_factory=list, description="Where this knowledge is applied")
    
    # Context Relevance
    experience_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to experience context")
    relationship_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to relationship context")
    # REFACTOR: Removed 'current_relevance'
    
    @validator('category')
    def validate_category(cls, v):
        common_categories = [
            'technology', 'science', 'art', 'music', 'sports', 'cooking',
            'health', 'finance', 'education', 'work', 'social', 'other'
        ]
        if v.lower() not in common_categories:
            logger.warning(f"Uncommon knowledge category: {v}")
        return v


class RelationshipMemoryNode(BaseMemoryNode):
    """
    Relationship Context Memory: Who I know + how I feel + how we interact
    
    Stores information about people, relationships, emotional connections,
    and interaction patterns. This combines social memory with emotional intelligence.
    """
    memory_context: MemoryContext = Field(default=MemoryContext.RELATIONSHIP, description="Memory context")
    
    # Person Details
    person_name: str = Field(description="Name of the person")
    relationship_type: str = Field(description="Type of relationship (friend, family, colleague, etc.)")
    relationship_strength: float = Field(default=0.5, ge=0.0, le=1.0, description="Strength of relationship (0-1)")
    
    # Emotional Connection
    emotional_connection: float = Field(default=0.0, ge=-1.0, le=1.0, description="Emotional connection strength (-1 to 1)")
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0, description="Trust level (0-1)")
    intimacy_level: float = Field(default=0.0, ge=0.0, le=1.0, description="Intimacy level (0-1)")
    
    # Interaction Patterns
    interaction_frequency: float = Field(default=0.0, ge=0.0, le=1.0, description="How often interactions occur")
    interaction_contexts: List[str] = Field(default_factory=list, description="Contexts of interactions (work, social, etc.)")
    communication_style: str = Field(default="casual", description="Preferred communication style")
    
    # Personal Characteristics
    personality_traits: List[str] = Field(default_factory=list, description="Known personality traits")
    interests: List[str] = Field(default_factory=list, description="Known interests and hobbies")
    skills: List[str] = Field(default_factory=list, description="Known skills and abilities")
    
    # Relationship Dynamics
    power_dynamic: Optional[str] = Field(None, description="Power dynamic in the relationship")
    support_patterns: List[str] = Field(default_factory=list, description="How this person provides support")
    conflict_patterns: List[str] = Field(default_factory=list, description="Known conflict patterns")
    
    # Context Relevance
    experience_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to experience context")
    knowledge_relevance: float = Field(default=0.5, ge=0.0, le=1.0, description="How relevant this is to knowledge context")
    # REFACTOR: Removed 'current_relevance'
    
    @validator('relationship_type')
    def validate_relationship_type(cls, v):
        common_types = [
            'friend', 'family', 'colleague', 'acquaintance', 'romantic',
            'mentor', 'student', 'neighbor', 'classmate', 'teammate', 'other'
        ]
        if v.lower() not in common_types:
            logger.warning(f"Unknown relationship type: {v}")
        return v


# REFACTOR: Removed the entire CurrentMemoryNode class
# (This class is redundant per Gap #5. L1's WorkingMemory class serves this role.)