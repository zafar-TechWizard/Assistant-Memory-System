from typing import List, Dict, Any, Set, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import re
from pathlib import Path

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Matcher = None
    PhraseMatcher = None

from utils.logger import UniversalLogger

logger = UniversalLogger.get_logger("entity_extraction")

class EntityType(Enum):
    """Types of entities that can be extracted"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    TANGIBLE = "TANGIBLE"
    INTANGIBLE = "INTANGIBLE"
    CUSTOM = "CUSTOM"


class EntityCategory(Enum):
    """High-level categories for entities"""
    TANGIBLE = "tangible"  # Physical: people, objects, places
    INTANGIBLE = "intangible"  # Abstract: ideas, emotions, concepts
    TEMPORAL = "temporal"  # Time-related: dates, times, durations
    MONETARY = "monetary"  # Money-related: prices, costs, budgets
    CUSTOM = "custom"  # User-defined custom entities


@dataclass
class Entity:
    """
    Represents an extracted entity with full metadata.
    """
    text: str
    normalized_text: str
    type: EntityType
    category: EntityCategory
    confidence: float
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary format"""
        return {
            'text': self.text,
            'normalized_text': self.normalized_text,
            'type': self.type.value,
            'category': self.category.value,
            'confidence': self.confidence,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'metadata': self.metadata
        }
    
    def __hash__(self):
        """Make Entity hashable for set operations"""
        return hash(self.normalized_text)
    
    def __eq__(self, other):
        """Equality based on normalized text"""
        if not isinstance(other, Entity):
            return False
        return self.normalized_text == other.normalized_text


class EntityExtractor:
    """
    Robust entity extraction with multiple strategies.
    """
    
    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
        enable_caching: bool = True,
        strict_spacy: bool = False,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize entity extractor.
        
        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
            custom_patterns: List of custom entity patterns
            enable_caching: Enable entity caching for performance
            strict_spacy: Raise error if spaCy unavailable (default: False)
            confidence_threshold: Minimum confidence to include entity
        """
        self.spacy_model = spacy_model
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching
        
        # Try to load spaCy
        self.nlp = None
        self.spacy_available = False
        
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                self.spacy_available = True
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                if strict_spacy:
                    raise RuntimeError(
                        f"spaCy model '{spacy_model}' not available. "
                        f"Install with: python -m spacy download {spacy_model}"
                    )
                logger.warning(
                    f"spaCy not available, using fallback extraction: {e}"
                )
        else:
            if strict_spacy:
                raise RuntimeError(
                    "spaCy not installed. Install with: pip install spacy"
                )
            logger.warning("⚠ spaCy not installed, using fallback extraction")
        
        # Initialize matchers if spaCy available
        if self.spacy_available:
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            self._setup_custom_patterns(custom_patterns or [])
        else:
            self.matcher = None
            self.phrase_matcher = None
        
        # Entity cache for performance
        self.entity_cache = {} if enable_caching else None
        
        # Tangible/Intangible classification keywords
        self.tangible_indicators = {
            'object', 'device', 'tool', 'machine', 'vehicle', 'car',
            'building', 'house', 'furniture', 'equipment', 'product',
            'laptop', 'phone', 'computer', 'book', 'table', 'chair'
        }
        
        self.intangible_indicators = {
            'idea', 'concept', 'thought', 'feeling', 'emotion', 'belief',
            'theory', 'principle', 'strategy', 'plan', 'philosophy',
            'approach', 'methodology', 'framework', 'mindset', 'attitude'
        }
    
    def extract_entities(
        self,
        text: str,
        return_format: str = "simple"
    ) -> Union[List[str], Dict[str, List[Entity]]]:
        """
        Extract entities from text.
        
        Args:
            text: Input text to extract entities from
            return_format: Output format
                - "simple": List[str] (backward compatible)
                - "detailed": Dict[str, List[Entity]] (grouped by type)
        
        Returns:
            List of entity strings OR detailed entity dictionary
            
        Example:
            >>> extractor.extract_entities("I met Sarah")
            ['sarah']
            
            >>> extractor.extract_entities("I met Sarah", "detailed")
            {'person': [Entity(text='Sarah', ...)]}
        """
        if not text or not text.strip():
            return [] if return_format == "simple" else {}
        
        # Check cache
        cache_key = f"{text}:{return_format}"
        if self.enable_caching and cache_key in self.entity_cache:
            return self.entity_cache[cache_key]
        
        # Extract entities
        if self.spacy_available:
            entities = self._extract_with_spacy(text)
        else:
            entities = self._extract_with_heuristics(text)
        
        # Filter by confidence threshold
        entities = [
            e for e in entities 
            if e.confidence >= self.confidence_threshold
        ]
        
        # Format output
        result = self._format_output(entities, return_format)
        
        # Cache results
        if self.enable_caching:
            self.entity_cache[cache_key] = result
        
        logger.debug(
            f"Extracted {len(entities)} entities from text: '{text[:50]}...'"
        )
        
        return result
    
    def extract_entities_detailed(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract entities with full metadata (convenience method).
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping entity types to Entity objects
        """
        return self.extract_entities(text, return_format="detailed")
    
    def add_custom_pattern(
        self,
        name: str,
        pattern: Union[List[Dict], List[str]],
        pattern_type: str = "phrase",
        entity_type: EntityType = EntityType.CUSTOM,
        entity_category: EntityCategory = EntityCategory.CUSTOM
    ):
        """
        Add custom entity pattern at runtime.
        
        Args:
            name: Pattern identifier
            pattern: Token pattern (list of dicts) or phrase list (list of strings)
            pattern_type: "token" or "phrase"
            entity_type: EntityType for matched entities
            entity_category: EntityCategory for matched entities
            
        Example:
            >>> extractor.add_custom_pattern(
            ...     "project_names",
            ...     ["SOFI", "Assistant Memory System"],
            ...     pattern_type="phrase"
            ... )
        """
        if not self.spacy_available:
            logger.warning("Cannot add patterns without spaCy")
            return
        
        if pattern_type == "token":
            self.matcher.add(name, [pattern])
        else:  # phrase
            phrases = [self.nlp.make_doc(p) for p in pattern]
            self.phrase_matcher.add(name, phrases)
        
        logger.info(f"Added custom pattern: {name} ({pattern_type})")
    
    def clear_cache(self):
        """Clear entity cache"""
        if self.entity_cache is not None:
            self.entity_cache.clear()
            logger.info("Entity cache cleared")
    

    def _extract_with_spacy(self, text: str) -> List[Entity]:
        """
        Extract entities using spaCy NER.
        
        Extracts:
        1. Named entities (PERSON, ORG, LOC, etc.)
        2. Custom pattern matches
        3. Noun chunks as concepts
        """
        doc = self.nlp(text)
        entities = []
        
        # 1. Extract named entities
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                normalized_text=ent.text.lower().strip(),
                type=self._map_spacy_label(ent.label_),
                category=self._classify_category(ent.text, ent.label_),
                confidence=0.85,  # spaCy NER confidence
                start_char=ent.start_char,
                end_char=ent.end_char,
                metadata={
                    'source': 'spacy_ner',
                    'spacy_label': ent.label_,
                    'lemma': ent.lemma_
                }
            )
            entities.append(entity)
        
        # 2. Extract custom pattern matches
        if self.matcher:
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entity = Entity(
                    text=span.text,
                    normalized_text=span.text.lower().strip(),
                    type=EntityType.CUSTOM,
                    category=EntityCategory.CUSTOM,
                    confidence=0.90,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    metadata={
                        'source': 'custom_pattern',
                        'pattern_id': self.nlp.vocab.strings[match_id]
                    }
                )
                entities.append(entity)
        
        # 3. Extract phrase matches
        if self.phrase_matcher:
            matches = self.phrase_matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entity = Entity(
                    text=span.text,
                    normalized_text=span.text.lower().strip(),
                    type=EntityType.CUSTOM,
                    category=EntityCategory.CUSTOM,
                    confidence=0.95,
                    start_char=span.start_char,
                    end_char=span.end_char,
                    metadata={
                        'source': 'phrase_match',
                        'pattern_id': self.nlp.vocab.strings[match_id]
                    }
                )
                entities.append(entity)
        
        # 4. Extract significant noun chunks as concepts
        for chunk in doc.noun_chunks:
            # Only multi-word chunks or important single words
            if len(chunk.text.split()) > 1 or chunk.root.pos_ in ['PROPN']:
                # Skip if already captured as named entity
                if any(e.text.lower() == chunk.text.lower() for e in entities):
                    continue
                
                entity = Entity(
                    text=chunk.text,
                    normalized_text=chunk.text.lower().strip(),
                    type=EntityType.CONCEPT,
                    category=self._classify_category(chunk.text, "CONCEPT"),
                    confidence=0.70,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={
                        'source': 'noun_chunk',
                        'chunk_type': 'noun_phrase',
                        'root': chunk.root.text
                    }
                )
                entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _extract_with_heuristics(self, text: str) -> List[Entity]:
        """
        Fallback heuristic extraction when spaCy unavailable.
        
        Heuristics:
        1. Capitalized words → likely PERSON or proper nouns
        2. Multi-word capitalized phrases → likely PERSON/ORG
        3. Long words (>6 chars) → likely significant concepts
        4. Quoted text → likely important concepts
        """
        entities = []
        
        # Split into words while preserving positions
        words = text.split()
        current_pos = 0
        
        i = 0
        while i < len(words):
            word = words[i]
            clean_word = ''.join(c for c in word if c.isalnum())
            
            if not clean_word:
                i += 1
                continue
            
            # Find word position in original text
            word_start = text.find(word, current_pos)
            word_end = word_start + len(word)
            current_pos = word_end
            
            # Heuristic 1: Multi-word capitalized phrases
            if i < len(words) - 1 and word[0].isupper():
                # Look ahead for more capitalized words
                phrase_words = [word]
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    phrase_words.append(words[j])
                    j += 1
                
                if len(phrase_words) > 1:
                    # Multi-word entity found
                    phrase = ' '.join(phrase_words)
                    phrase_end = text.find(phrase_words[-1], word_start) + len(phrase_words[-1])
                    
                    entity = Entity(
                        text=phrase,
                        normalized_text=phrase.lower().strip(),
                        type=EntityType.PERSON,
                        category=EntityCategory.TANGIBLE,
                        confidence=0.75,
                        start_char=word_start,
                        end_char=phrase_end,
                        metadata={'source': 'heuristic_multi_cap'}
                    )
                    entities.append(entity)
                    i = j
                    continue
            
            # Heuristic 2: Single capitalized word
            if len(clean_word) > 2 and word[0].isupper():
                entity = Entity(
                    text=clean_word,
                    normalized_text=clean_word.lower(),
                    type=EntityType.PERSON,
                    category=EntityCategory.TANGIBLE,
                    confidence=0.65,
                    start_char=word_start,
                    end_char=word_end,
                    metadata={'source': 'heuristic_cap'}
                )
                entities.append(entity)
            
            # Heuristic 3: Long words (likely significant)
            elif len(clean_word) > 6:
                entity = Entity(
                    text=clean_word,
                    normalized_text=clean_word.lower(),
                    type=EntityType.CONCEPT,
                    category=EntityCategory.INTANGIBLE,
                    confidence=0.55,
                    start_char=word_start,
                    end_char=word_end,
                    metadata={'source': 'heuristic_long'}
                )
                entities.append(entity)
            
            i += 1
        
        # Heuristic 4: Quoted text
        quoted_pattern = r'"([^"]+)"|\'([^\']+)\''
        for match in re.finditer(quoted_pattern, text):
            quoted_text = match.group(1) or match.group(2)
            if len(quoted_text) > 3:
                entity = Entity(
                    text=quoted_text,
                    normalized_text=quoted_text.lower().strip(),
                    type=EntityType.CONCEPT,
                    category=EntityCategory.INTANGIBLE,
                    confidence=0.80,
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={'source': 'heuristic_quoted'}
                )
                entities.append(entity)
        
        return self._deduplicate_entities(entities)
    
    def _map_spacy_label(self, label: str) -> EntityType:
        """Map spaCy entity labels to EntityType"""
        mapping = {
            'PERSON': EntityType.PERSON,
            'ORG': EntityType.ORGANIZATION,
            'GPE': EntityType.LOCATION,  # Geo-political entity
            'LOC': EntityType.LOCATION,
            'FAC': EntityType.LOCATION,  # Facility
            'DATE': EntityType.DATE,
            'TIME': EntityType.TIME,
            'MONEY': EntityType.MONEY,
            'PRODUCT': EntityType.PRODUCT,
            'EVENT': EntityType.EVENT,
            'WORK_OF_ART': EntityType.CONCEPT,
            'LAW': EntityType.CONCEPT,
            'LANGUAGE': EntityType.CONCEPT,
        }
        return mapping.get(label, EntityType.CONCEPT)
    
    def _classify_category(self, text: str, label: str) -> EntityCategory:
        """
        Classify entity as tangible/intangible/temporal/monetary.
        
        Rules:
        - PERSON, ORG, LOC, PRODUCT → TANGIBLE
        - DATE, TIME → TEMPORAL
        - MONEY → MONETARY
        - Contains tangible keywords → TANGIBLE
        - Contains intangible keywords → INTANGIBLE
        - Default → INTANGIBLE (for concepts)
        """
        text_lower = text.lower()
        
        # Temporal entities
        if label in ['DATE', 'TIME']:
            return EntityCategory.TEMPORAL
        
        # Monetary entities
        if label == 'MONEY':
            return EntityCategory.MONETARY
        
        # Tangible entities (physical)
        if label in ['PERSON', 'ORG', 'GPE', 'LOC', 'FAC', 'PRODUCT']:
            return EntityCategory.TANGIBLE
        
        # Check for tangible indicators
        if any(ind in text_lower for ind in self.tangible_indicators):
            return EntityCategory.TANGIBLE
        
        # Check for intangible indicators
        if any(ind in text_lower for ind in self.intangible_indicators):
            return EntityCategory.INTANGIBLE
        
        # Default to intangible for abstract concepts
        return EntityCategory.INTANGIBLE
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities, keeping highest confidence.
        
        Deduplication strategy:
        - Entities with same normalized_text are duplicates
        - Keep entity with highest confidence
        - If confidence equal, prefer longer original text
        """
        seen = {}
        for entity in entities:
            key = entity.normalized_text
            
            if key not in seen:
                seen[key] = entity
            else:
                # Keep higher confidence
                if entity.confidence > seen[key].confidence:
                    seen[key] = entity
                # If same confidence, keep longer text
                elif (entity.confidence == seen[key].confidence and 
                      len(entity.text) > len(seen[key].text)):
                    seen[key] = entity
        
        return list(seen.values())
    
    def _format_output(
        self,
        entities: List[Entity],
        return_format: str
    ) -> Union[List[str], Dict[str, List[Entity]]]:
        """
        Format output based on requested format.
        
        Args:
            entities: List of Entity objects
            return_format: "simple" or "detailed"
        
        Returns:
            List of strings OR dictionary grouped by type
        """
        if return_format == "simple":
            # Backward compatible: return list of normalized strings
            return [e.normalized_text for e in entities]
        
        else:  # detailed
            # Group by entity type
            grouped = {}
            for entity in entities:
                type_key = entity.type.value.lower()
                if type_key not in grouped:
                    grouped[type_key] = []
                grouped[type_key].append(entity)
            
            return grouped
    
    def _setup_custom_patterns(self, patterns: List[Dict[str, Any]]):
        """
        Setup custom entity patterns from configuration.
        
        Pattern format:
        {
            'name': 'pattern_name',
            'type': 'token' or 'phrase',
            'pattern': [...] or ['phrase1', 'phrase2']
        }
        """
        for pattern_config in patterns:
            name = pattern_config.get('name')
            pattern_type = pattern_config.get('type', 'phrase')
            pattern = pattern_config.get('pattern', [])
            
            if not name or not pattern:
                continue
            
            try:
                if pattern_type == 'token':
                    self.matcher.add(name, [pattern])
                else:  # phrase
                    phrases = [self.nlp.make_doc(p) for p in pattern]
                    self.phrase_matcher.add(name, phrases)
                
                logger.debug(f"Added custom pattern: {name}")
            except Exception as e:
                logger.error(f"Error adding pattern '{name}': {e}")


def create_entity_extractor(
    config: Optional[Dict[str, Any]] = None
) -> EntityExtractor:
    """
    Factory function to create EntityExtractor from config.
    
    Args:
        config: Configuration dictionary with keys:
            - spacy_model: str
            - custom_patterns: List[Dict]
            - enable_caching: bool
            - strict_spacy: bool
            - confidence_threshold: float
    
    Returns:
        Initialized EntityExtractor instance
        
    Example:
        >>> extractor = create_entity_extractor(config)
    """
    if config is None:
        config = {}
    
    return EntityExtractor(
        spacy_model=config.get('spacy_model', 'en_core_web_sm'),
        custom_patterns=config.get('custom_patterns', []),
        enable_caching=config.get('enable_caching', True),
        strict_spacy=config.get('strict_spacy', False),
        confidence_threshold=config.get('confidence_threshold', 0.5)
    )


