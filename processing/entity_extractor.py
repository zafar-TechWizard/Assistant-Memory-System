import importlib.util
import re
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Union

try:
    import spacy
    from spacy.matcher import Matcher, PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Matcher = None
    PhraseMatcher = None

# GLiNER: zero-shot NER with custom entity types. Optional dependency.
# Install: pip install gliner
# Catches concept-level entities (project, event, topic, emotion) that spaCy misses.
try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    GLiNER = None

# coreferee: spaCy plugin for co-reference resolution. Optional dependency.
COREFEREE_AVAILABLE = importlib.util.find_spec("coreferee") is not None

from memory.observability import observer


def _ensure_spacy_model(model_name: str) -> None:
    """Download the spaCy model if not already installed."""
    if not SPACY_AVAILABLE:
        return
    if not spacy.util.is_package(model_name):
        observer.info(f"spaCy model '{model_name}' not found — downloading...")
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            observer.info(f"spaCy model '{model_name}' installed")
        else:
            observer.warning(f"spaCy model download failed", model=model_name, stderr=result.stderr[:200])


def _ensure_coreferee_en() -> None:
    """Download coreferee English model data if not already installed."""
    if not COREFEREE_AVAILABLE:
        return
    try:
        import coreferee
        data_dir = Path(coreferee.__file__).parent / "data"
        if not data_dir.exists() or not any(data_dir.glob("en_*")):
            observer.info("coreferee English model not found — downloading...")
            result = subprocess.run(
                [sys.executable, "-m", "coreferee", "install", "en"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                observer.info("coreferee English model installed")
            else:
                observer.warning("coreferee English model download failed", stderr=result.stderr[:200])
    except Exception as e:
        observer.warning("coreferee model check failed", error=str(e))

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

    # Maximum number of (text, format) pairs to cache.
    # Old entries are evicted LRU-style when this limit is reached.
    _CACHE_MAX_SIZE: int = 512
    
    # GLiNER labels for personal-AI conversational text.
    # Broader than spaCy's NER — catches concept-level entities humans actually
    # reference in conversation.
    _GLINER_LABELS = [
        "person", "place", "organization", "project", "event",
        "topic", "emotion", "skill", "tool", "concept", "goal", "relationship",
    ]

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        custom_patterns: Optional[List[Dict[str, Any]]] = None,
        enable_caching: bool = True,
        strict_spacy: bool = False,
        confidence_threshold: float = 0.5,
        use_gliner: bool = True,
        gliner_model: str = "urchade/gliner_medium-v2.1",
        use_coreferee: bool = True,
    ):
        """
        Initialize entity extractor.

        Args:
            spacy_model: spaCy model name (default: en_core_web_sm)
            custom_patterns: List of custom entity patterns
            enable_caching: Enable entity caching for performance
            strict_spacy: Raise error if spaCy unavailable (default: False)
            confidence_threshold: Minimum confidence to include entity
            use_gliner: Try to load GLiNER for richer concept-level extraction
            gliner_model: GLiNER checkpoint to load (medium ~70MB)
            use_coreferee: Try to add coreferee to spaCy pipeline for pronoun resolution
        """
        self.spacy_model = spacy_model
        self.confidence_threshold = confidence_threshold
        self.enable_caching = enable_caching

        # ── Load spaCy ───────────────────────────────────────────────────────
        self.nlp = None
        self.spacy_available = False

        if SPACY_AVAILABLE:
            _ensure_spacy_model(spacy_model)
            try:
                self.nlp = spacy.load(spacy_model)
                self.spacy_available = True
                observer.info("loaded spaCy", model=spacy_model)
            except Exception as e:
                if strict_spacy:
                    raise RuntimeError(
                        f"spaCy model '{spacy_model}' not available. "
                        f"Install with: python -m spacy download {spacy_model}"
                    )
                observer.warning("spaCy load failed after download attempt", error=str(e))
        else:
            if strict_spacy:
                raise RuntimeError("spaCy not installed. Install with: pip install spacy")
            observer.warning("spaCy not installed — using fallback")

        # ── Add coreferee to spaCy pipeline (pronoun resolution) ─────────────
        self.coref_available = False
        if use_coreferee and self.spacy_available and COREFEREE_AVAILABLE:
            _ensure_coreferee_en()
            try:
                if "coreferee" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("coreferee")
                self.coref_available = True
                observer.info("coreferee pipeline enabled")
            except Exception as e:
                observer.warning("coreferee pipeline failed", error=str(e))
        elif use_coreferee and not COREFEREE_AVAILABLE:
            observer.info("coreferee not installed — pronoun resolution disabled")

        # ── Load GLiNER (zero-shot NER with custom labels) ───────────────────
        self.gliner_model = None
        if use_gliner and GLINER_AVAILABLE:
            try:
                self.gliner_model = GLiNER.from_pretrained(gliner_model)
                observer.info("loaded GLiNER", model=gliner_model)
            except Exception as e:
                observer.warning("GLiNER load failed — fallback to spaCy NER", error=str(e))
        elif use_gliner and not GLINER_AVAILABLE:
            observer.info("GLiNER not installed — using spaCy NER only")

        # Initialize matchers if spaCy available
        if self.spacy_available:
            self.matcher = Matcher(self.nlp.vocab)
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            self._setup_custom_patterns(custom_patterns or [])
        else:
            self.matcher = None
            self.phrase_matcher = None
        
        # Entity cache — bounded LRU via OrderedDict
        # Key: f"{text}:{return_format}"  Value: extraction result
        self._cache: OrderedDict = OrderedDict() if enable_caching else None
        
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
        if self._cache is not None and cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        # Extract entities — prefer the unified pipeline when GLiNER or spaCy is
        # available. Falls back to pure heuristics only if neither is loaded.
        if self.gliner_model is not None or self.spacy_available:
            entities = self._extract_combined(text, text)
        else:
            entities = self._extract_with_heuristics(text)
        
        # Filter by confidence threshold
        entities = [
            e for e in entities 
            if e.confidence >= self.confidence_threshold
        ]
        
        # Format output
        result = self._format_output(entities, return_format)
        
        # Cache result (bounded LRU)
        if self._cache is not None:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._CACHE_MAX_SIZE:
                self._cache.popitem(last=False)  # evict least-recently-used
        
        return result
    
    def extract_entities_detailed(self, text: str) -> Dict[str, List[Entity]]:
        """
        Extract entities with full metadata (convenience method).
        """
        return self.extract_entities(text, return_format="detailed")

    def extract_entities_with_context(
        self,
        current_message: str,
        recent_messages: Optional[List[str]] = None,
        return_format: str = "simple",
    ) -> Union[List[str], Dict[str, List[Entity]]]:
        """
        Sliding-window entity extraction.

        Combines recent_messages + current_message so that pronouns and implicit
        references in the current message resolve against entities mentioned in
        recent turns. Three real-world scenarios this fixes:

          Turn N-1: "I had a fight with Sarah yesterday."
          Turn N  : "she keeps doing this."   ← current message has zero entities
                    → sliding window sees "Sarah" from N-1 and surfaces it
                    → coreference (if available) explicitly resolves "she" → Sarah

          Turn N-1: "The deployment failed twice today."
          Turn N  : "I'm stressed about it."
                    → "deployment" is recovered as the topic anchor

          Turn N-1: "Mike from the team didn't show up."
          Turn N  : "he's been unreliable lately."
                    → "Mike" recovered, EXPERIENCE_TO_RELATIONSHIP path activates

        Returns the same shape as extract_entities — drop-in replacement when
        you have recent context available.
        """
        if not current_message or not current_message.strip():
            return [] if return_format == "simple" else {}

        recent_messages = recent_messages or []

        # Cache key includes both current and context — same context returns same result
        cache_key = f"ctx::{return_format}::{'|'.join(recent_messages[-3:])}::{current_message}"
        if self._cache is not None and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Combine for extraction. Newline separation keeps spaCy/GLiNER's sentence
        # boundaries clean — they treat each turn as its own sentence cluster but
        # coreference still spans them.
        combined = "\n".join(recent_messages[-3:] + [current_message])

        entities = self._extract_combined(combined, current_message)
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        result = self._format_output(entities, return_format)

        if self._cache is not None:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            if len(self._cache) > self._CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

        return result

    def _extract_combined(self, combined_text: str, current_message: str) -> List[Entity]:
        """
        Extract entities from combined recent+current text.

        Pipeline:
        1. GLiNER if available (best for concept-level entities)
        2. spaCy NER + noun chunks (named entities + concepts)
        3. coreferee chains expanded into the current message (pronoun → antecedent)

        All sources merge through _deduplicate_entities. Confidence preferences:
          GLiNER (0.80) ≥ phrase_match (0.95) > custom (0.90) > NER (0.85) > coref (0.78) > chunk (0.70)
        """
        entities: List[Entity] = []

        # ── GLiNER (preferred when available) ────────────────────────────────
        if self.gliner_model is not None:
            try:
                glin = self.gliner_model.predict_entities(
                    combined_text, self._GLINER_LABELS, threshold=0.5,
                )
                for g in glin:
                    text = str(g.get("text", "")).strip()
                    if not text or len(text) < 2:
                        continue
                    label_raw = str(g.get("label", "concept"))
                    entities.append(Entity(
                        text=text,
                        normalized_text=text.lower(),
                        type=self._gliner_label_to_type(label_raw),
                        category=self._gliner_label_to_category(label_raw),
                        confidence=float(g.get("score", 0.8)),
                        start_char=int(g.get("start", 0)),
                        end_char=int(g.get("end", 0)),
                        metadata={"source": "gliner", "gliner_label": label_raw},
                    ))
            except Exception as exc:
                observer.warning("gliner inference failed", error=str(exc))

        # ── spaCy NER + custom matchers + noun chunks ─────────────────────────
        if self.spacy_available:
            try:
                spacy_entities = self._extract_with_spacy(combined_text)
                entities.extend(spacy_entities)

                # Coreference: walk every chain and surface the antecedent for
                # any pronoun that lies inside the current_message portion.
                if self.coref_available:
                    coref_entities = self._extract_coreference_referents(
                        combined_text, current_message,
                    )
                    entities.extend(coref_entities)
            except Exception as exc:
                observer.warning("spacy extraction failed", error=str(exc))
        elif self.gliner_model is None:
            # Neither GLiNER nor spaCy — fallback heuristics
            entities.extend(self._extract_with_heuristics(combined_text))

        return self._deduplicate_entities(entities)

    def _extract_coreference_referents(
        self, combined_text: str, current_message: str,
    ) -> List[Entity]:
        """
        Use coreferee to resolve pronouns. For each chain, if any mention falls
        inside the current_message portion of combined_text, surface the head
        (canonical antecedent) as an entity.

        Example chain: ["Sarah" @0..5, "she" @45..48, "her" @80..83]
        If "she" or "her" is inside the current message, we surface "Sarah".
        """
        out: List[Entity] = []
        try:
            doc = self.nlp(combined_text)
        except Exception:
            return out

        chains = getattr(doc._, "coref_chains", None)
        if not chains:
            return out

        # Locate current_message span inside combined_text (last occurrence)
        msg_start = combined_text.rfind(current_message)
        if msg_start < 0:
            msg_start = 0
        msg_end = msg_start + len(current_message)

        for chain in chains:
            # chain.mentions is a list of Mention objects with token_indexes
            # The "head" is the most informative antecedent
            try:
                head_idx = getattr(chain, "most_specific_mention_index", 0)
                mentions = getattr(chain, "mentions", None)
                if mentions is None:
                    # Some coreferee versions expose mentions via indexing on chain
                    try:
                        head_mention = chain[head_idx]
                        mentions = [chain[i] for i in range(len(chain))]
                    except (TypeError, IndexError):
                        continue
                else:
                    head_mention = mentions[head_idx]

                head_tokens = [doc[i] for i in head_mention.token_indexes]
                head_text = " ".join(t.text for t in head_tokens).strip()
                if not head_text or len(head_text) < 2:
                    continue

                # Does any pronoun mention in this chain fall in the current message?
                touches_current = False
                for mention in mentions:
                    for tok_i in mention.token_indexes:
                        tok = doc[tok_i]
                        if msg_start <= tok.idx < msg_end:
                            touches_current = True
                            break
                    if touches_current:
                        break

                if not touches_current:
                    continue

                # Avoid surfacing the head itself if it's just a pronoun
                if head_text.lower() in {"he", "she", "it", "they", "him", "her", "them"}:
                    continue

                out.append(Entity(
                    text=head_text,
                    normalized_text=head_text.lower(),
                    type=EntityType.PERSON,   # coref antecedents in conversation are usually people
                    category=EntityCategory.TANGIBLE,
                    confidence=0.78,
                    start_char=head_tokens[0].idx,
                    end_char=head_tokens[-1].idx + len(head_tokens[-1]),
                    metadata={"source": "coreference", "chain_len": len(mentions)},
                ))
            except Exception:
                continue

        return out

    def _gliner_label_to_type(self, label: str) -> EntityType:
        """Map GLiNER labels to internal EntityType."""
        mapping = {
            "person":        EntityType.PERSON,
            "organization":  EntityType.ORGANIZATION,
            "place":         EntityType.LOCATION,
            "project":       EntityType.EVENT,
            "event":         EntityType.EVENT,
            "topic":         EntityType.CONCEPT,
            "emotion":       EntityType.CONCEPT,
            "skill":         EntityType.CONCEPT,
            "tool":          EntityType.PRODUCT,
            "concept":       EntityType.CONCEPT,
            "goal":          EntityType.CONCEPT,
            "relationship":  EntityType.PERSON,
        }
        return mapping.get(label.lower(), EntityType.CONCEPT)

    def _gliner_label_to_category(self, label: str) -> EntityCategory:
        tangible = {"person", "place", "organization", "tool", "relationship"}
        intangible = {"topic", "emotion", "skill", "concept", "goal", "project", "event"}
        ll = label.lower()
        if ll in tangible:
            return EntityCategory.TANGIBLE
        if ll in intangible:
            return EntityCategory.INTANGIBLE
        return EntityCategory.INTANGIBLE
    
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
            ...     ["Assistant Memory", "Assistant Memory System"],
            ...     pattern_type="phrase"
            ... )
        """
        if not self.spacy_available:
            observer.warning("cannot add patterns without spaCy")
            return

        if pattern_type == "token":
            self.matcher.add(name, [pattern])
        else:  # phrase
            phrases = [self.nlp.make_doc(p) for p in pattern]
            self.phrase_matcher.add(name, phrases)

    def clear_cache(self):
        """Clear entity cache."""
        if self._cache is not None:
            self._cache.clear()
    

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
            except Exception as e:
                observer.error("entity pattern add failed", exception=e, pattern=name)


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


