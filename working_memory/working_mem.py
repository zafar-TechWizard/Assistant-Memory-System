import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Lock
from contextlib import contextmanager
from context_manager import WorkingContextManager
# from memory.working_memory.context_manager import WorkingContextManager


WORKING_CONTEXT_FILE = Path("data/working_context.json")
ENTITY_EXPIRY_MINUTES = 15
ENTITY_EXPIRY_MS = ENTITY_EXPIRY_MINUTES * 60 * 1000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def current_time_seconds() -> float:
    """Get current time in seconds (Unix timestamp)."""
    return time.time()


def current_time_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)


def extract_entities(text: str) -> List[str]:
    """
    Extract entities from text.
    
    This is a placeholder implementation. In production, replace with:
    - NER (Named Entity Recognition) model
    - spaCy entity extraction
    - Custom entity extraction logic
    - LLM-based entity extraction
    
    Args:
        text: Input text to extract entities from
        
    Returns:
        List of extracted entity names
    """
    words = text.split()
    entities = []
    
    for word in words:
        clean_word = ''.join(c for c in word if c.isalnum())
        
        if len(clean_word) > 4 and (word[0].isupper() or len(clean_word) > 6):
            entities.append(clean_word.lower())
    
    seen = set()
    unique_entities = []
    for entity in entities:
        if entity not in seen:
            seen.add(entity)
            unique_entities.append(entity)
    
    return unique_entities


def fetch_long_term_memory(entity: str) -> List[Dict[str, Any]]:
    """
    Fetch relevant long-term memories for an entity.
    
    This is a placeholder implementation. In production, replace with:
    - Vector database query (Pinecone, Weaviate, Qdrant)
    - Graph database query (Neo4j)
    - RAG (Retrieval Augmented Generation) system
    - Semantic search over memory store
    
    Args:
        entity: Entity name to fetch memories for
        
    Returns:
        List of memory objects related to the entity
    """
    logger.info(f"Fetching long-term memories for entity: {entity}")
    
    return [
        {
            "entity": entity,
            "memory_type": "factual",
            "content": f"Long-term memory about {entity}",
            "relevance_score": 0.85,
            "timestamp": current_time_seconds(),
            "source": "placeholder"
        }
    ]


# ===========================
# Working Memory Class
# ===========================

class WorkingMemory:
    """
    Main Working Memory class implementing reactive processing.
    
    This class manages:
    - Entity extraction from messages
    - Active entity tracking with expiry
    - Long-term memory retrieval
    - Persistent storage of working context
    """
    
    def __init__(self, context_file: Optional[Path] = None):
        """
        Initialize Working Memory.
        
        Args:
            context_file: Path to working_context.json (optional)
        """
        self.context_file = context_file or WORKING_CONTEXT_FILE
        self.context_manager = WorkingContextManager(self.context_file)
        self.entity_expiry_ms = ENTITY_EXPIRY_MS
        
        logger.info(f"Working Memory initialized with file: {self.context_file}")
        logger.info(f"Entity expiry timeout: {ENTITY_EXPIRY_MINUTES} minutes ({self.entity_expiry_ms}ms)")
    
    def reactive_processing(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Main reactive processing method.
        
        Process flow:
        1. Load active entities from working_context.json
        2. Extract entities from messages
        3. Update current_entities in JSON
        4. For existing entities: refresh expiry time to 15 minutes
        5. For new entities: add to active_entities with 15 minute expiry
        6. Fetch long-term memories for new entities
        7. Update memories in working_context.json
        8. Save updated context back to file
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            
        Returns:
            Updated working context dictionary
            
        Example:
            >>> wm = WorkingMemory()
            >>> messages = [
            ...     {"role": "user", "content": "I met Sarah yesterday"},
            ...     {"role": "assistant", "content": "How is Sarah doing?"}
            ... ]
            >>> context = wm.reactive_processing(messages)
        """
        try:
            logger.info(f"Starting reactive processing for {len(messages)} message(s)")
            start_time = time.time()
            
            # Step 1: Load active entities from working_context.json
            working_context = self.context_manager.load()
            active_entities = self._parse_active_entities(working_context.get("active_entities", {}))
            
            logger.debug(f"Loaded {len(active_entities)} active entities")
            
            # Step 2: Extract entities from messages
            current_entities = self._extract_entities_from_messages(messages)
            logger.info(f"Extracted {len(current_entities)} entities from messages: {current_entities}")
            
            # Step 3: Update current_entities in working context
            working_context["current_entities"] = list(current_entities)
            
            # Step 4 & 5: Update active entities
            new_entities = self._update_active_entities(
                active_entities, 
                current_entities
            )
            
            logger.info(f"Found {len(new_entities)} new entities: {new_entities}")
            
            # Step 6: Fetch long-term memories for new entities
            new_memories = self._fetch_memories_for_entities(new_entities)
            logger.info(f"Retrieved {len(new_memories)} new memories")
            
            # Step 7: Update memories in working context
            existing_memories = working_context.get("memories", [])
            working_context["memories"] = self._merge_memories(
                existing_memories, 
                new_memories
            )
            
            working_context["active_entities"] = active_entities
            
            # Step 8: Save updated context
            self.context_manager.save(working_context)
            
            elapsed_time = (time.time() - start_time) * 1000
            logger.info(f"Reactive processing completed in {elapsed_time:.2f}ms")
            
            return working_context
            
        except Exception as e:
            logger.error(f"Error in reactive processing: {e}", exc_info=True)
            raise
    
    def _parse_active_entities(self, active_entities_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Parse active entities from JSON data.
        
        Args:
            active_entities_data: Dictionary of entity data from JSON
                                 Format: {"entity_name": expiry_time_ms}
            
        Returns:
            Dictionary mapping entity names to expiry times (in milliseconds)
        """
        entities = {}
        
        for entity_name, entity_data in active_entities_data.items():
            if isinstance(entity_data, dict):
                # Legacy format with full entity info - extract expiry_time and convert to ms
                expiry_time = entity_data.get("expiry_time", 0)
                # Convert from seconds to milliseconds if needed (legacy was in seconds)
                if expiry_time < 10000000000:  # If less than year 2286 in seconds, it's in seconds
                    expiry_time = int(expiry_time * 1000)
                entities[entity_name] = expiry_time
            else:
                # New simple format (just expiry time in ms)
                entities[entity_name] = entity_data
        
        return entities
    
    def _extract_entities_from_messages(self, messages: List[Dict[str, str]]) -> Set[str]:
        """
        Extract entities from all messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Set of unique entity names
        """
        all_entities = set()
        
        for message in messages:
            content = message.get("content", "")
            if content:
                entities = extract_entities(content)
                all_entities.update(entities)
        
        return all_entities
    
    def _update_active_entities(
        self, 
        active_entities: Dict[str, int], 
        current_entities: Set[str]
    ) -> Set[str]:
        """
        Update active entities with current entities.
        
        For existing entities: refresh expiry time to 15 minutes from now
        For new entities: add with 15 minute expiry
        
        Args:
            active_entities: Current active entities dictionary {entity_name: expiry_time_ms}
            current_entities: Entities extracted from messages
            
        Returns:
            Set of new entity names
        """
        current_time_ms_val = current_time_ms()
        new_expiry_time = current_time_ms_val + self.entity_expiry_ms
        new_entities = set()
        
        for entity in current_entities:
            if entity in active_entities:
                # Existing entity: refresh expiry time to 15 minutes from now
                active_entities[entity] = new_expiry_time
                logger.debug(f"Refreshed entity '{entity}' expiry to {new_expiry_time}")
            else:
                # New entity: add with 15 minute expiry
                active_entities[entity] = new_expiry_time
                new_entities.add(entity)
                logger.debug(f"Added new entity '{entity}' with expiry {new_expiry_time}")
        
        # Clean up expired entities
        self._remove_expired_entities(active_entities)
        
        return new_entities
    
    def _remove_expired_entities(self, active_entities: Dict[str, int]):
        """
        Remove expired entities from active entities.
        
        Args:
            active_entities: Dictionary of active entities {entity_name: expiry_time_ms}
        """
        current_time_ms_val = current_time_ms()
        expired_entities = []
        
        for entity_name, expiry_time in active_entities.items():
            if current_time_ms_val > expiry_time:
                expired_entities.append(entity_name)
        
        for entity_name in expired_entities:
            del active_entities[entity_name]
            logger.info(f"Removed expired entity: {entity_name}")
    
    def _fetch_memories_for_entities(self, entities: Set[str]) -> List[Dict[str, Any]]:
        """
        Fetch long-term memories for a set of entities.
        
        Args:
            entities: Set of entity names
            
        Returns:
            List of memory objects
        """
        all_memories = []
        
        for entity in entities:
            try:
                memories = fetch_long_term_memory(entity)
                all_memories.extend(memories)
            except Exception as e:
                logger.error(f"Error fetching memories for entity '{entity}': {e}")
        
        return all_memories
    
    def _merge_memories(
        self, 
        existing_memories: List[Dict[str, Any]], 
        new_memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge new memories with existing memories, avoiding duplicates.
        
        Args:
            existing_memories: Current memories in working context
            new_memories: Newly fetched memories
            
        Returns:
            Merged list of memories
        """
        # Create a set of existing memory identifiers
        existing_ids = set()
        for memory in existing_memories:
            # Create identifier from entity and content
            identifier = f"{memory.get('entity', '')}:{memory.get('content', '')}"
            existing_ids.add(identifier)
        
        # Add only new memories that don't exist
        merged = existing_memories.copy()
        
        for memory in new_memories:
            identifier = f"{memory.get('entity', '')}:{memory.get('content', '')}"
            if identifier not in existing_ids:
                merged.append(memory)
                existing_ids.add(identifier)
        
        logger.debug(f"Merged memories: {len(existing_memories)} existing + {len(new_memories)} new = {len(merged)} total")
        
        return merged

    def _enough_context(self, active_entities, current_entities):
        for entity in current_entities:
            if entity not in active_entities:
                return False
        return True
    
    
    def get_working_context(self,role: str, content: str) -> Dict[str, Any]:
        """
        Get the current working context with time-budgeted retrieval.
        
        This method attempts to build a complete working context within 500ms.
        If the context is incomplete, it will try to fetch additional memories
        for active entities until the time budget is exhausted.
        
        Returns:
            Current working context dictionary with:
            - active_entities: Dict of entity names to expiry times
            - current_entities: List of recently mentioned entities
            - memories: List of relevant long-term memories
            - notice: Optional message if context is incomplete
            
        Time Budget: 500ms maximum
        """
        formated_messages = [{"role": role, "content": content}]
        start_time = current_time_ms()
        
        current_message_entities = self._extract_entities_from_messages(formated_messages)

        # Step 1: Load current working context
        working_context = self.context_manager.load()
        active_entities = self._parse_active_entities(working_context.get("active_entities", {}))
        
        if self._enough_context(active_entities, current_message_entities):
            logger.info("Context is complete")
            return working_context
        

        # Step 2: Check if we have enough context
        # completeness_score = self._calculate_context_completeness(working_context, active_entities)
        
        # if completeness_score >= 0.8:
        #     logger.info(f"Context is complete (score: {completeness_score:.2f})")
        #     return working_context
        
        # Step 3: Try to improve context within 500ms time budget
        # logger.info(f"Context incomplete (score: {completeness_score:.2f}), attempting to fetch more memories")
        
        
        # Fetch memories for missing entities within time budget
        while (current_time_ms() - start_time) <= 500:
            
            try:
                working_context = self.context_manager.load()
                active_entities = self._parse_active_entities(working_context.get("active_entities", {}))
                
                if self._enough_context(active_entities, current_message_entities):
                    logger.info("Context is complete")
                    return working_context
                
               
                
                # Recalculate completeness
                # completeness_score = self._calculate_context_completeness(working_context, active_entities)
                
                
                
                
            except Exception as e:
                logger.error(f"Error fetching memories for entity '{entity}': {e}")
                continue
        
        # Step 4: Return context with notice if still incomplete
        elapsed_time = current_time_ms() - start_time
        
        # if completeness_score < 0.8:
        #     working_context["notice"] = "I need a bit more time to recall all relevant context."
        #     logger.warning(f"Context still incomplete after {elapsed_time}ms (score: {completeness_score:.2f})")
        # else:
            # Remove any previous notice if context is now complete
            # working_context.pop("notice", None)
            # logger.info(f"Context retrieval completed in {elapsed_time}ms (score: {completeness_score:.2f})")
        
        logger.info(f"Need More {elapsed_time}ms")
        
        return working_context
    
    def _calculate_context_completeness(
        self, 
        working_context: Dict[str, Any], 
        active_entities: Dict[str, int]
    ) -> float:
        """
        Calculate how complete the working context is.
        
        Scoring criteria:
        - Memory coverage: Do we have memories for active entities?
        - Memory count: Do we have sufficient memories overall?
        - Recency: Are memories recent and relevant?
        
        Args:
            working_context: Current working context dictionary
            active_entities: Dictionary of active entities
            
        Returns:
            Completeness score between 0.0 (empty) and 1.0 (complete)
        """
        if not active_entities:
            # No active entities means context is trivially complete
            return 1.0
        
        memories = working_context.get("memories", [])
        
        if not memories:
            # No memories at all
            return 0.0
        
        # Calculate entity coverage (what % of active entities have memories)
        entities_with_memories = set()
        for memory in memories:
            entity = memory.get("entity")
            if entity and entity in active_entities:
                entities_with_memories.add(entity)
        
        entity_coverage = len(entities_with_memories) / len(active_entities) if active_entities else 0.0
        
        # Calculate memory density (how many memories per entity)
        avg_memories_per_entity = len(memories) / len(active_entities) if active_entities else 0.0
        memory_density_score = min(avg_memories_per_entity / 2.0, 1.0)  # Normalize to 0-1, target ~2 memories per entity
        
        # Weighted score: entity coverage is more important than density
        completeness_score = (entity_coverage * 0.7) + (memory_density_score * 0.3)
        
        logger.debug(
            f"Completeness calculation: "
            f"entity_coverage={entity_coverage:.2f}, "
            f"memory_density={memory_density_score:.2f}, "
            f"final_score={completeness_score:.2f}"
        )
        
        return completeness_score
    
    def clear_expired_entities(self):
        """
        Manually trigger cleanup of expired entities.
        """
        working_context = self.context_manager.load()
        active_entities = self._parse_active_entities(working_context.get("active_entities", {}))
        
        self._remove_expired_entities(active_entities)
        
        working_context["active_entities"] = active_entities
        self.context_manager.save(working_context)
        
        logger.info("Expired entities cleanup completed")


# ===========================
# Example Usage
# ===========================

def main():
    """Example usage of the Working Memory system."""
    
    # Initialize working memory
    wm = WorkingMemory()
    
    # Example messages
    messages = [
        {
            "role": "user",
            "content": "I was talking to Sarah about the Python project yesterday"
        },
        {
            "role": "assistant",
            "content": "That sounds interesting! How is the project going?"
        },
        {
            "role": "user",
            "content": "Sarah mentioned that Michael is also working on it"
        }
    ]
    
    # Process messages
    print("\n" + "="*60)
    print("REACTIVE PROCESSING EXAMPLE")
    print("="*60)
    
    context = wm.reactive_processing(messages)
    
    print("\n=== CURRENT ENTITIES ===")
    print(json.dumps(context.get("current_entities", []), indent=2))
    
    print("\n=== ACTIVE ENTITIES ===")
    print(json.dumps(context.get("active_entities", {}), indent=2))
    
    print("\n=== MEMORIES ===")
    print(json.dumps(context.get("memories", [])[:3], indent=2))  # Show first 3
    print(f"... ({len(context.get('memories', []))} total memories)")
    
    # Demonstrate get_working_context
    print("\n" + "="*60)
    print("GET WORKING CONTEXT EXAMPLE")
    print("="*60)
    
    retrieved_context = wm.get_working_context("user", "I was talking to Sarah about the Python project yesterday today dayaftertommotow")
    
    print(f"\n=== CONTEXT RETRIEVAL ===")
    print(f"Active Entities: {len(retrieved_context.get('active_entities', {}))}")
    print(f"Total Memories: {len(retrieved_context.get('memories', []))}")
    
    if "notice" in retrieved_context:
        print(f"Notice: {retrieved_context['notice']}")
    else:
        print("Status: Context is complete")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
