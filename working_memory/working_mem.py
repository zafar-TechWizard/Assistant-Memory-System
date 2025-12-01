import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, Future
import threading
from utils.logger import UniversalLogger

from memory.config import config
from memory.working_memory.context_manager import WorkingContextManager
from memory.processing.conversationLogger import ConversationLogger
from memory.processing.entity_extractor import EntityExtractor

# TODO:
#  - Add proper Memory retrival.
#  - Implement get working context


# Initialize logger
logger = UniversalLogger.get_logger("working_memory")


def current_time_seconds() -> float:
    """Get current time in seconds (Unix timestamp)."""
    return time.time()


def current_time_ms() -> int:
    """Get current time in milliseconds."""
    return int(time.time() * 1000)



def fetch_long_term_memory(entity: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch relevant long-term memories for an entity.
    
    This is a placeholder implementation. In production, replace with:
    - Vector database query (Pinecone, Weaviate, Qdrant)
    - Graph database query (Neo4j)
    - RAG (Retrieval Augmented Generation) system
    - Semantic search over memory store
    
    Args:
        entity: Entity name to fetch memories for
        max_results: Maximum number of memories to return
        
    Returns:
        List of memory objects related to the entity
    """
    logger.info(f"Fetching long-term memories for entity: {entity}")
    
    # Placeholder implementation
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
            context_file: Optional path to working_context
        """
        # Load configuration from config file
        wm_config = config.get("working_memory", {})
        base_path = Path(config.get("base_path", "."))
        
        # File paths
        if context_file:
            self.context_file = context_file
        else:
            self.context_file = base_path / wm_config.get(
                "context_file", 
                "memory/working_memory/data/working_context.json"
            )
        
        # Timing configuration
        self.entity_expiry_minutes = wm_config.get("entity_expiry_minutes", 15)
        self.entity_expiry_ms = self.entity_expiry_minutes * 60 * 1000
        self.context_retrieval_timeout_ms = wm_config.get("context_retrieval_timeout_ms", 500)
        
        # Memory configuration
        self.max_memories_per_entity = wm_config.get("max_memories_per_entity", 5)
        self.enable_auto_cleanup = wm_config.get("enable_auto_cleanup", True)
        
        # Ensure data directory exists
        self.context_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize context manager
        self.context_manager = WorkingContextManager(self.context_file)
        
        # Initialize conversation logger
        self.conversation_logger = ConversationLogger()

        # Initialize entity extractor
        self.entity_extractor = EntityExtractor(strict_spacy=False)
        
        # Thread pool for async operations (logging, entity extraction)
        # Using 2 workers: 1 for logging, 1 for entity extraction
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="WorkingMemory")
        
        logger.info(
            f"Working Memory initialized: context_file={self.context_file}, "
            f"entity_expiry={self.entity_expiry_minutes}min, "
            f"timeout={self.context_retrieval_timeout_ms}ms"
        )

    
    def reactive_processing(self, role: str, content: str) -> Dict[str, Any]:
        """
        Main reactive processing method.        
        Args:
            role: Role of the message sender (user, assistant.)
            content: Content of the message
            
        Returns:
            Updated working context dictionary
            >>> context = wm.reactive_processing(messages)
        """
        try:
            logger.info(f"Starting reactive processing")
            start_time = time.time()

            # Extract entities asynchronously
            entity_future = self._executor.submit(
                self.entity_extractor.extract_entities, content
            )

            # Log message asynchronously (fire-and-forget with error handling)
            logging_future = self._executor.submit(
                self.conversation_logger.log_message, role, content
            )
            
            # Load active entities from working_context.json
            working_context = self.context_manager.load()
            active_entities = self._parse_active_entities(
                working_context.get("active_entities", {})
            )
            
            # Wait max 100ms for entity extraction (should be <10ms normally)
            current_entities = entity_future.result(timeout=0.10)
            logger.debug(f"Entity extraction completed: {len(current_entities)} entities")
            
            # Update current_entities in working context
            working_context["current_entities"] = list(current_entities)
            
            # Update active entities
            new_entities = self._update_active_entities(
                active_entities, 
                current_entities
            )

            # Fetch long-term memories for new entities
            new_memories = self._fetch_memories_for_entities(new_entities)
            logger.info(f"Retrieved {len(new_memories)} new memories")
            
            # Update memories in working context
            existing_memories = working_context.get("memories", [])
            working_context["memories"] = self._merge_memories(
                existing_memories, 
                new_memories
            )
            
            working_context["active_entities"] = active_entities
            
            # Save updated context
            self.context_manager.save(working_context)
            
            return working_context
            
        except Exception as e:
            logger.error(f"Error in reactive processing: {e}", exc_info=True)
            raise
    
    def get_working_context(self, role: str, content: str) -> Dict[str, Any]:
        """
        Get the current working context with time-budgeted retrieval.
        
        This method attempts to build a complete working context within the
        configured timeout (default 500ms). If the context is incomplete,
        it will wait for reactive processing to complete.
        
        Args:
            role: Message role (user/assistant)
            content: Message content
        
        Returns:
            Current working context dictionary with:
            - active_entities: Dict of entity names to expiry times
            - current_entities: List of recently mentioned entities
            - memories: List of relevant long-term memories
            - notice: Optional message if context is incomplete
        """
        start_time = current_time_ms()
        
        # Extract entities from current message
        current_message_entities = set(self.entity_extractor.extract_entities(content))
        
        # Step 1: Load current working context
        working_context = self.context_manager.load()
        active_entities = self._parse_active_entities(
            working_context.get("active_entities", {})
        )
        
        # Check if we have enough context
        if self._has_enough_context(active_entities, current_message_entities):
            logger.info("Context is complete")
            return working_context
        
        # Step 2: Wait for context to be complete within time budget
        timeout_ms = self.context_retrieval_timeout_ms
        
        while (current_time_ms() - start_time) <= timeout_ms:
            try:
                working_context = self.context_manager.load()
                active_entities = self._parse_active_entities(
                    working_context.get("active_entities", {})
                )
                
                if self._has_enough_context(active_entities, current_message_entities):
                    logger.info("Context is complete")
                    return working_context
                
                # Small sleep to avoid busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error checking context: {e}")
                break
        
        # Return context even if incomplete
        elapsed_time = current_time_ms() - start_time
        logger.info(f"Context retrieval completed in {elapsed_time}ms")
        
        return working_context
    
    
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
                # Legacy format with full entity info
                expiry_time = entity_data.get("expiry_time", 0)
                # Convert from seconds to milliseconds if needed
                if expiry_time < 10000000000:  # Year 2286 in seconds
                    expiry_time = int(expiry_time * 1000)
                entities[entity_name] = expiry_time
            else:
                # New simple format (just expiry time in ms)
                entities[entity_name] = entity_data
        
        return entities
    
    
    
    def _update_active_entities(
        self, 
        active_entities: Dict[str, int], 
        current_entities: Set[str]
    ) -> Set[str]:
        """
        Update active entities with current entities.
        
        For existing entities: refresh expiry time
        For new entities: add with expiry
        
        Args:
            active_entities: Current active entities {entity_name: expiry_time_ms}
            current_entities: Entities extracted from messages
            
        Returns:
            Set of new entity names
        """
        current_time_ms_val = current_time_ms()
        new_expiry_time = current_time_ms_val + self.entity_expiry_ms
        new_entities = set()
        
        for entity in current_entities:
            if entity in active_entities:
                # Existing entity: refresh expiry time
                active_entities[entity] = new_expiry_time
                logger.debug(f"Refreshed entity '{entity}' expiry to {new_expiry_time}")
            else:
                # New entity: add with expiry
                active_entities[entity] = new_expiry_time
                new_entities.add(entity)
                logger.debug(f"Added new entity '{entity}' with expiry {new_expiry_time}")
        
        # Clean up expired entities if enabled
        if self.enable_auto_cleanup:
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
    
    def _has_enough_context(
        self, 
        active_entities: Dict[str, int], 
        current_entities: Set[str]
    ) -> bool:
        """
        Check if we have enough context for current entities.
        
        Args:
            active_entities: Currently active entities
            current_entities: Entities from current message
            
        Returns:
            True if all current entities are in active entities
        """
        for entity in current_entities:
            if entity not in active_entities:
                return False
        return True
    
    # ===========================
    # Memory Management Methods
    # ===========================
    
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
                memories = fetch_long_term_memory(
                    entity, 
                    max_results=self.max_memories_per_entity
                )
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
            identifier = f"{memory.get('entity', '')}:{memory.get('content', '')}"
            existing_ids.add(identifier)
        
        # Add only new memories that don't exist
        merged = existing_memories.copy()
        
        for memory in new_memories:
            identifier = f"{memory.get('entity', '')}:{memory.get('content', '')}"
            if identifier not in existing_ids:
                merged.append(memory)
                existing_ids.add(identifier)
        
        logger.debug(
            f"Merged memories: {len(existing_memories)} existing + "
            f"{len(new_memories)} new = {len(merged)} total"
        )
        
        return merged
    
    def __del__(self):
        """Cleanup thread pool when WorkingMemory is destroyed."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True, cancel_futures=False)
                logger.debug("Thread pool executor shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")


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
    print(json.dumps(context.get("memories", [])[:3], indent=2))
    print(f"... ({len(context.get('memories', []))} total memories)")
    
    # Demonstrate get_working_context
    print("\n" + "="*60)
    print("GET WORKING CONTEXT EXAMPLE")
    print("="*60)
    
    retrieved_context = wm.get_working_context(
        "user", 
        "I was talking to Sarah about the Python project"
    )
    
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
