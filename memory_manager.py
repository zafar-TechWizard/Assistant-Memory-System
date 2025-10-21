import asyncio
from typing import Dict, Any, List, Optional
from uuid import uuid4

# Layer 1: The "Conscious Mind"
from sofi_memory.layer1_working_memory.context_manager import ContextManager

# Layer 2: The "Subconscious" Graph Infrastructure and Models
from sofi_memory.long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client

# Processing "Plugs": The bridge between L1 and L2
from sofi_memory.processing.retrieval_engine import RetrievalEngine
from sofi_memory.processing.conversation_logger import ConversationLogger

class UnifiedMemoryManager:
    """
    The main "plug-and-play" API for the AI assistant's memory.

    This class provides a single, simple interface to the entire two-layer memory
    system. It handles real-time context management (L1), long-term memory
    retrieval (L2), and conversation logging for background consolidation.
    """

    def __init__(self, user_id: str, session_id: Optional[str] = None):
        """
        Initializes the memory manager for a specific user and session.

        Args:
            user_id (str): The unique identifier for the user.
            session_id (Optional[str]): The ID for the conversation session. If not
                                        provided, a new one is generated.
        """
        self.user_id = user_id
        self.session_id = session_id or f"session_{uuid4()}"

        # Initialize Layer 1: The real-time "working memory"
        self.l1_manager = ContextManager(self.user_id, self.session_id)

        # Initialize Layer 2 components and the "plugs" that connect to it
        self.l2_client: Neo4jClient = create_neo4j_client()
        self.retrieval_engine = RetrievalEngine(self.l2_client)
        self.logger = ConversationLogger('conversation.json')
        
        # A flag to ensure we connect to the database only once per session
        self._is_db_connected = False
        print(f"UnifiedMemoryManager created for user '{self.user_id}'.")

    async def _ensure_db_connection(self):
        """Connects to the Neo4j database if not already connected."""
        if not self._is_db_connected:
            await self.l2_client.connect()
            self._is_db_connected = True
            print("Database connection established.")

    async def observe(self, role: str, message: str):
        """
        Observes a new message, logs it, and updates the real-time context.
        This is the primary INPUT method for the memory system.

        Args:
            role (str): The role of the speaker ('user' or 'assistant').
            message (str): The content of the message.
        """
        # 1. Log the raw conversation for future consolidation
        self.logger.log_message(self.user_id, role, message)
        
        # 2. Update the fast, real-time Layer 1 context
        self.l1_manager.observe_message(role, message)
        
        # In a real application, you would add a quick NLP step here
        # to update focus and mood from the message content.
        # e.g., self.l1_manager.update_focus(...)

    async def get_context_for_llm(self, query_text: str) -> str:
        """
        Builds the complete context string to be injected into the LLM prompt.
        This is the primary OUTPUT method of the memory system.

        Args:
            query_text (str): The latest message from the user, used for retrieval.

        Returns:
            str: A formatted string containing both short-term (L1) and
                 relevant long-term (L2) memories.
        """
        await self._ensure_db_connection()

        # 1. Get the real-time context from Layer 1
        l1_context_str = self.l1_manager.build_prompt_context()
        
        # 2. Use the retrieval engine to find relevant long-term memories
        retrieved_memories = await self.retrieval_engine.retrieve_memories(query_text)
        
        # 3. Format the retrieved memories into a clean string
        l2_context_str = self._format_retrieved_memories(retrieved_memories)
        
        return f"{l1_context_str}\n{l2_context_str}"

    def _format_retrieved_memories(self, memories: List[Dict[str, Any]]) -> str:
        """Formats the complex retrieval results into a simple string for the LLM."""
        if not memories:
            return "[No relevant long-term memories found]"

        formatted_string = "[Relevant Long-Term Memories]\n"
        for item in memories:
            node = item.get('node', {})
            score = item.get('score', 0.0)
            related = item.get('related_nodes', [])
            
            # Main memory found by vector search
            formatted_string += f"- Primary Memory (Relevance: {score:.2f}): "
            formatted_string += f"[{node.get('memory_context')}] {node.get('content')}\n"
            
            # Related memories found by graph traversal
            if related:
                for rel_node in related:
                    formatted_string += f"  - Linked Memory: [{rel_node.get('memory_context')}] {rel_node.get('content')}\n"
        
        return formatted_string

    async def disconnect(self):
        """Gracefully disconnects from the database."""
        if self._is_db_connected:
            await self.l2_client.disconnect()
            self._is_db_connected = False
            print("Database connection closed.")

# --- Example End-to-End Usage ---
async def main():
    print("\n--- Testing the Unified Memory Manager ---")
    
    # Imagine the AI assistant starts a new conversation with you
    user = "Zafar"
    memory = UnifiedMemoryManager(user_id=user)

    # You send your first message
    message1 = "Feeling a bit stuck on a project. I was talking to Akanksha about it."
    await memory.observe("user", message1)

    # The assistant needs to respond. It calls the memory manager to get context.
    print(f"\n--- Building context for assistant's response to: '{message1}' ---")
    context = await memory.get_context_for_llm(message1)
    
    # This full context string would be sent to the main LLM
    print("\n--- Full Context String for LLM Prompt ---")
    print(context)
    print("------------------------------------------")

    # Based on the context, the LLM generates a response, which is also observed.
    assistant_response = "I remember you mentioned Akanksha is your girlfriend. Talking things through can be really helpful!"
    await memory.observe("assistant", assistant_response)

    # Let's see how the L1 context has changed
    print("\n--- L1 context has now been updated with the assistant's response ---")
    print(memory.l1_manager.build_prompt_context())

    # Clean up the connection
    await memory.disconnect()

if __name__ == '__main__':
    # This runs the example usage function
    asyncio.run(main())
