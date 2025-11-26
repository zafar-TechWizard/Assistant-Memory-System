import asyncio
import logging
from typing import Dict, Any, Optional

# The Components 
from memory.working_memory.working_mem import WorkingMemory
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.processing.conversationLogger import ConversationLogger
from memory.processing.embedding_utils import EmbeddingUtils
from memory.config import config

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    The Main Interface for the Memory System.
    """

    def __init__(self):
        self.config = config
        self.is_initialized = False

        self.active_entities = {}
        self.current_entities = []
        
        # Components
        self.docker_manager = None  # Docker lifecycle manager
        self.logger = None          # Logs inputs for consolidation
        self.l2_client = None       # Neo4j Connection
        self.retriever = None       # L2 Reader
        self.working_memory = None  # L1 Processor & State Manager

    async def setup(self):
        """
        Boot sequence: Connects DBs, loads models, prepares the system.
        """
        logger.info("BOOT: Starting Memory System Setup...")

        # 0. Setup Docker Manager and ensure Neo4j is running
        logger.info("BOOT: Initializing Docker Manager...")
        self.docker_manager = DockerManager()
        
        try:
            # Start Docker container (idempotent - safe to call multiple times)
            self.docker_manager.start_docker()
            
            # Ensure Neo4j is ready to accept connections
            self.docker_manager.ensure_connection()
            
            logger.info("BOOT: Docker and Neo4j are ready")
        except Exception as e:
            logger.error(f"BOOT: Failed to start Docker/Neo4j: {e}")
            raise RuntimeError(
                f"Failed to start Neo4j Docker container. "
                f"Please ensure Docker Desktop is running. Error: {e}"
            )

        # 1. Setup Logging (For the 'Dreaming' process later)
        self.logger = ConversationLogger(
            user_id=self.config.user_id,
            filepath=str(self.config.conversation_log_path)
        )

        # 2. Setup Long Term Memory (L2) Infrastructure
        # Use SOFI configuration from docker_manager
        self.l2_client = create_neo4j_client(
            uri=SOFI_NEO4J_CONFIG["uri"],
            username=SOFI_NEO4J_CONFIG["username"],
            password=SOFI_NEO4J_CONFIG["password"],
            database=SOFI_NEO4J_CONFIG["database"]
        )
        await self.l2_client.connect()
        
        # 3. Setup Retrieval Engine (The Bridge between L1 and L2)
        # This engine handles the complex Graph+Vector queries
        self.retriever = MemoryRetrievalEngine(
            neo4j_client=self.l2_client,
            embedding_utils=EmbeddingUtils() 
        )

        # 4. Setup Working Memory (The Processor)
        # We inject the retriever so L1 can 'ask' L2 for data
        self.working_memory = WorkingMemory(
            user_id=self.config.user_id,
            retrieval_engine=self.retriever
        )

        self.is_initialized = True
        logger.info("BOOT: Memory System Ready.")

    async def observe(self, role: str, content: str):
        """
        Input Method: Takes a signal, logs it, and processes it into Working Context.
        """
        if not self.is_initialized:
            raise RuntimeError("Memory System not initialized. Run await setup() first.")

        # Step A: Log raw data (for nightly consolidation)
        # We do this first to ensure data safety
        self.logger.log_message(role, content)

        # Step B: Process into Working Memory (in background)
        # This updates the 'Working Context' state (History, Focus, L2 Retrieval)
        # Run in background task without waiting for completion
        message = [{"role": role, "content": content}]

        loop = asyncio.get_running_loop()

        # Fire-and-forget background thread
        asyncio.create_task(
            loop.run_in_executor(
                None,  # None = default ThreadPoolExecutor
                self.working_memory.reactive_processing,
                message
            )
        )


    def get_context(self, role: str, message: str):
        """
        Output Method: Returns the current state of the 'Mind' to the Assistant.
        """
        if not self.is_initialized:
            raise RuntimeError("Memory System not initialized.")

        return self.working_memory.get_working_context(role, message)

    async def shutdown(self, stop_docker: bool = False):
        """
        Graceful shutdown.
        
        Args:
            stop_docker: If True, stops the Docker container. Default False to leave it running.
        """
        if self.l2_client:
            await self.l2_client.disconnect()
        
        if stop_docker and self.docker_manager:
            logger.info("SHUTDOWN: Stopping Docker container...")
            self.docker_manager.stop_docker()
        
        logger.info("SHUTDOWN: Memory System stopped.")



# --- Quick Test Block ---
if __name__ == "__main__":
    async def main():
        manager = MemoryManager()
        try:
            await manager.setup()
            
            # simulate input
            await manager.observe("user", "I am working on the memory refactor.")
            
            # get context
            ctx = manager.get_context("user", "I am working on the memory refactor.")
            print("\n--- WORKING CONTEXT ---")
            print(ctx.to_prompt_header())
            
        finally:
            await manager.shutdown()

    asyncio.run(main())