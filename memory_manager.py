"""
SOFi Memory Manager — Central Orchestrator

Wires together the 3-tier cognitive architecture:
  L1: WorkingMemory   — in-memory active state (the whiteboard)
  L2: Neo4j Graph     — long-term persistent memory
  Processing:         — entity extraction, conversation logging, embeddings

Usage:
    manager = MemoryManager()
    await manager.setup()

    # On every incoming message:
    await manager.observe("user", "...")

    # Before LLM generates its response:
    ctx = manager.get_context("user", "...")
    # ctx is a dict → pass to prompt builder

    await manager.shutdown()
"""

import asyncio
import logging
from typing import Dict, Any, Optional

from memory.working_memory.working_mem import WorkingMemory
from memory.working_memory.working_context import WorkingContextManager
from memory.working_memory.workspace_watcher import WorkspaceWatcher
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.memory_router import MemoryRouter
from memory.processing.embedding_utils import EmbeddingUtils
from memory.config import config

from utils.logger import UniversalLogger

logger = UniversalLogger.get_logger("memory_manager")


class MemoryManager:
    """
    Central orchestrator for the SOFi Memory System.

    Public surface:
        await setup()               — boot everything
        await observe(role, text)   — ingest a message (non-blocking)
        get_context(role, text)     — get SOFi's current awareness dict
        await shutdown()            — graceful close
    """

    def __init__(self):
        self.config = config
        self.is_initialized: bool = False

        # Components (set during setup())
        self.docker_manager:   Optional[DockerManager]         = None
        self.l2_client                                         = None
        self.retriever:        Optional[MemoryRetrievalEngine] = None
        self.router:           Optional[MemoryRouter]          = None
        self.working_memory:   Optional[WorkingMemory]         = None
        self.context_manager:  Optional[WorkingContextManager] = None
        self._watcher:         Optional[WorkspaceWatcher]      = None

    # =========================================================================
    # BOOT
    # =========================================================================

    async def setup(self):
        """
        Boot sequence — must be awaited once before any other call.

        Steps:
          0. Start Docker container for Neo4j (idempotent)
          1. Connect Neo4j and create schema (indexes + constraints)
          2. Build embedding utility and retrieval engine
          3. Build working memory, wired to the retrieval engine
        """
        logger.info("=" * 50)
        logger.info("SOFi Memory System — Booting")
        logger.info("=" * 50)

        # 0. Docker / Neo4j health ─────────────────────────────────────────
        self.docker_manager = DockerManager()
        try:
            self.docker_manager.start_docker()
            await self.docker_manager.ensure_connection_async()
        except Exception as e:
            raise RuntimeError(
                f"Neo4j Docker startup failed. "
                f"Ensure Docker Desktop is running. Detail: {e}"
            ) from e

        # 1. Long-Term Memory (L2) ─────────────────────────────────────────
        logger.info("Connecting to Neo4j…")
        self.l2_client = create_neo4j_client(
            uri=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            database=self.config.database,
            # Sized for 15-20 concurrent router queries (Tier1+Tier2+backup)
            # plus headroom for spikes.  Each route() dispatches up to ~8
            # concurrent Cypher queries; pool of 30 covers 3 overlapping messages.
            max_connection_pool_size=30,
        )
        await self.l2_client.connect()

        # Create vector index + uniqueness constraints (idempotent)
        logger.info("Ensuring Neo4j schema (indexes + constraints)…")
        await self.l2_client.create_constraints_and_indexes()

        # 2. Retrieval Engine ──────────────────────────────────────────────
        logger.info("Loading embedding model…")
        embed = EmbeddingUtils()

        self.retriever = MemoryRetrievalEngine(
            neo4j_client=self.l2_client,
            embedding_utils=embed,
        )

        # 3. Memory Router (intent classifier + tiered dispatch) ──────────────
        logger.info("Initialising MemoryRouter…")
        self.router = MemoryRouter(engine=self.retriever)

        # 4. Working Context Manager (single source of truth) ─────────────────
        logger.info("Initialising WorkingContextManager…")
        self.context_manager = WorkingContextManager()

        # 5. Working Memory (L1) ───────────────────────────────────────────────
        loop = asyncio.get_running_loop()
        self.working_memory = WorkingMemory(
            user_id=self.config.user_id,
            retrieval_engine=self.retriever,
            memory_router=self.router,
            context_manager=self.context_manager,
            event_loop=loop,
        )

        # 6. WorkspaceWatcher (proactive notification monitor) ─────────────────
        logger.info("Starting WorkspaceWatcher…")
        self._watcher = WorkspaceWatcher(
            context_manager=self.context_manager,
            proactive_callback=self._on_proactive_notification,
        )
        self._watcher.start()

        self.is_initialized = True
        logger.info("=" * 50)
        logger.info("SOFi Memory System — Ready ✓")
        logger.info("=" * 50)

    # =========================================================================
    # OBSERVE  (input gate)
    # =========================================================================

    async def observe(self, role: str, content: str) -> None:
        """
        Observe a new message.

        Fires reactive_processing in a background thread and returns immediately.
        Also records user activity so WorkspaceWatcher can track conversation gaps.

        Args:
            role:    "user" | "assistant" | "system"
            content: The message text.
        """
        self._require_initialized()

        # Tell the watcher the user is active (resets gap timer)
        if role == "user" and self._watcher:
            self._watcher.record_user_activity()

        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None,
            self.working_memory.reactive_processing,
            role,
            content,
        )
        logger.debug(f"observe() dispatched background task (role={role})")

    # =========================================================================
    # GET CONTEXT  (output gate)
    # =========================================================================

    def get_context(self, role: str = "", message: str = "") -> Dict[str, Any]:
        """
        Return SOFi's current working context snapshot.

        Blocks for at most `context_retrieval_timeout_ms` milliseconds waiting
        for in-flight reactive_processing to complete, then returns the state.

        Returns a serialisable dict built from WorkingContextManager.snapshot().
        """
        self._require_initialized()
        return self.working_memory.get_working_context(role, message)

    def get_full_context(self):
        """
        Return the full WorkingContext dataclass snapshot — all four pillars.
        Use this for rich access (e.g., to read tiered memories, sofi state, workspace).
        """
        self._require_initialized()
        return self.context_manager.snapshot()

    # =========================================================================
    # SHUTDOWN
    # =========================================================================

    async def shutdown(self, stop_docker: bool = False) -> None:
        """Graceful shutdown — flush working memory, stop watcher, close connections."""
        if self._watcher:
            self._watcher.stop()

        if self.working_memory:
            self.working_memory.shutdown()

        if self.l2_client:
            await self.l2_client.disconnect()

        if stop_docker and self.docker_manager:
            logger.info("Stopping Neo4j Docker container…")
            self.docker_manager.stop_docker()

        logger.info("SOFi Memory System — Shut down.")

    # =========================================================================
    # PROACTIVE CALLBACK (stub — replace with real SOFi activation)
    # =========================================================================

    def _on_proactive_notification(self, item) -> None:
        """
        Called by WorkspaceWatcher when a proactive notification fires.
        Override this OR replace via dependency injection to activate SOFi.

        Default behaviour: log the event (safe no-op until the main assistant
        loop wires in a real activation callback).
        """
        logger.info(
            f"[manager] PROACTIVE: '{item.title}' "
            f"(type={item.type.value} source={item.source_agent})"
        )

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "MemoryManager is not initialized. "
                "Call `await manager.setup()` first."
            )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
async def _main():
    manager = MemoryManager()
    try:
        await manager.setup()

        test_msg = "I am building the SOFi memory system with Python and Neo4j."
        await manager.observe("user", test_msg)

        # Give the background thread ~400ms to finish
        await asyncio.sleep(0.4)

        ctx = manager.get_context("user", test_msg)
        print("\n=== WORKING CONTEXT ===")
        import json
        print(json.dumps(ctx, indent=2, default=str))

    finally:
        await manager.shutdown()


if __name__ == "__main__":
    asyncio.run(_main())