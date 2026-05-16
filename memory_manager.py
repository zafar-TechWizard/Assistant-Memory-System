"""
SOFi Memory Manager — Central Orchestrator

Wires together the 3-tier cognitive architecture:
  L1: WorkingMemory   — in-memory active state (the whiteboard)
  L2: Neo4j Graph     — long-term persistent memory
  Processing:         — entity extraction, conversation logging, embeddings

Usage:
    manager = MemoryManager(log=False, review=False)
    await manager.setup()

    # On every incoming message:
    await manager.observe("user", "...")

    # Before LLM generates its response:
    ctx = manager.get_context("user", "...")

    await manager.shutdown()

Observability:
    log=True    → diagnostic events to memory/data/logs/YYYY-MM-DD.log
    review=True → per-query traces to memory/data/reviews/observe/YYYY-MM-DD/
"""

import asyncio
from typing import Dict, Any, Optional

from memory.working_memory.working_mem import WorkingMemory
from memory.working_memory.working_context import WorkingContextManager
from memory.working_memory.workspace_watcher import WorkspaceWatcher
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.memory_router import MemoryRouter
from memory.processing.embedding_utils import EmbeddingUtils
from memory.observability import observer
from memory.config import config


class MemoryManager:
    """
    Central orchestrator for the SOFi Memory System.

    Public surface:
        await setup()               — boot everything
        await observe(role, text)   — ingest a message (non-blocking)
        get_context(role, text)     — get SOFi's current awareness dict
        await shutdown()            — graceful close
    """

    def __init__(self, log: bool = False, review: bool = False):
        """
        Args:
            log:    if True, diagnostic events are written to memory/data/logs/.
                    Use for debugging errors and verifying operation.
            review: if True, every observe() call writes a full pipeline trace
                    to memory/data/reviews/observe/. Use for behavioural analysis
                    and improving retrieval quality.
        """
        self.config = config
        self.is_initialized: bool = False

        observer.configure(log=log, review=review)

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
        observer.info("MemoryManager.setup starting")

        # 0. Docker / Neo4j health
        self.docker_manager = DockerManager()
        try:
            self.docker_manager.start_docker()
            await self.docker_manager.ensure_connection_async()
        except Exception as e:
            observer.error("Neo4j Docker startup failed", exception=e)
            raise RuntimeError(
                f"Neo4j Docker startup failed. "
                f"Ensure Docker Desktop is running. Detail: {e}"
            ) from e

        # 1. Long-Term Memory (L2)
        self.l2_client = create_neo4j_client(
            uri=self.config.neo4j_uri,
            username=self.config.neo4j_username,
            password=self.config.neo4j_password,
            database=self.config.database,
            max_connection_pool_size=30,
        )
        await self.l2_client.connect()
        await self.l2_client.create_constraints_and_indexes()

        # 2. Retrieval Engine
        embed = EmbeddingUtils()

        # Warmup: prime MiniLM + Neo4j pool so the first user message doesn't
        # pay the 400-800ms cold-start penalty.
        _warmup_loop = asyncio.get_running_loop()
        await _warmup_loop.run_in_executor(None, embed.generate_embedding, "warmup")
        await self.l2_client.execute_query("RETURN 1")

        from memory.long_term import reranker as _reranker_module
        await _warmup_loop.run_in_executor(None, _reranker_module.load_reranker)

        self.retriever = MemoryRetrievalEngine(
            neo4j_client=self.l2_client,
            embedding_utils=embed,
        )
        await self.retriever.ensure_fulltext_index()

        # 3. Memory Router
        self.router = MemoryRouter(engine=self.retriever)

        # 4. Working Context Manager (single source of truth)
        self.context_manager = WorkingContextManager()

        # 5. Working Memory (L1)
        loop = asyncio.get_running_loop()
        self.working_memory = WorkingMemory(
            user_id=self.config.user_id,
            retrieval_engine=self.retriever,
            memory_router=self.router,
            context_manager=self.context_manager,
            event_loop=loop,
        )

        # 6. WorkspaceWatcher
        self._watcher = WorkspaceWatcher(
            context_manager=self.context_manager,
            proactive_callback=self._on_proactive_notification,
        )
        self._watcher.start()

        self.is_initialized = True
        observer.info("MemoryManager.setup complete")

    # =========================================================================
    # OBSERVE  (input gate)
    # =========================================================================

    async def observe(self, role: str, content: str) -> None:
        """
        Observe a new message.

        Fires reactive_processing in a background thread and returns immediately.
        Also records user activity so WorkspaceWatcher can track conversation gaps.
        """
        self._require_initialized()

        if role == "user" and self._watcher:
            self._watcher.record_user_activity()

        loop = asyncio.get_running_loop()
        loop.run_in_executor(
            None,
            self.working_memory.reactive_processing,
            role,
            content,
        )

    # =========================================================================
    # GET CONTEXT  (output gate)
    # =========================================================================

    def get_context(self, role: str = "", message: str = "") -> Dict[str, Any]:
        """
        Return SOFi's current working context snapshot.

        Blocks for at most `context_retrieval_timeout_ms` milliseconds waiting
        for in-flight reactive_processing to complete, then returns the state.
        """
        self._require_initialized()
        return self.working_memory.get_working_context(role, message)

    def get_full_context(self):
        """Return the full WorkingContext dataclass snapshot — all four pillars."""
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
            self.docker_manager.stop_docker()

        observer.info("MemoryManager.shutdown complete")
        observer.shutdown()

    # =========================================================================
    # PROACTIVE CALLBACK (stub — replace with real SOFi activation)
    # =========================================================================

    def _on_proactive_notification(self, item) -> None:
        """
        Called by WorkspaceWatcher when a proactive notification fires.
        Override OR replace via dependency injection to activate SOFi.
        """
        observer.info(
            "proactive_notification",
            title=item.title,
            type=item.type.value,
            source=item.source_agent,
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
