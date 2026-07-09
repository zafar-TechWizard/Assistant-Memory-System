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
    log=True    → diagnostic events to BRAIN/memory/data/logs/YYYY-MM-DD.log
    review=True → per-query traces to BRAIN/memory/data/reviews/observe/YYYY-MM-DD/
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
            log:    if True, diagnostic events are written to BRAIN/memory/data/logs/.
                    Use for debugging errors and verifying operation.
            review: if True, every observe() call writes a full pipeline trace
                    to BRAIN/memory/data/reviews/observe/. Use for behavioural
                    analysis and improving retrieval quality.
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
          0. Ensure all operational directories exist (BRAIN/memory/data/...)
          1. Run parallel tasks:
             - Neo4j/Docker startup
             - Embedding model load & warmup
             - Reranker model load & warmup
             - WorkingMemory instantiation (EntityExtractor load)
          2. Assemble components and start WorkspaceWatcher
          3. End-to-end warmup
        """
        observer.info("MemoryManager.setup starting")
        import time

        # 0. Ensure operational directories exist before anything else
        dirs = self.config.ensure_directories()
        observer.info("directories ensured", **{k: str(v) for k, v in dirs.items()})

        _loop = asyncio.get_running_loop()

        # Task A: Docker and Neo4j
        async def _task_neo4j():
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
            
            self.l2_client = create_neo4j_client(
                uri=self.config.neo4j_uri,
                username=self.config.neo4j_username,
                password=self.config.neo4j_password,
                database=self.config.database,
                max_connection_pool_size=30,
            )
            await self.l2_client.connect()
            await self.l2_client.create_constraints_and_indexes()
            await self.l2_client.execute_query("RETURN 1")

        # Task B: Embeddings
        async def _task_embed():
            def _load_and_warmup():
                embed = EmbeddingUtils()
                embed.generate_embedding("warmup")
                return embed
            return await _loop.run_in_executor(None, _load_and_warmup)
        
        # Task C: Reranker
        async def _task_reranker():
            from memory.long_term import reranker as _reranker_module
            await _loop.run_in_executor(None, _reranker_module.load_reranker)

        # Task D: WorkingContextManager + WorkingMemory (Instantiates EntityExtractor which takes ~5-10s)
        async def _task_working_memory():
            self.context_manager = WorkingContextManager()
            def _init_working_mem():
                return WorkingMemory(
                    user_id=self.config.user_id,
                    retrieval_engine=None, # wired later
                    memory_router=None,    # wired later
                    context_manager=self.context_manager,
                    event_loop=_loop,
                )
            self.working_memory = await _loop.run_in_executor(None, _init_working_mem)

        # 1. Run all heavy loads concurrently
        _t0 = time.perf_counter()
        results = await asyncio.gather(
            _task_neo4j(),
            _task_embed(),
            _task_reranker(),
            _task_working_memory(),
            return_exceptions=False
        )
        embed = results[1]
        
        observer.info("parallel loading complete", ms=(time.perf_counter() - _t0) * 1000)

        self.retriever = MemoryRetrievalEngine(
            neo4j_client=self.l2_client,
            embedding_utils=embed,
        )
        await self.retriever.ensure_fulltext_index()

        self.router = MemoryRouter(engine=self.retriever)
        
        # Wire up dependencies missed in task_working_memory
        self.working_memory.retrieval_engine = self.retriever
        self.working_memory._router = self.router

        self._watcher = WorkspaceWatcher(
            context_manager=self.context_manager,
            proactive_callback=self._on_proactive_notification,
        )
        self._watcher.start()

        self.is_initialized = True  # set early so observe() doesn't refuse

        # 3. Background the warmups so boot completes instantly
        async def _run_warmups():
            try:
                await self.retriever.warmup()
                self.working_memory._processing_done.clear()
                await _loop.run_in_executor(
                    None,
                    self.working_memory.reactive_processing,
                    "system",
                    "warmup",
                )
            except Exception as exc:
                observer.warning("background warmups failed (non-fatal)", error=str(exc))
        
        # Fire and forget the warmup task
        _loop.create_task(_run_warmups())

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

        # Clear the "processing done" event in the caller's thread BEFORE
        # dispatching to the executor. Otherwise a get_context() racing the
        # executor's scheduling latency returns instantly (event still set
        # from the previous turn / startup), surfacing empty state.
        self.working_memory._processing_done.clear()

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

        SAFE to call from sync code. From async code use `get_context_async`
        instead — otherwise the threading.Event.wait inside freezes the event
        loop and starves the bridged Neo4j calls that reactive_processing
        scheduled, making this method return empty state every time.
        """
        self._require_initialized()
        return self.working_memory.get_working_context(role, message)

    async def get_context_async(
        self, role: str = "", message: str = ""
    ) -> Dict[str, Any]:
        """
        Async-friendly variant of get_context.

        Performs the blocking wait inside a thread executor so the event loop
        stays free to service the bridged coroutines (router.route, Neo4j
        queries) that reactive_processing scheduled via
        asyncio.run_coroutine_threadsafe.

        Use this from async code (brain.py, sofi.py). get_context (sync) is
        kept for callers that genuinely run from a non-async thread.
        """
        self._require_initialized()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.working_memory.get_working_context, role, message
        )

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
