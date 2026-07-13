"""
Memory Manager — Central Orchestrator

Wires together the 3-tier cognitive architecture:
  L1: WorkingMemory   — in-memory active state (the whiteboard)
  L2: Neo4j Graph     — long-term persistent memory
  Processing:         — entity extraction, conversation logging, embeddings

Usage:
    manager = MemoryManager()   # reads MEMORY_USER_ID, NEO4J_PASSWORD from env
    await manager.setup()

    # On every incoming message:
    await manager.observe("user", "...")

    # Before LLM generates its response:
    ctx = manager.get_context("user", "...")

    await manager.shutdown()

Or with explicit config (useful when env vars are not convenient):

    manager = MemoryManager(
        user_id="alice",
        neo4j_password="secret",
        base_dir="/path/to/data",
        on_proactive_notification=my_callback,
    )

Observability:
    log=True    → diagnostic events written to <base_dir>/BRAIN/data/logs/YYYY-MM-DD.log
    review=True → per-query traces written to <base_dir>/BRAIN/data/reviews/observe/YYYY-MM-DD/
"""

import asyncio
from pathlib import Path
from typing import Callable, Dict, Any, Optional

from memory.working_memory.working_mem import WorkingMemory
from memory.working_memory.working_context import WorkingContextManager
from memory.working_memory.workspace_watcher import WorkspaceWatcher
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.memory_router import MemoryRouter
from memory.processing.embedding_utils import EmbeddingUtils
from memory.observability import observer
from memory.config import MemoryConfig


class MemoryManager:
    """
    Central orchestrator for the memory system.

    Public surface:
        await setup()               — boot everything
        await observe(role, text)   — ingest a message (non-blocking)
        get_context(role, text)     — get current working context dict
        get_full_context()          — get WorkingContext dataclass (all four pillars)
        await shutdown()            — graceful close
    """

    def __init__(
        self,
        log: bool = False,
        review: bool = False,
        user_id: Optional[str] = None,
        base_dir: Optional[str | Path] = None,
        neo4j_password: Optional[str] = None,
        container_name: Optional[str] = None,
        assistant_name: Optional[str] = None,
        on_proactive_notification: Optional[Callable] = None,
    ):
        """
        Args:
            log:        Write diagnostic events to <base_dir>/data/logs/.
            review:     Write per-query pipeline traces to <base_dir>/data/reviews/.
            user_id:    Identity of the memory owner. Overrides MEMORY_USER_ID env var.
            base_dir:   Root data directory. Overrides MEMORY_BASE_DIR env var.
            neo4j_password: Neo4j password. Overrides NEO4J_PASSWORD env var.
            container_name: Docker container name. Overrides NEO4J_CONTAINER_NAME env var.
            assistant_name: Name the assistant calls itself; also the JSON section key
                in working_context.json (e.g. "aria", "nova"). Overrides
                MEMORY_ASSISTANT_NAME env var. Defaults to "assistant".
            on_proactive_notification: Callback invoked when WorkspaceWatcher fires a
                proactive notification. Receives a WorkspaceItem. If None, notifications
                are logged and discarded.
        """
        # Build config — if any override is provided, create a fresh instance
        # that merges those overrides with env vars for the remaining fields;
        # otherwise use the global singleton (backward-compatible path).
        if any(v is not None for v in [user_id, base_dir, neo4j_password, container_name, assistant_name]):
            kwargs: Dict[str, Any] = {}
            if user_id is not None:
                kwargs["user_id"] = user_id
            if base_dir is not None:
                kwargs["base_dir"] = Path(base_dir)
            if neo4j_password is not None:
                kwargs["neo4j_password"] = neo4j_password
            if container_name is not None:
                kwargs["container_name"] = container_name
            if assistant_name is not None:
                kwargs["assistant_name"] = assistant_name
            self.config = MemoryConfig(**kwargs)
            # Update the backing singleton so the lazy proxy (imported as
            # `config` throughout sub-components) returns these values.
            import memory.config as _cfg_mod
            _cfg_mod._config = self.config
        else:
            from memory.config import get_config
            self.config = get_config()

        self.is_initialized: bool = False
        self._proactive_callback: Optional[Callable] = on_proactive_notification

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
            self.docker_manager = DockerManager(config=self.config.to_docker_config())
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
        try:
            results = await asyncio.gather(
                _task_neo4j(),
                _task_embed(),
                _task_reranker(),
                _task_working_memory(),
                return_exceptions=False,
            )
        except Exception:
            # Clean up any partially-initialised resources before re-raising.
            if self.l2_client:
                try:
                    await self.l2_client.disconnect()
                except Exception:
                    pass
            if self.working_memory:
                try:
                    self.working_memory.shutdown()
                except Exception:
                    pass
            raise
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
            proactive_callback=self._dispatch_proactive_notification,
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
        Return the assistant's current working context snapshot (sync variant).

        Blocks for at most `context_retrieval_timeout_ms` milliseconds waiting
        for in-flight reactive_processing to complete, then returns the state.

        Only use from non-async callers. From async code use `get_context_async`
        — otherwise the threading.Event.wait inside freezes the event loop and
        starves the bridged Neo4j calls that reactive_processing scheduled.
        """
        self._require_initialized()
        return self.working_memory.get_working_context(role, message)

    async def get_context_async(
        self, role: str = "", message: str = ""
    ) -> Dict[str, Any]:
        """
        Async-safe variant of get_context. Waits for in-flight observe()
        processing inside a thread executor so the event loop stays free.
        Prefer this over get_context() from any async caller.
        """
        self._require_initialized()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self.working_memory.get_working_context, role, message
        )

    def get_full_context(self):
        """Return the full WorkingContext dataclass snapshot — all four pillars.

        Reads whatever state observe() already populated; does NOT trigger
        retrieval. Always call after observe() / get_context_async() has settled.
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
            self.docker_manager.stop_docker()

        observer.info("MemoryManager.shutdown complete")
        observer.shutdown()

    # =========================================================================
    # PROACTIVE CALLBACK
    # =========================================================================

    def _dispatch_proactive_notification(self, item) -> None:
        """
        Called by WorkspaceWatcher when a proactive notification fires.
        Delegates to the injected callback if one was provided at construction.
        """
        observer.info(
            "proactive_notification",
            title=item.title,
            type=item.type.value,
            source=item.source_agent,
        )
        if self._proactive_callback is not None:
            self._proactive_callback(item)

    def _on_proactive_notification(self, item) -> None:
        """
        Legacy entry point kept for callers that monkey-patch this method
        directly. Prefer passing on_proactive_notification= to __init__.
        """
        self._dispatch_proactive_notification(item)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "MemoryManager is not initialized. "
                "Call `await manager.setup()` first."
            )
