"""
Assistant Memory — Public Interface

A plug-and-play long-term + working memory module for AI assistants.

Quickstart (env vars):

    # Required env vars:
    #   MEMORY_USER_ID=alice
    #   NEO4J_PASSWORD=<your-password>
    #   MEMORY_BASE_DIR=/path/to/data   (optional, defaults to ~/.memory)

    from memory import MemoryManager

    manager = MemoryManager()
    await manager.setup()

    # Every incoming message:
    await manager.observe("user", text)

    # Before building the assistant's response:
    # observe() triggers retrieval; get_context_async() waits for it to finish.
    await manager.get_context_async("user", text)
    ctx = manager.get_full_context()   # WorkingContext dataclass — all four pillars

    # Graceful close:
    await manager.shutdown()

Or with explicit config (no env vars needed):

    manager = MemoryManager(
        user_id="alice",
        neo4j_password="<your-password>",
        base_dir="/path/to/data",
        assistant_name="aria",          # names ctx.assistant + the JSON section key
        on_proactive_notification=my_callback,
    )

Nothing else in this package is part of the public API.
"""

from memory.memory_manager import MemoryManager
from memory.config import MemoryConfig
from memory.working_memory.working_context import (
    WorkingContext,
    AssistantState,
    UserState,
    MemoryState,
)

__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "WorkingContext",
    "AssistantState",
    "UserState",
    "MemoryState",
]
