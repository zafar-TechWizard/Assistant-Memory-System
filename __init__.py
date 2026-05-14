"""
SOFi Memory System — Public Interface

External systems (brain, sofi) import ONLY from this package root:

    from memory import MemoryManager

    manager = MemoryManager()
    await manager.setup()

    # Every incoming message:
    await manager.observe("user", text)

    # Before building the prompt:
    context = manager.get_full_context()   # → WorkingContext (all four pillars)

    # Graceful close:
    await manager.shutdown()

Nothing else in this package is part of the public API.
"""

from memory.memory_manager import MemoryManager

__all__ = ["MemoryManager"]
