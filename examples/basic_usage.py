"""
Basic usage example for assistant-memory.

Prerequisites:
    pip install "assistant-memory[nlp]"
    python -m spacy download en_core_web_sm
    python -m coreferee install en

    Docker Desktop must be running.
"""
import asyncio
from memory import MemoryManager, WorkingContext


async def main() -> None:
    manager = MemoryManager(
        user_id="alice",
        neo4j_password="your-password",   # or set NEO4J_PASSWORD env var
        assistant_name="aria",
        log=True,                         # writes logs to BRAIN/data/logs/
    )

    print("Booting memory system...")
    await manager.setup()
    print("Ready.\n")

    # --- Conversation turn 1 ---
    await manager.observe("user", "Hi, I'm Alice. I work in Python and prefer short answers.")
    await manager.observe("assistant", "Got it, Alice — concise and Pythonic it is.")

    # --- Conversation turn 2 ---
    await manager.observe("user", "What do you know about me so far?")

    # Wait for in-flight retrieval triggered by observe()
    await manager.get_context_async("user", "What do you know about me so far?")

    # Read the typed four-pillar snapshot
    ctx: WorkingContext = manager.get_full_context()

    print("=== Memory snapshot ===")
    print(f"Must-know memories : {ctx.memory.must_know}")
    print(f"Active entities    : {list(ctx.memory.retrieval_meta.__dict__)}")
    print(f"User emotional state: {ctx.user.current_emotional_state}")
    print(f"Assistant mode     : {ctx.assistant.current_mode}")
    print(f"Time of day        : {ctx.assistant.time_of_day}")

    await manager.shutdown()
    print("\nShutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
