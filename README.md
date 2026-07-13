# Assistant Memory

A plug-and-play long-term + working memory module for AI assistants.

Drop it into any LLM-powered agent and get persistent, relationship-aware memory with zero vendor lock-in — everything runs locally.

---

## Architecture

Three cognitive tiers, all wired together automatically:

```
Your Assistant
      │
      ▼
MemoryManager
  ├── L1  WorkingMemory      — in-process active state, updated every turn
  ├── L2  Neo4j Graph        — persistent long-term memory (local Docker)
  └── Processing Pipeline   — entity extraction · embeddings · reranking
```

**Proactive notifications** — a background `WorkspaceWatcher` fires a callback when something is worth surfacing between turns, without polling.

---

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Neo4j runs in a container — no cloud account needed)

---

## Installation

```bash
pip install assistant-memory

# Full NLP stack (entity extraction, coreference resolution, zero-shot NER)
pip install "assistant-memory[nlp]"
python -m spacy download en_core_web_sm
python -m coreferee install en
```

> Model downloads (~150 MB for embeddings + reranker) happen automatically on first run.

---

## Quickstart

```python
import asyncio
from memory import MemoryManager

async def main():
    manager = MemoryManager(
        user_id="alice",
        neo4j_password="your-password",
        assistant_name="aria",          # optional — names the JSON section key
    )
    await manager.setup()

    # Ingest each message turn
    await manager.observe("user", "I prefer concise answers and I work in Python.")
    await manager.observe("assistant", "Got it, I'll keep things short and Pythonic.")

    # Before generating the next response — waits for retrieval to settle
    await manager.get_context_async("user", "What do you remember about me?")
    ctx = manager.get_full_context()

    print(ctx.memory.must_know)     # retrieved memories
    print(ctx.user.emotional_state) # inferred user state
    print(ctx.assistant.time_of_day)

    await manager.shutdown()

asyncio.run(main())
```

See [`examples/`](examples/) for more complete usage examples.

---

## Configuration

All options can be set via constructor args **or** environment variables. Constructor args take precedence.

| Constructor arg | Env var | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | `MEMORY_USER_ID` | ✅ | — | Identity of the memory owner |
| `neo4j_password` | `NEO4J_PASSWORD` | ✅ | — | Neo4j database password |
| `base_dir` | `MEMORY_BASE_DIR` | — | `~/.memory` | Root data directory |
| `container_name` | `NEO4J_CONTAINER_NAME` | — | `memory-neo4j` | Docker container name |
| `assistant_name` | `MEMORY_ASSISTANT_NAME` | — | `assistant` | Names `ctx.assistant` + JSON key |

Copy [`.env.example`](.env.example) to `.env` to get started.

---

## Data layout

All data lives under `<base_dir>/BRAIN/`:

```
~/.memory/BRAIN/
  runtime/session/   — working_context.json, conversation.json
  runtime/tasks/     — task records
  runtime/logs/      — diagnostic logs (when log=True)
  data/neo4j/        — Neo4j data volume (Docker mount)
  data/reviews/      — retrieval traces (when review=True)
```

---

## Public API

```python
manager = MemoryManager(user_id=..., neo4j_password=..., ...)
await manager.setup()

await manager.observe(role, content)          # ingest a message
await manager.get_context_async(role, text)   # wait for retrieval
ctx = manager.get_full_context()              # WorkingContext dataclass
await manager.shutdown()
```

`get_full_context()` returns a `WorkingContext` with four pillars:
- `ctx.memory` — `MemoryState` (retrieved facts, recent turns)
- `ctx.assistant` — `AssistantState` (mode, tone, datetime awareness)
- `ctx.user` — `UserState` (emotional state, focus, sentiment)
- `ctx.workspace` — `AgenticWorkspace` (tasks, notifications)

---

## Memory consolidation

Long-running sessions accumulate conversation turns. Run the consolidation pipeline
periodically to distill them into the Neo4j graph:

```bash
python -m memory.processing.consolidation_runner
```

Requires the [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated.

---

## License

MIT — see [LICENSE](LICENSE).
