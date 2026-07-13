<div align="center">

# Assistant Memory

**Production-grade long-term + working memory for AI assistants — fully local, zero vendor lock-in.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Neo4j](https://img.shields.io/badge/graph-Neo4j-blue)](https://neo4j.com)
[![Docker](https://img.shields.io/badge/requires-Docker-2496ed)](https://www.docker.com/products/docker-desktop/)

<br/>

*Give your AI assistant the memory of a person — not a search engine.*

</div>

---

## The Problem

Every LLM application today solves memory the same way: dump recent messages in the context window and run a similarity search when it fills up. This breaks in predictable ways:

- The assistant forgets things the moment the window rolls over
- It retrieves disconnected fragments with no sense of *how* memories relate
- It has no concept of *you* — your patterns, emotional state, or current focus
- Memories accumulate forever with no reasoning about what to keep or discard
- Everything depends on an external API you don't control

**Vector search is not memory. Association is.**

---

## What This Is

`assistant-memory` is a modular, plug-and-play Python library that gives any LLM assistant a three-tier cognitive memory system — inspired by how human memory actually works.

```
Your Assistant
      │
      ▼
MemoryManager
  ├── L1  WorkingMemory      — live, per-turn context updated in real time
  ├── L2  Neo4j Graph        — persistent knowledge graph (runs locally in Docker)
  └── Processing Pipeline   — NLP extraction · embeddings · reranker · consolidation
```

Drop it into your agent with five lines. Everything else is automatic.

---

## What Makes It Different

### Graph memory, not flat vectors

Memories are stored as typed nodes in a knowledge graph. Relationships carry weights and semantic types (`CAUSED`, `TRIGGERED`, `KNOWLEDGE_HIERARCHY`, `EXPERIENCE_CHAIN`, …). When you retrieve, the engine traverses the graph using **spreading activation** — the same mechanism the human brain uses to recall related ideas — not just cosine similarity.

```
user mentions "project deadline"
        │
        ▼ EXPERIENCE_CHAIN (0.80)
"last sprint retrospective" ─── CAUSED (0.90) ──► "team burned out"
        │
        └── KNOWLEDGE_HIERARCHY (0.70) ──► "scrum practices"
```

### Intent-aware retrieval

The router classifies every query by intent (`entity`, `factual`, `emotional`, `temporal`) and traverses different edge types with different conductivity weights for each intent. An emotional query follows `INFLUENCED → TRIGGERED → CAUSED` paths. A temporal query follows `HAPPENED_BEFORE → CONCURRENT` paths. The retrieval is shaped by *what you're asking*, not just *what words you used*.

### Four-pillar Working Context

Every turn, the assistant receives a typed `WorkingContext` — not a raw text blob:

| Pillar | What it holds |
|---|---|
| `ctx.memory` | Retrieved facts, must-know summaries, recent turns, retrieval confidence |
| `ctx.assistant` | Current mode, emotional tone, datetime awareness, recent commitments |
| `ctx.user` | Emotional state, current focus, mentioned entities, sentiment trajectory |
| `ctx.workspace` | Active tasks, pending notifications, reminders, sub-agent results |

### Agentic memory consolidation

Long-running sessions don't rot. A consolidation pipeline uses an LLM agent (Gemini CLI) to *reason* over the conversation and existing memories, producing a structured plan: **CREATE** new nodes, **UPDATE** existing ones, **ENHANCE** with new relationships, **CONTRADICT** superseded beliefs (never deleted — temporal lineage preserved). Python executes the plan deterministically against Neo4j.

### Proactive notifications

A background `WorkspaceWatcher` monitors the workspace between turns. It fires your callback when something is worth surfacing — a completed background task, a due reminder, a stale commitment — without polling, with priority-aware gap detection.

### Fully local

| Component | How it runs |
|---|---|
| Neo4j graph | Local Docker container — you own the data |
| Embeddings | `all-MiniLM-L6-v2` — runs on CPU, no GPU needed |
| Reranker | `ms-marco-MiniLM-L-6-v2` — local cross-encoder |
| NLP pipeline | spaCy + GLiNER + coreferee — installed once, runs forever |
| Consolidation | Gemini CLI — the only optional external call |

No OpenAI. No Pinecone. No monthly bill. No data leaving your machine.

---

## Installation

```bash
# Core package
pip install assistant-memory

# Full NLP stack — entity extraction, coreference resolution, zero-shot NER
pip install "assistant-memory[nlp]"
```

> **Models download automatically.** spaCy (`en_core_web_sm`, ~12 MB), coreferee English data, and the embedding/reranker models (~150 MB total) are fetched on first run — no manual step required.

**Prerequisites:**
- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — Neo4j runs as a local container

---

## Quickstart

```python
import asyncio
from memory import MemoryManager

async def main():
    manager = MemoryManager(
        user_id="alice",
        neo4j_password="your-password",
    )
    await manager.setup()  # boots Neo4j, loads models — ~8s on first run

    # Every message turn
    await manager.observe("user", "I'm building a CLI tool in Python. Keep answers short.")
    await manager.observe("assistant", "Got it — concise and Pythonic.")

    await manager.observe("user", "What do you remember about me?")

    # Wait for in-flight retrieval, then read the snapshot
    await manager.get_context_async("user", "What do you remember about me?")
    ctx = manager.get_full_context()

    print(ctx.memory.must_know)          # ["prefers short answers", "building Python CLI"]
    print(ctx.user.emotional_state)      # "focused"
    print(ctx.assistant.time_of_day)     # "afternoon"

    await manager.shutdown()

asyncio.run(main())
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a complete annotated example.

---

## Configuration

All options are available as constructor args **or** environment variables. Constructor args take precedence.

| Arg | Env var | Required | Default | Description |
|---|---|---|---|---|
| `user_id` | `MEMORY_USER_ID` | ✅ | — | Identity of the memory owner |
| `neo4j_password` | `NEO4J_PASSWORD` | ✅ | — | Neo4j database password |
| `base_dir` | `MEMORY_BASE_DIR` | | `~/.memory` | Root data directory |
| `container_name` | `NEO4J_CONTAINER_NAME` | | `memory-neo4j` | Docker container name |
| `assistant_name` | `MEMORY_ASSISTANT_NAME` | | `assistant` | Names `ctx.assistant` and the JSON section key |
| `log` | | | `False` | Write diagnostic events to `BRAIN/data/logs/` |
| `review` | | | `False` | Write per-query retrieval traces to `BRAIN/data/reviews/` |

Copy [`.env.example`](.env.example) to `.env` to get started with environment variables.

```python
# Zero-config if env vars are set
manager = MemoryManager()

# Explicit — good for testing, multi-user, or when you want no env vars
manager = MemoryManager(
    user_id="alice",
    neo4j_password="s3cr3t",
    base_dir="/data/alice",
    assistant_name="aria",
    on_proactive_notification=my_callback,
)
```

---

## Architecture Deep Dive

### L1 — Working Memory

An in-process, turn-by-turn active state. Every call to `observe()` fires `reactive_processing` in a background thread — it runs the NLP pipeline, updates entity state, and writes to the Working Context. Non-blocking: your assistant doesn't wait.

```
observe("user", text)
    │
    ├── EntityExtractor  → NER (spaCy + GLiNER) + coreference resolution (coreferee)
    ├── UserStateInferencer → emotional tone, focus, intent classification
    ├── MemoryRouter     → intent-aware query → spreading-activation retrieval
    └── WorkingContextManager → updates all four pillars atomically
```

### L2 — Neo4j Knowledge Graph

Three memory node types with rich metadata:

- **ExperienceMemory** — episodic events (`"Alice mentioned a project deadline on Tuesday"`)
- **KnowledgeMemory** — semantic facts (`"Alice prefers short answers"`, `"Alice works in Python"`)
- **RelationshipMemory** — people and relationships (`"Alice's manager is Bob"`)

Relationships are typed and carry frequency and strength fields that reinforce over repeated activation. The graph schema enforces single canonical nodes per person and concept (MERGE semantics), preventing duplicates.

### Processing Pipeline

| Component | Role | Degrades gracefully? |
|---|---|---|
| spaCy | Named entity recognition, token matching | ✅ core NLP disabled if absent |
| GLiNER | Zero-shot NER (projects, events, emotions, concepts) | ✅ skipped if absent |
| coreferee | Coreference resolution ("he" → "Bob") | ✅ skipped if absent |
| `all-MiniLM-L6-v2` | Sentence embeddings for semantic search | required |
| `ms-marco-MiniLM-L-6-v2` | Cross-encoder reranking | required |

Every NLP component either loads or skips — the module never refuses to boot.

### Spreading Activation Retrieval

Retrieval uses spreading activation, not a flat nearest-neighbor search. Each edge type has a base conductivity weight:

```python
BASE_WEIGHTS = {
    "CAUSED":              0.90,   # causal chain — highest conductivity
    "EXPERIENCE_CHAIN":    0.80,   # narrative thread
    "KNOWLEDGE_HIERARCHY": 0.70,   # conceptual abstraction
    "TEMPORAL":            0.50,   # time-ordered events
    "SIMILAR_TO":          0.15,   # deliberately low — prevents topic drift
}
```

Per-intent boosts adjust conductivity dynamically. An emotional query amplifies `INFLUENCED` and `TRIGGERED` paths. A temporal query amplifies `HAPPENED_BEFORE` and `CONCURRENT`. The result is retrieval that *follows the shape of the question*, not just keyword overlap.

---

## The Working Context

`get_full_context()` returns a strongly-typed `WorkingContext` dataclass. Use it to build your system prompt:

```python
ctx = manager.get_full_context()

# What the assistant should know right now
ctx.memory.must_know        # List[str] — high-priority recalled memories
ctx.memory.context          # List[str] — supporting context
ctx.memory.associations     # List[str] — loosely related memories
ctx.memory.recent_turns     # List[dict] — last N conversation turns
ctx.memory.retrieval_meta   # confidence score, latency, active entities

# About the assistant itself
ctx.assistant.current_mode        # "focused" | "creative" | "empathetic" | ...
ctx.assistant.current_datetime    # "Monday, 14 July 2026 at 15:42"
ctx.assistant.time_of_day         # "afternoon"
ctx.assistant.last_topics_discussed
ctx.assistant.last_commitments    # commitments made in the last response

# About the user
ctx.user.emotional_state          # inferred from message tone
ctx.user.current_focus            # most recently mentioned entity
ctx.user.mentioned_entities       # Set[str] — entities in this turn

# Agentic workspace
ctx.workspace.get_active_tasks()
ctx.workspace.get_pending_notifications()
```

---

## Proactive Notifications

Wire up a callback to receive proactive notifications — no polling required:

```python
def on_notification(item: WorkspaceItem) -> None:
    # item.title, item.type, item.priority, item.due_at
    # trigger your LLM with get_full_context() as the system prompt
    print(f"Proactive: {item.title}")

manager = MemoryManager(
    ...,
    on_proactive_notification=on_notification,
)
```

Add items from anywhere:

```python
from memory.working_memory.working_context import (
    WorkspaceItem, WorkspaceItemType, NotifyPriority
)
from datetime import datetime, timedelta

ctx = manager.get_full_context()
ctx.workspace.add_item(WorkspaceItem(
    title="Follow up on the CLI project",
    type=WorkspaceItemType.REMINDER,
    notify=True,
    notify_priority=NotifyPriority.NORMAL,  # fires after 30s user inactivity
    due_at=datetime.now() + timedelta(hours=1),
))
```

Priority rules:
- `URGENT` — fires immediately, interrupts regardless of conversation state
- `NORMAL` — fires after `gap_threshold_s` seconds of user inactivity (default 30s)
- `LOW` — never fires proactively; surfaces in `ctx.workspace` at the next user turn

---

## Memory Consolidation

Long-running sessions accumulate conversation turns. Run the consolidation pipeline to distill them into the graph:

```bash
python -m memory.processing.consolidation_runner
```

The pipeline:
1. **Fetches** existing graph memories likely relevant to the session via spreading activation
2. **Sends** the conversation + pre-fetched memories to a Gemini CLI agent
3. **Parses** the agent's structured consolidation plan (no free-form output)
4. **Executes** the plan deterministically against Neo4j:
   - `CREATE` — new memory nodes
   - `UPDATE` — patch fields on existing nodes
   - `ENHANCE` — add new relationships between existing nodes
   - `SKIP` — already known, nothing to do
   - `CONTRADICT` — mark superseded nodes (`superseded=true`), never delete

The agent never has direct graph access during reasoning — all relevant memories are pre-fetched and inlined into the prompt. This keeps the reasoning self-contained and the execution deterministic.

Requires the [Gemini CLI](https://github.com/google-gemini/gemini-cli) installed and authenticated.

---

## Data Layout

All data lives under `<base_dir>/BRAIN/`:

```
~/.memory/BRAIN/
  runtime/
    session/
      conversation.json       — turn-by-turn log
      working_context.json    — live 4-pillar state (schema v2.0)
    tasks/                    — workspace task records
    logs/                     — diagnostic events (when log=True)
  data/
    neo4j/                    — Neo4j data volume (Docker bind mount)
    reviews/                  — per-query retrieval traces (when review=True)
    consolidation_dry_runs/   — preview runs without writing to graph
```

---

## Observability

```python
# Diagnostic log — one rolling file per day under BRAIN/data/logs/
manager = MemoryManager(log=True, ...)

# Retrieval trace — per-query JSON under BRAIN/data/reviews/observe/
manager = MemoryManager(review=True, ...)
```

Both are off by default. `review=True` is useful during development to inspect exactly what the retrieval engine found and why.

---

## Comparison

| Feature | assistant-memory | MemGPT | Mem0 | LangChain Memory |
|---|:---:|:---:|:---:|:---:|
| Graph-based storage | ✅ | ✅ | ✅ | ❌ |
| Spreading activation retrieval | ✅ | ❌ | ❌ | ❌ |
| Intent-aware traversal | ✅ | ❌ | ❌ | ❌ |
| Fully local (no cloud API) | ✅ | ❌ | ❌ | ✅ |
| 4-pillar typed Working Context | ✅ | ❌ | ❌ | ❌ |
| Proactive notification system | ✅ | ❌ | ❌ | ❌ |
| Agentic consolidation reasoning | ✅ | ✅ | ❌ | ❌ |
| Graceful NLP degradation | ✅ | ❌ | — | — |
| Plug-and-play (5-line setup) | ✅ | ❌ | ✅ | ✅ |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, branch model, and PR checklist.

```bash
git clone https://github.com/your-org/assistant-memory
cd assistant-memory
pip install -e ".[nlp,dev]"
pytest
```

---

## Roadmap

- [ ] `pip install assistant-memory` on PyPI
- [ ] Managed Neo4j option (for deployment without Docker)
- [ ] Multi-user memory isolation
- [ ] OpenTelemetry tracing export
- [ ] REST API wrapper for language-agnostic use

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built for the open-source AI community. If this saves you three weeks of reinventing memory, consider giving it a ⭐.

</div>
