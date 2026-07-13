# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.0] — 2026-07-13

### Added

- **3-tier cognitive architecture**: WorkingMemory (L1 in-process) + Neo4j knowledge graph (L2 persistent) + entity extraction pipeline.
- **Concurrent async boot**: four initialization tasks (Neo4j, embeddings, reranker, working memory) run in parallel via `asyncio.gather` — typical boot time under 8 seconds on warm hardware.
- **WorkspaceWatcher proactive notifications**: background watcher fires a callback when the system determines something is worth surfacing between turns, without polling.
- **Local ML stack**: `all-MiniLM-L6-v2` embeddings, `ms-marco-MiniLM-L-6-v2` cross-encoder reranker, spaCy NER, GLiNER zero-shot entity extraction, coreferee coreference resolution — all local, no external API calls.
- **Graceful NLP degradation**: each NLP component degrades gracefully if its package is not installed; the module boots and runs on a minimal install.
- **Automatic NLP model download**: missing spaCy model and coreferee English data are downloaded automatically on first run.
- **Plug-and-play API**: single `MemoryManager` class; all config via constructor args or env vars; no hardcoded identity or paths.
- **Dynamic assistant section key**: `ctx.assistant` / `ctx.memory` / `ctx.user` JSON keys follow `config.assistant_name` — rename without touching the module.
- **Memory consolidation pipeline**: agentic Gemini CLI pipeline that distills conversation history into the Neo4j graph (`python -m memory.processing.consolidation_runner`).
- **Observability**: structured trace system with per-component log filtering and rolling log files (`log=True`), plus per-query pipeline traces (`review=True`).
