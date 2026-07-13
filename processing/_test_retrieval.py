"""
End-to-end retrieval testing against the populated graph.

Three levels:
  L1 -- Direct retrieval engine methods (BM25, spreading activation, emotional, ...)
  L2 -- Router with realistic queries (full pipeline minus working memory)
  L3 -- Full memory manager (observe + get_context)

Run: python -m memory.processing._test_retrieval
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from memory.config import config
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.infrastructure.neo4j_client import create_neo4j_client, Neo4jClient
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.long_term.memory_router import MemoryRouter, Intent
from memory.processing.embedding_utils import EmbeddingUtils


def _hr(title: str = "") -> None:
    print()
    print("=" * 76)
    if title: print(f"  {title}")
    if title: print("=" * 76)


def _row(label: str, value: Any) -> None:
    print(f"  {label:24}  {value}")


def _show_memories(mems: List[Dict], limit: int = 5, indent: str = "    ") -> None:
    if not mems:
        print(f"{indent}(no results)")
        return
    for i, m in enumerate(mems[:limit]):
        ident = (
            m.get("person_name")
            or m.get("concept")
            or (m.get("content") or m.get("root_content") or "?")[:80]
        )
        labels = m.get("type") or m.get("label") or "?"
        if isinstance(labels, list):
            labels = labels[0] if labels else "?"
        score = m.get("bm25_score") or m.get("score") or m.get("activation_score")
        score_str = f"  score={score:.2f}" if isinstance(score, (int, float)) else ""
        print(f"{indent}{i+1}. [{labels:18}]{score_str}  {ident}")
    if len(mems) > limit:
        print(f"{indent}... +{len(mems)-limit} more")


# ===============================================================================
# L1 -- DIRECT RETRIEVAL METHODS
# ===============================================================================

async def test_l1_direct_methods(eng: MemoryRetrievalEngine) -> None:
    _hr("L1 -- DIRECT RETRIEVAL ENGINE METHODS")

    # 1. BM25 -- single keyword
    print()
    print("  [1] bm25_search(['Sarah'])  -- single named entity")
    t = time.perf_counter()
    rows = await eng.bm25_search(query_terms=["Sarah"], limit=10)
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 2. BM25 -- multi-keyword (entity + concept)
    print()
    print("  [2] bm25_search(['Sarah', 'conflict'])  -- entity + theme")
    t = time.perf_counter()
    rows = await eng.bm25_search(query_terms=["Sarah", "conflict"], limit=10)
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 3. Recent
    print()
    print("  [3] get_recent_memories(days=10)")
    t = time.perf_counter()
    rows = await eng.get_recent_memories(days=10, limit=10)
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 4. Emotionally significant
    print()
    print("  [4] get_emotionally_significant_memories(min_intensity=0.4)")
    t = time.perf_counter()
    rows = await eng.get_emotionally_significant_memories(
        min_emotional_intensity=0.4, limit=10,
    )
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 5. Recent emotional (last 7 days, lower threshold)
    print()
    print("  [5] get_recent_emotional_memories(days=7, min=0.3)")
    t = time.perf_counter()
    rows = await eng.get_recent_emotional_memories(days=7, min_intensity=0.3, limit=10)
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 6. Topic search
    print()
    print("  [6] get_memories_by_topic('Sarah')")
    t = time.perf_counter()
    rows = await eng.get_memories_by_topic(topic="Sarah", limit=10)
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)

    # 7. Spreading activation from Sarah's node
    print()
    print("  [7] _spreading_activation(['Sarah'], intent='entity')")
    t = time.perf_counter()
    rows = await eng._spreading_activation(
        entities=["Sarah"], intent="entity", budget=20,
    )
    print(f"      {(time.perf_counter()-t)*1000:.0f}ms  ->  {len(rows)} results")
    _show_memories(rows)


# ===============================================================================
# L2 -- ROUTER-LEVEL TESTS
# ===============================================================================

async def test_l2_router(router: MemoryRouter) -> None:
    _hr("L2 -- ROUTER (full retrieval pipeline)")

    test_queries = [
        # (label, message, entities)
        ("ENTITY -- known person",          "Tell me about Sarah",                 ["Sarah"]),
        ("ENTITY -- multi-entity",          "What happened with Sarah and Marcus", ["Sarah", "Marcus"]),
        ("EMOTIONAL -- no entity",          "I'm feeling overwhelmed today",       []),
        ("EMOTIONAL -- with entity",        "I'm still stressed about Sarah",      ["Sarah"]),
        ("TEMPORAL -- last week",           "What happened last Thursday",         []),
        ("FACTUAL -- knowledge query",      "what do I know about async context managers", []),
        ("FACTUAL -- about person",         "what do I know about Alex",           ["Alex"]),
        ("AMBIENT -- greeting (should bypass)", "hello",                            []),
        ("Unknown entity",                  "tell me about Bobby",                ["Bobby"]),
    ]

    for label, msg, entities in test_queries:
        print()
        print(f"  [{label}]")
        print(f"  message:  {msg!r}")
        print(f"  entities: {entities}")
        t = time.perf_counter()
        try:
            result = await router.route(message=msg, entities=entities)
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            continue
        ms = (time.perf_counter() - t) * 1000
        print(f"  intent={result.intent.value} confidence={result.confidence:.2f}  "
              f"signals={result.signals_fired}  {ms:.0f}ms")
        print(f"  must_know={len(result.must_know)}  context={len(result.context)}  "
              f"associations={len(result.associations)}")
        if result.must_know:
            print("  must_know:")
            _show_memories(result.must_know, limit=3, indent="    ")
        if result.context and len(result.context) > 0:
            print("  context:")
            _show_memories(result.context, limit=2, indent="    ")


# ===============================================================================
# L3 -- FULL MEMORY MANAGER LOOP
# ===============================================================================

async def test_l3_memory_manager() -> None:
    """
    End-to-end test of the public API:
        MemoryManager.setup() -> observe() -> get_context() / get_full_context()

    Validates the working-memory async bridge, four-pillar WorkingContext,
    and that retrieval results actually surface through the wrapper layer.
    """
    from memory.memory_manager import MemoryManager

    _hr("L3 -- MEMORY MANAGER (full observe + get_context loop)")

    manager = MemoryManager(log=False, review=False)
    print()
    print("  booting MemoryManager...")
    t = time.perf_counter()
    await manager.setup()
    print(f"  setup done in {(time.perf_counter()-t)*1000:.0f}ms")

    # A realistic conversational sequence — each message exercises a different intent.
    turns = [
        ("user",       "Tell me about Sarah",                 "ENTITY"),
        ("user",       "I'm feeling overwhelmed today",       "EMOTIONAL"),
        ("user",       "what happened last Thursday",         "TEMPORAL"),
        ("user",       "what do I know about Alex",           "FACTUAL/ENTITY"),
        ("user",       "hello",                                "AMBIENT"),
    ]

    failures: List[str] = []

    for role, msg, expected_intent in turns:
        print()
        print(f"  [observe] role={role}  msg={msg!r}  expected={expected_intent}")

        # observe is non-blocking — fires reactive_processing in a background thread
        t_obs = time.perf_counter()
        await manager.observe(role, msg)
        obs_ms = (time.perf_counter() - t_obs) * 1000

        # Use get_context_async — sync get_context() would freeze the event
        # loop and starve the bridged Neo4j coroutines.
        t_ctx = time.perf_counter()
        ctx = await manager.get_context_async(role, msg)
        ctx_ms = (time.perf_counter() - t_ctx) * 1000

        meta   = ctx.get("retrieval_meta", {})
        tiered = ctx.get("tiered_memories", {})
        mk = tiered.get("must_know", [])
        ct = tiered.get("context", [])
        asn = tiered.get("associations", [])

        print(f"    observe={obs_ms:.0f}ms  get_context={ctx_ms:.0f}ms")
        print(f"    intent={meta.get('intent')}  confidence={meta.get('confidence', 0):.2f}  "
              f"signals={meta.get('signals_fired')}  retrieval_ms={meta.get('latency_ms', 0):.0f}")
        print(f"    must_know={len(mk)}  context={len(ct)}  associations={len(asn)}")

        # Hard assertions — flag, don't crash, so subsequent turns still run
        if not meta:
            failures.append(f"{msg!r}: retrieval_meta empty")
        # NOTE: "hello" can legitimately surface memories mid-conversation —
        # working memory carries active entities across turns, so the classifier
        # no longer sees an empty entity list. Pure AMBIENT bypass only fires
        # on the very first message with no active entities.
        if "ENTITY" in expected_intent or "EMOTIONAL" in expected_intent:
            # We seeded Sarah/Alex into the graph during consolidation, so something
            # should always come back unless retrieval is broken
            if not (mk or ct or asn):
                failures.append(f"{msg!r}: expected memories for {expected_intent}, got 0")

        if mk:
            _show_memories(mk, limit=2, indent="      mk> ")

    # Final probe — full WorkingContext snapshot (all four pillars)
    print()
    print("  [get_full_context] inspecting four pillars")
    full = manager.get_full_context()

    pillars = {
        "memory":    full.memory,
        "assistant": full.assistant,
        "user":      full.user,
        "workspace": full.workspace,
    }
    for name, p in pillars.items():
        if p is None:
            failures.append(f"WorkingContext.{name} is None")
            print(f"    [FAIL]  {name}: None")
        else:
            print(f"    [OK]    {name}: {type(p).__name__}")

    if full.assistant:
        _row("assistant.name",             full.assistant.name)
        _row("assistant.current_datetime", full.assistant.current_datetime)
        _row("assistant.time_of_day",      full.assistant.time_of_day)

    if full.user:
        _row("user.user_id",            full.user.user_id)
        _row("user.mentioned_entities", list(full.user.mentioned_entities or [])[:5])

    if full.memory:
        _row("memory.must_know",    len(full.memory.must_know or []))
        _row("memory.context",      len(full.memory.context or []))
        _row("memory.associations", len(full.memory.associations or []))
        _row("memory.recent_turns", len(full.memory.recent_turns or []))

    # Shutdown — leave Docker running for next test invocation
    await manager.shutdown(stop_docker=False)

    print()
    if failures:
        print(f"  [L3 FAIL]  {len(failures)} assertion(s) failed:")
        for f in failures:
            print(f"     - {f}")
    else:
        print("  [L3 PASS]  all assertions ok")


# ===============================================================================
# DRIVER
# ===============================================================================

async def main() -> None:
    _hr("RETRIEVAL TEST  --  BRAIN/memory/data/neo4j")

    # Ensure Docker + Neo4j up
    dm = DockerManager()
    if not dm.is_docker_running():
        print("  [FAIL] Docker daemon not running")
        return
    try:
        dm.start_docker()
        await dm.ensure_connection_async()
    except Exception as exc:
        print(f"  [FAIL] {exc}")
        return

    neo4j = create_neo4j_client(
        uri=config.neo4j_uri,
        username=config.neo4j_username,
        password=config.neo4j_password,
        database=config.database,
    )
    await neo4j.connect()

    # Graph snapshot
    nodes = await neo4j.execute_query("MATCH (n) RETURN labels(n)[0] AS l, count(n) AS c ORDER BY c DESC")
    edges = await neo4j.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")
    print()
    print("  Graph state:")
    for r in nodes: _row(r["l"], f"{r['c']} nodes")
    _row("edges", edges[0]["c"] if edges else 0)

    embed = EmbeddingUtils()
    eng = MemoryRetrievalEngine(neo4j_client=neo4j, embedding_utils=embed)
    router = MemoryRouter(engine=eng)

    # Load the cross-encoder reranker -- used in router._package
    from memory.long_term import reranker as _reranker
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _reranker.load_reranker)

    # Warm up Neo4j query plan cache so reported latencies reflect steady-state
    # behaviour rather than first-call cold-start.
    print()
    print("  warming up retrieval engine...")
    t = time.perf_counter()
    await eng.warmup()
    print(f"  warmup done in {(time.perf_counter()-t)*1000:.0f}ms")

    try:
        await test_l1_direct_methods(eng)
        await test_l2_router(router)
    finally:
        await neo4j.disconnect()

    # L3 needs its own MemoryManager (with its own Neo4j connection). Run it
    # after the L1/L2 client is fully released so we don't double-pool.
    await test_l3_memory_manager()

    print()
    print("=" * 76)
    print("  DONE")
    print("=" * 76)


if __name__ == "__main__":
    asyncio.run(main())
