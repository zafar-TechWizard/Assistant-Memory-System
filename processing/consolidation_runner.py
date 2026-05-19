"""
Consolidation Runner -- Standalone wrapper for running consolidation via Gemini CLI.

Pre-flight checks, then processes pending sessions from conversation.json,
calling the agentic engine for each. Failed sessions stay in the log for
next-night retry. Successful sessions are removed after processing.

Usage:
    # Pre-flight only -- verify everything is ready (no consolidation)
    python -m memory.processing.consolidation_runner --check-only

    # Default -- connect Neo4j, run consolidation on all pending sessions
    python -m memory.processing.consolidation_runner

    # Dry run -- produce plans but don't write to graph
    python -m memory.processing.consolidation_runner --dry-run

    # Specific session only
    python -m memory.processing.consolidation_runner --session-id session_xyz

    # Custom gemini cli path
    python -m memory.processing.consolidation_runner --gemini /usr/local/bin/gemini

    # Verbose log output (writes to BRAIN/memory/data/logs/ via observer)
    python -m memory.processing.consolidation_runner --log

Configure Gemini CLI manually BEFORE running:
    1. Authenticate (gemini auth login or set GEMINI_API_KEY)
    2. Set model: manual mode, gemini-3.1-pro-preview

This wrapper assumes the CLI is already configured. It does NOT change model
settings -- that's the user's responsibility per their workflow.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from memory.config import get_config
from memory.long_term.infrastructure.docker_manager import DockerManager
from memory.long_term.infrastructure.neo4j_client import Neo4jClient, create_neo4j_client
from memory.long_term.memory_retrieval_engine import MemoryRetrievalEngine
from memory.observability import observer
from memory.processing.consolidation import (
    AgenticConsolidationEngine,
    GeminiAgent,
    SessionResult,
    check_gemini_cli,
    check_gemini_auth,
)
from memory.processing.embedding_utils import EmbeddingUtils


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class ConsolidationRunner:
    """
    End-to-end runner. Owns connection lifecycle, pre-flight checks, and the
    per-session loop. Designed to be called from a script entry point.
    """

    def __init__(
        self,
        gemini_cli_path: str = "gemini",
        agent_timeout_s: int = 180,
        dry_run: bool = False,
        log_enabled: bool = False,
    ):
        self.gemini_cli_path = gemini_cli_path
        self.agent_timeout_s = agent_timeout_s
        self.dry_run = dry_run

        # Configure observer FIRST so all subsequent events get logged if requested
        observer.configure(log=log_enabled, review=False)

        self.cfg = get_config()
        self.neo4j: Optional[Neo4jClient] = None
        self.docker_manager: Optional[DockerManager] = None
        self.engine: Optional[AgenticConsolidationEngine] = None

    # -- Pre-flight -------------------------------------------------------------

    async def preflight(self, skip_conversation_check: bool = False) -> bool:
        """
        Verify everything's in place before running.

        Order:
          1. Gemini CLI installed
          2. Gemini auth working
          3. Docker daemon running
          4. Neo4j container started (creates/starts if needed)
          5. Neo4j connection ready
          6. conversation.json has pending sessions (optional -- skipped if --check-only)
        """
        print("-" * 72)
        print("CONSOLIDATION RUNNER -- PRE-FLIGHT")
        print("-" * 72)

        # 0. Ensure all operational directories exist
        dirs = self.cfg.ensure_directories()
        self._print_check(
            "Operational directories",
            True,
            f"{len(dirs)} dirs ready under {dirs['data_dir']}",
        )

        # 1. Gemini CLI binary
        ok_cli, msg_cli = await check_gemini_cli(self.gemini_cli_path)
        self._print_check("Gemini CLI installed", ok_cli, msg_cli)
        if not ok_cli:
            print()
            print("  -> Install Google's Gemini CLI:")
            print("      npm install -g @google/gemini-cli")
            print("    Or pass --gemini /path/to/binary")
            return False

        # 2. Gemini auth (sends a trivial test prompt)
        print("    (verifying auth -- this may take 30-60s on first call)")
        ok_auth, msg_auth = await check_gemini_auth(self.gemini_cli_path)
        self._print_check("Gemini auth", ok_auth, msg_auth)
        if not ok_auth:
            print()
            print("  -> First-time setup may need interactive auth:")
            print("      gemini             # then follow auth prompts")
            print("    Or set GEMINI_API_KEY env var")
            return False

        # 3 + 4. Docker + Neo4j container
        ok_docker, msg_docker = await self._ensure_docker_and_neo4j_container()
        self._print_check("Docker + Neo4j container", ok_docker, msg_docker)
        if not ok_docker:
            print()
            print("  -> Ensure Docker Desktop is installed and running")
            return False

        # 5. Neo4j connection (graph database is reachable)
        ok_neo, msg_neo = await self._check_neo4j_connection()
        self._print_check("Neo4j connection", ok_neo, msg_neo)
        if not ok_neo:
            print()
            print("  -> Neo4j container is up but Bolt is not accepting connections")
            print("    Check `docker logs sofi-neo4j-memory` for errors")
            return False

        # 6. Pending sessions (optional)
        if not skip_conversation_check:
            ok_log, msg_log = self._check_conversation_log()
            self._print_check("conversation.json", ok_log, msg_log)
            if not ok_log:
                print()
                print("  -> Nothing to consolidate. Exit cleanly.")
                return False

        # Final reminder about model selection
        print()
        print("! REMINDER: Your Gemini CLI should be configured with:")
        print("    model: gemini-3.1-pro-preview")
        print("    mode:  manual")
        print()
        if self.dry_run:
            print("  -> DRY RUN mode: plans will be produced, no graph writes")
        print("-" * 72)
        return True

    @staticmethod
    def _print_check(label: str, ok: bool, msg: str) -> None:
        mark = "[OK]" if ok else "[FAIL]"
        truncated = (msg[:80] + "...") if len(msg) > 80 else msg
        print(f"  {mark}  {label}: {truncated}")

    async def _ensure_docker_and_neo4j_container(self) -> Tuple[bool, str]:
        """
        Ensure Docker daemon is running AND the sofi-neo4j-memory container
        is up and healthy. Creates/starts container if needed.
        """
        try:
            self.docker_manager = DockerManager()
        except Exception as exc:
            return False, f"DockerManager init failed: {exc}"

        # Is Docker daemon itself running?
        if not self.docker_manager.is_docker_running():
            return False, "Docker daemon not running"

        # Start container if needed (idempotent -- does nothing if already up)
        try:
            self.docker_manager.start_docker()
        except RuntimeError as exc:
            return False, f"container start failed: {exc}"

        # Wait for Neo4j to accept connections (non-blocking)
        try:
            await self.docker_manager.ensure_connection_async()
        except RuntimeError as exc:
            return False, f"Neo4j readiness check failed: {exc}"

        if self.docker_manager.is_container_running():
            return True, f"container '{self.docker_manager.container_name}' is up"
        return False, "container did not stay running"

    async def _check_neo4j_connection(self) -> Tuple[bool, str]:
        """
        Connect with retries. HTTP and Bolt come up separately in Neo4j;
        the HTTP-readiness check from DockerManager doesn't guarantee Bolt is up.
        """
        attempts = 8
        delay_s = 4
        last_err = "unknown"
        for i in range(1, attempts + 1):
            try:
                self.neo4j = create_neo4j_client(
                    uri=self.cfg.neo4j_uri,
                    username=self.cfg.neo4j_username,
                    password=self.cfg.neo4j_password,
                    database=self.cfg.database,
                )
                await self.neo4j.connect()
                health = await self.neo4j.health_check()
                if health.get("status") == "healthy":
                    return True, (
                        f"{self.cfg.neo4j_uri} "
                        f"({health.get('response_time_ms', 0):.0f}ms, attempt {i})"
                    )
                last_err = str(health)
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {str(exc)[:120]}"
                try:
                    if self.neo4j:
                        await self.neo4j.disconnect()
                except Exception:
                    pass
                self.neo4j = None
            if i < attempts:
                await asyncio.sleep(delay_s)
        return False, f"after {attempts} attempts: {last_err}"

    def _check_conversation_log(self) -> Tuple[bool, str]:
        path = Path(self.cfg.conversation_log_path)
        if not path.exists():
            return False, f"file not found at {path}"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"parse error: {exc}"

        user_key = f"user_{self.cfg.user_id}"
        sessions = data.get(user_key, []) if isinstance(data, dict) else []
        n_sessions = len(sessions)
        n_turns = sum(len(s.get("conversations") or []) for s in sessions)
        if n_sessions == 0:
            return False, "no sessions to consolidate"
        return True, f"{n_sessions} sessions, {n_turns} total turns"

    # -- Engine setup -----------------------------------------------------------

    async def _build_engine(self) -> None:
        embed = EmbeddingUtils()
        retrieval = MemoryRetrievalEngine(
            neo4j_client=self.neo4j, embedding_utils=embed,
        )
        # Ensure ALL indexes + constraints exist before consolidation writes:
        #   - Vector indexes (semantic search after consolidation)
        #   - Uniqueness constraints (canonical person_name, concept)
        #   - Property indexes (timestamp, participants, concept, person_name)
        try:
            await self.neo4j.create_constraints_and_indexes()
        except Exception as exc:
            observer.warning("create_constraints_and_indexes failed", error=str(exc))

        # Ensure BM25 fulltext index too (used by pre-fetcher for context lookup)
        try:
            await retrieval.ensure_fulltext_index()
        except Exception as exc:
            observer.warning("ensure_fulltext_index failed (non-fatal)", error=str(exc))

        agent = GeminiAgent(
            cli_path=self.gemini_cli_path,
            timeout_s=self.agent_timeout_s,
        )
        self.engine = AgenticConsolidationEngine(
            neo4j=self.neo4j,
            embed=embed,
            retrieval=retrieval,
            agent=agent,
        )

    # -- Main loop --------------------------------------------------------------

    async def run(self, only_session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process pending sessions. Returns aggregate summary.
        """
        if self.engine is None:
            await self._build_engine()

        log_path = Path(self.cfg.conversation_log_path)
        log_data = self._load_log(log_path)
        user_key = f"user_{self.cfg.user_id}"
        sessions = log_data.get(user_key, []) if isinstance(log_data, dict) else []

        if only_session_id:
            sessions = [s for s in sessions if s.get("session_id") == only_session_id]
            if not sessions:
                print(f"\nSession {only_session_id} not found in conversation.json")
                return self._empty_summary()

        print()
        print("-" * 72)
        print(f"PROCESSING {len(sessions)} SESSION(S)")
        print("-" * 72)

        results: List[SessionResult] = []
        keep_sessions: List[Dict] = []

        for i, session in enumerate(sessions, 1):
            sid = session.get("session_id", "unknown")
            turns = len(session.get("conversations") or [])
            print()
            print(f"[{i}/{len(sessions)}] Session: {sid}  ({turns} turns)")

            if turns == 0:
                print(f"  -> empty session, skipping")
                continue

            start = datetime.now()
            try:
                if self.dry_run:
                    result = await self._dry_run_session(session)
                else:
                    result = await self.engine.consolidate_session(session)
            except Exception as exc:
                observer.error("session crashed in runner", exception=exc,
                                 session_id=sid)
                result = SessionResult(
                    session_id=sid, turns=turns,
                    error=f"{type(exc).__name__}: {exc}",
                )

            elapsed_s = (datetime.now() - start).total_seconds()

            self._print_session_result(result, elapsed_s)
            results.append(result)

            # On dry-run keep everything. On real run, keep only failed sessions.
            if self.dry_run or not result.succeeded:
                keep_sessions.append(session)

        # Persist remaining sessions (only on non-dry-run)
        if not self.dry_run and not only_session_id:
            log_data[user_key] = keep_sessions
            self._save_log(log_path, log_data)
        elif not self.dry_run and only_session_id:
            # When targeting one session, update only that one in the file
            self._update_one_session(log_path, only_session_id, results)

        summary = self._summarize(results)
        self._print_summary(summary)
        return summary

    async def _dry_run_session(self, session: Dict) -> SessionResult:
        """Run pipeline up to plan production but don't execute writes."""
        sid = str(session.get("session_id") or "unknown")
        turns = len(session.get("conversations") or [])
        result = SessionResult(session_id=sid, turns=turns)

        # Manually do fetch + agent (skip executor)
        from memory.processing.consolidation import _derive_session_date
        try:
            context = await self.engine.fetcher.fetch(session.get("conversations") or [])
            plan = await self.engine.agent.plan(
                session_id=sid,
                user_id=self.cfg.user_id,
                session_date=_derive_session_date(session),
                conversations=session.get("conversations") or [],
                existing_context=context,
            )
            if plan is None:
                result.error = "agent failed to produce a valid plan"
                return result
            result.operations_planned = len(plan.operations)
            result.reasoning = plan.reasoning
            result.summary = plan.session_summary
            for op in plan.operations:
                if op.operation == "CREATE":
                    result.nodes_created += 1
                elif op.operation == "UPDATE":
                    result.nodes_updated += 1
                elif op.operation == "ENHANCE":
                    result.nodes_enhanced += 1
                elif op.operation == "SKIP":
                    result.nodes_skipped += 1
                elif op.operation == "CONTRADICT":
                    result.nodes_superseded += 1
            result.edges_created = len(plan.edges)
            result.succeeded = True

            # Save plan to disk for inspection
            self._save_dry_run_plan(sid, plan)

        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        return result

    def _save_dry_run_plan(self, session_id: str, plan: Any) -> None:
        """Save dry-run plan JSON to <BRAIN/memory/data>/consolidation_dry_runs/."""
        try:
            out_dir = self.cfg.consolidation_dry_runs_dir
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = out_dir / f"{stamp}_{session_id}.json"

            # Serialize the plan
            serialized = {
                "session_id": plan.session_id,
                "reasoning": plan.reasoning,
                "session_summary": plan.session_summary,
                "overall_sentiment": plan.overall_sentiment,
                "operations": [
                    {
                        "op_index": op.op_index,
                        "operation": op.operation,
                        "reason": op.reason,
                        "target_id": op.target_id,
                        "update_fields": op.update_fields,
                        "enhance_additions": op.enhance_additions,
                        "memory": (
                            {
                                k: v for k, v in op.memory.__dict__.items()
                                if v not in (None, [], {})
                            } if op.memory else None
                        ),
                    }
                    for op in plan.operations
                ],
                "edges": [
                    {
                        "from_op_index": e.from_op_index,
                        "from_node_id": e.from_node_id,
                        "to_op_index": e.to_op_index,
                        "to_node_id": e.to_node_id,
                        "rel_type": e.rel_type,
                        "strength": e.strength,
                        "bidirectional": e.bidirectional,
                    }
                    for e in plan.edges
                ],
            }
            path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  -> plan saved: {path}")
        except Exception as exc:
            print(f"  -> couldn't save plan: {exc}")

    # -- Reporting --------------------------------------------------------------

    @staticmethod
    def _print_session_result(r: SessionResult, elapsed_s: float) -> None:
        mark = "[OK]" if r.succeeded else "[FAIL]"
        print(f"  {mark}  {elapsed_s:.1f}s  ", end="")
        if r.error:
            print(f"FAILED: {r.error}")
            return
        if r.operations_planned == 0:
            print(f"nothing worth memorizing  -- {r.summary}")
            return
        print(
            f"created={r.nodes_created} "
            f"updated={r.nodes_updated} "
            f"enhanced={r.nodes_enhanced} "
            f"skipped={r.nodes_skipped} "
            f"superseded={r.nodes_superseded} "
            f"edges={r.edges_created}"
        )
        if r.summary:
            print(f"       summary: {r.summary[:100]}")

    @staticmethod
    def _summarize(results: List[SessionResult]) -> Dict[str, Any]:
        return {
            "sessions_attempted": len(results),
            "sessions_succeeded": sum(1 for r in results if r.succeeded),
            "sessions_failed":    sum(1 for r in results if not r.succeeded),
            "total_nodes_created":    sum(r.nodes_created for r in results),
            "total_nodes_updated":    sum(r.nodes_updated for r in results),
            "total_nodes_enhanced":   sum(r.nodes_enhanced for r in results),
            "total_nodes_skipped":    sum(r.nodes_skipped for r in results),
            "total_nodes_superseded": sum(r.nodes_superseded for r in results),
            "total_edges_created":    sum(r.edges_created for r in results),
        }

    @staticmethod
    def _empty_summary() -> Dict[str, Any]:
        return {
            "sessions_attempted": 0, "sessions_succeeded": 0, "sessions_failed": 0,
            "total_nodes_created": 0, "total_nodes_updated": 0,
            "total_nodes_enhanced": 0, "total_nodes_skipped": 0,
            "total_nodes_superseded": 0, "total_edges_created": 0,
        }

    @staticmethod
    def _print_summary(s: Dict[str, Any]) -> None:
        print()
        print("-" * 72)
        print("CONSOLIDATION SUMMARY")
        print("-" * 72)
        print(f"  Sessions attempted:  {s['sessions_attempted']}")
        print(f"  Sessions succeeded:  {s['sessions_succeeded']}")
        print(f"  Sessions failed:     {s['sessions_failed']}")
        print()
        print(f"  Nodes created:       {s['total_nodes_created']}")
        print(f"  Nodes updated:       {s['total_nodes_updated']}")
        print(f"  Nodes enhanced:      {s['total_nodes_enhanced']}")
        print(f"  Nodes skipped:       {s['total_nodes_skipped']}")
        print(f"  Nodes superseded:    {s['total_nodes_superseded']}")
        print(f"  Edges created:       {s['total_edges_created']}")
        print("-" * 72)

    # -- File helpers -----------------------------------------------------------

    @staticmethod
    def _load_log(path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            observer.error("conversation log load failed", exception=exc)
            return {}

    @staticmethod
    def _save_log(path: Path, data: Dict) -> None:
        try:
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            observer.error("conversation log save failed", exception=exc)

    def _update_one_session(
        self, path: Path, session_id: str, results: List[SessionResult],
    ) -> None:
        """When --session-id is used, only remove THAT session if it succeeded."""
        data = self._load_log(path)
        user_key = f"user_{self.cfg.user_id}"
        sessions = data.get(user_key, []) if isinstance(data, dict) else []
        succeeded = any(r.succeeded and r.session_id == session_id for r in results)
        if succeeded:
            sessions = [s for s in sessions if s.get("session_id") != session_id]
            data[user_key] = sessions
            self._save_log(path, data)

    # -- Shutdown ---------------------------------------------------------------

    async def shutdown(self) -> None:
        if self.neo4j:
            try:
                await self.neo4j.disconnect()
            except Exception:
                pass
        observer.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY
# ═══════════════════════════════════════════════════════════════════════════════

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="consolidation_runner",
        description="Run agentic consolidation via Gemini CLI.",
    )
    p.add_argument("--gemini", default="gemini",
                    help="Path to gemini CLI binary (default: 'gemini' on PATH)")
    p.add_argument("--timeout", type=int, default=360,
                    help="Per-session agent timeout in seconds (default: 360). "
                         "Long sessions with 25+ turns may need 200-300s.")
    p.add_argument("--dry-run", action="store_true",
                    help="Produce plans but do not write to graph")
    p.add_argument("--session-id", default=None,
                    help="Process only this session ID")
    p.add_argument("--log", action="store_true",
                    help="Enable diagnostic logging to BRAIN/memory/data/logs/")
    p.add_argument("--check-only", action="store_true",
                    help="Run pre-flight checks only (no consolidation)")
    return p


async def _amain(args: argparse.Namespace) -> int:
    runner = ConsolidationRunner(
        gemini_cli_path=args.gemini,
        agent_timeout_s=args.timeout,
        dry_run=args.dry_run,
        log_enabled=args.log,
    )
    try:
        ok = await runner.preflight(skip_conversation_check=args.check_only)
        if not ok:
            return 1
        if args.check_only:
            print()
            print("[OK] All pre-flight checks passed. System is ready for consolidation.")
            print()
            return 0
        await runner.run(only_session_id=args.session_id)
        return 0
    finally:
        await runner.shutdown()


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
