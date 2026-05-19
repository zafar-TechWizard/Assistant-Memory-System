"""
SOFi Memory System — Typed Configuration

Single source of truth for all memory system settings.
Access via the module-level `config` singleton:

    from memory.config import config
    config.user_id
    config.neo4j_uri

Or via the factory function (backward-compatible with consolidation.py):

    from memory.config import get_config
    cfg = get_config()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class MemoryConfig:
    """
    Strongly-typed, attribute-accessible configuration for the SOFi memory system.
    All path properties are derived from the memory/ package root automatically.
    """

    # -------------------------------------------------------------------------
    # User Identity
    # -------------------------------------------------------------------------
    user_id: str = "Zafar"

    # -------------------------------------------------------------------------
    # Neo4j Database
    # -------------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "SofiAiAssistant"
    database: str = "neo4j"
    container_name: str = "sofi-neo4j-memory"
    neo4j_http_port: int = 7474
    neo4j_bolt_port: int = 7687

    # Docker / Health Check
    neo4j_health_check_max_attempts: int = 12
    neo4j_health_check_interval: int = 5   # seconds between poll attempts
    neo4j_post_start_wait: int = 10        # extra seconds after "Started" message

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # -------------------------------------------------------------------------
    # Working Memory
    # -------------------------------------------------------------------------
    entity_expiry_minutes: int = 15
    context_retrieval_timeout_ms: int = 300   # hard cap for get_context()
    max_memories_per_entity: int = 5
    max_total_memories: int = 50              # cap on whiteboard memory list
    enable_auto_cleanup: bool = True
    max_working_memory_turns: int = 10
    retrieval_threshold: float = 0.7

    # ── Working Context ────────────────────────────────────────────────────
    working_context_recent_turns: int = 5     # how many recent turns to include in MemoryState
    workspace_watcher_poll_interval_s: int = 5    # how often WorkspaceWatcher checks for flags
    workspace_watcher_gap_threshold_s: int = 30   # seconds of inactivity = "conversation gap"

    # -------------------------------------------------------------------------
    # Conversation Logging
    # -------------------------------------------------------------------------
    session_timeout_minutes: int = 30

    # -------------------------------------------------------------------------
    # Derived Paths
    #
    # CODE lives at:    <project>/memory/                  (this file's directory)
    # DATA lives at:    <project>/BRAIN/memory/data/       (operational state)
    #
    # All runtime artifacts (Neo4j graph files, conversation log, working
    # context, logs, review traces, dry-run plans) live under BRAIN/memory/
    # so the brain layer owns operational state cleanly while the code stays
    # in its own package.
    # -------------------------------------------------------------------------
    @property
    def _memory_root(self) -> Path:
        """Absolute path to the memory/ source-code package directory."""
        return Path(__file__).parent

    @property
    def _project_root(self) -> Path:
        """Absolute path to the project root (parent of memory/)."""
        return self._memory_root.parent

    @property
    def brain_dir(self) -> Path:
        """
        Absolute path to <project>/BRAIN/memory/.
        All operational runtime data lives under this directory.
        """
        p = self._project_root / "BRAIN" / "memory"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_dir(self) -> Path:
        """<project>/BRAIN/memory/data/ — created on first access."""
        p = self.brain_dir / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def conversation_log_path(self) -> Path:
        """<project>/BRAIN/memory/data/conversation.json"""
        return self.data_dir / "conversation.json"

    @property
    def context_file_path(self) -> Path:
        """<project>/BRAIN/memory/data/working_context.json"""
        return self.data_dir / "working_context.json"

    @property
    def neo4j_data_path(self) -> str:
        """
        <project>/BRAIN/memory/data/neo4j/ — Docker mounts this as /data
        inside the Neo4j container. Contains databases/, transactions/, dbms/.
        """
        p = self.data_dir / "neo4j"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    @property
    def logs_dir(self) -> Path:
        """<project>/BRAIN/memory/data/logs/ — observer log files."""
        p = self.data_dir / "logs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reviews_dir(self) -> Path:
        """<project>/BRAIN/memory/data/reviews/ — observer review traces."""
        p = self.data_dir / "reviews"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def consolidation_dry_runs_dir(self) -> Path:
        """<project>/BRAIN/memory/data/consolidation_dry_runs/ — agent plan dumps."""
        p = self.data_dir / "consolidation_dry_runs"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Legacy alias kept for backward compatibility
    @property
    def base_path(self) -> str:
        return str(self._project_root)

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------
    def ensure_directories(self) -> Dict[str, "Path"]:
        """
        Ensure every operational directory exists on disk. Called explicitly
        at startup so missing directories surface immediately rather than
        deep inside a writer that assumes its target exists.

        Returns a dict mapping logical names → resolved paths. Useful for
        confirming exactly what got created (the runner prints this).
        """
        # Accessing each property triggers its own mkdir(parents=True, exist_ok=True)
        return {
            "brain_dir":                  self.brain_dir,
            "data_dir":                   self.data_dir,
            "neo4j_data_path":            Path(self.neo4j_data_path),
            "logs_dir":                   self.logs_dir,
            "reviews_dir":                self.reviews_dir,
            "consolidation_dry_runs_dir": self.consolidation_dry_runs_dir,
        }


# ---------------------------------------------------------------------------
# Singleton instance — import this everywhere
# ---------------------------------------------------------------------------
config = MemoryConfig()


def get_config() -> MemoryConfig:
    """
    Factory function — returns the global config singleton.
    Backward-compatible with consolidation.py which imports get_config().
    """
    return config
