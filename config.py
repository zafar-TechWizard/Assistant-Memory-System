"""
Memory System — Typed Configuration

Single source of truth for all memory system settings.
Access via the module-level `config` singleton:

    from memory.config import config
    config.user_id
    config.neo4j_uri

Or construct with explicit values (useful for multi-tenant or testing):

    from memory.config import MemoryConfig
    cfg = MemoryConfig(user_id="alice", neo4j_password="secret")

Or override via environment variables (picked up automatically by the singleton):

    MEMORY_USER_ID        — required: identity of the memory owner
    NEO4J_PASSWORD        — required: Neo4j database password
    MEMORY_BASE_DIR       — optional: root data directory (default: ~/.memory)
    NEO4J_CONTAINER_NAME  — optional: Docker container name (default: memory-neo4j)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class MemoryConfig:
    """
    Strongly-typed, attribute-accessible configuration for the memory system.

    Required fields (no hardcoded defaults — must come from env vars or constructor args):
        user_id         identity of the memory owner
        neo4j_password  Neo4j database password

    Optional fields (safe defaults provided):
        base_dir        root data directory; all paths derive from here
        container_name  Docker container name for Neo4j
        ... (all other fields have sensible defaults)
    """

    # -------------------------------------------------------------------------
    # Identity — REQUIRED (no hardcoded default)
    # -------------------------------------------------------------------------
    user_id: str = field(default="")

    # Name the assistant calls itself and the JSON section key in working_context.json.
    # Defaults to "assistant" — set to your assistant's name (e.g. "aria", "nova").
    assistant_name: str = "assistant"

    # -------------------------------------------------------------------------
    # Neo4j Database
    # -------------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = field(default="")   # REQUIRED — no default
    database: str = "neo4j"
    container_name: str = field(default="")   # resolved in __post_init__
    neo4j_http_port: int = 7474
    neo4j_bolt_port: int = 7687

    # Connection pool
    # Bolt driver maintains a persistent connection pool. 30 is generous for a
    # single-user assistant; lower it (e.g. 10) if memory footprint matters.
    neo4j_max_connection_pool_size: int = 30

    # Docker / Health Check
    neo4j_health_check_max_attempts: int = 60
    neo4j_health_check_interval: int = 1   # seconds between poll attempts
    neo4j_post_start_wait: int = 2         # extra seconds after "Started" message

    # -------------------------------------------------------------------------
    # Storage Root
    # -------------------------------------------------------------------------
    base_dir: Optional[Path] = field(default=None)  # resolved in __post_init__

    # -------------------------------------------------------------------------
    # Embedding
    # -------------------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"

    # -------------------------------------------------------------------------
    # Working Memory
    # -------------------------------------------------------------------------
    entity_expiry_minutes: int = 15
    context_retrieval_timeout_ms: int = 1500
    # Why 1500ms: warm-state retrieval is 110-270ms, but the FIRST call after
    # setup pays cold-start cost for cross-encoder JIT (~200ms), Neo4j first
    # query plan (~150ms) and entity extractor warm-up (~200-400ms). Hard cap
    # must accommodate the cold path, not just the warm path. The router itself
    # gets 80% of this. (Verified empirically via L3 end-to-end test.)
    max_memories_per_entity: int = 5
    max_total_memories: int = 50
    enable_auto_cleanup: bool = True
    max_working_memory_turns: int = 10
    retrieval_threshold: float = 0.7

    # Working Context
    working_context_recent_turns: int = 5
    workspace_watcher_poll_interval_s: int = 5
    workspace_watcher_gap_threshold_s: int = 30

    # -------------------------------------------------------------------------
    # Conversation Logging
    # -------------------------------------------------------------------------
    session_timeout_minutes: int = 30

    # -------------------------------------------------------------------------
    # Initialization — resolve env vars and validate
    # -------------------------------------------------------------------------

    def __post_init__(self):
        # Resolve user_id — required, no silent fallback
        if not self.user_id:
            self.user_id = os.environ.get("MEMORY_USER_ID", "")
            if not self.user_id:
                raise ValueError(
                    "user_id is required. Set the MEMORY_USER_ID environment variable "
                    "or pass user_id= to MemoryConfig()."
                )

        # Resolve neo4j_password — required, never hardcoded
        if not self.neo4j_password:
            self.neo4j_password = os.environ.get("NEO4J_PASSWORD", "")
            if not self.neo4j_password:
                raise ValueError(
                    "neo4j_password is required. Set the NEO4J_PASSWORD environment variable "
                    "or pass neo4j_password= to MemoryConfig()."
                )

        # Resolve assistant_name — no validation needed, has a safe default
        if self.assistant_name == "assistant":
            env_name = os.environ.get("MEMORY_ASSISTANT_NAME")
            if env_name:
                self.assistant_name = env_name

        # Resolve container name — safe default, no product-specific branding
        if not self.container_name:
            self.container_name = os.environ.get("NEO4J_CONTAINER_NAME", "memory-neo4j")

        # Resolve base_dir — all data paths derive from this single root
        if self.base_dir is None:
            env_base = os.environ.get("MEMORY_BASE_DIR")
            self.base_dir = Path(env_base) if env_base else Path.home() / ".memory"
        else:
            self.base_dir = Path(self.base_dir)
        self.base_dir = self.base_dir.expanduser().resolve()

    # =========================================================================
    # Derived Paths — all derived from base_dir/BRAIN/
    #
    # Directory layout:
    #
    #   <base_dir>/
    #     BRAIN/
    #       runtime/          ← operational state (session data, logs, tasks)
    #         session/
    #           conversation.json
    #           working_context.json
    #         tasks/
    #         logs/
    #       data/             ← learning data (Neo4j, reviews, consolidation)
    #         neo4j/          ← Docker-mounted Neo4j data volume
    #         reviews/
    #         consolidation_dry_runs/
    # =========================================================================

    @property
    def _brain_root(self) -> Path:
        """<base_dir>/BRAIN/ — single root for all memory data."""
        return self.base_dir / "BRAIN"

    @property
    def runtime_dir(self) -> Path:
        """<base_dir>/BRAIN/runtime/ — all operational state."""
        return self._brain_root / "runtime"

    @property
    def session_dir(self) -> Path:
        """<base_dir>/BRAIN/runtime/session/ — per-session JSON state files."""
        return self.runtime_dir / "session"

    @property
    def conversation_log_path(self) -> Path:
        """<base_dir>/BRAIN/runtime/session/conversation.json"""
        return self.session_dir / "conversation.json"

    @property
    def context_file_path(self) -> Path:
        """<base_dir>/BRAIN/runtime/session/working_context.json"""
        return self.session_dir / "working_context.json"

    @property
    def tasks_dir(self) -> Path:
        """<base_dir>/BRAIN/runtime/tasks/ — delegate task record files."""
        return self.runtime_dir / "tasks"

    @property
    def logs_dir(self) -> Path:
        """<base_dir>/BRAIN/runtime/logs/ — daily logs + metrics."""
        return self.runtime_dir / "logs"

    @property
    def data_dir(self) -> Path:
        """<base_dir>/BRAIN/data/ — memory system's learning data."""
        return self._brain_root / "data"

    @property
    def neo4j_data_path(self) -> Path:
        """
        <base_dir>/BRAIN/data/neo4j/ — Docker mounts this as /data inside the
        Neo4j container. Contains databases/, transactions/, dbms/.
        """
        return self.data_dir / "neo4j"

    @property
    def reviews_dir(self) -> Path:
        """<base_dir>/BRAIN/data/reviews/ — observer review traces."""
        return self.data_dir / "reviews"

    @property
    def consolidation_dry_runs_dir(self) -> Path:
        """<base_dir>/BRAIN/data/consolidation_dry_runs/ — agent plan dumps."""
        return self.data_dir / "consolidation_dry_runs"

    # =========================================================================
    # Setup
    # =========================================================================

    def to_docker_config(self) -> Dict[str, Any]:
        """Build the Docker config dict expected by DockerManager."""
        return {
            "container_name":            self.container_name,
            "database":                  self.database,
            "username":                  self.neo4j_username,
            "password":                  self.neo4j_password,
            "data_path":                 str(self.neo4j_data_path),
            "uri":                       self.neo4j_uri,
            "http_port":                 self.neo4j_http_port,
            "bolt_port":                 self.neo4j_bolt_port,
            "health_check_max_attempts": self.neo4j_health_check_max_attempts,
            "health_check_interval":     self.neo4j_health_check_interval,
            "post_start_wait":           self.neo4j_post_start_wait,
        }

    def ensure_directories(self) -> Dict[str, Path]:
        """
        Ensure every operational directory exists on disk. Called once at
        startup from MemoryManager.setup() — path properties are pure
        computations and do NOT create directories themselves.

        Returns a dict mapping logical names → resolved paths.
        """
        dirs = {
            "runtime_dir":                self.runtime_dir,
            "session_dir":                self.session_dir,
            "tasks_dir":                  self.tasks_dir,
            "logs_dir":                   self.logs_dir,
            "data_dir":                   self.data_dir,
            "neo4j_data_path":            self.neo4j_data_path,
            "reviews_dir":                self.reviews_dir,
            "consolidation_dry_runs_dir": self.consolidation_dry_runs_dir,
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        return dirs


# ---------------------------------------------------------------------------
# Lazy singleton — deferred until first attribute access so that
# `from memory import MemoryManager` does not raise ValueError in
# environments where MEMORY_USER_ID / NEO4J_PASSWORD are not set as
# env vars (the constructor-args path sets them explicitly).
# ---------------------------------------------------------------------------

_config: Optional[MemoryConfig] = None


def get_config() -> MemoryConfig:
    """Build (or return the cached) global config singleton."""
    global _config
    if _config is None:
        _config = MemoryConfig()
    return _config


class _LazyConfigProxy:
    """
    Proxy for the global MemoryConfig singleton.

    Attribute access is forwarded to `get_config()`, so the real
    MemoryConfig is not constructed until something actually reads it.
    Code that does `from memory.config import config` gets this proxy;
    it behaves identically to a real MemoryConfig instance at runtime.
    """
    def __getattr__(self, name: str):
        return getattr(get_config(), name)


config = _LazyConfigProxy()
