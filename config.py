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
    # Derived Paths  (computed from the memory/ package directory)
    # -------------------------------------------------------------------------
    @property
    def _memory_root(self) -> Path:
        """Absolute path to the memory/ package directory."""
        return Path(__file__).parent

    @property
    def data_dir(self) -> Path:
        """Absolute path to memory/data/ — created on first access."""
        p = self._memory_root / "data"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def conversation_log_path(self) -> Path:
        return self.data_dir / "conversation.json"

    @property
    def context_file_path(self) -> Path:
        return self.data_dir / "working_context.json"

    @property
    def neo4j_data_path(self) -> str:
        return str(self.data_dir / "neo4j_production")

    # Keep `base_path` for legacy code that reads config.get("base_path")
    @property
    def base_path(self) -> str:
        return str(self._memory_root.parent)


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
