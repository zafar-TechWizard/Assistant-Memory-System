
import asyncio
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

from memory.observability import observer


def _get_default_neo4j_config() -> Dict[str, Any]:
    """Build the default Neo4j Docker config from the central memory config (lazy)."""
    from memory.config import get_config
    return get_config().to_docker_config()


class DockerManager:
    """
    Manages Docker container lifecycle for the Neo4j database.

    Provides idempotent start/stop/monitor operations with health checking.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Neo4j Docker config dict. If None, derived lazily from
                    the global MemoryConfig at first use.
        """
        self.config = config if config is not None else _get_default_neo4j_config()
        self.container_name = self.config["container_name"]
        self.data_path = Path(self.config["data_path"])

    def _run_command(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if check and result.returncode != 0:
            # Redact NEO4J_AUTH before logging — never write passwords to disk.
            safe_cmd = [
                "NEO4J_AUTH=<redacted>" if "NEO4J_AUTH=" in arg else arg
                for arg in command
            ]
            observer.error("docker command failed", stderr=result.stderr, cmd=" ".join(safe_cmd))
            raise RuntimeError(f"Command failed: {result.stderr}")

        return result

    def is_docker_running(self) -> bool:
        """Check if Docker daemon is running."""
        try:
            result = self._run_command(["docker", "info"], check=False)
            return result.returncode == 0
        except Exception as e:
            observer.error("docker status check failed", exception=e)
            return False

    def container_exists(self) -> bool:
        """Check if the Neo4j container exists."""
        result = self._run_command(
            ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
            check=False,
        )
        return self.container_name in result.stdout

    def is_container_running(self) -> bool:
        """Check if the Neo4j container is running."""
        result = self._run_command(
            ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
            check=False,
        )
        return self.container_name in result.stdout

    def start_docker(self) -> None:
        """
        Start the Neo4j Docker container (idempotent).

        - Starts the container if it exists but is stopped.
        - Creates and starts a new container if it doesn't exist.
        - Does nothing if the container is already running.

        Raises:
            RuntimeError: If Docker is not running or the container fails to start.
        """
        observer.info("starting Neo4j container", name=self.container_name)

        if not self.is_docker_running():
            raise RuntimeError(
                "Docker is not running. Please start Docker Desktop and retry."
            )

        if self.is_container_running():
            observer.info("container already running", name=self.container_name)
            return

        self.data_path.mkdir(parents=True, exist_ok=True)

        if self.container_exists():
            self._run_command(["docker", "start", self.container_name])
            observer.info("existing container started", name=self.container_name)
        else:
            docker_command = [
                "docker", "run",
                "-d",
                "--name", self.container_name,
                "-p", f"{self.config['http_port']}:7474",
                "-p", f"{self.config['bolt_port']}:7687",
                "-v", f"{self.data_path}:/data",
                "-e", f"NEO4J_AUTH={self.config['username']}/{self.config['password']}",
                "-e", "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes",
                "neo4j:5.15",
            ]
            self._run_command(docker_command)
            observer.info("new container created", name=self.container_name)

    async def ensure_connection_async(self) -> None:
        """
        Wait for Neo4j to become ready (async, non-blocking).

        Runs blocking I/O (subprocess, HTTP probe) in a thread executor so
        the event loop is never frozen during the health-check loop.

        Raises:
            RuntimeError: If the container stops unexpectedly or never becomes
                          ready within the configured number of attempts.
        """
        loop = asyncio.get_running_loop()

        def _container_running() -> bool:
            return self.is_container_running()

        def _http_probe(port: int) -> bool:
            import urllib.request
            from urllib.error import URLError
            try:
                resp = urllib.request.urlopen(f"http://localhost:{port}/", timeout=2)
                return resp.getcode() == 200
            except (URLError, Exception):
                return False

        if not await loop.run_in_executor(None, _container_running):
            raise RuntimeError(f"Container '{self.container_name}' is not running")

        max_attempts = self.config["health_check_max_attempts"]
        interval     = self.config["health_check_interval"]
        http_port    = self.config.get("http_port", 7474)

        for attempt in range(1, max_attempts + 1):
            await asyncio.sleep(interval)

            if not await loop.run_in_executor(None, _container_running):
                raise RuntimeError(
                    f"Container '{self.container_name}' stopped unexpectedly"
                )

            if await loop.run_in_executor(None, _http_probe, http_port):
                await asyncio.sleep(self.config["post_start_wait"])
                observer.info("Neo4j ready", attempt=attempt)
                return

        raise RuntimeError(
            f"Neo4j did not become ready after {max_attempts} attempts "
            f"({max_attempts * interval}s). Check Docker logs: "
            f"`docker logs {self.container_name}`"
        )

    def ensure_connection(self) -> None:
        """Synchronous health-check (CLI / non-async callers only)."""
        import urllib.request
        from urllib.error import URLError

        if not self.is_container_running():
            raise RuntimeError(f"Container '{self.container_name}' is not running")

        max_attempts = self.config["health_check_max_attempts"]
        interval     = self.config["health_check_interval"]
        http_port    = self.config.get("http_port", 7474)

        for attempt in range(1, max_attempts + 1):
            time.sleep(interval)

            if not self.is_container_running():
                raise RuntimeError(f"Container '{self.container_name}' stopped unexpectedly")

            try:
                response = urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
                if response.getcode() == 200:
                    time.sleep(self.config["post_start_wait"])
                    observer.info("Neo4j ready", attempt=attempt)
                    return
            except (URLError, Exception):
                pass

        raise RuntimeError(
            f"Neo4j did not become ready after {max_attempts} attempts. "
            f"Check Docker logs: `docker logs {self.container_name}`"
        )

    def stop_docker(self) -> None:
        """Stop the Neo4j Docker container (idempotent)."""
        if self.is_container_running():
            self._run_command(["docker", "stop", self.container_name])
            observer.info("container stopped", name=self.container_name)

    def remove_container(self) -> None:
        """Remove the Neo4j Docker container (data volume is preserved)."""
        if self.container_exists():
            if self.is_container_running():
                self.stop_docker()
            self._run_command(["docker", "rm", self.container_name])
            observer.info("container removed", name=self.container_name)

    def get_status(self) -> Dict[str, Any]:
        """Return status information about the Docker container and Neo4j."""
        safe_config = {k: v for k, v in self.config.items() if k != "password"}
        status: Dict[str, Any] = {
            "docker_running":    self.is_docker_running(),
            "container_exists":  self.container_exists(),
            "container_running": self.is_container_running(),
            "data_path":         str(self.data_path),
            "data_path_exists":  self.data_path.exists(),
            "config":            safe_config,
        }

        if self.container_exists():
            result = self._run_command(
                ["docker", "inspect", self.container_name], check=False
            )
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)[0]
                status["container_state"]   = info["State"]
                status["container_created"] = info["Created"]

        return status

    def get_logs(self, tail: int = 50) -> str:
        """Get recent logs from the Neo4j container."""
        if not self.container_exists():
            return f"Container '{self.container_name}' does not exist"
        result = self._run_command(
            ["docker", "logs", "--tail", str(tail), self.container_name], check=False
        )
        return result.stdout if result.returncode == 0 else result.stderr


def create_docker_manager(config: Optional[Dict[str, Any]] = None) -> DockerManager:
    """Factory function to create a DockerManager instance."""
    return DockerManager(config)


if __name__ == "__main__":
    import asyncio as _asyncio

    observer.configure(log=True)
    manager = DockerManager()
    try:
        status = manager.get_status()
        observer.info(
            "docker status",
            docker_running=status["docker_running"],
            container_exists=status["container_exists"],
            container_running=status["container_running"],
            data_path=str(status["data_path"]),
        )
        manager.start_docker()
        manager.ensure_connection()
        cfg = _get_default_neo4j_config()
        observer.info(
            "Neo4j ready for connections",
            browser=f"http://localhost:{cfg['http_port']}",
            database=cfg["database"],
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_docker()
    except Exception as e:
        observer.error("docker_manager CLI failed", exception=e)
        raise
