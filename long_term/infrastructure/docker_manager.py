
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

from memory.config import config as _mem_cfg
from memory.observability import observer


def _build_default_neo4j_config() -> Dict[str, Any]:
    """
    Build the default Neo4j Docker config from the central memory config.
    The data_path is resolved at runtime to <project>/BRAIN/memory/data/neo4j/
    so that all SOFi state lives under BRAIN/memory/ as a single root.
    """
    return {
        "container_name": _mem_cfg.container_name,
        "database":       _mem_cfg.database,
        "username":       _mem_cfg.neo4j_username,
        "password":       _mem_cfg.neo4j_password,
        "data_path":      _mem_cfg.neo4j_data_path,
        "uri":            _mem_cfg.neo4j_uri,
        "http_port":      _mem_cfg.neo4j_http_port,
        "bolt_port":      _mem_cfg.neo4j_bolt_port,
        "health_check_max_attempts": _mem_cfg.neo4j_health_check_max_attempts,
        "health_check_interval":     _mem_cfg.neo4j_health_check_interval,
        "post_start_wait":           _mem_cfg.neo4j_post_start_wait,
    }


# SOFI Neo4j Configuration — derived from memory config at import time
SOFI_NEO4J_CONFIG = _build_default_neo4j_config()


class DockerManager:
    """
    Manages Docker container lifecycle for SOFI's Neo4j database.
    
    This class provides methods to start, stop, and monitor the Neo4j Docker
    container with SOFI's specific configuration. It ensures idempotent operations
    and proper health checking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Docker manager with SOFI configuration.
        
        Args:
            config: Optional custom configuration. Defaults to SOFI_NEO4J_CONFIG
        """
        self.config = config or SOFI_NEO4J_CONFIG
        self.container_name = self.config["container_name"]
        self.data_path = Path(self.config["data_path"])
        
    def _run_command(self, command: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result"""
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )

        if check and result.returncode != 0:
            observer.error("docker command failed", stderr=result.stderr, cmd=" ".join(command))
            raise RuntimeError(f"Command failed: {result.stderr}")

        return result

    def is_docker_running(self) -> bool:
        """Check if Docker daemon is running"""
        try:
            result = self._run_command(["docker", "info"], check=False)
            return result.returncode == 0
        except Exception as e:
            observer.error("docker status check failed", exception=e)
            return False
    
    def container_exists(self) -> bool:
        """Check if the SOFI Neo4j container exists"""
        result = self._run_command(
            ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
            check=False
        )
        return self.container_name in result.stdout
    
    def is_container_running(self) -> bool:
        """Check if the SOFI Neo4j container is running"""
        result = self._run_command(
            ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
            check=False
        )
        return self.container_name in result.stdout
    
    def start_docker(self) -> None:
        """
        Start the SOFI Neo4j Docker container.
        
        This method is idempotent - it will:
        - Start the container if it exists but is stopped
        - Create and start a new container if it doesn't exist
        - Do nothing if the container is already running
        
        Raises:
            RuntimeError: If Docker is not running or container fails to start
        """
        observer.info("starting SOFI Neo4j container", name=self.container_name)

        # Check if Docker is running
        if not self.is_docker_running():
            raise RuntimeError("Docker is not running. Please start Docker Desktop.")

        # Check if container is already running
        if self.is_container_running():
            observer.info("container already running", name=self.container_name)
            return

        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)

        # If container exists but is stopped, start it
        if self.container_exists():
            self._run_command(["docker", "start", self.container_name])
            observer.info("existing container started", name=self.container_name)
        else:
            # Create and start new container
            docker_command = [
                "docker", "run",
                "-d",  # Detached mode
                "--name", self.container_name,
                "-p", f"{self.config['http_port']}:7474",  # HTTP
                "-p", f"{self.config['bolt_port']}:7687",  # Bolt
                "-v", f"{self.data_path}:/data",  # Persistent data volume
                "-e", f"NEO4J_AUTH={self.config['username']}/{self.config['password']}",
                "-e", "NEO4J_ACCEPT_LICENSE_AGREEMENT=yes",
                "neo4j:latest"
            ]

            self._run_command(docker_command)
            observer.info("new container created", name=self.container_name)
    
    async def ensure_connection_async(self) -> None:
        """
        Async version of ensure_connection.

        Uses asyncio.sleep instead of time.sleep so the event loop is NOT
        blocked while waiting for Neo4j to become ready.
        """
        import asyncio as _asyncio
        import urllib.request
        from urllib.error import URLError

        if not self.is_container_running():
            raise RuntimeError(f"Container '{self.container_name}' is not running")

        max_attempts = self.config["health_check_max_attempts"]
        interval     = self.config["health_check_interval"]
        http_port    = self.config.get("http_port", 7474)

        for attempt in range(1, max_attempts + 1):
            await _asyncio.sleep(interval)

            if not self.is_container_running():
                raise RuntimeError(
                    f"Container '{self.container_name}' stopped unexpectedly"
                )

            try:
                # Use a short timeout for the request
                response = urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
                if response.getcode() == 200:
                    await _asyncio.sleep(self.config["post_start_wait"])
                    observer.info("Neo4j ready", attempt=attempt)
                    return
            except (URLError, Exception):
                pass

        observer.warning("Neo4j health check max wait reached")

    def ensure_connection(self) -> None:
        """
        Ensure Neo4j is ready to accept connections.
        
        This method performs health checks by polling the Neo4j HTTP API,
        then waits additional time for Neo4j to be fully ready.
        
        Raises:
            RuntimeError: If container is not running or Neo4j fails to start
        """
        import urllib.request
        from urllib.error import URLError

        if not self.is_container_running():
            raise RuntimeError(f"Container '{self.container_name}' is not running")

        max_attempts = self.config["health_check_max_attempts"]
        interval = self.config["health_check_interval"]
        http_port = self.config.get("http_port", 7474)

        for attempt in range(1, max_attempts + 1):
            time.sleep(interval)

            # Check if container is still running
            if not self.is_container_running():
                raise RuntimeError(f"Container '{self.container_name}' stopped unexpectedly")

            try:
                # Use a short timeout for the request
                response = urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
                if response.getcode() == 200:
                    time.sleep(self.config['post_start_wait'])
                    observer.info("Neo4j ready", attempt=attempt)
                    return
            except (URLError, Exception):
                pass

        observer.warning("Neo4j health check max wait reached")
    
    def stop_docker(self) -> None:
        """
        Stop the SOFI Neo4j Docker container.

        This method is idempotent - it will do nothing if the container
        is already stopped.
        """
        if self.is_container_running():
            self._run_command(["docker", "stop", self.container_name])
            observer.info("container stopped", name=self.container_name)

    def remove_container(self) -> None:
        """
        Remove the SOFI Neo4j Docker container.

        Note: This does NOT delete the data - data persists in the volume.
        """
        if self.container_exists():
            # Stop first if running
            if self.is_container_running():
                self.stop_docker()

            self._run_command(["docker", "rm", self.container_name])
            observer.info("container removed", name=self.container_name)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get detailed status information about the Docker container and Neo4j.
        
        Returns:
            Dictionary with status information
        """
        status = {
            "docker_running": self.is_docker_running(),
            "container_exists": self.container_exists(),
            "container_running": self.is_container_running(),
            "data_path": str(self.data_path),
            "data_path_exists": self.data_path.exists(),
            "config": self.config
        }
        
        # Get container details if it exists
        if self.container_exists():
            result = self._run_command(
                ["docker", "inspect", self.container_name],
                check=False
            )
            if result.returncode == 0:
                import json
                container_info = json.loads(result.stdout)[0]
                status["container_state"] = container_info["State"]
                status["container_created"] = container_info["Created"]
        
        return status
    
    def get_logs(self, tail: int = 50) -> str:
        """
        Get recent logs from the Neo4j container.
        
        Args:
            tail: Number of lines to retrieve
            
        Returns:
            Log output as string
        """
        if not self.container_exists():
            return f"Container '{self.container_name}' does not exist"
        
        result = self._run_command(
            ["docker", "logs", "--tail", str(tail), self.container_name],
            check=False
        )
        
        return result.stdout if result.returncode == 0 else result.stderr


# Convenience function for easy import
def create_docker_manager(config: Optional[Dict[str, Any]] = None) -> DockerManager:
    """
    Factory function to create a DockerManager instance.
    
    Args:
        config: Optional custom configuration
        
    Returns:
        DockerManager instance
    """
    return DockerManager(config)


# CLI for manual testing — writes diagnostic events to memory/data/logs/
if __name__ == "__main__":
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
        observer.info(
            "Neo4j ready for connections",
            browser=f"http://localhost:{SOFI_NEO4J_CONFIG['http_port']}",
            database=SOFI_NEO4J_CONFIG["database"],
        )

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            manager.stop_docker()

    except Exception as e:
        observer.error("docker_manager CLI failed", exception=e)
        raise
