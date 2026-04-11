
import subprocess
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# SOFI Neo4j Configuration
SOFI_NEO4J_CONFIG = {
    "container_name": "sofi-neo4j-memory",
    "database": "neo4j",
    "username": "neo4j",
    "password": "SofiAiAssistant",
    "data_path": r"C:\Users\mdzaf\OneDrive\Desktop\assistant\BRAIN\MEMORY",
    "uri": "bolt://localhost:7687",
    "http_port": 7474,
    "bolt_port": 7687,
    "health_check_max_attempts": 12,
    "health_check_interval": 5,  # seconds
    "post_start_wait": 10  # seconds to wait after "Started" message
}


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
        logger.debug(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False
        )
        
        if check and result.returncode != 0:
            logger.error(f"Command failed: {result.stderr}")
            raise RuntimeError(f"Command failed: {result.stderr}")
        
        return result
    
    def is_docker_running(self) -> bool:
        """Check if Docker daemon is running"""
        try:
            result = self._run_command(["docker", "info"], check=False)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to check Docker status: {e}")
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
        logger.info("Starting SOFI Neo4j Docker container...")
        
        # Check if Docker is running
        if not self.is_docker_running():
            raise RuntimeError("Docker is not running. Please start Docker Desktop.")
        
        # Check if container is already running
        if self.is_container_running():
            logger.info(f"Container '{self.container_name}' is already running")
            return
        
        # Create data directory if it doesn't exist
        self.data_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {self.data_path}")
        
        # If container exists but is stopped, start it
        if self.container_exists():
            logger.info(f"Starting existing container '{self.container_name}'...")
            self._run_command(["docker", "start", self.container_name])
            logger.info(f"Container '{self.container_name}' started")
        else:
            # Create and start new container
            logger.info(f"Creating new container '{self.container_name}'...")
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
            logger.info(f"Container '{self.container_name}' created and started")
    
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

        logger.info(
            f"[async] Waiting for Neo4j to be ready "
            f"(max {max_attempts * interval}s)..."
        )

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
                    logger.info(f"[async] Neo4j ready (attempt {attempt}/{max_attempts})")
                    logger.info(
                        f"[async] Waiting {self.config['post_start_wait']}s "
                        f"for full initialisation..."
                    )
                    await _asyncio.sleep(self.config["post_start_wait"])
                    logger.info("[async] Neo4j fully ready.")
                    return
            except (URLError, Exception):
                pass

            logger.debug(f"[async] Still waiting... ({attempt}/{max_attempts})")

        logger.warning(
            "[async] Max wait time reached — Neo4j may not be fully ready."
        )

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

        logger.info("Ensuring Neo4j is ready to accept connections...")
        
        if not self.is_container_running():
            raise RuntimeError(f"Container '{self.container_name}' is not running")
        
        # Health check loop
        max_attempts = self.config["health_check_max_attempts"]
        interval = self.config["health_check_interval"]
        http_port = self.config.get("http_port", 7474)
        
        logger.info(f"Waiting for Neo4j to be ready (max {max_attempts * interval} seconds)...")
        
        for attempt in range(1, max_attempts + 1):
            time.sleep(interval)
            
            # Check if container is still running
            if not self.is_container_running():
                raise RuntimeError(f"Container '{self.container_name}' stopped unexpectedly")
            
            try:
                # Use a short timeout for the request
                response = urllib.request.urlopen(f"http://localhost:{http_port}/", timeout=2)
                if response.getcode() == 200:
                    logger.info(f"Neo4j is ready! (took ~{attempt * interval} seconds)")
                    # Wait additional time for Neo4j to fully accept connections
                    logger.info(f"Waiting additional {self.config['post_start_wait']} seconds for full initialization...")
                    time.sleep(self.config['post_start_wait'])
                    logger.info("Neo4j is fully ready to accept connections")
                    return
            except (URLError, Exception):
                pass
            
            logger.debug(f"  Waiting... (attempt {attempt}/{max_attempts})")
        
        logger.warning("Max wait time reached, Neo4j may not be fully ready")
    
    def stop_docker(self) -> None:
        """
        Stop the SOFI Neo4j Docker container.
        
        This method is idempotent - it will do nothing if the container
        is already stopped.
        """
        logger.info("Stopping SOFI Neo4j Docker container...")
        
        if self.is_container_running():
            self._run_command(["docker", "stop", self.container_name])
            logger.info(f"Container '{self.container_name}' stopped")
        else:
            logger.info(f"Container '{self.container_name}' is not running")
    
    def remove_container(self) -> None:
        """
        Remove the SOFI Neo4j Docker container.
        
        Note: This does NOT delete the data - data persists in the volume.
        """
        logger.info("Removing SOFI Neo4j Docker container...")
        
        if self.container_exists():
            # Stop first if running
            if self.is_container_running():
                self.stop_docker()
            
            self._run_command(["docker", "rm", self.container_name])
            logger.info(f"Container '{self.container_name}' removed")
        else:
            logger.info(f"Container '{self.container_name}' does not exist")
    
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


# CLI for manual testing
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    manager = DockerManager()
    
    print("\n" + "="*80)
    print("SOFI Neo4j Docker Manager")
    print("="*80)
    
    try:
        # Get status
        print("\n1. Checking current status...")
        status = manager.get_status()
        print(f"   Docker running: {status['docker_running']}")
        print(f"   Container exists: {status['container_exists']}")
        print(f"   Container running: {status['container_running']}")
        print(f"   Data path: {status['data_path']}")
        
        # Start Docker
        print("\n2. Starting Docker container...")
        manager.start_docker()
        
        # Ensure connection
        print("\n3. Ensuring Neo4j is ready...")
        manager.ensure_connection()
        
        print("\n" + "="*80)
        print("SUCCESS! Neo4j is ready")
        print("="*80)
        print(f"\nNeo4j Browser: http://localhost:{SOFI_NEO4J_CONFIG['http_port']}")
        print(f"Username: {SOFI_NEO4J_CONFIG['username']}")
        print(f"Password: {SOFI_NEO4J_CONFIG['password']}")
        print(f"Database: {SOFI_NEO4J_CONFIG['database']}")
        print("\nPress Ctrl+C to stop the container...")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nStopping container...")
            manager.stop_docker()
            print("Container stopped.")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        print(f"\nError: {e}")
