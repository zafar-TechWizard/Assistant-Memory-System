import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Lock
from contextlib import contextmanager


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkingContextManager:
    """
    Manages reading and writing to the working_context.json file.
    Provides thread-safe operations and error handling.
    """
    
    def __init__(self, file_path: Path):
        """
        Initialize the context manager.
        
        Args:
            file_path: Path to the working_context.json file
        """
        self.file_path = file_path
        self._lock = Lock()
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Ensure the working context file exists with proper structure."""
        if not self.file_path.exists():
            logger.warning(f"Working context file not found. Creating: {self.file_path}")
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            initial_data = {
                "active_entities": {},
                "current_entities": [],
                "memories": []
            }
            
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
    
    @contextmanager
    def _file_lock(self):
        """Context manager for thread-safe file operations."""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
    
    def load(self) -> Dict[str, Any]:
        """
        Load working context from file.
        
        Returns:
            Dictionary containing the working context
            
        Raises:
            IOError: If file cannot be read
            json.JSONDecodeError: If file contains invalid JSON
        """
        with self._file_lock():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Ensure required keys exist
                if "active_entities" not in data:
                    data["active_entities"] = {}
                if "current_entities" not in data:
                    data["current_entities"] = []
                if "memories" not in data:
                    data["memories"] = []
                
                logger.debug(f"Loaded working context: {len(data.get('active_entities', {}))} active entities")
                return data
                
            except FileNotFoundError:
                logger.error(f"Working context file not found: {self.file_path}")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in working context file: {e}")
                raise
            except Exception as e:
                logger.error(f"Error loading working context: {e}")
                raise
    
    def save(self, data: Dict[str, Any]):
        """
        Save working context to file.
        
        Args:
            data: Dictionary containing the working context
            
        Raises:
            IOError: If file cannot be written
        """
        with self._file_lock():
            try:
                # Create backup before writing
                if self.file_path.exists():
                    backup_path = self.file_path.with_suffix('.json.bak')
                    with open(self.file_path, 'r', encoding='utf-8') as f:
                        backup_data = f.read()
                    with open(backup_path, 'w', encoding='utf-8') as f:
                        f.write(backup_data)
                
                # Write new data
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                logger.debug(f"Saved working context: {len(data.get('active_entities', {}))} active entities")
                
            except Exception as e:
                logger.error(f"Error saving working context: {e}")
                raise
