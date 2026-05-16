import json
import os
import threading
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Optional
from memory.config import config
from memory.observability import observer


class ConversationLogger:
    """
    Manages logging conversations to a JSON file with session handling.
    Configuration is loaded from memory.config.
    """

    def __init__(self, user_id: Optional[str] = None, filepath: Optional[str] = None, session_timeout_minutes: Optional[int] = None):
        """
        Initialize ConversationLogger.
        
        Args:
            user_id: User ID (defaults to config value)
            filepath: Path to conversation log file (defaults to config value)
            session_timeout_minutes: Session timeout in minutes (defaults to config value)
        """
        # Set user ID
        self.user = user_id or getattr(config, "user_id", "default_user")
        
        # Set filepath
        if filepath:
            self.filepath = filepath
        else:
            self.filepath = str(getattr(config, "conversation_log_path", "memory/data/conversation.json"))
        
        # Set session timeout
        timeout_minutes = session_timeout_minutes or getattr(config, "session_timeout_minutes", 30)
        self.session_timeout = timedelta(minutes=timeout_minutes)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Thread-safety: prevents write race when two executor threads call
        # log_message() simultaneously (load-modify-save race condition).
        self._lock = threading.Lock()

        observer.info(
            "ConversationLogger initialized",
            user=self.user,
            filepath=self.filepath,
            timeout_min=timeout_minutes,
        )

    def _load_data(self) -> dict:
        """
        Safely loads the JSON data from the file.
        Returns an empty dictionary if the file doesn't exist or is empty.
        """
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            observer.warning("could not load conversation data", error=str(e))
            return {}

    def _save_data(self, data: dict):
        """Saves the given data to the JSON file with pretty printing."""
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            observer.error("conversation data save failed", exception=e)
            raise

    def _create_new_session(self) -> dict:
        """Creates the JSON structure for a new session."""
        return {
            "session_id": f"session_{uuid4()}",
            "start_time": datetime.utcnow().isoformat() + "Z",
            "conversations": []
        }

    def log_message(self, role: str, content: str):
        """
        Logs a new message for a user, handling session logic.

        Thread-safe: acquires _lock so concurrent calls from a ThreadPoolExecutor
        do not produce a load-modify-save race that silently drops messages.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
        """
        with self._lock:
            self._log_message_locked(role, content)

    def _log_message_locked(self, role: str, content: str):
        """Internal implementation — must only be called while _lock is held."""
        all_data = self._load_data()
        now = datetime.utcnow()

        # Get the user's conversation history, or create it if it doesn't exist
        user_key = f"user_{self.user}"
        user_sessions = all_data.get(user_key, [])

        current_session = None
        if user_sessions:
            # Get the last session to check for timeout
            last_session = user_sessions[-1]
            if last_session['conversations']:
                last_message_time_str = last_session['conversations'][-1]['timestamp']
                last_message_time = datetime.fromisoformat(last_message_time_str.replace('Z', '+00:00'))
                
                # Check if the last message is within the timeout window
                if now - last_message_time.replace(tzinfo=None) < self.session_timeout:
                    current_session = last_session

        # If no active session, create a new one
        if not current_session:
            current_session = self._create_new_session()
            user_sessions.append(current_session)
            observer.info("new conversation session", session_id=current_session["session_id"])

        # Create the new message "turn"
        new_turn = {
            "role": role,
            "content": content,
            "timestamp": now.isoformat() + "Z"
        }

        # Add the new turn to the current session
        current_session['conversations'].append(new_turn)

        # Update the data and save it
        all_data[user_key] = user_sessions
        self._save_data(all_data)
        # (lock released by context manager in log_message)
    
    def get_current_session(self) -> dict:
        """
        Get the current active session.
        
        Returns:
            Current session dictionary or None if no active session
        """
        all_data = self._load_data()
        user_key = f"user_{self.user}"
        user_sessions = all_data.get(user_key, [])
        
        if user_sessions:
            return user_sessions[-1]
        return None
    
    def get_conversation_history(self, max_turns: Optional[int] = None) -> list:
        """
        Get conversation history from current session.
        
        Args:
            max_turns: Maximum number of turns to return (None for all)
            
        Returns:
            List of conversation turns
        """
        session = self.get_current_session()
        if not session:
            return []
        
        conversations = session.get('conversations', [])
        
        if max_turns:
            return conversations[-max_turns:]
        return conversations
