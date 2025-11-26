import json
import os
from datetime import datetime, timedelta
from uuid import uuid4

class ConversationLogger:
    """
    Manages logging conversations to a JSON file with session handling.
    """

    def __init__(self, user_id: str, filepath: str, session_timeout_minutes: int = 30):
        
        self.user = user_id
        self.filepath = filepath
        self.session_timeout = timedelta(minutes=session_timeout_minutes)

    def _load_data(self) -> dict:
        """
        Safely loads the JSON data from the file.
        Returns an empty dictionary if the file doesn't exist or is empty.
        """
        if not os.path.exists(self.filepath):
            return {}
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_data(self, data: dict):
        """Saves the given data to the JSON file with pretty printing."""
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

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

        """
        all_data = self._load_data()
        now = datetime.utcnow()

        # Get the user's conversation history, or create it if it doesn't exist
        user_key = f"user_{self.user}"  # FIX: Use consistent format with consolidation
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
                    print(f"Continuing existing session: {current_session['session_id']}")

        # If no active session, create a new one
        if not current_session:
            current_session = self._create_new_session(self.user)
            user_sessions.append(current_session)
            print(f"Starting new session: {current_session['session_id']}")

        # Create the new message "turn"
        new_turn = {
            "role": role,
            "content": content,
            "timestamp": now.isoformat() + "Z",
            "consolidated": False
        }

        # Add the new turn to the current session
        current_session['conversations'].append(new_turn)

        # Update the data and save it
        all_data[user_key] = user_sessions
        self._save_data(all_data)
        print(f"Logged message for {self.user} in role {role}.")


