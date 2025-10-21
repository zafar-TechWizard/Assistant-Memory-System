import json
import os
from datetime import datetime, timedelta
from uuid import uuid4

class ConversationLogger:
    """
    Manages logging conversations to a JSON file with session handling.

    This class reads and writes to a JSON file that stores conversations
    grouped by user and session. It automatically determines if a new
    message belongs to an existing session or starts a new one based on
    an inactivity timeout.
    """

    def __init__(self, filepath: str, session_timeout_minutes: int = 30):
        """
        Initializes the logger.

        Args:
            filepath (str): The path to the conversation.json file.
            session_timeout_minutes (int): The number of minutes of inactivity
                                           before a new session is created.
        """
        self.filepath = filepath
        self.session_timeout = timedelta(minutes=session_timeout_minutes)
        print(f"ConversationLogger initialized. Session timeout is {session_timeout_minutes} minutes.")

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

    def _create_new_session(self, user_id: str) -> dict:
        """Creates the JSON structure for a new session."""
        return {
            "session_id": f"session_{uuid4()}",
            "start_time": datetime.utcnow().isoformat() + "Z",
            "turns": []
        }

    def log_message(self, user_id: str, role: str, content: str):
        """
        Logs a new message for a user, handling session logic.

        Args:
            user_id (str): The unique identifier for the user.
            role (str): The role of the sender ('user' or 'assistant').
            content (str): The content of the message.
        """
        all_data = self._load_data()
        now = datetime.utcnow()

        # Get the user's conversation history, or create it if it doesn't exist
        user_sessions = all_data.get(user_id, [])

        current_session = None
        if user_sessions:
            # Get the last session to check for timeout
            last_session = user_sessions[-1]
            if last_session['turns']:
                last_message_time_str = last_session['turns'][-1]['timestamp']
                last_message_time = datetime.fromisoformat(last_message_time_str.replace('Z', '+00:00'))
                
                # Check if the last message is within the timeout window
                if now - last_message_time.replace(tzinfo=None) < self.session_timeout:
                    current_session = last_session
                    print(f"Continuing existing session: {current_session['session_id']}")

        # If no active session, create a new one
        if not current_session:
            current_session = self._create_new_session(user_id)
            user_sessions.append(current_session)
            print(f"Starting new session: {current_session['session_id']}")

        # Create the new message "turn"
        new_turn = {
            "turn_id": f"msg_{uuid4()}",
            "role": role,
            "content": content,
            "timestamp": now.isoformat() + "Z",
            "consolidated": False
        }

        # Add the new turn to the current session
        current_session['turns'].append(new_turn)

        # Update the data and save it
        all_data[user_id] = user_sessions
        self._save_data(all_data)
        print(f"Logged message for {user_id} in role {role}.")


# --- Example Usage ---
if __name__ == '__main__':
    import time

    # Create a logger instance pointing to a file in the same directory
    # For testing, we'll use a short 1-minute timeout
    logger = ConversationLogger('conversation.json', session_timeout_minutes=1)
    
    USER_ID = "zafar_001"

    print("\n--- Simulating a conversation ---")
    logger.log_message(USER_ID, 'user', "Hey, how are you?")
    time.sleep(2) # Wait 2 seconds
    logger.log_message(USER_ID, 'assistant', "I'm doing well! What's on your mind?")
    time.sleep(2)
    logger.log_message(USER_ID, 'user', "I'm working on that Python script again...")

    print("\n--- Simulating a pause longer than the session timeout (1 minute) ---")
    time.sleep(61) # Wait for 61 seconds

    print("\n--- Simulating a new conversation after the timeout ---")
    logger.log_message(USER_ID, 'user', "Okay, I'm back. I need help with something new.")
    time.sleep(2)
    logger.log_message(USER_ID, 'assistant', "Of course! A new session has started. What can I help you with?")

    print("\nCheck the 'conversation.json' file to see the result.")
    # To clean up the created file after testing:
    # if os.path.exists('conversation.json'):
    #     os.remove('conversation.json')
