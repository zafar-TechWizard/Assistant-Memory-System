from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
from sofi_memory.long_term.models.node_models import CurrentMemoryNode, MemoryContext

class ContextManager:
    """
    Manages the real-time "working memory" (Layer 1) for the AI assistant.

    This class holds and manipulates the CurrentMemoryNode in-memory for a single
    conversation session. It tracks the immediate focus, emotional state, and
    short-term history, providing a real-time context buffer that informs
    both the AI's responses and its long-term memory retrieval.
    """

    def __init__(self, user_id: str, session_id: str):
        """
        Initializes the working memory for a new conversation session.

        Args:
            user_id (str): The ID of the user in the conversation.
            session_id (str): The unique ID for this conversation session.
        """
        self.user_id = user_id
        self.session_id = session_id
        self.last_interaction_time = datetime.utcnow()
        
        # This is the AI's "conscious mind" for the session.
        self.current_context_node = CurrentMemoryNode(
            content="Initializing new conversation context.",
            current_focus="greeting",
            time_context=self._get_time_of_day(),
            current_mood=0.1,  # Start with a neutral, slightly positive mood
            stress_level=0.0
        )
        
        # A short-term buffer for the most recent conversation turns.
        self.short_term_history: List[Dict[str, str]] = []
        print(f"L1 ContextManager initialized for session {self.session_id}")

    def _get_time_of_day(self) -> str:
        """Determines the general time of day."""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 21:
            return 'evening'
        else:
            return 'night'

    def observe_message(self, role: str, content: str):
        """
        Observes a new message, updating the short-term history and interaction time.

        Args:
            role (str): The role of the speaker ('user' or 'assistant').
            content (str): The text content of the message.
        """
        self.short_term_history.append({"role": role, "content": content})
        # Keep the history buffer from growing too large
        if len(self.short_term_history) > 10:
            self.short_term_history.pop(0)
            
        self.last_interaction_time = datetime.utcnow()
        self.current_context_node.content = f"Last message from {role}: {content}"
        print(f"L1 observed message from {role}.")

    def update_focus(self, new_focus: str, related_entities: Optional[List[str]] = None):
        """
        Updates the AI's current focus, typically after an intent/entity analysis.

        Args:
            new_focus (str): A concise description of the current topic.
            related_entities (Optional[List[str]]): A list of key people, places, or
                                                   concepts mentioned recently.
        """
        print(f"L1 focus updated to: '{new_focus}'")
        self.current_context_node.current_focus = new_focus
        if related_entities:
            # This primes the retrieval engine with known entities.
            self.current_context_node.recent_relationships = related_entities

    def update_mood(self, mood_change: float, stress_change: float):
        """
        Adjusts the current emotional context of the conversation.

        Args:
            mood_change (float): A value to add to the current mood (-1.0 to 1.0).
            stress_change (float): A value to add to the current stress level (0.0 to 1.0).
        """
        # Clamp values to their defined ranges from the model
        new_mood = self.current_context_node.current_mood + mood_change
        self.current_context_node.current_mood = max(-1.0, min(1.0, new_mood))
        
        new_stress = self.current_context_node.stress_level + stress_change
        self.current_context_node.stress_level = max(0.0, min(1.0, new_stress))
        print(f"L1 mood updated: Mood={self.current_context_node.current_mood:.2f}, Stress={self.current_context_node.stress_level:.2f}")

    def get_current_context(self) -> CurrentMemoryNode:
        """Returns the live CurrentMemoryNode object."""
        return self.current_context_node

    def build_prompt_context(self) -> str:
        """
        Creates a concise string summary of the current working memory.
        This is designed to be injected directly into an LLM prompt.
        """
        history_str = "\n".join([f"{turn['role']}: {turn['content']}" for turn in self.short_term_history])
        
        context_summary = f"""
[START of Real-Time Context]
Current Focus: {self.current_context_node.current_focus}
Current Mood: {self.current_context_node.current_mood:.2f} (from -1 sad to 1 happy)
Current Stress: {self.current_context_node.stress_level:.2f} (from 0 calm to 1 stressed)
Recent Conversation History (last {len(self.short_term_history)} turns):
{history_str}
[END of Real-Time Context]
"""
        return context_summary

# --- Example Usage ---
if __name__ == '__main__':
    import time

    print("\n--- Testing the Layer 1 ContextManager ---")
    user_id = "zafar_001"
    session_id = f"session_{uuid4()}"

    # A conversation begins
    l1_manager = ContextManager(user_id, session_id)

    # User sends a message
    l1_manager.observe_message("user", "Hey, I'm feeling a bit stressed about this Python bug.")
    # In a real app, an NLP tool would analyze the sentiment
    l1_manager.update_focus("Debugging Python Script", related_entities=["Python"])
    l1_manager.update_mood(mood_change=-0.3, stress_change=0.4)

    # Assistant responds
    l1_manager.observe_message("assistant", "I understand. Debugging can be tricky. I'm here to help.")
    
    # Let's see the context we would send to the LLM for its *next* response
    prompt_context = l1_manager.build_prompt_context()
    print("\n--- Context to be injected into LLM Prompt: ---")
    print(prompt_context)

    # The user mentions his girlfriend, Akanksha
    l1_manager.observe_message("user", "Thanks. I was talking to Akanksha about it earlier.")
    l1_manager.update_focus("Discussing bug with Akanksha", related_entities=["Python", "Akanksha"])

    prompt_context_2 = l1_manager.build_prompt_context()
    print("\n--- Updated Context for LLM Prompt: ---")
    print(prompt_context_2)
