"""
Short-term memory implementation for agent session management.
Provides a shared scratchpad for agents to post and retrieve findings
during a single analysis run.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json


@dataclass
class Message:
    """A single message in the conversation history."""

    agent: str
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "agent": self.agent,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class ShortTermMemory:
    """
    Short-term memory manager for a single analysis session.

    Maintains:
    - Message history for the current run
    - Shared scratchpad where agents post intermediate findings
    - Session metadata (ticker, start time, etc.)
    """

    def __init__(self, ticker: str, session_id: Optional[str] = None):
        """
        Initialize short-term memory for a session.

        Args:
            ticker: Stock ticker being analyzed
            session_id: Optional unique identifier for this session
        """
        self.ticker = ticker.upper()
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.messages: List[Message] = []
        self.scratchpad: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "ticker": self.ticker,
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
        }

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        return f"{self.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def add_message(
        self,
        agent: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            agent: Name of the agent posting the message
            role: Message role (system/user/assistant)
            content: Message content
            metadata: Optional metadata dictionary
        """
        message = Message(
            agent=agent,
            role=role,
            content=content,
            metadata=metadata or {},
        )
        self.messages.append(message)

    def get_messages(
        self, agent: Optional[str] = None, role: Optional[str] = None
    ) -> List[Message]:
        """
        Retrieve messages, optionally filtered by agent or role.

        Args:
            agent: Filter by agent name
            role: Filter by role

        Returns:
            List of matching messages
        """
        messages = self.messages
        if agent:
            messages = [m for m in messages if m.agent == agent]
        if role:
            messages = [m for m in messages if m.role == role]
        return messages

    def get_conversation_history(self, max_messages: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history in a format suitable for LLM context.

        Args:
            max_messages: Limit to most recent N messages

        Returns:
            List of message dictionaries
        """
        messages = self.messages[-max_messages:] if max_messages else self.messages
        return [
            {"role": msg.role, "content": f"[{msg.agent}] {msg.content}"}
            for msg in messages
        ]

    def post_to_scratchpad(self, key: str, value: Any, agent: str) -> None:
        """
        Post data to the shared scratchpad.

        Args:
            key: Scratchpad key
            value: Data to store
            agent: Name of the agent posting the data
        """
        self.scratchpad[key] = {
            "value": value,
            "agent": agent,
            "timestamp": datetime.now().isoformat(),
        }

    def get_from_scratchpad(self, key: str) -> Optional[Any]:
        """
        Retrieve data from the scratchpad.

        Args:
            key: Scratchpad key

        Returns:
            Stored value or None if not found
        """
        entry = self.scratchpad.get(key)
        return entry["value"] if entry else None

    def get_scratchpad_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all scratchpad contents.

        Returns:
            Dictionary of scratchpad entries
        """
        return {k: v["value"] for k, v in self.scratchpad.items()}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the entire session to a dictionary.

        Returns:
            Dictionary representation of the session
        """
        return {
            "metadata": self.metadata,
            "messages": [msg.to_dict() for msg in self.messages],
            "scratchpad": self.scratchpad,
        }

    def to_json(self) -> str:
        """
        Serialize the session to JSON.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2)

    def clear(self) -> None:
        """Clear all messages and scratchpad data."""
        self.messages.clear()
        self.scratchpad.clear()
