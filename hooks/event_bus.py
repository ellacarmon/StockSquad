"""
Event Bus for Agent Communication

Provides a publish-subscribe pattern for agents to communicate with each other
without tight coupling.
"""

from typing import Callable, Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


class AgentEvent(Enum):
    """
    Types of events that agents can emit and subscribe to.

    Events enable loose coupling between agents - an agent can react to
    another agent's findings without directly depending on it.
    """

    # Analysis lifecycle events
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_COMPLETE = "analysis_complete"
    ANALYSIS_FAILED = "analysis_failed"

    # Signal and recommendation events
    SIGNAL_GENERATED = "signal_generated"
    RECOMMENDATION_MADE = "recommendation_made"

    # Data and insight events
    DATA_COLLECTED = "data_collected"
    INSIGHT_DISCOVERED = "insight_discovered"
    RISK_DETECTED = "risk_detected"
    OPPORTUNITY_DETECTED = "opportunity_detected"

    # Conflict and consensus events
    CONFLICT_DETECTED = "conflict_detected"
    CONSENSUS_REACHED = "consensus_reached"

    # Memory events
    MEMORY_STORED = "memory_stored"
    MEMORY_RETRIEVED = "memory_retrieved"


@dataclass
class Event:
    """
    Container for event data.

    Attributes:
        event_type: Type of event
        source_agent: Name of agent that emitted the event
        ticker: Stock ticker related to this event (if applicable)
        data: Event-specific data payload
        timestamp: When the event was created
    """

    event_type: AgentEvent
    source_agent: str
    ticker: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class EventBus:
    """
    Global event bus for agent communication.

    Agents can publish events when they discover insights, and other agents
    can subscribe to react to those events. This enables collaborative
    analysis without tight coupling.

    Example:
        # Agent publishes an event
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL',
            data={'signal': 'strong_buy', 'score': 85}
        )

        # Another agent subscribes to react
        def on_signal(event):
            if event.data['signal'] == 'strong_buy':
                # Challenge the bullish case
                analyze_risks(event.ticker)

        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, on_signal)
    """

    _listeners: Dict[AgentEvent, List[Callable[[Event], None]]] = {}
    _event_history: List[Event] = []
    _max_history: int = 1000  # Keep last 1000 events

    @classmethod
    def subscribe(
        cls,
        event_type: AgentEvent,
        callback: Callable[[Event], None],
        agent_name: Optional[str] = None
    ) -> None:
        """
        Subscribe to an event type.

        Args:
            event_type: Type of event to listen for
            callback: Function to call when event is published.
                     Should accept an Event object as parameter.
            agent_name: Optional name of subscribing agent (for logging)
        """
        if event_type not in cls._listeners:
            cls._listeners[event_type] = []

        cls._listeners[event_type].append(callback)

        subscriber_name = agent_name or callback.__name__
        logger.debug(
            f"Subscribed {subscriber_name} to {event_type.value}"
        )

    @classmethod
    def unsubscribe(
        cls,
        event_type: AgentEvent,
        callback: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from an event type.

        Args:
            event_type: Type of event to stop listening for
            callback: The callback function to remove
        """
        if event_type in cls._listeners:
            try:
                cls._listeners[event_type].remove(callback)
                logger.debug(f"Unsubscribed from {event_type.value}")
            except ValueError:
                logger.warning(
                    f"Callback not found in {event_type.value} listeners"
                )

    @classmethod
    def publish(
        cls,
        event_type: AgentEvent,
        source_agent: str,
        ticker: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event_type: Type of event to publish
            source_agent: Name of agent publishing the event
            ticker: Stock ticker related to this event (if applicable)
            data: Event-specific data payload
        """
        # Create event object
        event = Event(
            event_type=event_type,
            source_agent=source_agent,
            ticker=ticker,
            data=data or {}
        )

        # Add to history
        cls._event_history.append(event)
        if len(cls._event_history) > cls._max_history:
            cls._event_history.pop(0)

        # Notify subscribers
        if event_type in cls._listeners:
            logger.debug(
                f"Publishing {event_type.value} from {source_agent} "
                f"to {len(cls._listeners[event_type])} subscriber(s)"
            )

            for callback in cls._listeners[event_type]:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(
                        f"Error in event handler {callback.__name__} "
                        f"for {event_type.value}: {e}",
                        exc_info=True
                    )
        else:
            logger.debug(
                f"Publishing {event_type.value} from {source_agent} "
                f"(no subscribers)"
            )

    @classmethod
    def get_history(
        cls,
        event_type: Optional[AgentEvent] = None,
        ticker: Optional[str] = None,
        source_agent: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Event]:
        """
        Get event history with optional filtering.

        Args:
            event_type: Filter by event type
            ticker: Filter by ticker
            source_agent: Filter by source agent
            limit: Maximum number of events to return (most recent)

        Returns:
            List of Event objects matching the filters
        """
        events = cls._event_history

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if ticker:
            events = [e for e in events if e.ticker == ticker]
        if source_agent:
            events = [e for e in events if e.source_agent == source_agent]

        # Apply limit (most recent)
        if limit:
            events = events[-limit:]

        return events

    @classmethod
    def clear_history(cls) -> None:
        """Clear event history. Useful for testing."""
        cls._event_history.clear()
        logger.debug("Cleared event history")

    @classmethod
    def clear_listeners(cls) -> None:
        """Clear all event listeners. Useful for testing."""
        cls._listeners.clear()
        logger.debug("Cleared all event listeners")

    @classmethod
    def get_listener_count(cls, event_type: AgentEvent) -> int:
        """
        Get the number of subscribers for an event type.

        Args:
            event_type: Event type to check

        Returns:
            Number of subscribers
        """
        return len(cls._listeners.get(event_type, []))

    @classmethod
    def list_event_types(cls) -> List[str]:
        """
        List all event types that have subscribers.

        Returns:
            List of event type names
        """
        return [event_type.value for event_type in cls._listeners.keys()]


# Convenience decorator for subscribing to events
def on_event(event_type: AgentEvent, agent_name: Optional[str] = None):
    """
    Decorator for subscribing a method to an event.

    Example:
        class MyAgent(BaseAgent):
            @on_event(AgentEvent.SIGNAL_GENERATED, agent_name='MyAgent')
            def handle_signal(self, event):
                print(f"Signal received: {event.data}")

    Args:
        event_type: Type of event to subscribe to
        agent_name: Optional name of subscribing agent

    Returns:
        Decorator function
    """
    def decorator(func: Callable[[Event], None]) -> Callable[[Event], None]:
        EventBus.subscribe(event_type, func, agent_name)
        return func
    return decorator
