"""
Event Hooks System for StockSquad

Enables agents to communicate and react to each other's actions via events.
"""

from hooks.event_bus import EventBus, AgentEvent

__all__ = ['EventBus', 'AgentEvent']
