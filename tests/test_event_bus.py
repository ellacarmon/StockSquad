"""
Tests for the Event Bus System

Tests EventBus, AgentEvent, and event-driven agent communication.
"""

import pytest
from datetime import datetime

from hooks.event_bus import EventBus, AgentEvent, Event


class TestEvent:
    """Test Event dataclass"""

    def test_event_creation(self):
        """Test creating an event"""
        event = Event(
            event_type=AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL',
            data={'signal': 'buy', 'score': 85}
        )

        assert event.event_type == AgentEvent.SIGNAL_GENERATED
        assert event.source_agent == 'TechnicalAgent'
        assert event.ticker == 'AAPL'
        assert event.data['signal'] == 'buy'
        assert isinstance(event.timestamp, datetime)

    def test_event_auto_timestamp(self):
        """Test that timestamp is auto-generated"""
        event = Event(
            event_type=AgentEvent.ANALYSIS_COMPLETE,
            source_agent='DataAgent'
        )
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)


class TestEventBus:
    """Test EventBus class"""

    def setup_method(self):
        """Clear event bus before each test"""
        EventBus.clear_history()
        EventBus.clear_listeners()

    def teardown_method(self):
        """Clear event bus after each test"""
        EventBus.clear_history()
        EventBus.clear_listeners()

    def test_publish_event(self):
        """Test publishing an event"""
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL',
            data={'signal': 'buy'}
        )

        history = EventBus.get_history()
        assert len(history) == 1
        assert history[0].event_type == AgentEvent.SIGNAL_GENERATED
        assert history[0].source_agent == 'TechnicalAgent'

    def test_subscribe_and_publish(self):
        """Test subscribing to and receiving events"""
        received_events = []

        def callback(event: Event):
            received_events.append(event)

        # Subscribe
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback)

        # Publish
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL',
            data={'signal': 'buy'}
        )

        # Verify callback was called
        assert len(received_events) == 1
        assert received_events[0].ticker == 'AAPL'
        assert received_events[0].data['signal'] == 'buy'

    def test_multiple_subscribers(self):
        """Test multiple subscribers to the same event"""
        callback1_called = []
        callback2_called = []

        def callback1(event: Event):
            callback1_called.append(event)

        def callback2(event: Event):
            callback2_called.append(event)

        # Subscribe both
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback1)
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback2)

        # Publish
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL'
        )

        # Both callbacks should be called
        assert len(callback1_called) == 1
        assert len(callback2_called) == 1

    def test_unsubscribe(self):
        """Test unsubscribing from events"""
        received_events = []

        def callback(event: Event):
            received_events.append(event)

        # Subscribe
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback)

        # Publish - should receive
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL'
        )
        assert len(received_events) == 1

        # Unsubscribe
        EventBus.unsubscribe(AgentEvent.SIGNAL_GENERATED, callback)

        # Publish again - should not receive
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='MSFT'
        )
        assert len(received_events) == 1  # Still 1, not 2

    def test_event_history(self):
        """Test event history tracking"""
        # Publish multiple events
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent1', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'Agent2', 'MSFT')
        EventBus.publish(AgentEvent.RISK_DETECTED, 'Agent3', 'NVDA')

        history = EventBus.get_history()
        assert len(history) == 3

    def test_filter_history_by_event_type(self):
        """Test filtering history by event type"""
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent1', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'Agent2', 'MSFT')
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent3', 'NVDA')

        signals = EventBus.get_history(event_type=AgentEvent.SIGNAL_GENERATED)
        assert len(signals) == 2
        assert all(e.event_type == AgentEvent.SIGNAL_GENERATED for e in signals)

    def test_filter_history_by_ticker(self):
        """Test filtering history by ticker"""
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent1', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'Agent2', 'AAPL')
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent3', 'NVDA')

        aapl_events = EventBus.get_history(ticker='AAPL')
        assert len(aapl_events) == 2
        assert all(e.ticker == 'AAPL' for e in aapl_events)

    def test_filter_history_by_source_agent(self):
        """Test filtering history by source agent"""
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'TechnicalAgent', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'DataAgent', 'MSFT')
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'TechnicalAgent', 'NVDA')

        tech_events = EventBus.get_history(source_agent='TechnicalAgent')
        assert len(tech_events) == 2
        assert all(e.source_agent == 'TechnicalAgent' for e in tech_events)

    def test_history_limit(self):
        """Test limiting history results"""
        for i in range(10):
            EventBus.publish(AgentEvent.SIGNAL_GENERATED, f'Agent{i}', 'AAPL')

        # Get only last 3
        history = EventBus.get_history(limit=3)
        assert len(history) == 3
        # Should be most recent (Agent7, Agent8, Agent9)
        assert history[0].source_agent == 'Agent7'
        assert history[-1].source_agent == 'Agent9'

    def test_clear_history(self):
        """Test clearing event history"""
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent1', 'AAPL')
        assert len(EventBus.get_history()) == 1

        EventBus.clear_history()
        assert len(EventBus.get_history()) == 0

    def test_clear_listeners(self):
        """Test clearing all listeners"""
        def callback(event: Event):
            pass

        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback)
        assert EventBus.get_listener_count(AgentEvent.SIGNAL_GENERATED) == 1

        EventBus.clear_listeners()
        assert EventBus.get_listener_count(AgentEvent.SIGNAL_GENERATED) == 0

    def test_get_listener_count(self):
        """Test getting listener count"""
        def callback1(event: Event):
            pass

        def callback2(event: Event):
            pass

        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback1)
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback2)

        count = EventBus.get_listener_count(AgentEvent.SIGNAL_GENERATED)
        assert count == 2

    def test_list_event_types(self):
        """Test listing event types with subscribers"""
        def callback(event: Event):
            pass

        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, callback)
        EventBus.subscribe(AgentEvent.RISK_DETECTED, callback)

        event_types = EventBus.list_event_types()
        assert 'signal_generated' in event_types
        assert 'risk_detected' in event_types

    def test_error_in_callback_doesnt_break_bus(self):
        """Test that errors in callbacks don't break the event bus"""
        successful_callback_called = []

        def failing_callback(event: Event):
            raise ValueError("Intentional error")

        def successful_callback(event: Event):
            successful_callback_called.append(event)

        # Subscribe both callbacks
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, failing_callback)
        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, successful_callback)

        # Publish - failing callback should not prevent successful callback
        EventBus.publish(AgentEvent.SIGNAL_GENERATED, 'Agent', 'AAPL')

        # Successful callback should still be called
        assert len(successful_callback_called) == 1


class TestAgentCommunication:
    """Integration tests for agent-to-agent communication via events"""

    def setup_method(self):
        """Clear event bus before each test"""
        EventBus.clear_history()
        EventBus.clear_listeners()

    def teardown_method(self):
        """Clear event bus after each test"""
        EventBus.clear_history()
        EventBus.clear_listeners()

    def test_devils_advocate_reacts_to_signal(self):
        """Test Devil's Advocate reacting to strong buy signal"""
        devils_advocate_reactions = []

        # Devil's Advocate subscribes to signals
        def on_signal_generated(event: Event):
            if event.data.get('signal') == 'strong_buy':
                # Challenge the bullish case
                devils_advocate_reactions.append({
                    'ticker': event.ticker,
                    'action': 'challenge_bullish_case',
                    'original_score': event.data['score']
                })

        EventBus.subscribe(AgentEvent.SIGNAL_GENERATED, on_signal_generated)

        # TechnicalAgent publishes strong buy signal
        EventBus.publish(
            AgentEvent.SIGNAL_GENERATED,
            source_agent='TechnicalAgent',
            ticker='AAPL',
            data={'signal': 'strong_buy', 'score': 95}
        )

        # Devil's Advocate should have reacted
        assert len(devils_advocate_reactions) == 1
        assert devils_advocate_reactions[0]['ticker'] == 'AAPL'
        assert devils_advocate_reactions[0]['action'] == 'challenge_bullish_case'

    def test_orchestrator_tracks_analysis_progress(self):
        """Test Orchestrator tracking analysis progress via events"""
        analysis_progress = {}

        def track_progress(event: Event):
            ticker = event.ticker
            if ticker not in analysis_progress:
                analysis_progress[ticker] = []
            analysis_progress[ticker].append({
                'agent': event.source_agent,
                'event': event.event_type.value
            })

        # Orchestrator subscribes to analysis events
        EventBus.subscribe(AgentEvent.ANALYSIS_STARTED, track_progress)
        EventBus.subscribe(AgentEvent.ANALYSIS_COMPLETE, track_progress)

        # Agents publish events
        EventBus.publish(AgentEvent.ANALYSIS_STARTED, 'DataAgent', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'DataAgent', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_STARTED, 'TechnicalAgent', 'AAPL')
        EventBus.publish(AgentEvent.ANALYSIS_COMPLETE, 'TechnicalAgent', 'AAPL')

        # Orchestrator should have tracked all events
        assert len(analysis_progress['AAPL']) == 4
        assert analysis_progress['AAPL'][0]['agent'] == 'DataAgent'
        assert analysis_progress['AAPL'][0]['event'] == 'analysis_started'

    def test_conflict_detection_between_agents(self):
        """Test detecting conflicts between agent recommendations"""
        conflicts_detected = []

        agent_recommendations = {}

        def track_recommendation(event: Event):
            ticker = event.ticker
            agent = event.source_agent
            recommendation = event.data['recommendation']

            if ticker not in agent_recommendations:
                agent_recommendations[ticker] = {}

            agent_recommendations[ticker][agent] = recommendation

            # Check for conflicts
            recommendations = agent_recommendations[ticker]
            if len(recommendations) >= 2:
                unique_recs = set(recommendations.values())
                if len(unique_recs) > 1:
                    # Conflict detected
                    conflicts_detected.append({
                        'ticker': ticker,
                        'recommendations': recommendations
                    })
                    EventBus.publish(
                        AgentEvent.CONFLICT_DETECTED,
                        source_agent='ConflictDetector',
                        ticker=ticker,
                        data={'recommendations': recommendations}
                    )

        EventBus.subscribe(AgentEvent.RECOMMENDATION_MADE, track_recommendation)

        # Agents make conflicting recommendations
        EventBus.publish(
            AgentEvent.RECOMMENDATION_MADE,
            'TechnicalAgent',
            'AAPL',
            data={'recommendation': 'buy'}
        )
        EventBus.publish(
            AgentEvent.RECOMMENDATION_MADE,
            'FundamentalsAgent',
            'AAPL',
            data={'recommendation': 'sell'}
        )

        # Conflict should be detected
        assert len(conflicts_detected) == 1
        assert conflicts_detected[0]['ticker'] == 'AAPL'

        # CONFLICT_DETECTED event should be in history
        conflicts = EventBus.get_history(event_type=AgentEvent.CONFLICT_DETECTED)
        assert len(conflicts) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
