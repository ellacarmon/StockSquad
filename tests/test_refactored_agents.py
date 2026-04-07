"""
Tests for refactored agents using the skills system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from skills import register_all_skills
from skills.registry import SkillsRegistry


class TestTechnicalAgentWithSkills:
    """Test TechnicalAgent with skills system"""

    @classmethod
    def setup_class(cls):
        """Register skills before tests"""
        SkillsRegistry.clear()
        register_all_skills()

    @classmethod
    def teardown_class(cls):
        """Clean up after tests"""
        SkillsRegistry.clear()

    def test_technical_agent_has_skills(self):
        """Test that TechnicalAgent gets skills injected"""
        from agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent()

        # Verify agent has required skills
        assert agent.has_skill('technical_indicators')
        assert agent.has_skill('ml_signals')

        # Verify agent can access skills
        assert agent.skills.technical_indicators is not None
        assert agent.skills.ml_signals is not None

    def test_technical_agent_capabilities(self):
        """Test agent capabilities reporting"""
        from agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent()
        capabilities = agent.get_capabilities()

        assert capabilities['agent_name'] == 'TechnicalAgent'
        assert 'technical_indicators' in capabilities['required_skills']
        assert 'ml_signals' in capabilities['required_skills']
        assert 'technical_indicators' in capabilities['available_skills']
        assert 'ml_signals' in capabilities['available_skills']

    def test_technical_agent_can_calculate_indicators(self):
        """Test that TechnicalAgent can use skills to calculate indicators"""
        from agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent()

        # Create fake price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        price_data = pd.DataFrame({
            'Open': np.random.uniform(100, 150, 100),
            'High': np.random.uniform(150, 160, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 150, 100),
            'Volume': np.random.uniform(1e6, 10e6, 100)
        }, index=dates)

        # Calculate indicators using the skill
        indicators = agent.skills.technical_indicators.calculate_all_indicators(price_data)

        # Verify indicators were calculated
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'moving_averages' in indicators
        assert 'trend' in indicators
        assert indicators['rsi']['value'] is not None

    def test_technical_agent_can_score_signal(self):
        """Test that TechnicalAgent can use skills to score signals"""
        from agents.technical_agent import TechnicalAgent

        agent = TechnicalAgent()

        # Create fake price data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        price_data = pd.DataFrame({
            'Open': np.random.uniform(100, 150, 100),
            'High': np.random.uniform(150, 160, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 150, 100),
            'Volume': np.random.uniform(1e6, 10e6, 100)
        }, index=dates)

        # Calculate indicators
        indicators = agent.skills.technical_indicators.calculate_all_indicators(price_data)

        # Score signal using the skill
        signal = agent.skills.ml_signals.score_signal(indicators)

        # Verify signal was generated
        assert 'signal_score' in signal
        assert 'direction' in signal
        assert 'confidence' in signal
        assert 'recommendation' in signal
        assert signal['direction'] in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_technical_agent_model_type_param(self):
        """Test that model_type parameter is passed to ML skill"""
        from agents.technical_agent import TechnicalAgent

        # Create agent with specific model type
        agent = TechnicalAgent(model_type='ensemble_unanimous')

        assert agent.model_type == 'ensemble_unanimous'

    def test_multiple_agents_share_skills(self):
        """Test that multiple agents share the same skill instances"""
        from agents.technical_agent import TechnicalAgent

        agent1 = TechnicalAgent()
        agent2 = TechnicalAgent()

        # Both should have the same skill instances (singletons)
        assert agent1.skills.technical_indicators is agent2.skills.technical_indicators
        assert agent1.skills.ml_signals is agent2.skills.ml_signals  # Same singleton instance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
