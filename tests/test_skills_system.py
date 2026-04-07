"""
Tests for the Skills System

Tests BaseSkill, SkillsRegistry, SkillsContainer, and BaseAgent.
"""

import pytest
import sys
import os
from typing import Any
import importlib.util

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skills.base import BaseSkill
from skills.registry import SkillsRegistry
from skills.container import SkillsContainer

# Import BaseAgent directly from the file to avoid loading agents package
spec = importlib.util.spec_from_file_location(
    "agents.base",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "agents", "base.py")
)
base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base_module)
BaseAgent = base_module.BaseAgent


# Test Skills for testing purposes
class MockMarketDataSkill(BaseSkill):
    """Mock skill for testing"""
    skill_name = "mock_market_data"
    description = "Mock market data skill for testing"
    version = "1.0.0"

    def execute(self, ticker: str, **kwargs) -> dict:
        return {
            'ticker': ticker,
            'price': 150.00,
            'volume': 1000000
        }

    def get_price(self, ticker: str) -> float:
        return 150.00


class MockTechnicalSkill(BaseSkill):
    """Mock technical indicators skill"""
    skill_name = "mock_technical"
    description = "Mock technical indicators"
    version = "1.0.0"

    def execute(self, data: dict, **kwargs) -> dict:
        return {'rsi': 65, 'macd': 2.5}

    def calculate_rsi(self, data: dict) -> float:
        return 65.0


class MockOptionalSkill(BaseSkill):
    """Mock optional skill"""
    skill_name = "mock_optional"
    description = "Mock optional skill"
    version = "1.0.0"

    def execute(self, **kwargs) -> str:
        return "optional_data"


# Test Agent
class MockAgent(BaseAgent):
    """Mock agent for testing"""
    agent_name = "MockAgent"
    agent_description = "Mock agent for testing"
    required_skills = ['mock_market_data', 'mock_technical']
    optional_skills = ['mock_optional']

    def analyze(self, ticker: str, **kwargs):
        data = self.skills.mock_market_data.execute(ticker)
        indicators = self.skills.mock_technical.execute(data)

        optional_data = None
        if self.has_skill('mock_optional'):
            optional_data = self.skills.mock_optional.execute()

        return {
            'agent': self.agent_name,
            'ticker': ticker,
            'data': data,
            'indicators': indicators,
            'optional': optional_data
        }


class TestBaseSkill:
    """Test BaseSkill class"""

    def test_skill_creation(self):
        """Test creating a skill"""
        skill = MockMarketDataSkill()
        assert skill.skill_name == "mock_market_data"
        assert skill.description == "Mock market data skill for testing"
        assert skill.version == "1.0.0"

    def test_skill_execute(self):
        """Test skill execution"""
        skill = MockMarketDataSkill()
        result = skill.execute('AAPL')
        assert result['ticker'] == 'AAPL'
        assert result['price'] == 150.00

    def test_skill_capabilities(self):
        """Test get_capabilities"""
        skill = MockMarketDataSkill()
        caps = skill.get_capabilities()
        assert caps['name'] == 'mock_market_data'
        assert 'get_price' in caps['methods']

    def test_skill_validate_params(self):
        """Test parameter validation"""
        skill = MockMarketDataSkill()
        # Should not raise
        skill.validate_params(['ticker'], ticker='AAPL')

        # Should raise
        with pytest.raises(ValueError, match="Missing required parameters"):
            skill.validate_params(['ticker', 'period'], ticker='AAPL')


class TestSkillsRegistry:
    """Test SkillsRegistry class"""

    def setup_method(self):
        """Clear registry before each test"""
        SkillsRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test"""
        SkillsRegistry.clear()

    def test_register_skill(self):
        """Test registering a skill"""
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        assert SkillsRegistry.is_registered('mock_market_data')
        assert 'mock_market_data' in SkillsRegistry.list_skills()

    def test_register_invalid_skill(self):
        """Test registering invalid skill raises error"""
        with pytest.raises(TypeError, match="must inherit from BaseSkill"):
            SkillsRegistry.register('invalid', str)

    def test_get_skill(self):
        """Test getting a skill class"""
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        skill_class = SkillsRegistry.get('mock_market_data')
        assert skill_class == MockMarketDataSkill

    def test_get_nonexistent_skill(self):
        """Test getting non-existent skill raises error"""
        with pytest.raises(ValueError, match="not registered"):
            SkillsRegistry.get('nonexistent')

    def test_get_instance(self):
        """Test getting a skill instance"""
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        instance = SkillsRegistry.get_instance('mock_market_data')
        assert isinstance(instance, MockMarketDataSkill)

    def test_singleton_instance(self):
        """Test singleton instances are cached"""
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill, singleton=True)
        instance1 = SkillsRegistry.get_instance('mock_market_data')
        instance2 = SkillsRegistry.get_instance('mock_market_data')
        assert instance1 is instance2  # Same instance

    def test_list_skills(self):
        """Test listing all skills"""
        SkillsRegistry.register('skill1', MockMarketDataSkill)
        SkillsRegistry.register('skill2', MockTechnicalSkill)
        skills = SkillsRegistry.list_skills()
        assert 'skill1' in skills
        assert 'skill2' in skills

    def test_unregister_skill(self):
        """Test unregistering a skill"""
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        assert SkillsRegistry.is_registered('mock_market_data')

        SkillsRegistry.unregister('mock_market_data')
        assert not SkillsRegistry.is_registered('mock_market_data')


class TestSkillsContainer:
    """Test SkillsContainer class"""

    def setup_method(self):
        """Register test skills before each test"""
        SkillsRegistry.clear()
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        SkillsRegistry.register('mock_technical', MockTechnicalSkill)
        SkillsRegistry.register('mock_optional', MockOptionalSkill)

    def teardown_method(self):
        """Clear registry after each test"""
        SkillsRegistry.clear()

    def test_container_creation(self):
        """Test creating a container with required skills"""
        container = SkillsContainer(['mock_market_data', 'mock_technical'])
        assert container.has('mock_market_data')
        assert container.has('mock_technical')

    def test_container_missing_required_skill(self):
        """Test container raises error if required skill missing"""
        with pytest.raises(ValueError, match="Required skill.*not registered"):
            SkillsContainer(['nonexistent_skill'])

    def test_container_optional_skills(self):
        """Test container loads optional skills if available"""
        container = SkillsContainer(
            required_skills=['mock_market_data'],
            optional_skills=['mock_optional', 'nonexistent']
        )
        assert container.has('mock_market_data')
        assert container.has('mock_optional')
        assert not container.has('nonexistent')  # Missing optional skill

    def test_container_attribute_access(self):
        """Test accessing skills via attributes"""
        container = SkillsContainer(['mock_market_data'])
        skill = container.mock_market_data
        assert isinstance(skill, MockMarketDataSkill)

    def test_container_attribute_access_missing(self):
        """Test accessing missing skill raises AttributeError"""
        container = SkillsContainer(['mock_market_data'])
        with pytest.raises(AttributeError, match="not available"):
            _ = container.nonexistent

    def test_container_in_operator(self):
        """Test 'in' operator"""
        container = SkillsContainer(['mock_market_data'])
        assert 'mock_market_data' in container
        assert 'nonexistent' not in container

    def test_container_list_available(self):
        """Test listing available skills"""
        container = SkillsContainer(
            required_skills=['mock_market_data'],
            optional_skills=['mock_optional']
        )
        available = container.list_available()
        assert 'mock_market_data' in available
        assert 'mock_optional' in available


class TestBaseAgent:
    """Test BaseAgent class"""

    def setup_method(self):
        """Register test skills before each test"""
        SkillsRegistry.clear()
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        SkillsRegistry.register('mock_technical', MockTechnicalSkill)
        SkillsRegistry.register('mock_optional', MockOptionalSkill)

    def teardown_method(self):
        """Clear registry after each test"""
        SkillsRegistry.clear()

    def test_agent_creation(self):
        """Test creating an agent"""
        agent = MockAgent()
        assert agent.agent_name == "MockAgent"
        assert agent.has_skill('mock_market_data')
        assert agent.has_skill('mock_technical')

    def test_agent_missing_required_skill(self):
        """Test agent raises error if required skill missing"""
        SkillsRegistry.unregister('mock_market_data')
        with pytest.raises(ValueError, match="Required skill.*not registered"):
            MockAgent()

    def test_agent_analyze(self):
        """Test agent analysis"""
        agent = MockAgent()
        result = agent.analyze('AAPL')

        assert result['agent'] == 'MockAgent'
        assert result['ticker'] == 'AAPL'
        assert result['data']['price'] == 150.00
        assert result['indicators']['rsi'] == 65
        assert result['optional'] == 'optional_data'

    def test_agent_analyze_without_optional(self):
        """Test agent works without optional skills"""
        SkillsRegistry.unregister('mock_optional')
        agent = MockAgent()
        result = agent.analyze('AAPL')

        assert result['optional'] is None  # Optional skill not available

    def test_agent_capabilities(self):
        """Test agent capabilities"""
        agent = MockAgent()
        caps = agent.get_capabilities()

        assert caps['agent_name'] == 'MockAgent'
        assert 'mock_market_data' in caps['required_skills']
        assert 'mock_optional' in caps['optional_skills']
        assert 'mock_market_data' in caps['available_skills']

    def test_agent_has_skill(self):
        """Test has_skill method"""
        agent = MockAgent()
        assert agent.has_skill('mock_market_data')
        assert not agent.has_skill('nonexistent')


class TestIntegration:
    """Integration tests for the complete skills system"""

    def setup_method(self):
        """Register test skills before each test"""
        SkillsRegistry.clear()
        SkillsRegistry.register('mock_market_data', MockMarketDataSkill)
        SkillsRegistry.register('mock_technical', MockTechnicalSkill)
        SkillsRegistry.register('mock_optional', MockOptionalSkill)

    def teardown_method(self):
        """Clear registry after each test"""
        SkillsRegistry.clear()

    def test_end_to_end_workflow(self):
        """Test complete workflow from registration to agent execution"""
        # Create agent
        agent = MockAgent()

        # Verify skills loaded
        assert agent.has_skill('mock_market_data')
        assert agent.has_skill('mock_technical')
        assert agent.has_skill('mock_optional')

        # Run analysis
        result = agent.analyze('AAPL')

        # Verify result
        assert result['ticker'] == 'AAPL'
        assert result['data']['price'] == 150.00
        assert result['indicators']['rsi'] == 65
        assert result['optional'] == 'optional_data'

    def test_multiple_agents_share_skills(self):
        """Test that multiple agents can share skill instances"""
        agent1 = MockAgent()
        agent2 = MockAgent()

        # Both agents should have access to the same skill instances (singletons)
        skill1 = agent1.skills.mock_market_data
        skill2 = agent2.skills.mock_market_data
        assert skill1 is skill2  # Same instance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
