"""
Skills System for StockSquad Agents

This module provides a flexible, extensible system for agent capabilities.
Agents declare required and optional skills, and the system dynamically
provides them with the appropriate tools.
"""

import logging

from skills.registry import SkillsRegistry
from skills.base import BaseSkill

__all__ = ['SkillsRegistry', 'BaseSkill', 'register_all_skills']

logger = logging.getLogger(__name__)


def register_all_skills():
    """
    Register all available skills in the system.

    This should be called at application startup to make all skills
    available to agents.
    """
    logger.info("Registering all skills...")

    # Import skills
    from skills.market_data_skill import MarketDataSkill
    from skills.technical_indicators_skill import TechnicalIndicatorsSkill
    from skills.ml_signals_skill import MLSignalsSkill

    # Register skills (singleton=True means one instance shared by all agents)
    SkillsRegistry.register('market_data', MarketDataSkill, singleton=True)
    SkillsRegistry.register('technical_indicators', TechnicalIndicatorsSkill, singleton=True)
    SkillsRegistry.register('ml_signals', MLSignalsSkill, singleton=True)

    logger.info(f"✓ Registered {len(SkillsRegistry.list_skills())} skills: {', '.join(SkillsRegistry.list_skills())}")
