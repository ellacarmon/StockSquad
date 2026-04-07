"""
Skills Container

Container that provides skills to an agent based on its requirements.
"""

from typing import Dict, List, Optional
import logging

from skills.registry import SkillsRegistry
from skills.base import BaseSkill


logger = logging.getLogger(__name__)


class SkillsContainer:
    """
    Container that provides skills to an agent.

    The container loads required and optional skills from the registry
    and provides them to the agent via attribute access.

    Example:
        container = SkillsContainer(['market_data'], ['news_api'])
        data = container.market_data.get_price_history('AAPL')
        if container.has('news_api'):
            news = container.news_api.get_latest_news('AAPL')
    """

    def __init__(
        self,
        required_skills: List[str],
        optional_skills: Optional[List[str]] = None
    ):
        """
        Initialize the skills container.

        Args:
            required_skills: Skills that must be available (raises error if missing)
            optional_skills: Skills to load if available (silent if missing)

        Raises:
            ValueError: If any required skill is not registered
        """
        self.required_skills = required_skills
        self.optional_skills = optional_skills or []
        self._loaded_skills: Dict[str, BaseSkill] = {}

        self._load_skills()

    def _load_skills(self) -> None:
        """
        Load all required and available optional skills.

        Raises:
            ValueError: If any required skill is not registered
        """
        # Load required skills
        for skill_name in self.required_skills:
            if not SkillsRegistry.is_registered(skill_name):
                available = ', '.join(SkillsRegistry.list_skills())
                raise ValueError(
                    f"Required skill '{skill_name}' is not registered. "
                    f"Available skills: {available}"
                )

            skill_instance = SkillsRegistry.get_instance(skill_name)
            self._loaded_skills[skill_name] = skill_instance
            logger.debug(f"Loaded required skill: {skill_name}")

        # Load optional skills if available
        for skill_name in self.optional_skills:
            if SkillsRegistry.is_registered(skill_name):
                skill_instance = SkillsRegistry.get_instance(skill_name)
                self._loaded_skills[skill_name] = skill_instance
                logger.debug(f"Loaded optional skill: {skill_name}")
            else:
                logger.debug(
                    f"Optional skill '{skill_name}' not available, skipping"
                )

    def has(self, skill_name: str) -> bool:
        """
        Check if a skill is available in this container.

        Args:
            skill_name: Name of the skill to check

        Returns:
            True if skill is loaded, False otherwise
        """
        return skill_name in self._loaded_skills

    def get(self, skill_name: str) -> Optional[BaseSkill]:
        """
        Get a skill by name.

        Args:
            skill_name: Name of the skill to retrieve

        Returns:
            The skill instance if available, None otherwise
        """
        return self._loaded_skills.get(skill_name)

    def list_available(self) -> List[str]:
        """
        List all available skills in this container.

        Returns:
            List of skill names that are loaded
        """
        return list(self._loaded_skills.keys())

    def get_all_capabilities(self) -> Dict[str, Dict]:
        """
        Get capabilities of all loaded skills.

        Returns:
            Dictionary mapping skill names to their capabilities
        """
        return {
            name: skill.get_capabilities()
            for name, skill in self._loaded_skills.items()
        }

    def __getattr__(self, skill_name: str) -> BaseSkill:
        """
        Allow accessing skills via attribute notation.

        This enables: container.market_data instead of container.get('market_data')

        Args:
            skill_name: Name of the skill to access

        Returns:
            The skill instance

        Raises:
            AttributeError: If skill is not available in this container
        """
        # Check if this is a loaded skill
        if skill_name in self._loaded_skills:
            return self._loaded_skills[skill_name]

        # If not found, raise AttributeError
        available = ', '.join(self._loaded_skills.keys())
        raise AttributeError(
            f"Skill '{skill_name}' is not available in this container. "
            f"Available skills: {available}"
        )

    def __contains__(self, skill_name: str) -> bool:
        """
        Support 'in' operator for checking skill availability.

        Example:
            if 'market_data' in container:
                data = container.market_data.get_price_history('AAPL')

        Args:
            skill_name: Name of the skill to check

        Returns:
            True if skill is loaded, False otherwise
        """
        return skill_name in self._loaded_skills

    def __repr__(self) -> str:
        """String representation of the container."""
        skills_list = ', '.join(self._loaded_skills.keys())
        return f"<SkillsContainer [{skills_list}]>"
