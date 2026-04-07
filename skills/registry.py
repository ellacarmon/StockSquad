"""
Skills Registry

Central registry for all available skills in the system.
Skills are registered once at startup and can be accessed by any agent.
"""

from typing import Dict, Type, List, Optional
import logging

from skills.base import BaseSkill


logger = logging.getLogger(__name__)


class SkillsRegistry:
    """
    Central registry of all available skills/tools.

    This is a singleton-like class that maintains a global registry of
    skills. Skills are registered by name and can be retrieved by any agent.
    """

    _skills: Dict[str, Type[BaseSkill]] = {}
    _instances: Dict[str, BaseSkill] = {}  # Cached instances for singleton skills

    @classmethod
    def register(
        cls,
        skill_name: str,
        skill_class: Type[BaseSkill],
        singleton: bool = True
    ) -> None:
        """
        Register a skill in the global registry.

        Args:
            skill_name: Unique name for the skill
            skill_class: Class that implements BaseSkill
            singleton: If True, only one instance of the skill will be created
                      and shared across all agents (default: True)

        Raises:
            ValueError: If skill_name is already registered
            TypeError: If skill_class doesn't inherit from BaseSkill
        """
        # Validate skill_class
        if not issubclass(skill_class, BaseSkill):
            raise TypeError(
                f"Skill class {skill_class.__name__} must inherit from BaseSkill"
            )

        # Check for duplicate registration
        if skill_name in cls._skills:
            logger.warning(
                f"Skill '{skill_name}' is already registered. "
                f"Overwriting with {skill_class.__name__}"
            )

        cls._skills[skill_name] = skill_class
        logger.info(f"Registered skill: {skill_name} ({skill_class.__name__})")

        # Pre-instantiate if singleton
        if singleton:
            cls._instances[skill_name] = skill_class()

    @classmethod
    def get(cls, skill_name: str) -> Type[BaseSkill]:
        """
        Get a skill class by name.

        Args:
            skill_name: Name of the skill to retrieve

        Returns:
            The skill class

        Raises:
            ValueError: If skill is not registered
        """
        if skill_name not in cls._skills:
            available = ', '.join(cls._skills.keys()) if cls._skills else 'none'
            raise ValueError(
                f"Skill '{skill_name}' not registered. "
                f"Available skills: {available}"
            )

        return cls._skills[skill_name]

    @classmethod
    def get_instance(cls, skill_name: str) -> BaseSkill:
        """
        Get a skill instance by name.

        For singleton skills, returns the cached instance.
        For non-singleton skills, creates a new instance.

        Args:
            skill_name: Name of the skill to retrieve

        Returns:
            An instance of the skill

        Raises:
            ValueError: If skill is not registered
        """
        if skill_name not in cls._skills:
            available = ', '.join(cls._skills.keys()) if cls._skills else 'none'
            raise ValueError(
                f"Skill '{skill_name}' not registered. "
                f"Available skills: {available}"
            )

        # Return cached instance if available
        if skill_name in cls._instances:
            return cls._instances[skill_name]

        # Create new instance
        skill_class = cls._skills[skill_name]
        return skill_class()

    @classmethod
    def list_skills(cls) -> List[str]:
        """
        List all registered skill names.

        Returns:
            List of skill names
        """
        return list(cls._skills.keys())

    @classmethod
    def get_all_capabilities(cls) -> Dict[str, Dict]:
        """
        Get capabilities of all registered skills.

        Returns:
            Dictionary mapping skill names to their capabilities
        """
        capabilities = {}
        for skill_name in cls._skills:
            instance = cls.get_instance(skill_name)
            capabilities[skill_name] = instance.get_capabilities()
        return capabilities

    @classmethod
    def is_registered(cls, skill_name: str) -> bool:
        """
        Check if a skill is registered.

        Args:
            skill_name: Name of the skill to check

        Returns:
            True if skill is registered, False otherwise
        """
        return skill_name in cls._skills

    @classmethod
    def unregister(cls, skill_name: str) -> None:
        """
        Unregister a skill.

        Useful for testing or dynamic skill management.

        Args:
            skill_name: Name of the skill to unregister
        """
        if skill_name in cls._skills:
            del cls._skills[skill_name]
            if skill_name in cls._instances:
                del cls._instances[skill_name]
            logger.info(f"Unregistered skill: {skill_name}")

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered skills.

        Useful for testing.
        """
        cls._skills.clear()
        cls._instances.clear()
        logger.info("Cleared all registered skills")
