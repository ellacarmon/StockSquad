"""
Base Skill Interface

All agent skills/tools inherit from BaseSkill and implement the execute method.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseSkill(ABC):
    """
    Base class for all agent skills/tools.

    Skills are reusable capabilities that agents can use to perform tasks.
    Each skill implements a consistent interface while providing specialized
    functionality.

    Attributes:
        skill_name: Unique identifier for the skill
        description: Human-readable description of what the skill does
        version: Semantic version of the skill implementation
    """

    skill_name: str = "base_skill"
    description: str = "Base skill class"
    version: str = "1.0.0"

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the skill's primary function.

        This is the main entry point for the skill. Subclasses should
        implement this method to provide the core functionality.

        Args:
            **kwargs: Skill-specific parameters

        Returns:
            Any: Skill-specific return value

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute()")

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return metadata about this skill's capabilities.

        Returns:
            Dict containing:
                - name: Skill name
                - description: What the skill does
                - version: Skill version
                - methods: List of public methods available
        """
        # Get all public methods (not starting with _)
        methods = [
            method_name
            for method_name in dir(self)
            if not method_name.startswith('_')
            and callable(getattr(self, method_name))
            and method_name not in ['execute', 'get_capabilities']
        ]

        return {
            'name': self.skill_name,
            'description': self.description,
            'version': self.version,
            'methods': methods
        }

    def validate_params(self, required_params: List[str], **kwargs) -> None:
        """
        Validate that required parameters are present.

        Args:
            required_params: List of required parameter names
            **kwargs: Parameters to validate

        Raises:
            ValueError: If any required parameter is missing
        """
        missing = [param for param in required_params if param not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required parameters for {self.skill_name}: {', '.join(missing)}"
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} '{self.skill_name}' v{self.version}>"
