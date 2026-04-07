"""
Base Agent Class with Skills Support

All StockSquad agents inherit from BaseAgent and can access the skills system.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import logging

from skills.container import SkillsContainer
from memory.short_term import ShortTermMemory


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all StockSquad agents with skills support.

    Agents declare their required and optional skills, and the BaseAgent
    automatically provides them via the skills container.

    Subclasses should:
    1. Define required_skills and optional_skills class attributes
    2. Implement the analyze() method
    3. Access skills via self.skills.<skill_name>

    Example:
        class TechnicalAgent(BaseAgent):
            required_skills = ['market_data', 'technical_indicators']
            optional_skills = ['options_data']

            def analyze(self, ticker: str):
                data = self.skills.market_data.get_price_history(ticker)
                indicators = self.skills.technical_indicators.calculate_all(data)
                return {'data': data, 'indicators': indicators}

    Attributes:
        required_skills: Skills that must be available for this agent
        optional_skills: Skills that enhance this agent if available
        agent_name: Human-readable name for this agent
        agent_description: Description of what this agent does
    """

    # Subclasses should override these
    required_skills: List[str] = []
    optional_skills: List[str] = []
    agent_name: str = "BaseAgent"
    agent_description: str = "Base agent class"

    def __init__(
        self,
        memory: Optional[ShortTermMemory] = None,
        **kwargs
    ):
        """
        Initialize the base agent.

        Args:
            memory: Optional short-term memory instance for agent communication
            **kwargs: Additional agent-specific configuration
        """
        self.memory = memory
        self.config = kwargs

        # Initialize skills container
        logger.info(
            f"Initializing {self.agent_name} with skills: "
            f"required={self.required_skills}, optional={self.optional_skills}"
        )

        try:
            self.skills = SkillsContainer(
                required_skills=self.required_skills,
                optional_skills=self.optional_skills
            )
            logger.info(
                f"{self.agent_name} loaded skills: {self.skills.list_available()}"
            )
        except ValueError as e:
            logger.error(f"Failed to initialize {self.agent_name}: {e}")
            raise

    @abstractmethod
    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Perform the agent's primary analysis task.

        This is the main entry point for the agent. Subclasses must
        implement this method to provide their specialized analysis.

        Args:
            ticker: Stock ticker symbol to analyze
            **kwargs: Agent-specific parameters

        Returns:
            Dictionary containing analysis results. Should include at least:
                - 'agent': Name of the agent
                - 'ticker': The ticker analyzed
                - Additional agent-specific results

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement analyze()"
        )

    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return this agent's capabilities and configuration.

        Returns:
            Dictionary containing:
                - agent_name: Name of the agent
                - description: What the agent does
                - required_skills: Skills this agent needs
                - optional_skills: Skills this agent can use if available
                - available_skills: Skills currently loaded
                - skill_capabilities: Detailed capabilities of each skill
        """
        return {
            'agent_name': self.agent_name,
            'description': self.agent_description,
            'required_skills': self.required_skills,
            'optional_skills': self.optional_skills,
            'available_skills': self.skills.list_available(),
            'skill_capabilities': self.skills.get_all_capabilities()
        }

    def has_skill(self, skill_name: str) -> bool:
        """
        Check if a skill is available to this agent.

        Args:
            skill_name: Name of the skill to check

        Returns:
            True if skill is loaded, False otherwise
        """
        return self.skills.has(skill_name)

    def log_analysis_start(self, ticker: str) -> None:
        """
        Log the start of an analysis.

        Args:
            ticker: Stock ticker being analyzed
        """
        logger.info(f"[{self.agent_name}] Starting analysis for {ticker}")
        if self.memory:
            self.memory.add_message(
                agent=self.agent_name,
                role="system",
                content=f"Starting {self.agent_name} analysis for {ticker}"
            )

    def log_analysis_complete(self, ticker: str, summary: str = None) -> None:
        """
        Log the completion of an analysis.

        Args:
            ticker: Stock ticker that was analyzed
            summary: Optional summary of the analysis
        """
        message = f"Completed analysis for {ticker}"
        if summary:
            message += f": {summary}"

        logger.info(f"[{self.agent_name}] {message}")
        if self.memory:
            self.memory.add_message(
                agent=self.agent_name,
                role="system",
                content=message
            )

    def log_error(self, ticker: str, error: Exception) -> None:
        """
        Log an error that occurred during analysis.

        Args:
            ticker: Stock ticker being analyzed
            error: The exception that occurred
        """
        logger.error(
            f"[{self.agent_name}] Error analyzing {ticker}: {error}",
            exc_info=True
        )
        if self.memory:
            self.memory.add_message(
                agent=self.agent_name,
                role="system",
                content=f"Error during analysis: {str(error)}"
            )

    def __repr__(self) -> str:
        """String representation of the agent."""
        skills_list = ', '.join(self.skills.list_available())
        return f"<{self.agent_name} skills=[{skills_list}]>"
