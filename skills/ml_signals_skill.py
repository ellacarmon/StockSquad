"""
ML Signals Skill

Wraps the SignalScorer to provide ML-based signal scoring capabilities to agents.
"""

from typing import Dict, Any, Optional

from skills.base import BaseSkill
from ml.signal_model import SignalScorer


class MLSignalsSkill(BaseSkill):
    """
    Provides ML-based signal scoring capabilities to agents.

    This skill wraps the SignalScorer class and provides methods to:
    - Generate signal scores based on technical indicators
    - Provide buy/sell/hold recommendations
    - Calculate confidence levels
    - Use either ML models (XGBoost, Random Forest, LightGBM) or rule-based fallback
    """

    skill_name = "ml_signals"
    description = "ML-based signal scoring for buy/sell/hold recommendations"
    version = "1.0.0"

    def __init__(self, use_ml: bool = True, model_type: str = "xgboost"):
        """
        Initialize the ML signals skill.

        Args:
            use_ml: Whether to use ML models (falls back to rules if unavailable)
            model_type: Type of ML model ('xgboost', 'random_forest', 'lightgbm',
                       or ensemble options like 'ensemble_unanimous', 'ensemble_voting')
        """
        self.scorer = SignalScorer(use_ml=use_ml, model_type=model_type)
        self.model_type = model_type

    def execute(
        self,
        indicators: Dict[str, Any],
        sentiment_result: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the skill's primary function: score the signal.

        Args:
            indicators: Dictionary of technical indicators
            sentiment_result: Optional sentiment analysis results
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with signal score, direction, confidence, and recommendation
        """
        return self.score_signal(indicators, sentiment_result)

    def score_signal(
        self,
        indicators: Dict[str, Any],
        sentiment_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a signal score based on technical indicators.

        Args:
            indicators: Dictionary of technical indicators
            sentiment_result: Optional SentimentAgent result for ML features

        Returns:
            Dictionary with signal, confidence, and reasoning:
                - signal_score: Score from -100 (bearish) to +100 (bullish)
                - direction: 'BULLISH', 'BEARISH', or 'NEUTRAL'
                - confidence: Confidence level 0-100
                - confidence_level: 'Very Low', 'Low', 'Medium', 'High', 'Very High'
                - recommendation: 'STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'
                - signals: List of reasoning points
                - model_version: Version of the model used
                - features_used: List of features considered
                - timestamp: When the signal was generated
        """
        return self.scorer.score_signal(indicators, sentiment_result)

    def get_confidence_level(self, confidence: int) -> str:
        """
        Categorize confidence score.

        Args:
            confidence: Confidence value (0-100)

        Returns:
            Confidence level string ('Very Low', 'Low', 'Medium', 'High', 'Very High')
        """
        return self.scorer._get_confidence_level(confidence)

    def format_for_llm(self, signal_data: Dict[str, Any]) -> str:
        """
        Format signal scoring results for LLM consumption.

        Args:
            signal_data: Signal scoring results from score_signal()

        Returns:
            Formatted string for LLM
        """
        return self.scorer.format_for_llm(signal_data)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ML model in use.

        Returns:
            Dictionary with model information:
                - model_version: Version string
                - model_type: Type of model
                - ml_powered: Whether using ML or rule-based
                - features: List of features used
        """
        return {
            'model_version': self.scorer.model_version,
            'model_type': self.model_type,
            'ml_powered': self.scorer.ml_engine is not None,
            'features': self.scorer.features_used
        }
