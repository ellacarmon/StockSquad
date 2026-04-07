"""
Technical Indicators Skill

Wraps the TechnicalIndicators class to provide technical analysis capabilities to agents.
"""

from typing import Dict, Any, List
import pandas as pd

from skills.base import BaseSkill
from tools.ta_indicators import TechnicalIndicators


class TechnicalIndicatorsSkill(BaseSkill):
    """
    Provides technical analysis indicators to agents.

    This skill wraps the TechnicalIndicators class and provides methods to:
    - Calculate RSI (Relative Strength Index)
    - Calculate MACD (Moving Average Convergence Divergence)
    - Calculate Moving Averages (SMA, EMA)
    - Calculate Bollinger Bands
    - Calculate volume indicators
    - Identify support/resistance levels
    - Calculate all indicators at once
    """

    skill_name = "technical_indicators"
    description = "Calculate technical analysis indicators (RSI, MACD, MAs, Bollinger Bands)"
    version = "1.0.0"

    def __init__(self):
        """Initialize the technical indicators skill."""
        self.calculator = TechnicalIndicators()

    def execute(self, price_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Execute the skill's primary function: calculate all indicators.

        Args:
            price_data: DataFrame with 'Close', 'High', 'Low', 'Volume' columns
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary containing all calculated indicators
        """
        return self.calculate_all_indicators(price_data)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        return self.calculator.calculate_rsi(prices, period)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            prices: Series of closing prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram' series
        """
        return self.calculator.calculate_macd(
            prices, fast_period, slow_period, signal_period
        )

    def calculate_moving_averages(
        self,
        prices: pd.Series,
        periods: List[int] = [20, 50, 200]
    ) -> Dict[str, pd.Series]:
        """
        Calculate Simple Moving Averages (SMA).

        Args:
            prices: Series of closing prices
            periods: List of periods to calculate

        Returns:
            Dictionary with MA series for each period
        """
        return self.calculator.calculate_moving_averages(prices, periods)

    def calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: Series of closing prices
            period: EMA period

        Returns:
            Series of EMA values
        """
        return self.calculator.calculate_ema(prices, period)

    def calculate_exponential_moving_averages(
        self,
        prices: pd.Series,
        periods: List[int] = [12, 26]
    ) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages for multiple periods.

        Args:
            prices: Series of closing prices
            periods: List of periods to calculate

        Returns:
            Dictionary with EMA series for each period
        """
        return self.calculator.calculate_exponential_moving_averages(prices, periods)

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of closing prices
            period: Moving average period
            std_dev: Standard deviation multiplier

        Returns:
            Dictionary with 'upper', 'middle', 'lower' bands
        """
        return self.calculator.calculate_bollinger_bands(prices, period, std_dev)

    def calculate_volume_indicators(
        self,
        prices: pd.Series,
        volumes: pd.Series,
        period: int = 20
    ) -> Dict[str, Any]:
        """
        Calculate volume-based indicators.

        Args:
            prices: Series of closing prices
            volumes: Series of volumes
            period: Period for calculations

        Returns:
            Dictionary with volume indicators
        """
        return self.calculator.calculate_volume_indicators(prices, volumes, period)

    def calculate_support_resistance(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> Dict[str, float]:
        """
        Identify support and resistance levels.

        Args:
            prices: Series of closing prices
            window: Lookback window

        Returns:
            Dictionary with support and resistance levels
        """
        return self.calculator.calculate_support_resistance(prices, window)

    def calculate_all_indicators(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all technical indicators for given price data.

        Args:
            price_data: DataFrame with 'Close', 'High', 'Low', 'Volume' columns

        Returns:
            Dictionary containing all calculated indicators
        """
        return self.calculator.calculate_all_indicators(price_data)

    def format_for_llm(self, indicators: Dict[str, Any]) -> str:
        """
        Format technical indicators as readable text for LLM.

        Args:
            indicators: Dictionary of calculated indicators

        Returns:
            Formatted string
        """
        return self.calculator.format_for_llm(indicators)
