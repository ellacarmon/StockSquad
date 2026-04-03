"""
Technical Analysis Indicators
Calculates RSI, MACD, Moving Averages, and other technical indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class TechnicalIndicators:
    """Calculate technical analysis indicators for stock price data."""

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of closing prices
            period: RSI period (default 14)

        Returns:
            Series of RSI values (0-100)
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(
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
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd = fast_ema - slow_ema
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal

        return {
            "macd": macd,
            "signal": signal,
            "histogram": histogram
        }

    @staticmethod
    def calculate_moving_averages(
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
        return {
            f"SMA_{period}": prices.rolling(window=period).mean()
            for period in periods
        }

    @staticmethod
    def calculate_ema(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            prices: Series of closing prices
            period: EMA period

        Returns:
            Series of EMA values
        """
        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_exponential_moving_averages(
        prices: pd.Series,
        periods: List[int] = [12, 26]
    ) -> Dict[str, pd.Series]:
        """
        Calculate Exponential Moving Averages (EMA) for multiple periods.

        Args:
            prices: Series of closing prices
            periods: List of periods to calculate

        Returns:
            Dictionary with EMA series for each period
        """
        return {
            f"EMA_{period}": prices.ewm(span=period, adjust=False).mean()
            for period in periods
        }

    @staticmethod
    def calculate_bollinger_bands(
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
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        return {
            "upper": middle + (std * std_dev),
            "middle": middle,
            "lower": middle - (std * std_dev)
        }

    @staticmethod
    def calculate_volume_indicators(
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
        avg_volume = volumes.rolling(window=period).mean()
        volume_ratio = volumes / avg_volume

        # On-Balance Volume (OBV)
        obv = (np.sign(prices.diff()) * volumes).fillna(0).cumsum()

        return {
            "average_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "obv": obv
        }

    @staticmethod
    def calculate_support_resistance(
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
        recent_prices = prices.tail(window)

        return {
            "support": float(recent_prices.min()),
            "resistance": float(recent_prices.max()),
            "current": float(prices.iloc[-1])
        }

    def calculate_all_indicators(
        self,
        price_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate all technical indicators for given price data.

        Args:
            price_data: DataFrame with 'Close', 'High', 'Low', 'Volume' columns

        Returns:
            Dictionary containing all calculated indicators
        """
        close_prices = price_data['Close']
        volumes = price_data['Volume']

        # Current price and basic stats
        current_price = float(close_prices.iloc[-1])
        price_high = float(price_data['High'].iloc[-1])
        price_low = float(price_data['Low'].iloc[-1])

        # RSI
        rsi = self.calculate_rsi(close_prices)
        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

        # MACD
        macd_data = self.calculate_macd(close_prices)
        current_macd = float(macd_data['macd'].iloc[-1]) if not pd.isna(macd_data['macd'].iloc[-1]) else None
        current_signal = float(macd_data['signal'].iloc[-1]) if not pd.isna(macd_data['signal'].iloc[-1]) else None
        current_histogram = float(macd_data['histogram'].iloc[-1]) if not pd.isna(macd_data['histogram'].iloc[-1]) else None

        # Moving Averages
        mas = self.calculate_moving_averages(close_prices, [20, 50, 200])
        ma_values = {
            key: float(val.iloc[-1]) if not pd.isna(val.iloc[-1]) else None
            for key, val in mas.items()
        }

        # Bollinger Bands
        bb = self.calculate_bollinger_bands(close_prices)
        bb_values = {
            key: float(val.iloc[-1]) if not pd.isna(val.iloc[-1]) else None
            for key, val in bb.items()
        }

        # Volume indicators
        vol_indicators = self.calculate_volume_indicators(close_prices, volumes)
        avg_volume = float(vol_indicators['average_volume'].iloc[-1]) if not pd.isna(vol_indicators['average_volume'].iloc[-1]) else None
        volume_ratio = float(vol_indicators['volume_ratio'].iloc[-1]) if not pd.isna(vol_indicators['volume_ratio'].iloc[-1]) else None

        # Support/Resistance
        support_resistance = self.calculate_support_resistance(close_prices)

        # Trend analysis
        sma_20 = ma_values.get('SMA_20')
        sma_50 = ma_values.get('SMA_50')
        sma_200 = ma_values.get('SMA_200')

        trend = "Unknown"
        if sma_20 and sma_50 and sma_200:
            if sma_20 > sma_50 > sma_200:
                trend = "Strong Uptrend"
            elif sma_20 > sma_50:
                trend = "Uptrend"
            elif sma_20 < sma_50 < sma_200:
                trend = "Strong Downtrend"
            elif sma_20 < sma_50:
                trend = "Downtrend"
            else:
                trend = "Sideways"

        # Price position relative to MAs
        price_position = {}
        if sma_20:
            price_position['vs_SMA20'] = ((current_price - sma_20) / sma_20) * 100
        if sma_50:
            price_position['vs_SMA50'] = ((current_price - sma_50) / sma_50) * 100
        if sma_200:
            price_position['vs_SMA200'] = ((current_price - sma_200) / sma_200) * 100

        # MACD signal
        macd_signal = "Neutral"
        if current_macd and current_signal:
            if current_macd > current_signal and current_histogram > 0:
                macd_signal = "Bullish"
            elif current_macd < current_signal and current_histogram < 0:
                macd_signal = "Bearish"

        # RSI signal
        rsi_signal = "Neutral"
        if current_rsi:
            if current_rsi > 70:
                rsi_signal = "Overbought"
            elif current_rsi < 30:
                rsi_signal = "Oversold"

        return {
            "current_price": current_price,
            "price_high": price_high,
            "price_low": price_low,
            "rsi": {
                "value": current_rsi,
                "signal": rsi_signal
            },
            "macd": {
                "macd": current_macd,
                "signal": current_signal,
                "histogram": current_histogram,
                "signal_interpretation": macd_signal
            },
            "moving_averages": ma_values,
            "bollinger_bands": bb_values,
            "volume": {
                "current": float(volumes.iloc[-1]),
                "average": avg_volume,
                "ratio": volume_ratio
            },
            "support_resistance": support_resistance,
            "trend": trend,
            "price_position": price_position
        }

    def format_for_llm(self, indicators: Dict[str, Any]) -> str:
        """
        Format technical indicators as readable text for LLM.

        Args:
            indicators: Dictionary of calculated indicators

        Returns:
            Formatted string
        """
        rsi = indicators['rsi']
        macd = indicators['macd']
        mas = indicators['moving_averages']
        bb = indicators['bollinger_bands']
        vol = indicators['volume']
        sr = indicators['support_resistance']
        trend = indicators['trend']
        price_pos = indicators['price_position']

        formatted = f"""
Technical Analysis Indicators
{'=' * 60}

Current Price: ${indicators['current_price']:.2f}
Trend: {trend}

Momentum Indicators:
- RSI (14): {rsi['value']:.2f} - {rsi['signal']}
- MACD: {macd['macd']:.4f} (Signal: {macd['signal']:.4f})
- MACD Signal: {macd['signal_interpretation']}

Moving Averages:
- SMA 20: ${mas.get('SMA_20', 0):.2f} ({price_pos.get('vs_SMA20', 0):+.2f}%)
- SMA 50: ${mas.get('SMA_50', 0):.2f} ({price_pos.get('vs_SMA50', 0):+.2f}%)
- SMA 200: ${mas.get('SMA_200', 0):.2f} ({price_pos.get('vs_SMA200', 0):+.2f}%)

Bollinger Bands (20, 2):
- Upper: ${bb.get('upper', 0):.2f}
- Middle: ${bb.get('middle', 0):.2f}
- Lower: ${bb.get('lower', 0):.2f}

Volume Analysis:
- Current: {vol['current']:,.0f}
- 20-day Average: {vol['average']:,.0f}
- Volume Ratio: {vol['ratio']:.2f}x

Support & Resistance:
- Resistance: ${sr['resistance']:.2f}
- Current: ${sr['current']:.2f}
- Support: ${sr['support']:.2f}

Technical Summary:
The stock is in a {trend.lower()}. RSI indicates {rsi['signal'].lower()} conditions.
MACD is showing {macd['signal_interpretation'].lower()} momentum.
Price is {price_pos.get('vs_SMA20', 0):+.2f}% from 20-day MA.
"""
        return formatted.strip()
