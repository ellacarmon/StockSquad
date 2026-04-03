"""
ML Signal Scoring Model
Generates confidence scores for price direction predictions based on technical indicators.
"""

from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime
from pathlib import Path

from ml.sentiment_features import extract_sentiment_features


class SignalScorer:
    """
    Machine learning-based signal scorer for stock price predictions.

    Supports both ML-based (trained models) and rule-based (fallback) scoring.
    """

    def __init__(self, use_ml: bool = True, model_type: str = "xgboost"):
        """
        Initialize the signal scorer.

        Args:
            use_ml: Whether to use ML models (falls back to rules if models unavailable)
            model_type: Type of ML model to use (xgboost, random_forest, lightgbm)
        """
        self.use_ml = use_ml
        self.model_type = model_type
        self.ml_engine = None

        # Try to load ML models
        if use_ml:
            try:
                models_dir = Path(__file__).parent / "models"
                if models_dir.exists():
                    from ml.inference.prediction_engine import PredictionEngine
                    # Pass the absolute path to models directory
                    self.ml_engine = PredictionEngine(models_dir=str(models_dir), model_type=model_type)
                    self.model_version = f"2.0-ml-{model_type}"
                    print(f"[SignalScorer] Loaded ML model: {model_type} from {models_dir}")
                else:
                    print(f"[SignalScorer] ML models directory not found at {models_dir}")
                    print(f"[SignalScorer] Using rule-based fallback")
                    self.model_version = "1.0-rule-based"
            except Exception as e:
                print(f"[SignalScorer] Failed to load ML models: {e}")
                print(f"[SignalScorer] Using rule-based fallback")
                self.model_version = "1.0-rule-based"
        else:
            self.model_version = "1.0-rule-based"

        self.features_used = [
            "rsi", "macd", "ma_alignment", "volume_ratio",
            "price_vs_ma", "trend_strength"
        ]

    def _get_confidence_level(self, confidence: int) -> str:
        """
        Categorize confidence score.

        Args:
            confidence: Confidence value (0-100)

        Returns:
            Confidence level string
        """
        if confidence > 70:
            return "Very High"
        elif confidence > 50:
            return "High"
        elif confidence > 30:
            return "Medium"
        elif confidence > 15:
            return "Low"
        else:
            return "Very Low"

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
            Dictionary with signal, confidence, and reasoning
        """
        # Try ML prediction first if available
        if self.ml_engine:
            try:
                print(f"[SignalScorer] Running ML prediction using {self.model_type} model...")
                prediction = self.ml_engine.predict(indicators, sentiment_result=sentiment_result)

                # Check if prediction contains error
                if "error" in prediction:
                    raise ValueError(f"PredictionEngine error: {prediction['error']}")

                print(f"[SignalScorer] ✓ ML Prediction: {prediction['direction']} ({prediction['confidence']:.1f}% confidence, {prediction['expected_return']:+.2f}% expected return)")
                # Convert ML prediction format to SignalScorer format
                return {
                    "signal_score": int(prediction["score"]),
                    "direction": prediction["direction"].upper(),
                    "confidence": int(prediction["confidence"]),
                    "confidence_level": self._get_confidence_level(int(prediction["confidence"])),
                    "recommendation": prediction["recommendation"],
                    "signals": [
                        f"ML Model: {prediction['direction']} with {prediction['confidence']:.1f}% confidence",
                        f"Expected 5-day return: {prediction['expected_return']:+.2f}%",
                        f"Direction probabilities: {prediction['direction_probabilities']['up']:.1f}% up, {prediction['direction_probabilities']['down']:.1f}% down"
                    ],
                    "model_version": self.model_version,
                    "features_used": self.features_used + (["news_sentiment"] if sentiment_result else []),
                    "timestamp": prediction["timestamp"],
                    "ml_powered": True,
                    "sentiment_features": extract_sentiment_features(sentiment_result) if sentiment_result else {}
                }
            except Exception as e:
                print(f"[SignalScorer] ML prediction failed: {e}, falling back to rule-based scoring")

        # Fall back to rule-based scoring
        print(f"[SignalScorer] Using rule-based scoring (ML not available)")
        # Extract indicator values
        rsi_value = indicators.get('rsi', {}).get('value')
        macd_data = indicators.get('macd', {})
        mas = indicators.get('moving_averages', {})
        volume_data = indicators.get('volume', {})
        trend = indicators.get('trend', 'Unknown')
        price_position = indicators.get('price_position', {})

        # Initialize score (-100 to +100, where negative = bearish, positive = bullish)
        score = 0
        signals = []

        # RSI Scoring (-20 to +20)
        if rsi_value:
            if rsi_value < 30:
                score += 15
                signals.append(f"RSI oversold at {rsi_value:.1f} (bullish)")
            elif rsi_value < 40:
                score += 5
                signals.append(f"RSI low at {rsi_value:.1f} (slightly bullish)")
            elif rsi_value > 70:
                score -= 15
                signals.append(f"RSI overbought at {rsi_value:.1f} (bearish)")
            elif rsi_value > 60:
                score -= 5
                signals.append(f"RSI elevated at {rsi_value:.1f} (slightly bearish)")

        # MACD Scoring (-20 to +20)
        macd = macd_data.get('macd')
        signal_line = macd_data.get('signal')
        histogram = macd_data.get('histogram')

        if macd and signal_line and histogram:
            if macd > signal_line and histogram > 0:
                score += 15
                signals.append("MACD bullish crossover")
            elif macd < signal_line and histogram < 0:
                score -= 15
                signals.append("MACD bearish crossover")

            # Histogram momentum
            if histogram > 0.1:
                score += 5
                signals.append("Strong bullish MACD momentum")
            elif histogram < -0.1:
                score -= 5
                signals.append("Strong bearish MACD momentum")

        # Moving Average Alignment (-25 to +25)
        sma_20 = mas.get('SMA_20')
        sma_50 = mas.get('SMA_50')
        sma_200 = mas.get('SMA_200')

        if sma_20 and sma_50 and sma_200:
            if sma_20 > sma_50 > sma_200:
                score += 25
                signals.append("Perfect bullish MA alignment (20>50>200)")
            elif sma_20 < sma_50 < sma_200:
                score -= 25
                signals.append("Perfect bearish MA alignment (20<50<200)")
            elif sma_20 > sma_50:
                score += 10
                signals.append("Short-term bullish (20>50 MA)")
            elif sma_20 < sma_50:
                score -= 10
                signals.append("Short-term bearish (20<50 MA)")

        # Price Position vs MA (-15 to +15)
        vs_sma_20 = price_position.get('vs_SMA20', 0)
        if vs_sma_20 > 5:
            score += 10
            signals.append(f"Price {vs_sma_20:.1f}% above 20-day MA (bullish)")
        elif vs_sma_20 > 2:
            score += 5
        elif vs_sma_20 < -5:
            score -= 10
            signals.append(f"Price {abs(vs_sma_20):.1f}% below 20-day MA (bearish)")
        elif vs_sma_20 < -2:
            score -= 5

        # Volume Analysis (-10 to +10)
        volume_ratio = volume_data.get('ratio')
        if volume_ratio:
            if volume_ratio > 1.5 and score > 0:
                score += 10
                signals.append(f"High volume ({volume_ratio:.2f}x) confirming bullish move")
            elif volume_ratio > 1.5 and score < 0:
                score -= 10
                signals.append(f"High volume ({volume_ratio:.2f}x) confirming bearish move")
            elif volume_ratio < 0.5:
                # Low volume reduces confidence
                score = int(score * 0.8)
                signals.append(f"Low volume ({volume_ratio:.2f}x) - reduced confidence")

        # Trend Scoring (-10 to +10)
        if "Strong Uptrend" in trend:
            score += 10
            signals.append("Strong uptrend in place")
        elif "Uptrend" in trend:
            score += 5
            signals.append("Uptrend in place")
        elif "Strong Downtrend" in trend:
            score -= 10
            signals.append("Strong downtrend in place")
        elif "Downtrend" in trend:
            score -= 5
            signals.append("Downtrend in place")

        # Normalize score to -100 to +100
        score = max(-100, min(100, score))

        # Convert to confidence and direction
        direction = "BULLISH" if score > 0 else "BEARISH" if score < 0 else "NEUTRAL"
        confidence = abs(score)

        # Categorize confidence
        if confidence > 70:
            confidence_level = "Very High"
        elif confidence > 50:
            confidence_level = "High"
        elif confidence > 30:
            confidence_level = "Medium"
        elif confidence > 15:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"

        # Generate recommendation
        if direction == "BULLISH" and confidence > 50:
            recommendation = "STRONG BUY"
        elif direction == "BULLISH" and confidence > 30:
            recommendation = "BUY"
        elif direction == "BEARISH" and confidence > 50:
            recommendation = "STRONG SELL"
        elif direction == "BEARISH" and confidence > 30:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        return {
            "signal_score": score,
            "direction": direction,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "recommendation": recommendation,
            "signals": signals,
            "model_version": self.model_version,
            "features_used": self.features_used,
            "timestamp": datetime.now().isoformat()
        }

    def format_for_llm(self, signal_data: Dict[str, Any]) -> str:
        """
        Format signal scoring results for LLM consumption.

        Args:
            signal_data: Signal scoring results

        Returns:
            Formatted string
        """
        signals_text = "\n".join([f"  • {s}" for s in signal_data['signals']])

        formatted = f"""
ML Signal Scoring Analysis
{'=' * 60}

Overall Signal: {signal_data['direction']}
Recommendation: {signal_data['recommendation']}
Confidence: {signal_data['confidence']}/100 ({signal_data['confidence_level']})

Signal Breakdown:
{signals_text}

Model: {signal_data['model_version']}
Features Used: {', '.join(signal_data['features_used'])}

Interpretation:
A score of {signal_data['signal_score']:+d}/100 indicates {signal_data['direction'].lower()} sentiment
with {signal_data['confidence_level'].lower()} confidence based on technical indicators.
"""
        return formatted.strip()


# Future enhancement: Trained ML model
class TrainedSignalModel:
    """
    Placeholder for future trained ML model implementation.

    This would use:
    - RandomForestClassifier or LightGBM
    - Historical S&P 500 data for training
    - Features: RSI, MACD, MA relationships, volume patterns
    - Target: Price direction 30 days forward
    """

    def __init__(self):
        raise NotImplementedError(
            "Trained model not yet implemented. "
            "Currently using rule-based SignalScorer. "
            "See project_brief.md Phase 2 for training plan."
        )
