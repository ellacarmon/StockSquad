"""
ML Prediction Engine
Makes predictions using trained models.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ml.sentiment_features import extract_sentiment_features


class PredictionEngine:
    """Makes stock predictions using trained ML models."""

    def __init__(self, models_dir: str = "../models", model_type: str = "xgboost"):
        """
        Initialize prediction engine.

        Args:
            models_dir: Directory containing trained models
            model_type: Type of model to use (xgboost, random_forest, lightgbm)
        """
        self.models_dir = Path(models_dir)
        self.model_type = model_type

        # Load models
        self.classifier = self._load_model(f"{model_type}_direction_classifier.joblib")
        self.regressor = self._load_model(f"{model_type}_return_regressor.joblib")

        # Load feature names
        feature_path = self.models_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = None

        # Load metadata
        metadata_path = self.models_dir / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _load_model(self, filename: str) -> Optional[Any]:
        """
        Load a trained model.

        Args:
            filename: Model filename

        Returns:
            Loaded model or None
        """
        model_path = self.models_dir / filename
        if model_path.exists():
            return joblib.load(model_path)
        return None

    def prepare_features(
        self,
        indicators: Dict[str, Any],
        sentiment_result: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Prepare features from technical indicators.

        Args:
            indicators: Dictionary with technical indicators from TechnicalIndicators.calculate_all_indicators()
            sentiment_result: Optional SentimentAgent result for live numeric features

        Returns:
            DataFrame with features
        """
        # Get current price data
        current_price = indicators.get('current_price', 0)

        # Extract base OHLCV features (use current values as defaults)
        features = {
            'open': current_price,  # We don't have open in indicators, use close
            'high': indicators.get('price_high', current_price),
            'low': indicators.get('price_low', current_price),
            'close': current_price,
            'volume': indicators.get('volume', {}).get('current', 0),
        }

        # RSI - extract value from dict
        rsi_data = indicators.get('rsi', {})
        features['rsi_14'] = rsi_data.get('value', 50) if isinstance(rsi_data, dict) else 50

        # MACD - already in correct format
        macd_data = indicators.get('macd', {})
        features.update({
            'macd': macd_data.get('macd', 0),
            'macd_signal': macd_data.get('signal', 0),
            'macd_hist': macd_data.get('histogram', 0),
        })

        # Moving Averages - note the key is 'moving_averages' not 'sma'
        mas = indicators.get('moving_averages', {})
        features.update({
            'sma_20': mas.get('SMA_20', current_price),
            'sma_50': mas.get('SMA_50', current_price),
            'sma_200': mas.get('SMA_200', current_price),
        })

        # EMAs - calculate from SMAs if not provided
        # Note: We need to add EMA calculation to TechnicalIndicators or use approximations
        features.update({
            'ema_12': mas.get('SMA_20', current_price),  # Approximate with SMA_20
            'ema_26': mas.get('SMA_50', current_price),  # Approximate with SMA_50
        })

        # Bollinger Bands - note key is 'bollinger_bands' not 'bollinger'
        bb = indicators.get('bollinger_bands', {})
        features.update({
            'bb_upper': bb.get('upper', current_price),
            'bb_middle': bb.get('middle', current_price),
            'bb_lower': bb.get('lower', current_price),
        })

        # Volume indicators - extract from volume dict
        volume_data = indicators.get('volume', {})
        features['volume_sma_20'] = volume_data.get('average', features['volume'])

        # Derived features - these are already calculated in indicators
        price_pos = indicators.get('price_position', {})
        if price_pos.get('vs_SMA20') is not None:
            features['price_to_sma20'] = price_pos['vs_SMA20'] / 100  # Convert from percentage
        else:
            features['price_to_sma20'] = (current_price / features['sma_20']) - 1 if features['sma_20'] > 0 else 0

        if price_pos.get('vs_SMA50') is not None:
            features['price_to_sma50'] = price_pos['vs_SMA50'] / 100
        else:
            features['price_to_sma50'] = (current_price / features['sma_50']) - 1 if features['sma_50'] > 0 else 0

        # Volume ratio
        features['volume_ratio'] = volume_data.get('ratio', 1.0)

        # Optional sentiment features from SentimentAgent
        if sentiment_result is None and isinstance(indicators.get('sentiment_features'), dict):
            features.update(indicators['sentiment_features'])
        else:
            features.update(extract_sentiment_features(sentiment_result))

        # Create DataFrame
        df = pd.DataFrame([features])

        # Ensure correct column order
        if self.feature_names:
            # Only keep features that were in training
            df = df[[col for col in self.feature_names if col in df.columns]]

        # Convert all columns to numeric (fix any object types)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        return df

    def predict(
        self,
        indicators: Dict[str, Any],
        sentiment_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make prediction for a stock.

        Args:
            indicators: Dictionary with technical indicators

        Returns:
            Prediction dictionary
        """
        if self.classifier is None or self.regressor is None:
            return {
                "error": "Models not loaded",
                "direction": "neutral",
                "confidence": 0,
                "expected_return": 0.0,
                "recommendation": "HOLD"
            }

        try:
            # Prepare features
            X = self.prepare_features(indicators, sentiment_result=sentiment_result)
            print(f"[PredictionEngine] Prepared {X.shape[1]} features for prediction")

            # Classification prediction (direction)
            direction_prob = self.classifier.predict_proba(X)[0]
            direction_class = int(self.classifier.predict(X)[0])
            confidence = max(direction_prob) * 100  # 0-100

            # Regression prediction (expected return)
            expected_return = float(self.regressor.predict(X)[0])

            print(f"[PredictionEngine] Raw prediction: class={direction_class}, probs={direction_prob}, return={expected_return:.2f}%")

            # Determine direction and recommendation
            if direction_class == 1:
                direction = "bullish"
                if confidence > 70 and expected_return > 3:
                    recommendation = "STRONG BUY"
                elif confidence > 60 and expected_return > 1:
                    recommendation = "BUY"
                else:
                    recommendation = "HOLD"
            else:
                direction = "bearish"
                if confidence > 70 and expected_return < -3:
                    recommendation = "STRONG SELL"
                elif confidence > 60 and expected_return < -1:
                    recommendation = "SELL"
                else:
                    recommendation = "HOLD"

            # Calculate score (-100 to +100)
            if direction == "bullish":
                score = min(100, confidence + (expected_return * 5))
            else:
                score = max(-100, -confidence + (expected_return * 5))

            return {
                "model_type": self.model_type,
                "direction": direction,
                "confidence": round(confidence, 1),
                "expected_return": round(expected_return, 2),
                "score": round(score, 1),
                "recommendation": recommendation,
                "direction_probabilities": {
                    "down": round(direction_prob[0] * 100, 1),
                    "up": round(direction_prob[1] * 100, 1)
                },
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            print(f"[PredictionEngine] Error making prediction: {e}")
            return {
                "error": str(e),
                "direction": "neutral",
                "confidence": 0,
                "expected_return": 0.0,
                "recommendation": "HOLD"
            }

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Model information dictionary
        """
        return {
            "model_type": self.model_type,
            "classifier_loaded": self.classifier is not None,
            "regressor_loaded": self.regressor is not None,
            "features": self.feature_names,
            "metadata": self.metadata
        }


if __name__ == "__main__":
    # Example usage
    print("\n" + "="*70)
    print("ML PREDICTION ENGINE TEST")
    print("="*70 + "\n")

    # Initialize engine
    engine = PredictionEngine(model_type="xgboost")

    # Get model info
    info = engine.get_model_info()
    print("Model Info:")
    print(f"  Type: {info['model_type']}")
    print(f"  Classifier loaded: {info['classifier_loaded']}")
    print(f"  Regressor loaded: {info['regressor_loaded']}")
    print(f"  Features: {len(info['features']) if info['features'] else 0}")

    # Test prediction with sample data
    sample_indicators = {
        'open': 175.0,
        'high': 178.0,
        'low': 174.5,
        'close': 177.0,
        'volume': 50000000,
        'rsi': 58,
        'macd': {'macd': 1.5, 'signal': 1.2, 'histogram': 0.3},
        'sma': {'20': 175.0, '50': 170.0, '200': 165.0},
        'ema': {'12': 176.0, '26': 174.0},
        'bollinger': {'upper': 180.0, 'middle': 175.0, 'lower': 170.0},
        'volume_sma': 45000000
    }

    print("\n" + "="*70)
    print("SAMPLE PREDICTION")
    print("="*70 + "\n")

    prediction = engine.predict(sample_indicators)

    if "error" in prediction:
        print(f"❌ Prediction failed: {prediction['error']}")
        print("\nThis is expected if models haven't been trained yet.")
        print("Train models first: python3 ml/training/train_models.py")
    else:
        print(f"Direction: {prediction['direction'].upper()}")
        print(f"Confidence: {prediction['confidence']}%")
        print(f"Expected Return (5d): {prediction['expected_return']:+.2f}%")
        print(f"Score: {prediction.get('score', 0):.1f}/100")
        print(f"Recommendation: {prediction['recommendation']}")

        if 'direction_probabilities' in prediction:
            print(f"\nProbabilities:")
            print(f"  Down: {prediction['direction_probabilities']['down']}%")
            print(f"  Up: {prediction['direction_probabilities']['up']}%")
