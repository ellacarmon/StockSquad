"""
Ensemble Predictor - Combines multiple ML models for better predictions.
Uses voting/averaging to generate more robust signals.
"""

from typing import Dict, Any, List
from pathlib import Path
import numpy as np

from ml.inference.prediction_engine import PredictionEngine


class EnsemblePredictor:
    """
    Ensemble predictor that combines XGBoost, Random Forest, and LightGBM.

    Strategies:
    - Voting: Only trade when majority agrees on direction
    - Averaging: Average confidence and expected returns
    - Unanimous: Only trade when all models agree (most conservative)
    """

    def __init__(self, models_dir: str = None, strategy: str = "voting"):
        """
        Initialize ensemble predictor.

        Args:
            models_dir: Path to models directory
            strategy: Ensemble strategy ("voting", "averaging", "unanimous")
        """
        if models_dir is None:
            models_dir = str(Path(__file__).parent.parent / "models")

        self.strategy = strategy

        # Initialize all three models
        print(f"[EnsemblePredictor] Loading models with {strategy} strategy...")
        self.models = {
            'xgboost': PredictionEngine(models_dir=models_dir, model_type='xgboost'),
            'random_forest': PredictionEngine(models_dir=models_dir, model_type='random_forest'),
            'lightgbm': PredictionEngine(models_dir=models_dir, model_type='lightgbm')
        }
        print(f"[EnsemblePredictor] ✓ Loaded 3 models: XGBoost, Random Forest, LightGBM")

    def predict(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate ensemble prediction from all models.

        Args:
            indicators: Technical indicators dictionary

        Returns:
            Ensemble prediction with combined signals
        """
        # Get predictions from all models
        predictions = {}
        for model_name, model in self.models.items():
            try:
                pred = model.predict(indicators)
                if 'error' not in pred:
                    predictions[model_name] = pred
            except Exception as e:
                print(f"[EnsemblePredictor] {model_name} failed: {e}")

        # If no models succeeded, return error
        if not predictions:
            return {
                'error': 'All models failed',
                'direction': 'neutral',
                'confidence': 0,
                'expected_return': 0.0,
                'recommendation': 'HOLD'
            }

        # Apply ensemble strategy
        if self.strategy == "voting":
            return self._voting_strategy(predictions)
        elif self.strategy == "averaging":
            return self._averaging_strategy(predictions)
        elif self.strategy == "unanimous":
            return self._unanimous_strategy(predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _voting_strategy(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Majority voting: Trade when 2+ models agree on direction.

        Averages confidence and expected return from agreeing models.
        """
        # Count votes for each direction
        bullish_votes = []
        bearish_votes = []

        for model_name, pred in predictions.items():
            direction = pred['direction'].lower()
            if direction == 'bullish':
                bullish_votes.append(pred)
            elif direction == 'bearish':
                bearish_votes.append(pred)

        # Determine consensus
        if len(bullish_votes) > len(bearish_votes):
            consensus_direction = 'bullish'
            consensus_predictions = bullish_votes
        elif len(bearish_votes) > len(bullish_votes):
            consensus_direction = 'bearish'
            consensus_predictions = bearish_votes
        else:
            # Tie - no consensus
            consensus_direction = 'neutral'
            consensus_predictions = list(predictions.values())

        # Calculate ensemble metrics
        avg_confidence = np.mean([p['confidence'] for p in consensus_predictions])
        avg_expected_return = np.mean([p['expected_return'] for p in consensus_predictions])

        # Boost confidence if unanimous
        if len(consensus_predictions) == len(predictions) and consensus_direction != 'neutral':
            confidence_boost = 10  # Add 10% for unanimous agreement
            avg_confidence = min(100, avg_confidence + confidence_boost)

        # Calculate score
        if consensus_direction == 'bullish':
            score = min(100, avg_confidence + (avg_expected_return * 5))
        elif consensus_direction == 'bearish':
            score = max(-100, -avg_confidence + (avg_expected_return * 5))
        else:
            score = 0

        # Determine recommendation
        recommendation = self._get_recommendation(consensus_direction, avg_confidence, avg_expected_return)

        # Build model agreement summary
        agreement_summary = f"{len(consensus_predictions)}/{len(predictions)} models agree"
        model_votes = ", ".join([
            f"{name}: {pred['direction']}"
            for name, pred in predictions.items()
        ])

        return {
            'model_type': 'ensemble_voting',
            'direction': consensus_direction,
            'confidence': round(avg_confidence, 1),
            'expected_return': round(avg_expected_return, 2),
            'score': round(score, 1),
            'recommendation': recommendation,
            'ensemble_agreement': agreement_summary,
            'model_votes': model_votes,
            'num_models_agreeing': len(consensus_predictions),
            'total_models': len(predictions),
            'direction_probabilities': {
                'bullish': len(bullish_votes) / len(predictions) * 100,
                'bearish': len(bearish_votes) / len(predictions) * 100
            },
            'individual_predictions': predictions
        }

    def _averaging_strategy(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Simple averaging: Average all predictions regardless of agreement.

        Direction is determined by average expected return sign.
        """
        # Average all metrics
        avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
        avg_expected_return = np.mean([p['expected_return'] for p in predictions.values()])

        # Determine direction from average expected return
        if avg_expected_return > 0.5:
            direction = 'bullish'
        elif avg_expected_return < -0.5:
            direction = 'bearish'
        else:
            direction = 'neutral'

        # Calculate score
        if direction == 'bullish':
            score = min(100, avg_confidence + (avg_expected_return * 5))
        elif direction == 'bearish':
            score = max(-100, -avg_confidence + (avg_expected_return * 5))
        else:
            score = 0

        recommendation = self._get_recommendation(direction, avg_confidence, avg_expected_return)

        return {
            'model_type': 'ensemble_averaging',
            'direction': direction,
            'confidence': round(avg_confidence, 1),
            'expected_return': round(avg_expected_return, 2),
            'score': round(score, 1),
            'recommendation': recommendation,
            'num_models': len(predictions),
            'individual_predictions': predictions
        }

    def _unanimous_strategy(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Unanimous voting: Only trade when ALL models agree on direction.

        Most conservative - highest precision, lowest recall.
        """
        # Check if all models agree
        directions = [p['direction'].lower() for p in predictions.values()]

        if len(set(directions)) == 1 and directions[0] != 'neutral':
            # All agree!
            consensus_direction = directions[0]

            # Average metrics with high confidence boost
            avg_confidence = np.mean([p['confidence'] for p in predictions.values()])
            avg_expected_return = np.mean([p['expected_return'] for p in predictions.values()])

            # Big confidence boost for unanimous agreement
            avg_confidence = min(100, avg_confidence + 15)

            # Calculate score
            if consensus_direction == 'bullish':
                score = min(100, avg_confidence + (avg_expected_return * 5))
            else:
                score = max(-100, -avg_confidence + (avg_expected_return * 5))

            recommendation = self._get_recommendation(consensus_direction, avg_confidence, avg_expected_return)

            return {
                'model_type': 'ensemble_unanimous',
                'direction': consensus_direction,
                'confidence': round(avg_confidence, 1),
                'expected_return': round(avg_expected_return, 2),
                'score': round(score, 1),
                'recommendation': recommendation,
                'ensemble_agreement': f"ALL {len(predictions)} models agree!",
                'unanimous': True,
                'individual_predictions': predictions
            }
        else:
            # No consensus - return neutral
            return {
                'model_type': 'ensemble_unanimous',
                'direction': 'neutral',
                'confidence': 0,
                'expected_return': 0.0,
                'score': 0,
                'recommendation': 'HOLD',
                'ensemble_agreement': 'No unanimous agreement',
                'unanimous': False,
                'model_disagreement': ", ".join([
                    f"{name}: {pred['direction']}"
                    for name, pred in predictions.items()
                ]),
                'individual_predictions': predictions
            }

    def _get_recommendation(self, direction: str, confidence: float, expected_return: float) -> str:
        """Generate trading recommendation based on direction and confidence."""
        if direction == 'bullish':
            if confidence > 70 and expected_return > 3:
                return 'STRONG BUY'
            elif confidence > 60 and expected_return > 1:
                return 'BUY'
            else:
                return 'HOLD'
        elif direction == 'bearish':
            if confidence > 70 and expected_return < -3:
                return 'STRONG SELL'
            elif confidence > 60 and expected_return < -1:
                return 'SELL'
            else:
                return 'HOLD'
        else:
            return 'HOLD'
