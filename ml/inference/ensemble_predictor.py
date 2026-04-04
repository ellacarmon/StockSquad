"""
Ensemble Predictor - Combines multiple ML models for better predictions.
Uses voting/averaging to generate more robust signals.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

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

        # Stacking meta-learner (fitted via fit_stacking_meta_learner)
        self._meta_learner: Optional[LogisticRegression] = None

        # Dynamic weighting: rolling accuracy tracker per model
        # {model_name: [accuracy_values]} — most recent entries used for weighting
        self._rolling_accuracy: Dict[str, List[float]] = {}

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
        elif self.strategy == "stacking":
            return self._stacking_strategy(predictions)
        elif self.strategy == "dynamic_weighting":
            return self._dynamic_weighting_strategy(predictions)
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

    def fit_stacking_meta_learner(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_models: dict,
    ) -> None:
        """
        Train a Logistic Regression meta-learner on out-of-fold predictions.

        Uses TimeSeriesSplit(n_splits=5) to generate OOF probabilities from each
        base model, then trains a LogisticRegression on those meta-features.

        Args:
            X: Feature DataFrame used to generate OOF predictions.
            y: Binary target series (0/1 direction labels).
            base_models: Dict mapping model name -> fitted sklearn-compatible classifier
                         with a predict_proba method.
        """
        tscv = TimeSeriesSplit(n_splits=5)
        n_samples = len(X)
        model_names = list(base_models.keys())

        # OOF probability matrix: one column per base model
        oof_probs = np.full((n_samples, len(model_names)), np.nan)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]

            for col_idx, model_name in enumerate(model_names):
                model = base_models[model_name]
                model.fit(X_train_fold, y_train_fold)
                probs = model.predict_proba(X_val_fold)
                # Use probability of positive class (index 1)
                oof_probs[val_idx, col_idx] = probs[:, 1]

        # Drop rows where any OOF prediction is missing (first fold train indices)
        valid_mask = ~np.isnan(oof_probs).any(axis=1)
        meta_X = oof_probs[valid_mask]
        meta_y = y.values[valid_mask]

        self._meta_learner = LogisticRegression(max_iter=1000)
        self._meta_learner.fit(meta_X, meta_y)

    def _stacking_strategy(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Stacking strategy: use the fitted meta-learner to combine base model predictions.

        If no meta-learner has been fitted, falls back to averaging.
        """
        if self._meta_learner is None:
            # Graceful fallback — no meta-learner fitted yet
            result = self._averaging_strategy(predictions)
            result['model_type'] = 'ensemble_stacking'
            result['stacking_note'] = 'meta-learner not fitted; used averaging fallback'
            return result

        model_names = list(predictions.keys())
        # Build meta-feature vector: one probability per base model
        meta_features = np.array([
            [predictions[name]['confidence'] / 100.0 for name in model_names]
        ])

        # Meta-learner predicts direction probability
        direction_prob = self._meta_learner.predict_proba(meta_features)[0, 1]  # P(bullish)
        confidence = round(direction_prob * 100, 1)

        if direction_prob > 0.55:
            direction = 'bullish'
        elif direction_prob < 0.45:
            direction = 'bearish'
        else:
            direction = 'neutral'

        avg_expected_return = float(np.mean([p['expected_return'] for p in predictions.values()]))
        recommendation = self._get_recommendation(direction, confidence, avg_expected_return)

        return {
            'model_type': 'ensemble_stacking',
            'direction': direction,
            'confidence': confidence,
            'expected_return': round(avg_expected_return, 2),
            'recommendation': recommendation,
            'direction_probability': round(direction_prob, 4),
            'num_models': len(predictions),
            'individual_predictions': predictions,
        }

    def update_model_weights(self, model_name: str, recent_accuracy: float) -> None:
        """
        Record a recent accuracy observation for a base model.

        Maintains a rolling window of up to 30 entries per model.

        Args:
            model_name: Name of the base model (e.g. "xgboost").
            recent_accuracy: Accuracy value in [0, 1] from a recent backtested period.
        """
        if model_name not in self._rolling_accuracy:
            self._rolling_accuracy[model_name] = []
        self._rolling_accuracy[model_name].append(recent_accuracy)
        # Keep only the last 30 observations (rolling 30-day window)
        self._rolling_accuracy[model_name] = self._rolling_accuracy[model_name][-30:]

    def _dynamic_weighting_strategy(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Dynamic weighting: weight each base model by its rolling average accuracy.

        Falls back to equal weights when no accuracy history is available.
        """
        model_names = list(predictions.keys())

        # Compute weights from rolling accuracy history
        weights = {}
        for name in model_names:
            history = self._rolling_accuracy.get(name, [])
            weights[name] = float(np.mean(history)) if history else 1.0

        total_weight = sum(weights.values())
        if total_weight == 0:
            total_weight = len(model_names)
            weights = {name: 1.0 for name in model_names}

        # Weighted average of confidence and expected return
        weighted_confidence = sum(
            predictions[name]['confidence'] * weights[name]
            for name in model_names
        ) / total_weight

        weighted_return = sum(
            predictions[name]['expected_return'] * weights[name]
            for name in model_names
        ) / total_weight

        # Determine direction from weighted confidence signals
        bullish_weight = sum(
            weights[name]
            for name in model_names
            if predictions[name]['direction'].lower() == 'bullish'
        )
        bearish_weight = sum(
            weights[name]
            for name in model_names
            if predictions[name]['direction'].lower() == 'bearish'
        )

        if bullish_weight > bearish_weight:
            direction = 'bullish'
        elif bearish_weight > bullish_weight:
            direction = 'bearish'
        else:
            direction = 'neutral'

        recommendation = self._get_recommendation(direction, weighted_confidence, weighted_return)

        return {
            'model_type': 'ensemble_dynamic_weighting',
            'direction': direction,
            'confidence': round(weighted_confidence, 1),
            'expected_return': round(weighted_return, 2),
            'recommendation': recommendation,
            'model_weights': {name: round(weights[name] / total_weight, 4) for name in model_names},
            'num_models': len(predictions),
            'individual_predictions': predictions,
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
