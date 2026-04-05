"""
Property-based tests for EnsemblePredictor.

**Validates: Requirements 6.1, 6.4**
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from unittest.mock import MagicMock, patch

from ml.inference.ensemble_predictor import EnsemblePredictor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {"direction", "confidence", "expected_return", "recommendation"}
VALID_DIRECTIONS = {"bullish", "bearish", "neutral"}
ALL_STRATEGIES = ["voting", "averaging", "unanimous", "stacking", "dynamic_weighting"]


def _make_prediction(direction: str, confidence: float, expected_return: float) -> dict:
    """Build a minimal prediction dict matching PredictionEngine output shape."""
    return {
        "direction": direction,
        "confidence": confidence,
        "expected_return": expected_return,
        "recommendation": "HOLD",
    }


def _make_ensemble(strategy: str) -> EnsemblePredictor:
    """Create an EnsemblePredictor with mocked PredictionEngine instances."""
    with patch("ml.inference.ensemble_predictor.PredictionEngine"):
        ep = EnsemblePredictor(models_dir="/fake/path", strategy=strategy)
    return ep


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

direction_st = st.sampled_from(["bullish", "bearish", "neutral"])
confidence_st = st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
return_st = st.floats(min_value=-20.0, max_value=20.0, allow_nan=False, allow_infinity=False)


@st.composite
def predictions_dict_st(draw):
    """Generate a dict of 1–5 base model predictions."""
    n = draw(st.integers(min_value=1, max_value=5))
    names = [f"model_{i}" for i in range(n)]
    return {
        name: _make_prediction(
            draw(direction_st),
            draw(confidence_st),
            draw(return_st),
        )
        for name in names
    }


# ---------------------------------------------------------------------------
# Property 11: every strategy returns required fields
# **Validates: Requirements 6.1, 6.4**
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ALL_STRATEGIES)
@given(predictions=predictions_dict_st())
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
def test_property_11_required_fields_present(strategy: str, predictions: dict):
    """
    Property 11: For any set of base model predictions, every ensemble strategy
    (voting, averaging, unanimous, stacking, dynamic_weighting) SHALL return a
    result dict containing direction, confidence, expected_return, and
    recommendation fields.

    **Validates: Requirements 6.1, 6.4**
    """
    ep = _make_ensemble(strategy)

    # Call the internal combine method directly to avoid PredictionEngine I/O
    if strategy == "voting":
        result = ep._voting_strategy(predictions)
    elif strategy == "averaging":
        result = ep._averaging_strategy(predictions)
    elif strategy == "unanimous":
        result = ep._unanimous_strategy(predictions)
    elif strategy == "stacking":
        result = ep._stacking_strategy(predictions)
    elif strategy == "dynamic_weighting":
        result = ep._dynamic_weighting_strategy(predictions)

    # All required fields must be present
    for field in REQUIRED_FIELDS:
        assert field in result, (
            f"Strategy '{strategy}' missing field '{field}'. Got keys: {list(result.keys())}"
        )

    # direction must be a valid value
    assert result["direction"] in VALID_DIRECTIONS, (
        f"Strategy '{strategy}' returned invalid direction: {result['direction']}"
    )

    # confidence must be a number
    assert isinstance(result["confidence"], (int, float)), (
        f"Strategy '{strategy}' confidence is not numeric: {result['confidence']}"
    )

    # expected_return must be a number
    assert isinstance(result["expected_return"], (int, float)), (
        f"Strategy '{strategy}' expected_return is not numeric: {result['expected_return']}"
    )

    # recommendation must be a non-empty string
    assert isinstance(result["recommendation"], str) and result["recommendation"], (
        f"Strategy '{strategy}' recommendation is empty or not a string"
    )


# ---------------------------------------------------------------------------
# Unit tests for stacking strategy
# ---------------------------------------------------------------------------

class TestStackingStrategy:
    def test_stacking_fallback_without_meta_learner(self):
        """Without a fitted meta-learner, stacking falls back to averaging."""
        ep = _make_ensemble("stacking")
        preds = {
            "xgboost": _make_prediction("bullish", 70.0, 2.0),
            "lightgbm": _make_prediction("bullish", 65.0, 1.5),
        }
        result = ep._stacking_strategy(preds)
        assert result["model_type"] == "ensemble_stacking"
        for field in REQUIRED_FIELDS:
            assert field in result

    def test_fit_stacking_meta_learner_stores_learner(self):
        """fit_stacking_meta_learner should populate self._meta_learner."""
        from sklearn.linear_model import LogisticRegression as LR
        from sklearn.datasets import make_classification

        ep = _make_ensemble("stacking")
        X_arr, y_arr = make_classification(n_samples=100, n_features=5, random_state=42)
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
        y = pd.Series(y_arr)

        # Create simple sklearn base models
        from sklearn.linear_model import LogisticRegression
        base_models = {
            "m1": LogisticRegression(max_iter=200),
            "m2": LogisticRegression(max_iter=200),
        }
        ep.fit_stacking_meta_learner(X, y, base_models)
        assert ep._meta_learner is not None
        assert isinstance(ep._meta_learner, LR)

    def test_stacking_with_fitted_meta_learner_returns_required_fields(self):
        """After fitting, stacking strategy must return all required fields."""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression

        ep = _make_ensemble("stacking")
        X_arr, y_arr = make_classification(n_samples=100, n_features=5, random_state=0)
        X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(5)])
        y = pd.Series(y_arr)

        base_models = {
            "m1": LogisticRegression(max_iter=200),
            "m2": LogisticRegression(max_iter=200),
        }
        ep.fit_stacking_meta_learner(X, y, base_models)

        preds = {
            "m1": _make_prediction("bullish", 72.0, 2.5),
            "m2": _make_prediction("bearish", 55.0, -1.0),
        }
        result = ep._stacking_strategy(preds)
        for field in REQUIRED_FIELDS:
            assert field in result


# ---------------------------------------------------------------------------
# Unit tests for dynamic_weighting strategy
# ---------------------------------------------------------------------------

class TestDynamicWeightingStrategy:
    def test_equal_weights_when_no_history(self):
        """With no accuracy history, all models get equal weight."""
        ep = _make_ensemble("dynamic_weighting")
        preds = {
            "xgboost": _make_prediction("bullish", 70.0, 2.0),
            "random_forest": _make_prediction("bullish", 60.0, 1.0),
            "lightgbm": _make_prediction("bullish", 65.0, 1.5),
        }
        result = ep._dynamic_weighting_strategy(preds)
        for field in REQUIRED_FIELDS:
            assert field in result
        # All weights should be equal (1/3 each)
        weights = result["model_weights"]
        assert abs(weights["xgboost"] - weights["random_forest"]) < 1e-6

    def test_update_model_weights_stores_values(self):
        """update_model_weights should record accuracy and cap at 30 entries."""
        ep = _make_ensemble("dynamic_weighting")
        for i in range(35):
            ep.update_model_weights("xgboost", 0.6)
        assert len(ep._rolling_accuracy["xgboost"]) == 30

    def test_higher_accuracy_model_gets_more_weight(self):
        """A model with higher rolling accuracy should receive a larger weight."""
        ep = _make_ensemble("dynamic_weighting")
        ep.update_model_weights("xgboost", 0.8)
        ep.update_model_weights("xgboost", 0.8)
        ep.update_model_weights("lightgbm", 0.4)
        ep.update_model_weights("lightgbm", 0.4)

        preds = {
            "xgboost": _make_prediction("bullish", 70.0, 2.0),
            "lightgbm": _make_prediction("bullish", 60.0, 1.0),
        }
        result = ep._dynamic_weighting_strategy(preds)
        assert result["model_weights"]["xgboost"] > result["model_weights"]["lightgbm"]

    def test_dynamic_weighting_returns_required_fields(self):
        ep = _make_ensemble("dynamic_weighting")
        ep.update_model_weights("xgboost", 0.7)
        preds = {
            "xgboost": _make_prediction("bullish", 70.0, 2.0),
            "random_forest": _make_prediction("bearish", 55.0, -1.0),
        }
        result = ep._dynamic_weighting_strategy(preds)
        for field in REQUIRED_FIELDS:
            assert field in result
