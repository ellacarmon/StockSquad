"""
Property-based tests for ModelTrainer.

**Validates: Requirements 3.6**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ml.training.train_models import ModelTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_training_df(n_samples: int, n_features: int, seed: int = 0) -> tuple:
    """Return (X_train, y_class_train, X_val, y_class_val, X_reg_train, y_reg_train, X_reg_val, y_reg_val)."""
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{i}" for i in range(n_features)]

    split = max(1, int(n_samples * 0.8))

    X = pd.DataFrame(rng.standard_normal((n_samples, n_features)), columns=feature_names)
    y_class = pd.Series((rng.random(n_samples) > 0.5).astype(int))
    y_reg = pd.Series(rng.standard_normal(n_samples))

    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_class_train, y_class_val = y_class.iloc[:split], y_class.iloc[split:]
    y_reg_train, y_reg_val = y_reg.iloc[:split], y_reg.iloc[split:]

    return X_train, y_class_train, X_val, y_class_val, y_reg_train, y_reg_val


# ---------------------------------------------------------------------------
# Property 7: ModelEvaluation always contains feature importances
# **Validates: Requirements 3.6**
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["xgboost", "random_forest", "lightgbm"])
@given(
    n_samples=st.integers(min_value=60, max_value=200),
    n_features=st.integers(min_value=3, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_property7_feature_importances_present_classifier(model_type, n_samples, n_features, seed):
    """
    Property 7: For any completed training run, the returned evaluation object
    SHALL contain a non-empty feature_importances dict mapping feature names to
    non-negative importance scores.

    **Validates: Requirements 3.6**
    """
    trainer = ModelTrainer(models_dir="/tmp/test_models")
    X_train, y_class_train, X_val, y_class_val, y_reg_train, y_reg_val = _make_training_df(
        n_samples, n_features, seed
    )

    _, metrics = trainer.train_direction_classifier(
        X_train, y_class_train, X_val, y_class_val, model_type=model_type
    )

    # Must have feature_importances key
    assert "feature_importances" in metrics, (
        f"feature_importances missing from {model_type} classifier metrics"
    )

    fi = metrics["feature_importances"]

    # Must be a non-empty dict
    assert isinstance(fi, dict), "feature_importances must be a dict"
    assert len(fi) > 0, "feature_importances must be non-empty"

    # Keys must be feature names
    assert set(fi.keys()) == set(X_train.columns), (
        "feature_importances keys must match training feature names"
    )

    # All importance scores must be non-negative
    for feat, score in fi.items():
        assert score >= 0.0, (
            f"Importance score for '{feat}' is negative: {score}"
        )


@pytest.mark.parametrize("model_type", ["xgboost", "random_forest", "lightgbm"])
@given(
    n_samples=st.integers(min_value=60, max_value=200),
    n_features=st.integers(min_value=3, max_value=15),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
def test_property7_feature_importances_present_regressor(model_type, n_samples, n_features, seed):
    """
    Property 7 (regressor): For any completed training run, the returned evaluation
    object SHALL contain a non-empty feature_importances dict mapping feature names
    to non-negative importance scores.

    **Validates: Requirements 3.6**
    """
    trainer = ModelTrainer(models_dir="/tmp/test_models")
    X_train, y_class_train, X_val, y_class_val, y_reg_train, y_reg_val = _make_training_df(
        n_samples, n_features, seed
    )

    _, metrics = trainer.train_return_regressor(
        X_train, y_reg_train, X_val, y_reg_val, model_type=model_type
    )

    assert "feature_importances" in metrics, (
        f"feature_importances missing from {model_type} regressor metrics"
    )

    fi = metrics["feature_importances"]

    assert isinstance(fi, dict), "feature_importances must be a dict"
    assert len(fi) > 0, "feature_importances must be non-empty"
    assert set(fi.keys()) == set(X_train.columns), (
        "feature_importances keys must match training feature names"
    )

    for feat, score in fi.items():
        assert score >= 0.0, (
            f"Importance score for '{feat}' is negative: {score}"
        )
