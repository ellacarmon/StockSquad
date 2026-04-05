"""
Property-based tests for HyperparameterTuner.

**Validates: Requirements 2.6, 2.7, 2.8, 2.9**

Property 5: For any completed Optuna study, every parameter in best_params
falls within the configured search bounds (e.g., XGBoost max_depth in [3,8],
learning_rate in [0.01, 0.3]).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from ml.training.hyperparameter_tuner import (
    HyperparameterTuner,
    DEFAULT_PARAMS,
    SEARCH_BOUNDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_classification_data(n_samples: int = 200, n_features: int = 5, seed: int = 0):
    """Generate a simple binary classification dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_samples, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.Series((X["f0"] + rng.standard_normal(n_samples) * 0.5 > 0).astype(int))
    return X, y


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestHyperparameterTunerUnit:
    """Unit tests for deterministic / structural behaviour."""

    def test_tune_xgboost_returns_dict(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="xgboost", n_trials=3, cv_splits=2)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_tune_lightgbm_returns_dict(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="lightgbm", n_trials=3, cv_splits=2)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_tune_random_forest_returns_dict(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="random_forest", n_trials=3, cv_splits=2)
        assert isinstance(params, dict)
        assert len(params) > 0

    def test_invalid_model_type_raises(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        with pytest.raises(ValueError, match="Unknown model_type"):
            tuner.tune(X, y, model_type="svm", n_trials=2, cv_splits=2)

    def test_get_study_results_before_tune_returns_empty(self):
        tuner = HyperparameterTuner()
        df = tuner.get_study_results()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["trial_number", "params", "value", "state"]
        assert len(df) == 0

    def test_get_study_results_after_tune_has_correct_columns(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        tuner.tune(X, y, model_type="xgboost", n_trials=3, cv_splits=2)
        df = tuner.get_study_results()
        assert set(df.columns) == {"trial_number", "params", "value", "state"}

    def test_get_study_results_trial_count_matches_n_trials(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        n_trials = 4
        tuner.tune(X, y, model_type="xgboost", n_trials=n_trials, cv_splits=2)
        df = tuner.get_study_results()
        assert len(df) == n_trials

    def test_fallback_to_defaults_when_all_trials_fail(self, monkeypatch):
        """When every trial returns -inf, best_params should be the defaults."""
        X, y = make_classification_data()
        tuner = HyperparameterTuner()

        # Force all trials to fail by patching _build_model to raise
        def bad_build(model_type, params):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(HyperparameterTuner, "_build_model", staticmethod(bad_build))
        params = tuner.tune(X, y, model_type="xgboost", n_trials=4, cv_splits=2)
        assert params == DEFAULT_PARAMS["xgboost"]

    def test_xgboost_params_within_bounds(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="xgboost", n_trials=5, cv_splits=2)
        bounds = SEARCH_BOUNDS["xgboost"]
        for key, (lo, hi) in bounds.items():
            if key in params:
                assert lo <= params[key] <= hi, (
                    f"xgboost.{key}={params[key]} outside [{lo}, {hi}]"
                )

    def test_lightgbm_params_within_bounds(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="lightgbm", n_trials=5, cv_splits=2)
        bounds = SEARCH_BOUNDS["lightgbm"]
        for key, (lo, hi) in bounds.items():
            if key in params:
                assert lo <= params[key] <= hi, (
                    f"lightgbm.{key}={params[key]} outside [{lo}, {hi}]"
                )

    def test_random_forest_params_within_bounds(self):
        X, y = make_classification_data()
        tuner = HyperparameterTuner()
        params = tuner.tune(X, y, model_type="random_forest", n_trials=5, cv_splits=2)
        bounds = SEARCH_BOUNDS["random_forest"]
        for key, (lo, hi) in bounds.items():
            if key in params:
                assert lo <= params[key] <= hi, (
                    f"random_forest.{key}={params[key]} outside [{lo}, {hi}]"
                )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    n_samples=st.integers(min_value=100, max_value=300),
    n_features=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_property5_xgboost_params_within_bounds(n_samples, n_features, seed):
    """
    **Validates: Requirements 2.6, 2.7**

    Property 5 (XGBoost): For any completed Optuna study, every numeric
    parameter in best_params SHALL fall within the configured search bounds.
    """
    X, y = make_classification_data(n_samples=n_samples, n_features=n_features, seed=seed)
    tuner = HyperparameterTuner()
    params = tuner.tune(X, y, model_type="xgboost", n_trials=5, cv_splits=2)

    bounds = SEARCH_BOUNDS["xgboost"]
    for key, (lo, hi) in bounds.items():
        if key in params:
            assert lo <= params[key] <= hi, (
                f"Property 5 violated: xgboost.{key}={params[key]} not in [{lo}, {hi}]"
            )


@given(
    n_samples=st.integers(min_value=100, max_value=300),
    n_features=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_property5_lightgbm_params_within_bounds(n_samples, n_features, seed):
    """
    **Validates: Requirements 2.8**

    Property 5 (LightGBM): For any completed Optuna study, every numeric
    parameter in best_params SHALL fall within the configured search bounds.
    """
    X, y = make_classification_data(n_samples=n_samples, n_features=n_features, seed=seed)
    tuner = HyperparameterTuner()
    params = tuner.tune(X, y, model_type="lightgbm", n_trials=5, cv_splits=2)

    bounds = SEARCH_BOUNDS["lightgbm"]
    for key, (lo, hi) in bounds.items():
        if key in params:
            assert lo <= params[key] <= hi, (
                f"Property 5 violated: lightgbm.{key}={params[key]} not in [{lo}, {hi}]"
            )


@given(
    n_samples=st.integers(min_value=100, max_value=300),
    n_features=st.integers(min_value=3, max_value=10),
    seed=st.integers(min_value=0, max_value=999),
)
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)
def test_property5_random_forest_params_within_bounds(n_samples, n_features, seed):
    """
    **Validates: Requirements 2.9**

    Property 5 (Random Forest): For any completed Optuna study, every numeric
    parameter in best_params SHALL fall within the configured search bounds.
    """
    X, y = make_classification_data(n_samples=n_samples, n_features=n_features, seed=seed)
    tuner = HyperparameterTuner()
    params = tuner.tune(X, y, model_type="random_forest", n_trials=5, cv_splits=2)

    bounds = SEARCH_BOUNDS["random_forest"]
    for key, (lo, hi) in bounds.items():
        if key in params:
            assert lo <= params[key] <= hi, (
                f"Property 5 violated: random_forest.{key}={params[key]} not in [{lo}, {hi}]"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
