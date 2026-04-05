"""Property-based and unit tests for ProbabilityCalibrator.

**Validates: Requirements 5.1, 5.2, 5.3**
"""

import logging

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from ml.inference.probability_calibrator import ProbabilityCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fitted_lr(n_samples: int = 300, n_features: int = 4, random_state: int = 42):
    """Return a pre-fitted LogisticRegression and matching calibration data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=2,
        random_state=random_state,
    )
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    y_ser = pd.Series(y)

    split = n_samples // 2
    model = LogisticRegression(max_iter=200)
    model.fit(X_df.iloc[:split], y_ser.iloc[:split])

    return model, X_df.iloc[split:], y_ser.iloc[split:]


# ---------------------------------------------------------------------------
# Property 10: calibrated probabilities always in [0, 1]
# **Validates: Requirements 5.3**
# ---------------------------------------------------------------------------

@given(
    raw_probs=st.lists(
        st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=500,
    )
)
@settings(max_examples=200)
def test_property_10_calibrated_probs_in_unit_interval_unfitted(raw_probs):
    """Property 10: For any raw probability array, calibrate() returns values in [0, 1].

    Tests the fallback path (calibrator not fitted).
    **Validates: Requirements 5.3**
    """
    calibrator = ProbabilityCalibrator()
    arr = np.array(raw_probs, dtype=float)
    result = calibrator.calibrate(arr)

    assert len(result) == len(arr), "Output length must match input length"
    assert np.all(result >= 0.0), f"Found value < 0: {result[result < 0]}"
    assert np.all(result <= 1.0), f"Found value > 1: {result[result > 1]}"


@given(
    raw_probs=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=200,
    )
)
@settings(max_examples=100)
def test_property_10_calibrated_probs_in_unit_interval_fitted(raw_probs):
    """Property 10: calibrate() returns values in [0, 1] even when fitted.

    Passes arbitrary [0,1] floats as raw probabilities to the fitted calibrator.
    **Validates: Requirements 5.3**
    """
    model, X_cal, y_cal = _make_fitted_lr()
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model, X_cal, y_cal, method="isotonic")

    arr = np.array(raw_probs, dtype=float)
    result = calibrator.calibrate(arr)

    assert len(result) == len(arr), "Output length must match input length"
    assert np.all(result >= 0.0), f"Found value < 0: {result[result < 0]}"
    assert np.all(result <= 1.0), f"Found value > 1: {result[result > 1]}"


# ---------------------------------------------------------------------------
# Unit tests — fit method (Requirements 5.1, 5.2)
# ---------------------------------------------------------------------------

def test_fit_isotonic_sets_calibrated_flag():
    """fit() with 'isotonic' method marks calibrator as fitted."""
    model, X_cal, y_cal = _make_fitted_lr()
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model, X_cal, y_cal, method="isotonic")
    assert calibrator._calibrated is True
    assert calibrator._calibration_fn is not None


def test_fit_platt_sets_calibrated_flag():
    """fit() with 'platt' method (Platt scaling) marks calibrator as fitted."""
    model, X_cal, y_cal = _make_fitted_lr()
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model, X_cal, y_cal, method="platt")
    assert calibrator._calibrated is True
    assert calibrator._calibration_fn is not None


def test_fit_too_few_samples_skips_calibration(caplog):
    """fit() skips calibration and logs warning when fewer than 100 samples."""
    model, X_cal, y_cal = _make_fitted_lr()
    small_X = X_cal.iloc[:50]
    small_y = y_cal.iloc[:50]

    calibrator = ProbabilityCalibrator()
    with caplog.at_level(logging.WARNING, logger="ml.inference.probability_calibrator"):
        calibrator.fit(model, small_X, small_y)

    assert calibrator._calibrated is False
    assert "Skipping calibration" in caplog.text


def test_fit_single_class_skips_calibration(caplog):
    """fit() skips calibration and logs warning when all labels are the same class."""
    model, X_cal, y_cal = _make_fitted_lr()
    single_class_y = pd.Series([0] * len(y_cal), index=y_cal.index)

    calibrator = ProbabilityCalibrator()
    with caplog.at_level(logging.WARNING, logger="ml.inference.probability_calibrator"):
        calibrator.fit(model, X_cal, single_class_y)

    assert calibrator._calibrated is False
    assert "Skipping calibration" in caplog.text


# ---------------------------------------------------------------------------
# Unit tests — calibrate method (Requirement 5.3)
# ---------------------------------------------------------------------------

def test_calibrate_unfitted_clips_to_unit_interval():
    """calibrate() on unfitted calibrator clips values to [0, 1]."""
    calibrator = ProbabilityCalibrator()
    raw = np.array([-0.5, 0.0, 0.5, 1.0, 1.5])
    result = calibrator.calibrate(raw)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
    assert len(result) == len(raw)


def test_calibrate_preserves_length():
    """calibrate() always returns an array of the same length as input."""
    model, X_cal, y_cal = _make_fitted_lr()
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model, X_cal, y_cal)

    for size in [1, 10, 100]:
        raw = np.random.rand(size)
        result = calibrator.calibrate(raw)
        assert len(result) == size


def test_calibrate_output_in_unit_interval_after_fit():
    """calibrate() returns values in [0, 1] after successful fit."""
    model, X_cal, y_cal = _make_fitted_lr()
    calibrator = ProbabilityCalibrator()
    calibrator.fit(model, X_cal, y_cal, method="isotonic")

    raw = np.linspace(0.0, 1.0, 50)
    result = calibrator.calibrate(raw)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
