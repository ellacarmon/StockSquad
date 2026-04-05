"""
Property-based tests for target engineering methods in ModelTrainer.

**Validates: Requirements 4.1, 4.2**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from ml.training.train_models import ModelTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _return_series(draw, min_size: int = 1) -> pd.Series:
    """Draw a pd.Series of float returns."""
    size = draw(st.integers(min_value=min_size, max_value=300))
    values = draw(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Property 8: Threshold labeling produces no samples where |return| < t
# **Validates: Requirements 4.1**
# ---------------------------------------------------------------------------

@given(
    data=st.data(),
    threshold=st.floats(min_value=0.001, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property8_threshold_labeling_excludes_band(data, threshold):
    """
    Property 8: For any return series and threshold t > 0, threshold labeling
    produces no samples where |return| < t.

    **Validates: Requirements 4.1**
    """
    y_return = _return_series(data.draw)
    trainer = ModelTrainer(models_dir="/tmp/test_models")

    labels, valid_index = trainer.apply_threshold_labeling(y_return, threshold)

    # All retained returns must have |return| >= threshold
    retained_returns = y_return.loc[valid_index]
    assert (retained_returns.abs() >= threshold).all(), (
        f"Found samples with |return| < {threshold} in threshold-labeled output. "
        f"Min |return| in output: {retained_returns.abs().min()}"
    )

    # Labels must only be 0 or 1
    assert labels.isin([0, 1]).all(), "Labels must be binary (0 or 1)"

    # Positive returns → label 1, negative returns → label 0
    assert (labels[retained_returns > 0] == 1).all(), "Returns > threshold should be labeled 1"
    assert (labels[retained_returns < 0] == 0).all(), "Returns < -threshold should be labeled 0"


# ---------------------------------------------------------------------------
# Property 9: Tertile labels are approximately balanced (25%–42% each class)
# **Validates: Requirements 4.2**
# ---------------------------------------------------------------------------

@given(
    data=st.data(),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property9_tertile_labels_approximately_balanced(data):
    """
    Property 9: For any return distribution with >= 30 samples, tertile labeling
    produces three classes each containing between 25% and 42% of total samples.

    Uses a non-degenerate distribution (at least 3 distinct values) to ensure
    tertile boundaries are meaningful.

    **Validates: Requirements 4.2**
    """
    size = data.draw(st.integers(min_value=30, max_value=300))
    # Draw values ensuring at least 3 distinct values so tertile splits are non-trivial
    base_values = data.draw(
        st.lists(
            st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=size,
            max_size=size,
        )
    )
    # Ensure at least 3 distinct values by replacing first 3 with known distinct values
    # if the drawn list is degenerate
    values = list(base_values)
    if len(set(values)) < 3:
        values[0] = -1.0
        values[min(1, len(values) - 1)] = 0.0
        values[min(2, len(values) - 1)] = 1.0
    y_return = pd.Series(values, dtype=float)

    trainer = ModelTrainer(models_dir="/tmp/test_models")
    labels = trainer.apply_tertile_labeling(y_return)

    n = len(labels)
    assert n == len(y_return), "Output length must match input length"

    # All three classes must be present
    assert set(labels.unique()).issubset({0, 1, 2}), "Labels must be in {0, 1, 2}"

    for cls in [0, 1, 2]:
        count = (labels == cls).sum()
        pct = count / n
        assert 0.25 <= pct <= 0.42, (
            f"Class {cls} has {pct:.2%} of samples (expected 25%–42%). "
            f"n={n}, count={count}"
        )
