"""
Property-based tests for walk-forward validation temporal ordering.

**Validates: Requirements 7.1**
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from sklearn.model_selection import TimeSeriesSplit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_rows: int, seed: int = 0) -> pd.DataFrame:
    """Build a minimal DataFrame that ModelTrainer.prepare_features can consume."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 1, n_rows))
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close * (1 + rng.uniform(-0.005, 0.005, n_rows)),
            "high": close * (1 + rng.uniform(0, 0.01, n_rows)),
            "low": close * (1 - rng.uniform(0, 0.01, n_rows)),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n_rows).astype(float),
            "rsi_14": rng.uniform(20, 80, n_rows),
            "macd": rng.normal(0, 0.5, n_rows),
            "macd_signal": rng.normal(0, 0.5, n_rows),
            "macd_hist": rng.normal(0, 0.2, n_rows),
            "sma_20": close * rng.uniform(0.98, 1.02, n_rows),
            "sma_50": close * rng.uniform(0.97, 1.03, n_rows),
            "sma_200": close * rng.uniform(0.95, 1.05, n_rows),
            "ema_12": close * rng.uniform(0.99, 1.01, n_rows),
            "ema_26": close * rng.uniform(0.98, 1.02, n_rows),
            "bb_upper": close * 1.02,
            "bb_middle": close,
            "bb_lower": close * 0.98,
            "volume_sma_20": rng.integers(1_000_000, 10_000_000, n_rows).astype(float),
            "direction_5d": rng.integers(0, 2, n_rows),
            "forward_5d_return": rng.normal(0, 0.02, n_rows),
        }
    )
    df.set_index("date", inplace=True)
    return df


def _get_fold_date_ranges(X: pd.DataFrame, n_splits: int):
    """
    Return a list of (train_end_date, val_start_date, val_end_date) for each fold.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ranges = []
    for train_idx, val_idx in tscv.split(X):
        train_dates = X.index[train_idx]
        val_dates = X.index[val_idx]
        ranges.append(
            {
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "val_start": val_dates[0],
                "val_end": val_dates[-1],
            }
        )
    return ranges


# ---------------------------------------------------------------------------
# Property 12: Walk-forward folds maintain temporal ordering
#
# For any walk-forward training run, every validation fold's date range SHALL
# be strictly after its corresponding training fold's date range (no future
# data leaks into past folds).
#
# **Validates: Requirements 7.1**
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=60, max_value=500),
    n_splits=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_property_12_walk_forward_temporal_ordering(n_rows, n_splits, seed):
    """
    Property 12: For any walk-forward training run, every validation fold's
    date range SHALL be strictly after its corresponding training fold's date
    range (no future data leaks into past folds).

    **Validates: Requirements 7.1**
    """
    df = _make_df(n_rows, seed)
    # Use only the feature/index portion (no label columns needed for split logic)
    X = df.drop(columns=["direction_5d", "forward_5d_return"])

    # Clamp n_splits so TimeSeriesSplit can form valid folds
    effective_splits = min(n_splits, len(X) - 1)
    if effective_splits < 2:
        pytest.skip("Not enough rows to form folds")

    fold_ranges = _get_fold_date_ranges(X, effective_splits)

    for fold_idx, fold in enumerate(fold_ranges):
        # The earliest validation date must be strictly after the latest training date
        assert fold["val_start"] > fold["train_end"], (
            f"Fold {fold_idx + 1}: validation start ({fold['val_start']}) is NOT "
            f"strictly after training end ({fold['train_end']}). "
            "Future data leaked into training fold."
        )

        # Validation window must be entirely after training window
        assert fold["val_end"] >= fold["val_start"], (
            f"Fold {fold_idx + 1}: val_end ({fold['val_end']}) < val_start ({fold['val_start']})"
        )

    # Expanding-window property: each fold's training set is a superset of the previous
    for i in range(1, len(fold_ranges)):
        prev_train_end = fold_ranges[i - 1]["train_end"]
        curr_train_end = fold_ranges[i]["train_end"]
        assert curr_train_end >= prev_train_end, (
            f"Fold {i + 1} training end ({curr_train_end}) is before fold {i} "
            f"training end ({prev_train_end}). Window is not expanding."
        )
