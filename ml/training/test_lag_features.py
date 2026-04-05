"""
Tests for FeatureEngineer.add_lag_features (task 1.1)

Validates: Requirements 1.1
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from ml.training.feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(close_prices):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(close_prices)
    return pd.DataFrame({
        "open":   close_prices,
        "high":   [p * 1.01 for p in close_prices],
        "low":    [p * 0.99 for p in close_prices],
        "close":  close_prices,
        "volume": [1_000_000] * n,
    })


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed for pure methods


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAddLagFeaturesUnit:
    """Unit tests with known inputs and expected outputs."""

    def test_columns_present_default_periods(self):
        """Output must contain return_1d, return_5d, return_10d, return_20d."""
        df = make_ohlcv([100.0] * 30)
        result = ENGINEER.add_lag_features(df)
        for col in ["return_1d", "return_5d", "return_10d", "return_20d"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_columns_present_custom_periods(self):
        """Custom periods produce correctly named columns."""
        df = make_ohlcv([100.0] * 15)
        result = ENGINEER.add_lag_features(df, periods=[3, 7])
        assert "return_3d" in result.columns
        assert "return_7d" in result.columns
        # Default columns should NOT be present when overriding periods
        assert "return_1d" not in result.columns

    def test_return_formula_correctness(self):
        """return_nd = (close - close.shift(n)) / close.shift(n)."""
        prices = [100.0, 110.0, 121.0, 133.1, 146.41]
        df = make_ohlcv(prices)
        result = ENGINEER.add_lag_features(df, periods=[1])

        # Row 1: (110 - 100) / 100 = 0.10
        assert abs(result["return_1d"].iloc[1] - 0.10) < 1e-9
        # Row 2: (121 - 110) / 110 ≈ 0.10
        assert abs(result["return_2d"].iloc[2] - 0.10) < 1e-9 if "return_2d" in result.columns else True
        assert abs(result["return_1d"].iloc[2] - (121 - 110) / 110) < 1e-9

    def test_first_n_rows_are_nan(self):
        """First n rows of return_nd must be NaN (no prior data)."""
        df = make_ohlcv(list(range(1, 26)))
        result = ENGINEER.add_lag_features(df, periods=[1, 5])
        assert result["return_1d"].iloc[0] != result["return_1d"].iloc[0]  # NaN check
        for i in range(5):
            assert pd.isna(result["return_5d"].iloc[i])

    def test_does_not_mutate_input(self):
        """Input DataFrame must not be modified."""
        df = make_ohlcv([100.0] * 25)
        original_cols = list(df.columns)
        ENGINEER.add_lag_features(df)
        assert list(df.columns) == original_cols

    def test_existing_columns_preserved(self):
        """All original columns must still be present in the output."""
        df = make_ohlcv([100.0] * 25)
        result = ENGINEER.add_lag_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_constant_price_returns_zero(self):
        """Constant price series should produce zero returns (after warm-up)."""
        df = make_ohlcv([50.0] * 25)
        result = ENGINEER.add_lag_features(df, periods=[1, 5])
        # After the first NaN rows, all returns should be 0
        assert (result["return_1d"].dropna() == 0.0).all()
        assert (result["return_5d"].dropna() == 0.0).all()


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    )
)
@settings(max_examples=200)
def test_lag_columns_always_present(prices):
    """
    **Validates: Requirements 1.1**

    For any valid OHLCV price DataFrame, after calling add_lag_features the
    output SHALL contain columns return_1d, return_5d, return_10d, return_20d.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_lag_features(df, periods=[1, 5, 10, 20])
    for col in ["return_1d", "return_5d", "return_10d", "return_20d"]:
        assert col in result.columns


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    )
)
@settings(max_examples=200)
def test_lag_return_formula_holds(prices):
    """
    **Validates: Requirements 1.1**

    For every non-NaN row i and period n, return_nd[i] == (close[i] - close[i-n]) / close[i-n].
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_lag_features(df, periods=[1, 5, 10, 20])
    close = result["close"].values

    for n, col in [(1, "return_1d"), (5, "return_5d"), (10, "return_10d"), (20, "return_20d")]:
        for i in range(n, len(close)):
            expected = (close[i] - close[i - n]) / close[i - n]
            actual = result[col].iloc[i]
            assert abs(actual - expected) < 1e-9, (
                f"Formula mismatch at row {i}, period {n}: expected {expected}, got {actual}"
            )


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    )
)
@settings(max_examples=200)
def test_first_n_rows_nan_for_each_period(prices):
    """
    **Validates: Requirements 1.1**

    The first n rows of return_nd SHALL be NaN because there is no prior data.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_lag_features(df, periods=[1, 5, 10, 20])

    for n, col in [(1, "return_1d"), (5, "return_5d"), (10, "return_10d"), (20, "return_20d")]:
        for i in range(n):
            assert pd.isna(result[col].iloc[i]), (
                f"Expected NaN at row {i} for period {n}, got {result[col].iloc[i]}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
