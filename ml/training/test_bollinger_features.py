"""
Property-based tests for FeatureEngineer.add_bollinger_features (task 1.5)

Validates: Requirements 1.5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from ml.training.feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bb_df(close_prices, bb_upper, bb_middle, bb_lower):
    """Build a minimal DataFrame with Bollinger Band columns."""
    n = len(close_prices)
    return pd.DataFrame({
        "close":     close_prices,
        "bb_upper":  bb_upper,
        "bb_middle": bb_middle,
        "bb_lower":  bb_lower,
    })


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed for pure methods


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAddBollingerFeaturesUnit:
    """Unit tests with known inputs and expected outputs."""

    def test_columns_present(self):
        """Output must contain bb_pct_b and bb_width."""
        df = make_bb_df(
            close_prices=[100.0] * 10,
            bb_upper=[105.0] * 10,
            bb_middle=[100.0] * 10,
            bb_lower=[95.0] * 10,
        )
        result = ENGINEER.add_bollinger_features(df)
        assert "bb_pct_b" in result.columns
        assert "bb_width" in result.columns

    def test_bb_pct_b_calculation(self):
        """bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)."""
        df = make_bb_df(
            close_prices=[100.0],
            bb_upper=[110.0],
            bb_middle=[100.0],
            bb_lower=[90.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        # (100 - 90) / (110 - 90) = 10/20 = 0.5
        assert abs(result["bb_pct_b"].iloc[0] - 0.5) < 1e-10

    def test_bb_width_calculation(self):
        """bb_width = (bb_upper - bb_lower) / bb_middle."""
        df = make_bb_df(
            close_prices=[100.0],
            bb_upper=[110.0],
            bb_middle=[100.0],
            bb_lower=[90.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        # (110 - 90) / 100 = 20/100 = 0.2
        assert abs(result["bb_width"].iloc[0] - 0.2) < 1e-10

    def test_bb_pct_b_nan_when_zero_bandwidth(self):
        """bb_pct_b must be NaN when bb_upper == bb_lower."""
        df = make_bb_df(
            close_prices=[100.0],
            bb_upper=[100.0],
            bb_middle=[100.0],
            bb_lower=[100.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        assert pd.isna(result["bb_pct_b"].iloc[0])

    def test_bb_width_nan_when_middle_zero(self):
        """bb_width must be NaN when bb_middle == 0."""
        df = make_bb_df(
            close_prices=[0.0],
            bb_upper=[1.0],
            bb_middle=[0.0],
            bb_lower=[-1.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        assert pd.isna(result["bb_width"].iloc[0])

    def test_does_not_mutate_input(self):
        """Input DataFrame must not be modified."""
        df = make_bb_df(
            close_prices=[100.0] * 5,
            bb_upper=[105.0] * 5,
            bb_middle=[100.0] * 5,
            bb_lower=[95.0] * 5,
        )
        original_cols = list(df.columns)
        ENGINEER.add_bollinger_features(df)
        assert list(df.columns) == original_cols

    def test_existing_columns_preserved(self):
        """All original columns must still be present in the output."""
        df = make_bb_df(
            close_prices=[100.0] * 5,
            bb_upper=[105.0] * 5,
            bb_middle=[100.0] * 5,
            bb_lower=[95.0] * 5,
        )
        result = ENGINEER.add_bollinger_features(df)
        for col in ["close", "bb_upper", "bb_middle", "bb_lower"]:
            assert col in result.columns

    def test_bb_pct_b_above_bands(self):
        """bb_pct_b > 1 when close is above bb_upper."""
        df = make_bb_df(
            close_prices=[115.0],
            bb_upper=[110.0],
            bb_middle=[100.0],
            bb_lower=[90.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        # (115 - 90) / (110 - 90) = 25/20 = 1.25
        assert result["bb_pct_b"].iloc[0] > 1.0

    def test_bb_pct_b_below_bands(self):
        """bb_pct_b < 0 when close is below bb_lower."""
        df = make_bb_df(
            close_prices=[85.0],
            bb_upper=[110.0],
            bb_middle=[100.0],
            bb_lower=[90.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        # (85 - 90) / (110 - 90) = -5/20 = -0.25
        assert result["bb_pct_b"].iloc[0] < 0.0

    def test_bb_width_non_negative_for_valid_bands(self):
        """bb_width >= 0 when bb_upper >= bb_lower and bb_middle > 0."""
        df = make_bb_df(
            close_prices=[100.0] * 5,
            bb_upper=[105.0, 106.0, 107.0, 108.0, 109.0],
            bb_middle=[100.0] * 5,
            bb_lower=[95.0, 94.0, 93.0, 92.0, 91.0],
        )
        result = ENGINEER.add_bollinger_features(df)
        valid = result["bb_width"].dropna()
        assert (valid >= 0).all()


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=1, max_value=200),
    middle=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    half_width=st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
    close_offset=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_bollinger_columns_always_present(n, middle, half_width, close_offset):
    """
    **Validates: Requirements 1.5**

    For any valid DataFrame with Bollinger Band columns, after calling
    add_bollinger_features the output SHALL contain bb_pct_b and bb_width columns.
    """
    df = make_bb_df(
        close_prices=[middle + close_offset] * n,
        bb_upper=[middle + half_width] * n,
        bb_middle=[middle] * n,
        bb_lower=[middle - half_width] * n,
    )
    result = ENGINEER.add_bollinger_features(df)
    assert "bb_pct_b" in result.columns, "Missing column: bb_pct_b"
    assert "bb_width" in result.columns, "Missing column: bb_width"


@given(
    n=st.integers(min_value=1, max_value=200),
    middle=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    half_width=st.floats(min_value=1e-6, max_value=1e4, allow_nan=False, allow_infinity=False),
    close_offset=st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300)
def test_bb_width_always_non_negative(n, middle, half_width, close_offset):
    """
    **Validates: Requirements 1.5**

    bb_width SHALL be >= 0 wherever it is not NaN, since bb_upper >= bb_lower
    (half_width > 0) and bb_middle > 0.
    """
    df = make_bb_df(
        close_prices=[middle + close_offset] * n,
        bb_upper=[middle + half_width] * n,
        bb_middle=[middle] * n,
        bb_lower=[middle - half_width] * n,
    )
    result = ENGINEER.add_bollinger_features(df)
    valid = result["bb_width"].dropna()
    assert (valid >= 0).all(), (
        f"Found negative bb_width values: {valid[valid < 0].tolist()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
