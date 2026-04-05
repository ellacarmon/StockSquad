"""
Property-based tests for FeatureEngineer.add_volatility_features (task 1.2)

Validates: Requirements 1.2
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

def make_ohlcv(close_prices):
    """Build a minimal OHLCV DataFrame from a list of close prices."""
    n = len(close_prices)
    return pd.DataFrame({
        "open":   close_prices,
        "high":   [p * 1.02 for p in close_prices],
        "low":    [p * 0.98 for p in close_prices],
        "close":  close_prices,
        "volume": [1_000_000] * n,
    })


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed for pure methods


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAddVolatilityFeaturesUnit:
    """Unit tests with known inputs and expected outputs."""

    def test_columns_present(self):
        """Output must contain atr_14, realized_vol_20d, vol_regime."""
        df = make_ohlcv([100.0] * 80)
        result = ENGINEER.add_volatility_features(df)
        for col in ["atr_14", "realized_vol_20d", "vol_regime"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_atr_non_negative(self):
        """ATR values must be >= 0 where not NaN."""
        prices = [100.0 + i * 0.5 for i in range(80)]
        df = make_ohlcv(prices)
        result = ENGINEER.add_volatility_features(df)
        valid = result["atr_14"].dropna()
        assert (valid >= 0).all()

    def test_realized_vol_non_negative(self):
        """realized_vol_20d must be >= 0 where not NaN."""
        prices = [100.0 + i * 0.3 for i in range(80)]
        df = make_ohlcv(prices)
        result = ENGINEER.add_volatility_features(df)
        valid = result["realized_vol_20d"].dropna()
        assert (valid >= 0).all()

    def test_vol_regime_binary(self):
        """vol_regime must be 0 or 1 where not NaN."""
        prices = [100.0 + i * 0.1 for i in range(80)]
        df = make_ohlcv(prices)
        result = ENGINEER.add_volatility_features(df)
        valid = result["vol_regime"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_does_not_mutate_input(self):
        """Input DataFrame must not be modified."""
        df = make_ohlcv([100.0] * 80)
        original_cols = list(df.columns)
        ENGINEER.add_volatility_features(df)
        assert list(df.columns) == original_cols

    def test_existing_columns_preserved(self):
        """All original columns must still be present in the output."""
        df = make_ohlcv([100.0] * 80)
        result = ENGINEER.add_volatility_features(df)
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_atr_warmup_nans(self):
        """First 13 rows of atr_14 must be NaN (need 14 rows for first value)."""
        df = make_ohlcv([100.0] * 80)
        result = ENGINEER.add_volatility_features(df)
        for i in range(13):
            assert pd.isna(result["atr_14"].iloc[i])
        assert not pd.isna(result["atr_14"].iloc[13])

    def test_realized_vol_warmup_nans(self):
        """First 20 rows of realized_vol_20d must be NaN."""
        df = make_ohlcv([100.0] * 80)
        result = ENGINEER.add_volatility_features(df)
        for i in range(20):
            assert pd.isna(result["realized_vol_20d"].iloc[i])
        assert not pd.isna(result["realized_vol_20d"].iloc[20])

    def test_vol_regime_nan_when_realized_vol_nan(self):
        """vol_regime must be NaN wherever realized_vol_20d is NaN."""
        df = make_ohlcv([100.0] * 80)
        result = ENGINEER.add_volatility_features(df)
        vol_nan_mask = result["realized_vol_20d"].isna()
        assert result["vol_regime"][vol_nan_mask].isna().all()


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=80,
        max_size=300,
    )
)
@settings(max_examples=200)
def test_volatility_columns_always_present(prices):
    """
    **Validates: Requirements 1.2**

    For any valid OHLCV price DataFrame, after calling add_volatility_features the
    output SHALL contain columns atr_14, realized_vol_20d, and vol_regime.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_volatility_features(df)
    for col in ["atr_14", "realized_vol_20d", "vol_regime"]:
        assert col in result.columns, f"Missing column: {col}"


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=80,
        max_size=300,
    )
)
@settings(max_examples=200)
def test_atr_always_non_negative(prices):
    """
    **Validates: Requirements 1.2**

    atr_14 values SHALL be >= 0 wherever they are not NaN.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_volatility_features(df)
    valid = result["atr_14"].dropna()
    assert (valid >= 0).all(), f"Found negative ATR values: {valid[valid < 0].tolist()}"


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=80,
        max_size=300,
    )
)
@settings(max_examples=200)
def test_realized_vol_always_non_negative(prices):
    """
    **Validates: Requirements 1.2**

    realized_vol_20d values SHALL be >= 0 wherever they are not NaN.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_volatility_features(df)
    valid = result["realized_vol_20d"].dropna()
    assert (valid >= 0).all(), f"Found negative realized vol values: {valid[valid < 0].tolist()}"


@given(
    prices=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=80,
        max_size=300,
    )
)
@settings(max_examples=200)
def test_vol_regime_always_binary(prices):
    """
    **Validates: Requirements 1.2**

    vol_regime values SHALL be in {0, 1} wherever they are not NaN.
    """
    df = make_ohlcv(prices)
    result = ENGINEER.add_volatility_features(df)
    valid = result["vol_regime"].dropna()
    assert set(valid.unique()).issubset({0.0, 1.0}), (
        f"vol_regime contains values outside {{0, 1}}: {valid.unique().tolist()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
