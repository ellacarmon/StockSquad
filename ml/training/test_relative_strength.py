"""
Property-based tests for FeatureEngineer.add_relative_strength (task 1.3)

Validates: Requirements 1.3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ml.training.feature_engineer import FeatureEngineer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_multi_ticker_df(ticker_prices: dict) -> pd.DataFrame:
    """
    Build a multi-ticker DataFrame from a dict of {ticker: [close_prices]}.
    All tickers share the same date range (integer index used as date proxy).
    """
    frames = []
    for ticker, prices in ticker_prices.items():
        n = len(prices)
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "date": pd.date_range("2020-01-01", periods=n, freq="B"),
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1_000_000] * n,
        }))
    return pd.concat(frames, ignore_index=True)


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed for pure methods


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAddRelativeStrengthUnit:
    """Unit tests with known inputs and expected outputs."""

    def test_column_present_with_spy(self):
        """rs_vs_spy_20d must be present when SPY is in the DataFrame."""
        df = make_multi_ticker_df({
            "AAPL": [100.0 + i for i in range(30)],
            "SPY":  [300.0 + i for i in range(30)],
        })
        result = ENGINEER.add_relative_strength(df)
        assert "rs_vs_spy_20d" in result.columns

    def test_column_present_without_spy(self):
        """rs_vs_spy_20d must be present (all NaN) when SPY is absent."""
        df = make_multi_ticker_df({
            "AAPL": [100.0 + i for i in range(30)],
        })
        result = ENGINEER.add_relative_strength(df)
        assert "rs_vs_spy_20d" in result.columns
        assert result["rs_vs_spy_20d"].isna().all()

    def test_does_not_mutate_input(self):
        """Input DataFrame must not be modified."""
        df = make_multi_ticker_df({
            "AAPL": [100.0] * 30,
            "SPY":  [300.0] * 30,
        })
        original_cols = list(df.columns)
        ENGINEER.add_relative_strength(df)
        assert list(df.columns) == original_cols

    def test_existing_columns_preserved(self):
        """All original columns must still be present in the output."""
        df = make_multi_ticker_df({
            "AAPL": [100.0] * 30,
            "SPY":  [300.0] * 30,
        })
        result = ENGINEER.add_relative_strength(df)
        for col in ["ticker", "date", "open", "high", "low", "close", "volume"]:
            assert col in result.columns

    def test_no_temp_columns_leaked(self):
        """Internal helper columns (_ret20, _spy_ret20) must not appear in output."""
        df = make_multi_ticker_df({
            "AAPL": [100.0 + i for i in range(30)],
            "SPY":  [300.0 + i for i in range(30)],
        })
        result = ENGINEER.add_relative_strength(df)
        assert "_ret20" not in result.columns
        assert "_spy_ret20" not in result.columns

    def test_rs_formula_correctness(self):
        """
        rs_vs_spy_20d = stock_20d_return / spy_20d_return.
        Use a simple case: AAPL doubles over 20 days, SPY stays flat (1% gain).
        """
        # AAPL: 100 -> 200 over 20 days (100% return)
        aapl_prices = [100.0] * 20 + [200.0]
        # SPY: 300 -> 303 over 20 days (~1% return)
        spy_prices = [300.0] * 20 + [303.0]

        df = make_multi_ticker_df({"AAPL": aapl_prices, "SPY": spy_prices})
        result = ENGINEER.add_relative_strength(df)

        aapl_row = result[(result["ticker"] == "AAPL")].iloc[-1]
        expected_stock_ret = (200.0 - 100.0) / 100.0   # 1.0
        expected_spy_ret   = (303.0 - 300.0) / 300.0   # 0.01
        expected_rs = expected_stock_ret / expected_spy_ret  # 100.0

        assert abs(aapl_row["rs_vs_spy_20d"] - expected_rs) < 1e-6

    def test_spy_zero_return_produces_nan(self):
        """When SPY 20d return is 0, rs_vs_spy_20d must be NaN (avoid division by zero)."""
        # SPY flat for 21 rows → 20d return = 0
        spy_prices = [300.0] * 21
        aapl_prices = [100.0 + i for i in range(21)]
        df = make_multi_ticker_df({"AAPL": aapl_prices, "SPY": spy_prices})
        result = ENGINEER.add_relative_strength(df)
        # The last row has a valid 20d window; SPY return is 0 → NaN
        aapl_last = result[(result["ticker"] == "AAPL")].iloc[-1]
        assert pd.isna(aapl_last["rs_vs_spy_20d"])

    def test_warmup_rows_are_nan(self):
        """First 20 rows per ticker must be NaN (no 20-day window yet)."""
        df = make_multi_ticker_df({
            "AAPL": [100.0 + i for i in range(30)],
            "SPY":  [300.0 + i for i in range(30)],
        })
        result = ENGINEER.add_relative_strength(df)
        aapl = result[result["ticker"] == "AAPL"].reset_index(drop=True)
        for i in range(20):
            assert pd.isna(aapl["rs_vs_spy_20d"].iloc[i]), (
                f"Expected NaN at warmup row {i}, got {aapl['rs_vs_spy_20d'].iloc[i]}"
            )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    stock_prices=st.lists(
        st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    ),
    spy_prices=st.lists(
        st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    ),
)
@settings(max_examples=200)
def test_rs_vs_spy_column_always_present_with_spy(stock_prices, spy_prices):
    """
    **Validates: Requirements 1.3**

    For any valid multi-ticker DataFrame that includes SPY, after calling
    add_relative_strength the output SHALL contain the rs_vs_spy_20d column.
    """
    n = min(len(stock_prices), len(spy_prices))
    df = make_multi_ticker_df({
        "AAPL": stock_prices[:n],
        "SPY":  spy_prices[:n],
    })
    result = ENGINEER.add_relative_strength(df)
    assert "rs_vs_spy_20d" in result.columns


@given(
    stock_prices=st.lists(
        st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    ),
)
@settings(max_examples=200)
def test_rs_vs_spy_column_always_present_without_spy(stock_prices):
    """
    **Validates: Requirements 1.3**

    For any valid multi-ticker DataFrame that does NOT include SPY, after calling
    add_relative_strength the output SHALL contain the rs_vs_spy_20d column (all NaN).
    """
    df = make_multi_ticker_df({"AAPL": stock_prices})
    result = ENGINEER.add_relative_strength(df)
    assert "rs_vs_spy_20d" in result.columns
    assert result["rs_vs_spy_20d"].isna().all()


@given(
    stock_prices=st.lists(
        st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    ),
    spy_prices=st.lists(
        st.floats(min_value=1.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=25,
        max_size=200,
    ),
)
@settings(max_examples=200)
def test_rs_vs_spy_finite_when_both_returns_nonzero(stock_prices, spy_prices):
    """
    **Validates: Requirements 1.3**

    When SPY data is present, rs_vs_spy_20d SHALL be finite (not inf) wherever
    both the stock and SPY have non-zero 20-day returns.
    """
    n = min(len(stock_prices), len(spy_prices))
    df = make_multi_ticker_df({
        "AAPL": stock_prices[:n],
        "SPY":  spy_prices[:n],
    })
    result = ENGINEER.add_relative_strength(df)

    aapl = result[result["ticker"] == "AAPL"]
    # Only check rows where rs_vs_spy_20d is not NaN
    valid = aapl["rs_vs_spy_20d"].dropna()
    assert not np.isinf(valid).any(), (
        f"Found inf values in rs_vs_spy_20d: {valid[np.isinf(valid)].tolist()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
