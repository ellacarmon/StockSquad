"""
Property-based tests for FeatureEngineer.add_calendar_features (task 1.4)

Validates: Requirements 1.4
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

def make_df(n: int, tickers: list[str] | None = None) -> pd.DataFrame:
    """Build a minimal DataFrame with a 'date' column (and optionally 'ticker')."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    if tickers:
        frames = []
        for ticker in tickers:
            frames.append(pd.DataFrame({
                "ticker": ticker,
                "date": dates,
                "close": np.random.uniform(10, 500, n),
            }))
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame({
        "date": dates,
        "close": np.random.uniform(10, 500, n),
    })


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestAddCalendarFeaturesUnit:
    """Unit tests with known inputs."""

    def test_columns_present(self):
        df = make_df(10)
        result = ENGINEER.add_calendar_features(df)
        assert "day_of_week" in result.columns
        assert "month" in result.columns
        assert "days_to_earnings" in result.columns

    def test_does_not_mutate_input(self):
        df = make_df(10)
        original_cols = list(df.columns)
        ENGINEER.add_calendar_features(df)
        assert list(df.columns) == original_cols

    def test_day_of_week_known_date(self):
        # 2020-01-06 is a Monday → dayofweek == 0
        df = pd.DataFrame({"date": pd.to_datetime(["2020-01-06"])})
        result = ENGINEER.add_calendar_features(df)
        assert result["day_of_week"].iloc[0] == 0

    def test_month_known_date(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2020-03-15"])})
        result = ENGINEER.add_calendar_features(df)
        assert result["month"].iloc[0] == 3

    def test_days_to_earnings_first_row(self):
        """First row should have days_to_earnings == 63 (next multiple of 63 from pos 0)."""
        df = make_df(5)
        result = ENGINEER.add_calendar_features(df)
        assert result["days_to_earnings"].iloc[0] == 63

    def test_days_to_earnings_non_negative(self):
        df = make_df(200)
        result = ENGINEER.add_calendar_features(df)
        valid = result["days_to_earnings"].dropna()
        assert (valid >= 0).all()

    def test_no_date_column_fills_nan(self):
        df = pd.DataFrame({"close": [100.0, 101.0]})
        result = ENGINEER.add_calendar_features(df)
        assert result["day_of_week"].isna().all()
        assert result["month"].isna().all()
        assert result["days_to_earnings"].isna().all()

    def test_string_date_converted(self):
        df = pd.DataFrame({"date": ["2020-01-06", "2020-01-07"]})
        result = ENGINEER.add_calendar_features(df)
        assert result["day_of_week"].iloc[0] == 0  # Monday

    def test_per_ticker_days_to_earnings(self):
        """Each ticker's days_to_earnings should reset independently."""
        df = make_df(10, tickers=["AAPL", "MSFT"])
        result = ENGINEER.add_calendar_features(df)
        for ticker in ["AAPL", "MSFT"]:
            first_row = result[result["ticker"] == ticker].iloc[0]
            assert first_row["days_to_earnings"] == 63


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

@given(
    n=st.integers(min_value=1, max_value=300),
    start_date=st.dates(
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=pd.Timestamp("2030-01-01").date(),
    ),
)
@settings(max_examples=200)
def test_day_of_week_always_in_range(n, start_date):
    """
    **Validates: Requirements 1.4**

    For any valid DataFrame with a date column, day_of_week SHALL be in [0, 6]
    for all non-NaN values.
    """
    dates = pd.date_range(start=start_date, periods=n, freq="D")
    df = pd.DataFrame({"date": dates})
    result = ENGINEER.add_calendar_features(df)
    valid = result["day_of_week"].dropna()
    assert (valid >= 0).all() and (valid <= 6).all(), (
        f"day_of_week out of range [0,6]: {valid[~((valid >= 0) & (valid <= 6))].tolist()}"
    )


@given(
    n=st.integers(min_value=1, max_value=300),
    start_date=st.dates(
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=pd.Timestamp("2030-01-01").date(),
    ),
)
@settings(max_examples=200)
def test_month_always_in_range(n, start_date):
    """
    **Validates: Requirements 1.4**

    For any valid DataFrame with a date column, month SHALL be in [1, 12]
    for all non-NaN values.
    """
    dates = pd.date_range(start=start_date, periods=n, freq="D")
    df = pd.DataFrame({"date": dates})
    result = ENGINEER.add_calendar_features(df)
    valid = result["month"].dropna()
    assert (valid >= 1).all() and (valid <= 12).all(), (
        f"month out of range [1,12]: {valid[~((valid >= 1) & (valid <= 12))].tolist()}"
    )


@given(
    n=st.integers(min_value=1, max_value=300),
    start_date=st.dates(
        min_value=pd.Timestamp("2000-01-01").date(),
        max_value=pd.Timestamp("2030-01-01").date(),
    ),
)
@settings(max_examples=200)
def test_days_to_earnings_always_non_negative(n, start_date):
    """
    **Validates: Requirements 1.4**

    For any valid DataFrame with a date column, days_to_earnings SHALL be >= 0
    for all non-NaN values.
    """
    dates = pd.date_range(start=start_date, periods=n, freq="D")
    df = pd.DataFrame({"date": dates})
    result = ENGINEER.add_calendar_features(df)
    valid = result["days_to_earnings"].dropna()
    assert (valid >= 0).all(), (
        f"days_to_earnings has negative values: {valid[valid < 0].tolist()}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
