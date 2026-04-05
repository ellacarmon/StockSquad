"""
Property-based tests for FeatureEngineer.select_features (task 1.6)

Validates: Requirements 1.7, 1.8, 1.9, 1.10
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from ml.training.feature_engineer import FeatureEngineer


ENGINEER = FeatureEngineer.__new__(FeatureEngineer)  # no DB needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_feature_df(n_rows: int, n_cols: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build a random feature DataFrame with no missing values."""
    data = rng.standard_normal((n_rows, n_cols))
    cols = [f"feat_{i}" for i in range(n_cols)]
    return pd.DataFrame(data, columns=cols)


def make_labels(n_rows: int, rng: np.random.Generator) -> pd.Series:
    """Build a binary label Series."""
    return pd.Series(rng.integers(0, 2, size=n_rows))


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestSelectFeaturesUnit:
    """Deterministic unit tests for select_features."""

    def test_returns_dataframe(self):
        rng = np.random.default_rng(0)
        X = make_feature_df(200, 10, rng)
        y = make_labels(200, rng)
        result = ENGINEER.select_features(X, y, top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_are_subset_of_input(self):
        rng = np.random.default_rng(1)
        X = make_feature_df(200, 15, rng)
        y = make_labels(200, rng)
        result = ENGINEER.select_features(X, y, top_n=10)
        assert set(result.columns).issubset(set(X.columns))

    def test_drops_high_missing_features(self):
        """Features with >5% NaN must not appear in output."""
        rng = np.random.default_rng(2)
        X = make_feature_df(200, 10, rng)
        # Inject >5% NaN into feat_0
        X.loc[:15, "feat_0"] = np.nan  # 16/200 = 8% missing
        y = make_labels(200, rng)
        result = ENGINEER.select_features(X, y, top_n=9)
        assert "feat_0" not in result.columns

    def test_drops_correlated_features(self):
        """One of a pair with |corr| > 0.95 must be dropped."""
        rng = np.random.default_rng(3)
        base = rng.standard_normal(200)
        # Build 12 independent features + 1 correlated pair so we have 13 total
        # (well above the <10 fallback threshold after pruning)
        indep = {f"ind_{i}": rng.standard_normal(200) for i in range(11)}
        X = pd.DataFrame({
            "a": base,
            "b": base + rng.standard_normal(200) * 0.01,  # corr ~0.9999 with a
            **indep,
        })
        y = make_labels(200, rng)
        result = ENGINEER.select_features(X, y, top_n=12)
        # Both 'a' and 'b' cannot both be in the result
        assert not ("a" in result.columns and "b" in result.columns)

    def test_top_n_respected(self):
        rng = np.random.default_rng(4)
        X = make_feature_df(300, 50, rng)
        y = make_labels(300, rng)
        result = ENGINEER.select_features(X, y, top_n=20)
        assert len(result.columns) <= 20

    def test_fallback_when_few_features_after_corr(self):
        """When <10 features survive correlation pruning, fallback keeps top min(20, top_n)."""
        rng = np.random.default_rng(5)
        # Build 12 features where 11 are nearly identical (high correlation)
        base = rng.standard_normal(300)
        data = {"independent": rng.standard_normal(300)}
        for i in range(11):
            data[f"corr_{i}"] = base + rng.standard_normal(300) * 0.001
        X = pd.DataFrame(data)
        y = make_labels(300, rng)
        # After corr pruning only ~2 features remain (< 10) → fallback to top min(20, top_n)
        result = ENGINEER.select_features(X, y, top_n=40)
        # Fallback: min(20, 40) = 20, but only 12 features exist so at most 12
        assert len(result.columns) <= 20

    def test_output_rows_unchanged(self):
        """Row count must be preserved."""
        rng = np.random.default_rng(6)
        X = make_feature_df(150, 20, rng)
        y = make_labels(150, rng)
        result = ENGINEER.select_features(X, y, top_n=10)
        assert len(result) == 150


# ---------------------------------------------------------------------------
# Property 2: output never has more than top_n columns
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=100, max_value=300),
    n_cols=st.integers(min_value=5, max_value=60),
    top_n=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=60_000)
def test_property2_output_never_exceeds_top_n(n_rows, n_cols, top_n, seed):
    """
    **Validates: Requirements 1.9**

    Property 2: For any input feature matrix and label series, the feature
    selection pipeline SHALL return a DataFrame with at most top_n columns.
    """
    rng = np.random.default_rng(seed)
    X = make_feature_df(n_rows, n_cols, rng)
    y = make_labels(n_rows, rng)
    result = ENGINEER.select_features(X, y, top_n=top_n)
    assert len(result.columns) <= top_n, (
        f"Expected <= {top_n} columns, got {len(result.columns)}"
    )


# ---------------------------------------------------------------------------
# Property 3: no pair of output features has |corr| >= 0.95
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=150, max_value=300),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=60_000)
def test_property3_no_high_correlation_in_output(n_rows, seed):
    """
    **Validates: Requirements 1.8**

    Property 3: For any pair of features in the selected feature set, their
    absolute Pearson correlation SHALL be less than 0.95.

    Uses synthetic data with injected correlated pairs to stress-test pruning.
    """
    rng = np.random.default_rng(seed)

    # Build 20 independent features
    n_independent = 20
    data = {f"ind_{i}": rng.standard_normal(n_rows) for i in range(n_independent)}

    # Inject 5 correlated pairs (|corr| > 0.95)
    for i in range(5):
        base = rng.standard_normal(n_rows)
        data[f"corr_a_{i}"] = base
        data[f"corr_b_{i}"] = base + rng.standard_normal(n_rows) * 0.05

    X = pd.DataFrame(data)
    y = make_labels(n_rows, rng)

    result = ENGINEER.select_features(X, y, top_n=15)

    # Skip check if fewer than 2 features selected
    if len(result.columns) < 2:
        return

    corr = result.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    max_corr = upper.stack().max() if not upper.stack().empty else 0.0

    assert max_corr < 0.95, (
        f"Found pair with |corr| = {max_corr:.4f} >= 0.95 in selected features"
    )


# ---------------------------------------------------------------------------
# Property 4: features with >5% missing values never appear in output
# ---------------------------------------------------------------------------

@given(
    n_rows=st.integers(min_value=100, max_value=300),
    n_cols=st.integers(min_value=5, max_value=40),
    n_high_missing=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=60_000)
def test_property4_high_missing_features_excluded(n_rows, n_cols, n_high_missing, seed):
    """
    **Validates: Requirements 1.7**

    Property 4: For any input feature matrix, features with more than 5%
    missing values SHALL NOT appear in the output of the feature selection
    pipeline.
    """
    assume(n_high_missing < n_cols)

    rng = np.random.default_rng(seed)
    X = make_feature_df(n_rows, n_cols, rng)
    y = make_labels(n_rows, rng)

    # Inject >5% NaN into the first n_high_missing columns
    high_missing_cols = [f"feat_{i}" for i in range(n_high_missing)]
    n_nan = int(n_rows * 0.10)  # 10% missing — clearly above threshold
    for col in high_missing_cols:
        nan_idx = rng.choice(n_rows, size=n_nan, replace=False)
        X.loc[nan_idx, col] = np.nan

    result = ENGINEER.select_features(X, y, top_n=min(n_cols - n_high_missing, 40))

    for col in high_missing_cols:
        assert col not in result.columns, (
            f"Feature '{col}' with >5% missing values appeared in output"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
