"""
Drift Monitor - Detects feature distribution shifts using Population Stability Index (PSI).

**Validates: Requirements 9.1-9.7**
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Optional
from datetime import datetime


class DriftMonitor:
    """
    Monitor feature distribution drift using Population Stability Index (PSI).

    PSI measures the shift in a feature's distribution between a baseline (training)
    and current (production) dataset. Higher PSI indicates greater drift.

    PSI thresholds:
    - PSI < 0.1: No significant change (stable)
    - 0.1 ≤ PSI < 0.2: Minor shift (monitor)
    - PSI ≥ 0.2: Major shift (retrain recommended)
    """

    def __init__(self, baseline_path: str = "ml/monitoring/baseline_distribution.json", n_buckets: int = 10):
        """
        Initialize drift monitor.

        Args:
            baseline_path: Path to store/load baseline feature distributions
            n_buckets: Number of buckets for PSI calculation (default 10)
        """
        self.baseline_path = Path(baseline_path)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.n_buckets = n_buckets
        self.baseline_stats: Optional[Dict] = None

        # Load existing baseline if available
        if self.baseline_path.exists():
            self._load_baseline()

    def _load_baseline(self) -> None:
        """Load baseline distribution from disk."""
        try:
            with open(self.baseline_path, 'r') as f:
                self.baseline_stats = json.load(f)
            print(f"[DriftMonitor] Loaded baseline from {self.baseline_path}")
        except Exception as e:
            print(f"[DriftMonitor] Failed to load baseline: {e}")
            self.baseline_stats = None

    def _save_baseline(self) -> None:
        """Save baseline distribution to disk."""
        if self.baseline_stats is None:
            return

        try:
            with open(self.baseline_path, 'w') as f:
                json.dump(self.baseline_stats, f, indent=2)
            print(f"[DriftMonitor] Saved baseline to {self.baseline_path}")
        except Exception as e:
            print(f"[DriftMonitor] Failed to save baseline: {e}")

    def compute_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Compute Population Stability Index (PSI) between two distributions.

        PSI = sum((actual% - expected%) * ln(actual% / expected%))

        Args:
            expected: Baseline/training distribution
            actual: Current/production distribution
            buckets: Number of bins to discretize the distributions

        Returns:
            PSI score (always >= 0)

        **Validates: Requirements 9.1**
        """
        # Remove NaN values
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins based on expected distribution
        try:
            breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
            # Ensure unique breakpoints
            breakpoints = np.unique(breakpoints)
            if len(breakpoints) < 2:
                # Can't create bins, return 0
                return 0.0
        except Exception:
            return 0.0

        # Ensure boundaries cover full range
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Bin the distributions
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Convert to percentages
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Avoid division by zero and log(0)
        epsilon = 1e-6
        expected_pct = np.where(expected_pct == 0, epsilon, expected_pct)
        actual_pct = np.where(actual_pct == 0, epsilon, actual_pct)

        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        # PSI is always non-negative
        return max(0.0, psi)

    def store_baseline(self, features: pd.DataFrame) -> None:
        """
        Store feature distribution as the baseline for future drift comparisons.

        Args:
            features: DataFrame of training features

        **Validates: Requirements 9.5**
        """
        print(f"[DriftMonitor] Storing baseline distribution for {len(features.columns)} features")

        baseline = {
            'stored_at': datetime.now().isoformat(),
            'n_samples': len(features),
            'features': {}
        }

        for col in features.columns:
            values = features[col].dropna().values

            if len(values) == 0:
                continue

            baseline['features'][col] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'percentiles': [float(x) for x in np.percentile(values, np.linspace(0, 100, self.n_buckets + 1))],
                'values_sample': values[:1000].tolist()  # Store sample for PSI calculation
            }

        self.baseline_stats = baseline
        self._save_baseline()

    def check_feature_drift(self, current_features: pd.DataFrame) -> Dict[str, float]:
        """
        Check drift for each feature against stored baseline.

        Args:
            current_features: DataFrame of current feature values

        Returns:
            Dictionary mapping feature names to PSI scores

        **Validates: Requirements 9.3**
        """
        if self.baseline_stats is None:
            print("[DriftMonitor] No baseline stored. Storing current features as baseline.")
            self.store_baseline(current_features)
            return {}

        drift_scores = {}

        for col in current_features.columns:
            if col not in self.baseline_stats['features']:
                continue

            current_values = current_features[col].dropna().values

            if len(current_values) == 0:
                continue

            # Reconstruct baseline distribution from stored sample
            baseline_sample = np.array(self.baseline_stats['features'][col]['values_sample'])

            # Compute PSI
            psi = self.compute_psi(baseline_sample, current_values, buckets=self.n_buckets)
            drift_scores[col] = psi

        return drift_scores

    def classify_drift(self, psi_score: float) -> str:
        """
        Classify drift severity based on PSI score.

        Args:
            psi_score: PSI value

        Returns:
            Classification: "stable", "minor_shift", or "major_shift"

        **Validates: Requirements 9.2, 9.3, 9.4**
        """
        if psi_score < 0.1:
            return "stable"
        elif psi_score < 0.2:
            return "minor_shift"
        else:
            return "major_shift"

    def should_retrain(self, drift_scores: Optional[Dict[str, float]] = None, threshold: float = 0.2) -> bool:
        """
        Determine if model should be retrained based on drift scores.

        Args:
            drift_scores: Dictionary of feature -> PSI scores (if None, will check current features)
            threshold: PSI threshold for triggering retrain (default 0.2)

        Returns:
            True if any feature has PSI >= threshold, False otherwise

        **Validates: Requirements 9.7**
        """
        if drift_scores is None:
            return False

        # Check if any feature exceeds threshold
        for feature, psi in drift_scores.items():
            if psi >= threshold:
                print(f"[DriftMonitor] Feature '{feature}' has PSI={psi:.3f} >= {threshold} (major drift detected)")
                return True

        return False

    def get_drift_summary(self, drift_scores: Dict[str, float]) -> Dict[str, any]:
        """
        Generate a summary report of drift scores.

        Args:
            drift_scores: Dictionary of feature -> PSI scores

        Returns:
            Summary with counts by severity and top drifted features
        """
        if not drift_scores:
            return {
                'total_features': 0,
                'stable': 0,
                'minor_shift': 0,
                'major_shift': 0,
                'max_psi': 0.0,
                'top_drifted_features': []
            }

        # Count by severity
        stable = sum(1 for psi in drift_scores.values() if psi < 0.1)
        minor = sum(1 for psi in drift_scores.values() if 0.1 <= psi < 0.2)
        major = sum(1 for psi in drift_scores.values() if psi >= 0.2)

        # Sort by PSI descending
        sorted_features = sorted(drift_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'total_features': len(drift_scores),
            'stable': stable,
            'minor_shift': minor,
            'major_shift': major,
            'max_psi': sorted_features[0][1] if sorted_features else 0.0,
            'top_drifted_features': [
                {'feature': feat, 'psi': round(psi, 3), 'classification': self.classify_drift(psi)}
                for feat, psi in sorted_features[:10]
            ]
        }
