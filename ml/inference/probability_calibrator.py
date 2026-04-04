"""Probability calibration for ML classifiers.

Wraps sklearn's calibration utilities to apply Platt scaling or isotonic
regression to raw model probabilities, making confidence scores more meaningful.
"""

import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Calibrates raw classifier probabilities using Platt scaling or isotonic regression.

    The calibrator learns a mapping from raw probabilities → calibrated probabilities
    using a held-out calibration set. After fitting, `calibrate()` accepts a 1-D array
    of raw probabilities and returns calibrated values in [0, 1].

    Usage:
        calibrator = ProbabilityCalibrator()
        calibrator.fit(model, X_cal, y_cal, method="isotonic")
        calibrated_probs = calibrator.calibrate(raw_probs)
    """

    def __init__(self):
        self._calibrated = False
        self._calibration_fn = None  # callable: raw_prob -> calibrated_prob

    def fit(
        self,
        model,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = "isotonic",
    ) -> None:
        """Fit calibration on a held-out calibration set.

        Learns a mapping from the model's raw positive-class probabilities to
        calibrated probabilities using the provided calibration data.

        Args:
            model: A pre-fitted sklearn-compatible classifier with predict_proba.
            X_cal: Calibration feature DataFrame.
            y_cal: Calibration labels Series.
            method: "isotonic" or "platt" (Platt scaling uses logistic/sigmoid).
        """
        # Graceful fallback: too few samples
        if len(X_cal) < 100:
            logger.warning(
                "Calibration set has only %d samples (< 100). "
                "Skipping calibration — predictions will be uncalibrated.",
                len(X_cal),
            )
            self._calibrated = False
            return

        # Graceful fallback: single class
        unique_classes = y_cal.unique()
        if len(unique_classes) < 2:
            logger.warning(
                "Calibration set contains only one class (%s). "
                "Skipping calibration — predictions will be uncalibrated.",
                unique_classes[0],
            )
            self._calibrated = False
            return

        # Get raw positive-class probabilities from the pre-fitted model
        raw_probs = model.predict_proba(X_cal)[:, 1]
        y_arr = np.array(y_cal)

        if method == "isotonic":
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(raw_probs, y_arr)
            self._calibration_fn = calibrator.predict

        else:
            # "platt" scaling: fit a logistic regression on raw probs
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            lr.fit(raw_probs.reshape(-1, 1), y_arr)
            self._calibration_fn = lambda p: lr.predict_proba(
                np.array(p).reshape(-1, 1)
            )[:, 1]

        self._calibrated = True

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities clipped to [0, 1].

        Args:
            raw_probs: 1-D array of raw model probabilities for the positive class.

        Returns:
            np.ndarray of the same length with values in [0, 1].
        """
        raw_arr = np.asarray(raw_probs, dtype=float).ravel()

        if not self._calibrated or self._calibration_fn is None:
            return np.clip(raw_arr, 0.0, 1.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calibrated = self._calibration_fn(raw_arr)

        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)
