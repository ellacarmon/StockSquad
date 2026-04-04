"""
HyperparameterTuner: Optuna-based Bayesian hyperparameter optimization.

Uses TimeSeriesSplit cross-validation to evaluate each trial, ensuring
no future data leaks into past folds during tuning.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Default parameters (fallback when >50% of trials fail)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
    },
    "lightgbm": {
        "num_leaves": 31,
        "min_data_in_leaf": 20,
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
    },
}

# ---------------------------------------------------------------------------
# Search space bounds (used for validation in tests)
# ---------------------------------------------------------------------------

SEARCH_BOUNDS = {
    "xgboost": {
        "n_estimators": (100, 500),
        "max_depth": (3, 8),
        "learning_rate": (0.01, 0.3),
        "min_child_weight": (1, 10),
        "gamma": (0.0, 5.0),
        "reg_alpha": (0.0, 10.0),
        "reg_lambda": (0.0, 10.0),
    },
    "lightgbm": {
        "num_leaves": (20, 150),
        "min_data_in_leaf": (10, 100),
        "feature_fraction": (0.5, 1.0),
        "bagging_fraction": (0.5, 1.0),
        "lambda_l1": (0.0, 10.0),
        "lambda_l2": (0.0, 10.0),
    },
    "random_forest": {
        "n_estimators": (100, 500),
        "max_depth": (5, 20),
        "min_samples_split": (2, 20),
    },
}


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization using Optuna with time-series CV.

    Supports XGBoost, LightGBM, and Random Forest model types.
    Falls back to default parameters when more than 50% of trials fail.
    """

    def __init__(self):
        self._study: Optional[optuna.Study] = None
        self._model_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str,
        n_trials: int = 50,
        cv_splits: int = 5,
    ) -> dict:
        """
        Run Optuna Bayesian search and return the best hyperparameter dict.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (rows must be in chronological order).
        y : pd.Series
            Target labels.
        model_type : str
            One of "xgboost", "lightgbm", "random_forest".
        n_trials : int
            Number of Optuna trials to run (default 50).
        cv_splits : int
            Number of TimeSeriesSplit folds (default 5).

        Returns
        -------
        dict
            Best hyperparameter dictionary, or default params if >50% of
            trials failed.
        """
        model_type = model_type.lower()
        if model_type not in DEFAULT_PARAMS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(DEFAULT_PARAMS.keys())}"
            )

        self._model_type = model_type
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, model_type)
            try:
                scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = self._build_model(model_type, params)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_train, y_train)
                    preds = model.predict(X_val)
                    scores.append(accuracy_score(y_val, preds))

                return float(np.mean(scores))
            except Exception as exc:
                logger.debug("Trial %d failed: %s", trial.number, exc)
                return float("-inf")

        self._study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        self._study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Check failure rate
        failed = sum(
            1
            for t in self._study.trials
            if t.value is not None and t.value == float("-inf")
        )
        failure_rate = failed / max(len(self._study.trials), 1)

        if failure_rate > 0.5:
            logger.warning(
                "HyperparameterTuner: %.0f%% of trials failed for %s. "
                "Falling back to default parameters.",
                failure_rate * 100,
                model_type,
            )
            return dict(DEFAULT_PARAMS[model_type])

        return dict(self._study.best_params)

    def get_study_results(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all Optuna trial results.

        Columns: trial_number, params, value, state
        """
        if self._study is None:
            return pd.DataFrame(
                columns=["trial_number", "params", "value", "state"]
            )

        rows = []
        for trial in self._study.trials:
            rows.append(
                {
                    "trial_number": trial.number,
                    "params": trial.params,
                    "value": trial.value,
                    "state": trial.state.name,
                }
            )
        return pd.DataFrame(rows, columns=["trial_number", "params", "value", "state"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _suggest_params(trial: optuna.Trial, model_type: str) -> dict:
        """Suggest hyperparameters for the given model type."""
        if model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            }
        elif model_type == "lightgbm":
            return {
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
                "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            }
        elif model_type == "random_forest":
            max_features_choice = trial.suggest_categorical(
                "max_features", ["sqrt", "log2", "auto"]
            )
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "max_features": max_features_choice,
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    @staticmethod
    def _build_model(model_type: str, params: dict):
        """Instantiate a sklearn-compatible model with the given params."""
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
        elif model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                **params,
                verbose=-1,
                random_state=42,
            )
        elif model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            # "auto" is not valid in newer sklearn; map to None (uses all features)
            p = dict(params)
            if p.get("max_features") == "auto":
                p["max_features"] = None
            return RandomForestClassifier(
                **p,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
