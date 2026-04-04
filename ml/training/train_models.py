"""
ML Model Training
Trains multiple models for stock price prediction.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime


@dataclass
class TrainingConfig:
    model_types: list = field(default_factory=lambda: ["xgboost", "random_forest", "lightgbm"])
    target_type: str = "binary"          # "binary", "threshold", "tertile"
    threshold_pct: float = 1.0           # used when target_type="threshold"
    n_cv_splits: int = 5
    n_optuna_trials: int = 50
    early_stopping_rounds: int = 20
    top_n_features: int = 40
    calibration_method: str = "isotonic" # "isotonic" or "platt"
    handle_imbalance: bool = True


class ModelTrainer:
    """Trains and evaluates ML models for stock prediction."""

    def __init__(self, models_dir: str = "ml/models"):
        """
        Initialize model trainer.

        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and labels for training.

        Args:
            df: Training data

        Returns:
            Tuple of (features, classification_labels, regression_labels)
        """
        # Feature columns
        feature_cols = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi_14', 'macd', 'macd_signal', 'macd_hist',
            'sma_20', 'sma_50', 'sma_200',
            'ema_12', 'ema_26',
            'bb_upper', 'bb_middle', 'bb_lower',
            'volume_sma_20',
            'news_sentiment_score',
            'news_sentiment_confidence',
            'news_sentiment_article_count',
            'news_sentiment_trend',
            'news_macro_sentiment',
            'news_company_expected_revenue_sentiment',
            'news_company_specific_sentiment',
            'news_industry_peer_sentiment',
        ]

        # Filter to only existing columns
        available_features = [col for col in feature_cols if col in df.columns]

        X = df[available_features].copy()

        # Add derived features
        if 'close' in X.columns and 'sma_20' in X.columns:
            X['price_to_sma20'] = (X['close'] / X['sma_20']) - 1

        if 'close' in X.columns and 'sma_50' in X.columns:
            X['price_to_sma50'] = (X['close'] / X['sma_50']) - 1

        if 'volume' in X.columns and 'volume_sma_20' in X.columns:
            X['volume_ratio'] = X['volume'] / X['volume_sma_20']

        # Neutral defaults for optional sentiment features if the joined table exists
        sentiment_defaults = {
            'news_sentiment_score': 50.0,
            'news_sentiment_confidence': 0.0,
            'news_sentiment_article_count': 0.0,
            'news_sentiment_trend': 0.0,
            'news_macro_sentiment': 0.0,
            'news_company_expected_revenue_sentiment': 0.0,
            'news_company_specific_sentiment': 0.0,
            'news_industry_peer_sentiment': 0.0,
        }
        for column, default_value in sentiment_defaults.items():
            if column in X.columns:
                X[column] = pd.to_numeric(X[column], errors='coerce').fillna(default_value)

        # Labels
        y_classification = df['direction_5d']  # Binary: 1=up, 0=down
        y_regression = df['forward_5d_return']  # Continuous: % return

        # Remove rows with missing values
        valid_mask = ~(X.isnull().any(axis=1) | y_classification.isnull() | y_regression.isnull())
        X = X[valid_mask]
        y_classification = y_classification[valid_mask]
        y_regression = y_regression[valid_mask]

        return X, y_classification, y_regression

    def apply_threshold_labeling(
        self, y_return: pd.Series, threshold_pct: float
    ) -> tuple:
        """
        Apply threshold-based labeling: label as 1 (up) if return > +threshold_pct,
        0 (down) if return < -threshold_pct. Samples within the band are excluded.

        Args:
            y_return: Series of return values (as percentages or decimals, consistent units)
            threshold_pct: Threshold value; samples where |return| < threshold_pct are excluded

        Returns:
            Tuple of (labels, valid_index) where labels contains 1/0 and valid_index
            is the index of rows where |return| >= threshold_pct
        """
        valid_mask = y_return.abs() >= threshold_pct
        valid_index = y_return.index[valid_mask]
        labels = (y_return[valid_mask] > 0).astype(int)
        return labels, valid_index

    def apply_tertile_labeling(self, y_return: pd.Series) -> pd.Series:
        """
        Apply tertile-based labeling: divide the return distribution into thirds.
        Returns 2 for top tertile, 1 for middle, 0 for bottom.

        Args:
            y_return: Series of return values

        Returns:
            Series of integer labels (0, 1, 2)
        """
        # Use pd.qcut with duplicates='drop' to handle ties; fall back to rank-based
        # assignment when qcut cannot form 3 distinct bins
        try:
            codes = pd.qcut(y_return, q=3, labels=[0, 1, 2], duplicates="drop")
            labels = codes.astype(int)
        except ValueError:
            # Fallback: rank-based tertile assignment
            ranks = y_return.rank(method="first")
            n = len(y_return)
            labels = pd.Series(0, index=y_return.index, dtype=int)
            labels[ranks > n / 3] = 1
            labels[ranks > 2 * n / 3] = 2
        return labels

    def train_direction_classifier(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "xgboost"
    ) -> Tuple[Any, Dict]:
        """
        Train a direction classifier (up/down).

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model (xgboost, random_forest, lightgbm)

        Returns:
            Tuple of (trained model, metrics dict)
        """
        print(f"\n[ModelTrainer] Training {model_type} direction classifier...")

        # Compute early-stopping validation split (20% of training data)
        es_split_idx = int(len(X_train) * 0.8)
        X_es_train = X_train.iloc[:es_split_idx]
        y_es_train = y_train.iloc[:es_split_idx]
        X_es_val = X_train.iloc[es_split_idx:]
        y_es_val = y_train.iloc[es_split_idx:]

        # Compute scale_pos_weight for XGBoost class imbalance
        neg_count = int((y_es_train == 0).sum())
        pos_count = int((y_es_train == 1).sum())
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=20,
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                is_unbalance=True,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train with early stopping for XGBoost and LightGBM
        if model_type == "xgboost":
            model.fit(
                X_es_train, y_es_train,
                eval_set=[(X_es_val, y_es_val)],
                verbose=False,
            )
        elif model_type == "lightgbm":
            model.fit(
                X_es_train, y_es_train,
                eval_set=[(X_es_val, y_es_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Feature importances
        feature_importances = dict(zip(X_train.columns, model.feature_importances_))

        # Metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
            "feature_importances": feature_importances,
        }

        print(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
        print(f"  Val Accuracy: {metrics['val_accuracy']:.4f}")
        print(f"  Val Precision: {metrics['val_precision']:.4f}")
        print(f"  Val Recall: {metrics['val_recall']:.4f}")
        print(f"  Val F1: {metrics['val_f1']:.4f}")

        return model, metrics

    def train_return_regressor(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "xgboost"
    ) -> Tuple[Any, Dict]:
        """
        Train a return regressor (predict % return).

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_type: Type of model (xgboost, random_forest, lightgbm)

        Returns:
            Tuple of (trained model, metrics dict)
        """
        print(f"\n[ModelTrainer] Training {model_type} return regressor...")

        # Compute early-stopping validation split (20% of training data)
        es_split_idx = int(len(X_train) * 0.8)
        X_es_train = X_train.iloc[:es_split_idx]
        y_es_train = y_train.iloc[:es_split_idx]
        X_es_val = X_train.iloc[es_split_idx:]
        y_es_val = y_train.iloc[es_split_idx:]

        if model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=20,
            )
        elif model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train with early stopping for XGBoost and LightGBM
        if model_type == "xgboost":
            model.fit(
                X_es_train, y_es_train,
                eval_set=[(X_es_val, y_es_val)],
                verbose=False,
            )
        elif model_type == "lightgbm":
            model.fit(
                X_es_train, y_es_train,
                eval_set=[(X_es_val, y_es_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Feature importances
        feature_importances = dict(zip(X_train.columns, model.feature_importances_))

        # Metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "train_mae": np.mean(np.abs(y_train - y_train_pred)),
            "val_mae": np.mean(np.abs(y_val - y_val_pred)),
            "feature_importances": feature_importances,
        }

        print(f"  Train MSE: {metrics['train_mse']:.4f}")
        print(f"  Val MSE: {metrics['val_mse']:.4f}")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Val R²: {metrics['val_r2']:.4f}")
        print(f"  Val MAE: {metrics['val_mae']:.4f}")

        return model, metrics

    def walk_forward_validation(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.

        Args:
            df: Full dataset
            n_splits: Number of time series splits
            model_type: Type of model to train

        Returns:
            Dictionary with validation results
        """
        print(f"\n[ModelTrainer] Walk-forward validation ({n_splits} splits)...")

        X, y_class, y_reg = self.prepare_features(df)

        tscv = TimeSeriesSplit(n_splits=n_splits)

        classification_metrics = []
        regression_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\n  Fold {fold}/{n_splits}")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_class_train, y_class_val = y_class.iloc[train_idx], y_class.iloc[val_idx]
            y_reg_train, y_reg_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]

            # Train classifier
            _, class_metrics = self.train_direction_classifier(
                X_train, y_class_train, X_val, y_class_val, model_type
            )
            classification_metrics.append(class_metrics)

            # Train regressor
            _, reg_metrics = self.train_return_regressor(
                X_train, y_reg_train, X_val, y_reg_val, model_type
            )
            regression_metrics.append(reg_metrics)

        # Average metrics across folds (excluding feature_importances which is a dict)
        avg_class_metrics = {
            key: np.mean([m[key] for m in classification_metrics])
            for key in classification_metrics[0].keys()
            if key != "feature_importances"
        }

        avg_reg_metrics = {
            key: np.mean([m[key] for m in regression_metrics])
            for key in regression_metrics[0].keys()
            if key != "feature_importances"
        }

        print(f"\n[ModelTrainer] Average Metrics Across {n_splits} Folds:")
        print(f"  Classification Accuracy: {avg_class_metrics['val_accuracy']:.4f}")
        print(f"  Classification F1: {avg_class_metrics['val_f1']:.4f}")
        print(f"  Regression R²: {avg_reg_metrics['val_r2']:.4f}")
        print(f"  Regression MAE: {avg_reg_metrics['val_mae']:.4f}")

        return {
            "classification": avg_class_metrics,
            "regression": avg_reg_metrics,
            "n_splits": n_splits,
            "model_type": model_type
        }

    def _validate_model_path(self, path: Path) -> None:
        """
        Validate that a model file path is within the allowed models directory.

        Args:
            path: Path to validate

        Raises:
            ValueError: If the path is outside ml/models/
        """
        resolved_path = path.resolve()
        resolved_models_dir = self.models_dir.resolve()
        try:
            resolved_path.relative_to(resolved_models_dir)
        except ValueError:
            raise ValueError(
                f"Model path '{resolved_path}' is outside the allowed models directory "
                f"'{resolved_models_dir}'. Refusing to load/save."
            )

    def _load_baseline_metrics(self) -> Dict[str, float]:
        """
        Load baseline validation accuracies from existing training_metadata.json.

        Returns:
            Dictionary mapping model_type -> baseline validation accuracy

        **Validates: Requirements 11.1**
        """
        metadata_path = self.models_dir / "training_metadata.json"

        if not metadata_path.exists():
            print("[ModelTrainer] No baseline metadata found. Any accuracy will be accepted.")
            return {}

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            baseline_accuracies = {}
            for model_type, model_info in metadata.get('models', {}).items():
                # Use walk-forward validation accuracy as baseline (more robust than single split)
                if 'classifier_walk_forward_metrics' in model_info:
                    baseline_acc = model_info['classifier_walk_forward_metrics'].get('val_accuracy', 0.0)
                else:
                    baseline_acc = model_info.get('classifier_metrics', {}).get('val_accuracy', 0.0)

                baseline_accuracies[model_type] = baseline_acc
                print(f"[ModelTrainer] Baseline accuracy for {model_type}: {baseline_acc:.4f}")

            return baseline_accuracies

        except Exception as e:
            print(f"[ModelTrainer] Error loading baseline metrics: {e}")
            return {}

    def _passes_regression_gate(
        self,
        model_type: str,
        new_val_accuracy: float,
        baseline_accuracies: Dict[str, float],
        min_baseline: float = 0.526
    ) -> bool:
        """
        Check if new model passes regression test gate.

        Args:
            model_type: Type of model being tested
            new_val_accuracy: New validation accuracy
            baseline_accuracies: Dictionary of baseline accuracies by model type
            min_baseline: Absolute minimum baseline (default 52.6% from requirements)

        Returns:
            True if model passes gate, False otherwise

        **Validates: Requirements 11.2**
        """
        # If we have a baseline for this model type, use it
        if model_type in baseline_accuracies:
            baseline = baseline_accuracies[model_type]
        else:
            # Otherwise use the minimum baseline from requirements
            baseline = min_baseline

        passes = new_val_accuracy >= baseline

        if not passes:
            print(f"\n❌ [ModelTrainer] REGRESSION TEST GATE FAILED for {model_type}:")
            print(f"   New validation accuracy: {new_val_accuracy:.4f}")
            print(f"   Baseline accuracy: {baseline:.4f}")
            print(f"   Model will NOT be saved.")
        else:
            print(f"\n✓ [ModelTrainer] Regression test gate PASSED for {model_type}")
            print(f"   New validation accuracy: {new_val_accuracy:.4f} >= baseline {baseline:.4f}")

        return passes

    def train_and_save_models(
        self,
        df: pd.DataFrame,
        n_cv_splits: int = 3,
        model_types: list = ["xgboost", "random_forest", "lightgbm"]
    ) -> Dict[str, Any]:
        """
        Train and save final models using expanding-window walk-forward validation
        followed by a final retraining on all available data.

        Implements regression test gate: only saves models that meet or exceed
        baseline validation accuracy.

        Args:
            df: Full dataset
            n_cv_splits: Number of walk-forward folds (minimum 3)
            model_types: List of model types to train

        Returns:
            Dictionary with trained models and metrics

        **Validates: Requirements 11.1, 11.2, 11.3**
        """
        print(f"\n" + "="*70)
        print("TRAINING FINAL MODELS")
        print("="*70 + "\n")

        # Load baseline metrics for regression test gate
        baseline_accuracies = self._load_baseline_metrics()

        X, y_class, y_reg = self.prepare_features(df)

        n_folds = max(3, n_cv_splits)
        print(f"Total samples: {len(X):,}")
        print(f"Features: {len(X.columns)}")
        print(f"Walk-forward folds: {n_folds}")

        results = {}

        for model_type in model_types:
            print(f"\n{'='*70}")
            print(f"MODEL TYPE: {model_type.upper()}")
            print(f"{'='*70}")

            # --- Step 1: Expanding-window walk-forward validation ---
            print(f"\n[ModelTrainer] Walk-forward validation ({n_folds} folds)...")
            tscv = TimeSeriesSplit(n_splits=n_folds)

            fold_class_metrics = []
            fold_reg_metrics = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                print(f"\n  Fold {fold}/{n_folds}")
                X_fold_train = X.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_class_fold_train = y_class.iloc[train_idx]
                y_class_fold_val = y_class.iloc[val_idx]
                y_reg_fold_train = y_reg.iloc[train_idx]
                y_reg_fold_val = y_reg.iloc[val_idx]

                _, cm = self.train_direction_classifier(
                    X_fold_train, y_class_fold_train,
                    X_fold_val, y_class_fold_val,
                    model_type,
                )
                fold_class_metrics.append(cm)

                _, rm = self.train_return_regressor(
                    X_fold_train, y_reg_fold_train,
                    X_fold_val, y_reg_fold_val,
                    model_type,
                )
                fold_reg_metrics.append(rm)

            # Average validation metrics across folds
            avg_class_metrics = {
                key: float(np.mean([m[key] for m in fold_class_metrics]))
                for key in fold_class_metrics[0].keys()
                if key != "feature_importances"
            }
            avg_reg_metrics = {
                key: float(np.mean([m[key] for m in fold_reg_metrics]))
                for key in fold_reg_metrics[0].keys()
                if key != "feature_importances"
            }

            print(f"\n  [Walk-Forward Avg] Classifier accuracy: {avg_class_metrics['val_accuracy']:.4f}, "
                  f"F1: {avg_class_metrics['val_f1']:.4f}")
            print(f"  [Walk-Forward Avg] Regressor R²: {avg_reg_metrics['val_r2']:.4f}, "
                  f"MAE: {avg_reg_metrics['val_mae']:.4f}")

            # --- Step 2: Regression Test Gate ---
            # Check if model passes gate based on walk-forward validation accuracy
            passes_gate = self._passes_regression_gate(
                model_type,
                avg_class_metrics['val_accuracy'],
                baseline_accuracies
            )

            if not passes_gate:
                # Skip saving this model
                print(f"  ⚠️ Skipping model save for {model_type} due to regression gate failure")
                results[model_type] = {
                    "classifier": {
                        "model": None,
                        "path": None,
                        "metrics": {},
                        "walk_forward_metrics": avg_class_metrics,
                        "gate_status": "FAILED"
                    },
                    "regressor": {
                        "model": None,
                        "path": None,
                        "metrics": {},
                        "walk_forward_metrics": avg_reg_metrics,
                        "gate_status": "FAILED"
                    },
                }
                continue

            # --- Step 3: Final retraining on ALL available data ---
            print(f"\n  Retraining final {model_type} model on all {len(X):,} samples...")

            # For final training we use all data; pass X as both train and val so
            # the method signature is satisfied (metrics on full data are informational only)
            classifier, class_metrics_final = self.train_direction_classifier(
                X, y_class, X, y_class, model_type
            )
            regressor, reg_metrics_final = self.train_return_regressor(
                X, y_reg, X, y_reg, model_type
            )

            # --- Step 4: Save final models ---
            classifier_path = self.models_dir / f"{model_type}_direction_classifier.joblib"
            regressor_path = self.models_dir / f"{model_type}_return_regressor.joblib"

            self._validate_model_path(classifier_path)
            self._validate_model_path(regressor_path)

            joblib.dump(classifier, classifier_path)
            joblib.dump(regressor, regressor_path)

            print(f"\n  ✓ Saved classifier: {classifier_path}")
            print(f"  ✓ Saved regressor: {regressor_path}")

            # Save feature names
            feature_names_path = self.models_dir / "feature_names.json"
            with open(feature_names_path, 'w') as f:
                json.dump(list(X.columns), f)

            results[model_type] = {
                "classifier": {
                    "model": classifier,
                    "path": str(classifier_path),
                    "metrics": class_metrics_final,
                    "walk_forward_metrics": avg_class_metrics,
                    "gate_status": "PASSED"
                },
                "regressor": {
                    "model": regressor,
                    "path": str(regressor_path),
                    "metrics": reg_metrics_final,
                    "walk_forward_metrics": avg_reg_metrics,
                    "gate_status": "PASSED"
                },
            }

        # Save training metadata (only for models that passed the gate)
        # **Validates: Requirements 11.3**
        passed_models = {
            model_type: {
                "classifier_metrics": results[model_type]["classifier"]["metrics"],
                "classifier_walk_forward_metrics": results[model_type]["classifier"]["walk_forward_metrics"],
                "regressor_metrics": results[model_type]["regressor"]["metrics"],
                "regressor_walk_forward_metrics": results[model_type]["regressor"]["walk_forward_metrics"],
                "gate_status": results[model_type]["classifier"]["gate_status"]
            }
            for model_type in model_types
            if results[model_type]["classifier"]["gate_status"] == "PASSED"
        }

        if passed_models:
            metadata = {
                "trained_at": datetime.now().isoformat(),
                "total_samples": len(X),
                "n_walk_forward_folds": n_folds,
                "features": list(X.columns),
                "models": passed_models
            }

            metadata_path = self.models_dir / "training_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"\n✓ Saved metadata: {metadata_path}")
            print(f"✓ {len(passed_models)}/{len(model_types)} models passed regression gate")
        else:
            print(f"\n⚠️  No models passed regression gate. Metadata NOT updated.")

        return results


if __name__ == "__main__":
    from ml.training.feature_engineer import FeatureEngineer

    print("\n" + "="*70)
    print("ML MODEL TRAINING")
    print("="*70 + "\n")

    # Load training data
    engineer = FeatureEngineer()
    df = engineer.get_training_data()

    if df.empty:
        print("❌ No training data available. Run prepare_training_data.py first.")
        exit(1)

    print(f"Loaded {len(df):,} training samples")

    # Initialize trainer
    trainer = ModelTrainer()

    # Walk-forward validation (optional)
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION")
    print("="*70)

    validation_results = trainer.walk_forward_validation(df, n_splits=3, model_type="xgboost")

    # Train final models (walk-forward + retrain on all data)
    results = trainer.train_and_save_models(
        df,
        n_cv_splits=3,
        model_types=["xgboost", "random_forest", "lightgbm"]
    )

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70 + "\n")

    print("Best model by validation accuracy:")
    best_model = max(
        results.items(),
        key=lambda x: x[1]["classifier"]["metrics"]["val_accuracy"]
    )
    print(f"  {best_model[0]}: {best_model[1]['classifier']['metrics']['val_accuracy']:.4f}")

    print("\nModels saved to: ml/models/")
    print("Next step: python3 ml/inference/prediction_engine.py")
