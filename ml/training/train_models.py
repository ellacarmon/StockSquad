"""
ML Model Training
Trains multiple models for stock price prediction.
"""

import pandas as pd
import numpy as np
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

        if model_type == "xgboost":
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "lightgbm":
            model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
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

        if model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
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

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # Metrics
        metrics = {
            "train_mse": mean_squared_error(y_train, y_train_pred),
            "val_mse": mean_squared_error(y_val, y_val_pred),
            "train_r2": r2_score(y_train, y_train_pred),
            "val_r2": r2_score(y_val, y_val_pred),
            "train_mae": np.mean(np.abs(y_train - y_train_pred)),
            "val_mae": np.mean(np.abs(y_val - y_val_pred)),
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

        # Average metrics across folds
        avg_class_metrics = {
            key: np.mean([m[key] for m in classification_metrics])
            for key in classification_metrics[0].keys()
        }

        avg_reg_metrics = {
            key: np.mean([m[key] for m in regression_metrics])
            for key in regression_metrics[0].keys()
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

    def train_and_save_models(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.8,
        model_types: list = ["xgboost", "random_forest", "lightgbm"]
    ) -> Dict[str, Any]:
        """
        Train and save final models.

        Args:
            df: Full dataset
            train_ratio: Ratio of data to use for training
            model_types: List of model types to train

        Returns:
            Dictionary with trained models and metrics
        """
        print(f"\n" + "="*70)
        print("TRAINING FINAL MODELS")
        print("="*70 + "\n")

        X, y_class, y_reg = self.prepare_features(df)

        # Time-based split (no shuffle for time series)
        split_idx = int(len(X) * train_ratio)
        X_train = X.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_class_train = y_class.iloc[:split_idx]
        y_class_val = y_class.iloc[split_idx:]
        y_reg_train = y_reg.iloc[:split_idx]
        y_reg_val = y_reg.iloc[split_idx:]

        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Features: {len(X.columns)}")

        results = {}

        for model_type in model_types:
            print(f"\n{'='*70}")
            print(f"MODEL TYPE: {model_type.upper()}")
            print(f"{'='*70}")

            # Train classifier
            classifier, class_metrics = self.train_direction_classifier(
                X_train, y_class_train, X_val, y_class_val, model_type
            )

            # Train regressor
            regressor, reg_metrics = self.train_return_regressor(
                X_train, y_reg_train, X_val, y_reg_val, model_type
            )

            # Save models
            classifier_path = self.models_dir / f"{model_type}_direction_classifier.joblib"
            regressor_path = self.models_dir / f"{model_type}_return_regressor.joblib"

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
                    "metrics": class_metrics
                },
                "regressor": {
                    "model": regressor,
                    "path": str(regressor_path),
                    "metrics": reg_metrics
                }
            }

        # Save training metadata
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "features": list(X.columns),
            "models": {
                model_type: {
                    "classifier_metrics": results[model_type]["classifier"]["metrics"],
                    "regressor_metrics": results[model_type]["regressor"]["metrics"]
                }
                for model_type in model_types
            }
        }

        metadata_path = self.models_dir / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Saved metadata: {metadata_path}")

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

    # Train final models
    results = trainer.train_and_save_models(
        df,
        train_ratio=0.8,
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
