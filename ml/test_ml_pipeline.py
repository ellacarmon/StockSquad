"""
ML Pipeline Testing Script
Tests the complete ML pipeline from data collection to prediction.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_data_collection():
    """Test data collection."""
    print("\n" + "="*70)
    print("TEST 1: DATA COLLECTION")
    print("="*70 + "\n")

    from ml.training.data_collector import DataCollector

    collector = DataCollector()

    # Test with just AAPL for speed
    print("Testing data collection for AAPL...")
    success = collector.fetch_stock_data("AAPL", period="1y")

    if success:
        print("✅ Data collection successful")

        # Check database
        conn = collector.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM price_history WHERE ticker = 'AAPL'")
        count = cursor.fetchone()[0]
        print(f"   Rows collected: {count}")

        conn.close()

        if count > 0:
            return True
        else:
            print("❌ No data in database")
            return False
    else:
        print("❌ Data collection failed")
        return False


def test_feature_engineering():
    """Test feature engineering."""
    print("\n" + "="*70)
    print("TEST 2: FEATURE ENGINEERING")
    print("="*70 + "\n")

    from ml.training.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()

    print("Calculating technical indicators for AAPL...")
    success1 = engineer.calculate_technical_indicators("AAPL")

    if not success1:
        print("❌ Technical indicator calculation failed")
        return False

    print("✅ Technical indicators calculated")

    print("\nCreating labels for AAPL...")
    success2 = engineer.create_labels("AAPL", forward_periods=[1, 5])

    if not success2:
        print("❌ Label creation failed")
        return False

    print("✅ Labels created")

    # Check database
    conn = engineer.collector.get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE ticker = 'AAPL'")
    ind_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM labels WHERE ticker = 'AAPL'")
    label_count = cursor.fetchone()[0]

    conn.close()

    print(f"   Technical indicators: {ind_count} rows")
    print(f"   Labels: {label_count} rows")

    if ind_count > 0 and label_count > 0:
        return True
    else:
        print("❌ No indicators or labels in database")
        return False


def test_data_export():
    """Test CSV export."""
    print("\n" + "="*70)
    print("TEST 3: TRAINING DATA EXPORT")
    print("="*70 + "\n")

    from ml.training.feature_engineer import FeatureEngineer

    engineer = FeatureEngineer()

    print("Exporting training data to CSV...")
    csv_path = engineer.export_training_data()

    if csv_path and Path(csv_path).exists():
        print(f"✅ Training data exported to: {csv_path}")

        import pandas as pd
        df = pd.read_csv(csv_path)
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        if len(df) > 0:
            return True, df
        else:
            print("❌ CSV is empty")
            return False, None
    else:
        print("❌ CSV export failed")
        return False, None


def test_model_training(df):
    """Test model training."""
    print("\n" + "="*70)
    print("TEST 4: MODEL TRAINING")
    print("="*70 + "\n")

    from ml.training.train_models import ModelTrainer

    if len(df) < 200:
        print(f"⚠️  Only {len(df)} samples - too small for robust training")
        print("   Skipping training test (need at least 200 samples)")
        print("   Run: python3 ml/training/prepare_training_data.py --max-tickers 10")
        return False

    trainer = ModelTrainer()

    print(f"Training on {len(df):,} samples...")
    print("Note: This may take a few minutes for XGBoost...\n")

    try:
        # Train just XGBoost for speed
        results = trainer.train_and_save_models(
            df,
            train_ratio=0.8,
            model_types=["xgboost"]
        )

        if "xgboost" in results:
            print("\n✅ Model training successful")

            # Check metrics
            class_metrics = results["xgboost"]["classifier"]["metrics"]
            reg_metrics = results["xgboost"]["regressor"]["metrics"]

            print(f"\nClassifier Performance:")
            print(f"   Accuracy: {class_metrics['val_accuracy']:.4f}")
            print(f"   F1 Score: {class_metrics['val_f1']:.4f}")

            print(f"\nRegressor Performance:")
            print(f"   R²: {reg_metrics['val_r2']:.4f}")
            print(f"   MAE: {reg_metrics['val_mae']:.4f}%")

            # Check if models exist
            models_dir = Path("ml/models")
            classifier_path = models_dir / "xgboost_direction_classifier.joblib"
            regressor_path = models_dir / "xgboost_return_regressor.joblib"

            if classifier_path.exists() and regressor_path.exists():
                print(f"\n✅ Models saved to: {models_dir}/")
                return True
            else:
                print("\n❌ Models not saved")
                return False
        else:
            print("❌ Training returned no results")
            return False

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test prediction engine."""
    print("\n" + "="*70)
    print("TEST 5: PREDICTION ENGINE")
    print("="*70 + "\n")

    from ml.inference.prediction_engine import PredictionEngine

    try:
        print("Loading prediction engine...")
        engine = PredictionEngine(model_type="xgboost")

        # Get model info
        info = engine.get_model_info()
        print(f"   Model type: {info['model_type']}")
        print(f"   Classifier loaded: {info['classifier_loaded']}")
        print(f"   Regressor loaded: {info['regressor_loaded']}")

        if not info['classifier_loaded'] or not info['regressor_loaded']:
            print("\n❌ Models not loaded")
            print("   Run: python3 ml/training/train_models.py")
            return False

        print("✅ Models loaded successfully")

        # Test prediction with sample data
        sample_indicators = {
            'open': 175.0,
            'high': 178.0,
            'low': 174.5,
            'close': 177.0,
            'volume': 50000000,
            'rsi': 58,
            'macd': {'macd': 1.5, 'signal': 1.2, 'histogram': 0.3},
            'sma': {'20': 175.0, '50': 170.0, '200': 165.0},
            'ema': {'12': 176.0, '26': 174.0},
            'bollinger': {'upper': 180.0, 'middle': 175.0, 'lower': 170.0},
            'volume_sma': 45000000
        }

        print("\nMaking test prediction...")
        prediction = engine.predict(sample_indicators)

        if "error" in prediction:
            print(f"❌ Prediction failed: {prediction['error']}")
            return False

        print("✅ Prediction successful\n")
        print(f"   Direction: {prediction['direction'].upper()}")
        print(f"   Confidence: {prediction['confidence']:.1f}%")
        print(f"   Expected Return (5d): {prediction['expected_return']:+.2f}%")
        print(f"   Recommendation: {prediction['recommendation']}")
        print(f"   Score: {prediction['score']:.1f}/100")

        return True

    except Exception as e:
        print(f"❌ Prediction engine failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_signal_scorer_integration():
    """Test SignalScorer ML integration."""
    print("\n" + "="*70)
    print("TEST 6: SIGNAL SCORER INTEGRATION")
    print("="*70 + "\n")

    from ml.signal_model import SignalScorer

    try:
        print("Initializing SignalScorer with ML...")
        scorer = SignalScorer(use_ml=True, model_type="xgboost")

        print(f"   Model version: {scorer.model_version}")
        print(f"   ML engine loaded: {scorer.ml_engine is not None}")

        # Test scoring with sample data
        sample_indicators = {
            'rsi': {'value': 58},
            'macd': {'macd': 1.5, 'signal': 1.2, 'histogram': 0.3},
            'moving_averages': {
                'SMA_20': 175.0,
                'SMA_50': 170.0,
                'SMA_200': 165.0
            },
            'volume': {'ratio': 1.2},
            'trend': 'Uptrend',
            'price_position': {'vs_SMA20': 1.1},
            'close': 177.0,
            'volume': 50000000,
            'open': 175.0,
            'high': 178.0,
            'low': 174.5,
            'sma': {'20': 175.0, '50': 170.0, '200': 165.0},
            'ema': {'12': 176.0, '26': 174.0},
            'bollinger': {'upper': 180.0, 'middle': 175.0, 'lower': 170.0},
            'volume_sma': 45000000
        }

        print("\nGenerating signal...")
        signal = scorer.score_signal(sample_indicators)

        print("✅ Signal generation successful\n")
        print(f"   Direction: {signal['direction']}")
        print(f"   Confidence: {signal['confidence']}/100")
        print(f"   Recommendation: {signal['recommendation']}")
        print(f"   Model version: {signal['model_version']}")
        print(f"   ML-powered: {signal.get('ml_powered', False)}")

        if signal.get('ml_powered'):
            print("\n✅ ML integration working! Agents will use ML predictions.")
        else:
            print("\n⚠️  Using rule-based fallback (ML models may not be loaded)")

        return True

    except Exception as e:
        print(f"❌ SignalScorer integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("STOCKSQUAD ML PIPELINE TEST SUITE")
    print("="*70)
    print("\nThis will test the complete ML pipeline:")
    print("1. Data collection")
    print("2. Feature engineering")
    print("3. Training data export")
    print("4. Model training (XGBoost only)")
    print("5. Prediction engine")
    print("6. SignalScorer integration")
    print("\nNote: Tests will use AAPL for speed. Full training uses S&P 100.")

    input("\nPress Enter to continue...")

    results = {}
    df = None

    # Test 1: Data Collection
    results['data_collection'] = test_data_collection()
    if not results['data_collection']:
        print("\n❌ DATA COLLECTION FAILED - Stopping tests")
        return

    # Test 2: Feature Engineering
    results['feature_engineering'] = test_feature_engineering()
    if not results['feature_engineering']:
        print("\n❌ FEATURE ENGINEERING FAILED - Stopping tests")
        return

    # Test 3: Data Export
    results['data_export'], df = test_data_export()
    if not results['data_export']:
        print("\n❌ DATA EXPORT FAILED - Stopping tests")
        return

    # Test 4: Model Training (optional if not enough data)
    results['model_training'] = test_model_training(df)

    # Test 5: Prediction (only if training succeeded)
    if results['model_training']:
        results['prediction'] = test_prediction()
    else:
        print("\n⏭️  Skipping prediction test (models not trained)")
        results['prediction'] = False

    # Test 6: SignalScorer Integration
    results['signal_scorer'] = test_signal_scorer_integration()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    passed_count = sum(results.values())
    total_count = len(results)

    print(f"\n{passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! ML pipeline is fully functional.")
        print("\nNext steps:")
        print("1. Collect more data: python3 ml/training/prepare_training_data.py --max-tickers 20")
        print("2. Train production models: python3 ml/training/train_models.py")
        print("3. Test with Telegram bot: Send '/analyze AAPL' to your bot")
    elif results['data_collection'] and results['feature_engineering'] and results['data_export']:
        print("\n⚠️  Core pipeline works, but need more data for training")
        print("\nNext step:")
        print("   python3 ml/training/prepare_training_data.py --max-tickers 20 --period 2y")
    else:
        print("\n❌ CRITICAL TESTS FAILED - Check error messages above")


if __name__ == "__main__":
    main()
