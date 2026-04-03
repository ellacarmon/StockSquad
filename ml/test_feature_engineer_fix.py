"""
Quick test for FeatureEngineer fixes
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_technical_indicators():
    """Test TechnicalIndicators static methods."""
    print("\n" + "="*70)
    print("TEST: Technical Indicators")
    print("="*70 + "\n")

    try:
        from tools.ta_indicators import TechnicalIndicators
        import pandas as pd
        import numpy as np

        # Create sample price data
        dates = pd.date_range('2024-01-01', periods=250, freq='D')
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(250) * 2),
            index=dates
        )
        volumes = pd.Series(
            50000000 + np.random.randn(250) * 5000000,
            index=dates
        )

        print("Testing static methods...")

        # Test RSI
        rsi = TechnicalIndicators.calculate_rsi(prices, period=14)
        print(f"  ✓ RSI: {len(rsi)} values")

        # Test MACD
        macd = TechnicalIndicators.calculate_macd(prices)
        print(f"  ✓ MACD: {len(macd['macd'])} values")

        # Test SMA
        sma = TechnicalIndicators.calculate_moving_averages(prices, [20, 50, 200])
        print(f"  ✓ SMA: {list(sma.keys())}")

        # Test EMA
        ema = TechnicalIndicators.calculate_exponential_moving_averages(prices, [12, 26])
        print(f"  ✓ EMA: {list(ema.keys())}")

        # Test Bollinger Bands
        bb = TechnicalIndicators.calculate_bollinger_bands(prices)
        print(f"  ✓ Bollinger: {list(bb.keys())}")

        # Test Volume
        vol = TechnicalIndicators.calculate_volume_indicators(prices, volumes)
        print(f"  ✓ Volume: {list(vol.keys())}")

        print("\n✅ All technical indicator methods work correctly!\n")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineer():
    """Test FeatureEngineer with sample data."""
    print("\n" + "="*70)
    print("TEST: FeatureEngineer")
    print("="*70 + "\n")

    try:
        from ml.training.feature_engineer import FeatureEngineer
        from ml.training.data_collector import DataCollector

        # Create test database with sample stock
        print("Setting up test database...")
        collector = DataCollector(db_path="ml/training/data/test_stocks.db")

        # Fetch AAPL data
        print("Fetching AAPL data (1 year)...")
        success = collector.fetch_stock_data("AAPL", period="1y")

        if not success:
            print("❌ Failed to fetch data")
            return False

        print("✅ Data fetched successfully")

        # Test feature engineering
        print("\nTesting feature engineering...")
        engineer = FeatureEngineer(db_path="ml/training/data/test_stocks.db")

        print("  Calculating technical indicators...")
        success1 = engineer.calculate_technical_indicators("AAPL")

        if not success1:
            print("  ❌ Failed to calculate indicators")
            return False

        print("  ✅ Technical indicators calculated")

        print("  Creating labels...")
        success2 = engineer.create_labels("AAPL", forward_periods=[1, 5, 10, 20])

        if not success2:
            print("  ❌ Failed to create labels")
            return False

        print("  ✅ Labels created")

        # Verify data
        print("\nVerifying data in database...")
        conn = engineer.collector.get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM technical_indicators WHERE ticker = 'AAPL'")
        ind_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM labels WHERE ticker = 'AAPL'")
        label_count = cursor.fetchone()[0]

        # Check label columns
        cursor.execute("PRAGMA table_info(labels)")
        columns = [row[1] for row in cursor.fetchall()]

        conn.close()

        print(f"  Technical indicators: {ind_count} rows")
        print(f"  Labels: {label_count} rows")
        print(f"  Label columns: {', '.join(columns)}")

        # Verify all expected columns exist
        expected_cols = ['direction_1d', 'direction_5d', 'direction_10d', 'direction_20d']
        missing = [col for col in expected_cols if col not in columns]

        if missing:
            print(f"\n⚠️  Missing columns: {missing}")
            return False

        print("\n✅ FeatureEngineer works correctly!\n")
        print("Cleanup: You can delete ml/training/data/test_stocks.db")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("FEATURE ENGINEER FIX VALIDATION")
    print("="*70)
    print("\nTesting fixes for:")
    print("1. TechnicalIndicators.calculate_all_indicators() error")
    print("2. Missing 'direction_10d' column error")

    results = {}

    # Test 1
    results['technical_indicators'] = test_technical_indicators()

    # Test 2
    results['feature_engineer'] = test_feature_engineer()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name.replace('_', ' ').title()}")

    if all(results.values()):
        print("\n🎉 ALL FIXES VERIFIED!")
        print("\nYou can now run the full ML pipeline:")
        print("  python3 ml/test_ml_pipeline.py")
    else:
        print("\n❌ Some tests failed. Check errors above.")
