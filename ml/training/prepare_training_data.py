"""
Training Data Preparation Script
Collects historical data and engineers features for ML training.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml.training.data_collector import HistoricalDataCollector
from ml.training.feature_engineer import FeatureEngineer
from tools.stock_universe import StockUniverse


def prepare_training_data(
    universe: str = "sp100",
    period: str = "5y",
    max_tickers: int = None,
    max_workers: int = 5
):
    """
    Prepare complete training dataset.

    Args:
        universe: Stock universe to use
        period: Historical period to fetch
        max_tickers: Maximum tickers to process (None = all)
        max_workers: Parallel workers for data collection
    """
    print("\n" + "="*70)
    print("STOCK SQUAD ML TRAINING DATA PREPARATION")
    print("="*70 + "\n")

    # Get tickers from universe
    stock_universe = StockUniverse()
    tickers = stock_universe.get_tickers(universe)

    if max_tickers:
        tickers = tickers[:max_tickers]

    print(f"Universe: {universe.upper()}")
    print(f"Tickers: {len(tickers)}")
    print(f"Period: {period}")
    print(f"Workers: {max_workers}")

    # Step 1: Collect historical data
    print("\n" + "="*70)
    print("STEP 1: COLLECTING HISTORICAL DATA")
    print("="*70 + "\n")

    collector = HistoricalDataCollector()
    results = collector.collect_multiple_tickers(
        tickers,
        period=period,
        max_workers=max_workers
    )

    successful_tickers = [t for t, success in results.items() if success]
    print(f"\nSuccessfully collected: {len(successful_tickers)}/{len(tickers)} tickers")

    # Step 2: Calculate technical indicators
    print("\n" + "="*70)
    print("STEP 2: ENGINEERING FEATURES")
    print("="*70 + "\n")

    engineer = FeatureEngineer()
    stats = engineer.process_all_tickers(max_tickers=max_tickers)

    # Step 3: Get summary
    print("\n" + "="*70)
    print("STEP 3: DATA SUMMARY")
    print("="*70 + "\n")

    summary = collector.get_data_summary()
    print(f"Total tickers: {summary['total_tickers']}")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Date range: {summary['date_range']}")

    print(f"\nSectors:")
    for sector, count in summary['sectors'].items():
        print(f"  {sector}: {count}")

    # Step 4: Get training data sample
    print("\n" + "="*70)
    print("STEP 4: TRAINING DATA PREVIEW")
    print("="*70 + "\n")

    training_data = engineer.get_training_data()
    print(f"Total training samples: {len(training_data):,}")
    print(f"Features: {len(training_data.columns)}")

    if not training_data.empty:
        print(f"\nColumns:")
        for col in training_data.columns:
            print(f"  - {col}")

        print(f"\nSample data:")
        print(training_data.head())

        # Check data quality
        print(f"\nData Quality:")
        total_cells = len(training_data) * len(training_data.columns)
        missing_cells = training_data.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        print(f"  Completeness: {completeness:.1f}%")

        missing_by_col = training_data.isnull().sum()
        if missing_by_col.sum() > 0:
            print(f"\n  Missing values by column:")
            for col, count in missing_by_col[missing_by_col > 0].items():
                pct = (count / len(training_data)) * 100
                print(f"    {col}: {count} ({pct:.1f}%)")

        # Export for training
        print(f"\nExporting training data...")
        collector.export_for_training()

    print("\n" + "="*70)
    print("PREPARATION COMPLETE!")
    print("="*70 + "\n")

    print("Next steps:")
    print("  1. Review data quality and completeness")
    print("  2. Train ML models: python3 ml/training/train_models.py")
    print("  3. Evaluate model performance")
    print("  4. Deploy models to production")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare ML training data")
    parser.add_argument("--universe", default="sp100", help="Stock universe (sp100, sp500, etc.)")
    parser.add_argument("--period", default="5y", help="Historical period (1y, 2y, 5y, max)")
    parser.add_argument("--max-tickers", type=int, help="Maximum tickers to process")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers")

    args = parser.parse_args()

    prepare_training_data(
        universe=args.universe,
        period=args.period,
        max_tickers=args.max_tickers,
        max_workers=args.workers
    )
