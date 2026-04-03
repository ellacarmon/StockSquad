#!/usr/bin/env python3
"""
Run Simple Backtest - CLI Interface

Usage examples:
    # Single ticker
    python3 ml/backtesting/run_backtest.py --ticker AAPL --start 2023-01-01 --end 2024-12-31

    # Multiple tickers
    python3 ml/backtesting/run_backtest.py --tickers AAPL MSFT NVDA --start 2023-01-01 --end 2024-12-31

    # With custom parameters
    python3 ml/backtesting/run_backtest.py --ticker AAPL --model lightgbm --confidence 70 --holding-days 10

    # Export results
    python3 ml/backtesting/run_backtest.py --ticker NVDA --start 2023-01-01 --end 2024-12-31 --export results.json
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

from ml.backtesting.simple_backtester import SimpleBacktester
from ml.backtesting.report import BacktestReport


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run simple backtest on ML stock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Ticker selection
    ticker_group = parser.add_mutually_exclusive_group(required=True)
    ticker_group.add_argument(
        '--ticker',
        type=str,
        help='Single ticker to backtest (e.g., AAPL)'
    )
    ticker_group.add_argument(
        '--tickers',
        type=str,
        nargs='+',
        help='Multiple tickers to backtest (e.g., AAPL MSFT NVDA)'
    )

    # Date range
    parser.add_argument(
        '--start',
        type=str,
        default=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        help='Start date (YYYY-MM-DD). Default: 1 year ago'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD). Default: today'
    )

    # ML model
    parser.add_argument(
        '--model',
        type=str,
        default='xgboost',
        choices=['xgboost', 'random_forest', 'lightgbm', 'ensemble_voting', 'ensemble_averaging', 'ensemble_unanimous'],
        help='ML model to use. Ensemble options combine all 3 models. Default: xgboost'
    )

    # Strategy parameters
    parser.add_argument(
        '--confidence',
        type=float,
        default=60.0,
        help='Minimum ML confidence threshold (0-100). Default: 60'
    )
    parser.add_argument(
        '--min-return',
        type=float,
        default=2.0,
        help='Minimum expected return threshold (%%). Default: 2.0'
    )
    parser.add_argument(
        '--holding-days',
        type=int,
        default=5,
        help='Number of days to hold each position. Default: 5'
    )
    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.2,
        help='Round-trip transaction cost (%%). Default: 0.2'
    )

    # Direction filter
    parser.add_argument(
        '--direction',
        type=str,
        default='both',
        choices=['bullish', 'bearish', 'both'],
        help='Trade direction filter. Default: both'
    )

    # Output options
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to JSON file'
    )
    parser.add_argument(
        '--export-trades',
        type=str,
        help='Export trade list to CSV file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed logging'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Print header
    if not args.quiet:
        print("\n" + "="*70)
        print("           SIMPLE BACKTEST - ML STOCK PREDICTIONS")
        print("="*70)

    # Initialize backtester
    backtester = SimpleBacktester(
        model_type=args.model,
        holding_days=args.holding_days,
        confidence_threshold=args.confidence,
        min_expected_return=args.min_return,
        transaction_cost_pct=args.transaction_cost
    )

    # Run backtest
    if args.ticker:
        # Single ticker
        results = backtester.backtest_ticker(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            direction=args.direction
        )
    else:
        # Multiple tickers
        results = backtester.backtest_multiple(
            tickers=args.tickers,
            start_date=args.start,
            end_date=args.end,
            direction=args.direction
        )

    # Generate report
    report = BacktestReport.generate_text_report(results)
    print(report)

    # Export if requested
    if args.export:
        BacktestReport.export_to_json(results, args.export)

    if args.export_trades:
        BacktestReport.export_trades_to_csv(results, args.export_trades)

    # Return exit code based on success
    if 'error' in results:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
