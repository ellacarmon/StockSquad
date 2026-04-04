"""
Simple Backtester - Test ML predictions without portfolio tracking.
Each signal is an independent trade with fixed holding period.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from ml.inference.prediction_engine import PredictionEngine
from ml.inference.ensemble_predictor import EnsemblePredictor
from ml.training.data_collector import HistoricalDataCollector
from tools.ta_indicators import TechnicalIndicators
from ml.backtesting.metrics import BacktestMetrics


class SimpleBacktester:
    """
    Simple backtesting engine that tests ML predictions on historical data.

    Each prediction is treated as an independent trade:
    - Entry: When ML signal meets criteria
    - Exit: Fixed holding period (e.g., 5 days) or signal reversal
    - No portfolio tracking, just per-trade P&L analysis
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        holding_days: int = 5,
        confidence_threshold: float = 60.0,
        min_expected_return: float = 2.0,
        transaction_cost_pct: float = 0.2
    ):
        """
        Initialize simple backtester.

        Args:
            model_type: ML model to use (xgboost, random_forest, lightgbm)
            holding_days: How long to hold each position
            confidence_threshold: Minimum ML confidence to trade (0-100)
            min_expected_return: Minimum expected return to trade (%)
            transaction_cost_pct: Round-trip transaction cost (%)
        """
        self.model_type = model_type
        self.holding_days = holding_days
        self.confidence_threshold = confidence_threshold
        self.min_expected_return = min_expected_return
        self.transaction_cost_pct = transaction_cost_pct / 100  # Convert to decimal

        # Initialize components
        models_dir = Path(__file__).parent.parent / "models"

        # Check if using ensemble
        if model_type.startswith('ensemble'):
            # Extract ensemble strategy (e.g., "ensemble_voting" -> "voting")
            strategy = model_type.replace('ensemble_', '') if '_' in model_type else 'voting'
            self.prediction_engine = EnsemblePredictor(
                models_dir=str(models_dir),
                strategy=strategy
            )
        else:
            self.prediction_engine = PredictionEngine(
                models_dir=str(models_dir),
                model_type=model_type
            )
        self.data_collector = HistoricalDataCollector()
        self.ta_calculator = TechnicalIndicators()

        print(f"[SimpleBacktester] Initialized with:")
        print(f"  Model: {model_type}")
        print(f"  Holding period: {holding_days} days")
        print(f"  Confidence threshold: {confidence_threshold}%")
        print(f"  Min expected return: {min_expected_return}%")
        print(f"  Transaction costs: {transaction_cost_pct}%")

    def backtest_ticker(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        direction: str = "both"  # "bullish", "bearish", or "both"
    ) -> Dict[str, Any]:
        """
        Run backtest on a single ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            direction: Trade direction ("bullish", "bearish", or "both")

        Returns:
            Dictionary with backtest results and metrics
        """
        print(f"\n[SimpleBacktester] Running backtest on {ticker}")
        print(f"  Period: {start_date} to {end_date}")
        print(f"  Direction: {direction}")

        # Fetch historical data
        try:
            price_data = self._fetch_data(ticker, start_date, end_date)
        except Exception as e:
            return {
                'ticker': ticker,
                'error': f"Failed to fetch data: {str(e)}",
                'total_trades': 0
            }

        if len(price_data) < 100:
            return {
                'ticker': ticker,
                'error': f"Insufficient data: only {len(price_data)} days",
                'total_trades': 0
            }

        print(f"[SimpleBacktester] Processing {len(price_data)} days of data")

        # Generate trades
        trades = self._simulate_trades(ticker, price_data, direction)

        print(f"[SimpleBacktester] Generated {len(trades)} trades")

        # Calculate metrics
        metrics = self._calculate_metrics(ticker, trades, price_data, start_date, end_date)

        return metrics

    def backtest_multiple(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        direction: str = "both"
    ) -> Dict[str, Any]:
        """
        Run backtest on multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            direction: Trade direction

        Returns:
            Aggregated results across all tickers
        """
        print(f"\n[SimpleBacktester] Running backtest on {len(tickers)} tickers")

        all_results = []
        all_trades = []

        for ticker in tickers:
            result = self.backtest_ticker(ticker, start_date, end_date, direction)
            all_results.append(result)

            if 'trades' in result:
                all_trades.extend(result['trades'])

        # Aggregate metrics across all tickers
        aggregated = self._aggregate_results(all_results, all_trades)
        aggregated['individual_results'] = all_results

        return aggregated

    def _fetch_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data."""
        # Add buffer for indicators (need ~50 days before start)
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        buffer_start = start_dt - timedelta(days=100)
        buffer_start_str = buffer_start.strftime("%Y-%m-%d")

        # Fetch from data collector
        price_data = self.data_collector.fetch_historical_data(
            ticker=ticker,
            start_date=buffer_start_str,
            end_date=end_date
        )

        if price_data is None or price_data.empty:
            raise ValueError(f"No data available for {ticker}")

        # Ensure column names are capitalized (TechnicalIndicators expects capitalized)
        # yfinance returns 'Open', 'High', 'Low', 'Close', 'Volume'
        # data_collector returns 'open', 'high', 'low', 'close', 'volume'
        rename_map = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }
        price_data = price_data.rename(columns=rename_map)

        return price_data

    def _simulate_trades(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        direction: str
    ) -> List[Dict[str, Any]]:
        """
        Simulate trades by walking through history.

        Key: Only use data available at each point in time (no lookahead bias)!
        """
        trades = []

        # Need enough data for indicators + holding period
        min_history = 50  # Days needed for technical indicators
        max_index = len(price_data) - self.holding_days - 1

        for i in range(min_history, max_index):
            # Get data available up to this point (NO FUTURE DATA!)
            historical_slice = price_data.iloc[:i+1]

            try:
                # Calculate technical indicators
                indicators = self.ta_calculator.calculate_all_indicators(historical_slice)

                # Generate ML prediction
                prediction = self.prediction_engine.predict(indicators)

                # Debug first few predictions
                if i - min_history < 3:
                    print(f"[DEBUG] Prediction at index {i}: {prediction}")

                # Check if we should trade this signal
                if self._should_trade(prediction, direction):
                    trade = self._execute_trade(
                        ticker=ticker,
                        price_data=price_data,
                        entry_index=i,
                        prediction=prediction
                    )

                    if trade:
                        trades.append(trade)

            except Exception as e:
                # Skip this date if prediction fails
                if i - min_history < 3:
                    print(f"[DEBUG] Prediction failed at index {i}: {str(e)}")
                continue

        return trades

    def _should_trade(self, prediction: Dict[str, Any], direction: str) -> bool:
        """
        Determine if we should trade based on ML prediction.

        Entry criteria:
        - Confidence above threshold
        - Expected return above minimum
        - Direction matches filter
        """
        # Check for prediction errors
        if 'error' in prediction:
            return False

        # Check confidence
        if prediction.get('confidence', 0) < self.confidence_threshold:
            return False

        # Check expected return magnitude
        expected_return = prediction.get('expected_return', 0)
        if abs(expected_return) < self.min_expected_return:
            return False

        # Check direction filter
        pred_direction = prediction.get('direction', '').lower()

        if direction == "bullish" and pred_direction != "bullish":
            return False
        elif direction == "bearish" and pred_direction != "bearish":
            return False

        return True

    def _execute_trade(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        entry_index: int,
        prediction: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Simulate executing a trade.

        Entry: At close of entry_index
        Exit: At close of entry_index + holding_days
        """
        entry_date = price_data.index[entry_index]
        entry_price = float(price_data.iloc[entry_index]['Close'])

        exit_index = entry_index + self.holding_days
        exit_date = price_data.index[exit_index]
        exit_price = float(price_data.iloc[exit_index]['Close'])

        # Calculate returns
        gross_return_pct = ((exit_price - entry_price) / entry_price) * 100
        net_return_pct = gross_return_pct - (self.transaction_cost_pct * 100)

        # Determine if prediction was correct
        actual_direction = "bullish" if exit_price > entry_price else "bearish"
        pred_direction = prediction.get('direction', '').lower()
        prediction_correct = (actual_direction == pred_direction)

        return {
            'ticker': ticker,
            'entry_date': entry_date.strftime('%Y-%m-%d'),
            'exit_date': exit_date.strftime('%Y-%m-%d'),
            'holding_days': self.holding_days,
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'gross_return_pct': round(gross_return_pct, 2),
            'net_return_pct': round(net_return_pct, 2),
            'ml_confidence': round(prediction.get('confidence', 0), 1),
            'ml_expected_return': round(prediction.get('expected_return', 0), 2),
            'ml_direction': pred_direction,
            'actual_direction': actual_direction,
            'prediction_correct': prediction_correct,
            'profitable': net_return_pct > 0
        }

    def _build_equity_curve(self, trades: List[Dict[str, Any]], start_date: str, end_date: str) -> pd.Series:
        """
        Build an equity curve from trade results.

        Args:
            trades: List of trade dictionaries
            start_date: Start date for equity curve
            end_date: End date for equity curve

        Returns:
            Series indexed by date with cumulative portfolio value (starting at 100)
        """
        if not trades:
            # Return a flat line at 100
            dates = pd.date_range(start_date, end_date, freq='D')
            return pd.Series(100.0, index=dates)

        # Create a DataFrame with all trade dates
        trade_df = pd.DataFrame(trades)
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])

        # Build daily equity curve
        date_range = pd.date_range(start_date, end_date, freq='D')
        equity = pd.Series(100.0, index=date_range)

        # Apply returns on exit dates
        for _, trade in trade_df.iterrows():
            exit_date = trade['exit_date']
            if exit_date in equity.index:
                # Apply percentage return to current equity
                return_multiplier = 1 + (trade['net_return_pct'] / 100)
                equity.loc[exit_date:] *= return_multiplier

        return equity

    def _calculate_metrics(
        self,
        ticker: str,
        trades: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Calculate performance metrics from trades."""
        if not trades:
            return {
                'ticker': ticker,
                'total_trades': 0,
                'error': 'No trades generated with current criteria'
            }

        df = pd.DataFrame(trades)

        # Basic statistics
        total_trades = len(trades)
        winning_trades = (df['net_return_pct'] > 0).sum()
        losing_trades = (df['net_return_pct'] <= 0).sum()
        win_rate = (winning_trades / total_trades) * 100

        # Return statistics
        avg_return = df['net_return_pct'].mean()
        median_return = df['net_return_pct'].median()
        total_return = df['net_return_pct'].sum()

        winners = df[df['net_return_pct'] > 0]
        losers = df[df['net_return_pct'] <= 0]

        avg_winner = winners['net_return_pct'].mean() if len(winners) > 0 else 0
        avg_loser = losers['net_return_pct'].mean() if len(losers) > 0 else 0

        best_trade = df['net_return_pct'].max()
        worst_trade = df['net_return_pct'].min()

        # ML prediction analysis
        prediction_accuracy = (df['prediction_correct'].sum() / total_trades) * 100
        avg_confidence = df['ml_confidence'].mean()

        # Profit factor
        gross_profit = winners['net_return_pct'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['net_return_pct'].sum()) if len(losers) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Buy and hold comparison
        buy_hold_return = self._calculate_buy_hold_return(price_data, start_date, end_date)

        # --- Enhanced Metrics (Task 8) ---
        # Build equity curve for portfolio-level metrics
        equity_curve = self._build_equity_curve(trades, start_date, end_date)

        # Calculate enhanced metrics
        sharpe_ratio = BacktestMetrics.calculate_sharpe_ratio_annualized(equity_curve)
        sortino_ratio = BacktestMetrics.calculate_sortino_ratio(equity_curve)
        max_drawdown = BacktestMetrics.calculate_max_drawdown(equity_curve)
        calmar_ratio = BacktestMetrics.calculate_calmar_ratio(equity_curve)

        # Calibration curve
        calibration_curve = BacktestMetrics.calculate_calibration_curve(trades)

        # Regime analysis
        regime_breakdown = BacktestMetrics.analyze_by_regime(trades, price_data)

        return {
            'ticker': ticker,
            'period': f"{start_date} to {end_date}",
            'model_type': self.model_type,
            'total_trades': total_trades,
            'winning_trades': int(winning_trades),
            'losing_trades': int(losing_trades),
            'win_rate': round(win_rate, 1),
            'avg_return_per_trade': round(avg_return, 2),
            'median_return': round(median_return, 2),
            'total_return': round(total_return, 2),
            'avg_winner': round(avg_winner, 2),
            'avg_loser': round(avg_loser, 2),
            'best_trade': round(best_trade, 2),
            'worst_trade': round(worst_trade, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'inf',
            'prediction_accuracy': round(prediction_accuracy, 1),
            'avg_ml_confidence': round(avg_confidence, 1),
            'buy_hold_return': round(buy_hold_return, 2),
            # Enhanced metrics
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown': round(max_drawdown, 2),
            'calmar_ratio': round(calmar_ratio, 2),
            'calibration_curve': calibration_curve,
            'regime_breakdown': regime_breakdown,
            'trades': trades
        }

    def _calculate_buy_hold_return(
        self,
        price_data: pd.DataFrame,
        start_date: str,
        end_date: str
    ) -> float:
        """Calculate buy-and-hold return for comparison."""
        try:
            # Find prices closest to start and end dates
            mask = (price_data.index >= start_date) & (price_data.index <= end_date)
            filtered_data = price_data[mask]

            if len(filtered_data) < 2:
                return 0.0

            start_price = filtered_data.iloc[0]['Close']
            end_price = filtered_data.iloc[-1]['Close']

            return ((end_price - start_price) / start_price) * 100
        except Exception:
            return 0.0

    def _aggregate_results(
        self,
        results: List[Dict[str, Any]],
        all_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Aggregate results across multiple tickers."""
        # Filter out error results
        valid_results = [r for r in results if 'error' not in r and r.get('total_trades', 0) > 0]

        if not valid_results:
            return {
                'total_tickers': len(results),
                'error': 'No valid results',
                'total_trades': 0
            }

        df_trades = pd.DataFrame(all_trades)

        return {
            'total_tickers': len(results),
            'successful_tickers': len(valid_results),
            'total_trades': len(all_trades),
            'win_rate': round((df_trades['profitable'].sum() / len(all_trades)) * 100, 1),
            'avg_return_per_trade': round(df_trades['net_return_pct'].mean(), 2),
            'total_return': round(df_trades['net_return_pct'].sum(), 2),
            'prediction_accuracy': round((df_trades['prediction_correct'].sum() / len(all_trades)) * 100, 1),
            'avg_confidence': round(df_trades['ml_confidence'].mean(), 1),
            'best_ticker': max(valid_results, key=lambda x: x['avg_return_per_trade'])['ticker'],
            'worst_ticker': min(valid_results, key=lambda x: x['avg_return_per_trade'])['ticker']
        }
