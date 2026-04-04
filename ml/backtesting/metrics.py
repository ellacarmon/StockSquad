"""
Backtest Metrics Calculator
Helper functions for calculating performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any


class BacktestMetrics:
    """Calculate various performance metrics from trade data."""

    @staticmethod
    def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
        """Calculate percentage of winning trades."""
        if not trades:
            return 0.0
        winning = sum(1 for t in trades if t['net_return_pct'] > 0)
        return (winning / len(trades)) * 100

    @staticmethod
    def calculate_profit_factor(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate profit factor (gross profits / gross losses).
        >1.0 means profitable, >2.0 is excellent.
        """
        if not trades:
            return 0.0

        gross_profit = sum(t['net_return_pct'] for t in trades if t['net_return_pct'] > 0)
        gross_loss = abs(sum(t['net_return_pct'] for t in trades if t['net_return_pct'] <= 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    @staticmethod
    def calculate_expectancy(trades: List[Dict[str, Any]]) -> float:
        """
        Calculate expectancy per trade.
        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        """
        if not trades:
            return 0.0

        winners = [t['net_return_pct'] for t in trades if t['net_return_pct'] > 0]
        losers = [t['net_return_pct'] for t in trades if t['net_return_pct'] <= 0]

        win_rate = len(winners) / len(trades)
        loss_rate = len(losers) / len(trades)

        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0

        return (win_rate * avg_win) - (loss_rate * avg_loss)

    @staticmethod
    def calculate_sharpe_ratio(
        trades: List[Dict[str, Any]],
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio from trade returns.

        Note: This is a simplified version using individual trade returns,
        not true portfolio daily returns.
        """
        if not trades:
            return 0.0

        returns = [t['net_return_pct'] / 100 for t in trades]  # Convert to decimal

        if len(returns) < 2:
            return 0.0

        # Mean excess return
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)

        if std_return == 0:
            return 0.0

        # Annualize
        sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)

        return sharpe

    @staticmethod
    def calculate_max_consecutive_wins(trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum streak of consecutive winning trades."""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade['net_return_pct'] > 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def calculate_max_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
        """Calculate maximum streak of consecutive losing trades."""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            if trade['net_return_pct'] <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def calculate_return_distribution(trades: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate distribution of returns across bins.

        Returns:
            Dict with bins as keys and counts as values
        """
        if not trades:
            return {}

        returns = [t['net_return_pct'] for t in trades]

        bins = {
            'big_loss (<-5%)': sum(1 for r in returns if r < -5),
            'loss (-5% to 0%)': sum(1 for r in returns if -5 <= r < 0),
            'small_gain (0% to 2%)': sum(1 for r in returns if 0 <= r < 2),
            'medium_gain (2% to 5%)': sum(1 for r in returns if 2 <= r < 5),
            'big_gain (>5%)': sum(1 for r in returns if r >= 5)
        }

        return bins

    @staticmethod
    def calculate_prediction_accuracy_by_confidence(
        trades: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze ML prediction accuracy segmented by confidence level.

        Returns:
            Dict with confidence buckets and their statistics
        """
        if not trades:
            return {}

        # Define confidence buckets
        buckets = {
            'high (>70%)': [],
            'medium (60-70%)': [],
            'low (50-60%)': []
        }

        for trade in trades:
            conf = trade['ml_confidence']
            if conf > 70:
                buckets['high (>70%)'].append(trade)
            elif conf > 60:
                buckets['medium (60-70%)'].append(trade)
            elif conf > 50:
                buckets['low (50-60%)'].append(trade)

        # Calculate stats for each bucket
        results = {}
        for bucket_name, bucket_trades in buckets.items():
            if bucket_trades:
                accuracy = sum(1 for t in bucket_trades if t['prediction_correct']) / len(bucket_trades) * 100
                win_rate = sum(1 for t in bucket_trades if t['net_return_pct'] > 0) / len(bucket_trades) * 100
                avg_return = np.mean([t['net_return_pct'] for t in bucket_trades])

                results[bucket_name] = {
                    'count': len(bucket_trades),
                    'accuracy': round(accuracy, 1),
                    'win_rate': round(win_rate, 1),
                    'avg_return': round(avg_return, 2)
                }

        return results

    @staticmethod
    def calculate_sharpe_ratio_annualized(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate annualized Sharpe ratio from an equity curve.

        Args:
            equity_curve: Series of portfolio values over time (indexed by date)
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Annualized Sharpe ratio: (mean_return / std_return) * sqrt(252)

        **Validates: Requirements 8.1**
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()

        if len(daily_returns) == 0:
            return 0.0

        mean_return = daily_returns.mean()
        std_return = daily_returns.std(ddof=1)

        if std_return == 0:
            return 0.0

        # Annualized Sharpe ratio
        sharpe = (mean_return / std_return) * np.sqrt(252)

        return sharpe

    @staticmethod
    def calculate_sortino_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.02, target_return: float = 0.0) -> float:
        """
        Calculate Sortino ratio using downside deviation only.

        Args:
            equity_curve: Series of portfolio values over time
            risk_free_rate: Annual risk-free rate (default 2%)
            target_return: Minimum acceptable return (default 0%)

        Returns:
            Annualized Sortino ratio

        **Validates: Requirements 8.2**
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate daily returns
        daily_returns = equity_curve.pct_change().dropna()

        if len(daily_returns) == 0:
            return 0.0

        mean_return = daily_returns.mean()

        # Calculate downside deviation (only negative returns relative to target)
        downside_returns = daily_returns[daily_returns < target_return]

        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.sqrt(np.mean(downside_returns ** 2))

        if downside_std == 0:
            return 0.0

        # Annualized Sortino ratio
        sortino = (mean_return / downside_std) * np.sqrt(252)

        return sortino

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum peak-to-trough drawdown.

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Maximum drawdown as a negative percentage (e.g., -15.5 for a 15.5% drawdown)

        **Validates: Requirements 8.3**
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - running_max) / running_max * 100

        # Maximum drawdown is the minimum value (most negative)
        max_dd = drawdown.min()

        return max_dd if not np.isnan(max_dd) else 0.0

    @staticmethod
    def calculate_calmar_ratio(equity_curve: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / abs(max drawdown)).

        Args:
            equity_curve: Series of portfolio values over time

        Returns:
            Calmar ratio

        **Validates: Requirements 8.4**
        """
        if len(equity_curve) < 2:
            return 0.0

        # Calculate total return
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]

        # Annualize the return
        days = len(equity_curve)
        years = days / 252
        if years == 0:
            return 0.0

        annual_return = (1 + total_return) ** (1 / years) - 1

        # Calculate max drawdown
        max_dd = BacktestMetrics.calculate_max_drawdown(equity_curve)

        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0

        # Calmar ratio
        calmar = annual_return / abs(max_dd / 100)

        return calmar

    @staticmethod
    def calculate_calibration_curve(trades: List[Dict[str, Any]], n_bins: int = 10) -> Dict[str, Any]:
        """
        Calculate calibration curve mapping confidence deciles to actual win rates.

        Args:
            trades: List of trade dictionaries with 'ml_confidence' and 'prediction_correct' fields
            n_bins: Number of confidence bins (default 10 for deciles)

        Returns:
            Dict with confidence bins and their actual win rates

        **Validates: Requirements 8.5**
        """
        if not trades:
            return {}

        df = pd.DataFrame(trades)

        if 'ml_confidence' not in df.columns or 'prediction_correct' not in df.columns:
            return {}

        # Create confidence bins
        try:
            df['confidence_bin'] = pd.qcut(
                df['ml_confidence'],
                q=n_bins,
                labels=False,
                duplicates='drop'
            )
        except ValueError:
            # If we can't create n_bins, try fewer
            try:
                df['confidence_bin'] = pd.qcut(
                    df['ml_confidence'],
                    q=5,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                return {}

        # Calculate actual win rate for each bin
        calibration = {}
        for bin_idx in sorted(df['confidence_bin'].unique()):
            bin_trades = df[df['confidence_bin'] == bin_idx]
            avg_confidence = bin_trades['ml_confidence'].mean()
            actual_win_rate = (bin_trades['prediction_correct'].sum() / len(bin_trades)) * 100

            calibration[f"bin_{int(bin_idx)}"] = {
                'avg_confidence': round(avg_confidence, 1),
                'actual_win_rate': round(actual_win_rate, 1),
                'count': len(bin_trades)
            }

        return calibration

    @staticmethod
    def analyze_by_regime(trades: List[Dict[str, Any]], price_data: pd.DataFrame, atr_period: int = 14) -> Dict[str, Any]:
        """
        Analyze performance broken down by market regime (trending vs. ranging).

        Args:
            trades: List of trade dictionaries with 'entry_date' field
            price_data: DataFrame with OHLC data
            atr_period: Period for ATR calculation (default 14)

        Returns:
            Dict with regime-specific performance metrics

        **Validates: Requirements 8.6**
        """
        if not trades or price_data.empty:
            return {}

        # Calculate regime indicators
        # Trending: ADX > 25 or large directional movement
        # Ranging: ADX < 25 or low directional movement

        # Simple regime classification using rolling volatility
        if 'Close' in price_data.columns:
            close_col = 'Close'
        elif 'close' in price_data.columns:
            close_col = 'close'
        else:
            return {}

        # Calculate 20-day rolling volatility
        returns = price_data[close_col].pct_change()
        rolling_vol = returns.rolling(window=20).std()

        # Calculate 50-day SMA
        sma_50 = price_data[close_col].rolling(window=50).mean()

        # Classify regime
        # Trending: price far from SMA and high volatility
        # Ranging: price near SMA and low volatility
        price_deviation = abs((price_data[close_col] - sma_50) / sma_50)

        median_vol = rolling_vol.median()
        median_dev = price_deviation.median()

        trending_trades = []
        ranging_trades = []

        for trade in trades:
            entry_date = pd.to_datetime(trade['entry_date'])

            if entry_date not in price_data.index:
                continue

            vol = rolling_vol.loc[entry_date] if entry_date in rolling_vol.index else median_vol
            dev = price_deviation.loc[entry_date] if entry_date in price_deviation.index else median_dev

            # Classify as trending if volatility > median AND deviation > median
            if vol > median_vol and dev > median_dev:
                trending_trades.append(trade)
            else:
                ranging_trades.append(trade)

        # Calculate metrics for each regime
        result = {}

        if trending_trades:
            trending_df = pd.DataFrame(trending_trades)
            result['trending'] = {
                'count': len(trending_trades),
                'win_rate': round((trending_df['net_return_pct'] > 0).sum() / len(trending_trades) * 100, 1),
                'avg_return': round(trending_df['net_return_pct'].mean(), 2),
                'prediction_accuracy': round((trending_df['prediction_correct'].sum() / len(trending_trades)) * 100, 1) if 'prediction_correct' in trending_df.columns else 0.0
            }

        if ranging_trades:
            ranging_df = pd.DataFrame(ranging_trades)
            result['ranging'] = {
                'count': len(ranging_trades),
                'win_rate': round((ranging_df['net_return_pct'] > 0).sum() / len(ranging_trades) * 100, 1),
                'avg_return': round(ranging_df['net_return_pct'].mean(), 2),
                'prediction_accuracy': round((ranging_df['prediction_correct'].sum() / len(ranging_trades)) * 100, 1) if 'prediction_correct' in ranging_df.columns else 0.0
            }

        return result
