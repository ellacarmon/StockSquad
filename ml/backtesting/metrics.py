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
