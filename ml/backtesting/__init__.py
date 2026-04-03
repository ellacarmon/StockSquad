"""
Simple backtesting engine for ML stock predictions.
Tests if ML signals would have been profitable on historical data.
"""

from .simple_backtester import SimpleBacktester
from .metrics import BacktestMetrics
from .report import BacktestReport

__all__ = ['SimpleBacktester', 'BacktestMetrics', 'BacktestReport']
