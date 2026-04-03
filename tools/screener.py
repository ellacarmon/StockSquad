"""
Stock Screener Engine
Core screening functionality with criteria-based filtering.
"""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


class CriteriaOperator(Enum):
    """Comparison operators for screening criteria."""
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "="
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    BETWEEN = "between"
    IN = "in"


@dataclass
class ScreenCriteria:
    """Defines a single screening criterion."""
    field: str  # e.g., "rsi", "pe_ratio", "price"
    operator: CriteriaOperator
    value: Any  # Comparison value
    weight: float = 1.0  # Weight for scoring (0-1)


@dataclass
class ScreenResult:
    """Result of screening a single stock."""
    ticker: str
    passed: bool
    score: float  # 0-100
    metrics: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class QuickMetrics:
    """Fast metric calculations without LLM."""

    @staticmethod
    def fetch_basic_data(ticker: str, period: str = "1mo") -> Optional[Dict[str, Any]]:
        """
        Fetch basic stock data quickly.

        Args:
            ticker: Stock ticker
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y)

        Returns:
            Dictionary with basic metrics or None if error
        """
        try:
            stock = yf.Ticker(ticker)

            # Get price data
            hist = stock.history(period=period)
            if hist.empty:
                return None

            # Get latest price data
            latest = hist.iloc[-1]
            current_price = latest['Close']
            volume = latest['Volume']

            # Calculate returns
            returns_1d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100 if len(hist) > 1 else 0
            returns_5d = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100 if len(hist) >= 5 else 0

            # Get info (cached by yfinance)
            info = stock.info

            # Extract key metrics
            metrics = {
                "ticker": ticker,
                "price": current_price,
                "volume": volume,
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "50d_avg": info.get("fiftyDayAverage"),
                "200d_avg": info.get("twoHundredDayAverage"),
                "profit_margin": info.get("profitMargins", 0) * 100 if info.get("profitMargins") else None,
                "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else None,
                "debt_to_equity": info.get("debtToEquity"),
                "revenue_growth": info.get("revenueGrowth", 0) * 100 if info.get("revenueGrowth") else None,
                "earnings_growth": info.get("earningsGrowth", 0) * 100 if info.get("earningsGrowth") else None,
                "returns_1d": returns_1d,
                "returns_5d": returns_5d,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
            }

            # Calculate simple technical indicators
            if len(hist) >= 14:
                metrics["rsi"] = QuickMetrics.calculate_rsi(hist['Close'], period=14)

            if len(hist) >= 20:
                metrics["sma_20"] = hist['Close'].rolling(window=20).mean().iloc[-1]
                price_vs_sma20 = ((current_price / metrics["sma_20"]) - 1) * 100
                metrics["price_vs_sma20"] = price_vs_sma20

            if len(hist) >= 50:
                metrics["sma_50"] = hist['Close'].rolling(window=50).mean().iloc[-1]

            # Volume analysis
            avg_volume = hist['Volume'].mean()
            metrics["avg_volume"] = avg_volume
            metrics["volume_ratio"] = volume / avg_volume if avg_volume > 0 else 1.0

            return metrics

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """
        Calculate RSI (Relative Strength Index).

        Args:
            prices: Price series
            period: RSI period

        Returns:
            RSI value (0-100)
        """
        if len(prices) < period + 1:
            return 50.0  # Neutral default

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


class ScreenerEngine:
    """Core screening engine."""

    def __init__(self, max_workers: int = 10):
        """
        Initialize screener engine.

        Args:
            max_workers: Maximum parallel workers for fetching data
        """
        self.max_workers = max_workers

    def evaluate_criteria(self, metrics: Dict[str, Any], criteria: ScreenCriteria) -> bool:
        """
        Evaluate a single criterion against metrics.

        Args:
            metrics: Stock metrics
            criteria: Criterion to evaluate

        Returns:
            True if criterion is met
        """
        value = metrics.get(criteria.field)

        if value is None:
            return False

        try:
            if criteria.operator == CriteriaOperator.GREATER_THAN:
                return value > criteria.value
            elif criteria.operator == CriteriaOperator.LESS_THAN:
                return value < criteria.value
            elif criteria.operator == CriteriaOperator.EQUAL:
                return value == criteria.value
            elif criteria.operator == CriteriaOperator.GREATER_EQUAL:
                return value >= criteria.value
            elif criteria.operator == CriteriaOperator.LESS_EQUAL:
                return value <= criteria.value
            elif criteria.operator == CriteriaOperator.BETWEEN:
                return criteria.value[0] <= value <= criteria.value[1]
            elif criteria.operator == CriteriaOperator.IN:
                return value in criteria.value
            else:
                return False
        except Exception:
            return False

    def calculate_score(
        self,
        metrics: Dict[str, Any],
        criteria_list: List[ScreenCriteria],
        passed_criteria: List[ScreenCriteria]
    ) -> float:
        """
        Calculate a score based on how many weighted criteria were met.

        Args:
            metrics: Stock metrics
            criteria_list: All criteria
            passed_criteria: Criteria that passed

        Returns:
            Score from 0-100
        """
        if not criteria_list:
            return 100.0

        total_weight = sum(c.weight for c in criteria_list)
        passed_weight = sum(c.weight for c in passed_criteria)

        return (passed_weight / total_weight) * 100 if total_weight > 0 else 0.0

    def screen_single_stock(
        self,
        ticker: str,
        criteria_list: List[ScreenCriteria],
        require_all: bool = True
    ) -> ScreenResult:
        """
        Screen a single stock against criteria.

        Args:
            ticker: Stock ticker
            criteria_list: List of screening criteria
            require_all: If True, all criteria must pass. If False, any criteria passing is sufficient.

        Returns:
            ScreenResult object
        """
        # Fetch metrics
        metrics = QuickMetrics.fetch_basic_data(ticker)

        if metrics is None:
            return ScreenResult(
                ticker=ticker,
                passed=False,
                score=0.0,
                reasons=["Failed to fetch data"]
            )

        # Evaluate criteria
        passed_criteria = []
        failed_criteria = []
        reasons = []

        for criteria in criteria_list:
            if self.evaluate_criteria(metrics, criteria):
                passed_criteria.append(criteria)
                reasons.append(f"✓ {criteria.field} {criteria.operator.value} {criteria.value}")
            else:
                failed_criteria.append(criteria)
                reasons.append(f"✗ {criteria.field} {criteria.operator.value} {criteria.value}")

        # Determine if passed
        if require_all:
            passed = len(failed_criteria) == 0
        else:
            passed = len(passed_criteria) > 0

        # Calculate score
        score = self.calculate_score(metrics, criteria_list, passed_criteria)

        return ScreenResult(
            ticker=ticker,
            passed=passed,
            score=score,
            metrics=metrics,
            reasons=reasons
        )

    def screen_multiple_stocks(
        self,
        tickers: List[str],
        criteria_list: List[ScreenCriteria],
        require_all: bool = True,
        min_score: float = 0.0
    ) -> List[ScreenResult]:
        """
        Screen multiple stocks in parallel.

        Args:
            tickers: List of stock tickers
            criteria_list: List of screening criteria
            require_all: If True, all criteria must pass
            min_score: Minimum score to include in results

        Returns:
            List of ScreenResult objects
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.screen_single_stock, ticker, criteria_list, require_all): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                try:
                    result = future.result()
                    if result.passed and result.score >= min_score:
                        results.append(result)
                except Exception as e:
                    ticker = future_to_ticker[future]
                    print(f"Error screening {ticker}: {e}")

        # Sort by score (descending)
        results.sort(key=lambda x: x.score, reverse=True)

        return results

    def quick_filter(
        self,
        tickers: List[str],
        filter_func: Callable[[Dict[str, Any]], bool]
    ) -> List[str]:
        """
        Quick filter tickers based on a custom function.

        Args:
            tickers: List of tickers
            filter_func: Function that takes metrics dict and returns bool

        Returns:
            List of tickers that passed the filter
        """
        passed = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(QuickMetrics.fetch_basic_data, ticker): ticker
                for ticker in tickers
            }

            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    metrics = future.result()
                    if metrics and filter_func(metrics):
                        passed.append(ticker)
                except Exception as e:
                    print(f"Error filtering {ticker}: {e}")

        return passed


if __name__ == "__main__":
    # Example usage
    print("Testing Stock Screener Engine...\n")

    # Create screening criteria
    criteria = [
        ScreenCriteria("rsi", CriteriaOperator.LESS_THAN, 30, weight=1.0),
        ScreenCriteria("pe_ratio", CriteriaOperator.LESS_THAN, 20, weight=0.8),
        ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.5, weight=0.5),
    ]

    # Screen a few stocks
    engine = ScreenerEngine(max_workers=5)
    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]

    print(f"Screening {len(test_tickers)} stocks...")
    results = engine.screen_multiple_stocks(test_tickers, criteria, require_all=False, min_score=30)

    print(f"\nFound {len(results)} matches:\n")
    for result in results:
        print(f"{result.ticker}: Score {result.score:.1f}/100")
        print(f"  Price: ${result.metrics.get('price', 0):.2f}")
        print(f"  RSI: {result.metrics.get('rsi', 0):.1f}")
        print(f"  P/E: {result.metrics.get('pe_ratio', 'N/A')}")
        print()
