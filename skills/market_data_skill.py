"""
Market Data Skill

Wraps the MarketDataFetcher to provide market data capabilities to agents.
"""

from typing import Dict, Any, List, Optional

from skills.base import BaseSkill
from tools.market_data import MarketDataFetcher


class MarketDataSkill(BaseSkill):
    """
    Provides market data fetching capabilities to agents.

    This skill wraps the MarketDataFetcher class and provides methods to:
    - Get stock information (company info, sector, market cap)
    - Get historical price data (OHLCV)
    - Get financial metrics and ratios
    - Get recent news articles
    - Get earnings dates

    All methods are provided through the underlying MarketDataFetcher.
    """

    skill_name = "market_data"
    description = "Fetch stock market data, prices, financials, and news"
    version = "1.0.0"

    def __init__(self):
        """Initialize the market data skill."""
        self.fetcher = MarketDataFetcher()

    def execute(self, ticker: str, period: str = "1y", **kwargs) -> Dict[str, Any]:
        """
        Execute the skill's primary function: get complete stock data.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period (default: 1y)
            **kwargs: Additional parameters (e.g., comparison_tickers)

        Returns:
            Dictionary containing comprehensive stock data
        """
        comparison_tickers = kwargs.get('comparison_tickers')
        return self.get_complete_stock_data(ticker, period, comparison_tickers)

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing stock information (name, sector, market cap, etc.)
        """
        return self.fetcher.get_stock_info(ticker)

    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        comparison_tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get historical price data (OHLCV).

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            comparison_tickers: List of peer tickers to compare with

        Returns:
            Dictionary with price history and summary statistics
        """
        return self.fetcher.get_price_history(
            ticker=ticker,
            period=period,
            interval=interval,
            comparison_tickers=comparison_tickers
        )

    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Get key financial metrics and ratios.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing financial metrics (valuation, profitability, growth, etc.)
        """
        return self.fetcher.get_financials(ticker)

    def get_recent_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news articles for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to retrieve

        Returns:
            List of news article dictionaries
        """
        return self.fetcher.get_recent_news(ticker, limit)

    def get_earnings_dates(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get historical and upcoming earnings dates.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings date dictionaries
        """
        return self.fetcher.get_earnings_dates(ticker)

    def get_complete_stock_data(
        self,
        ticker: str,
        period: str = "1y",
        comparison_tickers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive stock data including info, price history, and financials.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period
            comparison_tickers: List of peer tickers to compare with

        Returns:
            Dictionary containing all available stock data
        """
        return self.fetcher.get_complete_stock_data(
            ticker=ticker,
            period=period,
            comparison_tickers=comparison_tickers
        )

    def format_for_llm(self, stock_data: Dict[str, Any]) -> str:
        """
        Format stock data as a readable string for LLM consumption.

        Args:
            stock_data: Complete stock data dictionary

        Returns:
            Formatted string representation
        """
        return self.fetcher.format_for_llm(stock_data)
