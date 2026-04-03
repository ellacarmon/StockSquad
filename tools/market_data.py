"""
Market data fetching tools using yfinance with Polygon.io fallback.
Provides functions to retrieve OHLCV data and basic financial metrics.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json
import os
from pathlib import Path

import yfinance as yf
import pandas as pd
import requests
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class MarketDataFetcher:
    """Fetches market data and financials for stock tickers using yfinance."""

    def __init__(self):
        """Initialize the market data fetcher."""
        self.cache: Dict[str, Any] = {}

    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing stock information

        Raises:
            ValueError: If ticker is invalid or data cannot be fetched
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Check if we got valid data
            if not info or "symbol" not in info:
                raise ValueError(f"Invalid ticker: {ticker}")

            # Extract key information
            return {
                "symbol": info.get("symbol", ticker.upper()),
                "name": info.get("longName", info.get("shortName", "N/A")),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "N/A"),
                "description": info.get("longBusinessSummary", "N/A"),
            }
        except Exception as e:
            raise ValueError(f"Failed to fetch stock info for {ticker}: {str(e)}")

    def _fetch_from_polygon(self, ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Fetch price history from Polygon.io as fallback.

        Args:
            ticker: Stock ticker symbol
            period: Time period

        Returns:
            DataFrame with OHLCV data or None
        """
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print(f"[MarketDataFetcher] No POLYGON_API_KEY found - skipping Polygon fallback")
            return None

        try:
            # Calculate date range
            end = datetime.now()
            if period == "1d":
                start = end - timedelta(days=1)
            elif period == "5d":
                start = end - timedelta(days=5)
            elif period == "1mo":
                start = end - timedelta(days=30)
            elif period == "3mo":
                start = end - timedelta(days=90)
            elif period == "6mo":
                start = end - timedelta(days=180)
            elif period == "1y":
                start = end - timedelta(days=365)
            elif period == "2y":
                start = end - timedelta(days=730)
            elif period == "5y":
                start = end - timedelta(days=1825)
            else:  # max or others
                start = end - timedelta(days=3650)

            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')

            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': api_key
            }

            print(f"[MarketDataFetcher] Trying Polygon.io for {ticker}...")
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"[MarketDataFetcher] Polygon API error {response.status_code}")
                return None

            data = response.json()

            if data.get('status') != 'OK' or not data.get('results'):
                print(f"[MarketDataFetcher] No Polygon data for {ticker}")
                return None

            # Convert to DataFrame
            results = data['results']
            df = pd.DataFrame(results)

            # Rename columns to match yfinance format
            df = df.rename(columns={
                'v': 'Volume',
                'o': 'Open',
                'c': 'Close',
                'h': 'High',
                'l': 'Low',
                't': 'timestamp'
            })

            # Convert timestamp to datetime
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('Date')

            # Select columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

            print(f"[MarketDataFetcher] ✓ Polygon.io: {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            print(f"[MarketDataFetcher] Polygon fetch error: {e}")
            return None

    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
        comparison_tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get historical price data (OHLCV) with Polygon.io fallback, and optional peers.

        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            comparison_tickers: List of peer tickers to compare with

        Returns:
            Dictionary with price history and summary statistics
        """
        hist = None

        # Try yfinance first
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, interval=interval)

            if not hist.empty:
                print(f"[MarketDataFetcher] yfinance: {len(hist)} rows for {ticker}")
            else:
                print(f"[MarketDataFetcher] yfinance: No data for {ticker}")

        except Exception as e:
            print(f"[MarketDataFetcher] yfinance error for {ticker}: {e}")

        # If yfinance failed and interval is 1d, try Polygon
        if (hist is None or hist.empty) and interval == "1d":
            hist = self._fetch_from_polygon(ticker, period)

        # Final validation
        if hist is None or hist.empty:
            raise ValueError(f"No price history available for {ticker} from any source")

        # Calculate summary statistics
        latest = hist.iloc[-1]
        first = hist.iloc[0]
        price_change = latest["Close"] - first["Close"]
        price_change_pct = (price_change / first["Close"]) * 100

        # Calculate percent change from start for mapping
        hist['Percent_Change'] = ((hist['Close'] - first['Close']) / first['Close']) * 100
        
        # Rename Close to Ticker_Close and Percent_Change to Ticker_Percent
        hist = hist.rename(columns={'Close': f'{ticker.upper()}_Close', 'Percent_Change': f'{ticker.upper()}_Percent'})

        # Fetch comparison tickers
        if comparison_tickers:
            for comp_ticker in comparison_tickers:
                comp_ticker = comp_ticker.upper()
                try:
                    comp_stock = yf.Ticker(comp_ticker)
                    comp_hist = comp_stock.history(period=period, interval=interval)
                    if not comp_hist.empty:
                        # Reindex to match the main ticker's timeline
                        comp_hist = comp_hist.reindex(hist.index, method='ffill')
                        first_comp = comp_hist['Close'].iloc[0]
                        # For the first valid value, we can use the first non-NaN if needed, but ffill handles gaps.
                        if pd.isna(first_comp):
                            first_valid_idx = comp_hist['Close'].first_valid_index()
                            if first_valid_idx:
                                first_comp = comp_hist['Close'].loc[first_valid_idx]
                        
                        if first_comp and not pd.isna(first_comp):
                            # Calculate percentage change based on initial price
                            comp_percent = ((comp_hist['Close'] - first_comp) / first_comp) * 100
                            hist[f'{comp_ticker}_Close'] = comp_hist['Close']
                            hist[f'{comp_ticker}_Percent'] = comp_percent
                except Exception as e:
                    print(f"[MarketDataFetcher] Failed to fetch comparison ticker {comp_ticker}: {e}")

        return {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "data_points": len(hist),
            "start_date": hist.index[0].strftime("%Y-%m-%d"),
            "end_date": hist.index[-1].strftime("%Y-%m-%d"),
            "latest_price": float(latest["Close"]),
            "latest_volume": int(latest["Volume"]),
            "period_high": float(hist["High"].max()),
            "period_low": float(hist["Low"].min()),
            "price_change": float(price_change),
            "price_change_percent": float(price_change_pct),
            "average_volume": int(hist["Volume"].mean()),
            "data": hist.reset_index().to_dict("records"),
        }

    def get_financials(self, ticker: str) -> Dict[str, Any]:
        """
        Get key financial metrics and ratios.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary containing financial metrics
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract financial metrics
            financials = {
                "ticker": ticker.upper(),
                "valuation": {
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    "ev_to_revenue": info.get("enterpriseToRevenue"),
                    "ev_to_ebitda": info.get("enterpriseToEbitda"),
                },
                "profitability": {
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "gross_margin": info.get("grossMargins"),
                    "roe": info.get("returnOnEquity"),
                    "roa": info.get("returnOnAssets"),
                },
                "growth": {
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue": info.get("totalRevenue"),
                    "earnings": info.get("netIncomeToCommon"),
                },
                "financial_health": {
                    "total_cash": info.get("totalCash"),
                    "total_debt": info.get("totalDebt"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    "free_cash_flow": info.get("freeCashflow"),
                },
                "dividend": {
                    "dividend_rate": info.get("dividendRate"),
                    "dividend_yield": info.get("dividendYield"),
                    "payout_ratio": info.get("payoutRatio"),
                },
                "analyst_metrics": {
                    "target_high": info.get("targetHighPrice"),
                    "target_low": info.get("targetLowPrice"),
                    "target_mean": info.get("targetMeanPrice"),
                    "target_median": info.get("targetMedianPrice"),
                    "recommendation": info.get("recommendationKey"),
                    "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
                },
            }

            return financials
        except Exception as e:
            raise ValueError(f"Failed to fetch financials for {ticker}: {str(e)}")

    def get_recent_news(self, ticker: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent news articles for a ticker.

        Args:
            ticker: Stock ticker symbol
            limit: Maximum number of news articles to retrieve

        Returns:
            List of news article dictionaries
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news

            if not news:
                return []

            # Format news articles
            formatted_news = []
            for article in news[:limit]:
                formatted_news.append({
                    "title": article.get("title", ""),
                    "publisher": article.get("publisher", ""),
                    "link": article.get("link", ""),
                    "published": datetime.fromtimestamp(
                        article.get("providerPublishTime", 0)
                    ).isoformat() if article.get("providerPublishTime") else None,
                    "type": article.get("type", ""),
                    "thumbnail": article.get("thumbnail", {}).get("resolutions", [{}])[0].get("url") if article.get("thumbnail") else None,
                })

            return formatted_news
        except Exception as e:
            print(f"Warning: Failed to fetch news for {ticker}: {str(e)}")
            return []

    def get_earnings_dates(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get historical and upcoming earnings dates for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of earnings date dictionaries with date and type
        """
        try:
            stock = yf.Ticker(ticker)

            earnings_dates = []

            # Try to get earnings dates (historical)
            try:
                earnings_history = stock.earnings_dates
                if earnings_history is not None and not earnings_history.empty:
                    # Get dates from the last 2 years
                    for date_idx in earnings_history.index:
                        earnings_dates.append({
                            "date": date_idx.strftime("%Y-%m-%d"),
                            "type": "earnings",
                            "reported": earnings_history.loc[date_idx].get('Reported EPS') if hasattr(earnings_history.loc[date_idx], 'get') else None
                        })
            except Exception as e:
                print(f"[MarketDataFetcher] Could not fetch earnings_dates: {e}")

            # Try to get upcoming earnings from calendar
            try:
                calendar = stock.calendar
                if calendar is not None and 'Earnings Date' in calendar:
                    earnings_date = calendar['Earnings Date']
                    if isinstance(earnings_date, (list, tuple)) and len(earnings_date) > 0:
                        # Sometimes it's a range, take the first date
                        upcoming_date = earnings_date[0]
                        if pd.notna(upcoming_date):
                            earnings_dates.append({
                                "date": pd.to_datetime(upcoming_date).strftime("%Y-%m-%d"),
                                "type": "earnings",
                                "upcoming": True
                            })
                    elif pd.notna(earnings_date):
                        earnings_dates.append({
                            "date": pd.to_datetime(earnings_date).strftime("%Y-%m-%d"),
                            "type": "earnings",
                            "upcoming": True
                        })
            except Exception as e:
                print(f"[MarketDataFetcher] Could not fetch calendar: {e}")

            print(f"[MarketDataFetcher] Found {len(earnings_dates)} earnings dates for {ticker}")
            return earnings_dates

        except Exception as e:
            print(f"[MarketDataFetcher] Failed to fetch earnings dates for {ticker}: {str(e)}")
            return []

    def get_complete_stock_data(self, ticker: str, period: str = "1y", comparison_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get comprehensive stock data including info, price history, and financials.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period
            comparison_tickers: List of peer tickers to compare with

        Returns:
            Dictionary containing all available stock data
        """
        ticker = ticker.upper()

        try:
            data = {
                "ticker": ticker,
                "fetched_at": datetime.now().isoformat(),
                "info": self.get_stock_info(ticker),
                "price_history": self.get_price_history(ticker, period=period, comparison_tickers=comparison_tickers),
                "financials": self.get_financials(ticker),
                "recent_news": self.get_recent_news(ticker),
                "earnings_dates": self.get_earnings_dates(ticker),
            }

            return data
        except Exception as e:
            raise ValueError(f"Failed to fetch complete data for {ticker}: {str(e)}")

    def format_for_llm(self, stock_data: Dict[str, Any]) -> str:
        """
        Format stock data as a readable string for LLM consumption.

        Args:
            stock_data: Complete stock data dictionary

        Returns:
            Formatted string representation
        """
        info = stock_data.get("info", {})
        price = stock_data.get("price_history", {})
        financials = stock_data.get("financials", {})

        formatted = f"""
Stock Analysis Data for {stock_data.get('ticker', 'N/A')}
{'=' * 60}

Company Information:
- Name: {info.get('name', 'N/A')}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}
- Exchange: {info.get('exchange', 'N/A')}

Current Price & Market Data:
- Price: ${info.get('current_price', 0):.2f}
- Market Cap: ${info.get('market_cap', 0):,.0f}
- Period Change: {price.get('price_change_percent', 0):.2f}% ({price.get('period', 'N/A')})
- 52-Week High: ${price.get('period_high', 0):.2f}
- 52-Week Low: ${price.get('period_low', 0):.2f}

Valuation Metrics:
- P/E Ratio: {financials.get('valuation', {}).get('pe_ratio', 'N/A')}
- Forward P/E: {financials.get('valuation', {}).get('forward_pe', 'N/A')}
- Price to Book: {financials.get('valuation', {}).get('price_to_book', 'N/A')}
- PEG Ratio: {financials.get('valuation', {}).get('peg_ratio', 'N/A')}

Profitability:
- Profit Margin: {financials.get('profitability', {}).get('profit_margin', 'N/A')}
- Operating Margin: {financials.get('profitability', {}).get('operating_margin', 'N/A')}
- ROE: {financials.get('profitability', {}).get('roe', 'N/A')}

Financial Health:
- Debt to Equity: {financials.get('financial_health', {}).get('debt_to_equity', 'N/A')}
- Current Ratio: {financials.get('financial_health', {}).get('current_ratio', 'N/A')}

Analyst Recommendation: {financials.get('analyst_metrics', {}).get('recommendation', 'N/A')}
Target Price: ${financials.get('analyst_metrics', {}).get('target_mean', 0):.2f}
"""
        return formatted.strip()
