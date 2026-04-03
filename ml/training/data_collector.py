"""
Historical Data Collector for ML Training
Fetches and stores historical stock data for model training.
"""

import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests
import os

from ml.sentiment_features import extract_sentiment_features


class HistoricalDataCollector:
    """Collects and stores historical stock data for ML training."""

    def __init__(self, db_path: str = "ml/training/stock_data.db"):
        """
        Initialize data collector.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Price history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                adj_close REAL,
                UNIQUE(ticker, date)
            )
        """)

        # Stock info table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                ticker TEXT PRIMARY KEY,
                name TEXT,
                sector TEXT,
                industry TEXT,
                market_cap REAL,
                last_updated TIMESTAMP
            )
        """)

        # Technical indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                rsi_14 REAL,
                macd REAL,
                macd_signal REAL,
                macd_hist REAL,
                sma_20 REAL,
                sma_50 REAL,
                sma_200 REAL,
                ema_12 REAL,
                ema_26 REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                volume_sma_20 REAL,
                UNIQUE(ticker, date)
            )
        """)

        # Labels table (for supervised learning)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                forward_1d_return REAL,
                forward_5d_return REAL,
                forward_10d_return REAL,
                forward_20d_return REAL,
                direction_1d INTEGER,  -- 1=up, 0=down
                direction_5d INTEGER,
                direction_10d INTEGER,
                direction_20d INTEGER,
                max_drawdown_5d REAL,
                UNIQUE(ticker, date)
            )
        """)

        # Structured news sentiment snapshots for ML features
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_sentiment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                news_sentiment_score REAL,
                news_sentiment_confidence REAL,
                news_sentiment_article_count REAL,
                news_sentiment_trend REAL,
                news_macro_sentiment REAL,
                news_company_expected_revenue_sentiment REAL,
                news_company_specific_sentiment REAL,
                news_industry_peer_sentiment REAL,
                source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date)
            )
        """)

        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_ticker_date ON price_history(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tech_ticker_date ON technical_indicators(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_labels_ticker_date ON labels(ticker, date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_sentiment_ticker_date ON news_sentiment(ticker, date)")

        conn.commit()
        conn.close()

        print(f"[DataCollector] Database initialized at {self.db_path}")

    def fetch_from_polygon(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from Polygon.io as fallback.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with historical data or None
        """
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            print(f"[DataCollector] No POLYGON_API_KEY found in environment")
            return None

        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,
                'apiKey': api_key
            }

            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"[DataCollector] Polygon API error {response.status_code} for {ticker}")
                return None

            data = response.json()

            if data.get('status') != 'OK' or not data.get('results'):
                print(f"[DataCollector] No Polygon data for {ticker}")
                return None

            # Convert to DataFrame
            results = data['results']
            df = pd.DataFrame(results)

            # Rename columns to match yfinance format
            df = df.rename(columns={
                'v': 'volume',
                'o': 'open',
                'c': 'close',
                'h': 'high',
                'l': 'low',
                't': 'timestamp'
            })

            # Convert timestamp to datetime
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date')

            # Select and order columns
            df = df[['open', 'high', 'low', 'close', 'volume']]

            print(f"[DataCollector] Fetched {len(df)} rows from Polygon for {ticker}")
            return df

        except Exception as e:
            print(f"[DataCollector] Polygon fetch error for {ticker}: {e}")
            return None

    def fetch_historical_data(
        self,
        ticker: str,
        start_date: str = None,
        end_date: str = None,
        period: str = "5y"
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data for a ticker with fallback to Polygon.io.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            period: Period if dates not specified (1y, 2y, 5y, max)

        Returns:
            DataFrame with historical data
        """
        # Try yfinance first
        try:
            stock = yf.Ticker(ticker)

            if start_date and end_date:
                hist = stock.history(start=start_date, end=end_date)
            else:
                hist = stock.history(period=period)

            if not hist.empty and len(hist) >= 200:
                # Clean column names
                hist.columns = [col.lower().replace(' ', '_') for col in hist.columns]
                hist['ticker'] = ticker
                print(f"[DataCollector] yfinance: {len(hist)} rows for {ticker}")
                return hist
            else:
                print(f"[DataCollector] yfinance: Insufficient data for {ticker} ({len(hist)} rows)")

        except Exception as e:
            print(f"[DataCollector] yfinance error for {ticker}: {e}")

        # Try Polygon.io as fallback
        print(f"[DataCollector] Trying Polygon.io for {ticker}...")

        if not start_date or not end_date:
            # Calculate dates based on period
            end = datetime.now()
            if period == "1y":
                start = end - timedelta(days=365)
            elif period == "2y":
                start = end - timedelta(days=730)
            elif period == "5y":
                start = end - timedelta(days=1825)
            elif period == "10y":
                start = end - timedelta(days=3650)
            else:  # max
                start = end - timedelta(days=7300)  # ~20 years

            start_date = start.strftime('%Y-%m-%d')
            end_date = end.strftime('%Y-%m-%d')

        polygon_data = self.fetch_from_polygon(ticker, start_date, end_date)

        if polygon_data is not None and len(polygon_data) >= 200:
            polygon_data['ticker'] = ticker
            return polygon_data

        print(f"[DataCollector] ❌ Failed to fetch data for {ticker} from all sources")
        return None

    def store_price_history(self, ticker: str, data: pd.DataFrame):
        """
        Store price history in database.

        Args:
            ticker: Stock ticker
            data: DataFrame with price data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing data for this ticker to avoid duplicates
        cursor.execute("DELETE FROM price_history WHERE ticker = ?", (ticker,))

        # Prepare data
        data_to_store = data.reset_index()
        data_to_store['ticker'] = ticker
        data_to_store['date'] = pd.to_datetime(data_to_store['Date']).dt.date

        # Select relevant columns
        columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
        if 'adj_close' in data_to_store.columns:
            columns.append('adj_close')

        df = data_to_store[columns]

        # Insert new data
        df.to_sql('price_history', conn, if_exists='append', index=False, method='multi')

        conn.commit()
        conn.close()

    def store_stock_info(self, ticker: str):
        """
        Fetch and store stock metadata.

        Args:
            ticker: Stock ticker
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO stock_info (ticker, name, sector, industry, market_cap, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                info.get('longName', ticker),
                info.get('sector', 'Unknown'),
                info.get('industry', 'Unknown'),
                info.get('marketCap', 0),
                datetime.now()
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            print(f"[DataCollector] Error storing info for {ticker}: {e}")

    def store_sentiment_snapshot(
        self,
        ticker: str,
        date: str,
        sentiment_result: Dict,
        source: str = "sentiment_agent"
    ) -> bool:
        """
        Store numeric sentiment features for a ticker/date pair.

        Args:
            ticker: Stock ticker
            date: Snapshot date (YYYY-MM-DD)
            sentiment_result: SentimentAgent result payload
            source: Feature provenance label

        Returns:
            True if stored successfully
        """
        try:
            features = extract_sentiment_features(sentiment_result)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO news_sentiment (
                    ticker,
                    date,
                    news_sentiment_score,
                    news_sentiment_confidence,
                    news_sentiment_article_count,
                    news_sentiment_trend,
                    news_macro_sentiment,
                    news_company_expected_revenue_sentiment,
                    news_company_specific_sentiment,
                    news_industry_peer_sentiment,
                    source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ticker,
                date,
                features["news_sentiment_score"],
                features["news_sentiment_confidence"],
                features["news_sentiment_article_count"],
                features["news_sentiment_trend"],
                features["news_macro_sentiment"],
                features["news_company_expected_revenue_sentiment"],
                features["news_company_specific_sentiment"],
                features["news_industry_peer_sentiment"],
                source
            ))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"[DataCollector] Error storing sentiment snapshot for {ticker}: {e}")
            return False

    def collect_multiple_tickers(
        self,
        tickers: List[str],
        period: str = "5y",
        max_workers: int = 5
    ) -> Dict[str, bool]:
        """
        Collect data for multiple tickers in parallel.

        Args:
            tickers: List of stock tickers
            period: Historical period to fetch
            max_workers: Number of parallel workers

        Returns:
            Dictionary mapping ticker to success status
        """
        results = {}
        total = len(tickers)

        print(f"[DataCollector] Collecting {total} tickers with {max_workers} workers...")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._collect_single, ticker, period): ticker
                for ticker in tickers
            }

            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                try:
                    success = future.result()
                    results[ticker] = success
                    status = "✓" if success else "✗"
                    print(f"  [{i}/{total}] {status} {ticker}")
                except Exception as e:
                    print(f"  [{i}/{total}] ✗ {ticker}: {e}")
                    results[ticker] = False

                # Rate limiting
                time.sleep(0.2)

        successful = sum(1 for v in results.values() if v)
        print(f"[DataCollector] Complete: {successful}/{total} successful")

        return results

    def _collect_single(self, ticker: str, period: str) -> bool:
        """
        Collect data for a single ticker.

        Args:
            ticker: Stock ticker
            period: Historical period

        Returns:
            True if successful
        """
        try:
            # Fetch price data
            data = self.fetch_historical_data(ticker, period=period)
            if data is None or data.empty:
                return False

            # Store price history
            self.store_price_history(ticker, data)

            # Store stock info
            self.store_stock_info(ticker)

            return True

        except Exception as e:
            print(f"[DataCollector] Error with {ticker}: {e}")
            return False

    def get_data_summary(self) -> Dict:
        """
        Get summary of stored data.

        Returns:
            Dictionary with data statistics
        """
        conn = sqlite3.connect(self.db_path)

        summary = {}

        # Count tickers
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM price_history")
        summary['total_tickers'] = cursor.fetchone()[0]

        # Count data points
        cursor.execute("SELECT COUNT(*) FROM price_history")
        summary['total_records'] = cursor.fetchone()[0]

        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM price_history")
        min_date, max_date = cursor.fetchone()
        summary['date_range'] = f"{min_date} to {max_date}"

        # Sectors
        cursor.execute("SELECT sector, COUNT(*) FROM stock_info GROUP BY sector")
        summary['sectors'] = dict(cursor.fetchall())

        conn.close()

        return summary

    def export_for_training(
        self,
        output_path: str = "ml/training/training_data.csv"
    ):
        """
        Export data in format ready for ML training.

        Args:
            output_path: Path to save CSV file
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                p.*,
                t.*,
                l.*,
                s.sector,
                s.industry
            FROM price_history p
            LEFT JOIN technical_indicators t ON p.ticker = t.ticker AND p.date = t.date
            LEFT JOIN labels l ON p.ticker = l.ticker AND p.date = l.date
            LEFT JOIN stock_info s ON p.ticker = s.ticker
            WHERE l.forward_5d_return IS NOT NULL
            ORDER BY p.ticker, p.date
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"[DataCollector] Exported {len(df)} records to {output_path}")

        return df


if __name__ == "__main__":
    # Example usage
    collector = HistoricalDataCollector()

    # Test with a few tickers
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "IREN", "TSLA", "META", "BRK-B", "JPM", "BBAI", "PLTR"]

    print("\n" + "="*60)
    print("COLLECTING HISTORICAL DATA")
    print("="*60 + "\n")

    results = collector.collect_multiple_tickers(test_tickers, period="2y", max_workers=3)

    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60 + "\n")

    summary = collector.get_data_summary()
    print(f"Total tickers: {summary['total_tickers']}")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Date range: {summary['date_range']}")
    print(f"\nSectors:")
    for sector, count in summary['sectors'].items():
        print(f"  {sector}: {count}")
