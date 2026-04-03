"""
Feature Engineering for ML Models
Calculates technical indicators and creates labels for training.
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Optional
from pathlib import Path
from tools.ta_indicators import TechnicalIndicators


class FeatureEngineer:
    """Engineers features and labels for ML training."""

    def __init__(self, db_path: str = "ml/training/stock_data.db"):
        """
        Initialize feature engineer.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)

    def calculate_technical_indicators(self, ticker: str) -> bool:
        """
        Calculate and store technical indicators for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            True if successful
        """
        try:
            # Load price data
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT date, open, high, low, close, volume
                FROM price_history
                WHERE ticker = '{ticker}'
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn, parse_dates=['date'])

            if df.empty or len(df) < 200:
                print(f"[FeatureEngineer] Insufficient data for {ticker}")
                conn.close()
                return False

            # Set date as index
            df.set_index('date', inplace=True)

            # Calculate indicators using static methods for full series
            close_prices = df['close']
            volumes = df['volume']

            rsi = TechnicalIndicators.calculate_rsi(close_prices, period=14)
            macd_data = TechnicalIndicators.calculate_macd(close_prices)
            sma = TechnicalIndicators.calculate_moving_averages(close_prices, [20, 50, 200])
            ema = TechnicalIndicators.calculate_exponential_moving_averages(close_prices, [12, 26])
            bb = TechnicalIndicators.calculate_bollinger_bands(close_prices)
            vol_indicators = TechnicalIndicators.calculate_volume_indicators(close_prices, volumes)

            # Package indicators
            indicators = {
                'rsi': rsi.values,
                'macd': {
                    'macd': macd_data['macd'].values,
                    'signal': macd_data['signal'].values,
                    'histogram': macd_data['histogram'].values
                },
                'sma': {
                    '20': sma['SMA_20'].values,
                    '50': sma['SMA_50'].values,
                    '200': sma['SMA_200'].values
                },
                'ema': {
                    '12': ema['EMA_12'].values,
                    '26': ema['EMA_26'].values
                },
                'bollinger': {
                    'upper': bb['upper'].values,
                    'middle': bb['middle'].values,
                    'lower': bb['lower'].values
                },
                'volume_sma': vol_indicators['average_volume'].values
            }

            # Prepare data for storage
            tech_data = []
            for i, date in enumerate(df.index):
                tech_data.append({
                    'ticker': ticker,
                    'date': date.date(),
                    'rsi_14': indicators['rsi'][i] if i < len(indicators['rsi']) else None,
                    'macd': indicators['macd']['macd'][i] if i < len(indicators['macd']['macd']) else None,
                    'macd_signal': indicators['macd']['signal'][i] if i < len(indicators['macd']['signal']) else None,
                    'macd_hist': indicators['macd']['histogram'][i] if i < len(indicators['macd']['histogram']) else None,
                    'sma_20': indicators['sma']['20'][i] if i < len(indicators['sma']['20']) else None,
                    'sma_50': indicators['sma']['50'][i] if i < len(indicators['sma']['50']) else None,
                    'sma_200': indicators['sma']['200'][i] if i < len(indicators['sma']['200']) else None,
                    'ema_12': indicators['ema']['12'][i] if i < len(indicators['ema']['12']) else None,
                    'ema_26': indicators['ema']['26'][i] if i < len(indicators['ema']['26']) else None,
                    'bb_upper': indicators['bollinger']['upper'][i] if i < len(indicators['bollinger']['upper']) else None,
                    'bb_middle': indicators['bollinger']['middle'][i] if i < len(indicators['bollinger']['middle']) else None,
                    'bb_lower': indicators['bollinger']['lower'][i] if i < len(indicators['bollinger']['lower']) else None,
                    'volume_sma_20': indicators['volume_sma'][i] if i < len(indicators['volume_sma']) else None,
                })

            # Delete existing indicators for this ticker to avoid duplicates
            cursor = conn.cursor()
            cursor.execute("DELETE FROM technical_indicators WHERE ticker = ?", (ticker,))

            # Store in database
            tech_df = pd.DataFrame(tech_data)
            tech_df.to_sql('technical_indicators', conn, if_exists='append', index=False, method='multi')

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"[FeatureEngineer] Error calculating indicators for {ticker}: {e}")
            return False

    def create_labels(self, ticker: str, forward_periods: list = [1, 5, 10, 20]) -> bool:
        """
        Create forward-looking labels for supervised learning.

        Args:
            ticker: Stock ticker
            forward_periods: List of forward-looking periods in days

        Returns:
            True if successful
        """
        try:
            # Load price data
            conn = sqlite3.connect(self.db_path)

            # Check if labels table exists and has correct schema
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='labels'")
            table_exists = cursor.fetchone() is not None

            if table_exists:
                # Get existing columns
                cursor.execute("PRAGMA table_info(labels)")
                existing_columns = [row[1] for row in cursor.fetchall()]

                # Check if all required columns exist
                required_columns = ['ticker', 'date'] + \
                                 [f'forward_{p}d_return' for p in forward_periods] + \
                                 [f'direction_{p}d' for p in forward_periods] + \
                                 ['max_drawdown_5d']

                missing_columns = [col for col in required_columns if col not in existing_columns]

                if missing_columns:
                    print(f"[FeatureEngineer] Recreating labels table with correct schema...")
                    cursor.execute("DROP TABLE labels")
                    conn.commit()

            query = f"""
                SELECT date, close
                FROM price_history
                WHERE ticker = '{ticker}'
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn, parse_dates=['date'])

            if df.empty:
                conn.close()
                return False

            # Calculate forward returns
            labels_data = []

            for i in range(len(df) - max(forward_periods)):
                current_price = df.iloc[i]['close']
                date = df.iloc[i]['date'].date()

                label = {
                    'ticker': ticker,
                    'date': date,
                }

                # Calculate returns for each period
                for period in forward_periods:
                    if i + period < len(df):
                        future_price = df.iloc[i + period]['close']
                        return_pct = ((future_price - current_price) / current_price) * 100

                        label[f'forward_{period}d_return'] = return_pct
                        label[f'direction_{period}d'] = 1 if return_pct > 0 else 0

                # Calculate max drawdown over 5 days
                if i + 5 < len(df):
                    future_prices = df.iloc[i:i+6]['close'].values
                    peak = future_prices[0]
                    max_dd = 0

                    for price in future_prices[1:]:
                        if price > peak:
                            peak = price
                        dd = ((price - peak) / peak) * 100
                        max_dd = min(max_dd, dd)

                    label['max_drawdown_5d'] = max_dd

                labels_data.append(label)

            # Delete existing labels for this ticker to avoid duplicates
            cursor.execute("DELETE FROM labels WHERE ticker = ?", (ticker,))

            # Store in database
            if labels_data:
                labels_df = pd.DataFrame(labels_data)
                labels_df.to_sql('labels', conn, if_exists='append', index=False, method='multi')

            conn.commit()
            conn.close()

            return True

        except Exception as e:
            print(f"[FeatureEngineer] Error creating labels for {ticker}: {e}")
            return False

    def process_all_tickers(self, max_tickers: Optional[int] = None) -> dict:
        """
        Process all tickers in database.

        Args:
            max_tickers: Maximum number of tickers to process (None = all)

        Returns:
            Dictionary with processing stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get list of tickers
        cursor.execute("SELECT DISTINCT ticker FROM price_history")
        tickers = [row[0] for row in cursor.fetchall()]

        if max_tickers:
            tickers = tickers[:max_tickers]

        conn.close()

        total = len(tickers)
        print(f"[FeatureEngineer] Processing {total} tickers...")

        indicators_success = 0
        labels_success = 0

        for i, ticker in enumerate(tickers, 1):
            print(f"  [{i}/{total}] Processing {ticker}...")

            # Calculate indicators
            if self.calculate_technical_indicators(ticker):
                indicators_success += 1

            # Create labels
            if self.create_labels(ticker):
                labels_success += 1

        print(f"\n[FeatureEngineer] Complete!")
        print(f"  Indicators: {indicators_success}/{total}")
        print(f"  Labels: {labels_success}/{total}")

        return {
            'total': total,
            'indicators_success': indicators_success,
            'labels_success': labels_success
        }

    def get_training_data(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get processed training data.

        Args:
            min_date: Minimum date (YYYY-MM-DD)
            max_date: Maximum date (YYYY-MM-DD)

        Returns:
            DataFrame with features and labels
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news_sentiment'")
        has_news_sentiment = cursor.fetchone() is not None

        # Build query
        query = """
            SELECT
                p.ticker,
                p.date,
                p.open,
                p.high,
                p.low,
                p.close,
                p.volume,
                t.rsi_14,
                t.macd,
                t.macd_signal,
                t.macd_hist,
                t.sma_20,
                t.sma_50,
                t.sma_200,
                t.ema_12,
                t.ema_26,
                t.bb_upper,
                t.bb_middle,
                t.bb_lower,
                t.volume_sma_20,
                l.forward_1d_return,
                l.forward_5d_return,
                l.forward_10d_return,
                l.forward_20d_return,
                l.direction_1d,
                l.direction_5d,
                l.max_drawdown_5d,
                s.sector,
                s.industry
            FROM price_history p
            INNER JOIN technical_indicators t ON p.ticker = t.ticker AND p.date = t.date
            INNER JOIN labels l ON p.ticker = l.ticker AND p.date = l.date
            LEFT JOIN stock_info s ON p.ticker = s.ticker
        """

        if has_news_sentiment:
            query += """
            LEFT JOIN news_sentiment ns ON p.ticker = ns.ticker AND p.date = ns.date
            """
            query = query.replace(
                "s.industry",
                """s.industry,
                ns.news_sentiment_score,
                ns.news_sentiment_confidence,
                ns.news_sentiment_article_count,
                ns.news_sentiment_trend,
                ns.news_macro_sentiment,
                ns.news_company_expected_revenue_sentiment,
                ns.news_company_specific_sentiment,
                ns.news_industry_peer_sentiment"""
            )

        query += """
            WHERE 1=1
        """

        if min_date:
            query += f" AND p.date >= '{min_date}'"
        if max_date:
            query += f" AND p.date <= '{max_date}'"

        query += " ORDER BY p.ticker, p.date"

        df = pd.read_sql_query(query, conn, parse_dates=['date'])
        conn.close()

        return df


if __name__ == "__main__":
    # Example usage
    engineer = FeatureEngineer()

    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60 + "\n")

    # Process all tickers
    stats = engineer.process_all_tickers(max_tickers=10)  # Start with 10 for testing

    print("\n" + "="*60)
    print("TRAINING DATA")
    print("="*60 + "\n")

    # Get training data
    training_data = engineer.get_training_data()
    print(f"Total training samples: {len(training_data):,}")
    print(f"Features: {training_data.columns.tolist()}")
    print(f"\nSample:")
    print(training_data.head())

    # Check for missing values
    missing = training_data.isnull().sum()
    print(f"\nMissing values:")
    print(missing[missing > 0])
