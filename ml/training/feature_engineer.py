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

    def __init__(self, db_path: str = "ml/training/stock_data.db", outlier_z_threshold: float = 5.0):
        """
        Initialize feature engineer.

        Args:
            db_path: Path to SQLite database
            outlier_z_threshold: Z-score threshold for outlier detection (default 5.0)
        """
        self.db_path = Path(db_path)
        self.outlier_z_threshold = outlier_z_threshold
        self.data_quality_stats = {
            'total_rows': 0,
            'invalid_ohlcv': 0,
            'outliers_detected': 0,
            'rows_excluded': 0
        }

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

    def add_lag_features(self, df: pd.DataFrame, periods: list[int] = None) -> pd.DataFrame:
        """
        Compute lag return features for the given periods.

        The lag return for period n is: (close - close.shift(n)) / close.shift(n)
        Resulting columns are named return_1d, return_5d, return_10d, return_20d.

        Args:
            df: DataFrame with a 'close' column (index should be date-ordered)
            periods: List of lag periods in days. Defaults to [1, 5, 10, 20].

        Returns:
            DataFrame with lag return columns added in-place (copy returned).
        """
        if periods is None:
            periods = [1, 5, 10, 20]

        df = df.copy()
        close = df['close']

        for n in periods:
            shifted = close.shift(n)
            df[f'return_{n}d'] = (close - shifted) / shifted

        return df

    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute volatility features: ATR-14, 20-day realized volatility, and vol regime.

        - atr_14: 14-period rolling mean of True Range.
          True Range = max(high-low, |high-prev_close|, |low-prev_close|). ATR >= 0.
        - realized_vol_20d: 20-day rolling std of daily log returns log(close/close.shift(1)).
          realized_vol >= 0.
        - vol_regime: Binary (0 or 1). 1 if realized_vol_20d > its 60-day rolling median, else 0.

        Args:
            df: DataFrame with 'high', 'low', 'close' columns (date-ordered).

        Returns:
            Copy of df with atr_14, realized_vol_20d, vol_regime columns added.
        """
        df = df.copy()

        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low'] - prev_close).abs(),
        ], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14, min_periods=14).mean()

        log_returns = np.log(df['close'] / df['close'].shift(1))
        df['realized_vol_20d'] = log_returns.rolling(window=20, min_periods=20).std()

        rolling_median = df['realized_vol_20d'].rolling(window=60, min_periods=1).median()
        df['vol_regime'] = (df['realized_vol_20d'] > rolling_median).astype(float)
        df.loc[df['realized_vol_20d'].isna(), 'vol_regime'] = np.nan

        return df

    def add_relative_strength(self, df: pd.DataFrame, benchmark: str = "SPY") -> pd.DataFrame:
        """
        Compute 20-day relative strength of each stock vs a benchmark (default SPY).

        rs_vs_spy_20d = stock_20d_return / spy_20d_return
        where 20d_return = (close - close.shift(20)) / close.shift(20), grouped by ticker.

        If the benchmark ticker is not present in the DataFrame, rs_vs_spy_20d is filled
        with NaN and a warning is logged.

        Args:
            df: Multi-ticker DataFrame with 'ticker', 'date', and 'close' columns.

        Returns:
            Copy of df with rs_vs_spy_20d column added.
        """
        import logging
        logger = logging.getLogger(__name__)

        df = df.copy()

        # Compute 20-day return for every row, grouped by ticker
        def _ret20(group):
            c = group['close']
            return (c - c.shift(20)) / c.shift(20)

        df['_ret20'] = df.groupby('ticker', group_keys=False)['close'].transform(
            lambda c: (c - c.shift(20)) / c.shift(20)
        )

        # Extract SPY 20d returns keyed by date
        spy_rows = df[df['ticker'] == benchmark]
        if spy_rows.empty:
            logger.warning(
                "[FeatureEngineer] Benchmark '%s' not found in DataFrame; "
                "rs_vs_spy_20d will be NaN for all rows.",
                benchmark,
            )
            df['rs_vs_spy_20d'] = np.nan
        else:
            spy_ret = spy_rows.set_index('date')['_ret20'].rename('_spy_ret20')
            df = df.join(spy_ret, on='date', how='left')
            # Avoid division by zero: where spy return is 0, result is NaN
            spy_zero = df['_spy_ret20'] == 0
            df['rs_vs_spy_20d'] = df['_ret20'] / df['_spy_ret20']
            df.loc[spy_zero, 'rs_vs_spy_20d'] = np.nan
            # Replace inf values (shouldn't occur after zero-guard, but be safe)
            df['rs_vs_spy_20d'] = df['rs_vs_spy_20d'].replace([np.inf, -np.inf], np.nan)
            df.drop(columns=['_spy_ret20'], inplace=True)

        df.drop(columns=['_ret20'], inplace=True)
        return df

    def add_bollinger_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Bollinger Band position features: bb_pct_b and bb_width.

        - bb_pct_b: Percent B = (close - bb_lower) / (bb_upper - bb_lower).
          Represents the position of close within the Bollinger Bands.
          Set to NaN when bb_upper == bb_lower (zero bandwidth).
        - bb_width: Band width = (bb_upper - bb_lower) / bb_middle.
          Normalized width of the bands.
          Set to NaN when bb_middle == 0.

        Requires columns: close, bb_upper, bb_middle, bb_lower.

        Args:
            df: DataFrame with bb_upper, bb_middle, bb_lower, and close columns.

        Returns:
            Copy of df with bb_pct_b and bb_width columns added.
        """
        df = df.copy()

        bandwidth = df['bb_upper'] - df['bb_lower']

        # bb_pct_b: NaN where bandwidth is zero
        bb_pct_b = (df['close'] - df['bb_lower']) / bandwidth
        bb_pct_b = bb_pct_b.where(bandwidth != 0, other=np.nan)
        df['bb_pct_b'] = bb_pct_b

        # bb_width: NaN where bb_middle is zero
        bb_width = bandwidth / df['bb_middle']
        bb_width = bb_width.where(df['bb_middle'] != 0, other=np.nan)
        df['bb_width'] = bb_width

        return df

    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute calendar features: day_of_week, month, and days_to_earnings.

        - day_of_week: Integer 0-6 (Monday=0, Sunday=6) from the 'date' column.
        - month: Integer 1-12 from the 'date' column.
        - days_to_earnings: Days until the next earnings date, approximated as the
          number of days until the next multiple of 63 trading days from the start
          of the series, computed per ticker.

        If the 'date' column is not datetime, it is converted first.
        If no 'date' column exists, all three features are filled with NaN.

        Args:
            df: DataFrame with a 'date' column and optionally a 'ticker' column.

        Returns:
            Copy of df with day_of_week, month, and days_to_earnings columns added.
        """
        df = df.copy()

        if 'date' not in df.columns:
            df['day_of_week'] = np.nan
            df['month'] = np.nan
            df['days_to_earnings'] = np.nan
            return df

        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])

        df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['month'] = df['date'].dt.month             # 1-12

        # days_to_earnings: per-ticker, days until next multiple of 63 from series start
        earnings_cycle = 63  # approximate quarterly trading days

        def _days_to_earnings(group):
            positions = np.arange(len(group))
            # Next multiple of 63 from position 0 (i.e., 63, 126, 189, ...)
            next_earnings = ((positions // earnings_cycle) + 1) * earnings_cycle
            return pd.Series(next_earnings - positions, index=group.index)

        if 'ticker' in df.columns:
            df['days_to_earnings'] = df.groupby('ticker', group_keys=False).apply(
                _days_to_earnings, include_groups=False
            )
        else:
            df['days_to_earnings'] = _days_to_earnings(df)

        return df

    def select_features(self, X: pd.DataFrame, y: pd.Series, top_n: int = 40) -> pd.DataFrame:
        """
        Select the most informative features from X.

        Steps:
        1. Drop features with >5% missing values (NaN fraction > 0.05).
        2. Drop features with pairwise absolute Pearson correlation > 0.95
           (keep the first of each correlated pair).
        3. Rank remaining features by XGBoost feature importance and keep top `top_n`.

        Fallback: If after steps 1-2 fewer than 10 features remain, skip step 2
        and instead keep top 20 features by XGBoost importance from the features
        that passed step 1 only. A warning is logged.

        Args:
            X: Feature DataFrame (rows = samples, columns = features).
            y: Target Series aligned with X.
            top_n: Number of top features to keep (default 40).

        Returns:
            DataFrame with only the selected feature columns.
        """
        import logging
        import xgboost as xgb

        logger = logging.getLogger(__name__)

        # Step 1: Drop features with >5% missing values
        missing_frac = X.isnull().mean()
        passed_missing = missing_frac[missing_frac <= 0.05].index.tolist()
        X_clean = X[passed_missing]

        # Step 2: Drop highly correlated features (>0.95 absolute Pearson)
        corr_matrix = X_clean.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
        X_after_corr = X_clean.drop(columns=to_drop)

        # Fallback: if fewer than 10 features remain after correlation pruning,
        # skip step 2 and use top 20 from step 1 features only
        use_fallback = len(X_after_corr.columns) < 10
        if use_fallback:
            logger.warning(
                "[FeatureEngineer] select_features: only %d features remain after "
                "correlation pruning (< 10). Skipping correlation pruning and "
                "keeping top 20 features by XGBoost importance from step-1 features.",
                len(X_after_corr.columns),
            )
            X_for_importance = X_clean
            n_keep = min(20, top_n)
        else:
            X_for_importance = X_after_corr
            n_keep = top_n

        # Step 3: Rank by XGBoost feature importance and keep top n_keep
        # Fill NaNs with median for XGBoost fitting
        X_filled = X_for_importance.fillna(X_for_importance.median())
        clf = xgb.XGBClassifier(
            n_estimators=100,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_filled, y)

        importances = pd.Series(clf.feature_importances_, index=X_for_importance.columns)
        top_features = importances.nlargest(n_keep).index.tolist()

        return X[top_features]

    def validate_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data quality and remove invalid/anomalous rows.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Cleaned DataFrame with invalid rows removed

        **Validates: Requirements 10.1, 10.2, 10.3**
        """
        if df.empty:
            return df

        initial_rows = len(df)
        self.data_quality_stats['total_rows'] = initial_rows

        # Step 1: Validate OHLCV values
        # Check: high >= low, all OHLCV values > 0
        ohlcv_valid_mask = (
            (df['high'] >= df['low']) &
            (df['open'] > 0) &
            (df['high'] > 0) &
            (df['low'] > 0) &
            (df['close'] > 0) &
            (df['volume'] > 0)
        )

        invalid_ohlcv_count = (~ohlcv_valid_mask).sum()
        if invalid_ohlcv_count > 0:
            print(f"[FeatureEngineer] Found {invalid_ohlcv_count} rows with invalid OHLCV values (high < low or non-positive values)")
            self.data_quality_stats['invalid_ohlcv'] = invalid_ohlcv_count

        df = df[ohlcv_valid_mask].copy()

        # Step 2: Detect outliers using z-score on price changes
        if len(df) > 10:  # Need enough data for meaningful z-score
            # Calculate daily returns
            df_sorted = df.sort_values(['ticker', 'date'])
            df_sorted['price_change'] = df_sorted.groupby('ticker')['close'].pct_change()

            # Calculate z-score for price changes
            price_changes = df_sorted['price_change'].dropna()
            if len(price_changes) > 0:
                mean_change = price_changes.mean()
                std_change = price_changes.std()

                if std_change > 0:
                    df_sorted['z_score'] = (df_sorted['price_change'] - mean_change) / std_change

                    # Flag outliers
                    outlier_mask = df_sorted['z_score'].abs() > self.outlier_z_threshold

                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        print(f"[FeatureEngineer] Detected {outlier_count} outliers (|z-score| > {self.outlier_z_threshold})")
                        self.data_quality_stats['outliers_detected'] = outlier_count

                        # Log some outlier examples
                        outliers = df_sorted[outlier_mask][['ticker', 'date', 'close', 'price_change', 'z_score']].head(5)
                        for _, row in outliers.iterrows():
                            print(f"  {row['ticker']} on {row['date']}: {row['price_change']*100:.2f}% change (z={row['z_score']:.2f})")

                    # Remove outliers
                    df = df_sorted[~outlier_mask].drop(columns=['price_change', 'z_score'])
                else:
                    df = df_sorted.drop(columns=['price_change'], errors='ignore')
            else:
                df = df_sorted.drop(columns=['price_change'], errors='ignore')

        # Calculate total rows excluded
        final_rows = len(df)
        rows_excluded = initial_rows - final_rows
        self.data_quality_stats['rows_excluded'] = rows_excluded

        if rows_excluded > 0:
            exclusion_pct = (rows_excluded / initial_rows) * 100
            print(f"[FeatureEngineer] Data quality check: excluded {rows_excluded} / {initial_rows} rows ({exclusion_pct:.1f}%)")

        return df

    def get_data_quality_stats(self) -> dict:
        """
        Get data quality statistics from the last validation run.

        Returns:
            Dictionary with validation statistics
        """
        return self.data_quality_stats.copy()

    def get_training_data(
        self,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get processed training data with data quality validation.

        Args:
            min_date: Minimum date (YYYY-MM-DD)
            max_date: Maximum date (YYYY-MM-DD)

        Returns:
            DataFrame with features and labels (invalid rows excluded)
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

        # Validate data quality before feature engineering
        if not df.empty:
            df = self.validate_data_quality(df)

        # Add lag return features to the pipeline output
        if not df.empty:
            df = self.add_lag_features(df, periods=[1, 5, 10, 20])
            df = self.add_volatility_features(df)
            df = self.add_relative_strength(df)
            df = self.add_calendar_features(df)
            df = self.add_bollinger_features(df)

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
