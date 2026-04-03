"""
Pre-built Stock Screens
Common screening strategies ready to use.
"""

from typing import Dict, List, Optional
from tools.screener import ScreenCriteria, CriteriaOperator, ScreenerEngine, ScreenResult
from tools.stock_universe import StockUniverse


class PrebuiltScreens:
    """Collection of pre-built stock screens."""

    def __init__(self):
        """Initialize pre-built screens."""
        self.engine = ScreenerEngine(max_workers=10)
        self.universe = StockUniverse()

    def oversold(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find oversold stocks (RSI < 30) with potential for reversal.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("rsi", CriteriaOperator.LESS_THAN, 30, weight=1.0),
            ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.2, weight=0.5),  # Above average volume
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=50
        )

        return results[:limit]

    def overbought(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find overbought stocks (RSI > 70) potentially due for pullback.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("rsi", CriteriaOperator.GREATER_THAN, 70, weight=1.0),
            ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.2, weight=0.5),
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=50
        )

        return results[:limit]

    def breakout(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find stocks breaking above their 50-day moving average with volume.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("price_vs_sma20", CriteriaOperator.GREATER_THAN, 0, weight=1.0),  # Above 20-day MA
            ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.5, weight=0.8),  # High volume
            ScreenCriteria("returns_5d", CriteriaOperator.GREATER_THAN, 2, weight=0.6),  # Up 2%+ in 5 days
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=60
        )

        return results[:limit]

    def momentum(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find stocks with strong upward momentum.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("returns_5d", CriteriaOperator.GREATER_THAN, 3, weight=1.0),  # Up 3%+ in 5 days
            ScreenCriteria("rsi", CriteriaOperator.BETWEEN, [50, 70], weight=0.8),  # Bullish but not overbought
            ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.2, weight=0.6),
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=60
        )

        return results[:limit]

    def value(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find undervalued stocks based on fundamental metrics.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("pe_ratio", CriteriaOperator.LESS_THAN, 20, weight=1.0),
            ScreenCriteria("price_to_book", CriteriaOperator.LESS_THAN, 3, weight=0.8),
            ScreenCriteria("dividend_yield", CriteriaOperator.GREATER_THAN, 2, weight=0.6),
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=False,  # Any criteria can match
            min_score=40
        )

        return results[:limit]

    def growth(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find high-growth stocks with strong fundamentals.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("revenue_growth", CriteriaOperator.GREATER_THAN, 15, weight=1.0),  # 15%+ revenue growth
            ScreenCriteria("earnings_growth", CriteriaOperator.GREATER_THAN, 10, weight=0.9),  # 10%+ earnings growth
            ScreenCriteria("profit_margin", CriteriaOperator.GREATER_THAN, 10, weight=0.7),  # 10%+ margins
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=False,
            min_score=40
        )

        return results[:limit]

    def quality(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find high-quality stocks with strong balance sheets.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("roe", CriteriaOperator.GREATER_THAN, 15, weight=1.0),  # ROE > 15%
            ScreenCriteria("debt_to_equity", CriteriaOperator.LESS_THAN, 1.0, weight=0.9),  # Low debt
            ScreenCriteria("profit_margin", CriteriaOperator.GREATER_THAN, 15, weight=0.8),  # High margins
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=False,
            min_score=50
        )

        return results[:limit]

    def dividend(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find stocks with high dividend yields.

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("dividend_yield", CriteriaOperator.GREATER_THAN, 3, weight=1.0),  # 3%+ yield
            ScreenCriteria("pe_ratio", CriteriaOperator.LESS_THAN, 25, weight=0.7),  # Reasonable valuation
            ScreenCriteria("debt_to_equity", CriteriaOperator.LESS_THAN, 1.5, weight=0.6),  # Manageable debt
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=60
        )

        return results[:limit]

    def reversal(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find potential reversal candidates (oversold with improving momentum).

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("rsi", CriteriaOperator.BETWEEN, [25, 40], weight=1.0),  # Oversold but recovering
            ScreenCriteria("returns_1d", CriteriaOperator.GREATER_THAN, 0, weight=0.8),  # Up today
            ScreenCriteria("volume_ratio", CriteriaOperator.GREATER_THAN, 1.3, weight=0.7),  # Above avg volume
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=True,
            min_score=60
        )

        return results[:limit]

    def contrarian(
        self,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Find beaten-down stocks with strong fundamentals (contrarian play).

        Args:
            universe: Stock universe to screen
            sector: Optional sector filter
            limit: Maximum results to return

        Returns:
            List of ScreenResult objects
        """
        tickers = self._get_tickers(universe, sector)

        criteria = [
            ScreenCriteria("returns_5d", CriteriaOperator.LESS_THAN, -5, weight=1.0),  # Down 5%+ recently
            ScreenCriteria("roe", CriteriaOperator.GREATER_THAN, 10, weight=0.9),  # Still profitable
            ScreenCriteria("pe_ratio", CriteriaOperator.LESS_THAN, 20, weight=0.8),  # Reasonable valuation
            ScreenCriteria("debt_to_equity", CriteriaOperator.LESS_THAN, 1.5, weight=0.7),  # Healthy balance sheet
        ]

        results = self.engine.screen_multiple_stocks(
            tickers,
            criteria,
            require_all=False,
            min_score=50
        )

        return results[:limit]

    def list_screens(self) -> Dict[str, str]:
        """
        List all available pre-built screens.

        Returns:
            Dictionary mapping screen names to descriptions
        """
        return {
            "oversold": "RSI < 30, potential reversal candidates",
            "overbought": "RSI > 70, potentially overextended",
            "breakout": "Breaking above moving averages with volume",
            "momentum": "Strong upward price momentum",
            "value": "Undervalued based on P/E, P/B, dividend yield",
            "growth": "High revenue and earnings growth",
            "quality": "Strong ROE, low debt, high margins",
            "dividend": "High dividend yield (3%+)",
            "reversal": "Oversold stocks showing early signs of recovery",
            "contrarian": "Beaten-down stocks with strong fundamentals",
        }

    def run_screen(
        self,
        screen_name: str,
        universe: str = "sp100",
        sector: Optional[str] = None,
        limit: int = 10
    ) -> List[ScreenResult]:
        """
        Run a pre-built screen by name.

        Args:
            screen_name: Name of the screen to run
            universe: Stock universe
            sector: Optional sector filter
            limit: Maximum results

        Returns:
            List of ScreenResult objects
        """
        screen_methods = {
            "oversold": self.oversold,
            "overbought": self.overbought,
            "breakout": self.breakout,
            "momentum": self.momentum,
            "value": self.value,
            "growth": self.growth,
            "quality": self.quality,
            "dividend": self.dividend,
            "reversal": self.reversal,
            "contrarian": self.contrarian,
        }

        if screen_name not in screen_methods:
            raise ValueError(f"Unknown screen: {screen_name}. Available: {list(screen_methods.keys())}")

        return screen_methods[screen_name](universe=universe, sector=sector, limit=limit)

    def _get_tickers(self, universe: str, sector: Optional[str] = None) -> List[str]:
        """
        Get tickers for screening.

        Args:
            universe: Universe name
            sector: Optional sector filter

        Returns:
            List of tickers
        """
        if sector:
            tickers = self.universe.get_tickers_by_sector(sector, universe)
        else:
            tickers = self.universe.get_tickers(universe)

        return tickers


if __name__ == "__main__":
    # Example usage
    print("Testing Pre-built Screens...\n")

    screens = PrebuiltScreens()

    # List available screens
    print("Available screens:")
    for name, description in screens.list_screens().items():
        print(f"  {name}: {description}")

    print("\n" + "="*60)
    print("Running 'oversold' screen on S&P 100...")
    print("="*60 + "\n")

    results = screens.run_screen("oversold", limit=5)

    if results:
        print(f"Found {len(results)} oversold stocks:\n")
        for i, result in enumerate(results, 1):
            metrics = result.metrics
            print(f"{i}. {result.ticker} - Score: {result.score:.1f}/100")
            print(f"   Price: ${metrics.get('price', 0):.2f}")
            print(f"   RSI: {metrics.get('rsi', 0):.1f}")
            print(f"   P/E: {metrics.get('pe_ratio', 'N/A')}")
            print(f"   Volume Ratio: {metrics.get('volume_ratio', 0):.2f}x")
            print(f"   5d Return: {metrics.get('returns_5d', 0):+.1f}%")
            print()
    else:
        print("No stocks found matching criteria.")

    print("="*60)
    print("Running 'value' screen on Technology sector...")
    print("="*60 + "\n")

    tech_value = screens.run_screen("value", sector="Technology", limit=3)

    if tech_value:
        print(f"Found {len(tech_value)} value tech stocks:\n")
        for i, result in enumerate(tech_value, 1):
            metrics = result.metrics
            print(f"{i}. {result.ticker} - Score: {result.score:.1f}/100")
            print(f"   Price: ${metrics.get('price', 0):.2f}")
            print(f"   P/E: {metrics.get('pe_ratio', 'N/A')}")
            print(f"   P/B: {metrics.get('price_to_book', 'N/A')}")
            print(f"   Dividend Yield: {metrics.get('dividend_yield', 0):.2f}%")
            print()
    else:
        print("No value tech stocks found.")
