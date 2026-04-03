"""
Batch Stock Analyzer
Tiered analysis system: Quick Filter → Medium Analysis → Deep Analysis
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import time

from tools.screener import ScreenResult, QuickMetrics
from tools.ta_indicators import TechnicalIndicators
from agents.technical_agent import TechnicalAgent
from agents.fundamentals_agent import FundamentalsAgent


@dataclass
class AnalysisLevel:
    """Analysis depth levels."""
    QUICK = "quick"  # Basic metrics only (1-2s)
    MEDIUM = "medium"  # Technical + Fundamentals (10-15s)
    DEEP = "deep"  # Full 7-agent analysis (60-90s)


@dataclass
class BatchAnalysisResult:
    """Result of batch analysis."""
    ticker: str
    level: str  # quick, medium, or deep
    score: float  # 0-100
    signal: str  # bullish, bearish, neutral
    metrics: Dict[str, Any]
    analysis: Optional[str] = None  # LLM analysis (medium/deep only)
    execution_time: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BatchAnalyzer:
    """Tiered batch analysis system."""

    def __init__(self, max_workers: int = 5):
        """
        Initialize batch analyzer.

        Args:
            max_workers: Maximum parallel workers
        """
        self.max_workers = max_workers

    def quick_analysis(self, ticker: str) -> BatchAnalysisResult:
        """
        Quick analysis - basic metrics only, no LLM.

        Args:
            ticker: Stock ticker

        Returns:
            BatchAnalysisResult with quick metrics
        """
        start_time = time.time()

        metrics = QuickMetrics.fetch_basic_data(ticker)
        if not metrics:
            return BatchAnalysisResult(
                ticker=ticker,
                level=AnalysisLevel.QUICK,
                score=0.0,
                signal="error",
                metrics={},
                execution_time=time.time() - start_time
            )

        # Calculate simple score
        score = self._calculate_quick_score(metrics)
        signal = self._determine_signal(score)

        return BatchAnalysisResult(
            ticker=ticker,
            level=AnalysisLevel.QUICK,
            score=score,
            signal=signal,
            metrics=metrics,
            execution_time=time.time() - start_time
        )

    def medium_analysis(self, ticker: str) -> BatchAnalysisResult:
        """
        Medium analysis - technical + fundamentals with LLM.

        Args:
            ticker: Stock ticker

        Returns:
            BatchAnalysisResult with technical and fundamental analysis
        """
        start_time = time.time()

        # Get quick metrics first
        metrics = QuickMetrics.fetch_basic_data(ticker, period="3mo")
        if not metrics:
            return BatchAnalysisResult(
                ticker=ticker,
                level=AnalysisLevel.MEDIUM,
                score=0.0,
                signal="error",
                metrics={},
                execution_time=time.time() - start_time
            )

        # Run technical agent (lightweight)
        try:
            tech_agent = TechnicalAgent()
            tech_agent.create_assistant()
            tech_result = tech_agent.analyze_technical(ticker)
            tech_agent.cleanup()

            if tech_result:
                metrics["technical_analysis"] = tech_result
                metrics["technical_signal"] = tech_result.get("signal", {}).get("direction", "neutral")
        except Exception as e:
            print(f"Warning: Technical analysis failed for {ticker}: {e}")
            metrics["technical_signal"] = "neutral"

        # Run fundamentals agent (lightweight)
        try:
            fund_agent = FundamentalsAgent()
            fund_agent.create_assistant()
            fund_result = fund_agent.analyze_fundamentals(ticker)
            fund_agent.cleanup()

            if fund_result:
                metrics["fundamental_analysis"] = fund_result
                metrics["fundamental_score"] = fund_result.get("composite_score", 50)
        except Exception as e:
            print(f"Warning: Fundamental analysis failed for {ticker}: {e}")
            metrics["fundamental_score"] = 50

        # Calculate combined score
        score = self._calculate_medium_score(metrics)
        signal = self._determine_signal(score)

        # Generate summary analysis
        analysis = self._generate_medium_summary(ticker, metrics)

        return BatchAnalysisResult(
            ticker=ticker,
            level=AnalysisLevel.MEDIUM,
            score=score,
            signal=signal,
            metrics=metrics,
            analysis=analysis,
            execution_time=time.time() - start_time
        )

    def deep_analysis(self, ticker: str) -> BatchAnalysisResult:
        """
        Deep analysis - full 7-agent orchestration.

        Args:
            ticker: Stock ticker

        Returns:
            BatchAnalysisResult with comprehensive analysis
        """
        start_time = time.time()

        # Import here to avoid circular dependency
        from agents.orchestrator import OrchestratorAgent

        try:
            orchestrator = OrchestratorAgent()
            result = orchestrator.analyze_stock(ticker)

            if not result.get("success", False):
                return BatchAnalysisResult(
                    ticker=ticker,
                    level=AnalysisLevel.DEEP,
                    score=0.0,
                    signal="error",
                    metrics={},
                    analysis=result.get("error", "Analysis failed"),
                    execution_time=time.time() - start_time
                )

            # Extract score from analysis
            score = self._extract_score_from_report(result.get("final_report", ""))
            signal = self._determine_signal(score)

            return BatchAnalysisResult(
                ticker=ticker,
                level=AnalysisLevel.DEEP,
                score=score,
                signal=signal,
                metrics=result.get("agent_results", {}),
                analysis=result.get("final_report", ""),
                execution_time=time.time() - start_time
            )

        except Exception as e:
            print(f"Error in deep analysis for {ticker}: {e}")
            return BatchAnalysisResult(
                ticker=ticker,
                level=AnalysisLevel.DEEP,
                score=0.0,
                signal="error",
                metrics={},
                analysis=str(e),
                execution_time=time.time() - start_time
            )

    def batch_quick_analysis(self, tickers: List[str]) -> List[BatchAnalysisResult]:
        """
        Run quick analysis on multiple tickers in parallel.

        Args:
            tickers: List of stock tickers

        Returns:
            List of BatchAnalysisResult objects
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.quick_analysis, ticker): ticker for ticker in tickers}

            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    ticker = futures[future]
                    print(f"Error analyzing {ticker}: {e}")

        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def tiered_screening(
        self,
        tickers: List[str],
        quick_top_n: int = 20,
        medium_top_n: int = 10,
        deep_top_n: int = 5
    ) -> Dict[str, List[BatchAnalysisResult]]:
        """
        Run tiered screening: Quick → Medium → Deep.

        Args:
            tickers: Initial list of tickers to screen
            quick_top_n: Top N from quick analysis to pass to medium
            medium_top_n: Top N from medium analysis to pass to deep
            deep_top_n: Top N from deep analysis to return

        Returns:
            Dictionary with results at each tier
        """
        print(f"🔍 Tier 1: Quick analysis on {len(tickers)} stocks...")
        quick_results = self.batch_quick_analysis(tickers)
        quick_top = [r for r in quick_results if r.signal != "error"][:quick_top_n]
        print(f"   ✓ Found {len(quick_top)} candidates")

        print(f"\n📊 Tier 2: Medium analysis on top {len(quick_top)} stocks...")
        medium_results = []
        for result in quick_top:
            medium_result = self.medium_analysis(result.ticker)
            medium_results.append(medium_result)
            print(f"   ✓ {result.ticker}: {medium_result.score:.1f}/100 ({medium_result.signal})")

        medium_results.sort(key=lambda x: x.score, reverse=True)
        medium_top = [r for r in medium_results if r.signal != "error"][:medium_top_n]

        print(f"\n🎯 Tier 3: Deep analysis on top {min(len(medium_top), deep_top_n)} stocks...")
        deep_results = []
        for i, result in enumerate(medium_top[:deep_top_n], 1):
            print(f"   [{i}/{min(len(medium_top), deep_top_n)}] Analyzing {result.ticker}...")
            deep_result = self.deep_analysis(result.ticker)
            deep_results.append(deep_result)
            print(f"       ✓ Score: {deep_result.score:.1f}/100 ({deep_result.execution_time:.1f}s)")

        deep_results.sort(key=lambda x: x.score, reverse=True)

        return {
            "quick": quick_results,
            "medium": medium_results,
            "deep": deep_results,
            "summary": {
                "total_screened": len(tickers),
                "quick_passed": len(quick_top),
                "medium_passed": len(medium_top),
                "deep_analyzed": len(deep_results)
            }
        }

    def _calculate_quick_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate score from quick metrics (rule-based).

        Args:
            metrics: Stock metrics

        Returns:
            Score from 0-100
        """
        score = 50.0  # Neutral baseline

        # RSI contribution
        rsi = metrics.get("rsi", 50)
        if rsi < 30:
            score += 20  # Oversold, bullish
        elif rsi > 70:
            score -= 20  # Overbought, bearish
        elif 40 <= rsi <= 60:
            score += 5  # Neutral zone

        # Price vs moving average
        price_vs_sma20 = metrics.get("price_vs_sma20", 0)
        if price_vs_sma20 > 5:
            score += 10  # Well above MA
        elif price_vs_sma20 > 0:
            score += 5  # Above MA
        elif price_vs_sma20 < -5:
            score -= 10  # Well below MA

        # Volume
        volume_ratio = metrics.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            score += 5  # High volume (conviction)

        # Valuation (P/E)
        pe = metrics.get("pe_ratio")
        if pe and 10 < pe < 25:
            score += 10  # Reasonable valuation
        elif pe and pe > 50:
            score -= 5  # Expensive

        # Momentum
        returns_5d = metrics.get("returns_5d", 0)
        if returns_5d > 5:
            score += 10  # Strong momentum
        elif returns_5d > 0:
            score += 5  # Positive momentum
        elif returns_5d < -5:
            score -= 10  # Weak

        return max(0, min(100, score))

    def _calculate_medium_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate score from medium analysis (technical + fundamental).

        Args:
            metrics: Stock metrics

        Returns:
            Score from 0-100
        """
        # Start with quick score
        score = self._calculate_quick_score(metrics)

        # Adjust with technical analysis
        tech_signal = metrics.get("technical_signal", "neutral")
        if tech_signal == "bullish":
            score += 15
        elif tech_signal == "bearish":
            score -= 15

        # Adjust with fundamental score
        fund_score = metrics.get("fundamental_score", 50)
        # Normalize fundamental score influence
        fund_adjustment = (fund_score - 50) / 5  # -10 to +10 range
        score += fund_adjustment

        return max(0, min(100, score))

    def _determine_signal(self, score: float) -> str:
        """
        Determine trading signal from score.

        Args:
            score: Score from 0-100

        Returns:
            Signal: bullish, bearish, or neutral
        """
        if score >= 65:
            return "bullish"
        elif score <= 35:
            return "bearish"
        else:
            return "neutral"

    def _generate_medium_summary(self, ticker: str, metrics: Dict[str, Any]) -> str:
        """
        Generate text summary for medium analysis.

        Args:
            ticker: Stock ticker
            metrics: Analysis metrics

        Returns:
            Summary text
        """
        tech_signal = metrics.get("technical_signal", "neutral")
        fund_score = metrics.get("fundamental_score", 50)
        price = metrics.get("price", 0)
        rsi = metrics.get("rsi", 50)

        summary = f"{ticker} Analysis Summary\n\n"
        summary += f"Price: ${price:.2f}\n"
        summary += f"Technical Signal: {tech_signal.upper()}\n"
        summary += f"Fundamental Score: {fund_score:.0f}/100\n"
        summary += f"RSI: {rsi:.1f}\n\n"

        if tech_signal == "bullish" and fund_score > 60:
            summary += "Strong buy candidate with positive technical and fundamental signals."
        elif tech_signal == "bearish" or fund_score < 40:
            summary += "Caution advised. Weak technical or fundamental indicators."
        else:
            summary += "Mixed signals. Further analysis recommended."

        return summary

    def _extract_score_from_report(self, report: str) -> float:
        """
        Extract overall score from deep analysis report.

        Args:
            report: Final report text

        Returns:
            Estimated score (0-100)
        """
        # Simple heuristic based on recommendation keywords
        report_lower = report.lower()

        if "strong buy" in report_lower or "highly recommended" in report_lower:
            return 85.0
        elif "buy" in report_lower and "not" not in report_lower.split("buy")[0][-20:]:
            return 70.0
        elif "hold" in report_lower:
            return 50.0
        elif "sell" in report_lower:
            return 30.0
        else:
            return 50.0  # Neutral default


if __name__ == "__main__":
    # Test batch analyzer
    print("Testing Batch Analyzer...\n")

    analyzer = BatchAnalyzer(max_workers=5)

    # Test quick analysis
    print("1. Quick Analysis Test")
    print("="*60)
    test_tickers = ["AAPL", "MSFT", "GOOGL"]
    results = analyzer.batch_quick_analysis(test_tickers)

    for result in results:
        print(f"{result.ticker}: {result.score:.1f}/100 ({result.signal}) - {result.execution_time:.2f}s")

    print("\n2. Tiered Screening Test (Small Sample)")
    print("="*60)

    # Small test with 10 stocks
    small_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]

    tiered_results = analyzer.tiered_screening(
        small_universe,
        quick_top_n=5,
        medium_top_n=3,
        deep_top_n=2
    )

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nTop Deep Analysis Results:")
    for i, result in enumerate(tiered_results["deep"][:3], 1):
        print(f"\n{i}. {result.ticker}")
        print(f"   Score: {result.score:.1f}/100")
        print(f"   Signal: {result.signal.upper()}")
        print(f"   Time: {result.execution_time:.1f}s")
