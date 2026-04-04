"""
Backtest Report Generator
Formats backtest results into readable reports.
"""

from typing import Dict, Any, List
import json


class BacktestReport:
    """Generate formatted reports from backtest results."""

    @staticmethod
    def generate_text_report(results: Dict[str, Any]) -> str:
        """
        Generate human-readable text report.

        Args:
            results: Backtest results dictionary

        Returns:
            Formatted report string
        """
        if 'error' in results:
            return f"\n❌ Backtest Error: {results['error']}\n"

        # Check if multi-ticker or single ticker
        if 'individual_results' in results:
            return BacktestReport._generate_multi_ticker_report(results)
        else:
            return BacktestReport._generate_single_ticker_report(results)

    @staticmethod
    def _generate_single_ticker_report(results: Dict[str, Any]) -> str:
        """Generate report for single ticker backtest."""
        ticker = results.get('ticker', 'N/A')
        period = results.get('period', 'N/A')
        model = results.get('model_type', 'N/A')

        report = f"""
{'═' * 70}
              BACKTEST RESULTS - {ticker}
{'═' * 70}

Period:         {period}
ML Model:       {model}

{'─' * 70}
TRADING STATISTICS
{'─' * 70}
Total Trades:           {results['total_trades']}
Winning Trades:         {results['winning_trades']} ({results['win_rate']}%)
Losing Trades:          {results['losing_trades']} ({100 - results['win_rate']:.1f}%)

{'─' * 70}
RETURNS
{'─' * 70}
Average Return/Trade:   {results['avg_return_per_trade']:+.2f}%
Median Return/Trade:    {results['median_return']:+.2f}%
Total Return (all):     {results['total_return']:+.2f}%

Average Winner:         {results['avg_winner']:+.2f}%
Average Loser:          {results['avg_loser']:+.2f}%

Best Trade:             {results['best_trade']:+.2f}%
Worst Trade:            {results['worst_trade']:+.2f}%

Profit Factor:          {results['profit_factor']}

{'─' * 70}
RISK-ADJUSTED METRICS
{'─' * 70}
Sharpe Ratio:           {results.get('sharpe_ratio', 'N/A')}
Sortino Ratio:          {results.get('sortino_ratio', 'N/A')}
Max Drawdown:           {results.get('max_drawdown', 'N/A')}%
Calmar Ratio:           {results.get('calmar_ratio', 'N/A')}

{'─' * 70}
ML MODEL PERFORMANCE
{'─' * 70}
Prediction Accuracy:    {results['prediction_accuracy']}%
Avg ML Confidence:      {results['avg_ml_confidence']}%
"""

        # Add calibration curve if available
        calibration_curve = results.get('calibration_curve', {})
        if calibration_curve:
            report += f"\n{'─' * 70}\nCALIBRATION CURVE\n{'─' * 70}\n"
            report += "Confidence Bin → Actual Win Rate:\n"
            for bin_key in sorted(calibration_curve.keys()):
                bin_data = calibration_curve[bin_key]
                report += f"  {bin_data['avg_confidence']:.1f}% → {bin_data['actual_win_rate']:.1f}% ({bin_data['count']} trades)\n"

        # Add regime breakdown if available
        regime_breakdown = results.get('regime_breakdown', {})
        if regime_breakdown:
            report += f"\n{'─' * 70}\nREGIME ANALYSIS\n{'─' * 70}\n"
            if 'trending' in regime_breakdown:
                trending = regime_breakdown['trending']
                report += f"Trending Markets:\n"
                report += f"  Trades: {trending['count']}, Win Rate: {trending['win_rate']}%, "
                report += f"Avg Return: {trending['avg_return']:+.2f}%, Accuracy: {trending['prediction_accuracy']}%\n"
            if 'ranging' in regime_breakdown:
                ranging = regime_breakdown['ranging']
                report += f"Ranging Markets:\n"
                report += f"  Trades: {ranging['count']}, Win Rate: {ranging['win_rate']}%, "
                report += f"Avg Return: {ranging['avg_return']:+.2f}%, Accuracy: {ranging['prediction_accuracy']}%\n"

        report += f"""
{'─' * 70}
BENCHMARK COMPARISON
{'─' * 70}
Buy & Hold Return:      {results['buy_hold_return']:+.2f}%
Strategy Return:        {results['total_return']:+.2f}%
"""

        # Add verdict
        if results['total_return'] > results['buy_hold_return']:
            diff = results['total_return'] - results['buy_hold_return']
            verdict = f"✓ Strategy BEAT buy-and-hold by {diff:+.2f}%"
        else:
            diff = results['buy_hold_return'] - results['total_return']
            verdict = f"✗ Strategy UNDERPERFORMED buy-and-hold by {diff:.2f}%"

        report += f"\n{'─' * 70}\nVERDICT\n{'─' * 70}\n{verdict}\n"

        # Add assessment
        assessment = BacktestReport._assess_performance(results)
        report += f"\n{assessment}\n"

        report += f"{'═' * 70}\n"

        return report

    @staticmethod
    def _generate_multi_ticker_report(results: Dict[str, Any]) -> str:
        """Generate report for multi-ticker backtest."""
        report = f"""
{'═' * 70}
         MULTI-TICKER BACKTEST RESULTS
{'═' * 70}

Tickers Tested:         {results['total_tickers']}
Successful Backtests:   {results['successful_tickers']}

{'─' * 70}
AGGREGATE STATISTICS
{'─' * 70}
Total Trades:           {results['total_trades']}
Win Rate:               {results['win_rate']}%
Avg Return/Trade:       {results['avg_return_per_trade']:+.2f}%
Total Return (all):     {results['total_return']:+.2f}%

ML Prediction Accuracy: {results['prediction_accuracy']}%
Avg ML Confidence:      {results['avg_confidence']}%

{'─' * 70}
TICKER PERFORMANCE
{'─' * 70}
Best Performer:         {results['best_ticker']}
Worst Performer:        {results['worst_ticker']}

{'─' * 70}
INDIVIDUAL TICKER RESULTS
{'─' * 70}
"""

        # Add individual ticker summaries
        individual = results.get('individual_results', [])
        for r in individual:
            if 'error' not in r and r.get('total_trades', 0) > 0:
                report += f"\n{r['ticker']:6s}: "
                report += f"{r['total_trades']:3d} trades, "
                report += f"{r['win_rate']:5.1f}% win rate, "
                report += f"{r['avg_return_per_trade']:+6.2f}% avg return"

        report += f"\n\n{'═' * 70}\n"

        return report

    @staticmethod
    def _assess_performance(results: Dict[str, Any]) -> str:
        """
        Provide qualitative assessment of backtest results.

        Args:
            results: Backtest results

        Returns:
            Assessment string
        """
        win_rate = results['win_rate']
        avg_return = results['avg_return_per_trade']
        prediction_acc = results['prediction_accuracy']
        profit_factor = results.get('profit_factor', 0)

        assessment = "ASSESSMENT:\n" + "─" * 70 + "\n"

        # Win rate assessment
        if win_rate > 60:
            assessment += "✓ Excellent win rate (>60%) - Strong signal quality\n"
        elif win_rate > 55:
            assessment += "✓ Good win rate (55-60%) - Signals have edge\n"
        elif win_rate > 50:
            assessment += "○ Marginal win rate (50-55%) - Slight edge\n"
        else:
            assessment += "✗ Poor win rate (<50%) - No edge detected\n"

        # Average return assessment
        if avg_return > 2:
            assessment += "✓ Strong average returns (>2%) per trade\n"
        elif avg_return > 1:
            assessment += "○ Moderate average returns (1-2%) per trade\n"
        elif avg_return > 0:
            assessment += "○ Weak average returns (0-1%) per trade\n"
        else:
            assessment += "✗ Negative average returns - Losing strategy\n"

        # Prediction accuracy
        if prediction_acc > 60:
            assessment += "✓ High ML accuracy (>60%) - Model is predictive\n"
        elif prediction_acc > 55:
            assessment += "○ Moderate ML accuracy (55-60%)\n"
        elif prediction_acc > 50:
            assessment += "○ Weak ML accuracy (50-55%) - Barely better than random\n"
        else:
            assessment += "✗ Poor ML accuracy (<50%) - Model not predictive\n"

        # Profit factor
        if profit_factor != 'inf':
            pf = float(profit_factor)
            if pf > 2:
                assessment += "✓ Excellent profit factor (>2.0) - Wins >> Losses\n"
            elif pf > 1.5:
                assessment += "✓ Good profit factor (1.5-2.0)\n"
            elif pf > 1:
                assessment += "○ Marginal profit factor (1.0-1.5)\n"
            else:
                assessment += "✗ Poor profit factor (<1.0) - Losses > Wins\n"

        # Overall verdict
        assessment += "\nOVERALL: "
        if win_rate > 55 and avg_return > 1.5 and prediction_acc > 55:
            assessment += "Strategy shows PROMISE - Consider live paper trading"
        elif win_rate > 50 and avg_return > 0.5:
            assessment += "Strategy has WEAK EDGE - Needs optimization"
        else:
            assessment += "Strategy NEEDS WORK - Not ready for live trading"

        return assessment

    @staticmethod
    def export_to_json(results: Dict[str, Any], output_path: str):
        """
        Export results to JSON file.

        Args:
            results: Backtest results
            output_path: Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n[Report] Results exported to {output_path}")

    @staticmethod
    def export_trades_to_csv(results: Dict[str, Any], output_path: str):
        """
        Export trade list to CSV.

        Args:
            results: Backtest results
            output_path: Path to save CSV file
        """
        import csv

        trades = results.get('trades', [])
        if not trades:
            print("[Report] No trades to export")
            return

        # Get all keys from first trade
        fieldnames = list(trades[0].keys())

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trades)

        print(f"\n[Report] {len(trades)} trades exported to {output_path}")
