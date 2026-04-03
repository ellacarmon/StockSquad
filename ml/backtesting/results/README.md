# Backtesting Results

This directory contains example backtest results showing ML model performance on historical data.

## Example Results

### `example_AAPL_ensemble_unanimous.json`

**Ensemble Unanimous Strategy on AAPL (2024)**

- **Strategy**: Ensemble Unanimous (all 3 models must agree)
- **Period**: 2024-01-01 to 2024-12-31
- **Results**:
  - Win Rate: 60.3%
  - Profit Factor: 2.02
  - Prediction Accuracy: 74.6%
  - Average Return/Trade: +1.26%
  - Total Trades: 63
  - vs Buy & Hold: +6.95% outperformance

**Key Insights:**
- High conviction trades (all models agree) show 74.6% accuracy
- Profit factor > 2.0 means winners are 2x larger than losers
- Conservative strategy (only 63 trades/year) but high quality signals

## Running Your Own Backtests

```bash
# Single ticker
PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
    --ticker AAPL \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --model ensemble_unanimous \
    --export my_results.json

# Multiple tickers
for ticker in AAPL MSFT NVDA GOOGL; do
    PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
        --ticker $ticker \
        --model ensemble_unanimous \
        --export results_${ticker}.json
done
```

## Understanding the Results

### Key Metrics

- **Win Rate**: % of profitable trades (>50% is good, >60% is excellent)
- **Profit Factor**: Gross profits / Gross losses (>1.5 is good, >2.0 is excellent)
- **Prediction Accuracy**: % of correct direction predictions
- **Avg Return/Trade**: Expected value per trade after transaction costs
- **Total Return**: Sum of all trade returns (NOT portfolio return)

### Important Notes

⚠️ **These are backtested results, not live trading results**
- Past performance does NOT guarantee future results
- Transaction costs (0.2% round-trip) are included
- Walk-forward validation prevents lookahead bias
- Results are for educational purposes only

### Model Comparison (AAPL 2024)

| Model | Win Rate | Profit Factor | Accuracy | Trades |
|-------|----------|---------------|----------|--------|
| Ensemble Unanimous | 60.3% | 2.02 | 74.6% | 63 |
| XGBoost | 54.5% | 1.41 | 60.3% | 189 |
| Random Forest | 49.3% | 1.25 | 70.4% | 142 |
| LightGBM | 50.6% | 1.32 | 48.7% | 158 |

**Winner**: Ensemble Unanimous - Best quality signals

## Next Steps

1. Test on different tickers (tech, value, commodities)
2. Try different time periods (bull markets, bear markets)
3. Optimize confidence thresholds
4. Test longer holding periods (10-20 days)
5. Compare against other benchmarks

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Stock trading involves substantial risk of loss. Always consult with a qualified financial advisor before making investment decisions.
