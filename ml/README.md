# StockSquad ML Pipeline

Complete machine learning infrastructure for training and deploying stock prediction models.

## Overview

The ML pipeline consists of four main stages:

1. **Data Collection** - Fetch historical stock data and store in SQLite
2. **Feature Engineering** - Calculate technical indicators and create labels
3. **Model Training** - Train XGBoost, Random Forest, and LightGBM models
4. **Inference** - Make real-time predictions using trained models

## Architecture

```
┌─────────────────────┐
│  Data Collection    │  ← yfinance API
│  (data_collector)   │
└──────────┬──────────┘
           │
           ↓ SQLite Database
┌─────────────────────┐
│ Feature Engineering │  ← Calculate indicators
│ (feature_engineer)  │    Create forward labels
└──────────┬──────────┘
           │
           ↓ Training-ready CSV
┌─────────────────────┐
│  Model Training     │  ← XGBoost, RF, LightGBM
│  (train_models)     │    Walk-forward validation
└──────────┬──────────┘
           │
           ↓ Saved .joblib models
┌─────────────────────┐
│ Prediction Engine   │  ← Real-time inference
│ (prediction_engine) │    Integrated with agents
└─────────────────────┘
```

## Quick Start

### 1. macOS Users: Install OpenMP First

**XGBoost requires OpenMP on macOS.** Install it before proceeding:

```bash
# Run the automated installer:
./ml/install_xgboost_macos.sh

# Or manually:
brew install libomp
```

This takes 5-10 minutes (installs cmake dependency).

**Skip this step if you're on Linux or Windows.**

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `scikit-learn>=1.3.0`
- `xgboost>=2.0.0`
- `lightgbm>=4.0.0`
- `joblib>=1.3.0`
- `yfinance>=0.2.36`

### 3. Collect Training Data

Collect 5 years of historical data for S&P 100 stocks:

```bash
python3 ml/training/prepare_training_data.py \
    --universe sp100 \
    --period 5y \
    --workers 10
```

Options:
- `--universe`: `sp100` (100 stocks) or `sp500` (500 stocks)
- `--period`: `1y`, `2y`, `5y`, `10y`, `max`
- `--workers`: Number of parallel workers (default: 5)
- `--max-tickers`: Limit number of tickers (useful for testing)

**Output:**
- SQLite database: `ml/training/data/stocks.db`
- Training CSV: `ml/training/data/training_data.csv`

**Database Schema:**

| Table | Columns |
|-------|---------|
| `price_history` | ticker, date, open, high, low, close, volume, adj_close |
| `stock_info` | ticker, name, sector, industry, market_cap, employees |
| `technical_indicators` | ticker, date, rsi_14, macd, sma_20, sma_50, sma_200, ema_12, ema_26, bb_upper, bb_middle, bb_lower, volume_sma_20 |
| `labels` | ticker, date, forward_1d_return, forward_5d_return, forward_10d_return, forward_20d_return, direction_1d, direction_5d, direction_10d, direction_20d |

### 4. Train Models

Train XGBoost, Random Forest, and LightGBM models:

```bash
python3 ml/training/train_models.py
```

**Training Process:**

1. **Data Preparation**
   - Load training data from CSV
   - Remove rows with missing values
   - Create train/validation split (80/20, time-based)

2. **Walk-Forward Validation**
   - 3-fold time series cross-validation
   - Ensures no look-ahead bias
   - Reports average metrics across folds

3. **Final Model Training**
   - Train on 80% of data
   - Validate on most recent 20%
   - Save models and metadata

**Model Types:**

| Model | Classifier | Regressor | Best For |
|-------|-----------|-----------|----------|
| XGBoost | Direction (up/down) | % Return | Fast inference, good accuracy |
| Random Forest | Direction | % Return | Robust, interpretable |
| LightGBM | Direction | % Return | Large datasets, very fast |

**Saved Artifacts:**

```
ml/models/
├── xgboost_direction_classifier.joblib
├── xgboost_return_regressor.joblib
├── random_forest_direction_classifier.joblib
├── random_forest_return_regressor.joblib
├── lightgbm_direction_classifier.joblib
├── lightgbm_return_regressor.joblib
├── feature_names.json
└── training_metadata.json
```

**Example Output:**

```
TRAINING FINAL MODELS
======================================================================

Training samples: 112,456
Validation samples: 28,115
Features: 21

======================================================================
MODEL TYPE: XGBOOST
======================================================================

[ModelTrainer] Training xgboost direction classifier...
  Train Accuracy: 0.6234
  Val Accuracy: 0.5823
  Val Precision: 0.5912
  Val Recall: 0.5745
  Val F1: 0.5827

[ModelTrainer] Training xgboost return regressor...
  Train MSE: 12.3456
  Val MSE: 15.2341
  Train R²: 0.3421
  Val R²: 0.2876
  Val MAE: 2.8934

  ✓ Saved classifier: ml/models/xgboost_direction_classifier.joblib
  ✓ Saved regressor: ml/models/xgboost_return_regressor.joblib
```

### 5. Make Predictions

Test the prediction engine:

```bash
python3 ml/inference/prediction_engine.py
```

**Prediction Output:**

```python
{
    "model_type": "xgboost",
    "direction": "bullish",
    "confidence": 85.3,  # 0-100
    "expected_return": +4.2,  # % return over 5 days
    "score": 89.5,  # -100 to +100
    "recommendation": "BUY",  # BUY, SELL, HOLD, STRONG BUY, STRONG SELL
    "direction_probabilities": {
        "down": 14.7,
        "up": 85.3
    },
    "timestamp": "2026-04-01T14:30:00.123456"
}
```

## Azure ML Training

For large-scale training on Azure ML managed compute:

### Prerequisites

1. **Azure ML workspace created** in Azure Portal
2. **Azure CLI** installed and authenticated: `az login`
3. **Azure ML SDK** installed:
   ```bash
   pip install -r ml/azure_ml/requirements-azure-ml.txt
   ```
4. **Compute cluster** created (or will be auto-created)

### Validate Setup

Before submitting jobs, validate your Azure ML setup:

```bash
python3 ml/azure_ml/test_azure_ml_setup.py
```

This will check:
- Azure ML SDK installation and version
- Azure Identity SDK
- Azure CLI authentication status
- Command API compatibility

### Submit Training Job

```bash
python3 ml/azure_ml/train_on_azure.py \
    --subscription-id <your-subscription-id> \
    --resource-group <your-resource-group> \
    --workspace <your-workspace-name> \
    --compute cpu-cluster
```

**Monitor Job:**

- View job in Azure ML Studio (URL provided in output)
- Or check status via CLI:

```bash
python3 ml/azure_ml/train_on_azure.py \
    --subscription-id <subscription-id> \
    --resource-group <resource-group> \
    --workspace <workspace-name> \
    --check-job <job-name>
```

**Azure ML Benefits:**

- Managed compute clusters (auto-scaling)
- Distributed training across multiple nodes
- MLflow experiment tracking
- Model versioning and registry
- CI/CD integration

## Integration with Agents

The ML models are automatically integrated into the existing TechnicalAgent via the SignalScorer class.

### How It Works

1. **SignalScorer Initialization** (`ml/signal_model.py`)
   ```python
   scorer = SignalScorer(use_ml=True, model_type="xgboost")
   ```

2. **Automatic ML Loading**
   - Attempts to load trained models from `ml/models/`
   - Falls back to rule-based scoring if models unavailable
   - No code changes required in agents

3. **TechnicalAgent Usage** (`agents/technical_agent.py`)
   ```python
   self.signal_scorer = SignalScorer()  # Already integrated!
   signal_data = self.signal_scorer.score_signal(indicators)
   ```

4. **Transparent Fallback**
   - If ML models exist → Use ML predictions
   - If ML models missing → Use rule-based scoring
   - Agents work either way

### Model Selection

Change model type by modifying `technical_agent.py`:

```python
# Use XGBoost (default)
self.signal_scorer = SignalScorer(use_ml=True, model_type="xgboost")

# Use Random Forest
self.signal_scorer = SignalScorer(use_ml=True, model_type="random_forest")

# Use LightGBM
self.signal_scorer = SignalScorer(use_ml=True, model_type="lightgbm")

# Force rule-based (no ML)
self.signal_scorer = SignalScorer(use_ml=False)
```

## Features

### Technical Indicators (Features)

| Category | Features |
|----------|----------|
| **Price** | open, high, low, close |
| **Volume** | volume, volume_sma_20, volume_ratio |
| **Momentum** | rsi_14, macd, macd_signal, macd_hist |
| **Trend** | sma_20, sma_50, sma_200, ema_12, ema_26 |
| **Volatility** | bb_upper, bb_middle, bb_lower |
| **Derived** | price_to_sma20, price_to_sma50 |

### Labels (Targets)

| Label | Description |
|-------|-------------|
| `forward_5d_return` | % return over next 5 trading days (regression target) |
| `direction_5d` | 1 if price goes up, 0 if down (classification target) |

Other forward periods available: 1d, 10d, 20d

### Model Performance Metrics

**Classification (Direction):**
- Accuracy: % of correct direction predictions
- Precision: Of predicted "up", how many were actually up
- Recall: Of actual "up" days, how many were predicted
- F1 Score: Harmonic mean of precision and recall

**Regression (Return):**
- MSE: Mean squared error
- R²: Proportion of variance explained (0 = no better than mean, 1 = perfect)
- MAE: Mean absolute error in percentage points

## Best Practices

### Data Collection

✅ **Do:**
- Collect at least 3-5 years of data for robust training
- Use S&P 100 for faster experimentation
- Use S&P 500 for production models
- Re-collect data monthly to stay current

❌ **Don't:**
- Train on less than 1 year of data
- Mix data from different sources
- Ignore data quality issues (check for missing values)

### Model Training

✅ **Do:**
- Always use walk-forward validation
- Monitor overfitting (train vs validation metrics)
- Train multiple model types and compare
- Save training metadata for reproducibility

❌ **Don't:**
- Use random shuffle (violates time series assumptions)
- Overfit to recent market conditions
- Skip validation step
- Ignore class imbalance (bull vs bear markets)

### Production Deployment

✅ **Do:**
- Retrain models monthly
- Monitor prediction accuracy over time
- Keep rule-based fallback active
- Log all predictions for analysis

❌ **Don't:**
- Deploy without validation
- Rely solely on ML (use rule-based + ML)
- Ignore model drift
- Skip A/B testing

## Troubleshooting

### XGBoost Library Error (macOS)

**Error:** `XGBoost Library (libxgboost.dylib) could not be loaded` or `Library not loaded: @rpath/libomp.dylib`

**Cause:** OpenMP runtime (libomp) is not installed on macOS.

**Solution:**

```bash
# Option 1: Run automated installer
./ml/install_xgboost_macos.sh

# Option 2: Manual installation
brew install libomp

# Option 3: If still not working, reinstall XGBoost
pip uninstall xgboost
pip install xgboost

# Test that it works:
python3 -c "import xgboost; print(f'✅ XGBoost {xgboost.__version__} loaded')"
```

**Note:** This installation takes 5-10 minutes as it installs cmake as a dependency.

### "No training data available"

```bash
# Run data collection first:
python3 ml/training/prepare_training_data.py --max-tickers 10
```

### "Models not loaded, using rule-based fallback"

```bash
# Train models first:
python3 ml/training/train_models.py
```

### "Not enough data for training"

- Increase `--period` (e.g., `5y` instead of `1y`)
- Reduce `forward_periods` in feature engineering
- Check for data gaps in SQLite database

### Azure ML Authentication Issues

```bash
# Login to Azure:
az login

# Set subscription:
az account set --subscription <subscription-id>

# Verify access:
az ml workspace show --name <workspace-name> --resource-group <rg>
```

### Azure ML SDK Version Issues

**Error:** `TypeError: Command.__init__() missing 1 required keyword-only argument: 'component'`

**Cause:** Azure ML SDK v2 API changes between versions

**Solution:** The code now uses `command()` function instead of `Command` class. If you still see this error:

```bash
# Upgrade Azure ML SDK:
pip install --upgrade azure-ai-ml

# Verify version (should be 1.12.0+):
python -c "import azure.ai.ml; print(azure.ai.ml.__version__)"
```

If issues persist, check the Azure ML SDK [migration guide](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-migrate-from-v1).

### Low Model Accuracy

- **Normal for stock prediction!** 55-60% accuracy is good for directional prediction
- Markets are inherently noisy and unpredictable
- Use ensemble of models + rule-based scoring
- Focus on high-confidence predictions only

## Future Enhancements

### Phase 4 Roadmap

- [ ] Sentiment integration (news, social media as features)
- [ ] Alternative data (options flow, insider trading)
- [ ] Ensemble models (combine XGBoost + RF + LightGBM)
- [ ] Online learning (incremental updates)
- [ ] Backtesting engine with transaction costs
- [ ] Model interpretability (SHAP values)
- [ ] Automated hyperparameter tuning
- [ ] Multi-horizon predictions (1d, 5d, 20d combined)

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Time Series Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Azure ML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## Support

For issues or questions:
1. Check existing GitHub issues
2. Review training logs in `ml/training/logs/`
3. Verify data quality in SQLite database
4. Test with small dataset first (`--max-tickers 10`)
