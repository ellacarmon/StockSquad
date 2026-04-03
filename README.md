# StockSquad

**A Multi-Agent Stock Research System powered by Azure AI Foundry**

StockSquad is a Python-based multi-agent system that uses specialized AI agents to collaborate on stock analysis. Each agent has a distinct role and memory, working together to produce comprehensive investment research reports.

## Features

### Multi-Agent System (Phase 2)
- **7 Specialized Agents**: Orchestrator, Data, Technical, Sentiment, Social Media, Fundamentals, Devil's Advocate
- **Intelligent Run Monitoring**: Automatic detection and cancellation of stuck assistant runs
- **Diagnostic Tools**: Built-in debugging for agent performance and thread analysis

### Memory & Intelligence (Phase 1)
- **Dual Memory System**:
  - Short-term memory for session-based agent communication
  - Long-term memory using ChromaDB for persistent analysis storage with semantic search
- **Azure OpenAI Integration**: Uses Azure OpenAI Assistants API with DefaultAzureCredential

### Machine Learning (Phase 4)
- **Trained ML Models**: XGBoost, Random Forest, LightGBM for stock prediction
- **Complete ML Pipeline**: Data collection → Feature engineering → Training → Inference
- **Azure ML Integration**: Distributed training on managed compute clusters
- **Walk-Forward Validation**: Proper time series cross-validation to prevent overfitting
- **Hybrid Scoring**: ML-based predictions with rule-based fallback

### Data & Analysis
- **Market Data Collection**: Real-time and historical data via yfinance
- **Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands, volume analysis
- **Sentiment Analysis**: News and social media sentiment scoring
- **Fundamental Analysis**: Financial ratios and company metrics

### Interfaces (Phase 3)
- **Rich CLI Interface**: Beautiful terminal UI with progress indicators
- **Telegram Bot**: Analyze stocks via Telegram with `/analyze` command
- **Analysis History**: Retrieve and reference past analyses

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OrchestratorAgent                          │
│         (Coordinates workflow & synthesizes reports)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                 ┌─────────┴─────────┐
                 │                   │
        ┌────────▼────────┐   ┌─────▼──────────┐
        │   DataAgent     │   │ TechnicalAgent │
        │ (Market data)   │   │ (TA + ML model)│
        └─────────────────┘   └────────────────┘
                 │                   │
        ┌────────▼────────┐   ┌─────▼──────────┐
        │ SentimentAgent  │   │ SocialAgent    │
        │ (News analysis) │   │ (Social media) │
        └─────────────────┘   └────────────────┘
                 │                   │
        ┌────────▼────────┐   ┌─────▼──────────┐
        │ FundamentalsAg. │   │ DevilsAdvocate │
        │ (Financials)    │   │ (Challenges)   │
        └─────────────────┘   └────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
   ┌────▼─────────┐  ┌───▼────────────┐
   │ Short-term   │  │  Long-term     │
   │ Memory       │  │  Memory        │
   │ (Session)    │  │  (ChromaDB)    │
   └──────────────┘  └────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      ML Pipeline (Phase 4)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data Collection → Feature Engineering → Model Training        │
│     (yfinance)         (Indicators)         (XGBoost/RF)       │
│                                                  │              │
│                                                  ↓              │
│                                          Prediction Engine      │
│                                                  │              │
│                                                  ↓              │
│                                          TechnicalAgent         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Python 3.11 or higher
- Azure subscription with:
  - Azure OpenAI Service with GPT-4o deployment
  - Azure OpenAI embeddings deployment (text-embedding-ada-002)
- Azure CLI installed and authenticated (`az login`)
- **macOS users only:** OpenMP runtime for XGBoost (see Installation step 1.5)

## Installation

### 1. Clone and Set Up Environment

```bash
# Navigate to project directory
cd StockSquad

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 1.5. macOS Only: Install OpenMP for XGBoost

**XGBoost requires OpenMP on macOS.** Run this in a separate terminal (takes 5-10 minutes):

```bash
# Automated installer:
./ml/install_xgboost_macos.sh

# Or manually:
brew install libomp

# Test:
python3 -c "import xgboost; print('✅ XGBoost loaded')"
```

**Linux/Windows users can skip this step.**

See `ml/README.md` for troubleshooting.

### 2. Configure Azure Credentials

Copy the example environment file and fill in your Azure credentials:

```bash
cp .env.example .env
```

Edit `.env` with your Azure credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://<your-resource-name>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-api-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# ChromaDB Configuration
CHROMA_DB_PATH=./chroma_db

# Logging
LOG_LEVEL=INFO
```

### 3. Azure OpenAI Setup

1. Create an Azure OpenAI resource in Azure Portal
2. Deploy models:
   - GPT-4o (for chat/analysis)
   - text-embedding-ada-002 (for embeddings)
3. Note your endpoint URL (e.g., `https://your-resource.openai.azure.com/`)
4. Get your API key from the Azure Portal (or use Azure AD authentication)

**Authenticate with Azure:**
```bash
az login
```

StockSquad uses `DefaultAzureCredential` which automatically uses your Azure CLI credentials.

## Usage

### Quick Start with Telegram Bot

1. **Set up the bot** (see `telegram_bot/README.md`):
   ```bash
   python3 telegram_bot/bot.py
   ```

2. **Analyze stocks via Telegram**:
   - Send: `/analyze AAPL`
   - Bot will coordinate all 7 agents
   - Receive comprehensive analysis in minutes

### Analyze a Stock via CLI

```bash
python main.py analyze AAPL
```

With options:
```bash
# Analyze with custom period
python main.py analyze NVDA --period 6mo

# Save report to file
python main.py analyze MSFT --save

# Show raw data collected
python main.py analyze TSLA --show-data
```

### View Analysis History

```bash
# View past analyses for a ticker
python main.py history AAPL

# Limit number of results
python main.py history AAPL --limit 3
```

### System Information

```bash
# View memory statistics
python main.py stats

# View configuration
python main.py config

# View version
python main.py version
```

### Machine Learning Pipeline

Train ML models to enhance prediction accuracy:

```bash
# 1. Test the ML pipeline
python3 ml/test_ml_pipeline.py

# 2. Collect training data (S&P 100, 5 years)
python3 ml/training/prepare_training_data.py --universe sp100 --period 5y

# 3. Train models (XGBoost, Random Forest, LightGBM)
python3 ml/training/train_models.py

# 4. Test predictions
python3 ml/inference/prediction_engine.py

# 5. (Optional) Train on Azure ML
python3 ml/azure_ml/train_on_azure.py \
    --subscription-id <id> \
    --resource-group <rg> \
    --workspace <workspace>
```

Once trained, models are automatically integrated into TechnicalAgent. See `ml/README.md` for detailed documentation.

### Backtesting ML Models

Test how well ML models would have performed on historical data:

```bash
# Single model backtest
PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
    --ticker AAPL \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --model xgboost \
    --holding-days 5 \
    --confidence 60

# Compare all 3 models
for model in xgboost random_forest lightgbm; do
    echo "Testing $model..."
    PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
        --ticker AAPL \
        --start 2024-01-01 \
        --end 2024-12-31 \
        --model $model
done

# **Ensemble models (RECOMMENDED - Best Performance)**
# Unanimous: Only trade when ALL 3 models agree (highest win rate!)
PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
    --ticker AAPL \
    --model ensemble_unanimous \
    --confidence 60

# Results on AAPL 2024:
#   Ensemble Unanimous: 61.9% win rate, 1.92 profit factor, 69.8% accuracy ✨
#   XGBoost alone:      54.5% win rate, 1.41 profit factor, 60.3% accuracy
#   Random Forest:      49.3% win rate, 1.25 profit factor, 70.4% accuracy

# Export results for analysis
PYTHONPATH=. python3 ml/backtesting/run_backtest.py \
    --ticker NVDA \
    --model ensemble_unanimous \
    --export results.json \
    --export-trades trades.csv
```

**Key Findings from Backtesting:**
- **Ensemble Unanimous** strategy achieves 61.9% win rate (vs 50% random)
- High conviction trades (all 3 models agree) show 69.8% prediction accuracy
- Profit factor of 1.92 indicates winners 2x larger than losers
- Transaction costs (0.2% round-trip) included in all results

## Project Structure

```
StockSquad/
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py        # Workflow orchestration
│   ├── data_agent.py          # Market data collection
│   ├── technical_agent.py     # Technical analysis + ML
│   ├── sentiment_agent.py     # News sentiment analysis
│   ├── social_media_agent.py  # Social media sentiment
│   ├── fundamentals_agent.py  # Financial analysis
│   ├── devils_advocate.py     # Challenge analysis
│   ├── assistant_utils.py     # Intelligent run monitoring
│   └── run_diagnostics.py     # Debugging tools
├── memory/
│   ├── __init__.py
│   ├── short_term.py          # Session memory
│   └── long_term.py           # ChromaDB vector store
├── tools/
│   ├── __init__.py
│   ├── market_data.py         # yfinance wrapper
│   ├── ta_indicators.py       # Technical indicators
│   └── news_api.py            # News fetching
├── ml/
│   ├── signal_model.py        # ML + rule-based signal scorer
│   ├── training/
│   │   ├── data_collector.py  # Historical data collection
│   │   ├── feature_engineer.py# Feature engineering
│   │   ├── train_models.py    # Model training
│   │   └── prepare_training_data.py
│   ├── inference/
│   │   ├── prediction_engine.py  # Real-time predictions
│   │   └── ensemble_predictor.py # Ensemble model voting
│   ├── backtesting/
│   │   ├── simple_backtester.py  # Walk-forward backtesting
│   │   ├── metrics.py         # Performance metrics
│   │   ├── report.py          # Results reporting
│   │   └── run_backtest.py    # CLI interface
│   ├── azure_ml/
│   │   ├── train_on_azure.py  # Azure ML job submission
│   │   └── conda_env.yml      # Azure ML environment
│   ├── models/                # Trained models (.joblib)
│   ├── test_ml_pipeline.py    # ML pipeline tests
│   └── README.md              # ML documentation
├── telegram_bot/
│   ├── bot.py                 # Telegram bot implementation
│   └── README.md              # Bot setup instructions
├── data/                      # Data storage directory
├── chroma_db/                 # ChromaDB storage
├── reports/                   # Saved reports
├── main.py                    # CLI entry point
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## How It Works

1. **User Request**: You request analysis for a ticker via CLI
2. **Orchestration**: OrchestratorAgent checks long-term memory for past analyses
3. **Data Collection**: DataAgent fetches market data, financials, and news
4. **Synthesis**: OrchestratorAgent synthesizes findings into a comprehensive report
5. **Storage**: Analysis is stored in ChromaDB with embeddings for future reference
6. **Output**: Formatted report displayed in terminal (and optionally saved)

## Example Output

```
╭─────────────────────────────────────────╮
│      StockSquad Analysis                │
│ Ticker: AAPL | Period: 1y              │
╰─────────────────────────────────────────╯

[OrchestratorAgent] Analyzing AAPL...
[OrchestratorAgent] Checking for past analyses...
[OrchestratorAgent] Found 2 past analysis(es)
[OrchestratorAgent] Requesting data collection from DataAgent...
[OrchestratorAgent] Data collection complete
[OrchestratorAgent] Synthesizing final report...
[OrchestratorAgent] Report synthesis complete
[OrchestratorAgent] Storing analysis in long-term memory...

Analysis complete!

================================================================================

# Executive Summary

Apple Inc. (AAPL) demonstrates strong financial health with...

[Full report continues...]

================================================================================
```

## Memory System

### Short-Term Memory
- In-memory storage for current analysis session
- Shared scratchpad where agents post intermediate findings
- Message history for agent communication
- Cleared after each analysis

### Long-Term Memory
- Persistent storage using ChromaDB
- Vector embeddings for semantic search
- Indexed by ticker and date
- Agents reference past analyses in new reports

## Troubleshooting

### Configuration Errors

```
Error: Failed to load configuration
```

**Solution**: Ensure `.env` file exists and contains Azure OpenAI endpoint and API key.

### Authentication Errors

```
Error: DefaultAzureCredential failed to retrieve a token
```

**Solution**: Run `az login` to authenticate with Azure CLI, or set up Azure credentials properly.

### Import Errors

```
ModuleNotFoundError: No module named 'X'
```

**Solution**: Ensure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Authentication Errors

```
Error: DefaultAzureCredential failed to retrieve a token
```

**Solution**:
1. Run `az login` to authenticate
2. Ensure you have access to the Azure OpenAI resource
3. Verify your Azure subscription is active

## Development Roadmap

### Phase 1 - Foundation ✅ Complete
- [x] Basic project structure
- [x] DataAgent and OrchestratorAgent
- [x] Short-term and long-term memory
- [x] CLI interface
- [x] Azure OpenAI Assistants API integration
- [x] DefaultAzureCredential authentication

### Phase 2 - The Squad ✅ Complete
- [x] TechnicalAgent with TA indicators
- [x] SentimentAgent with news analysis
- [x] SocialMediaAgent for social sentiment
- [x] FundamentalsAgent with ratio analysis
- [x] DevilsAdvocateAgent for challenge/debate
- [x] Intelligent run monitoring with stuck detection
- [x] Diagnostic tools for debugging

### Phase 3 - Interfaces ✅ Complete
- [x] Telegram bot integration
- [x] Bot command handlers (/analyze, /help, /history)
- [x] Rich formatting and progress updates
- [x] Analysis history via Telegram

### Phase 4 - Intelligence & ML ✅ Complete
- [x] Historical data collection pipeline with Polygon.io fallback
- [x] Feature engineering with 21 technical indicators
- [x] ML model training (XGBoost, Random Forest, LightGBM)
- [x] Prediction engine for real-time inference
- [x] Azure ML integration for distributed training
- [x] Hybrid scoring (ML + rule-based fallback)
- [x] **Backtesting engine** with walk-forward validation & transaction costs
- [x] **Ensemble predictor** (Voting, Averaging, Unanimous strategies)
- [x] Model comparison & performance metrics (win rate, profit factor, Sharpe ratio)
- [ ] Alternative data integration (options flow, insider trading)
- [ ] Advanced agents (OptionsAgent, MacroAgent, InsiderAgent)

### Phase 5 - Production (Planned)
- [ ] Azure AI Foundry Evaluation integration
- [ ] Model A/B testing framework
- [ ] Automated model retraining pipeline
- [ ] Performance dashboards
- [ ] Alerting and monitoring
- [ ] Multi-user support
- [ ] API endpoint deployment

## Contributing

This is a learning project focused on mastering multi-agent orchestration and Azure AI Foundry.

## Resources

- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [Azure OpenAI Assistants API](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/assistant)
- [Azure Identity (DefaultAzureCredential)](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [yfinance Documentation](https://github.com/ranaroussi/yfinance)

## License

MIT License - See [LICENSE](LICENSE) file for details.

**⚠️ DISCLAIMER:** This software is for educational and research purposes only. It is NOT financial advice and should NOT be used as the sole basis for investment decisions. Stock trading involves substantial risk of loss. Always do your own research and consult with a qualified financial advisor before making investment decisions.

---

**Created**: March 2026 | **Stack**: Python 3.11+ + Azure AI Foundry + ChromaDB
