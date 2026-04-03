# Changelog

## [Unreleased] - 2026-03-31

### Changed - Authentication Refactor

**Simplified Azure Authentication: Removed Azure AI Foundry dependency, now using DefaultAzureCredential**

#### What Changed

1. **Removed Azure AI Foundry Project requirement**
   - No longer need Azure AI Foundry project connection string
   - Direct Azure OpenAI integration instead

2. **Updated authentication method**
   - Now uses `DefaultAzureCredential` from `azure-identity`
   - Automatically works with `az login` credentials
   - More enterprise-friendly and secure

3. **Code changes**
   - `agents/data_agent.py`: Updated to use `AzureOpenAI` client with `beta.assistants` API
   - `agents/orchestrator.py`: Updated to use `AzureOpenAI` client with `beta.assistants` API
   - `config.py`: Removed `azure_ai_project_connection_string` field
   - `requirements.txt`: Removed `azure-ai-projects` dependency

4. **Configuration changes**
   - `.env.example`: Removed `AZURE_AI_PROJECT_CONNECTION_STRING`
   - Now only requires:
     - `AZURE_OPENAI_ENDPOINT`
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_DEPLOYMENT_NAME`
     - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`
     - `AZURE_OPENAI_API_VERSION`

5. **Documentation updates**
   - `README.md`: Updated setup instructions
   - `QUICKSTART.md`: Simplified quickstart guide
   - `AUTHENTICATION_GUIDE.md`: New comprehensive authentication guide
   - Removed references to Azure AI Foundry project setup

#### Migration Guide

**Before:**
```env
AZURE_AI_PROJECT_CONNECTION_STRING=<connection-string>
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
```

**After:**
```env
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

**Setup:**
```bash
# Just login with Azure CLI
az login

# That's it! DefaultAzureCredential handles the rest
python main.py analyze AAPL
```

#### Benefits

- ✅ Simpler setup (no Azure AI Foundry project needed)
- ✅ Standard Azure authentication pattern
- ✅ Works with existing Azure CLI credentials
- ✅ More secure (leverages Azure AD)
- ✅ Enterprise-ready
- ✅ Fewer dependencies

#### Technical Details

**Old approach:**
```python
from azure.ai.projects import AIProjectClient

client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=connection_string
)
```

**New approach:**
```python
from openai import AzureOpenAI
from azure.identity import get_bearer_token_provider, DefaultAzureCredential

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
)
```

#### Files Modified

- `agents/data_agent.py` - Updated authentication and API calls
- `agents/orchestrator.py` - Updated authentication and API calls
- `config.py` - Removed connection string field
- `.env.example` - Removed connection string
- `requirements.txt` - Removed azure-ai-projects
- `README.md` - Updated documentation
- `QUICKSTART.md` - Simplified setup guide

#### Files Added

- `AUTHENTICATION_GUIDE.md` - Comprehensive authentication documentation

---

## [0.1.0] - 2026-03-31 - Initial Release

### Added

**Phase 1 - Foundation Implementation**

#### Core Features

1. **Multi-Agent System**
   - `OrchestratorAgent`: Workflow coordination and report synthesis
   - `DataAgent`: Market data and financials collection

2. **Dual Memory Architecture**
   - `ShortTermMemory`: Session-based scratchpad with message history
   - `LongTermMemory`: ChromaDB vector store with semantic search

3. **Market Data Integration**
   - yfinance wrapper for OHLCV data
   - Financial metrics (P/E, margins, debt ratios)
   - Company information and news

4. **CLI Interface**
   - `analyze` - Run stock analysis
   - `history` - View past analyses
   - `stats` - Memory statistics
   - `config` - View configuration
   - Rich terminal output

#### Technical Stack

- Python 3.11+
- Azure OpenAI (GPT-4o, text-embedding-ada-002)
- ChromaDB for vector storage
- yfinance for market data
- Typer + Rich for CLI

#### Project Structure

```
StockSquad/
├── agents/          # Agent implementations
├── memory/          # Short-term and long-term memory
├── tools/           # Market data fetchers
├── main.py          # CLI entry point
├── config.py        # Configuration management
└── requirements.txt # Dependencies
```

#### Documentation

- `README.md` - Complete project documentation
- `QUICKSTART.md` - 5-minute setup guide
- `EMBEDDINGS_EXPLAINED.md` - Deep dive on embeddings
- `project_brief.md` - Original project specification

#### Phase 1 Deliverables (Complete)

- ✅ Project structure and setup
- ✅ DataAgent with OHLCV + financials
- ✅ OrchestratorAgent with task orchestration
- ✅ Short-term memory (session scratchpad)
- ✅ Long-term memory (ChromaDB)
- ✅ CLI interface
- ✅ Documentation

---

## [Phase 2] - 2026-03-31 - The Squad

### Added - Full Multi-Agent System

**Implemented complete agent squad with 2,869 lines of new code**

#### New Agents

1. **TechnicalAgent** (`agents/technical_agent.py`)
   - Calculates RSI, MACD, Moving Averages, Bollinger Bands
   - Integrates ML signal scoring
   - Identifies trends, momentum, support/resistance
   - Generates technical recommendations

2. **SentimentAgent** (`agents/sentiment_agent.py`)
   - Analyzes news headlines and articles
   - Uses embeddings for theme detection
   - Identifies sentiment trends (bullish/neutral/bearish)
   - Tracks narrative shifts

3. **FundamentalsAgent** (`agents/fundamentals_agent.py`)
   - Analyzes financial ratios and metrics
   - Evaluates valuation (P/E, PEG, P/B, etc.)
   - Assesses profitability, growth, financial health
   - Generates fundamental scores and ratings

4. **DevilsAdvocateAgent** (`agents/devils_advocate.py`)
   - Challenges consensus from other agents
   - Identifies risks and blind spots
   - Presents counter-arguments
   - Ensures balanced perspective

#### New Tools & Infrastructure

1. **Technical Indicators** (`tools/ta_indicators.py`)
   - RSI, MACD, SMA, EMA, Bollinger Bands
   - Volume indicators and OBV
   - Support/Resistance detection
   - Trend classification

2. **ML Signal Scoring** (`ml/signal_model.py`)
   - Rule-based signal scorer
   - Scores from -100 to +100
   - Confidence levels and recommendations
   - Ready for trained model upgrade

#### Enhanced Features

1. **Multi-Agent Orchestration**
   - Coordinated workflow across 5 specialized agents
   - Sequential analysis: Data → Technical → Sentiment → Fundamentals → Devil's Advocate
   - Comprehensive synthesis of all perspectives

2. **Comprehensive Reports**
   - Executive summary
   - Technical analysis summary
   - Fundamental analysis summary
   - Sentiment analysis summary
   - Risk assessment (Devil's Advocate)
   - Investment thesis (bull vs bear case)
   - Final recommendation with confidence

3. **Memory Integration**
   - All agent results stored in long-term memory
   - Cross-analysis comparisons
   - Past analysis references

#### Phase 2 Deliverables (Complete)

- ✅ TechnicalAgent with TA indicators
- ✅ ML signal scoring model (rule-based)
- ✅ SentimentAgent with embeddings
- ✅ FundamentalsAgent with ratio analysis
- ✅ DevilsAdvocateAgent
- ✅ Multi-agent coordination
- ✅ Long-term memory integration

---

## Planned (Phase 3)

### Polish & Production

- Backtesting framework
- Azure AI Foundry Evaluation
- Streamlit UI
- Memory decay/aging
- Performance optimization
- Production deployment guide

---

**Legend:**
- ✅ Completed
- 🚧 In Progress
- 📅 Planned
