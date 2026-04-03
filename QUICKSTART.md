# StockSquad Quick Start Guide

Get StockSquad up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Set Up Azure Credentials

1. **Create Azure OpenAI Resource**
   - Go to Azure Portal
   - Create an Azure OpenAI resource
   - Deploy GPT-4o model
   - Deploy text-embedding-ada-002 model
   - Note your endpoint URL and API key

2. **Configure Environment**

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required `.env` variables:
```env
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Step 3: Authenticate with Azure

```bash
# Login to Azure CLI
az login
```

This ensures DefaultAzureCredential can authenticate.

## Step 4: Run Your First Analysis

```bash
python main.py analyze AAPL
```

Expected output:
- DataAgent fetches market data
- OrchestratorAgent synthesizes findings
- Comprehensive report displayed
- Analysis stored in memory

## Step 5: Explore Commands

```bash
# View past analyses
python main.py history AAPL

# Check memory stats
python main.py stats

# View configuration
python main.py config

# Get help
python main.py --help
```

## Common Issues

### "Failed to load configuration"
**Fix**: Ensure `.env` file exists with Azure OpenAI endpoint and API key

### "DefaultAzureCredential failed"
**Fix**:
1. Run `az login` to authenticate with Azure
2. Verify you have access to the Azure OpenAI resource
3. Check your Azure subscription is active

### "Module not found"
**Fix**: Activate virtual environment and install dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

- Try analyzing different tickers (MSFT, NVDA, TSLA, GOOGL)
- Save reports with `--save` flag
- View raw data with `--show-data` flag
- Analyze different time periods with `--period 6mo`

## Example Usage

```bash
# Basic analysis
python main.py analyze MSFT

# 6-month analysis with saved report
python main.py analyze NVDA --period 6mo --save

# Show all collected data
python main.py analyze TSLA --show-data

# View history
python main.py history MSFT --limit 5
```

## Architecture at a Glance

```
User → CLI → OrchestratorAgent
              ├─> DataAgent (fetches data)
              ├─> Short-term memory (session)
              └─> Long-term memory (ChromaDB)
```

## What Happens During Analysis

1. OrchestratorAgent checks for past analyses in ChromaDB
2. DataAgent fetches:
   - Current stock info (price, market cap, sector)
   - Price history (OHLCV data)
   - Financial metrics (P/E, margins, ratios)
   - Recent news articles
3. Data posted to shared scratchpad
4. OrchestratorAgent synthesizes comprehensive report
5. Report stored with embeddings for future reference

## Files You Need to Know

- `main.py` - CLI entry point
- `config.py` - Configuration management
- `.env` - Your Azure credentials (create from .env.example)
- `agents/orchestrator.py` - Workflow coordinator
- `agents/data_agent.py` - Data collection
- `memory/long_term.py` - ChromaDB storage

## Getting Help

```bash
# Command-specific help
python main.py analyze --help
python main.py history --help

# General help
python main.py --help
```

Happy analyzing!
