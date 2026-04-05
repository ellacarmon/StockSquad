# StockSquad Azure Deployment Guide

This guide walks you through deploying StockSquad to Azure using Infrastructure as Code (Bicep) and GitHub Actions CI/CD.

## 📋 Prerequisites

1. **Azure Account** with an active subscription
2. **Azure CLI** installed ([Install Guide](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli))
3. **GitHub Account** with this repository
4. **Azure OpenAI** service already provisioned

## 💰 Estimated Monthly Cost

| Resource | SKU/Tier | Cost/Month |
|----------|----------|------------|
| App Service Plan | B1 (Basic) | $13.14 |
| Azure Files | Standard, 10GB | $2.00 |
| Application Insights | Pay-as-you-go | $5-10 |
| Log Analytics | Pay-as-you-go | $2-5 |
| Azure OpenAI | Usage-based | $20-40 |
| **Total** | | **$42-70/month** |

---

## 🚀 Deployment Steps

### Step 1: Azure CLI Login

```bash
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_ID"

# Verify
az account show
```

### Step 2: Create Resource Group

```bash
# Create a resource group (adjust location as needed)
az group create \
  --name rg-stocksquad-prod \
  --location eastus

# Verify
az group show --name rg-stocksquad-prod
```

### Step 3: Deploy Infrastructure with Bicep

```bash
# Navigate to project root
cd /path/to/StockSquad

# Deploy infrastructure
az deployment group create \
  --resource-group rg-stocksquad-prod \
  --template-file infra/main.bicep \
  --parameters \
    azureOpenAiEndpoint="https://YOUR_NAME.openai.azure.com/" \
    azureOpenAiApiKey="YOUR_API_KEY" \
    azureOpenAiDeploymentName="gpt-4o" \
    azureOpenAiEmbeddingDeploymentName="text-embedding-ada-002" \
    telegramBotToken="YOUR_TELEGRAM_BOT_TOKEN" \
    environment=prod \
    appServiceSku=B1

# This will create:
# - App Service Plan (B1)
# - App Service (Linux, Python 3.11)
# - Storage Account with File Share
# - Application Insights
# - Log Analytics Workspace
```

**Note the outputs** - you'll need the `appServiceName` for GitHub Actions.

### Step 4: Configure GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions → New repository secret

Create the following secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AZURE_CREDENTIALS` | Azure service principal credentials | `{"clientId": "...", "clientSecret": "...", ...}` |
| `AZURE_SUBSCRIPTION_ID` | Your Azure subscription ID | `12345678-1234-1234-1234-123456789012` |
| `AZURE_RESOURCE_GROUP` | Resource group name | `rg-stocksquad-prod` |
| `AZURE_WEBAPP_NAME` | App Service name from Bicep output | `stocksquad-app-prod-abc123` |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | `https://yourname.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key | `sk-...` |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Chat model deployment name | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | Embedding model deployment name | `text-embedding-ada-002` |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token (optional) | `123456:ABC-DEF...` |
| `XAI_API_KEY` | xAI/Grok API key (optional) | `xai-...` |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key (optional) | `DEMO` |

#### Creating Azure Service Principal

```bash
# Create service principal for GitHub Actions
az ad sp create-for-rbac \
  --name "github-actions-stocksquad" \
  --role contributor \
  --scopes /subscriptions/YOUR_SUBSCRIPTION_ID/resourceGroups/rg-stocksquad-prod \
  --sdk-auth

# Copy the entire JSON output and paste as AZURE_CREDENTIALS secret
```

### Step 5: Trigger Deployment

**Option 1: Push to main branch**
```bash
git add .
git commit -m "Initial deployment [deploy-infra]"
git push origin main
```

**Option 2: Manual trigger**
- Go to GitHub → Actions → "Deploy StockSquad to Azure" → Run workflow

### Step 6: Verify Deployment

```bash
# Get your App Service URL
az webapp show \
  --name YOUR_WEBAPP_NAME \
  --resource-group rg-stocksquad-prod \
  --query defaultHostName -o tsv

# Test the API
curl https://YOUR_WEBAPP_NAME.azurewebsites.net/api/reports

# View logs
az webapp log tail \
  --name YOUR_WEBAPP_NAME \
  --resource-group rg-stocksquad-prod
```

---

## 📊 Monitoring & Logs

### Application Insights
```bash
# Get Application Insights URL
az monitor app-insights component show \
  --app stocksquad-insights-prod-* \
  --resource-group rg-stocksquad-prod \
  --query 'appId' -o tsv

# Open in browser
open "https://portal.azure.com/#@/resource/subscriptions/YOUR_SUB_ID/resourceGroups/rg-stocksquad-prod/providers/microsoft.insights/components/stocksquad-insights-prod-*/overview"
```

### Live Logs
```bash
# Stream logs from App Service
az webapp log tail \
  --name YOUR_WEBAPP_NAME \
  --resource-group rg-stocksquad-prod

# Download logs
az webapp log download \
  --name YOUR_WEBAPP_NAME \
  --resource-group rg-stocksquad-prod \
  --log-file logs.zip
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically:

1. ✅ Builds React frontend
2. ✅ Installs Python dependencies
3. ✅ Creates deployment package
4. ✅ Deploys infrastructure (on manual trigger or `[deploy-infra]` in commit message)
5. ✅ Deploys application to App Service
6. ✅ Runs health checks

**Trigger infrastructure update:**
```bash
git commit -m "Update infrastructure [deploy-infra]"
git push
```

**Regular code deployment:**
```bash
git commit -m "Fix bug in stock analyzer"
git push  # Infrastructure unchanged, only app code deployed
```

---

## 🛠️ Troubleshooting

### App won't start
```bash
# Check logs
az webapp log tail -n YOUR_WEBAPP_NAME -g rg-stocksquad-prod

# Restart app
az webapp restart -n YOUR_WEBAPP_NAME -g rg-stocksquad-prod

# Check startup script
az webapp config show -n YOUR_WEBAPP_NAME -g rg-stocksquad-prod --query 'linuxFxVersion'
```

### ChromaDB connection issues
```bash
# Verify file share is mounted
az webapp config storage-account list \
  -n YOUR_WEBAPP_NAME \
  -g rg-stocksquad-prod

# Check file share exists
az storage share list \
  --account-name YOUR_STORAGE_ACCOUNT
```

### Frontend not loading
```bash
# Check if dist folder exists in deployment
az webapp ssh -n YOUR_WEBAPP_NAME -g rg-stocksquad-prod
# Then run: ls -la ui/web/dist
```

### High costs
```bash
# Check cost analysis
az consumption usage list \
  --start-date 2024-01-01 \
  --end-date 2024-01-31

# Scale down to F1 (Free tier) for testing
az appservice plan update \
  --name YOUR_APP_SERVICE_PLAN \
  --resource-group rg-stocksquad-prod \
  --sku F1
```

---

## 📝 Updating Configuration

### Update environment variables
```bash
az webapp config appsettings set \
  --name YOUR_WEBAPP_NAME \
  --resource-group rg-stocksquad-prod \
  --settings \
    LOG_LEVEL=DEBUG \
    AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
```

### Scale up/down
```bash
# Scale up to B2 (more memory)
az appservice plan update \
  --name YOUR_APP_SERVICE_PLAN \
  --resource-group rg-stocksquad-prod \
  --sku B2

# Scale down to B1
az appservice plan update \
  --name YOUR_APP_SERVICE_PLAN \
  --resource-group rg-stocksquad-prod \
  --sku B1
```

---

## 🗑️ Cleanup

To delete all resources:

```bash
# Delete entire resource group (removes everything)
az group delete \
  --name rg-stocksquad-prod \
  --yes \
  --no-wait

# Verify deletion
az group exists --name rg-stocksquad-prod
```

---

## 🔐 Security Best Practices

1. ✅ **Never commit secrets** - All secrets are in GitHub Secrets
2. ✅ **Use managed identities** when possible (future enhancement)
3. ✅ **HTTPS only** - Enforced in Bicep template
4. ✅ **TLS 1.2 minimum** - Configured in App Service
5. ✅ **Disable FTP** - Configured in Bicep
6. ✅ **Private storage** - Blob public access disabled

---

## 📚 Next Steps

1. **Set up custom domain** (optional)
   ```bash
   az webapp config hostname add \
     --webapp-name YOUR_WEBAPP_NAME \
     --resource-group rg-stocksquad-prod \
     --hostname stocksquad.yourdomain.com
   ```

2. **Enable auto-scaling** (requires Standard tier)
   ```bash
   az monitor autoscale create \
     --resource-group rg-stocksquad-prod \
     --resource YOUR_APP_SERVICE_PLAN \
     --resource-type Microsoft.Web/serverfarms \
     --name autoscale-stocksquad \
     --min-count 1 \
     --max-count 3 \
     --count 1
   ```

3. **Set up staging slot** (requires Standard tier)
   ```bash
   az webapp deployment slot create \
     --name YOUR_WEBAPP_NAME \
     --resource-group rg-stocksquad-prod \
     --slot staging
   ```

---

## 💡 Support

- **Azure Documentation**: https://docs.microsoft.com/azure
- **Bicep Reference**: https://docs.microsoft.com/azure/azure-resource-manager/bicep
- **GitHub Actions**: https://docs.github.com/actions

For issues, check Application Insights or contact your team.
