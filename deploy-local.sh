#!/bin/bash
# Local deployment script for StockSquad to Azure Container Apps
# Reads parameters from .env file

set -e

echo "========================================="
echo "StockSquad - Local Deployment to Azure"
echo "========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found"
    exit 1
fi

# Load .env file
echo "📄 Loading environment variables from .env..."
export $(grep -v '^#' .env | xargs)

# Check required variables
REQUIRED_VARS=(
    "AZURE_OPENAI_ENDPOINT"
    "AZURE_OPENAI_API_KEY"
    "AZURE_OPENAI_DEPLOYMENT_NAME"
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    "TELEGRAM_BOT_TOKEN"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ Error: $var is not set in .env"
        exit 1
    fi
done

# Azure configuration (you can customize these)
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-stocksquad}"
LOCATION="${AZURE_LOCATION:-eastus2}"
DEPLOYMENT_NAME="stocksquad-$(date +%Y%m%d-%H%M%S)"

echo ""
echo "Configuration:"
echo "  Resource Group: $RESOURCE_GROUP"
echo "  Location: $LOCATION"
echo "  Deployment: $DEPLOYMENT_NAME"
echo ""

# Ask for confirmation
read -p "Continue with deployment? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Deployment cancelled"
    exit 1
fi

# Create resource group if it doesn't exist
echo "🔧 Creating resource group (if needed)..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output none

# ACR configuration
ACR_NAME="${ACR_NAME:-cloudcopilotacr}"
ACR_RESOURCE_GROUP="${ACR_RESOURCE_GROUP:-cloudcopilot-swn0a-rg}"
DOCKER_IMAGE="${DOCKER_IMAGE:-${ACR_NAME}.azurecr.io/stocksquad:latest}"

# Deploy infrastructure
echo "🚀 Deploying infrastructure..."
DEPLOYMENT_OUTPUT=$(az deployment group create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$DEPLOYMENT_NAME" \
    --template-file ./infra/container-apps.bicep \
    --parameters \
        permitIoApiKey="$PERMIT_IO_API_KEY" \
        jwtSecretKey="$JWT_SECRET_KEY" \
        emailProvider="acs" \
        acsConnectionString="$ACS_CONNECTION_STRING" \
        emailFrom="$EMAIL_FROM" \
        azureOpenAiEndpoint="$AZURE_OPENAI_ENDPOINT" \
        azureOpenAiApiKey="$AZURE_OPENAI_API_KEY" \
        azureOpenAiDeploymentName="$AZURE_OPENAI_DEPLOYMENT_NAME" \
        azureOpenAiEmbeddingDeploymentName="$AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME" \
        telegramBotToken="$TELEGRAM_BOT_TOKEN" \
        xaiApiKey="${XAI_API_KEY:-}" \
        alphaVantageApiKey="${ALPHA_VANTAGE_API_KEY:-}" \
        environment=prod \
        acrName="$ACR_NAME" \
        acrResourceGroup="$ACR_RESOURCE_GROUP" \
        dockerImage="$DOCKER_IMAGE" \
    --output json)

# Extract outputs
CONTAINER_APP_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerAppName.value')
APP_URL=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerAppUrl.value')

echo ""
echo "✅ Infrastructure deployed successfully!"
echo ""
echo "Using existing ACR: $ACR_NAME"
echo "Using Docker image: $DOCKER_IMAGE"
echo ""
echo "ℹ️  Note: Using pre-existing image from ACR"
echo "   To update the container image, run: ./update-container.sh"

echo ""
echo "========================================="
echo "✅ Deployment Complete!"
echo "========================================="
echo "App URL: $APP_URL"
echo "Container App: $CONTAINER_APP_NAME"
echo "Resource Group: $RESOURCE_GROUP"
echo "========================================="
echo ""
echo "Test your deployment:"
echo "  curl $APP_URL/api/reports"
echo ""
echo "View logs:"
echo "  az containerapp logs show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --follow"
echo ""
