#!/bin/bash
# Quick update script - only rebuilds and pushes container image
# Use this after infrastructure is already deployed

set -e

echo "========================================="
echo "StockSquad - Update Container"
echo "========================================="

# Azure configuration
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-stocksquad}"
ACR_NAME="${ACR_NAME:-cloudcopilotacr}"
ACR_RESOURCE_GROUP="${ACR_RESOURCE_GROUP:-cloudcopilot}"

echo "🔍 Finding deployed resources..."
echo "  Resource Group: $RESOURCE_GROUP"
echo "  ACR: $ACR_NAME (in resource group: $ACR_RESOURCE_GROUP)"

ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"

# Get Container App name
CONTAINER_APP_NAME=$(az containerapp list \
    --resource-group "$RESOURCE_GROUP" \
    --query "[0].name" -o tsv)

if [ -z "$CONTAINER_APP_NAME" ]; then
    echo "❌ Error: No Container App found in resource group $RESOURCE_GROUP"
    exit 1
fi
IMAGE_TAG=$(git rev-parse --short HEAD)
echo "  ACR: $ACR_NAME"
echo "  Container App: $CONTAINER_APP_NAME"
echo "  Image tag: $IMAGE_TAG"
echo ""

# Build and push Docker image
echo "🐳 Building Docker image..."
docker build -t stocksquad:$IMAGE_TAG  --platform linux/amd64 .

echo "🔐 Logging in to ACR..."
az acr login --name "$ACR_NAME"

echo "🏷️  Tagging image..."
docker tag stocksquad:$IMAGE_TAG "$ACR_LOGIN_SERVER/stocksquad:$IMAGE_TAG"

echo "📤 Pushing image to ACR..."
docker push "$ACR_LOGIN_SERVER/stocksquad:$IMAGE_TAG"

# Update Container App
echo "🔄 Updating Container App..."
az containerapp update \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --image "$ACR_LOGIN_SERVER/stocksquad:$IMAGE_TAG" \
    --output none

# Get app URL
APP_URL=$(az containerapp show \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query properties.configuration.ingress.fqdn -o tsv)

echo ""
echo "========================================="
echo "✅ Container Updated!"
echo "========================================="
echo "App URL: https://$APP_URL"
echo ""
echo "Test your deployment:"
echo "  curl https://$APP_URL/api/reports"
echo ""
