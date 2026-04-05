// StockSquad Azure Infrastructure
// Deploys App Service, Storage, and Application Insights within $100/month budget

@description('Name of the application (used for resource naming)')
param appName string = 'stocksquad'

@description('Azure region for resources')
param location string = resourceGroup().location

@description('Environment name (dev, staging, prod)')
@allowed([
  'dev'
  'staging'
  'prod'
])
param environment string = 'prod'

@description('App Service SKU')
@allowed([
  'B1'  // Basic - $13.14/month - 1.75GB RAM (Recommended)
  'B2'  // Basic - $26.28/month - 3.5GB RAM
  'S1'  // Standard - $69.35/month - 1.75GB RAM
])
param appServiceSku string = 'B1'

@description('Azure OpenAI endpoint (e.g., https://yourname.openai.azure.com/)')
param azureOpenAiEndpoint string = ''

@description('Azure OpenAI API key (optional if using managed identity)')
@secure()
param azureOpenAiApiKey string = ''

@description('Azure OpenAI deployment name for chat')
param azureOpenAiDeploymentName string = ''

@description('Azure OpenAI deployment name for embeddings')
param azureOpenAiEmbeddingDeploymentName string = ''

@description('Telegram bot token')
@secure()
param telegramBotToken string = ''

@description('xAI/Grok API key (optional)')
@secure()
param xaiApiKey string = ''

@description('Alpha Vantage API key (optional)')
@secure()
param alphaVantageApiKey string = ''

// Variables
var uniqueSuffix = uniqueString(resourceGroup().id)
var appServicePlanName = '${appName}-plan-${environment}-${uniqueSuffix}'
var appServiceName = '${appName}-app-${environment}-${uniqueSuffix}'
var storageAccountName = 'st${appName}${environment}${take(uniqueSuffix, 8)}'
var fileShareName = 'chromadb'
var appInsightsName = '${appName}-insights-${environment}-${uniqueSuffix}'
var logAnalyticsName = '${appName}-logs-${environment}-${uniqueSuffix}'

// Base app settings (always included)
var baseAppSettings = [
  {
    name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
    value: 'false'
  }
  {
    name: 'WEBSITE_HTTPLOGGING_RETENTION_DAYS'
    value: '3'
  }
  {
    name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
    value: appInsights.properties.ConnectionString
  }
  {
    name: 'ApplicationInsightsAgent_EXTENSION_VERSION'
    value: '~3'
  }
  {
    name: 'TELEGRAM_BOT_TOKEN'
    value: telegramBotToken
  }
  {
    name: 'XAI_API_KEY'
    value: xaiApiKey
  }
  {
    name: 'ALPHA_VANTAGE_API_KEY'
    value: alphaVantageApiKey
  }
  {
    name: 'CHROMA_DB_PATH'
    value: '/mnt/chromadb'
  }
  {
    name: 'LOG_LEVEL'
    value: environment == 'prod' ? 'INFO' : 'DEBUG'
  }
]

// Azure OpenAI settings (only included if endpoint is provided)
var openAiAppSettings = !empty(azureOpenAiEndpoint) ? [
  {
    name: 'AZURE_OPENAI_ENDPOINT'
    value: azureOpenAiEndpoint
  }
  {
    name: 'AZURE_OPENAI_API_KEY'
    value: azureOpenAiApiKey
  }
  {
    name: 'AZURE_OPENAI_DEPLOYMENT_NAME'
    value: azureOpenAiDeploymentName
  }
  {
    name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'
    value: azureOpenAiEmbeddingDeploymentName
  }
  {
    name: 'AZURE_OPENAI_API_VERSION'
    value: '2024-02-15-preview'
  }
] : []

var allAppSettings = concat(baseAppSettings, openAiAppSettings)

// Log Analytics Workspace (required for Application Insights)
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    features: {
      enableLogAccessUsingOnlyResourcePermissions: true
    }
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    IngestionMode: 'LogAnalytics'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Storage Account for ChromaDB persistence
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'  // Locally redundant storage (cheapest)
  }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    allowBlobPublicAccess: false
  }
  tags: {
    environment: environment
    application: appName
  }
}

// File Share for ChromaDB
resource fileShare 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-01-01' = {
  name: '${storageAccount.name}/default/${fileShareName}'
  properties: {
    shareQuota: 10  // 10GB quota (~$2/month)
    enabledProtocols: 'SMB'
  }
}

// App Service Plan
resource appServicePlan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: appServiceSku
    tier: startsWith(appServiceSku, 'B') ? 'Basic' : 'Standard'
  }
  kind: 'linux'
  properties: {
    reserved: true  // Required for Linux
  }
  tags: {
    environment: environment
    application: appName
  }
}

// App Service (Web App)
resource appService 'Microsoft.Web/sites@2023-01-01' = {
  name: appServiceName
  location: location
  kind: 'app,linux'
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      alwaysOn: appServiceSku != 'F1' && appServiceSku != 'D1'  // Always On requires Basic or higher
      ftpsState: 'Disabled'
      minTlsVersion: '1.2'
      healthCheckPath: '/api/reports'
      appSettings: allAppSettings
      azureStorageAccounts: {
        chromadb: {
          type: 'AzureFiles'
          accountName: storageAccount.name
          shareName: fileShareName
          mountPath: '/mnt/chromadb'
          accessKey: storageAccount.listKeys().keys[0].value
        }
      }
    }
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Outputs
output appServiceName string = appService.name
output appServiceUrl string = 'https://${appService.properties.defaultHostName}'
output storageAccountName string = storageAccount.name
output appInsightsName string = appInsights.name
output appInsightsInstrumentationKey string = appInsights.properties.InstrumentationKey
output resourceGroupName string = resourceGroup().name
