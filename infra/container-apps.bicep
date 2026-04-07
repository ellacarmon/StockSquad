// StockSquad Azure Infrastructure - Container Apps
// Modern serverless container platform, better than App Service for containers

@description('Name of the application')
param appName string = 'stocksquad'

@description('Azure region')
param location string = resourceGroup().location

@description('Environment (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'prod'

@description('Azure OpenAI endpoint')
param azureOpenAiEndpoint string = ''

@description('Azure OpenAI API key')
@secure()
param azureOpenAiApiKey string = ''

@description('Azure OpenAI deployment name for chat')
param azureOpenAiDeploymentName string = ''

@description('Azure OpenAI deployment name for embeddings')
param azureOpenAiEmbeddingDeploymentName string = ''

@description('Telegram bot token')
@secure()
param telegramBotToken string = ''

@description('xAI/Grok API key')
@secure()
param xaiApiKey string = ''

@description('Alpha Vantage API key')
@secure()
param alphaVantageApiKey string = ''

@description('Existing ACR name')
param acrName string = 'cloudcopilotacr'

@description('ACR resource group name')
param acrResourceGroup string = 'cloudcopilot-swn0a-rg'

@description('Docker image to deploy')
param dockerImage string = 'cloudcopilotacr.azurecr.io/stocksquad:5'

@description('Permit.io API key for authorization')
@secure()
param permitIoApiKey string = ''

@description('Permit.io PDP URL (Policy Decision Point)')
param permitIoPdpUrl string = 'https://cloudpdp.api.permit.io'

@description('JWT secret key for token signing')
@secure()
param jwtSecretKey string = ''

@description('JWT token expiration in minutes')
param jwtExpireMinutes string = '10080'

@description('Email service provider (resend, sendgrid, smtp, acs)')
param emailProvider string = 'resend'

@description('Email service API key (Resend or SendGrid)')
@secure()
param emailApiKey string = ''

@description('Email from address')
param emailFrom string = 'noreply@stocksquad.app'

@description('SMTP host (if using SMTP provider)')
param smtpHost string = ''

@description('SMTP port (if using SMTP provider)')
param smtpPort string = '587'

@description('SMTP username (if using SMTP provider)')
param smtpUser string = ''

@description('SMTP password (if using SMTP provider)')
@secure()
param smtpPassword string = ''

@description('Azure Communication Services connection string (if using ACS provider)')
@secure()
param acsConnectionString string = ''

// Variables
var uniqueSuffix = uniqueString(resourceGroup().id)
var shortSuffix = take(uniqueSuffix, 6)
var containerAppName = '${appName}-${environment}'
var containerEnvName = '${appName}-env'
var storageAccountName = 'st${appName}'
var fileShareName = 'chromadb'
var appInsightsName = '${appName}-ai'
var logAnalyticsName = '${appName}-logs'

// Environment variables
var environmentVariables = concat([
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
    value: '/tmp/chromadb'  // Using local ephemeral storage
  }
  {
    name: 'AZURE_STORAGE_ACCOUNT_NAME'
    value: storageAccount.name  // For blob backup of analyses
  }
  {
    name: 'LOG_LEVEL'
    value: environment == 'prod' ? 'INFO' : 'DEBUG'
  }
  {
    name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'
    value: appInsights.properties.ConnectionString
  }
  {
    name: 'PERMIT_IO_API_KEY'
    value: permitIoApiKey
  }
  {
    name: 'PERMIT_IO_PDP_URL'
    value: permitIoPdpUrl
  }
  {
    name: 'JWT_SECRET_KEY'
    value: jwtSecretKey
  }
  {
    name: 'JWT_EXPIRE_MINUTES'
    value: jwtExpireMinutes
  }
  {
    name: 'EMAIL_PROVIDER'
    value: emailProvider
  }
  {
    name: 'EMAIL_FROM'
    value: emailFrom
  }
], concat(!empty(azureOpenAiEndpoint) ? [
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
] : [], emailProvider == 'resend' && !empty(emailApiKey) ? [
  {
    name: 'RESEND_API_KEY'
    value: emailApiKey
  }
] : emailProvider == 'sendgrid' && !empty(emailApiKey) ? [
  {
    name: 'SENDGRID_API_KEY'
    value: emailApiKey
  }
] : emailProvider == 'smtp' && !empty(smtpHost) ? [
  {
    name: 'SMTP_HOST'
    value: smtpHost
  }
  {
    name: 'SMTP_PORT'
    value: smtpPort
  }
  {
    name: 'SMTP_USER'
    value: smtpUser
  }
  {
    name: 'SMTP_PASSWORD'
    value: smtpPassword
  }
] : (emailProvider == 'acs' || emailProvider == 'azure') && !empty(acsConnectionString) ? [
  {
    name: 'ACS_CONNECTION_STRING'
    value: acsConnectionString
  }
] : []))

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: logAnalyticsName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
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
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Reference existing Azure Container Registry in different resource group
resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: acrName
  scope: resourceGroup(acrResourceGroup)
}

// Storage Account for ChromaDB
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: {
    name: 'Standard_LRS'
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
    shareQuota: 10
    enabledProtocols: 'SMB'
  }
}

// Container Apps Environment
resource containerEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: containerEnvName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Storage for Container Apps Environment
resource containerEnvStorage 'Microsoft.App/managedEnvironments/storages@2023-05-01' = {
  name: 'chromadb'
  parent: containerEnv
  properties: {
    azureFile: {
      accountName: storageAccount.name
      accountKey: storageAccount.listKeys().keys[0].value
      shareName: fileShareName
      accessMode: 'ReadWrite'
    }
  }
}

// Container App
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: containerAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    managedEnvironmentId: containerEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        allowInsecure: false
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: 'system'
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'stocksquad'
          image: dockerImage
          resources: {
            cpu: json('2.0')
            memory: '4Gi'
          }
          env: environmentVariables
          // ChromaDB uses SQLite which doesn't work well on network storage
          // Using local ephemeral storage instead - data will be lost on restart
          // but analyses can be regenerated
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '50'
              }
            }
          }
        ]
      }
    }
  }
  tags: {
    environment: environment
    application: appName
  }
}

// Role Assignments for Managed Identity
// Built-in role IDs (these are the same across all Azure subscriptions)
var storageBlobDataContributorRoleId = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe' // Storage Blob Data Contributor

// Grant AcrPull to Container App Managed Identity - deployed at ACR's resource group scope via module
module acrPullRoleAssignment 'acr-role-assignment.bicep' = {
  name: 'acrPullRoleAssignment'
  scope: resourceGroup(acrResourceGroup)
  params: {
    acrName: acrName
    principalId: containerApp.identity.principalId
  }
}

// Grant Storage Blob Data Contributor to Container App (for ChromaDB)
resource storageBlobRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, containerApp.id, storageBlobDataContributorRoleId)
  scope: storageAccount
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributorRoleId)
    principalId: containerApp.identity.principalId
    principalType: 'ServicePrincipal'
  }
}

// Outputs
output containerAppName string = containerApp.name
output containerAppFqdn string = containerApp.properties.configuration.ingress.fqdn
output containerAppUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output acrName string = acr.name
output acrLoginServer string = acr.properties.loginServer
output storageAccountName string = storageAccount.name
output appInsightsName string = appInsights.name
output resourceGroupName string = resourceGroup().name
