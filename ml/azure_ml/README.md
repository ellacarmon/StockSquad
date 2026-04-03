# Azure ML Training

This directory contains scripts for submitting training jobs to Azure ML managed compute.

## Quick Start

### 1. Install Azure ML SDK

```bash
pip install -r requirements-azure-ml.txt
```

### 2. Validate Setup

```bash
python3 test_azure_ml_setup.py
```

### 3. Submit Training Job

```bash
python3 train_on_azure.py \
    --subscription-id <your-subscription-id> \
    --resource-group <your-resource-group> \
    --workspace <your-workspace-name> \
    --compute cpu-cluster
```

## Files

- `train_on_azure.py` - Main script for submitting training jobs
- `test_azure_ml_setup.py` - Validation script for Azure ML setup
- `conda_env.yml` - Environment specification for Azure ML compute
- `requirements-azure-ml.txt` - Required packages for Azure ML SDK

## Troubleshooting

### Command API Error

If you see: `TypeError: Command.__init__() missing 1 required keyword-only argument: 'component'`

**Solution:**
```bash
pip install --upgrade azure-ai-ml
python3 test_azure_ml_setup.py  # Validate the fix
```

### Authentication Issues

If you see authentication errors:

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription <subscription-id>

# Verify access to workspace
az ml workspace show --name <workspace-name> --resource-group <rg>
```

### SDK Version Check

```bash
python -c "import azure.ai.ml; print(azure.ai.ml.__version__)"
```

Should be `>= 1.12.0`

## Why Use Azure ML?

- **Managed Compute**: No need to provision VMs
- **Auto-scaling**: Compute scales based on demand
- **Experiment Tracking**: MLflow integration
- **Model Registry**: Version and deploy models
- **Cost Management**: Pay only for compute time used

## Local Training vs Azure ML

| Feature | Local Training | Azure ML |
|---------|---------------|----------|
| **Setup** | Simple (`pip install`) | Requires Azure account |
| **Cost** | Free (your machine) | Pay for compute time |
| **Speed** | Limited by your hardware | Scale to powerful VMs |
| **Dataset Size** | Limited by RAM | Handle large datasets |
| **Experiments** | Manual tracking | Automatic MLflow tracking |
| **Best For** | Development & testing | Production training |

## Next Steps

After training completes:

1. Models are saved to Azure ML workspace
2. Download models and place in `ml/models/`
3. Models automatically integrate with TechnicalAgent

See main ML README for more details: `ml/README.md`
