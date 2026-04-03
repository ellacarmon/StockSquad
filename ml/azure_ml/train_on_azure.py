"""
Azure ML Training Script
Submit training job to Azure ML.
"""

import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes


def submit_training_job(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    compute_name: str = "cpu-cluster",
    experiment_name: str = "stocksquad-training"
):
    """
    Submit training job to Azure ML.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        workspace_name: Azure ML workspace name
        compute_name: Compute cluster name
        experiment_name: Experiment name
    """
    print("\n" + "="*70)
    print("SUBMITTING TRAINING JOB TO AZURE ML")
    print("="*70 + "\n")

    # Authenticate
    credential = DefaultAzureCredential()

    # Get ML Client
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    print(f"✓ Connected to workspace: {workspace_name}")

    # Define environment
    env = Environment(
        name="stocksquad-training-env",
        description="Environment for StockSquad ML training",
        conda_file="ml/azure_ml/conda_env.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )

    print(f"✓ Environment defined")

    # Define training command
    from azure.ai.ml import command

    command_job = command(
        code="./",  # Local path to code
        command="python ml/training/train_models.py",
        environment=env,
        compute=compute_name,
        experiment_name=experiment_name,
        display_name="StockSquad Model Training",
        description="Train XGBoost, Random Forest, and LightGBM models for stock prediction"
    )

    print(f"✓ Training command configured")
    print(f"  Compute: {compute_name}")
    print(f"  Experiment: {experiment_name}")

    # Submit job
    print(f"\n📤 Submitting job...")
    job = ml_client.jobs.create_or_update(command_job)

    print(f"\n✅ Job submitted!")
    print(f"  Job name: {job.name}")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"\n🔗 View job in Azure ML Studio:")
    print(f"  {job.services['Studio'].endpoint}")

    return job


def check_job_status(
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    job_name: str
):
    """
    Check status of a training job.

    Args:
        subscription_id: Azure subscription ID
        resource_group: Resource group name
        workspace_name: Azure ML workspace name
        job_name: Job name to check
    """
    credential = DefaultAzureCredential()

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name
    )

    job = ml_client.jobs.get(job_name)

    print(f"\n📊 Job Status: {job.status}")
    print(f"  Name: {job.name}")
    print(f"  Created: {job.creation_context.created_at}")
    print(f"  Duration: {job.properties.get('duration', 'N/A')}")

    if job.status == "Completed":
        print(f"  ✅ Training completed successfully!")
    elif job.status == "Failed":
        print(f"  ❌ Training failed")
        print(f"  Error: {job.properties.get('error', 'Unknown error')}")
    elif job.status in ["Running", "Queued", "Preparing"]:
        print(f"  ⏳ Training in progress...")

    return job


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Submit Azure ML training job")
    parser.add_argument("--subscription-id", required=True, help="Azure subscription ID")
    parser.add_argument("--resource-group", required=True, help="Resource group name")
    parser.add_argument("--workspace", required=True, help="Azure ML workspace name")
    parser.add_argument("--compute", default="cpu-cluster", help="Compute cluster name")
    parser.add_argument("--check-job", help="Check status of existing job")

    args = parser.parse_args()

    if args.check_job:
        check_job_status(
            args.subscription_id,
            args.resource_group,
            args.workspace,
            args.check_job
        )
    else:
        submit_training_job(
            args.subscription_id,
            args.resource_group,
            args.workspace,
            args.compute
        )
