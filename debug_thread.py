"""
Debug script to check thread status
"""

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from config import get_settings

# Get the thread ID from your logs
THREAD_ID = "thread_4zrzhIwBeEUUrwyVWaWASPZe"  # From your error logs
RUN_ID = "run_0oxktKLhRQ6hfA4tzZ1cIODQ"  # From your error logs

settings = get_settings()

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    api_version=settings.azure_openai_api_version,
    azure_endpoint=settings.azure_openai_endpoint,
    azure_ad_token_provider=token_provider,
)

print("Checking run status...")
try:
    run = client.beta.threads.runs.retrieve(
        thread_id=THREAD_ID,
        run_id=RUN_ID
    )

    print(f"\n{'='*60}")
    print(f"Thread ID: {THREAD_ID}")
    print(f"Run ID: {RUN_ID}")
    print(f"Status: {run.status}")
    print(f"Created: {run.created_at}")
    print(f"Model: {run.model}")
    print(f"{'='*60}\n")

    if run.status == "failed":
        print(f"❌ Run failed:")
        print(f"   Error: {run.last_error}")
    elif run.status in ["queued", "in_progress"]:
        print(f"⏳ Run is still {run.status}")
        print(f"\nTo cancel this run:")
        print(f"  run = client.beta.threads.runs.cancel(thread_id='{THREAD_ID}', run_id='{RUN_ID}')")

        # Option to cancel
        cancel = input("\nCancel this run? (y/n): ")
        if cancel.lower() == 'y':
            print("Cancelling...")
            run = client.beta.threads.runs.cancel(
                thread_id=THREAD_ID,
                run_id=RUN_ID
            )
            print(f"✅ Run cancelled. New status: {run.status}")
    elif run.status == "completed":
        print("✅ Run completed successfully")
    else:
        print(f"ℹ️  Run status: {run.status}")

except Exception as e:
    print(f"❌ Error: {e}")
