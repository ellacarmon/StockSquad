"""
Run Diagnostics
Analyze stuck or slow assistant runs to understand what's happening.
"""

from typing import Dict, Any
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from config import get_settings


def diagnose_run(thread_id: str, run_id: str) -> Dict[str, Any]:
    """
    Diagnose a potentially stuck assistant run.

    Args:
        thread_id: Thread ID
        run_id: Run ID

    Returns:
        Diagnostic information dictionary
    """
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

    print(f"\n{'='*70}")
    print(f"DIAGNOSING RUN")
    print(f"{'='*70}\n")

    # Get run details
    try:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        print(f"📋 Basic Info:")
        print(f"   Thread ID: {thread_id}")
        print(f"   Run ID: {run_id}")
        print(f"   Status: {run.status}")
        print(f"   Model: {run.model}")
        print(f"   Created At: {run.created_at}")
        print(f"   Started At: {run.started_at if run.started_at else 'Not started'}")
        print(f"   Completed At: {run.completed_at if run.completed_at else 'Not completed'}")

        # Check for errors
        if run.last_error:
            print(f"\n❌ Error Details:")
            print(f"   Code: {run.last_error.code}")
            print(f"   Message: {run.last_error.message}")

        # Check usage stats
        if run.usage:
            print(f"\n📊 Token Usage:")
            print(f"   Prompt Tokens: {run.usage.prompt_tokens}")
            print(f"   Completion Tokens: {run.usage.completion_tokens}")
            print(f"   Total Tokens: {run.usage.total_tokens}")

        # Get thread messages to see context size
        print(f"\n💬 Thread Messages:")
        messages = client.beta.threads.messages.list(thread_id=thread_id, limit=10)
        total_messages = len(messages.data)
        print(f"   Total Messages: {total_messages}")

        # Calculate approximate context size
        total_chars = 0
        for msg in messages.data[:10]:
            if hasattr(msg.content[0], 'text'):
                total_chars += len(msg.content[0].text.value)

        print(f"   Recent Messages Size: ~{total_chars:,} characters")
        print(f"   Estimated Tokens: ~{total_chars // 4:,} (rough estimate)")

        # Check if context is too large
        if total_chars > 50000:
            print(f"   ⚠️  WARNING: Very large context! This might cause slowness.")

        # Check run steps
        print(f"\n🔍 Run Steps:")
        try:
            steps = client.beta.threads.runs.steps.list(
                thread_id=thread_id,
                run_id=run_id,
                limit=20
            )

            if steps.data:
                print(f"   Total Steps: {len(steps.data)}")
                for i, step in enumerate(steps.data[:5], 1):
                    print(f"   {i}. {step.type} - {step.status}")

                if len(steps.data) > 20:
                    print(f"   ⚠️  WARNING: {len(steps.data)} steps! This might indicate a loop.")
            else:
                print(f"   No steps recorded yet")
        except Exception as e:
            print(f"   Could not retrieve steps: {e}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if run.status in ["queued", "in_progress"]:
            print(f"   • Run is still active")
            print(f"   • Consider cancelling if it's been running too long")
            print(f"   • To cancel: client.beta.threads.runs.cancel(thread_id, run_id)")

        if run.status == "failed":
            print(f"   • Run failed - check error details above")

        if total_chars > 50000:
            print(f"   • Large context detected - consider:")
            print(f"     - Simplifying prompts")
            print(f"     - Using shorter agent responses")
            print(f"     - Creating new threads more frequently")

        if run.usage and run.usage.total_tokens > 100000:
            print(f"   • High token usage - may be hitting limits")

        print(f"\n{'='*70}\n")

        return {
            "status": run.status,
            "created_at": run.created_at,
            "started_at": run.started_at,
            "total_messages": total_messages,
            "context_size_chars": total_chars,
            "estimated_tokens": total_chars // 4,
            "has_error": run.last_error is not None,
            "error": run.last_error.__dict__ if run.last_error else None,
            "usage": run.usage.__dict__ if run.usage else None,
        }

    except Exception as e:
        print(f"❌ Error diagnosing run: {e}")
        return {"error": str(e)}


def cancel_run(thread_id: str, run_id: str) -> bool:
    """
    Cancel a stuck run.

    Args:
        thread_id: Thread ID
        run_id: Run ID

    Returns:
        True if successfully cancelled
    """
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

    try:
        print(f"🛑 Cancelling run {run_id}...")
        run = client.beta.threads.runs.cancel(
            thread_id=thread_id,
            run_id=run_id
        )
        print(f"✅ Run cancelled. New status: {run.status}")
        return True
    except Exception as e:
        print(f"❌ Failed to cancel: {e}")
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 agents/run_diagnostics.py <thread_id> <run_id>")
        print("\nExample:")
        print("  python3 agents/run_diagnostics.py thread_4zrzhIwBeEUUrwyVWaWASPZe run_0oxktKLhRQ6hfA4tzZ1cIODQ")
        sys.exit(1)

    thread_id = sys.argv[1]
    run_id = sys.argv[2]

    # Run diagnostics
    result = diagnose_run(thread_id, run_id)

    # Offer to cancel if stuck
    if result.get("status") in ["queued", "in_progress"]:
        response = input("\n🛑 Cancel this run? (y/n): ")
        if response.lower() == 'y':
            cancel_run(thread_id, run_id)
