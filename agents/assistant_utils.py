"""
Assistant Utilities
Helper functions for Azure OpenAI Assistants API with intelligent monitoring.
"""

import time
from typing import Any, Optional
from openai import AzureOpenAI
from datetime import datetime


class AssistantTimeoutError(Exception):
    """Raised when assistant run exceeds timeout."""
    pass


class AssistantStuckError(Exception):
    """Raised when assistant appears to be stuck in a loop."""
    pass


def wait_for_run_completion(
    client: AzureOpenAI,
    thread_id: str,
    run_id: str,
    timeout: int = 180,  # 3 minutes default
    poll_interval: float = 1.0,
    max_same_status_count: int = 60  # Cancel if stuck in same status for 60 polls
) -> Any:
    """
    Wait for an assistant run to complete with intelligent monitoring.

    Args:
        client: Azure OpenAI client
        thread_id: Thread ID
        run_id: Run ID
        timeout: Maximum seconds to wait (default: 180)
        poll_interval: Seconds between polls (default: 1.0)
        max_same_status_count: Cancel if status doesn't change after this many polls

    Returns:
        Completed run object

    Raises:
        AssistantTimeoutError: If run doesn't complete within timeout
        AssistantStuckError: If run appears stuck in a loop
    """
    start_time = time.time()
    elapsed = 0
    poll_count = 0
    last_status = None
    same_status_count = 0

    print(f"[AssistantMonitor] Starting run monitor (timeout: {timeout}s)")

    while elapsed < timeout:
        poll_count += 1

        try:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
        except Exception as e:
            print(f"[AssistantMonitor] Error retrieving run: {e}")
            raise

        # Track if status is changing
        if run.status == last_status:
            same_status_count += 1
        else:
            same_status_count = 0
            last_status = run.status
            print(f"[AssistantMonitor] Status changed to: {run.status} (poll #{poll_count})")

        # Detect stuck runs
        if same_status_count >= max_same_status_count:
            print(f"[AssistantMonitor] ⚠️  Run stuck in '{run.status}' for {same_status_count} polls")

            # Try to cancel the stuck run
            try:
                print(f"[AssistantMonitor] Attempting to cancel stuck run...")
                client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
                raise AssistantStuckError(
                    f"Run stuck in '{run.status}' status for {same_status_count} consecutive polls. "
                    f"This likely indicates an infinite loop or stuck assistant. Run has been cancelled."
                )
            except Exception as e:
                raise AssistantStuckError(
                    f"Run stuck in '{run.status}' and failed to cancel: {e}"
                )

        # Check terminal states
        if run.status == "completed":
            print(f"[AssistantMonitor] ✓ Run completed after {poll_count} polls ({elapsed:.1f}s)")
            return run

        elif run.status == "failed":
            error_msg = f"Assistant run failed: {run.last_error if run.last_error else 'Unknown error'}"
            print(f"[AssistantMonitor] ✗ {error_msg}")
            raise RuntimeError(error_msg)

        elif run.status == "cancelled":
            print(f"[AssistantMonitor] ✗ Run was cancelled")
            raise RuntimeError("Assistant run was cancelled")

        elif run.status == "expired":
            print(f"[AssistantMonitor] ✗ Run expired")
            raise RuntimeError("Assistant run expired")

        elif run.status == "requires_action":
            # This shouldn't happen in wait_for_run_completion
            # Caller should handle requires_action separately
            print(f"[AssistantMonitor] ⚠️  Run requires action (unexpected here)")
            return run

        elif run.status in ["queued", "in_progress"]:
            # Still running, log progress periodically
            if poll_count % 10 == 0:
                print(f"[AssistantMonitor] Still {run.status}... (poll #{poll_count}, {elapsed:.0f}s elapsed)")

            time.sleep(poll_interval)
            elapsed = time.time() - start_time

        else:
            error_msg = f"Unknown run status: {run.status}"
            print(f"[AssistantMonitor] ✗ {error_msg}")
            raise RuntimeError(error_msg)

    # Timeout reached - cancel the run
    print(f"[AssistantMonitor] ⏱️  Timeout reached ({timeout}s). Cancelling run...")
    try:
        client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run_id)
    except Exception as e:
        print(f"[AssistantMonitor] Failed to cancel: {e}")

    raise AssistantTimeoutError(
        f"Assistant run exceeded timeout of {timeout}s after {poll_count} polls. "
        f"Final status: {run.status}. Run has been cancelled."
    )


def wait_for_run_with_actions(
    client: AzureOpenAI,
    thread_id: str,
    run_id: str,
    tool_executor: callable,
    timeout: int = 180,
    poll_interval: float = 1.0
) -> Any:
    """
    Wait for assistant run with support for tool/function calls.

    Args:
        client: Azure OpenAI client
        thread_id: Thread ID
        run_id: Run ID
        tool_executor: Function to execute tools (takes tool_name, arguments)
        timeout: Maximum seconds to wait
        poll_interval: Seconds between polls

    Returns:
        Completed run object

    Raises:
        AssistantTimeoutError: If run doesn't complete within timeout
    """
    start_time = time.time()
    elapsed = 0

    while elapsed < timeout:
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )

        if run.status == "completed":
            return run

        elif run.status == "failed":
            error_msg = f"Assistant run failed: {run.last_error}"
            raise RuntimeError(error_msg)

        elif run.status == "cancelled":
            raise RuntimeError("Assistant run was cancelled")

        elif run.status == "expired":
            raise RuntimeError("Assistant run expired")

        elif run.status == "requires_action":
            # Handle tool calls
            import json
            tool_calls = run.required_action.submit_tool_outputs.tool_calls
            tool_outputs = []

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                try:
                    output = tool_executor(tool_name, arguments)
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps(output) if not isinstance(output, str) else output
                    })
                except Exception as e:
                    tool_outputs.append({
                        "tool_call_id": tool_call.id,
                        "output": json.dumps({"error": str(e)})
                    })

            # Submit tool outputs
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )

        elif run.status in ["queued", "in_progress"]:
            time.sleep(poll_interval)
            elapsed = time.time() - start_time

        else:
            raise RuntimeError(f"Unknown run status: {run.status}")

    # Timeout reached
    raise AssistantTimeoutError(
        f"Assistant run exceeded timeout of {timeout}s. "
        f"Current status: {run.status}"
    )
