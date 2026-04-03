"""
ChatAgent: Interactive Q&A about stock analysis reports.

Allows users to ask questions about stored analyses, get clarifications,
and optionally search the web for additional context.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.long_term import LongTermMemory


class ChatAgent:
    """
    Chat agent for interactive Q&A about stock analysis reports.

    Features:
    - Answer questions about stored analysis reports
    - Reference multiple analyses for comparison
    - Optionally search web for additional context
    - Maintain conversation history
    """

    AGENT_NAME = "ChatAgent"

    # Base instructions for chat agent
    AGENT_INSTRUCTIONS = """You are a helpful financial analysis assistant that helps users understand and discuss stock analysis reports.

Your capabilities:
1. Answer questions about specific stock analyses
2. Clarify technical terms and concepts
3. Compare analyses across different time periods
4. Provide context and explanations
5. Search the web when explicitly requested for additional information

Guidelines:
- Base answers primarily on the analysis report data provided
- Be clear when information comes from the report vs. your general knowledge
- If asked to search the web, acknowledge you'll do so before providing web-based info
- Maintain context from the conversation history
- Be concise but thorough
- Always include disclaimers that this is not financial advice

When user asks about:
- Specific metrics: Quote directly from the report
- Comparisons: Reference multiple reports if available
- Market context: Can use general knowledge or web search if requested
- Predictions: Emphasize uncertainty and refer to report's risk factors

Never:
- Provide personalized financial advice
- Guarantee future returns
- Override the analysis report's conclusions without clear reasoning
"""

    def __init__(self, web_search_enabled: bool = False):
        """
        Initialize the ChatAgent.

        Args:
            web_search_enabled: Whether to allow web search for additional context
        """
        self.settings = get_settings()
        self.client = self._initialize_client()
        self.assistant = None
        self.long_term_memory = LongTermMemory()
        self.web_search_enabled = web_search_enabled
        self.conversation_history: List[Dict[str, str]] = []

    def _initialize_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client."""
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
        )

    def create_assistant(self):
        """Create the Azure OpenAI assistant."""
        if self.assistant is None:
            instructions = self.AGENT_INSTRUCTIONS

            if self.web_search_enabled:
                instructions += "\n\nWeb search is ENABLED. When users ask for latest information or market context, you can search the web."
            else:
                instructions += "\n\nWeb search is DISABLED. Rely only on the provided analysis reports and your general knowledge."

            self.assistant = self.client.beta.assistants.create(
                model=self.settings.azure_openai_deployment_name,
                name=self.AGENT_NAME,
                instructions=instructions,
            )
        return self.assistant

    def chat(
        self,
        user_message: str,
        ticker: Optional[str] = None,
        doc_id: Optional[str] = None,
        include_web_search: bool = False
    ) -> Dict[str, Any]:
        """
        Process a chat message about stock analysis.

        Args:
            user_message: The user's question or message
            ticker: Optional ticker to load recent analyses for
            doc_id: Optional specific document ID to reference
            include_web_search: Whether to search web for this query

        Returns:
            Dictionary with response and metadata
        """
        print(f"[{self.AGENT_NAME}] Processing question: {user_message[:50]}...")

        # Load relevant analysis context
        context = self._build_context(ticker=ticker, doc_id=doc_id)

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })

        # Build the full prompt with context
        full_prompt = self._build_prompt(
            user_message=user_message,
            context=context,
            include_web_search=include_web_search
        )

        # Create assistant and thread
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        # Send message
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=full_prompt
        )

        # Run assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Wait for completion
        from agents.assistant_utils import wait_for_run_completion, AssistantTimeoutError
        try:
            run = wait_for_run_completion(
                client=self.client,
                thread_id=thread.id,
                run_id=run.id,
                timeout=120
            )
        except AssistantTimeoutError as e:
            print(f"[{self.AGENT_NAME}] Timeout: {e}")
            return {
                "response": "I'm sorry, the request timed out. Please try again with a simpler question.",
                "error": "timeout"
            }

        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        assistant_messages = [
            msg for msg in messages.data
            if msg.role == "assistant" and msg.created_at > message.created_at
        ]

        if not assistant_messages:
            return {
                "response": "I'm sorry, I couldn't generate a response.",
                "error": "no_response"
            }

        response_text = assistant_messages[0].content[0].text.value

        # Add to conversation history
        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.now().isoformat()
        })

        print(f"[{self.AGENT_NAME}] Response generated")

        return {
            "response": response_text,
            "context_used": context is not None,
            "web_search_used": include_web_search,
            "thread_id": thread.id,
            "timestamp": datetime.now().isoformat()
        }

    def _build_context(
        self,
        ticker: Optional[str] = None,
        doc_id: Optional[str] = None
    ) -> Optional[str]:
        """
        Build context from stored analyses.

        Args:
            ticker: Load recent analyses for this ticker
            doc_id: Load specific document

        Returns:
            Context string or None
        """
        context_parts = []

        # Load specific document if requested
        if doc_id:
            try:
                analysis = self.long_term_memory.get_analysis_by_id(doc_id)
                if analysis:
                    context_parts.append(
                        f"=== Analysis Report (ID: {doc_id}) ===\n"
                        f"Ticker: {analysis.get('ticker', 'Unknown')}\n"
                        f"Date: {analysis.get('timestamp', 'Unknown')}\n\n"
                        f"{analysis.get('final_report', 'No report available')}\n"
                    )
            except Exception as e:
                print(f"[{self.AGENT_NAME}] Failed to load document {doc_id}: {e}")

        # Load recent analyses for ticker
        elif ticker:
            try:
                analyses = self.long_term_memory.retrieve_past_analyses(
                    ticker=ticker.upper(),
                    limit=2
                )

                for i, analysis in enumerate(analyses, 1):
                    context_parts.append(
                        f"=== Analysis Report #{i} ===\n"
                        f"Ticker: {analysis.get('ticker', 'Unknown')}\n"
                        f"Date: {analysis.get('timestamp', 'Unknown')}\n\n"
                        f"{analysis.get('final_report', 'No report available')}\n"
                    )
            except Exception as e:
                print(f"[{self.AGENT_NAME}] Failed to load analyses for {ticker}: {e}")

        return "\n\n".join(context_parts) if context_parts else None

    def _build_prompt(
        self,
        user_message: str,
        context: Optional[str],
        include_web_search: bool
    ) -> str:
        """Build the full prompt with context and history."""
        prompt_parts = []

        # Add context if available
        if context:
            prompt_parts.append("ANALYSIS REPORT CONTEXT:\n")
            prompt_parts.append(context)
            prompt_parts.append("\n" + "="*80 + "\n")

        # Add conversation history if exists
        if len(self.conversation_history) > 1:
            prompt_parts.append("CONVERSATION HISTORY:\n")
            for msg in self.conversation_history[-6:]:  # Last 3 exchanges
                role = msg["role"].upper()
                content = msg["content"][:200]  # Truncate for context
                prompt_parts.append(f"{role}: {content}\n")
            prompt_parts.append("\n" + "="*80 + "\n")

        # Add web search instruction if requested
        if include_web_search:
            prompt_parts.append(
                "USER HAS REQUESTED WEB SEARCH: Please search for latest market information "
                "or news to supplement your answer if needed.\n\n"
            )

        # Add the current question
        prompt_parts.append(f"USER QUESTION:\n{user_message}\n\n")

        # Add reminder
        prompt_parts.append(
            "Please answer the question based on the analysis report context provided above. "
            "Be specific and reference the report data. Include a disclaimer that this is not financial advice."
        )

        return "".join(prompt_parts)

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return self.conversation_history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print(f"[{self.AGENT_NAME}] Conversation history cleared")

    def cleanup(self):
        """Clean up assistant resources."""
        if self.assistant:
            try:
                self.client.beta.assistants.delete(self.assistant.id)
                self.assistant = None
            except Exception as e:
                print(f"Warning: Failed to delete assistant: {e}")
