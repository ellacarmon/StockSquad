"""
DataAgent: Fetches market data and financial information for stocks.
Uses yfinance to retrieve OHLCV data, financials, and company information.
"""

from typing import Dict, Any, Optional
import json

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from tools.market_data import MarketDataFetcher
from memory.short_term import ShortTermMemory


class DataAgent:
    """
    Agent responsible for fetching and formatting market data.

    This agent:
    - Retrieves price history, volume, and financial statements
    - Fetches company information and recent news
    - Formats data for analysis by other agents
    - Posts findings to shared memory
    """

    AGENT_NAME = "DataAgent"
    AGENT_INSTRUCTIONS = """You are the DataAgent for StockSquad, a specialized agent responsible for gathering market data.

Your responsibilities:
1. Fetch comprehensive stock data including price history, volume, and financial metrics
2. Retrieve company information, sector, and industry details
3. Gather recent news articles related to the stock
4. Format all data clearly and concisely for other agents

When asked to analyze a ticker:
1. Identify 2-3 major industry peers/competitors for the ticker.
2. Use get_complete_stock_data to fetch all available information, making sure to pass the peers into the comparison_tickers argument.
3. Summarize the key findings
4. Highlight any notable trends or metrics
5. Return a structured report

Always provide accurate, factual data. If data is unavailable, clearly state this."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the DataAgent.

        Args:
            memory: Optional short-term memory instance for sharing data
        """
        self.settings = get_settings()
        self.memory = memory
        self.market_data = MarketDataFetcher()

        # Initialize Azure OpenAI Client with DefaultAzureCredential
        self.client = self._initialize_client()

        # Define tools available to this agent
        self.tools = self._define_tools()

        # Assistant will be created when needed
        self.assistant = None

    def _initialize_client(self) -> AzureOpenAI:
        """Initialize Azure OpenAI client with DefaultAzureCredential."""
        # Get token provider for Azure OpenAI scope
        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        return AzureOpenAI(
            api_version=self.settings.azure_openai_api_version,
            azure_endpoint=self.settings.azure_openai_endpoint,
            azure_ad_token_provider=token_provider,
        )

    def _define_tools(self) -> list:
        """
        Define the function tools available to this agent.

        Returns:
            List of FunctionTool definitions
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_complete_stock_data",
                    "description": "Fetch comprehensive stock data including price history, financials, company info, and recent news",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol (e.g., AAPL, MSFT, NVDA)"
                            },
                            "period": {
                                "type": "string",
                                "description": "Historical data period",
                                "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                                "default": "1y"
                            },
                            "comparison_tickers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of 2-3 major industry peers to compare against"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_info",
                    "description": "Get basic company information and current price",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_financials",
                    "description": "Get detailed financial metrics and ratios",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Stock ticker symbol"
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            }
        ]

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool function.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments

        Returns:
            JSON string with results
        """
        try:
            if tool_name == "get_complete_stock_data":
                ticker = arguments["ticker"]
                period = arguments.get("period", "1y")
                comparison_tickers = arguments.get("comparison_tickers", [])

                # Try to fetch data
                try:
                    result = self.market_data.get_complete_stock_data(ticker, period, comparison_tickers)
                except ValueError as e:
                    # Data fetch failed completely
                    error_msg = f"❌ No price data for {ticker}"
                    print(f"[{self.AGENT_NAME}] {error_msg}: {str(e)}")
                    return json.dumps({
                        "error": error_msg,
                        "ticker": ticker,
                        "data_points": 0,
                        "message": "Unable to fetch price history. Ticker may be invalid, delisted, or not supported.",
                        "details": str(e)
                    })

                # Validate we got actual price data
                price_hist = result.get("price_history", {})
                data_points = price_hist.get("data_points", 0) if isinstance(price_hist, dict) else 0

                if data_points < 20:
                    error_msg = f"❌ Insufficient price data for {ticker}. Cannot perform analysis."
                    print(f"[{self.AGENT_NAME}] {error_msg}")
                    return json.dumps({
                        "error": error_msg,
                        "ticker": ticker,
                        "data_points": data_points,
                        "message": "This ticker may be delisted, invalid, or have insufficient trading history."
                    })

                # Post to scratchpad if memory is available
                if self.memory:
                    self.memory.post_to_scratchpad(
                        key=f"{ticker}_data",
                        value=result,
                        agent=self.AGENT_NAME
                    )

                return json.dumps(result)

            elif tool_name == "get_stock_info":
                ticker = arguments["ticker"]
                result = self.market_data.get_stock_info(ticker)
                return json.dumps(result)

            elif tool_name == "get_financials":
                ticker = arguments["ticker"]
                result = self.market_data.get_financials(ticker)
                return json.dumps(result)

            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    def create_assistant(self):
        """Create the Azure OpenAI assistant if not already created."""
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create(
                model=self.settings.azure_openai_deployment_name,
                name=self.AGENT_NAME,
                instructions=self.AGENT_INSTRUCTIONS,
                tools=self.tools,
            )
        return self.assistant

    def analyze_ticker(self, ticker: str, period: str = "1y") -> Dict[str, Any]:
        """
        Analyze a stock ticker and return comprehensive data.

        Args:
            ticker: Stock ticker symbol
            period: Historical data period

        Returns:
            Dictionary containing analysis results
        """
        # Create assistant if needed
        assistant = self.create_assistant()

        # Create a thread for this analysis
        thread = self.client.beta.threads.create()

        # Send message to assistant
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"Please fetch and analyze complete stock data for {ticker} with {period} of price history."
        )

        # Run the assistant
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Wait for completion and handle tool calls with timeout
        from agents.assistant_utils import wait_for_run_with_actions, AssistantTimeoutError
        try:
            run = wait_for_run_with_actions(
                client=self.client,
                thread_id=thread.id,
                run_id=run.id,
                tool_executor=self._execute_tool,
                timeout=180  # 3 minutes max
            )
        except AssistantTimeoutError as e:
            print(f"[{self.AGENT_NAME}] Timeout: {e}")
            return {"error": "Analysis timed out after 3 minutes"}

        # Get the assistant's response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)

        # Find the assistant's response
        assistant_messages = [
            msg for msg in messages.data
            if msg.role == "assistant" and msg.created_at > message.created_at
        ]

        if assistant_messages:
            response = assistant_messages[0].content[0].text.value
        else:
            response = "No response from assistant"

        # Get the data from memory if available
        stock_data = None
        if self.memory:
            stock_data = self.memory.get_from_scratchpad(f"{ticker}_data")

        # Add message to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=response,
                metadata={"ticker": ticker, "period": period}
            )

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "response": response,
            "stock_data": stock_data,
            "thread_id": thread.id,
            "run_id": run.id,
        }

    def cleanup(self):
        """Clean up assistant resources."""
        if self.assistant:
            try:
                self.client.beta.assistants.delete(self.assistant.id)
                self.assistant = None
            except Exception as e:
                print(f"Warning: Failed to delete assistant: {e}")
