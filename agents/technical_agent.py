"""
TechnicalAgent: Performs technical analysis with indicators and ML signal scoring.
Analyzes price patterns, momentum, and generates trading signals.
"""

from typing import Dict, Any, Optional
import json
import pandas as pd

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from tools.ta_indicators import TechnicalIndicators
from ml.signal_model import SignalScorer
from memory.short_term import ShortTermMemory


class TechnicalAgent:
    """
    Agent responsible for technical analysis and signal generation.

    This agent:
    - Calculates technical indicators (RSI, MACD, MAs)
    - Generates ML-based signal scores
    - Identifies trends and patterns
    - Provides trading recommendations
    """

    AGENT_NAME = "TechnicalAgent"
    AGENT_INSTRUCTIONS = """You are the TechnicalAgent for StockSquad, specializing in technical analysis.

Your responsibilities:
1. Analyze technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
2. Interpret ML-based signal scores and confidence levels
3. Identify trends, support/resistance levels, and momentum
4. Provide clear, actionable trading signals
5. Explain technical patterns in plain language

When analyzing data:
1. Review all provided technical indicators
2. Assess the ML signal score and its confidence
3. Consider multiple timeframes (short, medium, long-term)
4. Identify key support and resistance levels
5. Evaluate volume confirmation
6. Provide a balanced technical perspective

Report Structure:
- Technical Summary (2-3 sentences)
- Trend Analysis
- Momentum Indicators (RSI, MACD)
- Moving Average Analysis
- Volume Analysis
- ML Signal Score Interpretation
- Key Levels (support/resistance)
- Technical Recommendation

Always base your analysis on the data. Be objective and highlight both bullish and bearish signals."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the TechnicalAgent.

        Args:
            memory: Optional short-term memory instance
        """
        self.settings = get_settings()
        self.memory = memory
        self.ta_calculator = TechnicalIndicators()
        self.signal_scorer = SignalScorer()

        # Initialize Azure OpenAI Client
        self.client = self._initialize_client()
        self.assistant = None

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
            self.assistant = self.client.beta.assistants.create(
                model=self.settings.azure_openai_deployment_name,
                name=self.AGENT_NAME,
                instructions=self.AGENT_INSTRUCTIONS,
            )
        return self.assistant

    def analyze_technicals(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        sentiment_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform technical analysis on price data.

        Args:
            ticker: Stock ticker
            price_data: DataFrame with OHLCV data
            sentiment_result: Optional SentimentAgent output to feed into the ML model

        Returns:
            Dictionary with technical analysis results
        """
        print(f"[{self.AGENT_NAME}] Calculating technical indicators for {ticker}...")

        # Calculate all technical indicators
        indicators = self.ta_calculator.calculate_all_indicators(price_data)

        # Generate ML signal score
        signal_score = self.signal_scorer.score_signal(indicators, sentiment_result=sentiment_result)

        # Format for display
        ta_formatted = self.ta_calculator.format_for_llm(indicators)
        signal_formatted = self.signal_scorer.format_for_llm(signal_score)

        # Store in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_technical_indicators",
                value=indicators,
                agent=self.AGENT_NAME
            )
            self.memory.post_to_scratchpad(
                key=f"{ticker}_signal_score",
                value=signal_score,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Technical analysis complete")

        # Generate report using assistant
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        analysis_prompt = f"""Please analyze the following technical data for {ticker} and provide a comprehensive technical analysis report.

{ta_formatted}

{signal_formatted}

Please provide:
1. Technical Summary (what the charts are telling us)
2. Trend Analysis (short, medium, long-term)
3. Momentum Assessment (RSI and MACD interpretation)
4. Moving Average Analysis and price position
5. Volume Analysis
6. ML Signal Score Interpretation
7. Key Support and Resistance Levels
8. Technical Recommendation (with rationale)

Be specific, objective, and actionable."""

        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=analysis_prompt
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id,
        )

        # Wait for completion with timeout
        from agents.assistant_utils import wait_for_run_completion, AssistantTimeoutError
        try:
            run = wait_for_run_completion(
                client=self.client,
                thread_id=thread.id,
                run_id=run.id,
                timeout=180  # 3 minutes max
            )
        except AssistantTimeoutError as e:
            print(f"[{self.AGENT_NAME}] Timeout: {e}")
            return None

        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        assistant_messages = [
            msg for msg in messages.data
            if msg.role == "assistant" and msg.created_at > message.created_at
        ]

        if assistant_messages:
            report = assistant_messages[0].content[0].text.value
        else:
            report = "Technical analysis report generation failed"

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={"ticker": ticker, "type": "technical_analysis"}
            )

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "indicators": indicators,
            "signal_score": signal_score,
            "report": report,
            "thread_id": thread.id,
        }

    def cleanup(self):
        """Clean up assistant resources."""
        if self.assistant:
            try:
                self.client.beta.assistants.delete(self.assistant.id)
                self.assistant = None
            except Exception as e:
                print(f"Warning: Failed to delete assistant: {e}")
