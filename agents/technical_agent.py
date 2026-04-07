"""
TechnicalAgent: Performs technical analysis with indicators and ML signal scoring.
Analyzes price patterns, momentum, and generates trading signals.

Refactored to use the skills system.
"""

from typing import Dict, Any, Optional
import pandas as pd

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from agents.base import BaseAgent
from memory.short_term import ShortTermMemory
from hooks.event_bus import EventBus, AgentEvent


class TechnicalAgent(BaseAgent):
    """
    Agent responsible for technical analysis and signal generation.

    This agent:
    - Calculates technical indicators (RSI, MACD, MAs)
    - Generates ML-based signal scores
    - Identifies trends and patterns
    - Provides trading recommendations

    Uses skills: technical_indicators, ml_signals
    """

    # BaseAgent attributes
    agent_name = "TechnicalAgent"
    agent_description = "Performs technical analysis with indicators and ML signal scoring"
    required_skills = ['technical_indicators', 'ml_signals']
    optional_skills = []

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

    def __init__(self, memory: Optional[ShortTermMemory] = None, model_type: str = "xgboost"):
        """
        Initialize the TechnicalAgent.

        Args:
            memory: Optional short-term memory instance
            model_type: ML model type to use for signal scoring (e.g. 'xgboost', 'ensemble_unanimous')
        """
        # Initialize BaseAgent (loads skills)
        super().__init__(memory=memory, model_type=model_type)

        self.settings = get_settings()
        self.model_type = model_type

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
                name=self.agent_name,
                instructions=self.AGENT_INSTRUCTIONS,
            )
        return self.assistant

    def analyze(self, ticker: str, **kwargs) -> Dict[str, Any]:
        """
        Main analysis method required by BaseAgent.

        This is a wrapper around analyze_technicals for BaseAgent compatibility.

        Args:
            ticker: Stock ticker
            **kwargs: Additional parameters (price_data, sentiment_result)

        Returns:
            Dictionary with technical analysis results
        """
        price_data = kwargs.get('price_data')
        sentiment_result = kwargs.get('sentiment_result')

        if price_data is None:
            raise ValueError("price_data is required for technical analysis")

        return self.analyze_technicals(ticker, price_data, sentiment_result)

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
        self.log_analysis_start(ticker)

        print(f"[{self.agent_name}] Calculating technical indicators for {ticker}...")

        # Use skills instead of hardcoded tools
        # Calculate all technical indicators using the skill
        indicators = self.skills.technical_indicators.calculate_all_indicators(price_data)

        # Generate ML signal score using the skill
        signal_score = self.skills.ml_signals.score_signal(indicators, sentiment_result=sentiment_result)

        # Format for display
        ta_formatted = self.skills.technical_indicators.format_for_llm(indicators)
        signal_formatted = self.skills.ml_signals.format_for_llm(signal_score)

        # Store in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_technical_indicators",
                value=indicators,
                agent=self.agent_name
            )
            self.memory.post_to_scratchpad(
                key=f"{ticker}_signal_score",
                value=signal_score,
                agent=self.agent_name
            )

        print(f"[{self.agent_name}] Technical analysis complete")

        # Publish event: signal generated
        if signal_score.get('confidence', 0) > 50:  # Only publish strong signals
            EventBus.publish(
                AgentEvent.SIGNAL_GENERATED,
                source_agent=self.agent_name,
                ticker=ticker,
                data={
                    'signal': signal_score['recommendation'],
                    'direction': signal_score['direction'],
                    'confidence': signal_score['confidence'],
                    'score': signal_score['signal_score']
                }
            )

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
            error_msg = f"Timeout: {e}"
            print(f"[{self.agent_name}] {error_msg}")
            self.log_error(ticker, e)
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
                agent=self.agent_name,
                role="assistant",
                content=report,
                metadata={"ticker": ticker, "type": "technical_analysis"}
            )

        # Publish completion event
        EventBus.publish(
            AgentEvent.ANALYSIS_COMPLETE,
            source_agent=self.agent_name,
            ticker=ticker,
            data={'indicators': indicators, 'signal_score': signal_score}
        )

        self.log_analysis_complete(ticker, f"Signal: {signal_score['recommendation']}")

        return {
            "agent": self.agent_name,
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
