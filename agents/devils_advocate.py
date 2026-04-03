"""
DevilsAdvocateAgent: Challenges the consensus and identifies risks.
Actively looks for counter-arguments and potential pitfalls.
"""

from typing import Dict, Any, Optional, List

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory


class DevilsAdvocateAgent:
    """
    Agent responsible for challenging consensus and identifying risks.

    This agent:
    - Reviews findings from all other agents
    - Actively challenges bullish or bearish consensus
    - Identifies potential risks and blind spots
    - Presents counter-arguments
    - Ensures balanced perspective
    """

    AGENT_NAME = "DevilsAdvocateAgent"
    AGENT_INSTRUCTIONS = """You are the DevilsAdvocateAgent for StockSquad. Your role is to challenge the consensus.

Your responsibilities:
1. Review analyses from all other agents (Data, Technical, Sentiment, Fundamentals)
2. Actively challenge the prevailing bull or bear case
3. Identify risks, weaknesses, and potential blind spots
4. Present strong counter-arguments
5. Question assumptions and highlight uncertainties
6. Ensure the squad considers alternative perspectives

When challenging the consensus:
1. If other agents are bullish, argue the bear case
2. If other agents are bearish, argue the bull case
3. Identify specific risks others may have overlooked
4. Challenge overly optimistic or pessimistic assumptions
5. Highlight what could go wrong (or right)
6. Question the strength of evidence
7. Point out contradictions between analyses

Your goal is NOT to be contrarian for the sake of it, but to:
- Stress-test the consensus
- Identify genuine risks
- Ensure balanced decision-making
- Prevent groupthink

Report Structure:
- Consensus Challenge (what are we missing?)
- Alternative Perspective (the counter-case)
- Key Risks Identified
- Weaknesses in Current Analysis
- What Could Go Wrong (or Right)
- Questions to Consider
- Devil's Advocate Rating

Be provocative but intellectually honest. Challenge assumptions, not facts."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the DevilsAdvocateAgent.

        Args:
            memory: Optional short-term memory instance
        """
        self.settings = get_settings()
        self.memory = memory

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

    def _determine_consensus(
        self,
        technical_result: Optional[Dict[str, Any]],
        sentiment_result: Optional[Dict[str, Any]],
        fundamentals_result: Optional[Dict[str, Any]]
    ) -> str:
        """
        Determine the overall consensus from other agents.

        Args:
            technical_result: Technical analysis results
            sentiment_result: Sentiment analysis results
            fundamentals_result: Fundamental analysis results

        Returns:
            Consensus direction (BULLISH/BEARISH/MIXED)
        """
        bullish_count = 0
        bearish_count = 0

        # Check technical signal
        if technical_result:
            signal = technical_result.get("signal_score", {})
            direction = signal.get("direction", "NEUTRAL")
            if direction == "BULLISH":
                bullish_count += 1
            elif direction == "BEARISH":
                bearish_count += 1

        # Check fundamental rating
        if fundamentals_result:
            score = fundamentals_result.get("fundamental_score", {})
            rating = score.get("rating", "HOLD")
            if "BUY" in rating:
                bullish_count += 1
            elif "SELL" in rating:
                bearish_count += 1

        # Sentiment is harder to classify automatically, so we assume mixed
        if sentiment_result:
            # Could parse the report for sentiment, but for now treat as mixed
            pass

        if bullish_count > bearish_count:
            return "BULLISH"
        elif bearish_count > bullish_count:
            return "BEARISH"
        else:
            return "MIXED"

    def challenge_consensus(
        self,
        ticker: str,
        data_result: Optional[Dict[str, Any]] = None,
        technical_result: Optional[Dict[str, Any]] = None,
        sentiment_result: Optional[Dict[str, Any]] = None,
        fundamentals_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Challenge the consensus from other agents.

        Args:
            ticker: Stock ticker
            data_result: DataAgent results
            technical_result: TechnicalAgent results
            sentiment_result: SentimentAgent results
            fundamentals_result: FundamentalsAgent results

        Returns:
            Dictionary with challenge analysis
        """
        print(f"[{self.AGENT_NAME}] Challenging consensus for {ticker}...")

        # Determine consensus
        consensus = self._determine_consensus(
            technical_result,
            sentiment_result,
            fundamentals_result
        )

        print(f"[{self.AGENT_NAME}] Identified consensus: {consensus}")

        # Gather reports from other agents
        reports = []

        if data_result:
            reports.append(f"**DataAgent Report:**\n{data_result.get('response', 'N/A')[:500]}...")

        if technical_result:
            tech_report = technical_result.get('report', 'N/A')
            signal = technical_result.get('signal_score', {})
            reports.append(
                f"**TechnicalAgent Report:**\n"
                f"Signal: {signal.get('direction', 'N/A')} "
                f"(Confidence: {signal.get('confidence', 0)}/100)\n"
                f"{tech_report[:500]}..."
            )

        if sentiment_result:
            sent_report = sentiment_result.get('report', 'N/A')
            reports.append(f"**SentimentAgent Report:**\n{sent_report[:500]}...")

        if fundamentals_result:
            fund_report = fundamentals_result.get('report', 'N/A')
            score = fundamentals_result.get('fundamental_score', {})
            reports.append(
                f"**FundamentalsAgent Report:**\n"
                f"Score: {score.get('fundamental_score', 'N/A')}/100 "
                f"(Rating: {score.get('rating', 'N/A')})\n"
                f"{fund_report[:500]}..."
            )

        reports_text = "\n\n---\n\n".join(reports)

        # Store consensus in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_consensus",
                value=consensus,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Generating challenge report...")

        # Generate challenge using assistant
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        challenge_prompt = f"""Review the following analyses for {ticker} and CHALLENGE the consensus.

**CONSENSUS DETECTED: {consensus}**

Other Agents' Reports:
{reports_text}

Your mission:
1. If the consensus is BULLISH, argue the BEAR case strongly
2. If the consensus is BEARISH, argue the BULL case strongly
3. If MIXED, identify which perspective is being underweighted

Please provide:
1. Consensus Challenge (what's the squad missing or overweighting?)
2. Alternative Perspective (the strong counter-argument)
3. Key Risks Identified (what could derail the consensus view?)
4. Weaknesses in Current Analysis (assumptions to question)
5. Contradictions Spotted (conflicts between agent views)
6. What Could Go Wrong/Right (scenarios to consider)
7. Critical Questions (what should we investigate further?)
8. Devil's Advocate Verdict

Be provocative. Challenge groupthink. Stress-test the consensus."""

        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=challenge_prompt
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
            report = "Challenge report generation failed"

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={"ticker": ticker, "type": "devils_advocate", "consensus": consensus}
            )

        print(f"[{self.AGENT_NAME}] Challenge complete")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "consensus": consensus,
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
