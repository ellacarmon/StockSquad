"""
FundamentalsAgent: Analyzes financial fundamentals and key ratios.
Evaluates valuation, profitability, growth, and financial health.
"""

from typing import Dict, Any, Optional
import json

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory


class FundamentalsAgent:
    """
    Agent responsible for fundamental analysis and financial ratio assessment.

    This agent:
    - Analyzes financial statements and ratios
    - Evaluates valuation metrics (P/E, PEG, P/B, EV/EBITDA)
    - Assesses profitability (margins, ROE, ROA)
    - Reviews growth metrics (revenue, earnings)
    - Examines financial health (debt, cash flow)
    """

    AGENT_NAME = "FundamentalsAgent"
    AGENT_INSTRUCTIONS = """You are the FundamentalsAgent for StockSquad, specializing in fundamental analysis.

Your responsibilities:
1. Analyze financial statements and key ratios
2. Assess valuation (is the stock overvalued, undervalued, or fairly valued?)
3. Evaluate profitability and operating efficiency
4. Review growth trends and sustainability
5. Examine financial health and balance sheet strength
6. Compare metrics to industry standards where possible

When analyzing fundamentals:
1. Review all provided financial metrics
2. Assess valuation in context (not just absolute numbers)
3. Look for trends (improving/deteriorating metrics)
4. Identify strengths and weaknesses
5. Consider industry context
6. Flag any concerning metrics

Report Structure:
- Fundamental Summary (2-3 sentences)
- Valuation Assessment
- Profitability Analysis
- Growth Profile
- Financial Health (debt, cash, liquidity)
- Key Strengths
- Key Concerns
- Fundamental Rating

Provide objective, numbers-based analysis. Explain what the metrics mean in plain language."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the FundamentalsAgent.

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

    def _calculate_fundamental_score(self, financials: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate a composite fundamental score.

        Args:
            financials: Financial metrics dictionary

        Returns:
            Dictionary with scoring results
        """
        score = 50  # Start neutral
        signals = []

        # Valuation scoring
        valuation = financials.get("valuation", {})
        pe_ratio = valuation.get("pe_ratio")
        peg_ratio = valuation.get("peg_ratio")

        if pe_ratio:
            if pe_ratio < 15:
                score += 10
                signals.append(f"Attractive P/E ratio ({pe_ratio:.1f})")
            elif pe_ratio > 30:
                score -= 10
                signals.append(f"High P/E ratio ({pe_ratio:.1f})")

        if peg_ratio:
            if peg_ratio < 1:
                score += 10
                signals.append(f"Strong PEG ratio ({peg_ratio:.2f})")
            elif peg_ratio > 2:
                score -= 5
                signals.append(f"Elevated PEG ratio ({peg_ratio:.2f})")

        # Profitability scoring
        profitability = financials.get("profitability", {})
        profit_margin = profitability.get("profit_margin")
        roe = profitability.get("roe")

        if profit_margin:
            if profit_margin > 0.20:
                score += 10
                signals.append(f"Strong profit margin ({profit_margin*100:.1f}%)")
            elif profit_margin < 0.05:
                score -= 10
                signals.append(f"Weak profit margin ({profit_margin*100:.1f}%)")

        if roe:
            if roe > 0.15:
                score += 10
                signals.append(f"Healthy ROE ({roe*100:.1f}%)")
            elif roe < 0.05:
                score -= 5
                signals.append(f"Low ROE ({roe*100:.1f}%)")

        # Growth scoring
        growth = financials.get("growth", {})
        revenue_growth = growth.get("revenue_growth")
        earnings_growth = growth.get("earnings_growth")

        if revenue_growth:
            if revenue_growth > 0.15:
                score += 10
                signals.append(f"Strong revenue growth ({revenue_growth*100:.1f}%)")
            elif revenue_growth < 0:
                score -= 10
                signals.append(f"Declining revenue ({revenue_growth*100:.1f}%)")

        if earnings_growth:
            if earnings_growth > 0.15:
                score += 5
                signals.append(f"Strong earnings growth ({earnings_growth*100:.1f}%)")
            elif earnings_growth < 0:
                score -= 5
                signals.append(f"Declining earnings ({earnings_growth*100:.1f}%)")

        # Financial health scoring
        health = financials.get("financial_health", {})
        debt_to_equity = health.get("debt_to_equity")
        current_ratio = health.get("current_ratio")

        if debt_to_equity is not None:
            if debt_to_equity < 50:
                score += 5
                signals.append(f"Low debt levels (D/E: {debt_to_equity:.1f})")
            elif debt_to_equity > 150:
                score -= 10
                signals.append(f"High debt levels (D/E: {debt_to_equity:.1f})")

        if current_ratio:
            if current_ratio > 1.5:
                score += 5
                signals.append(f"Strong liquidity (CR: {current_ratio:.2f})")
            elif current_ratio < 1:
                score -= 10
                signals.append(f"Liquidity concerns (CR: {current_ratio:.2f})")

        # Normalize to 0-100
        score = max(0, min(100, score))

        # Rating
        if score >= 80:
            rating = "STRONG BUY"
        elif score >= 65:
            rating = "BUY"
        elif score >= 35:
            rating = "HOLD"
        elif score >= 20:
            rating = "SELL"
        else:
            rating = "STRONG SELL"

        return {
            "fundamental_score": score,
            "rating": rating,
            "signals": signals
        }

    def analyze_fundamentals(
        self,
        ticker: str,
        financials: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze fundamental financial data.

        Args:
            ticker: Stock ticker
            financials: Financial metrics dictionary

        Returns:
            Dictionary with fundamental analysis
        """
        print(f"[{self.AGENT_NAME}] Analyzing fundamentals for {ticker}...")

        # Calculate fundamental score
        fundamental_score = self._calculate_fundamental_score(financials)

        # Format financials for display
        financials_formatted = json.dumps(financials, indent=2)

        # Store in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_fundamental_score",
                value=fundamental_score,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Generating fundamental report...")

        # Generate report using assistant
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        analysis_prompt = f"""Please analyze the fundamental financial data for {ticker}.

Financial Metrics:
{financials_formatted}

Fundamental Score:
- Score: {fundamental_score['fundamental_score']}/100
- Rating: {fundamental_score['rating']}
- Key Signals:
{chr(10).join(['  • ' + s for s in fundamental_score['signals']])}

Please provide:
1. Fundamental Summary (overall financial health)
2. Valuation Assessment (overvalued/undervalued/fair)
3. Profitability Analysis (margins, returns)
4. Growth Profile (revenue and earnings trends)
5. Financial Health (balance sheet, debt, liquidity)
6. Key Strengths (what's working well)
7. Key Concerns (red flags or weaknesses)
8. Fundamental Rating and Rationale

Be specific with numbers and provide context. Explain what metrics mean for investors."""

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
            report = "Fundamental analysis report generation failed"

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={"ticker": ticker, "type": "fundamental_analysis"}
            )

        print(f"[{self.AGENT_NAME}] Fundamental analysis complete")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "fundamental_score": fundamental_score,
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
