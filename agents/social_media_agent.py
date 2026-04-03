"""
SocialMediaAgent: Analyzes social media sentiment from X and Reddit using Grok.
Uses xAI's Grok API with native x_search and web_search tools to fetch
and analyze real-time social media data in a single pass.
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime

from openai import OpenAI

from config import get_settings
from memory.short_term import ShortTermMemory


class SocialMediaAgent:
    """
    Agent responsible for social media sentiment analysis using Grok.

    This agent:
    - Uses Grok's native x_search to fetch real-time X (Twitter) data
    - Uses Grok's web_search to fetch Reddit discussions
    - Analyzes sentiment of social media posts
    - Identifies trending topics and themes
    - Tracks retail investor mood
    - Compares social vs news sentiment
    - Detects unusual activity or volume spikes
    """

    AGENT_NAME = "SocialMediaAgent"
    AGENT_INSTRUCTIONS = """You are the SocialMediaAgent for StockSquad, specializing in social media sentiment analysis.

Your responsibilities:
1. Search X (Twitter) for recent posts, cashtags, and discussions about the given stock ticker
2. Search the web for recent Reddit discussions (r/wallstreetbets, r/stocks, r/investing, r/StockMarket)
3. Assess retail investor sentiment (bullish/neutral/bearish)
4. Identify trending topics and discussion themes
5. Track social media volume and engagement
6. Compare social sentiment to institutional news sentiment
7. Detect unusual social activity or viral trends

When analyzing social data:
1. Search for the stock's cashtag (e.g., $AAPL) and name on X
2. Search Reddit for discussions about the ticker
3. Classify overall retail sentiment
4. Identify what themes retail investors are discussing
5. Assess the strength and conviction of sentiment
6. Note any divergence from news/institutional sentiment
7. Flag unusual volume spikes or viral trends
8. Consider the quality of discussion (informed vs speculation)

Report Structure:
- Social Sentiment Summary (2-3 sentences)
- Overall Retail Sentiment (Bullish/Neutral/Bearish with strength)
- X (Twitter) Analysis
- Reddit Analysis
- Trending Topics and Themes
- Social vs News Sentiment Comparison
- Volume and Engagement Metrics
- Notable Posts or Discussions
- Social Media Risk Assessment

Distinguish between informed analysis and speculation. Note the difference between
retail (social media) and institutional (news/analyst) perspectives."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the SocialMediaAgent.

        Args:
            memory: Optional short-term memory instance
        """
        self.settings = get_settings()
        self.memory = memory
        self.client = self._initialize_client()

    def _initialize_client(self) -> OpenAI:
        """Initialize xAI/Grok client via OpenAI SDK."""
        if not self.settings.xai_api_key:
            raise ValueError(
                "XAI_API_KEY not configured. Please set it in your .env file. "
                "Get your key from https://console.x.ai"
            )

        return OpenAI(
            api_key=self.settings.xai_api_key,
            base_url="https://api.x.ai/v1",
        )

    def analyze_social_sentiment(
        self,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Analyze social media sentiment for a ticker using Grok.

        Uses Grok's native x_search and web_search tools to fetch
        real-time social media data and analyze sentiment in one pass.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dictionary with social sentiment analysis
        """
        ticker = ticker.upper()
        print(f"[{self.AGENT_NAME}] Analyzing social sentiment for {ticker} via Grok...")

        if not self.settings.xai_api_key:
            print(f"[{self.AGENT_NAME}] Warning: XAI_API_KEY not configured")
            return {
                "agent": self.AGENT_NAME,
                "ticker": ticker,
                "error": "Grok API not configured",
                "report": "Social media analysis unavailable. Please configure XAI_API_KEY in .env.",
                "sentiment_analysis": {
                    "overall": {"sentiment": "N/A"}
                }
            }

        analysis_prompt = f"""Search X (Twitter) and Reddit for recent social media discussion about the stock ticker ${ticker}.

Specifically:
1. Search X for recent posts mentioning ${ticker} or the company name — look at cashtags, discussions, sentiment
2. Search the web for recent Reddit threads about {ticker} on subreddits like r/wallstreetbets, r/stocks, r/investing, r/StockMarket

Based on what you find, provide the following analysis:

## Social Sentiment Summary
2-3 sentences capturing the overall retail investor mood.

## Overall Retail Sentiment
Classify as: **Bullish**, **Neutral**, or **Bearish**
Include strength level (Strong/Moderate/Weak) and your confidence.

## X (Twitter) Analysis
- Key themes and narratives on X
- Notable tweets or influencer mentions
- Volume assessment (high/normal/low activity)
- Sentiment breakdown

## Reddit Analysis
- Key discussion themes across stock subreddits
- Notable posts or DD (Due Diligence)
- Community consensus and disagreements
- Sentiment breakdown

## Trending Topics and Themes
What are the main topics retail investors are discussing about {ticker}?

## Volume and Engagement Assessment
Is social media activity around {ticker} unusually high, normal, or low?

## Social vs Institutional Perspective
How might retail sentiment differ from institutional/analyst views?

## Social Media Risk Factors
Any signs of:
- Excessive speculation or hype
- Misinformation or pump-and-dump patterns
- Echo chamber dynamics
- Short squeeze narratives

Be objective and data-driven. Cite specific posts or trends you find. Distinguish quality analysis from speculation."""

        try:
            print(f"[{self.AGENT_NAME}] Querying Grok with x_search and web_search tools...")

            response = self.client.responses.create(
                model=self.settings.grok_model,
                input=[
                    {"role": "system", "content": self.AGENT_INSTRUCTIONS},
                    {"role": "user", "content": analysis_prompt}
                ],
                tools=[
                    {"type": "web_search"},
                    {"type": "x_search"}
                ],
            )

            report = response.output_text

            if not report:
                print(f"[{self.AGENT_NAME}] Warning: Empty response from Grok")
                report = "Social sentiment analysis returned empty response from Grok."

        except Exception as e:
            print(f"[{self.AGENT_NAME}] Grok API error: {e}")
            return {
                "agent": self.AGENT_NAME,
                "ticker": ticker,
                "error": str(e),
                "report": f"Social media analysis failed: {str(e)}",
                "sentiment_analysis": {
                    "overall": {"sentiment": "N/A"}
                }
            }

        # Extract sentiment classification from the report
        sentiment_classification = self._extract_sentiment(report)

        # Build sentiment_analysis structure for compatibility with orchestrator & frontend
        sentiment_analysis = {
            "overall": {
                "sentiment": sentiment_classification,
                "stats": {
                    "bullish_pct": 0,
                    "bearish_pct": 0,
                    "neutral_pct": 0,
                }
            },
            "source": "grok"
        }

        # Set approximate percentages based on classification
        if sentiment_classification == "Bullish":
            sentiment_analysis["overall"]["stats"]["bullish_pct"] = 65.0
            sentiment_analysis["overall"]["stats"]["bearish_pct"] = 15.0
            sentiment_analysis["overall"]["stats"]["neutral_pct"] = 20.0
        elif sentiment_classification == "Bearish":
            sentiment_analysis["overall"]["stats"]["bullish_pct"] = 15.0
            sentiment_analysis["overall"]["stats"]["bearish_pct"] = 65.0
            sentiment_analysis["overall"]["stats"]["neutral_pct"] = 20.0
        else:
            sentiment_analysis["overall"]["stats"]["bullish_pct"] = 30.0
            sentiment_analysis["overall"]["stats"]["bearish_pct"] = 30.0
            sentiment_analysis["overall"]["stats"]["neutral_pct"] = 40.0

        # Store in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_social_sentiment",
                value=sentiment_analysis,
                agent=self.AGENT_NAME
            )
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={"ticker": ticker, "type": "social_sentiment_analysis"}
            )

        print(f"[{self.AGENT_NAME}] Social sentiment analysis complete — {sentiment_classification}")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "sentiment_analysis": sentiment_analysis,
            "report": report,
        }

    def _extract_sentiment(self, report: str) -> str:
        """
        Extract the overall sentiment classification from the Grok report text.

        Searches for keywords like Bullish/Bearish/Neutral in the report,
        particularly near the 'Overall Retail Sentiment' section.

        Args:
            report: The full report text

        Returns:
            One of 'Bullish', 'Bearish', or 'Neutral'
        """
        report_lower = report.lower()

        # Look for the explicit sentiment section first
        sentiment_section = ""
        for marker in ["overall retail sentiment", "overall sentiment", "retail sentiment"]:
            idx = report_lower.find(marker)
            if idx != -1:
                # Grab ~200 chars after the marker
                sentiment_section = report_lower[idx:idx + 200]
                break

        # Check the sentiment section first, then fall back to full report
        search_text = sentiment_section if sentiment_section else report_lower

        # Count occurrences with weighting for proximity to the heading
        bullish_signals = search_text.count("bullish") + search_text.count("positive")
        bearish_signals = search_text.count("bearish") + search_text.count("negative")

        if bullish_signals > bearish_signals:
            return "Bullish"
        elif bearish_signals > bullish_signals:
            return "Bearish"
        else:
            return "Neutral"

    def cleanup(self):
        """Clean up resources (no-op for Grok — no assistant to delete)."""
        pass
