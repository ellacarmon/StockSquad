"""
SentimentAgent V2: Enhanced news sentiment analysis using FinGPT-style instruction formatting.

References:
- FinGPT: Open-Source Financial Large Language Models (2023)
- arXiv:2306.06031
- Structured prompt engineering for financial news parsing
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory


class SentimentAgentV2:
    """
    Enhanced sentiment agent using FinGPT-inspired structured prompting.

    Key improvements:
    - Separates news by category (Macro, Company, Industry, Earnings)
    - Structured JSON output format
    - Explicit sentiment scoring (0-100 scale)
    - Impact classification (High/Medium/Low)
    - Time-based news prioritization
    """

    AGENT_NAME = "SentimentAgent"

    # FinGPT-style instruction template
    AGENT_INSTRUCTIONS = """You are a financial news sentiment analyst specializing in structured analysis.

Your task is to analyze news articles and produce a structured JSON output with the following fields:

1. overall_sentiment: Object with:
   - direction: "bullish", "neutral", or "bearish"
   - score: 0-100 (0=very bearish, 50=neutral, 100=very bullish)
   - confidence: 0-100 (how confident in this assessment)

2. news_categories: Object categorizing news into:
   - macro_news: News about broader economy, Fed policy, inflation, etc.
   - company_specific: News directly about the company (earnings, products, management)
   - industry_trends: News about the sector/competitors
   - earnings_revenue: Specific financial results and guidance

3. key_drivers: Array of objects, each with:
   - theme: Category name (e.g., "Revenue Growth", "Regulatory Risk")
   - sentiment: "positive", "neutral", or "negative"
   - impact: "high", "medium", or "low"
   - evidence: Brief quote or summary from news

4. sentiment_shift: Object with:
   - trend: "improving", "stable", or "deteriorating"
   - previous_period_comparison: Description of how sentiment has changed
   - momentum: "strong", "moderate", or "weak"

5. risk_factors: Array of strings listing identified risks
6. positive_catalysts: Array of strings listing bullish factors

Analyze objectively. Base conclusions on evidence from the news provided."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """Initialize the enhanced SentimentAgent."""
        self.settings = get_settings()
        self.memory = memory
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

    def _categorize_news(self, news_articles: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Categorize news into FinGPT-style buckets using keyword matching.

        Returns:
            Dict with categories: macro_news, company_specific, industry_trends, earnings_revenue
        """
        categories = {
            "macro_news": [],
            "company_specific": [],
            "industry_trends": [],
            "earnings_revenue": []
        }

        # Keywords for categorization
        macro_keywords = ["fed", "federal reserve", "inflation", "interest rate", "gdp", "economy",
                          "recession", "unemployment", "jobs report", "treasury", "central bank"]

        earnings_keywords = ["earnings", "revenue", "eps", "profit", "quarter", "q1", "q2", "q3", "q4",
                            "guidance", "forecast", "outlook", "beat", "miss", "sales"]

        industry_keywords = ["sector", "industry", "competitor", "market share", "peers",
                            "competition", "rival", "versus"]

        for article in news_articles:
            title = article.get("title", "").lower()

            # Categorize based on keywords (articles can belong to multiple categories)
            if any(keyword in title for keyword in earnings_keywords):
                categories["earnings_revenue"].append(article)
            elif any(keyword in title for keyword in macro_keywords):
                categories["macro_news"].append(article)
            elif any(keyword in title for keyword in industry_keywords):
                categories["industry_trends"].append(article)
            else:
                categories["company_specific"].append(article)

        return categories

    def _format_news_fingpt_style(
        self,
        ticker: str,
        news_categories: Dict[str, List[Dict]]
    ) -> str:
        """
        Format news using FinGPT-inspired structured template.

        Separates news by category with clear delimiters for better LLM parsing.
        """
        formatted_sections = []

        # Header
        formatted_sections.append(f"# News Analysis for {ticker}")
        formatted_sections.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        formatted_sections.append("")

        # Macro News Section
        formatted_sections.append("## [MACRO NEWS]")
        if news_categories["macro_news"]:
            for i, article in enumerate(news_categories["macro_news"][:5], 1):
                formatted_sections.append(f"{i}. {article.get('title', 'N/A')}")
                formatted_sections.append(f"   Publisher: {article.get('publisher', 'Unknown')}")
                formatted_sections.append(f"   Date: {article.get('published', 'N/A')}")
                formatted_sections.append("")
        else:
            formatted_sections.append("No macro news available.")
            formatted_sections.append("")

        # Company-Specific News Section
        formatted_sections.append("## [COMPANY-SPECIFIC NEWS]")
        if news_categories["company_specific"]:
            for i, article in enumerate(news_categories["company_specific"][:8], 1):
                formatted_sections.append(f"{i}. {article.get('title', 'N/A')}")
                formatted_sections.append(f"   Publisher: {article.get('publisher', 'Unknown')}")
                formatted_sections.append(f"   Date: {article.get('published', 'N/A')}")
                formatted_sections.append("")
        else:
            formatted_sections.append("No company-specific news available.")
            formatted_sections.append("")

        # Earnings & Revenue News Section
        formatted_sections.append("## [EARNINGS & REVENUE NEWS]")
        if news_categories["earnings_revenue"]:
            for i, article in enumerate(news_categories["earnings_revenue"][:5], 1):
                formatted_sections.append(f"{i}. {article.get('title', 'N/A')}")
                formatted_sections.append(f"   Publisher: {article.get('publisher', 'Unknown')}")
                formatted_sections.append(f"   Date: {article.get('published', 'N/A')}")
                formatted_sections.append("")
        else:
            formatted_sections.append("No earnings/revenue news available.")
            formatted_sections.append("")

        # Industry Trends Section
        formatted_sections.append("## [INDUSTRY TRENDS]")
        if news_categories["industry_trends"]:
            for i, article in enumerate(news_categories["industry_trends"][:5], 1):
                formatted_sections.append(f"{i}. {article.get('title', 'N/A')}")
                formatted_sections.append(f"   Publisher: {article.get('publisher', 'Unknown')}")
                formatted_sections.append(f"   Date: {article.get('published', 'N/A')}")
                formatted_sections.append("")
        else:
            formatted_sections.append("No industry trend news available.")
            formatted_sections.append("")

        return "\n".join(formatted_sections)

    def _create_structured_prompt(
        self,
        ticker: str,
        formatted_news: str,
        total_articles: int
    ) -> str:
        """
        Create FinGPT-style structured analysis prompt.
        """
        return f"""Analyze the following categorized news for {ticker} and provide a structured JSON response.

{formatted_news}

TOTAL NEWS VOLUME: {total_articles} articles

REQUIRED OUTPUT FORMAT (JSON):
{{
  "overall_sentiment": {{
    "direction": "bullish|neutral|bearish",
    "score": <0-100>,
    "confidence": <0-100>,
    "summary": "<1-2 sentence explanation>"
  }},
  "news_categories": {{
    "macro_news": {{
      "count": <number>,
      "sentiment": "positive|neutral|negative",
      "key_points": ["<point1>", "<point2>"]
    }},
    "company_specific": {{
      "count": <number>,
      "sentiment": "positive|neutral|negative",
      "key_points": ["<point1>", "<point2>"]
    }},
    "earnings_revenue": {{
      "count": <number>,
      "sentiment": "positive|neutral|negative",
      "key_points": ["<point1>", "<point2>"]
    }},
    "industry_trends": {{
      "count": <number>,
      "sentiment": "positive|neutral|negative",
      "key_points": ["<point1>", "<point2>"]
    }}
  }},
  "key_drivers": [
    {{
      "theme": "<theme name>",
      "sentiment": "positive|neutral|negative",
      "impact": "high|medium|low",
      "evidence": "<quote or summary from news>"
    }}
  ],
  "sentiment_shift": {{
    "trend": "improving|stable|deteriorating",
    "momentum": "strong|moderate|weak",
    "description": "<explanation of trend>"
  }},
  "risk_factors": ["<risk1>", "<risk2>", ...],
  "positive_catalysts": ["<catalyst1>", "<catalyst2>", ...],
  "media_attention": {{
    "level": "high|medium|low",
    "focus_areas": ["<area1>", "<area2>"]
  }}
}}

Respond ONLY with valid JSON. Be objective and evidence-based."""

    def analyze_sentiment(
        self,
        ticker: str,
        news_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment using FinGPT-inspired structured approach.

        Args:
            ticker: Stock ticker
            news_articles: List of news articles

        Returns:
            Dictionary with enhanced sentiment analysis
        """
        print(f"[{self.AGENT_NAME}] Analyzing sentiment for {ticker} (FinGPT-style)...")

        if not news_articles:
            return self._empty_sentiment_response(ticker)

        # Categorize news
        news_categories = self._categorize_news(news_articles)

        # Format news using FinGPT structure
        formatted_news = self._format_news_fingpt_style(ticker, news_categories)

        # Create structured prompt
        analysis_prompt = self._create_structured_prompt(
            ticker,
            formatted_news,
            len(news_articles)
        )

        # Store categorized news in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_news_categories",
                value=news_categories,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Generating structured sentiment analysis...")

        # Generate analysis using assistant
        assistant = self.create_assistant()
        thread = self.client.beta.threads.create()

        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=analysis_prompt
        )

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
                timeout=180
            )
        except AssistantTimeoutError as e:
            print(f"[{self.AGENT_NAME}] Timeout: {e}")
            return self._empty_sentiment_response(ticker)

        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        assistant_messages = [
            msg for msg in messages.data
            if msg.role == "assistant" and msg.created_at > message.created_at
        ]

        if not assistant_messages:
            return self._empty_sentiment_response(ticker)

        raw_response = assistant_messages[0].content[0].text.value

        # Try to parse JSON response
        structured_sentiment = self._parse_json_response(raw_response)

        # Generate readable report from structured data
        report = self._generate_report_from_structured_data(ticker, structured_sentiment)

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={
                    "ticker": ticker,
                    "type": "sentiment_analysis_v2",
                    "structured_data": structured_sentiment
                }
            )

        print(f"[{self.AGENT_NAME}] Sentiment analysis complete")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "news_count": len(news_articles),
            "news_categories": news_categories,
            "structured_sentiment": structured_sentiment,
            "report": report,
            "thread_id": thread.id,
        }

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        try:
            # Try direct parse first
            return json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Fallback: return empty structure
            print(f"[{self.AGENT_NAME}] Warning: Could not parse JSON response")
            return {}

    def _generate_report_from_structured_data(
        self,
        ticker: str,
        structured_data: Dict[str, Any]
    ) -> str:
        """Generate human-readable report from structured sentiment data."""
        if not structured_data:
            return f"Sentiment analysis for {ticker} - No structured data available"

        report_lines = [
            f"# Sentiment Analysis for {ticker}",
            "",
            "## Overall Sentiment",
        ]

        # Overall sentiment
        overall = structured_data.get("overall_sentiment", {})
        if overall:
            direction = overall.get("direction", "neutral").upper()
            score = overall.get("score", 50)
            confidence = overall.get("confidence", 50)
            summary = overall.get("summary", "")

            report_lines.extend([
                f"**Direction**: {direction} ({score}/100)",
                f"**Confidence**: {confidence}%",
                f"**Summary**: {summary}",
                ""
            ])

        # Key drivers
        key_drivers = structured_data.get("key_drivers", [])
        if key_drivers:
            report_lines.append("## Key Sentiment Drivers")
            for driver in key_drivers[:5]:
                theme = driver.get("theme", "Unknown")
                sentiment = driver.get("sentiment", "neutral")
                impact = driver.get("impact", "medium")
                evidence = driver.get("evidence", "")

                emoji = "✓" if sentiment == "positive" else "✗" if sentiment == "negative" else "○"
                report_lines.append(f"{emoji} **{theme}** ({impact} impact, {sentiment})")
                if evidence:
                    report_lines.append(f"   {evidence}")
            report_lines.append("")

        # Sentiment shift
        shift = structured_data.get("sentiment_shift", {})
        if shift:
            trend = shift.get("trend", "stable")
            momentum = shift.get("momentum", "moderate")
            description = shift.get("description", "")

            report_lines.extend([
                "## Sentiment Trend",
                f"**Trend**: {trend.capitalize()} ({momentum} momentum)",
                f"{description}",
                ""
            ])

        # Risk factors
        risks = structured_data.get("risk_factors", [])
        if risks:
            report_lines.append("## Risk Factors")
            for risk in risks[:5]:
                report_lines.append(f"- {risk}")
            report_lines.append("")

        # Positive catalysts
        catalysts = structured_data.get("positive_catalysts", [])
        if catalysts:
            report_lines.append("## Positive Catalysts")
            for catalyst in catalysts[:5]:
                report_lines.append(f"- {catalyst}")
            report_lines.append("")

        # Media attention
        media = structured_data.get("media_attention", {})
        if media:
            level = media.get("level", "medium")
            focus_areas = media.get("focus_areas", [])
            report_lines.extend([
                "## Media Attention",
                f"**Level**: {level.capitalize()}",
                f"**Focus Areas**: {', '.join(focus_areas) if focus_areas else 'General coverage'}",
                ""
            ])

        return "\n".join(report_lines)

    def _empty_sentiment_response(self, ticker: str) -> Dict[str, Any]:
        """Return empty sentiment response when no news available."""
        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "news_count": 0,
            "structured_sentiment": {},
            "report": f"No news available for sentiment analysis of {ticker}",
            "thread_id": None,
        }

    def cleanup(self):
        """Clean up assistant resources."""
        if self.assistant:
            try:
                self.client.beta.assistants.delete(self.assistant.id)
                self.assistant = None
            except Exception as e:
                print(f"Warning: Failed to delete assistant: {e}")
