"""
SentimentAgent: Analyzes news sentiment and narrative trends using embeddings.
Uses FinGPT-style sectioned prompts to produce structured JSON sentiment output.
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import re

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory


class SentimentAgent:
    """
    Agent responsible for news sentiment analysis and narrative tracking.

    This agent:
    - Analyzes news headlines and articles
    - Classifies sentiment (positive, negative, neutral)
    - Identifies key themes and narratives
    - Tracks narrative shifts over time
    - Uses embeddings for semantic clustering
    """

    AGENT_NAME = "SentimentAgent"
    AGENT_INSTRUCTIONS = """You are the SentimentAgent for StockSquad, specializing in financial news sentiment analysis.

Your job is to convert grouped market news into structured JSON.

Core rules:
1. Separate macro news from company-specific news.
2. Separate earnings, guidance, revenue, and estimate-related items into "company_expected_revenue_news".
3. Use only evidence present in the supplied headlines and metadata.
4. Do not invent facts, quotes, numbers, or timelines.
5. If evidence is thin, lower confidence and say so explicitly.
6. Respond with valid JSON only.

Sentiment labels:
- overall direction: bullish, neutral, bearish
- item/category sentiment: positive, neutral, negative
- trend: improving, stable, deteriorating
- media attention: low, medium, high
- impact: high, medium, low

Keep the output concise, evidence-based, and machine-readable."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """
        Initialize the SentimentAgent.

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

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        response = self.client.embeddings.create(
            input=text,
            model=self.settings.azure_openai_embedding_deployment_name,
        )
        return response.data[0].embedding

    def _categorize_news(self, news_articles: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize news into FinGPT-style buckets.

        The categories are designed to make the prompt easier for the model to parse:
        macro news is separated from company estimate/guidance news and from general
        company headlines.
        """
        categories = {
            "macro_news": [],
            "company_expected_revenue_news": [],
            "company_specific_news": [],
            "industry_peer_news": [],
        }

        macro_keywords = [
            "fed", "federal reserve", "inflation", "interest rate", "rates",
            "gdp", "economy", "economic", "recession", "treasury", "jobs",
            "employment", "unemployment", "cpi", "ppi", "consumer confidence",
            "central bank", "tariff", "oil", "yield"
        ]
        expected_revenue_keywords = [
            "earnings", "revenue", "sales", "guidance", "forecast", "outlook",
            "estimate", "estimates", "expects", "expected", "eps", "quarter",
            "q1", "q2", "q3", "q4", "beat", "miss", "profit", "margin"
        ]
        industry_keywords = [
            "sector", "industry", "peer", "peers", "competitor", "competitors",
            "rival", "rivals", "market share", "semiconductor", "banking",
            "retail", "software", "energy", "auto", "airline"
        ]

        for article in news_articles:
            title = article.get("title", "")
            title_lower = title.lower()

            if any(keyword in title_lower for keyword in expected_revenue_keywords):
                categories["company_expected_revenue_news"].append(article)
            elif any(keyword in title_lower for keyword in macro_keywords):
                categories["macro_news"].append(article)
            elif any(keyword in title_lower for keyword in industry_keywords):
                categories["industry_peer_news"].append(article)
            else:
                categories["company_specific_news"].append(article)

        return categories

    def _format_article(self, article: Dict[str, Any], index: int) -> List[str]:
        """Format a single article in a compact, deterministic structure."""
        return [
            f"{index}. Headline: {article.get('title', 'N/A')}",
            f"   Publisher: {article.get('publisher', 'Unknown')}",
            f"   Published: {article.get('published', 'N/A')}",
        ]

    def _format_news_fingpt_style(
        self,
        ticker: str,
        news_categories: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        """
        Format news with explicit section labels inspired by FinGPT prompt style.
        """
        sections = [
            f"[Ticker]: {ticker}",
            f"[Analysis Date]: {datetime.now().strftime('%Y-%m-%d')}",
            "",
        ]

        category_labels = {
            "macro_news": "Macro News",
            "company_expected_revenue_news": "Company Expected Revenue News",
            "company_specific_news": "Company-Specific News",
            "industry_peer_news": "Industry and Peer News",
        }

        for category_key, label in category_labels.items():
            sections.append(f"[{label}]")
            articles = news_categories.get(category_key, [])

            if not articles:
                sections.append("None")
                sections.append("")
                continue

            for index, article in enumerate(articles[:6], 1):
                sections.extend(self._format_article(article, index))
            sections.append("")

        return "\n".join(sections)

    def _create_structured_prompt(
        self,
        ticker: str,
        formatted_news: str,
        theme_analysis: Dict[str, Any],
        total_articles: int
    ) -> str:
        """Create a structured, FinGPT-style prompt that targets valid JSON output."""
        theme_distribution = json.dumps(theme_analysis.get("themes", {}), indent=2)

        return f"""Analyze the following news for {ticker}.

The news is intentionally split into labeled sections so you can distinguish:
- broad market or economic context
- company expected revenue / earnings / guidance items
- general company-specific developments
- industry and peer context

{formatted_news}

[Theme Analysis]
Dominant Theme: {theme_analysis.get('dominant_theme', 'Mixed')}
Theme Distribution: {theme_distribution}
News Volume: {total_articles}

Return valid JSON using this exact top-level structure:
{{
  "ticker": "{ticker}",
  "overall_sentiment": {{
    "direction": "bullish|neutral|bearish",
    "score": 0,
    "confidence": 0,
    "summary": "1-2 sentence evidence-based summary"
  }},
  "news_sections": {{
    "macro_news": {{
      "count": 0,
      "sentiment": "positive|neutral|negative",
      "themes": ["theme"],
      "headline_refs": ["headline"]
    }},
    "company_expected_revenue_news": {{
      "count": 0,
      "sentiment": "positive|neutral|negative",
      "themes": ["theme"],
      "headline_refs": ["headline"]
    }},
    "company_specific_news": {{
      "count": 0,
      "sentiment": "positive|neutral|negative",
      "themes": ["theme"],
      "headline_refs": ["headline"]
    }},
    "industry_peer_news": {{
      "count": 0,
      "sentiment": "positive|neutral|negative",
      "themes": ["theme"],
      "headline_refs": ["headline"]
    }}
  }},
  "key_drivers": [
    {{
      "theme": "short theme name",
      "sentiment": "positive|neutral|negative",
      "impact": "high|medium|low",
      "evidence": "reference the relevant headline text"
    }}
  ],
  "notable_highlights": [
    {{
      "headline": "headline text",
      "category": "macro_news|company_expected_revenue_news|company_specific_news|industry_peer_news",
      "sentiment": "positive|neutral|negative",
      "impact": "high|medium|low",
      "reason": "brief explanation"
    }}
  ],
  "narrative_assessment": {{
    "dominant_narrative": "main narrative",
    "secondary_narratives": ["narrative"],
    "speculation_vs_fact": "separate hard evidence from market interpretation"
  }},
  "sentiment_trend": {{
    "trend": "improving|stable|deteriorating",
    "momentum": "strong|moderate|weak",
    "explanation": "brief explanation"
  }},
  "media_attention": {{
    "level": "low|medium|high",
    "reason": "brief explanation"
  }},
  "risk_factors": ["risk"],
  "positive_catalysts": ["catalyst"]
}}

Requirements:
1. Output valid JSON only. No markdown fences.
2. Use only supplied headlines and metadata as evidence.
3. Treat "company_expected_revenue_news" separately from general company news.
4. If a section is empty, keep its count at 0 and use neutral sentiment.
5. Keep each array concise and high-signal."""

    def _parse_structured_response(self, raw_response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from model output, including common fenced-code variants."""
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            pass

        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return None

        brace_match = re.search(r"(\{.*\})", raw_response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                return None

        return None

    def _analyze_sentiment_themes(self, news_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment themes using embeddings.

        Args:
            news_articles: List of news articles

        Returns:
            Dictionary with theme analysis
        """
        if not news_articles:
            return {
                "themes": [],
                "dominant_theme": "No news available",
                "theme_count": 0
            }

        # Generate embeddings for each headline
        embeddings = []
        for article in news_articles[:10]:  # Limit to 10 most recent
            title = article.get("title", "")
            if title:
                try:
                    embedding = self._generate_embedding(title)
                    embeddings.append({
                        "title": title,
                        "embedding": embedding,
                        "publisher": article.get("publisher", "Unknown")
                    })
                except Exception as e:
                    print(f"Warning: Failed to embed headline: {e}")

        # Simple theme detection based on keywords
        themes = {
            "growth": ["revenue", "sales", "growth", "expansion", "increase"],
            "earnings": ["earnings", "profit", "income", "eps", "quarter"],
            "innovation": ["product", "launch", "innovation", "technology", "new"],
            "risk": ["concern", "decline", "fall", "risk", "worry", "challenge"],
            "market": ["market", "sector", "industry", "competition"],
            "guidance": ["forecast", "outlook", "guidance", "expect", "target"]
        }

        theme_counts = {theme: 0 for theme in themes}

        for article in news_articles[:10]:
            title_lower = article.get("title", "").lower()
            for theme, keywords in themes.items():
                if any(keyword in title_lower for keyword in keywords):
                    theme_counts[theme] += 1

        # Get dominant theme
        dominant_theme = max(theme_counts, key=theme_counts.get) if max(theme_counts.values()) > 0 else "Mixed"

        return {
            "themes": theme_counts,
            "dominant_theme": dominant_theme,
            "theme_count": len([c for c in theme_counts.values() if c > 0]),
            "embedding_count": len(embeddings)
        }

    def analyze_sentiment(
        self,
        ticker: str,
        news_articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles.

        Args:
            ticker: Stock ticker
            news_articles: List of news articles

        Returns:
            Dictionary with sentiment analysis
        """
        print(f"[{self.AGENT_NAME}] Analyzing sentiment for {ticker}...")

        if not news_articles:
            empty_payload = {
                "ticker": ticker,
                "overall_sentiment": {
                    "direction": "neutral",
                    "score": 50,
                    "confidence": 100,
                    "summary": "No recent news articles were available for sentiment analysis."
                },
                "news_sections": {
                    "macro_news": {"count": 0, "sentiment": "neutral", "themes": [], "headline_refs": []},
                    "company_expected_revenue_news": {"count": 0, "sentiment": "neutral", "themes": [], "headline_refs": []},
                    "company_specific_news": {"count": 0, "sentiment": "neutral", "themes": [], "headline_refs": []},
                    "industry_peer_news": {"count": 0, "sentiment": "neutral", "themes": [], "headline_refs": []}
                },
                "key_drivers": [],
                "notable_highlights": [],
                "narrative_assessment": {
                    "dominant_narrative": "No active news narrative",
                    "secondary_narratives": [],
                    "speculation_vs_fact": "No news evidence provided."
                },
                "sentiment_trend": {
                    "trend": "stable",
                    "momentum": "weak",
                    "explanation": "No recent news flow to infer a trend."
                },
                "media_attention": {
                    "level": "low",
                    "reason": "No recent articles were available."
                },
                "risk_factors": [],
                "positive_catalysts": []
            }
            return {
                "agent": self.AGENT_NAME,
                "ticker": ticker,
                "theme_analysis": {
                    "themes": [],
                    "dominant_theme": "No news available",
                    "theme_count": 0
                },
                "news_categories": {
                    "macro_news": [],
                    "company_expected_revenue_news": [],
                    "company_specific_news": [],
                    "industry_peer_news": [],
                },
                "news_count": 0,
                "structured_sentiment": empty_payload,
                "report": json.dumps(empty_payload, indent=2),
                "thread_id": None,
            }

        # Analyze themes
        theme_analysis = self._analyze_sentiment_themes(news_articles)
        news_categories = self._categorize_news(news_articles[:15])
        formatted_news = self._format_news_fingpt_style(ticker, news_categories)
        analysis_prompt = self._create_structured_prompt(
            ticker=ticker,
            formatted_news=formatted_news,
            theme_analysis=theme_analysis,
            total_articles=len(news_articles)
        )

        # Store in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_sentiment_themes",
                value=theme_analysis,
                agent=self.AGENT_NAME
            )
            self.memory.post_to_scratchpad(
                key=f"{ticker}_news_categories",
                value=news_categories,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Generating sentiment report...")

        # Generate report using assistant
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

        structured_sentiment = {}
        if assistant_messages:
            raw_response = assistant_messages[0].content[0].text.value
            parsed_response = self._parse_structured_response(raw_response)
            if parsed_response:
                structured_sentiment = parsed_response
                report = json.dumps(parsed_response, indent=2)
            else:
                report = raw_response
        else:
            report = "Sentiment analysis report generation failed"

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={
                    "ticker": ticker,
                    "type": "sentiment_analysis",
                    "structured_data": structured_sentiment
                }
            )

        print(f"[{self.AGENT_NAME}] Sentiment analysis complete")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "theme_analysis": theme_analysis,
            "news_categories": news_categories,
            "news_count": len(news_articles),
            "structured_sentiment": structured_sentiment,
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
