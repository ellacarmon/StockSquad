"""
SocialSentimentAgent: Social media sentiment analysis using xpoz.ai

Analyzes social media mentions across Twitter/X, Reddit, TikTok, and Instagram
to capture retail investor sentiment and viral trends.

References:
- xpoz.ai API documentation
- Multi-platform sentiment aggregation
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from config import get_settings
from memory.short_term import ShortTermMemory


class SocialSentimentAgent:
    """
    Social media sentiment agent using xpoz.ai for multi-platform analysis.

    Key features:
    - Query Twitter/X, Reddit, TikTok, Instagram via xpoz.ai
    - Platform-specific sentiment breakdowns
    - Engagement metrics (likes, shares, comments)
    - Viral trend detection
    - Influencer tracking
    - Structured JSON output format
    """

    AGENT_NAME = "SocialSentimentAgent"

    # Agent instructions for LLM
    AGENT_INSTRUCTIONS = """You are a social media sentiment analyst specializing in retail investor sentiment.

Your task is to analyze social media posts from Twitter/X, Reddit, TikTok, and Instagram and produce structured analysis.

Output Structure (JSON):

1. overall_sentiment: Object with:
   - direction: "bullish", "neutral", or "bearish"
   - score: 0-100 (0=very bearish, 50=neutral, 100=very bullish)
   - confidence: 0-100 (confidence in assessment)
   - summary: Brief explanation

2. platform_breakdown: Object with sentiment by platform:
   - twitter: {sentiment, key_themes, engagement_level, post_count}
   - reddit: {sentiment, key_themes, engagement_level, post_count}
   - tiktok: {sentiment, key_themes, engagement_level, post_count}
   - instagram: {sentiment, key_themes, engagement_level, post_count}

3. engagement_metrics: Object with:
   - total_posts: Number of posts analyzed
   - total_engagement: Sum of likes, shares, comments
   - engagement_trend: "increasing", "stable", or "decreasing"
   - viral_potential: "high", "medium", or "low"

4. key_themes: Array of dominant discussion topics with:
   - theme: Topic name
   - sentiment: "positive", "neutral", or "negative"
   - platform: Primary platform discussing this
   - volume: "high", "medium", or "low"

5. influencer_mentions: Array of high-impact posts from verified/popular accounts

6. risk_signals: Array of concerning patterns (pump-and-dump, coordinated campaigns, controversy)

7. retail_momentum: Object with:
   - direction: "bullish", "neutral", or "bearish"
   - strength: "strong", "moderate", or "weak"
   - wsb_activity: Reddit WallStreetBets activity level

Analyze objectively. Flag suspicious coordinated activity. Weight verified accounts higher."""

    def __init__(self, memory: Optional[ShortTermMemory] = None):
        """Initialize the SocialSentimentAgent."""
        self.settings = get_settings()
        self.memory = memory
        self.client = self._initialize_client()
        self.assistant = None

        # Initialize xpoz.ai client if API key is available
        self.xpoz_client = self._initialize_xpoz_client()

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

    def _initialize_xpoz_client(self):
        """Initialize xpoz.ai client."""
        if not self.settings.xpoz_api_key:
            print(f"[{self.AGENT_NAME}] Warning: No xpoz_api_key found. Social sentiment will be disabled.")
            print(f"[{self.AGENT_NAME}] Get a free API key at https://www.xpoz.ai/get-token")
            return None

        try:
            # Import xpoz SDK
            import xpoz
            print(f"[{self.AGENT_NAME}] xpoz package imported successfully (v{xpoz.__version__ if hasattr(xpoz, '__version__') else 'unknown'})")
            client = xpoz.Client(api_key=self.settings.xpoz_api_key)
            print(f"[{self.AGENT_NAME}] xpoz.ai client initialized successfully")
            return client
        except ImportError as e:
            print(f"[{self.AGENT_NAME}] Warning: xpoz package not installed: {e}")
            print(f"[{self.AGENT_NAME}] Run: pip install xpoz")
            return None
        except Exception as e:
            print(f"[{self.AGENT_NAME}] Warning: Failed to initialize xpoz client: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_assistant(self):
        """Create the Azure OpenAI assistant."""
        if self.assistant is None:
            self.assistant = self.client.beta.assistants.create(
                model=self.settings.azure_openai_deployment_name,
                name=self.AGENT_NAME,
                instructions=self.AGENT_INSTRUCTIONS,
            )
        return self.assistant

    def _fetch_social_posts(
        self,
        ticker: str,
        days_back: int = 7,
        max_results_per_platform: int = 100
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch social media posts from xpoz.ai.

        Args:
            ticker: Stock ticker symbol
            days_back: How many days back to search
            max_results_per_platform: Max results per platform

        Returns:
            Dictionary with posts grouped by platform
        """
        if not self.xpoz_client:
            return {"twitter": [], "reddit": [], "tiktok": [], "instagram": []}

        platforms = {
            "twitter": [],
            "reddit": [],
            "tiktok": [],
            "instagram": []
        }

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Search query variations for the ticker
        queries = [
            f"${ticker}",  # Cashtag format
            f"#{ticker}",  # Hashtag format
            ticker,        # Plain ticker
        ]

        print(f"[{self.AGENT_NAME}] Fetching social posts for {ticker} from last {days_back} days...")

        try:
            for query in queries:
                # Query each platform via xpoz.ai
                # Note: Actual API methods may differ - this is a template based on typical REST APIs

                # Twitter/X
                try:
                    twitter_posts = self.xpoz_client.search_posts(
                        query=query,
                        platform="twitter",
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        limit=max_results_per_platform
                    )
                    platforms["twitter"].extend(twitter_posts)
                except Exception as e:
                    print(f"[{self.AGENT_NAME}] Twitter query failed: {e}")

                # Reddit
                try:
                    reddit_posts = self.xpoz_client.search_posts(
                        query=query,
                        platform="reddit",
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        limit=max_results_per_platform
                    )
                    platforms["reddit"].extend(reddit_posts)
                except Exception as e:
                    print(f"[{self.AGENT_NAME}] Reddit query failed: {e}")

                # TikTok
                try:
                    tiktok_posts = self.xpoz_client.search_posts(
                        query=query,
                        platform="tiktok",
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        limit=max_results_per_platform
                    )
                    platforms["tiktok"].extend(tiktok_posts)
                except Exception as e:
                    print(f"[{self.AGENT_NAME}] TikTok query failed: {e}")

                # Instagram
                try:
                    instagram_posts = self.xpoz_client.search_posts(
                        query=query,
                        platform="instagram",
                        start_date=start_date.isoformat(),
                        end_date=end_date.isoformat(),
                        limit=max_results_per_platform
                    )
                    platforms["instagram"].extend(instagram_posts)
                except Exception as e:
                    print(f"[{self.AGENT_NAME}] Instagram query failed: {e}")

            # Deduplicate posts by ID within each platform
            for platform in platforms:
                seen_ids = set()
                unique_posts = []
                for post in platforms[platform]:
                    post_id = post.get('id')
                    if post_id and post_id not in seen_ids:
                        seen_ids.add(post_id)
                        unique_posts.append(post)
                platforms[platform] = unique_posts

            # Sort by engagement (likes + comments + shares)
            for platform in platforms:
                platforms[platform].sort(
                    key=lambda p: (
                        p.get('likes', 0) +
                        p.get('comments', 0) +
                        p.get('shares', 0)
                    ),
                    reverse=True
                )

            print(f"[{self.AGENT_NAME}] Fetched {sum(len(v) for v in platforms.values())} posts across platforms")

        except Exception as e:
            print(f"[{self.AGENT_NAME}] Error fetching social posts: {e}")

        return platforms

    def _format_posts_for_llm(
        self,
        ticker: str,
        posts_by_platform: Dict[str, List[Dict]]
    ) -> str:
        """Format social media posts for LLM analysis."""
        sections = [
            f"# Social Media Analysis for {ticker}",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}",
            ""
        ]

        total_posts = sum(len(posts) for posts in posts_by_platform.values())
        sections.append(f"TOTAL POSTS: {total_posts} across all platforms")
        sections.append("")

        # Twitter/X section
        sections.append("## [TWITTER/X POSTS]")
        twitter_posts = posts_by_platform.get("twitter", [])
        if twitter_posts:
            sections.append(f"Post Count: {len(twitter_posts)}")
            for i, post in enumerate(twitter_posts[:20], 1):  # Top 20 by engagement
                sections.append(f"{i}. {post.get('text', 'N/A')[:200]}")
                sections.append(f"   Engagement: {post.get('likes', 0)} likes, {post.get('retweets', 0)} retweets")
                sections.append(f"   Author: {post.get('author', 'Unknown')} (Followers: {post.get('author_followers', 0)})")
                sections.append("")
        else:
            sections.append("No Twitter posts found.")
            sections.append("")

        # Reddit section
        sections.append("## [REDDIT POSTS]")
        reddit_posts = posts_by_platform.get("reddit", [])
        if reddit_posts:
            sections.append(f"Post Count: {len(reddit_posts)}")
            for i, post in enumerate(reddit_posts[:20], 1):
                sections.append(f"{i}. {post.get('title', 'N/A')}")
                sections.append(f"   Text: {post.get('text', 'N/A')[:200]}")
                sections.append(f"   Subreddit: r/{post.get('subreddit', 'Unknown')}")
                sections.append(f"   Engagement: {post.get('upvotes', 0)} upvotes, {post.get('comments', 0)} comments")
                sections.append("")
        else:
            sections.append("No Reddit posts found.")
            sections.append("")

        # TikTok section
        sections.append("## [TIKTOK POSTS]")
        tiktok_posts = posts_by_platform.get("tiktok", [])
        if tiktok_posts:
            sections.append(f"Post Count: {len(tiktok_posts)}")
            for i, post in enumerate(tiktok_posts[:20], 1):
                sections.append(f"{i}. {post.get('description', 'N/A')[:200]}")
                sections.append(f"   Engagement: {post.get('likes', 0)} likes, {post.get('shares', 0)} shares")
                sections.append(f"   Author: {post.get('author', 'Unknown')} (Followers: {post.get('author_followers', 0)})")
                sections.append("")
        else:
            sections.append("No TikTok posts found.")
            sections.append("")

        # Instagram section
        sections.append("## [INSTAGRAM POSTS]")
        instagram_posts = posts_by_platform.get("instagram", [])
        if instagram_posts:
            sections.append(f"Post Count: {len(instagram_posts)}")
            for i, post in enumerate(instagram_posts[:20], 1):
                sections.append(f"{i}. {post.get('caption', 'N/A')[:200]}")
                sections.append(f"   Engagement: {post.get('likes', 0)} likes, {post.get('comments', 0)} comments")
                sections.append(f"   Author: {post.get('author', 'Unknown')} (Followers: {post.get('author_followers', 0)})")
                sections.append("")
        else:
            sections.append("No Instagram posts found.")
            sections.append("")

        return "\n".join(sections)

    def _create_analysis_prompt(
        self,
        ticker: str,
        formatted_posts: str
    ) -> str:
        """Create structured analysis prompt for LLM."""
        return f"""Analyze the following social media posts for {ticker} and provide structured JSON analysis.

{formatted_posts}

REQUIRED OUTPUT FORMAT (JSON):
{{
  "overall_sentiment": {{
    "direction": "bullish|neutral|bearish",
    "score": <0-100>,
    "confidence": <0-100>,
    "summary": "<1-2 sentence explanation>"
  }},
  "platform_breakdown": {{
    "twitter": {{
      "sentiment": "positive|neutral|negative",
      "key_themes": ["<theme1>", "<theme2>"],
      "engagement_level": "high|medium|low",
      "post_count": <number>
    }},
    "reddit": {{
      "sentiment": "positive|neutral|negative",
      "key_themes": ["<theme1>", "<theme2>"],
      "engagement_level": "high|medium|low",
      "post_count": <number>
    }},
    "tiktok": {{
      "sentiment": "positive|neutral|negative",
      "key_themes": ["<theme1>", "<theme2>"],
      "engagement_level": "high|medium|low",
      "post_count": <number>
    }},
    "instagram": {{
      "sentiment": "positive|neutral|negative",
      "key_themes": ["<theme1>", "<theme2>"],
      "engagement_level": "high|medium|low",
      "post_count": <number>
    }}
  }},
  "engagement_metrics": {{
    "total_posts": <number>,
    "engagement_trend": "increasing|stable|decreasing",
    "viral_potential": "high|medium|low"
  }},
  "key_themes": [
    {{
      "theme": "<theme name>",
      "sentiment": "positive|neutral|negative",
      "platform": "<primary platform>",
      "volume": "high|medium|low"
    }}
  ],
  "influencer_mentions": [
    {{
      "platform": "<platform>",
      "author": "<username>",
      "followers": <number>,
      "sentiment": "positive|neutral|negative",
      "snippet": "<key quote>"
    }}
  ],
  "risk_signals": ["<risk1>", "<risk2>", ...],
  "retail_momentum": {{
    "direction": "bullish|neutral|bearish",
    "strength": "strong|moderate|weak",
    "wsb_activity": "high|medium|low"
  }}
}}

Respond ONLY with valid JSON. Be objective and flag suspicious coordinated activity."""

    def analyze_sentiment(
        self,
        ticker: str,
        days_back: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze social media sentiment for a ticker.

        Args:
            ticker: Stock ticker symbol
            days_back: Number of days to look back

        Returns:
            Dictionary with social sentiment analysis
        """
        print(f"[{self.AGENT_NAME}] Analyzing social sentiment for {ticker}...")

        if not self.xpoz_client:
            print(f"[{self.AGENT_NAME}] xpoz.ai client not available. Skipping social sentiment.")
            return self._empty_response(ticker)

        # Fetch social media posts
        posts_by_platform = self._fetch_social_posts(ticker, days_back)

        total_posts = sum(len(posts) for posts in posts_by_platform.values())
        if total_posts == 0:
            print(f"[{self.AGENT_NAME}] No social media posts found for {ticker}")
            return self._empty_response(ticker)

        # Format for LLM
        formatted_posts = self._format_posts_for_llm(ticker, posts_by_platform)

        # Create analysis prompt
        analysis_prompt = self._create_analysis_prompt(ticker, formatted_posts)

        # Store raw posts in memory
        if self.memory:
            self.memory.post_to_scratchpad(
                key=f"{ticker}_social_posts",
                value=posts_by_platform,
                agent=self.AGENT_NAME
            )

        print(f"[{self.AGENT_NAME}] Generating social sentiment analysis...")

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
            return self._empty_response(ticker)

        # Get response
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        assistant_messages = [
            msg for msg in messages.data
            if msg.role == "assistant" and msg.created_at > message.created_at
        ]

        if not assistant_messages:
            return self._empty_response(ticker)

        raw_response = assistant_messages[0].content[0].text.value

        # Parse JSON response
        structured_sentiment = self._parse_json_response(raw_response)

        # Generate readable report
        report = self._generate_report(ticker, structured_sentiment, posts_by_platform)

        # Add to memory
        if self.memory:
            self.memory.add_message(
                agent=self.AGENT_NAME,
                role="assistant",
                content=report,
                metadata={
                    "ticker": ticker,
                    "type": "social_sentiment_analysis",
                    "structured_data": structured_sentiment
                }
            )

        print(f"[{self.AGENT_NAME}] Social sentiment analysis complete")

        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "post_count": total_posts,
            "posts_by_platform": posts_by_platform,
            "structured_sentiment": structured_sentiment,
            "report": report,
            "thread_id": thread.id,
        }

    def _parse_json_response(self, raw_response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        try:
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
            print(f"[{self.AGENT_NAME}] Warning: Could not parse JSON response")
            return {}

    def _generate_report(
        self,
        ticker: str,
        structured_data: Dict[str, Any],
        posts_by_platform: Dict[str, List[Dict]]
    ) -> str:
        """Generate human-readable report from structured data."""
        if not structured_data:
            return f"Social sentiment analysis for {ticker} - No structured data available"

        report_lines = [
            f"# Social Media Sentiment for {ticker}",
            "",
            "## Overall Social Sentiment",
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

        # Platform breakdown
        platform_breakdown = structured_data.get("platform_breakdown", {})
        if platform_breakdown:
            report_lines.append("## Platform Breakdown")
            for platform, data in platform_breakdown.items():
                sentiment = data.get("sentiment", "neutral")
                engagement = data.get("engagement_level", "low")
                post_count = len(posts_by_platform.get(platform, []))
                themes = data.get("key_themes", [])

                report_lines.append(f"### {platform.capitalize()}")
                report_lines.append(f"- **Posts**: {post_count}")
                report_lines.append(f"- **Sentiment**: {sentiment}")
                report_lines.append(f"- **Engagement**: {engagement}")
                if themes:
                    report_lines.append(f"- **Key Themes**: {', '.join(themes)}")
                report_lines.append("")

        # Key themes
        key_themes = structured_data.get("key_themes", [])
        if key_themes:
            report_lines.append("## Key Discussion Themes")
            for theme in key_themes[:5]:
                theme_name = theme.get("theme", "Unknown")
                sentiment = theme.get("sentiment", "neutral")
                platform = theme.get("platform", "")
                volume = theme.get("volume", "medium")

                emoji = "✓" if sentiment == "positive" else "✗" if sentiment == "negative" else "○"
                report_lines.append(f"{emoji} **{theme_name}** ({volume} volume on {platform})")
            report_lines.append("")

        # Influencer mentions
        influencers = structured_data.get("influencer_mentions", [])
        if influencers:
            report_lines.append("## Notable Influencer Mentions")
            for inf in influencers[:5]:
                author = inf.get("author", "Unknown")
                platform = inf.get("platform", "")
                followers = inf.get("followers", 0)
                sentiment = inf.get("sentiment", "neutral")
                snippet = inf.get("snippet", "")

                report_lines.append(f"- **@{author}** ({platform}, {followers:,} followers)")
                report_lines.append(f"  Sentiment: {sentiment}")
                if snippet:
                    report_lines.append(f"  \"{snippet}\"")
            report_lines.append("")

        # Retail momentum
        momentum = structured_data.get("retail_momentum", {})
        if momentum:
            direction = momentum.get("direction", "neutral")
            strength = momentum.get("strength", "moderate")
            wsb = momentum.get("wsb_activity", "low")

            report_lines.extend([
                "## Retail Momentum",
                f"**Direction**: {direction.capitalize()} ({strength} strength)",
                f"**WallStreetBets Activity**: {wsb}",
                ""
            ])

        # Risk signals
        risks = structured_data.get("risk_signals", [])
        if risks:
            report_lines.append("## Risk Signals")
            for risk in risks:
                report_lines.append(f"- {risk}")
            report_lines.append("")

        # Engagement metrics
        engagement = structured_data.get("engagement_metrics", {})
        if engagement:
            trend = engagement.get("engagement_trend", "stable")
            viral = engagement.get("viral_potential", "low")

            report_lines.extend([
                "## Engagement Metrics",
                f"**Trend**: {trend.capitalize()}",
                f"**Viral Potential**: {viral.capitalize()}",
                ""
            ])

        return "\n".join(report_lines)

    def _empty_response(self, ticker: str) -> Dict[str, Any]:
        """Return empty response when no data available."""
        return {
            "agent": self.AGENT_NAME,
            "ticker": ticker,
            "post_count": 0,
            "posts_by_platform": {"twitter": [], "reddit": [], "tiktok": [], "instagram": []},
            "structured_sentiment": {},
            "report": f"No social media data available for {ticker}",
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
