# xpoz.ai Integration Setup Guide

## Overview

StockSquad now integrates with **xpoz.ai** to analyze social media sentiment across Twitter/X, Reddit, TikTok, and Instagram. This adds a crucial layer of retail investor sentiment analysis alongside traditional news sources.

## What is xpoz.ai?

- **Multi-platform social media API**: Access 1.5B+ posts from Twitter/X, Reddit, TikTok, and Instagram
- **Unified interface**: Single API for all platforms (no need for expensive Twitter API keys)
- **Real-time data**: Capture viral trends and retail momentum as they happen
- **Cost-effective**: Free tier includes 100,000 results/month

## Pricing

- **Free**: 100,000 results/month (no credit card required)
- **Pro**: $20/month for 1M results
- **Max**: $200/month for 10M results

For most StockSquad use cases, the **free tier is sufficient**.

---

## Setup Instructions

### Step 1: Get Your Free xpoz.ai API Key

1. Visit **https://www.xpoz.ai/get-token**
2. Sign up or log in (uses OAuth with Google account)
3. Copy your API key

**Note**: No credit card required for the free tier.

---

### Step 2: Install the xpoz Python SDK

The xpoz SDK has already been added to `requirements.txt`. Install it with:

```bash
pip install -r requirements.txt
```

Or install just the xpoz package:

```bash
pip install xpoz
```

---

### Step 3: Configure Your API Key

Add your xpoz.ai API key to your `.env` file:

```bash
# .env
XPOZ_API_KEY=your_xpoz_api_key_here
```

The configuration is already set up in `config.py` to read this environment variable.

---

### Step 4: Test the Integration

Run the test suite to verify everything is working:

```bash
python tests/test_social_sentiment.py
```

This will run 4 tests:
1. ✓ xpoz.ai client initialization
2. ✓ SocialSentimentAgent initialization
3. ✓ Fetch social media posts
4. ✓ Full sentiment analysis pipeline

**Expected output**: All tests should pass if the API key is configured correctly.

---

## Usage

### Basic Usage: Analyze Social Sentiment

```python
from agents.social_sentiment_agent import SocialSentimentAgent

# Initialize the agent
agent = SocialSentimentAgent()

# Analyze social sentiment for a ticker
result = agent.analyze_sentiment(
    ticker="AAPL",
    days_back=7  # Analyze last 7 days of posts
)

# Access the results
print(f"Total posts: {result['post_count']}")
print(f"Report:\n{result['report']}")

# Access structured sentiment data
sentiment = result['structured_sentiment']
overall = sentiment['overall_sentiment']
print(f"Overall sentiment: {overall['direction']} (score: {overall['score']}/100)")

# Platform breakdown
for platform, data in sentiment['platform_breakdown'].items():
    print(f"{platform}: {data['sentiment']} ({data['post_count']} posts)")

# Cleanup
agent.cleanup()
```

---

### Advanced: Platform-Specific Analysis

```python
# Get raw posts by platform
result = agent.analyze_sentiment("TSLA", days_back=3)
posts_by_platform = result['posts_by_platform']

# Analyze Twitter sentiment specifically
twitter_posts = posts_by_platform['twitter']
print(f"Twitter posts: {len(twitter_posts)}")

# Check Reddit WallStreetBets activity
reddit_posts = posts_by_platform['reddit']
wsb_posts = [p for p in reddit_posts if p.get('subreddit') == 'wallstreetbets']
print(f"WallStreetBets mentions: {len(wsb_posts)}")

# Identify influencer mentions
structured = result['structured_sentiment']
influencers = structured.get('influencer_mentions', [])
for inf in influencers:
    print(f"@{inf['author']} ({inf['followers']:,} followers): {inf['sentiment']}")
```

---

## Integration with StockSquad Pipeline

### Combining News + Social Sentiment

The `SocialSentimentAgent` works alongside the existing `SentimentAgentV2` (news sentiment):

```python
from agents.sentiment_agent_v2 import SentimentAgentV2
from agents.social_sentiment_agent import SocialSentimentAgent

# Analyze news sentiment
news_agent = SentimentAgentV2()
news_result = news_agent.analyze_sentiment(ticker="NVDA", news_articles=articles)

# Analyze social sentiment
social_agent = SocialSentimentAgent()
social_result = social_agent.analyze_sentiment(ticker="NVDA", days_back=7)

# Compare news vs. social sentiment
news_score = news_result['structured_sentiment']['overall_sentiment']['score']
social_score = social_result['structured_sentiment']['overall_sentiment']['score']

print(f"News sentiment: {news_score}/100")
print(f"Social sentiment: {social_score}/100")

if abs(news_score - social_score) > 20:
    print("⚠ Divergence detected between news and social sentiment!")
```

---

## What Data Is Available?

### Per Post:
- **Text/Content**: Post text, caption, or description
- **Engagement**: Likes, comments, shares, retweets
- **Author**: Username, follower count, verification status
- **Metadata**: Timestamp, hashtags, mentions, media URLs
- **Platform**: twitter, reddit, tiktok, instagram

### Aggregate Metrics:
- **Sentiment**: Bullish/Neutral/Bearish across platforms
- **Trends**: Viral potential, engagement momentum
- **Themes**: Key discussion topics
- **Risk Signals**: Pump-and-dump detection, coordinated campaigns
- **Influencer Activity**: High-impact posts from verified accounts

---

## API Rate Limits

### Free Tier (100,000 results/month)
- Approximately **3,300 results per day**
- For 10 stocks analyzed 10 times/day = **100 queries/day** ✓ (well within limits)

### Tips to Stay Within Limits:
1. **Cache results**: Social sentiment doesn't change every second
2. **Prioritize high-volume stocks**: Focus on actively traded stocks
3. **Adjust `days_back`**: Use fewer days for frequent queries (e.g., `days_back=3`)
4. **Limit results per platform**: Set `max_results_per_platform=50` instead of 100

---

## Troubleshooting

### Error: "xpoz package not installed"
**Solution**: Run `pip install xpoz`

### Error: "No xpoz_api_key found"
**Solution**: Add `XPOZ_API_KEY=your_key` to your `.env` file

### Error: "API authentication failed"
**Solution**:
- Verify your API key is correct
- Check that you copied the full key from https://www.xpoz.ai/get-token
- Ensure no extra spaces or quotes in the `.env` file

### Warning: "No posts found"
**Possible causes**:
- Ticker has low social media activity (try TSLA, AAPL, NVDA, GME for testing)
- Date range too narrow (try `days_back=7` or `days_back=14`)
- Ticker not mentioned in the queried platforms

### Timeout during analysis
**Solution**:
- The LLM analysis can take 1-2 minutes for large datasets
- This is normal for the first run
- Consider reducing `max_results_per_platform` for faster processing

---

## File Structure

```
StockSquad/
├── agents/
│   ├── social_sentiment_agent.py      # NEW: Social sentiment agent
│   └── sentiment_agent_v2.py           # EXISTING: News sentiment agent
│
├── tests/
│   └── test_social_sentiment.py        # NEW: Test suite
│
├── config.py                           # UPDATED: Added xpoz_api_key
├── requirements.txt                    # UPDATED: Added xpoz>=0.1.0
├── .env                                # ADD: XPOZ_API_KEY=your_key
└── XPOZ_INTEGRATION_SETUP.md          # This file
```

---

## Next Steps: Phase 2

After validating Phase 1 (this setup), the next phases include:

- **Phase 2**: ML Feature Engineering
  - Add social sentiment features to `ml/sentiment_features.py`
  - Create `social_metrics` skill for engagement analysis
  - Integrate with TechnicalAgent's ML signal scoring

- **Phase 3**: Advanced Analysis
  - Multi-platform sentiment aggregation
  - Viral trend detection
  - Risk signal identification (pump-and-dump warnings)

- **Phase 4**: UI Integration
  - Display social sentiment in web UI
  - Add platform-specific breakdowns
  - Show "Social Buzz" indicator

- **Phase 5**: Optimization
  - Caching strategy for cost reduction
  - Rate limit monitoring
  - Model retraining with social features

---

## Support

- **xpoz.ai Documentation**: https://www.xpoz.ai/
- **GitHub (xpoz)**: https://github.com/XPOZpublic
- **StockSquad Issues**: Report integration issues in the main repo

---

## Summary

✅ **What you get**:
- Real-time social media sentiment across 4 platforms
- Early warning signals (social often precedes news)
- Retail investor momentum tracking
- Meme stock detection
- Influencer sentiment analysis

✅ **Cost**: Free for most use cases (100K results/month)

✅ **Time to setup**: ~5 minutes

🚀 **Start analyzing social sentiment today!**
