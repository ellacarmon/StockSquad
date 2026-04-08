"""
Test script for SocialSentimentAgent and xpoz.ai integration.

Usage:
    python tests/test_social_sentiment.py

Requirements:
    - XPOZ_API_KEY environment variable set
    - Azure OpenAI credentials configured
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.social_sentiment_agent import SocialSentimentAgent
from config import get_settings
import json


def test_xpoz_client():
    """Test xpoz.ai client initialization."""
    print("=" * 80)
    print("TEST 1: xpoz.ai Client Initialization")
    print("=" * 80)

    try:
        settings = get_settings()
        if not settings.xpoz_api_key:
            print("❌ XPOZ_API_KEY not configured in environment")
            print("   Please get a free API key from https://www.xpoz.ai/get-token")
            print("   Then add to .env: XPOZ_API_KEY=your_key_here")
            return False

        print(f"✓ xpoz_api_key configured: {settings.xpoz_api_key[:10]}...")

        # Try to import xpoz package
        try:
            import xpoz
            print("✓ xpoz package is installed")
        except ImportError:
            print("❌ xpoz package not installed")
            print("   Run: pip install xpoz")
            return False

        # Try to initialize client
        try:
            client = xpoz.Client(api_key=settings.xpoz_api_key)
            print("✓ xpoz.Client initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize xpoz.Client: {e}")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_agent_initialization():
    """Test SocialSentimentAgent initialization."""
    print("\n" + "=" * 80)
    print("TEST 2: SocialSentimentAgent Initialization")
    print("=" * 80)

    try:
        agent = SocialSentimentAgent()
        print("✓ SocialSentimentAgent initialized")

        if agent.xpoz_client:
            print("✓ xpoz_client is available")
        else:
            print("❌ xpoz_client is None (check API key and package installation)")
            return False

        return True

    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fetch_posts():
    """Test fetching social media posts."""
    print("\n" + "=" * 80)
    print("TEST 3: Fetch Social Media Posts")
    print("=" * 80)

    try:
        agent = SocialSentimentAgent()

        if not agent.xpoz_client:
            print("⚠ Skipping: xpoz_client not available")
            return False

        # Test with a popular stock ticker
        ticker = "AAPL"
        print(f"Fetching posts for {ticker}...")

        posts_by_platform = agent._fetch_social_posts(
            ticker=ticker,
            days_back=7,
            max_results_per_platform=10
        )

        print("\nResults by platform:")
        total_posts = 0
        for platform, posts in posts_by_platform.items():
            count = len(posts)
            total_posts += count
            print(f"  {platform}: {count} posts")

        if total_posts > 0:
            print(f"\n✓ Successfully fetched {total_posts} posts")

            # Show sample post from each platform
            for platform, posts in posts_by_platform.items():
                if posts:
                    print(f"\n  Sample {platform} post:")
                    sample = posts[0]
                    print(f"    Text: {str(sample.get('text', sample.get('title', 'N/A')))[:100]}...")
                    print(f"    Engagement: {sample.get('likes', 0)} likes")

            return True
        else:
            print("⚠ No posts found (may be normal for less active tickers)")
            return True

    except Exception as e:
        print(f"❌ Error fetching posts: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sentiment_analysis():
    """Test full sentiment analysis pipeline."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Sentiment Analysis Pipeline")
    print("=" * 80)

    try:
        agent = SocialSentimentAgent()

        if not agent.xpoz_client:
            print("⚠ Skipping: xpoz_client not available")
            return False

        # Test with a popular ticker
        ticker = "TSLA"
        print(f"Running full sentiment analysis for {ticker}...")
        print("(This may take 1-2 minutes due to LLM processing)")

        result = agent.analyze_sentiment(ticker=ticker, days_back=3)

        print("\nAnalysis Results:")
        print(f"  Ticker: {result['ticker']}")
        print(f"  Total Posts: {result['post_count']}")

        if result.get('structured_sentiment'):
            print("\n  Structured Sentiment:")
            structured = result['structured_sentiment']

            # Overall sentiment
            overall = structured.get('overall_sentiment', {})
            if overall:
                print(f"    Direction: {overall.get('direction', 'N/A')}")
                print(f"    Score: {overall.get('score', 'N/A')}/100")
                print(f"    Confidence: {overall.get('confidence', 'N/A')}%")

            # Platform breakdown
            platform_breakdown = structured.get('platform_breakdown', {})
            if platform_breakdown:
                print("\n  Platform Sentiment:")
                for platform, data in platform_breakdown.items():
                    sentiment = data.get('sentiment', 'N/A')
                    post_count = data.get('post_count', 0)
                    if post_count > 0:
                        print(f"    {platform}: {sentiment} ({post_count} posts)")

        print("\n✓ Sentiment analysis completed successfully")
        print("\nGenerated Report:")
        print("-" * 80)
        print(result.get('report', 'No report generated'))
        print("-" * 80)

        # Save result to file for inspection
        output_file = Path(__file__).parent / f"social_sentiment_{ticker}_test.json"
        with open(output_file, 'w') as f:
            # Convert result to JSON-serializable format
            json_result = {
                'ticker': result['ticker'],
                'post_count': result['post_count'],
                'structured_sentiment': result.get('structured_sentiment', {}),
                'report': result.get('report', '')
            }
            json.dump(json_result, f, indent=2)

        print(f"\n✓ Full result saved to: {output_file}")

        # Cleanup
        agent.cleanup()

        return True

    except Exception as e:
        print(f"❌ Error in sentiment analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "XPOZ.AI INTEGRATION TEST SUITE" + " " * 27 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    results = {
        "Client Initialization": test_xpoz_client(),
        "Agent Initialization": test_agent_initialization(),
        "Fetch Posts": test_fetch_posts(),
        "Sentiment Analysis": test_sentiment_analysis(),
    }

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 80)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("=" * 80)

    if passed == total:
        print("\n🎉 All tests passed! xpoz.ai integration is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check configuration and API key.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
