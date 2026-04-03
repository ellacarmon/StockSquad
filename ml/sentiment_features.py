"""
Utilities for converting SentimentAgent output into numeric ML features.
"""

from typing import Any, Dict, Optional


def _map_sentiment_label(value: Optional[str]) -> float:
    mapping = {
        "positive": 1.0,
        "bullish": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
        "bearish": -1.0,
    }
    return mapping.get((value or "").strip().lower(), 0.0)


def _map_trend_label(value: Optional[str]) -> float:
    mapping = {
        "improving": 1.0,
        "stable": 0.0,
        "deteriorating": -1.0,
    }
    return mapping.get((value or "").strip().lower(), 0.0)


def extract_sentiment_features(sentiment_result: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """
    Convert structured SentimentAgent output into numeric features.

    Defaults are intentionally neutral so the ML pipeline can operate even when
    sentiment data is absent or a model was trained before sentiment features
    existed.
    """
    default_features = {
        "news_sentiment_score": 50.0,
        "news_sentiment_confidence": 0.0,
        "news_sentiment_article_count": 0.0,
        "news_sentiment_trend": 0.0,
        "news_macro_sentiment": 0.0,
        "news_company_expected_revenue_sentiment": 0.0,
        "news_company_specific_sentiment": 0.0,
        "news_industry_peer_sentiment": 0.0,
    }

    if not sentiment_result:
        return default_features

    structured = sentiment_result.get("structured_sentiment", sentiment_result)
    if not isinstance(structured, dict):
        return default_features

    overall = structured.get("overall_sentiment", {})
    sections = structured.get("news_sections", {})
    trend = structured.get("sentiment_trend", {})

    features = default_features.copy()
    features["news_sentiment_score"] = float(overall.get("score", 50) or 50)
    features["news_sentiment_confidence"] = float(overall.get("confidence", 0) or 0)
    features["news_sentiment_article_count"] = float(
        sentiment_result.get("news_count", structured.get("news_count", 0)) or 0
    )
    features["news_sentiment_trend"] = _map_trend_label(trend.get("trend"))
    features["news_macro_sentiment"] = _map_sentiment_label(
        sections.get("macro_news", {}).get("sentiment")
    )
    features["news_company_expected_revenue_sentiment"] = _map_sentiment_label(
        sections.get("company_expected_revenue_news", {}).get("sentiment")
    )
    features["news_company_specific_sentiment"] = _map_sentiment_label(
        sections.get("company_specific_news", {}).get("sentiment")
    )
    features["news_industry_peer_sentiment"] = _map_sentiment_label(
        sections.get("industry_peer_news", {}).get("sentiment")
    )

    return features
