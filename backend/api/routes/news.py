"""
API routes for news collection and sentiment analysis.
"""

import logging
from fastapi import APIRouter, HTTPException

from config.settings import settings
from data.news_collector import NewsCollector
from data.storage import DatabaseManager
from ai.sentiment import SentimentAnalyzer

logger = logging.getLogger(__name__)
router = APIRouter()

collector = NewsCollector()
analyzer = SentimentAnalyzer()


@router.post("/collect/{symbol}")
async def collect_and_analyze(symbol: str):
    """Collect news for a symbol, analyze sentiment, and save to database."""
    # Step 1: Collect news
    news_items = collector.collect_all(symbol)
    if not news_items:
        return {
            "symbol": symbol,
            "collected": 0,
            "saved": 0,
            "news": [],
            "overall": {"overall_summary": "ไม่พบข่าวสำหรับหุ้นนี้", "overall_score": 0, "overall_label": "neutral"},
        }

    # Step 2: Analyze sentiment
    overall = {"overall_summary": "", "overall_score": 0, "overall_label": "neutral"}
    if analyzer.is_available():
        news_items, overall = analyzer.analyze_batch(news_items, symbol)

    # Step 3: Save to database
    db = DatabaseManager(settings.db_path)
    saved = db.save_news_sentiment(news_items)

    return {
        "symbol": symbol,
        "collected": len(news_items),
        "saved": saved,
        "news": news_items,
        "overall": overall,
    }


@router.get("/{symbol}")
async def get_news(symbol: str, limit: int = 50, period: str = "30d"):
    """Get stored news articles with sentiment for a symbol.
    period: '1d', '7d', '30d', 'all'
    """
    db = DatabaseManager(settings.db_path)
    news = db.get_news_sentiment(symbol, limit=limit, period=period)
    return {"symbol": symbol, "news": news, "total": len(news), "period": period}


@router.get("/{symbol}/summary")
async def get_market_summary(symbol: str, period: str = "7d"):
    """Generate AI market summary from recent news."""
    db = DatabaseManager(settings.db_path)
    news = db.get_news_sentiment(symbol, limit=30, period=period)

    if not news:
        return {
            "symbol": symbol,
            "summary": "ยังไม่มีข่าวในระบบ กรุณากด Collect News ก่อน",
            "overall_score": 0,
            "overall_label": "neutral",
            "news_count": 0,
        }

    # Calculate aggregate stats
    scores = [n.get("sentiment_score", 0) for n in news]
    avg_score = sum(scores) / len(scores) if scores else 0
    bullish = sum(1 for s in scores if s > 0.1)
    bearish = sum(1 for s in scores if s < -0.1)
    label = "bullish" if avg_score > 0.15 else "bearish" if avg_score < -0.15 else "neutral"

    return {
        "symbol": symbol,
        "overall_score": round(avg_score, 3),
        "overall_label": label,
        "news_count": len(news),
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": len(news) - bullish - bearish,
        "period": period,
    }


@router.get("/{symbol}/daily")
async def get_daily_sentiment(symbol: str, days: int = 30):
    """Get daily average sentiment (for feature integration)."""
    db = DatabaseManager(settings.db_path)
    daily = db.get_daily_sentiment(symbol, days=days)
    return {"symbol": symbol, "daily_sentiment": daily}
