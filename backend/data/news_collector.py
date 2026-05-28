"""
News collector: fetches financial news from Yahoo Finance, NewsAPI, and RSS feeds.
"""

import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collect financial news from multiple sources."""

    # RSS feeds for market news
    RSS_FEEDS = [
        {
            "name": "Investing.com",
            "url": "https://www.investing.com/rss/news_301.rss",
            "category": "global",
        },
        {
            "name": "CNBC Asia",
            "url": "https://www.cnbc.com/id/104568959/device/rss/rss.html",
            "category": "asia",
        },
    ]

    # Map stock symbols to search keywords for NewsAPI
    SYMBOL_KEYWORDS = {
        # Thai
        "PTT.BK": "PTT Thailand oil energy",
        "ADVANC.BK": "AIS Advanced Info Thailand telecom",
        "SCC.BK": "SCG Siam Cement Thailand",
        "KBANK.BK": "Kasikornbank Thailand banking",
        "SCB.BK": "SCB Thailand banking",
        "AOT.BK": "AOT Airports Thailand aviation",
        "CPALL.BK": "CP All Thailand retail",
        "GULF.BK": "Gulf Energy Thailand power",
        "BEM.BK": "BEM Bangkok Expressway Metro",
        # US
        "AAPL": "Apple iPhone Mac",
        "MSFT": "Microsoft Azure AI",
        "GOOGL": "Google Alphabet search AI",
        "AMZN": "Amazon AWS ecommerce",
        "NVDA": "NVIDIA GPU AI chips",
        "TSLA": "Tesla electric vehicles Elon Musk",
        "META": "Meta Facebook Instagram",
        "JPM": "JPMorgan Chase banking",
        "V": "Visa payments fintech",
        "JNJ": "Johnson Johnson pharmaceutical",
        "XOM": "Exxon Mobil oil energy",
        "WMT": "Walmart retail",
        "TSM": "TSMC Taiwan Semiconductor chips",
        "BRK-B": "Berkshire Hathaway Warren Buffett",
        "COIN": "Coinbase crypto bitcoin",
    }

    def collect_yahoo_news(self, symbol: str) -> list[dict]:
        """Collect news from Yahoo Finance for a specific stock."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            raw_news = ticker.news or []

            news_items = []
            for item in raw_news[:10]:
                # yfinance 0.2.x returns different structure
                if isinstance(item, dict):
                    # Try new format first
                    content = item.get("content", {}) if isinstance(item.get("content"), dict) else {}
                    title = content.get("title") or item.get("title", "No title")
                    summary = content.get("summary") or item.get("summary", "")
                    
                    provider = content.get("provider", {})
                    source = provider.get("displayName", "Yahoo Finance") if isinstance(provider, dict) else "Yahoo Finance"
                    
                    canonical = content.get("canonicalUrl", {})
                    url = canonical.get("url", item.get("link", "")) if isinstance(canonical, dict) else item.get("link", "")
                    
                    pub_time = content.get("pubDate") or item.get("providerPublishTime", 0)
                    if isinstance(pub_time, (int, float)) and pub_time > 0:
                        published = datetime.fromtimestamp(pub_time).isoformat()
                    else:
                        published = datetime.now().isoformat()
                    
                    news_items.append({
                        "title": title,
                        "summary": summary[:500] if summary else "",
                        "source": source,
                        "url": url,
                        "published_at": published,
                        "symbol": symbol,
                        "source_type": "yahoo",
                    })

            logger.info("Collected %d Yahoo news for %s", len(news_items), symbol)
            return news_items

        except Exception as e:
            logger.error("Yahoo news collection failed for %s: %s", symbol, e)
            return []

    def collect_newsapi(self, symbol: str) -> list[dict]:
        """Collect news from NewsAPI.org."""
        import requests

        api_key = os.getenv("NEWSAPI_KEY")
        if not api_key:
            logger.warning("NEWSAPI_KEY not set, skipping NewsAPI")
            return []

        # Build search query from symbol
        query = self.SYMBOL_KEYWORDS.get(symbol, symbol.replace(".BK", "") + " Thailand stock")

        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 10,
                    "apiKey": api_key,
                },
                timeout=15,
            )
            data = resp.json()

            if data.get("status") != "ok":
                logger.warning("NewsAPI error: %s", data.get("message", "Unknown"))
                return []

            news_items = []
            for article in data.get("articles", []):
                news_items.append({
                    "title": article.get("title", "No title"),
                    "summary": (article.get("description") or "")[:500],
                    "source": article.get("source", {}).get("name", "NewsAPI"),
                    "url": article.get("url", ""),
                    "published_at": article.get("publishedAt", datetime.now().isoformat()),
                    "symbol": symbol,
                    "source_type": "newsapi",
                })

            logger.info("Collected %d NewsAPI articles for %s", len(news_items), symbol)
            return news_items

        except Exception as e:
            logger.error("NewsAPI collection failed: %s", e)
            return []

    def collect_rss_news(self, symbol: Optional[str] = None) -> list[dict]:
        """Collect news from RSS feeds."""
        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed, skipping RSS")
            return []

        news_items = []
        for feed_config in self.RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_config["url"])
                for entry in feed.entries[:5]:
                    published = ""
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6]).isoformat()
                    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                        published = datetime(*entry.updated_parsed[:6]).isoformat()
                    else:
                        published = datetime.now().isoformat()

                    news_items.append({
                        "title": getattr(entry, "title", "No title"),
                        "summary": getattr(entry, "summary", "")[:500],
                        "source": feed_config["name"],
                        "url": getattr(entry, "link", ""),
                        "published_at": published,
                        "symbol": symbol or "MARKET",
                        "source_type": "rss",
                    })
            except Exception as e:
                logger.warning("RSS fetch failed for %s: %s", feed_config["name"], e)

        logger.info("Collected %d RSS news items", len(news_items))
        return news_items

    def collect_all(self, symbol: str) -> list[dict]:
        """Collect news from all sources."""
        all_news = []
        all_news.extend(self.collect_yahoo_news(symbol))
        all_news.extend(self.collect_newsapi(symbol))
        all_news.extend(self.collect_rss_news(symbol))

        # Deduplicate by title
        seen = set()
        unique = []
        for item in all_news:
            title_key = item["title"].strip().lower()[:50]
            if title_key not in seen:
                seen.add(title_key)
                unique.append(item)

        logger.info("Total unique news for %s: %d", symbol, len(unique))
        return unique
