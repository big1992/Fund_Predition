"""
AI-powered sentiment analysis for financial news using OpenAI GPT.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze financial news sentiment using GPT."""

    def __init__(self):
        self.client = None
        self.model = "gpt-4o-mini"
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
        except Exception as e:
            logger.warning("Failed to init OpenAI for sentiment: %s", e)

    def is_available(self) -> bool:
        return self.client is not None

    def analyze_batch(self, news_items: list[dict], symbol: str = "") -> list[dict]:
        """
        Analyze sentiment for a batch of news articles.
        Returns enriched news items with sentiment_score, sentiment_label, etc.
        """
        if not self.client or not news_items:
            return news_items

        # Build news text for GPT
        news_texts = []
        for i, item in enumerate(news_items[:15]):  # Limit to 15
            news_texts.append(f"{i+1}. [{item.get('source', '?')}] {item['title']}")
            if item.get("summary"):
                news_texts.append(f"   {item['summary'][:200]}")

        news_block = "\n".join(news_texts)

        prompt = f"""Analyze the sentiment of these financial news articles related to {symbol or 'the stock market'}.

For EACH article, provide:
- sentiment_score: float from -1.0 (very bearish) to +1.0 (very bullish), 0 = neutral
- sentiment_label: "bullish", "bearish", or "neutral"
- key_topics: list of 2-3 key topics
- summary_th: 1-sentence Thai summary

Also provide an overall_summary in Thai summarizing the market mood.

NEWS:
{news_block}

Respond in STRICT JSON format:
{{
  "articles": [
    {{"index": 1, "sentiment_score": 0.5, "sentiment_label": "bullish", "key_topics": ["earnings", "growth"], "summary_th": "..."}},
    ...
  ],
  "overall_summary": "สรุปภาพรวม...",
  "overall_score": 0.3,
  "overall_label": "slightly_bullish"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analyst. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content.strip()
            # Clean markdown fences
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1]
                if raw.endswith("```"):
                    raw = raw[:-3]

            result = json.loads(raw)
            articles = result.get("articles", [])

            # Merge sentiment data back into news items
            for art in articles:
                idx = art.get("index", 0) - 1
                if 0 <= idx < len(news_items):
                    news_items[idx]["sentiment_score"] = float(art.get("sentiment_score", 0))
                    news_items[idx]["sentiment_label"] = art.get("sentiment_label", "neutral")
                    news_items[idx]["key_topics"] = art.get("key_topics", [])
                    news_items[idx]["summary_th"] = art.get("summary_th", "")

            # Set defaults for items not analyzed
            for item in news_items:
                if "sentiment_score" not in item:
                    item["sentiment_score"] = 0.0
                    item["sentiment_label"] = "neutral"
                    item["key_topics"] = []
                    item["summary_th"] = ""

            logger.info(
                "Sentiment analysis complete: %d articles, overall=%s (%.2f)",
                len(articles),
                result.get("overall_label", "?"),
                result.get("overall_score", 0),
            )

            return news_items, {
                "overall_summary": result.get("overall_summary", ""),
                "overall_score": result.get("overall_score", 0),
                "overall_label": result.get("overall_label", "neutral"),
            }

        except json.JSONDecodeError as e:
            logger.error("Failed to parse sentiment JSON: %s", e)
            for item in news_items:
                item.setdefault("sentiment_score", 0.0)
                item.setdefault("sentiment_label", "neutral")
                item.setdefault("key_topics", [])
                item.setdefault("summary_th", "")
            return news_items, {"overall_summary": "", "overall_score": 0, "overall_label": "neutral"}

        except Exception as e:
            logger.error("Sentiment analysis failed: %s", e)
            for item in news_items:
                item.setdefault("sentiment_score", 0.0)
                item.setdefault("sentiment_label", "neutral")
                item.setdefault("key_topics", [])
                item.setdefault("summary_th", "")
            return news_items, {"overall_summary": "", "overall_score": 0, "overall_label": "neutral"}
