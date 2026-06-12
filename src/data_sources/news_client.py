"""
News fetching from GDELT (no API key) and NewsAPI (optional key).
Computes simple rule-based sentiment scores on headlines.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx

from src.models import NewsItem, SourceProvenanceAnchor

logger = logging.getLogger(__name__)


def build_news_anchor(item: NewsItem, index: int = 0) -> SourceProvenanceAnchor:
    """Build a provenance anchor for a single news headline."""
    return SourceProvenanceAnchor(
        anchor_key=f"[NEWS-{item.published_at.strftime('%Y-%m-%d')}-{index}]",
        source_name=item.source or "News wire",
        source_url=item.url or "",
        section_reference="Headline",
        extracted_at=datetime.utcnow(),
        verbatim_snippet=item.title[:240],
    )

# ── Sentiment word lists (lightweight, no model needed locally) ────────────────

NEGATIVE_WORDS = {
    "bankrupt", "bankruptcy", "default", "fraud", "lawsuit", "sanction",
    "recall", "investigation", "scandal", "loss", "losses", "decline",
    "downgrade", "warning", "risk", "concern", "fail", "failure", "halt",
    "layoff", "layoffs", "cut", "cuts", "restructur", "probe", "fine",
    "penalty", "violation", "breach", "hack", "cyberattack", "shortage",
    "disruption", "delay", "strike", "fire", "explosion", "accident",
}

POSITIVE_WORDS = {
    "profit", "growth", "record", "beat", "exceed", "expand", "expansion",
    "partnership", "agreement", "contract", "award", "launch", "innovate",
    "upgrade", "increase", "rise", "gain", "strong", "robust", "dividend",
}


def _simple_sentiment(text: str) -> float:
    """Returns score between -1 and +1 based on keyword matching."""
    words = text.lower().split()
    neg = sum(1 for w in words if any(n in w for n in NEGATIVE_WORDS))
    pos = sum(1 for w in words if any(p in w for p in POSITIVE_WORDS))
    total = neg + pos
    if total == 0:
        return 0.0
    return round((pos - neg) / total, 3)


def _is_risk_relevant(title: str) -> bool:
    risk_terms = {
        "bankrupt", "sanction", "fraud", "recall", "shutdown", "hack",
        "breach", "lawsuit", "investigation", "strike", "shortage",
        "disruption", "fire", "explosion", "default", "downgrade",
    }
    title_lower = title.lower()
    return any(term in title_lower for term in risk_terms)


# ── GDELT client (no API key required) ────────────────────────────────────────

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


async def fetch_gdelt_news(
    entity_name: str,
    entity_id: str,
    days_back: int = 30,
    max_articles: int = 15,
) -> list[NewsItem]:
    """Fetch recent news from GDELT for a given company name."""
    params = {
        "query": f'"{entity_name}" sourcelang:english',
        "mode": "artlist",
        "maxrecords": max_articles,
        "format": "json",
        "timespan": f"{days_back}d",
    }
    items: list[NewsItem] = []
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(GDELT_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            for art in articles:
                title = art.get("title", "")
                pub_str = art.get("seendate", "")
                try:
                    pub_dt = datetime.strptime(pub_str[:14], "%Y%m%dT%H%M%S")
                except Exception:
                    pub_dt = datetime.utcnow() - timedelta(days=1)
                sentiment = _simple_sentiment(title)
                items.append(NewsItem(
                    entity_id=entity_id,
                    title=title,
                    source=art.get("domain", "unknown"),
                    published_at=pub_dt,
                    url=art.get("url", ""),
                    sentiment_score=sentiment,
                    risk_relevant=_is_risk_relevant(title),
                    summary="",
                ))
        except Exception as e:
            logger.warning(f"GDELT fetch failed for '{entity_name}': {e}")
    return items


# ── NewsAPI client (optional key) ─────────────────────────────────────────────

NEWSAPI_URL = "https://newsapi.org/v2/everything"


async def fetch_newsapi(
    entity_name: str,
    entity_id: str,
    api_key: str,
    days_back: int = 14,
    max_articles: int = 10,
) -> list[NewsItem]:
    """Fetch news from NewsAPI.org (requires free API key)."""
    if not api_key:
        return []
    from_date = (datetime.utcnow() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": entity_name,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": max_articles,
        "language": "en",
        "apiKey": api_key,
    }
    items: list[NewsItem] = []
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(NEWSAPI_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            for art in data.get("articles", []):
                title = art.get("title", "") or ""
                content = (art.get("description", "") or "")
                text = f"{title} {content}"
                pub_str = art.get("publishedAt", "")
                try:
                    pub_dt = datetime.strptime(pub_str, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pub_dt = datetime.utcnow()
                items.append(NewsItem(
                    entity_id=entity_id,
                    title=title,
                    source=art.get("source", {}).get("name", "unknown"),
                    published_at=pub_dt,
                    url=art.get("url", ""),
                    sentiment_score=_simple_sentiment(text),
                    risk_relevant=_is_risk_relevant(title),
                    summary=art.get("description", "")[:200],
                ))
        except Exception as e:
            logger.warning(f"NewsAPI fetch failed for '{entity_name}': {e}")
    return items


async def fetch_news(
    entity_name: str,
    entity_id: str,
    newsapi_key: str = "",
    days_back: int = 30,
) -> list[NewsItem]:
    """
    Fetch news from all available sources, deduplicate, and return.
    GDELT is always tried first; NewsAPI is layered on top if key present.
    """
    gdelt_task = fetch_gdelt_news(entity_name, entity_id, days_back=days_back)
    newsapi_task = fetch_newsapi(entity_name, entity_id, newsapi_key, days_back=days_back)
    gdelt_results, newsapi_results = await asyncio.gather(gdelt_task, newsapi_task)

    # Merge and deduplicate by title
    all_items = gdelt_results + newsapi_results
    seen_titles: set[str] = set()
    unique: list[NewsItem] = []
    for item in all_items:
        if item.title not in seen_titles:
            seen_titles.add(item.title)
            unique.append(item)

    # Sort by date descending
    unique.sort(key=lambda x: x.published_at, reverse=True)
    return unique
