"""
Aggregator: fans out all data source fetches for a single entity concurrently
and assembles a FootprintData object.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime

from src.models import Entity, FootprintData, InternalVendorRecord
from src.data_sources.yfinance_client import fetch_financial_metrics
from src.data_sources.news_client import fetch_news
from src.data_sources.sec_edgar import fetch_recent_filings
from src.data_sources.wikipedia_client import fetch_company_description
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Internal vendor registry loader ───────────────────────────────────────────

_vendor_registry_cache: dict[str, InternalVendorRecord] = {}


def load_vendor_registry(path: Path | None = None) -> dict[str, InternalVendorRecord]:
    """Load synthetic internal vendor registry from JSON. Keyed by lower-case vendor name."""
    global _vendor_registry_cache
    if _vendor_registry_cache:
        return _vendor_registry_cache

    fpath = path or settings.synthetic_data_path
    if not fpath.exists():
        logger.warning(f"Vendor registry not found at {fpath}")
        return {}

    with open(fpath) as f:
        data = json.load(f)

    for record in data.get("vendor_records", []):
        obj = InternalVendorRecord(**record)
        _vendor_registry_cache[obj.vendor_name.lower()] = obj

    logger.info(f"Loaded {len(_vendor_registry_cache)} internal vendor records")
    return _vendor_registry_cache


def get_internal_record(entity_name: str) -> InternalVendorRecord | None:
    registry = load_vendor_registry()
    return registry.get(entity_name.lower())


# ── Main aggregation function ─────────────────────────────────────────────────

async def aggregate_entity_footprint(
    entity: Entity,
    newsapi_key: str = "",
    sec_user_agent: str = "VendorRiskIntel/1.0 dev@example.com",
) -> FootprintData:
    """
    Fetch all available data for a single entity concurrently.
    Returns a populated FootprintData object.
    """
    logger.info(f"Aggregating footprint for: {entity.name}")

    # Fan out all fetches concurrently
    financials_task   = fetch_financial_metrics(entity.id, entity.ticker)
    news_task         = fetch_news(entity.name, entity.id, newsapi_key=newsapi_key)
    filings_task      = fetch_recent_filings(entity.id, entity.ticker, sec_user_agent)
    description_task  = fetch_company_description(entity.name)

    financials, news_items, filings, description = await asyncio.gather(
        financials_task,
        news_task,
        filings_task,
        description_task,
        return_exceptions=True,
    )

    # Handle exceptions gracefully — partial data is better than failure
    if isinstance(financials, Exception):
        logger.warning(f"Financials failed for {entity.name}: {financials}")
        financials = None
    if isinstance(news_items, Exception):
        logger.warning(f"News failed for {entity.name}: {news_items}")
        news_items = []
    if isinstance(filings, Exception):
        logger.warning(f"Filings failed for {entity.name}: {filings}")
        filings = []
    if isinstance(description, Exception):
        description = ""

    # Internal vendor record (synchronous lookup)
    internal_record = get_internal_record(entity.name)

    # Compute derived fields
    news_sentiments = [n.sentiment_score for n in news_items] if news_items else []
    avg_sentiment   = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
    neg_count       = sum(1 for s in news_sentiments if s < -0.1)
    risk_headlines  = [n.title for n in news_items if n.risk_relevant][:5]

    return FootprintData(
        entity_id=entity.id,
        entity_name=entity.name,
        fetch_date=datetime.utcnow(),
        financials=financials,
        news_items=news_items or [],
        sec_filings=filings or [],
        internal_record=internal_record,
        description=description or entity.description,
        news_sentiment_avg=round(avg_sentiment, 3),
        negative_news_count=neg_count,
        risk_news_headlines=risk_headlines,
    )


async def aggregate_all_entities(
    entities: list[Entity],
    newsapi_key: str = "",
    sec_user_agent: str = "",
    concurrency: int = 5,
) -> dict[str, FootprintData]:
    """
    Aggregate footprints for all entities with bounded concurrency.
    Uses a semaphore to avoid hammering public APIs.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_fetch(entity: Entity) -> tuple[str, FootprintData]:
        async with semaphore:
            fp = await aggregate_entity_footprint(entity, newsapi_key, sec_user_agent)
            return entity.id, fp

    tasks = [bounded_fetch(e) for e in entities]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    footprints: dict[str, FootprintData] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Footprint aggregation error: {result}")
            continue
        entity_id, fp = result
        footprints[entity_id] = fp

    logger.info(f"Aggregated footprints for {len(footprints)}/{len(entities)} entities")
    return footprints
