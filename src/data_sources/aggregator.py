"""
Aggregator: fans out all data source fetches for a single entity concurrently
and assembles a FootprintData object.
"""

import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

from src.models import (
    Entity, FootprintData, InternalVendorRecord, DriverEvidence,
    FinancialMetrics, NewsItem, SECFiling,
)
from src.data_sources.yfinance_client import (
    fetch_financial_metrics, build_financial_evidence, build_financial_evidence_list,
)
from src.data_sources.news_client import (
    fetch_news, build_news_evidence, fetch_gdelt_country_events,
)
from src.data_sources.sec_edgar import (
    fetch_recent_filings, build_filing_evidence, no_us_filing_evidence,
)
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


# ── Provenance collection ─────────────────────────────────────────────────────

def _collect_provenance(
    entity: Entity,
    financials: FinancialMetrics | None,
    news_items: list[NewsItem],
    filings: list[SECFiling],
    sec_evidence: list[DriverEvidence] | None = None,
    geo_events: list[DriverEvidence] | None = None,
) -> dict[str, DriverEvidence]:
    """
    Build the provenance map (DriverEvidence) for whatever real data was actually
    fetched. Keyed by field path (e.g. "financials.altman_z", "sec.10-K",
    "news.headline_0", "geo.event_0") so the dashboard can attach each piece of
    evidence to the figure it supports.
    """
    anchors: dict[str, DriverEvidence] = {}

    # Financials — anchor only when we have a ticker and at least one real metric.
    if financials is not None and entity.ticker and (financials.data_quality or 0) > 0.2:
        ev = build_financial_evidence(entity.ticker, financials)
        anchors["financials"] = ev
        if financials.altman_z_score is not None:
            anchors["financials.altman_z"] = ev

    # SEC filings — prefer the inline evidence captured at fetch time; fall back
    # to building from the filing objects.
    if sec_evidence:
        for i, ev in enumerate(sec_evidence[:5]):
            anchors[f"sec.{ev.value or i}"] = ev
    else:
        for filing in filings:
            key = f"sec.{filing.form_type}"
            if key not in anchors:
                anchors[key] = build_filing_evidence(filing)

    # News — anchor the first few risk-relevant (or otherwise top) headlines.
    cited = [n for n in news_items if n.risk_relevant] or news_items
    for i, item in enumerate(cited[:3]):
        if item.url:   # only cite a headline we can actually link to
            anchors[f"news.headline_{i}"] = build_news_evidence(item, i)

    # Geopolitical events (country-level GDELT).
    for i, ev in enumerate((geo_events or [])[:3]):
        if ev.source_url:
            anchors[f"geo.event_{i}"] = ev

    return anchors


# ── Mock footprint (offline, no network) ──────────────────────────────────────

# Deterministic per-ticker financial stubs so the mock path produces varied,
# realistic scores and provenance without any network call. Values are
# illustrative but plausible for the worked Apple supply-chain example.
_MOCK_FINANCIALS: dict[str, dict] = {
    "TSM":   {"market_cap_usd": 9.0e11, "revenue_ttm_usd": 8.0e10, "debt_to_equity": 28.0,  "current_ratio": 2.4, "altman_z_score": 6.1,  "revenue_growth_yoy_pct": 9.0},
    "HNHPF": {"market_cap_usd": 5.5e10, "revenue_ttm_usd": 2.1e11, "debt_to_equity": 65.0,  "current_ratio": 1.4, "altman_z_score": 3.2,  "revenue_growth_yoy_pct": -3.0},
    "SSNLF": {"market_cap_usd": 3.7e11, "revenue_ttm_usd": 2.0e11, "debt_to_equity": 12.0,  "current_ratio": 2.1, "altman_z_score": 4.8,  "revenue_growth_yoy_pct": 4.0},
    "AVGO":  {"market_cap_usd": 6.0e11, "revenue_ttm_usd": 4.6e10, "debt_to_equity": 110.0, "current_ratio": 1.0, "altman_z_score": 2.6,  "revenue_growth_yoy_pct": 12.0},
    "GLW":   {"market_cap_usd": 3.0e10, "revenue_ttm_usd": 1.3e10, "debt_to_equity": 78.0,  "current_ratio": 1.5, "altman_z_score": 2.9,  "revenue_growth_yoy_pct": 1.0},
    "T":     {"market_cap_usd": 1.3e11, "revenue_ttm_usd": 1.2e11, "debt_to_equity": 130.0, "current_ratio": 0.7, "altman_z_score": 1.9,  "revenue_growth_yoy_pct": -1.0},
    "AMZN":  {"market_cap_usd": 1.9e12, "revenue_ttm_usd": 5.7e11, "debt_to_equity": 55.0,  "current_ratio": 1.1, "altman_z_score": 4.0,  "revenue_growth_yoy_pct": 11.0},
    "GOOGL": {"market_cap_usd": 2.1e12, "revenue_ttm_usd": 3.1e11, "debt_to_equity": 9.0,   "current_ratio": 2.0, "altman_z_score": 8.5,  "revenue_growth_yoy_pct": 13.0},
    "ARM":   {"market_cap_usd": 1.4e11, "revenue_ttm_usd": 3.2e9,  "debt_to_equity": 5.0,   "current_ratio": 4.5, "altman_z_score": 9.2,  "revenue_growth_yoy_pct": 20.0},
    "ASML":  {"market_cap_usd": 3.5e11, "revenue_ttm_usd": 2.8e10, "debt_to_equity": 30.0,  "current_ratio": 1.6, "altman_z_score": 7.0,  "revenue_growth_yoy_pct": 6.0},
    "AMAT":  {"market_cap_usd": 1.5e11, "revenue_ttm_usd": 2.7e10, "debt_to_equity": 35.0,  "current_ratio": 2.3, "altman_z_score": 6.4,  "revenue_growth_yoy_pct": 3.0},
    "SHECY": {"market_cap_usd": 7.0e10, "revenue_ttm_usd": 1.5e10, "debt_to_equity": 8.0,   "current_ratio": 3.0, "altman_z_score": 5.5,  "revenue_growth_yoy_pct": 2.0},
    "SFTBY": {"market_cap_usd": 9.0e10, "revenue_ttm_usd": 4.2e10, "debt_to_equity": 180.0, "current_ratio": 1.2, "altman_z_score": 1.4,  "revenue_growth_yoy_pct": -8.0},
    "APD":   {"market_cap_usd": 6.0e10, "revenue_ttm_usd": 1.2e10, "debt_to_equity": 70.0,  "current_ratio": 1.8, "altman_z_score": 3.4,  "revenue_growth_yoy_pct": 0.0},
    "LIN":   {"market_cap_usd": 2.1e11, "revenue_ttm_usd": 3.3e10, "debt_to_equity": 60.0,  "current_ratio": 0.9, "altman_z_score": 3.1,  "revenue_growth_yoy_pct": 2.0},
    # Fix 2: Pegatron (TWSE: 4938.TW) — asset-heavy contract manufacturer. Mid-range
    # Altman Z (grey zone), modest growth, moderate leverage, thin liquidity. Stops
    # fin_score collapsing to the 50.0 missing-data default and populates fin_evidence.
    "4938.TW": {"market_cap_usd": 6.0e9, "revenue_ttm_usd": 4.0e10, "debt_to_equity": 85.0,  "current_ratio": 1.1, "altman_z_score": 2.1,  "revenue_growth_yoy_pct": 3.0},
}


def _build_mock_footprint(entity: Entity) -> FootprintData:
    """
    Build a fully-populated FootprintData for an entity without any network
    access. Produces realistic financials, a stub 10-K filing, a stub news
    headline, and the matching provenance anchors so the mock path exercises
    the entire grounding schema end to end.
    """
    from src.data_sources.sec_edgar import NON_US_LISTED
    from src.data_sources.news_client import _COUNTRY_QUERY

    ticker = entity.ticker
    financials: FinancialMetrics | None = None
    filings: list[SECFiling] = []
    news_items: list[NewsItem] = []
    sec_evidence: list[DriverEvidence] = []
    fin_evidence: list[DriverEvidence] = []

    is_foreign = entity.name.lower() in NON_US_LISTED

    if ticker and ticker.upper() in _MOCK_FINANCIALS:
        data = _MOCK_FINANCIALS[ticker.upper()]
        financials = FinancialMetrics(
            entity_id=entity.id,
            fetch_date=datetime.utcnow(),
            data_quality=0.9,
            **data,
        )
        # Issue 5: per-metric financial provenance for the mock path.
        fin_evidence = build_financial_evidence_list(ticker, financials)
        # A risk-flagged 10-K for distressed names; a clean one otherwise.
        # Foreign entities are not US-listed, so they get the no-filings entry.
        if not is_foreign:
            z = data.get("altman_z_score", 5.0)
            flags = ["going concern"] if z < 1.81 else (["impairment"] if z < 3.0 else [])
            filing = SECFiling(
                entity_id=entity.id,
                form_type="10-K",
                filed_at=datetime.utcnow(),
                accession_number="0000000000-24-000000",
                description="10-K annual report (mock)",
                risk_flags=flags,
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={entity.name.replace(' ', '+')}&type=10-K",
            )
            filings.append(filing)
            sec_evidence.append(build_filing_evidence(filing))

    # Compliance evidence is never empty: foreign / no-ticker entities get the
    # explicit no-US-filings entry (E2 mock parity).
    if not sec_evidence:
        sec_evidence.append(no_us_filing_evidence(entity.name, ticker))

    # Geopolitical events — one stub GDELT headline per entity, country-flavoured.
    country = entity.hq_country or "US"
    country_name = _COUNTRY_QUERY.get(country, country)
    geo_events: list[DriverEvidence] = [DriverEvidence(
        label=f"GDELT ({country_name}): trade-policy and supply-chain coverage monitored for {country_name}",
        source_url=f"https://www.gdeltproject.org/search/?query={country_name.replace(' ', '+')}",
        retrieved_at=datetime.utcnow(),
        value=datetime.utcnow().strftime("%Y-%m-%d"),
    )]

    # A single representative headline per entity (sentiment follows financial health).
    if ticker:
        z = _MOCK_FINANCIALS.get(ticker.upper(), {}).get("altman_z_score", 5.0)
        if z < 1.81:
            title = f"{entity.name} faces liquidity pressure as covenant concerns mount"
            risk = True
        elif z < 3.0:
            title = f"{entity.name} warns of margin headwinds amid supply constraints"
            risk = True
        else:
            title = f"{entity.name} reports steady demand across core product lines"
            risk = False
        news_items.append(NewsItem(
            entity_id=entity.id,
            title=title,
            source="Reuters",
            published_at=datetime.utcnow() - timedelta(days=2),
            url=f"https://www.reuters.com/search/news?blob={entity.name.replace(' ', '+')}",
            sentiment_score=-0.4 if risk else 0.3,
            risk_relevant=risk,
            summary="",
        ))

    internal_record = get_internal_record(entity.name)
    anchors = _collect_provenance(entity, financials, news_items, filings,
                                  sec_evidence=sec_evidence, geo_events=geo_events)

    news_sentiments = [n.sentiment_score for n in news_items]
    avg_sentiment   = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
    neg_count       = sum(1 for s in news_sentiments if s < -0.1)
    risk_headlines  = [n.title for n in news_items if n.risk_relevant][:5]

    return FootprintData(
        entity_id=entity.id,
        entity_name=entity.name,
        fetch_date=datetime.utcnow(),
        financials=financials,
        news_items=news_items,
        sec_filings=filings,
        internal_record=internal_record,
        description=entity.description,
        news_sentiment_avg=round(avg_sentiment, 3),
        negative_news_count=neg_count,
        risk_news_headlines=risk_headlines,
        provenance_anchors=anchors,
        sec_evidence=sec_evidence,
        geo_events=geo_events,
        fin_evidence=fin_evidence,
    )


# ── Main aggregation function ─────────────────────────────────────────────────

async def aggregate_entity_footprint(
    entity: Entity,
    newsapi_key: str = "",
    sec_user_agent: str = "VendorRiskIntel/1.0 dev@example.com",
    llm_backend: str = "mock",
) -> FootprintData:
    """
    Fetch all available data for a single entity concurrently.
    Returns a populated FootprintData object.

    In mock mode we skip the network entirely and return deterministic stub data
    plus provenance anchors, so the offline demo exercises the full grounding
    schema without depending on live yfinance/SEC/GDELT availability.
    """
    if llm_backend == "mock":
        logger.info(f"Aggregating MOCK footprint for: {entity.name}")
        return _build_mock_footprint(entity)

    logger.info(f"Aggregating footprint for: {entity.name}")

    # Fan out all fetches concurrently
    financials_task   = fetch_financial_metrics(entity.id, entity.ticker)
    news_task         = fetch_news(entity.name, entity.id, newsapi_key=newsapi_key)
    filings_task      = fetch_recent_filings(
        entity.id, entity.ticker, sec_user_agent, entity_name=entity.name
    )
    description_task  = fetch_company_description(entity.name)
    # Geopolitical events are jurisdiction-level — fetched once per country and
    # cached inside news_client, regardless of how many vendors share a country.
    geo_task          = fetch_gdelt_country_events(entity.hq_country)

    financials, news_items, filings_result, description, geo_events = await asyncio.gather(
        financials_task,
        news_task,
        filings_task,
        description_task,
        geo_task,
        return_exceptions=True,
    )

    # Handle exceptions gracefully — partial data is better than failure
    if isinstance(financials, Exception):
        logger.warning(f"Financials failed for {entity.name}: {financials}")
        financials = None
    if isinstance(news_items, Exception):
        logger.warning(f"News failed for {entity.name}: {news_items}")
        news_items = []
    if isinstance(filings_result, Exception):
        logger.warning(f"Filings failed for {entity.name}: {filings_result}")
        filings_result = ([], [no_us_filing_evidence(entity.name, entity.ticker)])
    if isinstance(description, Exception):
        description = ""
    if isinstance(geo_events, Exception):
        geo_events = []

    filings, sec_evidence = filings_result
    # Compliance evidence is never empty.
    if not sec_evidence:
        sec_evidence = [no_us_filing_evidence(entity.name, entity.ticker)]

    # Issue 5: per-metric financial provenance (live path).
    fin_evidence = (
        build_financial_evidence_list(entity.ticker, financials)
        if financials is not None and entity.ticker else []
    )

    # Internal vendor record (synchronous lookup)
    internal_record = get_internal_record(entity.name)

    # Compute derived fields
    news_sentiments = [n.sentiment_score for n in news_items] if news_items else []
    avg_sentiment   = sum(news_sentiments) / len(news_sentiments) if news_sentiments else 0.0
    neg_count       = sum(1 for s in news_sentiments if s < -0.1)
    risk_headlines  = [n.title for n in news_items if n.risk_relevant][:5]

    # Record provenance for whatever real data we actually fetched.
    anchors = _collect_provenance(
        entity, financials, news_items or [], filings or [],
        sec_evidence=sec_evidence, geo_events=geo_events or [],
    )

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
        provenance_anchors=anchors,
        sec_evidence=sec_evidence,
        geo_events=geo_events or [],
        fin_evidence=fin_evidence,
    )


async def aggregate_all_entities(
    entities: list[Entity],
    newsapi_key: str = "",
    sec_user_agent: str = "",
    concurrency: int = 5,
    llm_backend: str = "mock",
) -> dict[str, FootprintData]:
    """
    Aggregate footprints for all entities with bounded concurrency.
    Uses a semaphore to avoid hammering public APIs.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_fetch(entity: Entity) -> tuple[str, FootprintData]:
        async with semaphore:
            fp = await aggregate_entity_footprint(
                entity, newsapi_key, sec_user_agent, llm_backend=llm_backend
            )
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
