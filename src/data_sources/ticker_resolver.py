"""
Ticker resolver — grounds LLM-generated entities to real market symbols.

The watchlist LLM is unreliable at emitting correct stock tickers: it hallucinates
symbols or returns null, which collapses every downstream financial/SEC/news fetch
and starves the risk scores. This module resolves each entity's *company name* to a
real Yahoo Finance symbol via the public search endpoint, then writes it back onto
the entity. Results are cached so repeat runs are fast and deterministic and the
demo does not depend on a live network call.

Entities with no resolvable public listing are kept (not dropped) and marked
``is_public = False`` so the rest of the pipeline can label them honestly as
private/unlisted rather than silently defaulting their data.

This is best-effort grounding: name→symbol search can occasionally mis-match an
obscure private supplier. That trade-off is disclosed in the README.
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Optional

import httpx

from config.settings import settings

if TYPE_CHECKING:
    from src.models import Entity

logger = logging.getLogger(__name__)

YAHOO_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_CACHE_FILE = settings.cache_dir / "ticker_resolutions.json"
_HEADERS = {"User-Agent": "VendorRiskIntel/1.0 (vendor-risk-research)"}
_REQUEST_TIMEOUT = 8
_CONCURRENCY = 6

# Sentinel cached for "searched, nothing found" so we don't re-hit the network.
_UNLISTED = "__UNLISTED__"

# Exchange preference for picking a *primary* listing over cross-listings / BDRs.
# Higher = preferred. Yahoo's top search hit is often a foreign depositary receipt
# (e.g. TSMC34.SA, AP3.DE) that has sparse fundamentals in yfinance; we down-rank
# those in favour of the home/primary exchange.
_EXCHANGE_RANK = {
    "NMS": 10, "NGM": 10, "NYQ": 10, "NCM": 9, "ASE": 9, "PCX": 8,   # US majors
    "PNK": 7, "OQB": 6, "OQX": 6,                                     # US OTC (ADRs)
    "TAI": 8, "TWO": 7,                                               # Taiwan
    "KSC": 8, "KOE": 7,                                               # Korea
    "JPX": 8, "TYO": 8, "OSA": 7,                                     # Japan
    "LSE": 8,                                                         # London
    "AMS": 8, "EBS": 7, "PAR": 7,                                     # EU primaries
    "GER": 4, "FRA": 3, "STU": 2, "MUN": 2, "BER": 2, "HAM": 2,      # German venues
    "DUS": 2, "SAO": 1, "MEX": 1, "SGO": 1, "BUE": 1, "SES": 2,      # other secondaries
}
_DEFAULT_EXCHANGE_RANK = 5


# ── Cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> dict[str, str]:
    if _CACHE_FILE.exists():
        try:
            return json.loads(_CACHE_FILE.read_text())
        except Exception as e:
            logger.warning(f"Ticker cache unreadable ({e}); starting fresh")
    return {}


def _save_cache(cache: dict[str, str]) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except Exception as e:
        logger.warning(f"Could not persist ticker cache: {e}")


# ── Search ─────────────────────────────────────────────────────────────────────

def _name_match_bonus(query: str, quote: dict) -> int:
    """Small bonus when the quote's name contains query tokens (>=4 chars)."""
    label = f"{quote.get('shortname', '')} {quote.get('longname', '')}".lower()
    tokens = [t for t in query.lower().replace(",", " ").split() if len(t) >= 4]
    return sum(1 for t in tokens if t in label)


async def _validate_symbol(client: httpx.AsyncClient, symbol: str) -> bool:
    """True if the symbol is a real, quotable security (Yahoo chart returns data)."""
    try:
        r = await client.get(
            YAHOO_CHART_URL.format(symbol=symbol),
            params={"range": "5d", "interval": "1d"},
            headers=_HEADERS,
        )
        if r.status_code != 200:
            return False
        return bool(r.json().get("chart", {}).get("result"))
    except Exception:
        return False


async def _search_symbol(client: httpx.AsyncClient, name: str) -> Optional[str]:
    """
    Return the best primary-listing EQUITY symbol for a company name, or None.
    Ranks candidates by exchange preference + name match so we pick the home/major
    listing rather than a thinly-traded cross-listing or BDR.
    """
    try:
        r = await client.get(
            YAHOO_SEARCH_URL,
            params={"q": name, "quotesCount": 10, "newsCount": 0},
            headers=_HEADERS,
        )
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
    except Exception as e:
        logger.warning(f"Ticker search failed for '{name}': {e}")
        return None

    equities = [q for q in quotes if q.get("quoteType") == "EQUITY" and q.get("symbol")]
    candidates = equities or [q for q in quotes if q.get("symbol")]
    if not candidates:
        return None

    def score(q: dict) -> tuple[int, int]:
        exch = _EXCHANGE_RANK.get(q.get("exchange", ""), _DEFAULT_EXCHANGE_RANK)
        return (exch + _name_match_bonus(name, q), -len(q["symbol"]))

    best = max(candidates, key=score)
    return best["symbol"]


# ── Per-entity resolution ──────────────────────────────────────────────────────

async def _resolve_one(
    client: httpx.AsyncClient,
    entity: "Entity",
    cache: dict[str, str],
    cache_lock: asyncio.Lock,
) -> str:
    """
    Resolve one entity's ticker in place. Returns an outcome tag for logging:
    ``cached`` | ``validated`` | ``resolved`` | ``kept_llm`` | ``unlisted``.

    Order matters: we trust a correct LLM ticker over a name search, because the
    search can mis-match a foreign cross-listing. We only search when the LLM gave
    no ticker, or gave one that isn't a real security.
    """
    key = entity.name.strip().lower()
    llm_ticker = entity.ticker

    # 1. Cache hit — authoritative, no network.
    if key in cache:
        cached = cache[key]
        if cached == _UNLISTED:
            entity.ticker = None
            entity.is_public = False
        else:
            entity.ticker = cached
            entity.is_public = True
        return "cached"

    # 2. Validate an LLM-provided ticker first — keep it if it's a real security.
    if llm_ticker and await _validate_symbol(client, llm_ticker):
        entity.is_public = True
        async with cache_lock:
            cache[key] = llm_ticker
        return "validated"

    # 3. No (valid) ticker — search by company name for the primary listing.
    symbol = await _search_symbol(client, entity.name)
    if symbol:
        entity.ticker = symbol
        entity.is_public = True
        async with cache_lock:
            cache[key] = symbol
        return "resolved"

    # 4. Search found nothing. If the LLM gave a ticker, keep it as a best-effort
    #    fallback (likely a transient search miss) and do NOT cache, so a later run
    #    can still resolve it properly.
    if llm_ticker:
        entity.ticker = llm_ticker
        entity.is_public = True
        return "kept_llm"

    # 5. No public listing at all — keep the node, label it honestly.
    entity.ticker = None
    entity.is_public = False
    async with cache_lock:
        cache[key] = _UNLISTED
    return "unlisted"


# ── Public API ─────────────────────────────────────────────────────────────────

async def resolve_entity_tickers(entities: list["Entity"]) -> dict[str, int]:
    """
    Resolve/validate tickers for all entities in place. Cache-first, network only
    for misses, fully concurrent. Never raises — resolution is best-effort and
    degrades gracefully. Returns a summary dict (outcome → count) for logging.
    """
    if not entities:
        return {}

    cache = _load_cache()
    cache_lock = asyncio.Lock()
    sem = asyncio.Semaphore(_CONCURRENCY)

    async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
        async def bounded(entity: "Entity") -> str:
            async with sem:
                return await _resolve_one(client, entity, cache, cache_lock)

        outcomes = await asyncio.gather(
            *(bounded(e) for e in entities), return_exceptions=True
        )

    summary: dict[str, int] = {}
    for o in outcomes:
        if isinstance(o, Exception):
            logger.warning(f"Ticker resolution task errored: {o}")
            continue
        summary[o] = summary.get(o, 0) + 1

    _save_cache(cache)
    logger.info(
        "Ticker resolution: "
        + " ".join(f"{k}={v}" for k, v in sorted(summary.items()))
    )
    return summary
