"""
Fetches company descriptions from Wikipedia.
Used to populate entity.description and provide context to the LLM.
"""

import logging
import asyncio
from functools import lru_cache

import wikipediaapi

logger = logging.getLogger(__name__)

_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="VendorRiskIntel/1.0",
)


@lru_cache(maxsize=256)
def _fetch_wiki_summary_sync(company_name: str) -> str:
    """Cached synchronous Wikipedia summary fetch."""
    page = _wiki.page(company_name)
    if page.exists():
        return page.summary[:800]   # First 800 chars is plenty for context
    # Try without common suffixes
    for suffix in [" Inc", " Corp", " Ltd", " Holdings", " Group"]:
        short = company_name.replace(suffix, "").strip()
        page = _wiki.page(short)
        if page.exists():
            return page.summary[:800]
    return ""


async def fetch_company_description(company_name: str) -> str:
    """Async wrapper for Wikipedia summary fetch."""
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _fetch_wiki_summary_sync, company_name)
    except Exception as e:
        logger.warning(f"Wikipedia fetch failed for '{company_name}': {e}")
        return ""
