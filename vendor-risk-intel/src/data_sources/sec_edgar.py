"""
Fetches SEC EDGAR filings for US-listed companies.
Uses the public EDGAR full-text search API — no API key required.
https://efts.sec.gov/LATEST/search-index?q=...
"""

import logging
from datetime import datetime
from typing import Optional

import httpx

from src.models import SECFiling

logger = logging.getLogger(__name__)

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"

# Keywords that indicate risk-relevant filings
RISK_KEYWORDS = [
    "material weakness", "going concern", "restatement", "impairment",
    "sanctions", "investigation", "regulatory action", "supply chain disruption",
    "geopolitical", "bankruptcy", "default", "covenant violation",
]


def _extract_risk_flags(text: str) -> list[str]:
    text_lower = text.lower()
    return [kw for kw in RISK_KEYWORDS if kw in text_lower]


async def get_cik_for_ticker(
    ticker: str,
    user_agent: str,
) -> Optional[str]:
    """Look up SEC CIK number for a given ticker symbol."""
    url = "https://www.sec.gov/files/company_tickers.json"
    async with httpx.AsyncClient(timeout=15, headers={"User-Agent": user_agent}) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            ticker_upper = ticker.upper()
            for _, company in data.items():
                if company.get("ticker", "").upper() == ticker_upper:
                    return str(company["cik_str"]).zfill(10)
        except Exception as e:
            logger.warning(f"CIK lookup failed for {ticker}: {e}")
    return None


async def fetch_recent_filings(
    entity_id: str,
    ticker: Optional[str],
    user_agent: str,
    max_filings: int = 5,
    form_types: Optional[list[str]] = None,
) -> list[SECFiling]:
    """
    Fetch recent SEC filings for a company by ticker.
    Returns a list of SECFiling objects, flagging risk-relevant content.
    """
    if not ticker:
        return []

    form_types = form_types or ["10-K", "10-Q", "8-K"]
    filings: list[SECFiling] = []

    cik = await get_cik_for_ticker(ticker, user_agent)
    if not cik:
        logger.info(f"No CIK found for ticker {ticker}")
        return []

    url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik}.json"
    async with httpx.AsyncClient(timeout=20, headers={"User-Agent": user_agent}) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()

            recent = data.get("filings", {}).get("recent", {})
            form_list  = recent.get("form", [])
            date_list  = recent.get("filingDate", [])
            accn_list  = recent.get("accessionNumber", [])
            desc_list  = recent.get("primaryDocument", [])

            count = 0
            for i, form in enumerate(form_list):
                if form not in form_types:
                    continue
                if count >= max_filings:
                    break
                try:
                    filed_at = datetime.strptime(date_list[i], "%Y-%m-%d")
                except Exception:
                    filed_at = datetime.utcnow()

                desc = desc_list[i] if i < len(desc_list) else ""
                flags = _extract_risk_flags(desc)

                accession_number = accn_list[i] if i < len(accn_list) else ""
                filing_url = None
                if accession_number:
                    filing_url = f"https://www.sec.gov/edgar/search/#/q={accession_number}"

                filings.append(SECFiling(
                    entity_id=entity_id,
                    form_type=form,
                    filed_at=filed_at,
                    accession_number=accession_number,
                    description=f"{form} filing — {date_list[i] if i < len(date_list) else 'unknown'}",
                    risk_flags=flags,
                    url=filing_url,
                ))
                count += 1

        except Exception as e:
            logger.warning(f"EDGAR fetch failed for CIK {cik} ({ticker}): {e}")

    logger.info(f"Fetched {len(filings)} SEC filings for {ticker}")
    return filings
