"""
Fetches SEC EDGAR filings for US-listed companies.
Uses the public EDGAR full-text search API — no API key required.
https://efts.sec.gov/LATEST/search-index?q=...
"""

import logging
from datetime import datetime
from typing import Optional

import httpx

from src.models import SECFiling, DriverEvidence

logger = logging.getLogger(__name__)

# Entities with no US listing — EDGAR full-text search will return nothing for
# these, so we emit an explicit no-filings DriverEvidence rather than a blank.
# (E2) The compliance dimension must never be left unattributed for them.
NON_US_LISTED = {
    "tsmc", "samsung electronics", "foxconn", "shin-etsu chemical", "pegatron",
    "asml", "softbank group", "arm holdings",
}


def build_filing_evidence(filing: SECFiling) -> DriverEvidence:
    """
    Build a DriverEvidence for a single SEC filing, captured inline at fetch
    time. The URL is the direct EDGAR document link resolved on the SECFiling.
    """
    filed = filing.filed_at.strftime("%Y-%m-%d")
    if filing.risk_flags:
        label = f"{filing.form_type} filed {filed} — risk flags: {', '.join(filing.risk_flags)}"
    else:
        label = f"{filing.form_type} filed {filed}"
    return DriverEvidence(
        label=label,
        source_url=filing.url or "https://www.sec.gov/cgi-bin/browse-edgar",
        retrieved_at=datetime.utcnow(),
        value=filing.form_type,
    )


def edgar_search_url(ticker_or_name: str) -> str:
    """The EDGAR full-text search URL we attempt for an entity (used as the
    source link on the explicit no-filings evidence for foreign entities)."""
    q = ticker_or_name.replace(" ", "+")
    return f"https://efts.sec.gov/LATEST/search-index?q={q}"


def no_us_filing_evidence(entity_name: str, ticker: Optional[str]) -> DriverEvidence:
    """The single explicit DriverEvidence emitted for non-US-listed entities (E2)."""
    return DriverEvidence(
        label="No SEC filings — non-US-listed entity",
        source_url=edgar_search_url(ticker or entity_name),
        retrieved_at=datetime.utcnow(),
        value="non-us-listed",
    )

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
    entity_name: str = "",
) -> tuple[list[SECFiling], list[DriverEvidence]]:
    """
    Fetch recent SEC filings for a company by ticker (E2).

    Returns (filings, evidence) where `evidence` is a list of DriverEvidence
    captured inline at fetch time — one per filing for US-listed entities, or a
    single explicit "No SEC filings — non-US-listed entity" entry for foreign
    ones / entities with no resolvable CIK. Compliance evidence is never empty.
    """
    form_types = form_types or ["10-K", "10-Q", "8-K"]
    filings: list[SECFiling] = []
    evidence: list[DriverEvidence] = []

    # Foreign / non-US-listed entities never have EDGAR results — emit the
    # explicit no-filings evidence and skip the network call entirely.
    if entity_name and entity_name.lower() in NON_US_LISTED:
        return [], [no_us_filing_evidence(entity_name, ticker)]

    if not ticker:
        return [], [no_us_filing_evidence(entity_name or entity_id, ticker)]

    cik = await get_cik_for_ticker(ticker, user_agent)
    if not cik:
        logger.info(f"No CIK found for ticker {ticker}")
        return [], [no_us_filing_evidence(entity_name or ticker, ticker)]

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

                filing = SECFiling(
                    entity_id=entity_id,
                    form_type=form,
                    filed_at=filed_at,
                    accession_number=accession_number,
                    description=f"{form} filing — {date_list[i] if i < len(date_list) else 'unknown'}",
                    risk_flags=flags,
                    url=filing_url,
                )
                filings.append(filing)
                evidence.append(build_filing_evidence(filing))
                count += 1

        except Exception as e:
            logger.warning(f"EDGAR fetch failed for CIK {cik} ({ticker}): {e}")

    # A US-listed entity that returned no filings still gets an explicit entry.
    if not evidence:
        evidence.append(no_us_filing_evidence(entity_name or ticker, ticker))

    logger.info(f"Fetched {len(filings)} SEC filings for {ticker}")
    return filings, evidence
