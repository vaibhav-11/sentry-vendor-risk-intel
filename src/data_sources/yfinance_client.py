"""
Fetches financial metrics from Yahoo Finance via yfinance.
Works for any publicly listed company. Private companies return
partial data with degraded data_quality score.
"""

import logging
from datetime import datetime
from typing import Optional

import yfinance as yf
import numpy as np

from src.models import FinancialMetrics, DriverEvidence

logger = logging.getLogger(__name__)


def build_financial_evidence(
    ticker: str,
    fin: FinancialMetrics,
) -> DriverEvidence:
    """
    Build a DriverEvidence for financial metrics sourced from Yahoo Finance.
    The URL points at the public Yahoo quote page for the ticker so the cited
    figures are followable to their provider.
    """
    snippet_bits = []
    if fin.market_cap_usd:
        snippet_bits.append(f"Market cap ${fin.market_cap_usd/1e9:.1f}B")
    if fin.altman_z_score is not None:
        snippet_bits.append(f"Altman Z {fin.altman_z_score:.2f}")
    if fin.debt_to_equity is not None:
        snippet_bits.append(f"D/E {fin.debt_to_equity:.1f}")
    snippet = "; ".join(snippet_bits) or "Financial metrics retrieved from provider"

    return DriverEvidence(
        label=f"Yahoo Finance ({ticker.upper()}): {snippet}",
        source_url=f"https://finance.yahoo.com/quote/{ticker.upper()}",
        retrieved_at=datetime.utcnow(),
        value=snippet,
    )


def build_financial_evidence_list(
    ticker: str,
    fin: FinancialMetrics,
) -> list[DriverEvidence]:
    """
    Emit one DriverEvidence per computed financial metric (Issue 5) — Altman
    Z-score, revenue growth YoY, D/E ratio, current ratio — each linking back to
    the Yahoo quote page. Labels mirror the driver strings the financial scorer
    produces so the inspector evidence lines up 1:1 with the rendered drivers.
    """
    if fin is None or not ticker:
        return []
    url = f"https://finance.yahoo.com/quote/{ticker.upper()}"
    now = datetime.utcnow()
    out: list[DriverEvidence] = []

    if fin.altman_z_score is not None:
        z = fin.altman_z_score
        zone = ("Distress Zone (High Insolvency Risk)" if z < 1.81
                else "Grey Zone (Unsettled Volatility)" if z < 2.99
                else "Safe Zone (Strong Capital Buffer)")
        out.append(DriverEvidence(
            label=f"Altman Z-Score {z:.2f}: {zone}",
            source_url=url, retrieved_at=now, value=f"{z:.2f}",
        ))
    if fin.revenue_growth_yoy_pct is not None:
        g = fin.revenue_growth_yoy_pct
        if g < -15:
            lbl = f"Revenue structural decline: {g:.1f}% YoY contraction"
        elif g < 0:
            lbl = f"Revenue contraction warning: {g:.1f}% YoY drop"
        else:
            lbl = f"Revenue expansion verified: +{g:.1f}% YoY growth"
        out.append(DriverEvidence(
            label=lbl, source_url=url, retrieved_at=now, value=f"{g:.1f}%",
        ))
    if fin.debt_to_equity is not None:
        de = fin.debt_to_equity
        posture = ("Hyper-leveraged structural posture" if de > 200
                   else "Elevated balance sheet debt load" if de > 100
                   else "Conservative leverage alignment")
        out.append(DriverEvidence(
            label=f"Debt-to-Equity {de:.1f}%: {posture}",
            source_url=url, retrieved_at=now, value=f"{de:.1f}%",
        ))
    if fin.current_ratio is not None:
        cr = fin.current_ratio
        liq = ("Illiquidity threat (Current Liabilities exceed Assets)" if cr < 1.0
               else "Fluid working asset liquidity position")
        out.append(DriverEvidence(
            label=f"Current Ratio {cr:.2f}: {liq}",
            source_url=url, retrieved_at=now, value=f"{cr:.2f}",
        ))
    return out


def _safe_float(val, default=None) -> Optional[float]:
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def compute_altman_z_score(
    working_capital: float,
    total_assets: float,
    retained_earnings: float,
    ebit: float,
    market_cap: float,
    total_liabilities: float,
    revenue: float,
) -> Optional[float]:
    """
    Altman Z-Score (public company variant).
    Z > 2.99 = Safe zone
    1.81 < Z < 2.99 = Grey zone
    Z < 1.81 = Distress zone
    """
    try:
        if total_assets <= 0 or total_liabilities <= 0:
            return None
        x1 = working_capital / total_assets
        x2 = retained_earnings / total_assets
        x3 = ebit / total_assets
        x4 = market_cap / total_liabilities
        x5 = revenue / total_assets
        return round(1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + x5, 2)
    except ZeroDivisionError:
        return None


async def fetch_financial_metrics(
    entity_id: str,
    ticker: Optional[str],
) -> FinancialMetrics:
    """
    Fetch financial data for an entity by ticker symbol.
    Returns a FinancialMetrics object with available data fields.
    If no ticker, returns an empty metrics object with low data_quality.
    """
    base = FinancialMetrics(entity_id=entity_id, fetch_date=datetime.utcnow())

    if not ticker:
        base.data_quality = 0.1
        return base

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        base.market_cap_usd    = _safe_float(info.get("marketCap"))
        base.stock_price       = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        base.revenue_ttm_usd   = _safe_float(info.get("totalRevenue"))
        base.gross_margin_pct  = _safe_float(info.get("grossMargins"))
        base.net_income_ttm_usd = _safe_float(info.get("netIncomeToCommon"))
        base.total_debt_usd    = _safe_float(info.get("totalDebt"))
        base.cash_usd          = _safe_float(info.get("totalCash"))
        base.debt_to_equity    = _safe_float(info.get("debtToEquity"))
        base.current_ratio     = _safe_float(info.get("currentRatio"))

        # Revenue growth YoY
        rev_growth = _safe_float(info.get("revenueGrowth"))
        if rev_growth is not None:
            base.revenue_growth_yoy_pct = rev_growth * 100

        # 30-day price change
        hist = stock.history(period="1mo")
        if not hist.empty and len(hist) >= 2:
            price_start = hist["Close"].iloc[0]
            price_end   = hist["Close"].iloc[-1]
            if price_start > 0:
                base.price_change_30d_pct = round(
                    (price_end - price_start) / price_start * 100, 2
                )

        # Altman Z-Score (approximate from available info fields)
        bs = stock.balance_sheet
        if not bs.empty:
            try:
                total_assets = _safe_float(bs.loc["Total Assets"].iloc[0]) or 0
                total_liab   = _safe_float(bs.loc["Total Liabilities Net Minority Interest"].iloc[0]) or 0
                curr_assets  = _safe_float(bs.loc["Current Assets"].iloc[0]) or 0
                curr_liab    = _safe_float(bs.loc["Current Liabilities"].iloc[0]) or 0
                ret_earnings = _safe_float(bs.loc["Retained Earnings"].iloc[0]) or 0
                working_cap  = curr_assets - curr_liab

                income = stock.income_stmt
                ebit = 0.0
                if not income.empty and "EBIT" in income.index:
                    ebit = _safe_float(income.loc["EBIT"].iloc[0]) or 0.0

                z = compute_altman_z_score(
                    working_cap, total_assets, ret_earnings, ebit,
                    base.market_cap_usd or 0,
                    total_liab,
                    base.revenue_ttm_usd or 0,
                )
                base.altman_z_score = z
            except (KeyError, IndexError):
                pass

        # Data quality based on how many fields we populated
        filled = sum(1 for v in [
            base.market_cap_usd, base.revenue_ttm_usd, base.debt_to_equity,
            base.current_ratio, base.altman_z_score
        ] if v is not None)
        base.data_quality = round(filled / 5, 2)

        logger.info(f"Fetched financials for {ticker}: quality={base.data_quality}")

    except Exception as e:
        logger.warning(f"yfinance error for {ticker}: {e}")
        base.data_quality = 0.2

    return base
