"""
Multi-dimensional risk scoring engine.
Computes Financial, Operational, Compliance, and Geopolitical
dimension scores, then combines them into a weighted composite.
All logic is pure Python/NumPy — no GPU required.
"""

import logging
import yaml
from pathlib import Path
from typing import Optional

from src.models import (
    Entity, FootprintData, RiskScore, DimensionScore, RiskLevel
)
from config.settings import settings

logger = logging.getLogger(__name__)

# ── Country Risk Index (simplified Basel AML Index proxy) ─────────────────────
# Score 0-100; higher = more risky. Source: public country risk indices.
COUNTRY_RISK: dict[str, float] = {
    "US": 10, "GB": 12, "DE": 11, "JP": 14, "NL": 10, "IE": 13,
    "TW": 45, "KR": 25, "CN": 65, "SG": 15, "IN": 40, "MX": 50,
    "RU": 85, "IR": 92, "KP": 98, "BY": 80, "MM": 78,
    "BR": 45, "AU": 11, "CA": 10, "FR": 13, "SE": 8,
    "CH": 9,  "IL": 38, "SA": 42, "AE": 30, "VN": 48,
}
DEFAULT_COUNTRY_RISK = 40.0  # Unknown country

# ── OFAC / Sanctions watchlist (illustrative — check real OFAC in production) ──
SANCTIONED_COUNTRIES = {"RU", "IR", "KP", "BY", "SY", "CU", "SD", "VE"}


def _load_weights(path: Optional[Path] = None) -> dict:
    fpath = path or settings.risk_weights_path
    with open(fpath) as f:
        return yaml.safe_load(f)


# ── Financial Dimension ───────────────────────────────────────────────────────

def _score_financial(fp: FootprintData) -> DimensionScore:
    fin = fp.financials
    score = 50.0    # default neutral score for missing data
    drivers: list[str] = []
    gaps: list[str] = []
    points: list[float] = []

    if fin is None:
        gaps.append("No financial data available (private company or fetch error)")
        return DimensionScore(score=50.0, confidence=0.2, key_drivers=drivers, data_gaps=gaps)

    # Altman Z-Score (0-100 mapped from Z-score)
    if fin.altman_z_score is not None:
        z = fin.altman_z_score
        if z < 1.81:
            z_score = 85.0
            drivers.append(f"Altman Z-Score {z:.2f} — distress zone (< 1.81)")
        elif z < 2.99:
            z_score = 50.0
            drivers.append(f"Altman Z-Score {z:.2f} — grey zone")
        else:
            z_score = 15.0
        points.append(z_score)
    else:
        gaps.append("Altman Z-Score: insufficient balance sheet data")

    # Revenue growth trend
    if fin.revenue_growth_yoy_pct is not None:
        g = fin.revenue_growth_yoy_pct
        if g < -15:
            rev_score = 80.0
            drivers.append(f"Revenue declining {g:.1f}% YoY")
        elif g < -5:
            rev_score = 60.0
            drivers.append(f"Revenue declining {g:.1f}% YoY")
        elif g < 0:
            rev_score = 45.0
        else:
            rev_score = max(5.0, 30.0 - g)   # growth reduces score
        points.append(rev_score)
    else:
        gaps.append("Revenue growth: no YoY data")

    # Debt-to-Equity
    if fin.debt_to_equity is not None:
        de = fin.debt_to_equity
        if de > 200:
            de_score = 85.0
            drivers.append(f"D/E ratio {de:.0f} — highly leveraged")
        elif de > 100:
            de_score = 65.0
            drivers.append(f"D/E ratio {de:.0f} — elevated leverage")
        elif de > 50:
            de_score = 40.0
        else:
            de_score = 15.0
        points.append(de_score)
    else:
        gaps.append("Debt-to-equity: not available")

    # Current ratio
    if fin.current_ratio is not None:
        cr = fin.current_ratio
        if cr < 1.0:
            cr_score = 80.0
            drivers.append(f"Current ratio {cr:.2f} — potential liquidity risk")
        elif cr < 1.5:
            cr_score = 50.0
        else:
            cr_score = 15.0
        points.append(cr_score)
    else:
        gaps.append("Current ratio: not available")

    if points:
        score = round(sum(points) / len(points), 1)
        confidence = min(1.0, 0.4 + 0.15 * len(points))
    else:
        score = 50.0
        confidence = 0.2

    return DimensionScore(
        score=score,
        confidence=round(confidence, 2),
        key_drivers=drivers[:3],
        data_gaps=gaps,
    )


# ── Operational Dimension ─────────────────────────────────────────────────────

def _score_operational(entity: Entity, fp: FootprintData) -> DimensionScore:
    internal = fp.internal_record
    drivers: list[str] = []
    gaps: list[str] = []
    points: list[float] = []

    if internal:
        # Spend concentration
        if internal.spend_percentage > 15:
            pts = 80.0
            drivers.append(f"Spend concentration: {internal.spend_percentage:.1f}% of total vendor spend")
        elif internal.spend_percentage > 8:
            pts = 55.0
            drivers.append(f"Spend concentration: {internal.spend_percentage:.1f}%")
        else:
            pts = 20.0
        points.append(pts)

        # Single source flag
        if internal.single_source:
            points.append(90.0)
            drivers.append("Single-source vendor — no approved alternate")
        elif not internal.alternate_vendor_available:
            points.append(65.0)
            drivers.append("No alternate vendor currently available")
        else:
            points.append(15.0)

        # Business continuity
        if not internal.business_continuity_plan:
            points.append(75.0)
            drivers.append("No business continuity plan on file")
        else:
            points.append(20.0)

        # Audit score
        if internal.audit_score is not None:
            if internal.audit_score < 60:
                points.append(80.0)
                drivers.append(f"Audit score {internal.audit_score:.0f}/100 — below threshold")
            elif internal.audit_score < 75:
                points.append(50.0)
            else:
                points.append(15.0)
        else:
            gaps.append("No audit score available")

        # Incidents
        if internal.incidents_last_12m > 2:
            points.append(70.0)
            drivers.append(f"{internal.incidents_last_12m} incidents in last 12 months")
        elif internal.incidents_last_12m > 0:
            points.append(40.0)
    else:
        gaps.append("No internal vendor record — using entity-level proxies")
        # Fall back to entity-level signals
        if entity.importance_score >= 9:
            points.append(60.0)
            drivers.append("High-importance entity with no internal spend data")
        else:
            points.append(40.0)

    score    = round(sum(points) / len(points), 1) if points else 50.0
    confidence = 0.9 if internal else 0.4

    return DimensionScore(
        score=score,
        confidence=confidence,
        key_drivers=drivers[:3],
        data_gaps=gaps,
    )


# ── Compliance Dimension ──────────────────────────────────────────────────────

def _score_compliance(entity: Entity, fp: FootprintData) -> DimensionScore:
    drivers: list[str] = []
    gaps: list[str] = []
    points: list[float] = []

    # Sanctions check (country-level)
    if entity.hq_country in SANCTIONED_COUNTRIES:
        points.append(95.0)
        drivers.append(f"HQ country {entity.hq_country} on sanctions list")

    # Internal compliance data
    internal = fp.internal_record
    if internal:
        # Certification gaps
        expected_certs = {"ISO27001", "SOC2"}
        actual_certs   = set(internal.compliance_certifications)
        missing        = expected_certs - actual_certs
        if missing:
            pts = 60.0 if len(missing) == 2 else 35.0
            points.append(pts)
            drivers.append(f"Missing certifications: {', '.join(missing)}")
        else:
            points.append(10.0)

        # GDPR DPA
        if not internal.gdpr_dpa_signed:
            points.append(55.0)
            drivers.append("GDPR Data Processing Agreement not signed")
        else:
            points.append(5.0)
    else:
        gaps.append("No internal compliance record")
        points.append(40.0)

    # SEC filing risk flags
    all_flags = [flag for f in fp.sec_filings for flag in f.risk_flags]
    if "going concern" in all_flags or "material weakness" in all_flags:
        points.append(80.0)
        drivers.append(f"SEC filing flags: {', '.join(set(all_flags[:2]))}")
    elif all_flags:
        points.append(45.0)
        drivers.append(f"SEC filing flags: {all_flags[0]}")

    score      = round(sum(points) / len(points), 1) if points else 30.0
    confidence = 0.85 if internal else 0.5

    return DimensionScore(
        score=score,
        confidence=confidence,
        key_drivers=drivers[:3],
        data_gaps=gaps,
    )


# ── Geopolitical Dimension ────────────────────────────────────────────────────

def _score_geopolitical(entity: Entity, fp: FootprintData) -> DimensionScore:
    drivers: list[str] = []
    gaps: list[str] = []

    country = entity.hq_country or (
        fp.internal_record.geographic_risk_country if fp.internal_record else ""
    )
    country_risk = COUNTRY_RISK.get(country, DEFAULT_COUNTRY_RISK)

    if country_risk > 60:
        drivers.append(f"High-risk jurisdiction: {country} (risk index {country_risk:.0f})")
    elif country_risk > 35:
        drivers.append(f"Elevated-risk jurisdiction: {country} (risk index {country_risk:.0f})")

    # Taiwan / China specific — key for semiconductor supply chains
    trade_war_bonus = 0.0
    if country in {"TW", "CN"}:
        trade_war_bonus = 15.0
        drivers.append("Cross-strait geopolitical exposure (TW/CN)")
    elif country in {"KR", "JP"} and entity.industry in {
        "Semiconductor Fabrication", "Memory & Display", "Semiconductor Equipment"
    }:
        trade_war_bonus = 8.0
        drivers.append("US export control exposure (advanced semiconductor)")

    score = min(100.0, round(country_risk + trade_war_bonus, 1))

    if not country:
        gaps.append("HQ country unknown — using default risk score")
        score = DEFAULT_COUNTRY_RISK

    confidence = 0.75 if country else 0.4

    return DimensionScore(
        score=score,
        confidence=confidence,
        key_drivers=drivers[:3],
        data_gaps=gaps,
    )


# ── Composite Scorer ──────────────────────────────────────────────────────────

def _risk_level(score: float) -> RiskLevel:
    if score >= 80: return RiskLevel.CRITICAL
    if score >= 65: return RiskLevel.HIGH
    if score >= 45: return RiskLevel.MEDIUM
    return RiskLevel.LOW


def score_entity(entity: Entity, fp: FootprintData) -> RiskScore:
    """
    Compute a full multi-dimensional RiskScore for a single entity.
    Loads weights from config/risk_weights.yaml.
    """
    weights_config = _load_weights()
    dims = weights_config["dimensions"]

    fin_dim  = _score_financial(fp)
    ops_dim  = _score_operational(entity, fp)
    comp_dim = _score_compliance(entity, fp)
    geo_dim  = _score_geopolitical(entity, fp)

    w_fin  = dims["financial"]["weight"]
    w_ops  = dims["operational"]["weight"]
    w_comp = dims["compliance"]["weight"]
    w_geo  = dims["geopolitical"]["weight"]

    composite = round(
        fin_dim.score  * w_fin  +
        ops_dim.score  * w_ops  +
        comp_dim.score * w_comp +
        geo_dim.score  * w_geo,
        1,
    )
    composite = max(0.0, min(100.0, composite))

    return RiskScore(
        entity_id=entity.id,
        entity_name=entity.name,
        composite_score=composite,
        risk_level=_risk_level(composite),
        financial=fin_dim,
        operational=ops_dim,
        compliance=comp_dim,
        geopolitical=geo_dim,
    )


def score_all_entities(
    entities: list[Entity],
    footprints: dict[str, FootprintData],
) -> dict[str, RiskScore]:
    """Score all entities and return a dict keyed by entity_id."""
    scores: dict[str, RiskScore] = {}
    for entity in entities:
        fp = footprints.get(entity.id)
        if fp is None:
            logger.warning(f"No footprint for {entity.name} — skipping scoring")
            continue
        scores[entity.id] = score_entity(entity, fp)
        logger.debug(f"{entity.name}: {scores[entity.id].composite_score:.1f}/100 "
                     f"({scores[entity.id].risk_level.value})")
    logger.info(f"Scored {len(scores)} entities")
    return scores
