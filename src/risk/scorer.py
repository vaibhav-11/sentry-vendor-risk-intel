"""
Multi-dimensional risk scoring engine.
Computes Financial, Operational, Compliance, and Geopolitical
dimension scores, then combines them into a weighted composite.
All logic is pure Python/NumPy — no GPU required.
"""

import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import (
    Entity, FootprintData, RiskScore, DimensionScore, RiskLevel, DriverEvidence,
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

# ── Compliance scoring constants (E2) ─────────────────────────────────────────
# Deterministic compliance scoring driven by SEC EDGAR evidence rather than
# magic numbers. Higher = more compliance risk.
COMPLIANCE_BASELINE      = 45.0   # neutral starting point before evidence adjusts it
RECENT_10K_CREDIT        = 18.0   # a recent annual report is a good-governance signal
RECENT_8K_PENALTY        = 8.0    # each recent 8-K (material event) nudges risk up
NO_FILINGS_DEFAULT       = 45.0   # no resolvable filings (e.g. non-US-listed) — not a failure,
                                  # just absence of SEC filings; must not outscore US entities
                                  # that carry active filing flags purely due to absent filings
EIGHT_K_LOOKBACK_DAYS    = 90

# ── OFAC / Sanctions watchlist (illustrative — check real OFAC in production) ──
SANCTIONED_COUNTRIES = {"RU", "IR", "KP", "BY", "SY", "CU", "SD", "VE"}

# ── Country name / alias → ISO-2 normalization ────────────────────────────────
# The watchlist LLM and some seed data emit full country names; COUNTRY_RISK and
# every downstream geo lookup is keyed by ISO-2. Normalize at entity construction
# so hq_country is *always* ISO-2 by the time it reaches the scorer. Without this,
# every COUNTRY_RISK lookup misses and geo flatlines at DEFAULT_COUNTRY_RISK.
_COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "taiwan": "TW", "republic of china": "TW",
    "china": "CN", "people's republic of china": "CN", "prc": "CN", "mainland china": "CN",
    "south korea": "KR", "korea, republic of": "KR", "republic of korea": "KR", "korea": "KR",
    "north korea": "KP",
    "japan": "JP",
    "germany": "DE",
    "netherlands": "NL", "the netherlands": "NL", "holland": "NL",
    "ireland": "IE",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "britain": "GB", "england": "GB",
    "united states": "US", "united states of america": "US", "usa": "US", "u.s.": "US",
    "u.s.a.": "US", "america": "US",
    "singapore": "SG",
    "india": "IN",
    "mexico": "MX",
    "russia": "RU", "russian federation": "RU",
    "iran": "IR",
    "belarus": "BY",
    "myanmar": "MM", "burma": "MM",
    "brazil": "BR",
    "australia": "AU",
    "canada": "CA",
    "france": "FR",
    "sweden": "SE",
    "switzerland": "CH",
    "israel": "IL",
    "saudi arabia": "SA",
    "united arab emirates": "AE", "uae": "AE",
    "vietnam": "VN",
}

# Set of all ISO-2 codes we recognise (risk index keys ∪ sanctioned ∪ alias targets).
_KNOWN_ISO2: set[str] = (
    set(COUNTRY_RISK) | SANCTIONED_COUNTRIES | set(_COUNTRY_NAME_TO_ISO2.values())
)


def normalize_country(value: str) -> str:
    """
    Normalize a country value to an ISO-2 code.

    - Full names and common aliases ("China", "United Kingdom", "UK") map to ISO-2.
    - Values that are already valid 2-letter ISO codes pass through (upper-cased).
    - Empty / unknown values pass through unchanged so callers can decide how to
      handle them (the geo scorer treats "" as an explicit data gap).
    """
    if not value:
        return value
    raw = value.strip()
    if not raw:
        return value

    # Alias lookup first so 2-letter aliases like "UK" → "GB" aren't short-circuited
    # by the ISO-2 passthrough below.
    mapped = _COUNTRY_NAME_TO_ISO2.get(raw.lower())
    if mapped:
        return mapped

    # Already an ISO-2 code (e.g. "tw", "US") — upper-case and pass through.
    if len(raw) == 2 and raw.isalpha():
        return raw.upper()

    # Unknown longer string — return upper-cased original so it's at least visible;
    # the scorer will fall back to DEFAULT_COUNTRY_RISK for anything off the index.
    return raw.upper()


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
# Refactor inside src/risk/scorer.py

def _score_financial(fp: FootprintData) -> DimensionScore:
    fin = fp.financials
    drivers: list[str] = []
    gaps: list[str] = []
    points: list[float] = []

    if fin is None:
        gaps.append("No financial telemetry discovered in current pipeline iteration")
        return DimensionScore(score=50.0, confidence=0.2, key_drivers=["Missing historical balance sheet tracking"], data_gaps=gaps)

    # Altman Z-Score Driver Tracking
    if fin.altman_z_score is not None:
        z = fin.altman_z_score
        if z < 1.81:
            points.append(85.0)
            drivers.append(f"Altman Z-Score {z:.2f}: Distress Zone (High Insolvency Risk)")
        elif z < 2.99:
            points.append(50.0)
            drivers.append(f"Altman Z-Score {z:.2f}: Grey Zone (Unsettled Volatility)")
        else:
            points.append(15.0)
            drivers.append(f"Altman Z-Score {z:.2f}: Safe Zone (Strong Capital Buffer)")
    else:
        gaps.append("Altman Z-Score: Insufficient operational working capital data")

    # Revenue Growth Driver Tracking
    if fin.revenue_growth_yoy_pct is not None:
        g = fin.revenue_growth_yoy_pct
        if g < -15:
            points.append(80.0)
            drivers.append(f"Revenue structural decline: {g:.1f}% YoY contraction")
        elif g < 0:
            points.append(45.0)
            drivers.append(f"Revenue contraction warning: {g:.1f}% YoY drop")
        else:
            points.append(max(5.0, 30.0 - g))
            drivers.append(f"Revenue expansion verified: +{g:.1f}% YoY growth")
    else:
        gaps.append("Revenue Growth: Trailing metric not exposed")

    # Leverage (Debt to Equity)
    if fin.debt_to_equity is not None:
        de = fin.debt_to_equity
        if de > 200:
            points.append(85.0)
            drivers.append(f"Debt-to-Equity {de:.1f}%: Hyper-leveraged structural posture")
        elif de > 100:
            points.append(65.0)
            drivers.append(f"Debt-to-Equity {de:.1f}%: Elevated balance sheet debt load")
        else:
            points.append(15.0)
            drivers.append(f"Debt-to-Equity {de:.1f}%: Conservative leverage alignment")

    # Liquidity (Current Ratio)
    if fin.current_ratio is not None:
        cr = fin.current_ratio
        if cr < 1.0:
            points.append(80.0)
            drivers.append(f"Current Ratio {cr:.2f}: Illiquidity threat (Current Liabilities exceed Assets)")
        else:
            points.append(15.0)
            drivers.append(f"Current Ratio {cr:.2f}: Fluid working asset liquidity position")

    score = round(sum(points) / len(points), 1) if points else 50.0
    confidence = min(1.0, 0.4 + 0.15 * len(points))

    return DimensionScore(
        score=score,
        confidence=round(confidence, 2),
        key_drivers=drivers,  # Send the full array of actual metrics down the pipeline
        data_gaps=gaps,
        evidence=list(fp.fin_evidence),  # Issue 5: per-metric URL-bearing provenance
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

        # G1: a clean bill of health is itself evidence — state it explicitly
        # rather than emitting an empty driver list (e.g. Samsung Electronics).
        if not drivers:
            drivers.append("No operational flags — vendor meets all monitored thresholds")
    else:
        gaps.append("No internal vendor record — using entity-level proxies")
        # Fall back to entity-level signals
        if entity.importance_score >= 9:
            points.append(60.0)
            drivers.append("High-importance entity with no internal spend data")
        else:
            points.append(40.0)
        # G1: make the gap explicit rather than silent.
        drivers.append("No internal vendor record — operational score set to default")

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
    """
    Deterministic compliance scoring driven by SEC EDGAR evidence (E2).

    Every branch appends a DriverEvidence so the score is fully attributable:
      - recent 10-K present  → baseline reduced (good governance signal)
      - 8-K filings in last 90 days → score nudged up per filing (material events)
      - no filings found     → NO_FILINGS_DEFAULT, explicit no-US-filings evidence
    Sanctions and internal-certification signals layer on top, each cited.
    """
    drivers: list[str] = []
    gaps: list[str] = []
    evidence: list[DriverEvidence] = list(fp.sec_evidence)  # already URL-bearing, inline

    sec_ev = fp.sec_evidence
    has_real_filings = any(ev.value not in ("non-us-listed", None) for ev in sec_ev)
    has_10k = any((ev.value or "").upper() == "10-K" for ev in sec_ev)
    recent_8k = sum(
        1 for ev in sec_ev
        if (ev.value or "").upper() == "8-K"
        and (datetime.utcnow() - ev.retrieved_at).days <= EIGHT_K_LOOKBACK_DAYS
    )
    all_flags = [flag for f in fp.sec_filings for flag in f.risk_flags]

    # ── SEC-driven base score ──────────────────────────────────────────────
    if not has_real_filings:
        # Absence of US filings is NOT a compliance failure — it is the neutral
        # default. We deliberately cap the entire no-filings branch at this value
        # (Issue 4): a non-US-listed entity must never outscore a US entity that
        # carries active SEC filing flags purely because it doesn't file here.
        score = NO_FILINGS_DEFAULT
        drivers.append("No SEC filings on record (non-US-listed or unmatched entity)")
        gaps.append("No US securities filings — compliance inferred from other signals")
    else:
        score = COMPLIANCE_BASELINE
        # The 10-K credit is a *clean* governance signal — only award it when the
        # filing carries no risk flags. A flagged 10-K is not a clean bill.
        if has_10k and not all_flags:
            score -= RECENT_10K_CREDIT
            drivers.append("Recent 10-K on file, no flags — annual disclosure clean")
        elif has_10k:
            drivers.append("Recent 10-K on file (carries risk flags — see below)")
        if recent_8k:
            score += RECENT_8K_PENALTY * recent_8k
            drivers.append(f"{recent_8k} 8-K material-event filing(s) in last 90 days")

    # ── SEC risk flags from filing content ─────────────────────────────────
    if "going concern" in all_flags or "material weakness" in all_flags:
        score += 30.0
        drivers.append(f"SEC filing flags: {', '.join(set(all_flags[:2]))}")
    elif all_flags:
        score += 12.0
        drivers.append(f"SEC filing flags: {all_flags[0]}")

    # ── Internal certification / GDPR signals ──────────────────────────────
    internal = fp.internal_record
    if internal:
        expected_certs = {"ISO27001", "SOC2"}
        missing = expected_certs - set(internal.compliance_certifications)
        if missing:
            score += 10.0 if len(missing) == 2 else 5.0
            drivers.append(f"Missing certifications: {', '.join(sorted(missing))}")
        if not internal.gdpr_dpa_signed:
            score += 6.0
            drivers.append("GDPR Data Processing Agreement not signed")
    else:
        gaps.append("No internal compliance record")

    # Issue 4: hard ceiling on the no-filings path so absence of SEC filings
    # (even with cert/GDPR gaps) cannot exceed a US filer carrying active flags.
    # Applied BEFORE sanctions so a sanctioned non-filer still scores critically.
    if not has_real_filings:
        score = min(score, NO_FILINGS_DEFAULT)

    # ── Sanctions check (country-level) — never capped ─────────────────────
    if entity.hq_country in SANCTIONED_COUNTRIES:
        score += 40.0
        drivers.append(f"HQ country {entity.hq_country} on sanctions list")

    score      = round(max(0.0, min(100.0, score)), 1)
    confidence = 0.85 if has_real_filings else 0.55

    return DimensionScore(
        score=score,
        confidence=confidence,
        key_drivers=drivers[:4],
        data_gaps=gaps,
        evidence=evidence,
    )


# ── Geopolitical Dimension ────────────────────────────────────────────────────

def _score_geopolitical(entity: Entity, fp: FootprintData) -> DimensionScore:
    """
    Deterministic geopolitical scoring (F1). Always emits at minimum:
      (1) a country-risk baseline DriverEvidence explaining the index value,
      (2) any GDELT event entries fetched for the entity's country.
    The portfolio-HHI contribution is appended later by attach_geo_hhi_evidence()
    once the graph metrics are computed (HHI needs the full portfolio).
    """
    drivers: list[str] = []
    gaps: list[str] = []
    evidence: list[DriverEvidence] = []

    country = entity.hq_country or (
        fp.internal_record.geographic_risk_country if fp.internal_record else ""
    )
    # Defensive: normalize in case a fallback source still carries a full name.
    country = normalize_country(country)
    country_risk = COUNTRY_RISK.get(country, DEFAULT_COUNTRY_RISK)

    # (1) Country-risk baseline — ALWAYS emitted, even for low-risk jurisdictions.
    band = (
        "elevated geopolitical exposure" if country_risk > 60
        else "moderate geopolitical exposure" if country_risk > 35
        else "low geopolitical exposure"
    )
    baseline_label = f"{country or 'Unknown'} country risk index: {country_risk:.0f}/100 — {band}"
    drivers.append(baseline_label)
    evidence.append(DriverEvidence(
        label=baseline_label,
        source_url="https://www.baselgovernance.org/basel-aml-index",
        retrieved_at=datetime.utcnow(),
        value=f"{country_risk:.0f}",
    ))

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

    # (2) GDELT country events captured at fetch time.
    evidence.extend(fp.geo_events[:3])

    confidence = 0.75 if country else 0.4

    return DimensionScore(
        score=score,
        confidence=confidence,
        key_drivers=drivers[:4],
        data_gaps=gaps,
        evidence=evidence,
    )


def attach_geo_hhi_evidence(
    risk_scores: dict[str, RiskScore],
    entities: list[Entity],
    hhi: float,
    country_distribution: dict[str, float],
) -> None:
    """
    F1: append the portfolio geo-concentration HHI contribution to each node's
    geopolitical evidence. Called after graph metrics are computed (HHI needs the
    full portfolio). Each node's evidence cross-references its own country share,
    making the stat-card HHI figure explainable in the per-node inspector.
    """
    entity_country = {e.id: normalize_country(e.hq_country) for e in entities}
    for eid, score in risk_scores.items():
        country = entity_country.get(eid, "")
        share = country_distribution.get(country)
        if share is not None:
            label = (
                f"Portfolio geo-concentration HHI: {hhi:.0f} — "
                f"{country} accounts for {share:.1f}% of supply spend"
            )
        else:
            label = f"Portfolio geo-concentration HHI: {hhi:.0f}"
        score.geopolitical.key_drivers.append(label)
        score.geopolitical.evidence.append(DriverEvidence(
            label=label,
            source_url="#geo-distribution-container",
            retrieved_at=datetime.utcnow(),
            value=f"{share:.1f}%" if share is not None else None,
        ))


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
