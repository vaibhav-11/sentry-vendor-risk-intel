"""
Risk Agent — LangGraph node.
Scores every entity and runs graph-level cascade analysis.
Also calls the LLM to generate per-entity risk narratives in batch.
"""

import asyncio
import logging
from typing import Any

from src.models import PipelineState, RiskAlert, AlertSeverity
from src.risk.scorer import score_all_entities, attach_geo_hhi_evidence
from src.graph.supply_chain_graph import build_graph, attach_risk_scores, entities_to_relationships
from src.graph.cascading_risk import compute_graph_metrics
from src.risk.alternatives import attach_alternatives
from src.llm.interface import get_llm_client
from config.prompts import (
    SYSTEM_RISK_ANALYST, ENTITY_NARRATIVE_PROMPT, ALERT_PROMPT
)
from config.settings import settings
import yaml
import uuid

logger = logging.getLogger(__name__)


def _load_thresholds() -> dict:
    with open(settings.risk_weights_path) as f:
        return yaml.safe_load(f).get("thresholds", {
            "critical": 80, "high": 65, "medium": 45
        })


def _build_alert(entity_id: str, entity_name: str, score_obj, alert_raw: str) -> RiskAlert:
    """Parse LLM alert JSON and return a RiskAlert."""
    import json, re
    thresholds = _load_thresholds()
    score = score_obj.composite_score

    # Determine threshold name breached
    if score >= thresholds["critical"]:
        threshold_name = "critical"
    elif score >= thresholds["high"]:
        threshold_name = "high"
    else:
        threshold_name = "medium"

    try:
        clean = re.sub(r"```(?:json)?", "", alert_raw).strip()
        data  = json.loads(clean)
        return RiskAlert(
            alert_id=str(uuid.uuid4())[:8],
            entity_id=entity_id,
            entity_name=entity_name,
            alert_title=data.get("alert_title", f"{entity_name} risk alert"),
            severity=AlertSeverity(data.get("severity", threshold_name)),
            summary=data.get("summary", ""),
            recommended_action=data.get("recommended_action", ""),
            escalate_to=data.get("escalate_to", "CPO"),
            time_sensitivity=data.get("time_sensitivity", "1-week"),
            triggering_score=score,
            triggering_dimension="composite",
        )
    except Exception:
        return RiskAlert(
            alert_id=str(uuid.uuid4())[:8],
            entity_id=entity_id,
            entity_name=entity_name,
            alert_title=f"{entity_name} — {threshold_name.upper()} risk detected",
            severity=AlertSeverity(threshold_name),
            summary=f"{entity_name} has exceeded the {threshold_name} risk threshold "
                    f"with a composite score of {score:.0f}/100.",
            recommended_action="Review vendor risk profile and engage procurement team",
            escalate_to="CPO",
            time_sensitivity="1-week",
            triggering_score=score,
        )


async def risk_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: score all entities, run cascade analysis, generate
    narratives and alerts via LLM batch call.
    """
    ps = PipelineState(**state)
    ps.stage = "risk_scoring"
    llm = get_llm_client(ps.llm_backend)
    thresholds = _load_thresholds()

    # ── 1. Score all entities ──────────────────────────────────────────────
    logger.info(f"[Risk] Scoring {len(ps.entities)} entities")
    supply_chain_entities = [e for e in ps.entities if e.depth_level > 0]
    ps.risk_scores = score_all_entities(supply_chain_entities, ps.footprint_data)

    # ── 2. Build graph and run cascade analysis ────────────────────────────
    rels = ps.relationships or entities_to_relationships(ps.entities)
    G = build_graph(ps.entities, rels)
    G = attach_risk_scores(G, ps.risk_scores)
    ps.graph_metrics = compute_graph_metrics(G, ps.risk_scores, footprint_data=ps.footprint_data)

    # ── 2b. F1: append portfolio-HHI evidence to each node's geo dimension ──
    # HHI needs the full portfolio, so it's only available post-graph-metrics.
    attach_geo_hhi_evidence(
        ps.risk_scores,
        supply_chain_entities,
        ps.graph_metrics.geo_concentration_hhi,
        ps.graph_metrics.country_spend_distribution,
    )

    # ── 2c. G2: rank pre-vetted alternatives into each node's backups via LLM ──
    await attach_alternatives(ps, llm)

    # ── 3. Batch LLM: generate risk narratives for all scored entities ─────
    logger.info(f"[Risk] Generating narratives for {len(ps.risk_scores)} entities via LLM")

    narrative_prompts = []
    narrative_entity_ids = []
    for entity_id, score in ps.risk_scores.items():
        entity = ps.entity_by_id(entity_id)
        fp     = ps.footprint_data.get(entity_id)
        if entity is None or fp is None:
            continue

        fin_summary  = ""
        if fp.financials:
            fin = fp.financials
            fin_summary = (
                f"Market Cap: ${fin.market_cap_usd/1e9:.1f}B | "
                f"Revenue growth: {fin.revenue_growth_yoy_pct:.1f}% | "
                f"D/E: {fin.debt_to_equity:.1f} | "
                f"Z-Score: {fin.altman_z_score}"
            ) if all(v is not None for v in [
                fin.market_cap_usd, fin.revenue_growth_yoy_pct,
                fin.debt_to_equity, fin.altman_z_score
            ]) else "Limited financial data available"

        news_summary = "; ".join(fp.risk_news_headlines[:3]) or "No significant news"
        internal_summary = ""
        if fp.internal_record:
            ir = fp.internal_record
            internal_summary = (
                f"Annual spend: ${ir.annual_spend_usd/1e6:.0f}M "
                f"({ir.spend_percentage:.1f}% of total) | "
                f"Single-source: {ir.single_source} | "
                f"Audit score: {ir.audit_score}"
            )

        prompt = ENTITY_NARRATIVE_PROMPT.format(
            entity_name=entity.name,
            entity_type=entity.entity_type.value,
            relationship_description=entity.relationship_to_parent or "Supply chain entity",
            industry=entity.industry,
            hq_country=entity.hq_country,
            financial_summary=fin_summary or "Not available",
            news_summary=news_summary,
            internal_summary=internal_summary or "No internal data",
            risk_score=score.composite_score,
            fin_score=score.financial.score,
            ops_score=score.operational.score,
            comp_score=score.compliance.score,
            geo_score=score.geopolitical.score,
            financial_drivers="\n".join(
                f"- {d}" for d in score.financial.key_drivers
            ) or "- No financial drivers recorded",
            operational_drivers="\n".join(
                f"- {d}" for d in score.operational.key_drivers
            ) or "- No operational drivers recorded",
            compliance_drivers="\n".join(
                f"- {d}" for d in score.compliance.key_drivers
            ) or "- No compliance drivers recorded",
            geopolitical_drivers="\n".join(
                f"- {d}" for d in score.geopolitical.key_drivers
            ) or "- No geopolitical drivers recorded",
        )
        narrative_prompts.append(prompt)
        narrative_entity_ids.append(entity_id)

    if narrative_prompts:
        narratives = await llm.generate_batch(
            narrative_prompts, system=SYSTEM_RISK_ANALYST, max_tokens=2048
        )
        for entity_id, narrative in zip(narrative_entity_ids, narratives):
            ps.risk_scores[entity_id].narrative = narrative

    # ── 4. Generate alerts for entities breaching thresholds ──────────────
    alert_prompts: list[str] = []
    alert_entity_ids: list[str] = []

    for entity_id, score in ps.risk_scores.items():
        if score.composite_score >= thresholds["medium"]:
            entity = ps.entity_by_id(entity_id)
            fp     = ps.footprint_data.get(entity_id)
            drivers = (
                score.financial.key_drivers[:2] +
                score.operational.key_drivers[:1]
            )
            triggering_signal = (
                fp.risk_news_headlines[0] if fp and fp.risk_news_headlines
                else "Threshold breach on composite score"
            )
            threshold_name = (
                "critical" if score.composite_score >= thresholds["critical"]
                else "high" if score.composite_score >= thresholds["high"]
                else "medium"
            )
            prompt = ALERT_PROMPT.format(
                entity_name=entity.name if entity else entity_id,
                risk_score=score.composite_score,
                threshold_name=threshold_name,
                risk_drivers="; ".join(drivers) or "composite risk score",
                triggering_signal=triggering_signal,
            )
            alert_prompts.append(prompt)
            alert_entity_ids.append(entity_id)

    if alert_prompts:
        alert_responses = await llm.generate_batch(
            alert_prompts, system=SYSTEM_RISK_ANALYST, temperature=0.1
        )
        for entity_id, alert_raw in zip(alert_entity_ids, alert_responses):
            score = ps.risk_scores[entity_id]
            entity = ps.entity_by_id(entity_id)
            alert = _build_alert(entity_id, entity.name if entity else entity_id,
                                  score, alert_raw)
            ps.alerts.append(alert)

    # Sort alerts: critical → high → medium
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    ps.alerts.sort(key=lambda a: severity_order.get(a.severity.value, 9))

    logger.info(
        f"[Risk] Complete: {len(ps.risk_scores)} scored, {len(ps.alerts)} alerts generated"
    )
    return ps.model_dump()
