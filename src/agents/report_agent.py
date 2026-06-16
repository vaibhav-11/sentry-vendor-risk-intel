"""
Report Agent — LangGraph node.
Generates the executive HTML risk report using the LLM.
"""

import logging
import time
from datetime import datetime
from typing import Any

from src.models import PipelineState, RiskLevel
from src.llm.interface import get_llm_client
from src.llm.metrics import add_latency
from config.prompts import (
    SYSTEM_RISK_ANALYST, EXECUTIVE_REPORT_PROMPT, CASCADE_SUMMARY_PROMPT
)

logger = logging.getLogger(__name__)


def _summarise_top_entities(risk_scores: dict, entities: list, n: int = 5) -> str:
    sorted_scores = sorted(
        risk_scores.values(), key=lambda s: s.composite_score, reverse=True
    )[:n]
    lines = []
    for i, score in enumerate(sorted_scores, 1):
        entity = next((e for e in entities if e.id == score.entity_id), None)
        name   = entity.name if entity else score.entity_id
        level  = score.risk_level.value.upper()
        lines.append(
            f"{i}. {name} — Score: {score.composite_score:.0f}/100 [{level}] | "
            f"Fin: {score.financial.score:.0f} | "
            f"Ops: {score.operational.score:.0f} | "
            f"Compliance: {score.compliance.score:.0f} | "
            f"Geo: {score.geopolitical.score:.0f}"
        )
    return "\n".join(lines)


def _summarise_vulnerabilities(risk_scores: dict, entities: list) -> str:
    all_drivers: list[str] = []
    for score in risk_scores.values():
        all_drivers.extend(score.financial.key_drivers)
        all_drivers.extend(score.operational.key_drivers)
        all_drivers.extend(score.compliance.key_drivers)
        all_drivers.extend(score.geopolitical.key_drivers)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for d in all_drivers:
        if d not in seen:
            seen.add(d)
            unique.append(d)
    return "\n".join(f"• {d}" for d in unique[:8])


def _summarise_cascade(graph_metrics, entities: list) -> str:
    if graph_metrics is None:
        return "Graph analysis not available."
    spof_names = []
    for eid in graph_metrics.single_points_of_failure[:3]:
        entity = next((e for e in entities if e.id == eid), None)
        spof_names.append(entity.name if entity else eid)

    top_cascade_names = []
    for eid in graph_metrics.top_cascade_risks[:3]:
        nm = graph_metrics.node_metrics.get(eid)
        entity = next((e for e in entities if e.id == eid), None)
        name   = entity.name if entity else eid
        br     = nm.blast_radius_pct if nm else 0
        top_cascade_names.append(f"{name} (blast radius: {br:.0f}%)")

    lines = [
        f"Graph density: {graph_metrics.density:.3f}",
        f"Single points of failure: {', '.join(spof_names) or 'None identified'}",
        f"Top cascade risk nodes: {'; '.join(top_cascade_names) or 'None'}",
        f"Max supply chain depth: {graph_metrics.max_depth}",
    ]
    return "\n".join(lines)


async def report_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: generate executive HTML report.
    """
    ps = PipelineState(**state)
    ps.stage = "report"
    _t0 = time.perf_counter()
    llm = get_llm_client(ps.llm_backend)

    critical_count = sum(
        1 for s in ps.risk_scores.values() if s.risk_level == RiskLevel.CRITICAL
    )
    high_count = sum(
        1 for s in ps.risk_scores.values() if s.risk_level == RiskLevel.HIGH
    )

    prompt = EXECUTIVE_REPORT_PROMPT.format(
        target_company=ps.target_company,
        analysis_date=datetime.utcnow().strftime("%d %B %Y"),
        total_entities=len(ps.entities),
        critical_count=critical_count,
        high_count=high_count,
        top_entities_summary=_summarise_top_entities(ps.risk_scores, ps.entities),
        vulnerabilities_summary=_summarise_vulnerabilities(ps.risk_scores, ps.entities),
        cascade_summary=_summarise_cascade(ps.graph_metrics, ps.entities),
    )

    try:
        report_html = await llm.generate(prompt, system=SYSTEM_RISK_ANALYST,
                                          max_tokens=1500, label="report_agent")
        ps.report_html = report_html
    except Exception as e:
        ps.add_error(f"Report generation failed: {e}")
        ps.report_html = f"<p>Report generation failed: {e}</p>"
        logger.error(f"[Report] Error: {e}")

    logger.info(f"[Report] Executive report generated ({len(ps.report_html)} chars)")

    add_latency("report_agent", time.perf_counter() - _t0)
    return ps.model_dump()
