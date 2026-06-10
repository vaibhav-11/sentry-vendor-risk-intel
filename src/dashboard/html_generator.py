"""
HTML Dashboard Generator.
Takes a completed PipelineState and renders a fully self-contained
single-file HTML dashboard — no server required, opens in any browser.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from src.models import PipelineState, RiskLevel

logger = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _score_to_colour(score: float) -> str:
    if score >= 80: return "#ef4444"
    if score >= 65: return "#f97316"
    if score >= 45: return "#eab308"
    return "#22c55e"


def _build_vis_nodes(ps: PipelineState) -> list[dict]:
    nodes = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        score     = score_obj.composite_score if score_obj else 0
        level     = score_obj.risk_level.value if score_obj else "unknown"
        colour    = _score_to_colour(score)
        size      = 20 + entity.importance_score * 3   

        var_val = 0.0
        backups = []
        if ps.graph_metrics and entity.id in ps.graph_metrics.node_metrics:
            nm_data = ps.graph_metrics.node_metrics[entity.id]
            var_val = nm_data.value_at_risk_usd
            backups = nm_data.alternative_suppliers

        tooltip_text = (
            f"Entity: {entity.name}\n"
            f"Type: {entity.entity_type.value.title()}\n"
            f"HQ Region: {entity.hq_country or 'Unknown'}\n"
            f"Risk Score: {score:.1f}/100\n"
            f"Value-at-Risk: ${var_val:,.2f}"
        )

        nodes.append({
            "id":    entity.id,
            "label": entity.name,
            "title": tooltip_text,
            "color": {"background": colour, "border": "#0f172a",
                      "highlight": {"background": colour, "border": "#f8fafc"}},
            "size":  size,
            "font":  {"color": "#f8fafc", "size": 12},
            "group": entity.entity_type.value,
            "level": entity.depth_level,
            "meta": {
                "entity_id":   entity.id,
                "name":        entity.name,
                "ticker":      entity.ticker or "—",
                "type":        entity.entity_type.value,
                "industry":    entity.industry or "General Operations",
                "country":     entity.hq_country or "US",
                "depth":       entity.depth_level,
                "score":       round(score, 1),
                "risk_level":  level,
                "narrative":   score_obj.narrative if score_obj else "Normal baseline operations detected.",
                "fin_score":   round(score_obj.financial.score, 1) if score_obj else 50.0,
                "ops_score":   round(score_obj.operational.score, 1) if score_obj else 50.0,
                "comp_score":  round(score_obj.compliance.score, 1) if score_obj else 50.0,
                "geo_score":   round(score_obj.geopolitical.score, 1) if score_obj else 50.0,
                "value_at_risk": var_val,
                "backups":     backups,
                "fin_drivers": score_obj.financial.key_drivers if score_obj and score_obj.financial.key_drivers else ["Stable operating margins", "Adequate liquidity buffers"],
                "ops_drivers": score_obj.operational.key_drivers if score_obj and score_obj.operational.key_drivers else ["Standard business continuity plan active"],
                "evidence":    [],
            },
        })
    return nodes


def _build_vis_edges(ps: PipelineState) -> list[dict]:
    edges = []
    for rel in ps.relationships:
        width = max(1, int(rel.dependency_strength * 5))
        colour = "#ef4444" if rel.is_single_source else "#64748b"
        edges.append({
            "from":   rel.source_id,
            "to":     rel.target_id,
            "label":  rel.relationship_type.replace("_", " "),
            "width":  width,
            "color":  {"color": colour, "highlight": "#f8fafc"},
            "arrows": "to",
            "font":   {"size": 9, "color": "#94a3b8", "strokeWidth": 0},
        })
    return edges


def _build_risk_table(ps: PipelineState) -> list[dict]:
    rows = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        if not score_obj:
            continue
        var_val = 0.0
        if ps.graph_metrics and entity.id in ps.graph_metrics.node_metrics:
            var_val = ps.graph_metrics.node_metrics[entity.id].value_at_risk_usd

        nodes_lookup = _build_vis_nodes(ps)
        matched_node = next((n for n in nodes_lookup if n["id"] == entity.id), None)
        meta_data = matched_node["meta"] if matched_node else {}

        rows.append({
            "entity_id": entity.id,
            "name":      entity.name,
            "ticker":    entity.ticker or "—",
            "type":      entity.entity_type.value.title(),
            "country":   entity.hq_country or "—",
            "score":     round(score_obj.composite_score, 1),
            "level":     score_obj.risk_level.value,
            "fin":       round(score_obj.financial.score, 1),
            "ops":       round(score_obj.operational.score, 1),
            "comp":      round(score_obj.compliance.score, 1),
            "geo":       round(score_obj.geopolitical.score, 1),
            "value_at_risk": var_val,
            "meta":      meta_data
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _build_summary_stats(ps: PipelineState) -> dict:
    scores = list(ps.risk_scores.values())
    portfolio_val = ps.graph_metrics.total_portfolio_value_usd if ps.graph_metrics else 0.0
    var_total = ps.graph_metrics.total_value_at_risk_usd if ps.graph_metrics else 0.0
    hhi_val = ps.graph_metrics.geo_concentration_hhi if ps.graph_metrics else 0.0

    return {
        "total_entities":   len(ps.entities),
        "critical_count":   sum(1 for s in scores if s.risk_level == RiskLevel.CRITICAL),
        "high_count":       sum(1 for s in scores if s.risk_level == RiskLevel.HIGH),
        "medium_count":     sum(1 for s in scores if s.risk_level == RiskLevel.MEDIUM),
        "low_count":        sum(1 for s in scores if s.risk_level == RiskLevel.LOW),
        "alert_count":      len(ps.alerts),
        "avg_score":        round(sum(s.composite_score for s in scores) / len(scores), 1) if scores else 0,
        "spof_count":       len(ps.graph_metrics.single_points_of_failure) if ps.graph_metrics else 0,
        "total_portfolio_value": portfolio_val,
        "total_value_at_risk": var_total,
        "geo_concentration_hhi": hhi_val
    }


def generate_dashboard_html(ps: PipelineState) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("dashboard.html.j2")

    context = {
        "target_company":  ps.target_company,
        "generated_at":    datetime.utcnow().strftime("%d %B %Y %H:%M UTC"),
        "run_id":          ps.run_id,
        "stats":           _build_summary_stats(ps),
        "vis_nodes_json":  json.dumps(_build_vis_nodes(ps)),
        "vis_edges_json":  json.dumps(_build_vis_edges(ps)),
        "risk_table_json": json.dumps(_build_risk_table(ps)),
        "alerts_json":     json.dumps([a.model_dump() for a in ps.alerts], default=str),
        "chart_data_json": json.dumps({
            "geo_distribution": ps.graph_metrics.country_spend_distribution if ps.graph_metrics else {}
        }),
        "uploaded_docs_json": json.dumps(ps.uploaded_documents),
        "report_html":     ps.report_html,
        "error_count":     len(ps.errors),
        "errors":          ps.errors,
    }

    return template.render(**context)


async def dashboard_node(state: dict[str, Any]) -> dict[str, Any]:
    from config.settings import settings
    ps = PipelineState(**state)
    ps.stage = "dashboard"

    try:
        html = generate_dashboard_html(ps)
        ps.dashboard_html = html
        filename  = f"{ps.target_company.lower().replace(' ', '_')}_{ps.run_id}.html"
        out_path  = settings.output_dir / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        logger.info(f"[Dashboard] Saved to {out_path}")
    except Exception as e:
        ps.add_error(f"Dashboard assembly failure: {e}")
        logger.error(f"[Dashboard] Error: {e}", exc_info=True)

    return ps.model_dump()