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

from jinja2 import Environment, FileSystemLoader, BaseLoader
from src.models import PipelineState, RiskLevel, AlertSeverity

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


# ── Colour helpers ─────────────────────────────────────────────────────────────

RISK_COLOURS = {
    RiskLevel.CRITICAL: {"bg": "#ef4444", "text": "#fff",    "border": "#b91c1c"},
    RiskLevel.HIGH:     {"bg": "#f97316", "text": "#fff",    "border": "#c2410c"},
    RiskLevel.MEDIUM:   {"bg": "#eab308", "text": "#1e293b", "border": "#a16207"},
    RiskLevel.LOW:      {"bg": "#22c55e", "text": "#fff",    "border": "#15803d"},
}
UNKNOWN_COLOUR = {"bg": "#64748b", "text": "#fff", "border": "#475569"}


def _risk_colour(level_str: str) -> dict:
    try:
        return RISK_COLOURS[RiskLevel(level_str)]
    except (ValueError, KeyError):
        return UNKNOWN_COLOUR


def _score_to_colour(score: float) -> str:
    """Map a 0-100 score to a hex colour for vis.js nodes."""
    if score >= 80: return "#ef4444"
    if score >= 65: return "#f97316"
    if score >= 45: return "#eab308"
    return "#22c55e"


# ── Data serialisers ───────────────────────────────────────────────────────────

def _build_vis_nodes(ps: PipelineState) -> list[dict]:
    """Convert entities + risk scores into vis.js DataSet node format."""
    nodes = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        score     = score_obj.composite_score if score_obj else 0
        level     = score_obj.risk_level.value if score_obj else "unknown"
        colour    = _score_to_colour(score)
        size      = 20 + entity.importance_score * 3   # 23–50px

        tooltip_parts = [
            f"<b>{entity.name}</b>",
            f"Type: {entity.entity_type.value.title()}",
            f"Country: {entity.hq_country or 'Unknown'}",
        ]
        if score_obj:
            tooltip_parts += [
                f"Risk Score: {score:.0f}/100",
                f"Risk Level: {level.upper()}",
                f"Financial: {score_obj.financial.score:.0f}",
                f"Operational: {score_obj.operational.score:.0f}",
            ]

        nodes.append({
            "id":    entity.id,
            "label": entity.name,
            "title": "<br>".join(tooltip_parts),
            "color": {"background": colour, "border": "#0f172a",
                      "highlight": {"background": colour, "border": "#f8fafc"}},
            "size":  size,
            "font":  {"color": "#f8fafc", "size": 12},
            "group": entity.entity_type.value,
            "level": entity.depth_level,
            # Extra data for drill-down modal
            "meta": {
                "entity_id":   entity.id,
                "name":        entity.name,
                "ticker":      entity.ticker or "—",
                "type":        entity.entity_type.value,
                "industry":    entity.industry,
                "country":     entity.hq_country,
                "depth":       entity.depth_level,
                "score":       round(score, 1),
                "risk_level":  level,
                "narrative":   score_obj.narrative if score_obj else "",
                "fin_score":   round(score_obj.financial.score, 1) if score_obj else 0,
                "ops_score":   round(score_obj.operational.score, 1) if score_obj else 0,
                "comp_score":  round(score_obj.compliance.score, 1) if score_obj else 0,
                "geo_score":   round(score_obj.geopolitical.score, 1) if score_obj else 0,
                "fin_drivers": score_obj.financial.key_drivers if score_obj else [],
                "ops_drivers": score_obj.operational.key_drivers if score_obj else [],
            },
        })
    return nodes


def _build_vis_edges(ps: PipelineState) -> list[dict]:
    """Convert relationships into vis.js edge format."""
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
    """Sorted risk score table for the Risk Scores tab."""
    rows = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        if not score_obj:
            continue
        rows.append({
            "name":    entity.name,
            "ticker":  entity.ticker or "—",
            "type":    entity.entity_type.value.title(),
            "country": entity.hq_country or "—",
            "score":   round(score_obj.composite_score, 1),
            "level":   score_obj.risk_level.value,
            "fin":     round(score_obj.financial.score, 1),
            "ops":     round(score_obj.operational.score, 1),
            "comp":    round(score_obj.compliance.score, 1),
            "geo":     round(score_obj.geopolitical.score, 1),
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _build_alerts_data(ps: PipelineState) -> list[dict]:
    """Serialise alerts for the Alerts tab."""
    return [
        {
            "id":        a.alert_id,
            "entity":    a.entity_name,
            "title":     a.alert_title,
            "severity":  a.severity.value,
            "summary":   a.summary,
            "action":    a.recommended_action,
            "escalate":  a.escalate_to,
            "timing":    a.time_sensitivity,
            "score":     round(a.triggering_score, 1),
            "triggered": a.triggered_at.strftime("%Y-%m-%d %H:%M UTC"),
        }
        for a in ps.alerts
    ]


def _build_summary_stats(ps: PipelineState) -> dict:
    """Summary stat cards at the top of the dashboard."""
    scores = list(ps.risk_scores.values())
    return {
        "total_entities":   len(ps.entities),
        "critical_count":   sum(1 for s in scores if s.risk_level == RiskLevel.CRITICAL),
        "high_count":       sum(1 for s in scores if s.risk_level == RiskLevel.HIGH),
        "medium_count":     sum(1 for s in scores if s.risk_level == RiskLevel.MEDIUM),
        "low_count":        sum(1 for s in scores if s.risk_level == RiskLevel.LOW),
        "alert_count":      len(ps.alerts),
        "avg_score":        round(
            sum(s.composite_score for s in scores) / len(scores), 1
        ) if scores else 0,
        "spof_count":       len(
            ps.graph_metrics.single_points_of_failure
        ) if ps.graph_metrics else 0,
    }


def _build_chart_data(ps: PipelineState) -> dict:
    """Data for Plotly charts."""
    table = _build_risk_table(ps)[:15]   # Top 15 for bar chart
    return {
        "bar_names":  [r["name"][:20] for r in table],
        "bar_scores": [r["score"] for r in table],
        "bar_colors": [_score_to_colour(r["score"]) for r in table],
        "dimension_names":  ["Financial", "Operational", "Compliance", "Geopolitical"],
    }


# ── Main generator ─────────────────────────────────────────────────────────────

def generate_dashboard_html(ps: PipelineState) -> str:
    """
    Render the full self-contained HTML dashboard from a completed PipelineState.
    """
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
        "alerts_json":     json.dumps(_build_alerts_data(ps)),
        "chart_data_json": json.dumps(_build_chart_data(ps)),
        "report_html":     ps.report_html,
        "error_count":     len(ps.errors),
        "errors":          ps.errors,
    }

    return template.render(**context)


def save_dashboard(html: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Dashboard saved: {output_path} ({len(html):,} bytes)")
    return output_path


# ── LangGraph node ─────────────────────────────────────────────────────────────

async def dashboard_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: generate and save the HTML dashboard."""
    from config.settings import settings
    ps = PipelineState(**state)
    ps.stage = "dashboard"

    try:
        html = generate_dashboard_html(ps)
        ps.dashboard_html = html

        # Save to outputs dir
        filename  = f"{ps.target_company.lower().replace(' ', '_')}_{ps.run_id}.html"
        out_path  = settings.output_dir / filename
        save_dashboard(html, out_path)
        logger.info(f"[Dashboard] Saved to {out_path}")
    except Exception as e:
        ps.add_error(f"Dashboard generation failed: {e}")
        logger.error(f"[Dashboard] Error: {e}", exc_info=True)

    return ps.model_dump()
