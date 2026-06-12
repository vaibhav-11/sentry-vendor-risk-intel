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
from config.settings import settings

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
        lineage_payload = {}
        if ps.graph_metrics and entity.id in ps.graph_metrics.node_metrics:
            nm_data = ps.graph_metrics.node_metrics[entity.id]
            var_val = nm_data.value_at_risk_usd
            if nm_data.mathematical_lineage:
                lineage_payload = nm_data.mathematical_lineage.model_dump()

        # Map dynamic evidence registry data from the footprint layer
        provenance_payload = {}
        fp_record = ps.footprint_data.get(entity.id)
        if fp_record and hasattr(fp_record, 'provenance_anchors'):
            provenance_payload = {k: v.model_dump() for k, v in fp_record.provenance_anchors.items()}

        nodes.append({
            "id":    entity.id,
            "label": entity.name,
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
                "score":       round(score, 1),
                "risk_level":  level,
                "narrative":   score_obj.narrative if score_obj else "Baseline audit cycle complete.",
                
                # REMOVE STRIPPED FALLBACKS — MAP DIRECTLY TO DATA SCHEMAS
                "fin_score":   round(score_obj.financial.score, 1) if score_obj else 50.0,
                "ops_score":   round(score_obj.operational.score, 1) if score_obj else 50.0,
                "comp_score":  round(score_obj.compliance.score, 1) if score_obj else 50.0,
                "geo_score":   round(score_obj.geopolitical.score, 1) if score_obj else 50.0,
                
                "fin_drivers": score_obj.financial.key_drivers if score_obj else [],
                "ops_drivers": score_obj.operational.key_drivers if score_obj else [],
                
                "value_at_risk": var_val,
                "mathematical_lineage": lineage_payload,
                "provenance_anchors": provenance_payload
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


def _human_age(dt: datetime, now: datetime) -> str:
    """Human-friendly relative age string for a retrieval timestamp."""
    delta = now - dt
    secs = delta.total_seconds()
    if secs < 0:
        return "just now"
    days = int(secs // 86400)
    if days >= 1:
        return f"{days} day{'s' if days != 1 else ''} ago"
    hours = int(secs // 3600)
    if hours >= 1:
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    mins = int(secs // 60)
    return f"{mins} minute{'s' if mins != 1 else ''} ago"


def _build_freshness_panel(ps: PipelineState) -> dict:
    """
    Build the data-freshness panel: per-source retrieval timestamps drawn from
    the A3 provenance anchors, plus per-node composite deltas vs. the last cached
    run. This is the honest, statically-deliverable replacement for the old fake
    'refresh' animation — a self-contained HTML file has no backend to re-fetch.
    """
    now = datetime.utcnow()

    # ── Per-source retrieval timestamps (latest extracted_at per source_name) ──
    latest_by_source: dict[str, datetime] = {}
    for fp in ps.footprint_data.values():
        for anchor in fp.provenance_anchors.values():
            ts = anchor.extracted_at
            if anchor.source_name not in latest_by_source or ts > latest_by_source[anchor.source_name]:
                latest_by_source[anchor.source_name] = ts

    sources = [
        {
            "source": name,
            "retrieved_at": ts.strftime("%d %b %Y %H:%M UTC"),
            "age": _human_age(ts, now),
        }
        for name, ts in sorted(latest_by_source.items())
    ]

    # ── Change-since-last-run: diff current composites against the cache ──
    current = {
        eid: round(s.composite_score, 1) for eid, s in ps.risk_scores.items()
    }
    prior = _load_last_run_cache(ps.target_company)

    deltas: list[dict] = []
    baseline = prior is None
    if not baseline:
        for eid, score in current.items():
            name = ps.risk_scores[eid].entity_name
            if eid in prior:
                delta = round(score - prior[eid], 1)
                if abs(delta) >= 0.1:
                    deltas.append({"name": name, "delta": delta, "current": score})
            else:
                deltas.append({"name": name, "delta": None, "current": score, "new": True})
        deltas.sort(key=lambda d: abs(d["delta"]) if d["delta"] is not None else 1e9, reverse=True)

    # Persist this run's composites for the next diff.
    _save_last_run_cache(ps.target_company, current)

    return {
        "sources": sources,
        "deltas": deltas[:12],
        "baseline": baseline,
        "generated_at": now.strftime("%d %b %Y %H:%M UTC"),
    }


def _cache_path(company: str) -> Path:
    safe = company.lower().replace(" ", "_").replace("/", "_")
    return settings.cache_dir / f"{safe}_last_run.json"


def _load_last_run_cache(company: str) -> dict[str, float] | None:
    path = _cache_path(company)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data.get("composites")
    except Exception as e:
        logger.warning(f"Failed to read last-run cache: {e}")
        return None


def _save_last_run_cache(company: str, composites: dict[str, float]) -> None:
    path = _cache_path(company)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "saved_at": datetime.utcnow().isoformat(),
            "composites": composites,
        }, indent=2))
    except Exception as e:
        logger.warning(f"Failed to write last-run cache: {e}")


def _build_playbooks(ps: PipelineState) -> list[dict]:
    """
    Generate action playbooks from the actual highest-risk entities in this run,
    rather than two hardcoded generic emails. Targets the highest composite score
    and any single-source vendors flagged in the internal registry.
    """
    playbooks: list[dict] = []

    scored = [
        (eid, s) for eid, s in ps.risk_scores.items()
        if (e := ps.entity_by_id(eid)) and e.depth_level > 0
    ]
    if not scored:
        return playbooks

    # 1. Highest-composite vendor → continuity / contingency outreach.
    top_eid, top_score = max(scored, key=lambda kv: kv[1].composite_score)
    top_entity = ps.entity_by_id(top_eid)
    top_drivers = (top_score.financial.key_drivers + top_score.operational.key_drivers)[:3]
    driver_text = "; ".join(top_drivers) if top_drivers else "elevated composite risk score"
    playbooks.append({
        "id": 1,
        "title": f"Mitigate {top_entity.name} exposure (composite {top_score.composite_score:.0f}/100)",
        "subject": f"Risk review & continuity assurance — {top_entity.name}",
        "body": (
            f"Dear {top_entity.name} Account Team,\n\n"
            f"Our third-party risk monitoring has flagged {top_entity.name} at a composite "
            f"risk score of {top_score.composite_score:.0f}/100 ({top_score.risk_level.value}). "
            f"Key drivers: {driver_text}.\n\n"
            f"To support continuity planning, please provide your current business continuity "
            f"plan and any updated financial disclosures within 5 business days.\n\n"
            f"Regards,\nStrategic Procurement & Risk Operations"
        ),
    })

    # 2. Single-source vendor (if any) → dual-sourcing initiation.
    single_source = None
    for eid, s in scored:
        fp = ps.footprint_data.get(eid)
        if fp and fp.internal_record and fp.internal_record.single_source:
            single_source = (eid, s, fp)
            break

    if single_source:
        eid, s, fp = single_source
        e = ps.entity_by_id(eid)
        playbooks.append({
            "id": 2,
            "title": f"Initiate dual-sourcing for {e.name} (single-source dependency)",
            "subject": f"Dual-sourcing & resilience plan — {e.name}",
            "body": (
                f"Dear Sourcing Operations Team,\n\n"
                f"{e.name} is currently a single-source vendor with no approved alternate "
                f"(composite risk {s.composite_score:.0f}/100). This represents a concentrated "
                f"operational dependency.\n\n"
                f"Please initiate qualification of at least one alternate supplier and report "
                f"a target qualification timeline within 10 business days.\n\n"
                f"Regards,\nProcurement Resilience Office"
            ),
        })
    else:
        # Fall back to the second-highest composite vendor.
        ranked = sorted(scored, key=lambda kv: kv[1].composite_score, reverse=True)
        if len(ranked) > 1:
            eid, s = ranked[1]
            e = ps.entity_by_id(eid)
            playbooks.append({
                "id": 2,
                "title": f"Enhanced monitoring — {e.name} (composite {s.composite_score:.0f}/100)",
                "subject": f"Quarterly risk check-in — {e.name}",
                "body": (
                    f"Dear {e.name} Account Team,\n\n"
                    f"{e.name} carries a composite risk score of {s.composite_score:.0f}/100 "
                    f"({s.risk_level.value}) in our current assessment cycle. We are scheduling "
                    f"an enhanced quarterly review.\n\n"
                    f"Please confirm availability and share updated compliance certifications.\n\n"
                    f"Regards,\nVendor Risk Management"
                ),
            })

    return playbooks


def _build_dataroom_docs(ps: PipelineState) -> list[dict]:
    """
    Build an *illustrative* document-ingestion list keyed to node IDs that
    actually exist in this run's graph (the old VAULT_DOCUMENTS referenced
    hardcoded IDs like 'tsmc-tw' that may not be present). These are clearly
    labelled as roadmap/illustrative on the tab — no live RAG processing happens.
    """
    vendors = [
        e for e in ps.entities
        if e.depth_level > 0 and ps.risk_scores.get(e.id)
    ]
    vendors.sort(key=lambda e: ps.risk_scores[e.id].composite_score, reverse=True)

    docs: list[dict] = []
    for e in vendors[:3]:
        docs.append({
            "name": f"MSA_{e.name.replace(' ', '_')}.pdf",
            "type": "Master Service Agreement",
            "size": "—",
            "linked_node": e.id,
            "linked_name": e.name,
        })
    return docs


def generate_dashboard_html(ps: PipelineState) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    template = env.get_template("dashboard.html.j2")

    context = {
        "target_company":  ps.target_company,
        "generated_at":    datetime.utcnow().strftime("%d %B %Y %H:%M UTC"),
        "run_id":          ps.run_id,
        "stats":           _build_summary_stats(ps),
        "vis_nodes_json":  json.dumps(_build_vis_nodes(ps), default=str),
        "vis_edges_json":  json.dumps(_build_vis_edges(ps), default=str),
        "risk_table_json": json.dumps(_build_risk_table(ps), default=str),
        "alerts_json":     json.dumps([a.model_dump() for a in ps.alerts], default=str),
        "chart_data_json": json.dumps({
            "geo_distribution": ps.graph_metrics.country_spend_distribution if ps.graph_metrics else {}
        }),
        "dataroom_docs_json": json.dumps(_build_dataroom_docs(ps)),
        "playbooks_json":   json.dumps(_build_playbooks(ps)),
        "freshness_json":   json.dumps(_build_freshness_panel(ps)),
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