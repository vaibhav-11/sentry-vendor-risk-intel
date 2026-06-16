"""
HTML Dashboard Generator.
Takes a completed PipelineState and renders a fully self-contained
single-file HTML dashboard — no server required, opens in any browser.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from src.models import PipelineState, RiskLevel
from src.llm.metrics import add_latency
from config.settings import settings

logger = logging.getLogger(__name__)
TEMPLATES_DIR = Path(__file__).parent / "templates"


def _score_to_colour(score: float) -> str:
    if score >= 80: return "#ef4444"
    if score >= 65: return "#f97316"
    if score >= 45: return "#eab308"
    return "#22c55e"


# ── B2: node radius proportional to contract spend (sqrt-scaled) ──
# Square-root keeps a $22B vendor from visually swamping a $1B one while still
# making spend legible. The target (Apple) is pinned to MAX_NODE_SIZE by design.
MIN_NODE_SIZE = 18.0
MAX_NODE_SIZE = 60.0      # ceiling for vendors; the target sits strictly above this
TARGET_NODE_SIZE = 72.0  # target (Apple) is largest by design, regardless of its own spend
# k tuned so the spend range stays differentiated below the cap:
# ~$1B -> ~27px, ~$8.5B -> ~44px, ~$22B -> ~58px (just under the 60 ceiling).
NODE_SIZE_K   = 0.00028


def _spend_for_entity(entity) -> float:
    """Contract spend used for sizing/VaR; falls back to the importance proxy
    used in cascading_risk.py when no procurement record exists."""
    if entity.annual_spend_usd is not None:
        return float(entity.annual_spend_usd)
    return float(entity.importance_score * 1_850_000)


def _node_size_for_spend(spend: float) -> float:
    import math
    return round(max(MIN_NODE_SIZE, min(MAX_NODE_SIZE,
                 MIN_NODE_SIZE + NODE_SIZE_K * math.sqrt(max(spend, 0.0)))), 1)


def _build_vis_nodes(ps: PipelineState) -> list[dict]:
    nodes = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        score     = score_obj.composite_score if score_obj else 0
        level     = score_obj.risk_level.value if score_obj else "unknown"
        colour    = _score_to_colour(score)

        spend     = _spend_for_entity(entity)
        is_target = entity.entity_type.value == "target" or entity.depth_level == 0
        size      = TARGET_NODE_SIZE if is_target else _node_size_for_spend(spend)

        var_val = 0.0
        is_spof = False
        lineage_payload = {}
        if ps.graph_metrics and entity.id in ps.graph_metrics.node_metrics:
            nm_data = ps.graph_metrics.node_metrics[entity.id]
            var_val = nm_data.value_at_risk_usd
            is_spof = nm_data.is_single_point_of_failure
            if nm_data.mathematical_lineage:
                lineage_payload = nm_data.mathematical_lineage.model_dump()

        # Map dynamic evidence registry data from the footprint layer
        provenance_payload = {}
        fp_record = ps.footprint_data.get(entity.id)
        if fp_record and hasattr(fp_record, 'provenance_anchors'):
            provenance_payload = {k: v.model_dump() for k, v in fp_record.provenance_anchors.items()}

        # Tooltip surfaces the visual encoding: size ∝ spend, ring colour = risk band,
        # ring thickness = exposure (VaR). The target itself is not scored.
        if is_target:
            tooltip = (
                f"{entity.name}\n"
                f"Type: Target — not scored  |  {entity.hq_country or 'US'}\n"
                f"Contract spend: ${spend:,.0f}\n"
                f"(node size ∝ √spend)"
            )
        else:
            tooltip = (
                f"{entity.name}\n"
                f"Type: {entity.entity_type.value.title()}  |  {entity.hq_country or 'US'}\n"
                f"Contract spend: ${spend:,.0f}\n"
                f"Composite risk: {round(score, 1)}/100\n"
                f"Value-at-Risk: ${var_val:,.0f}\n"
                f"(node size ∝ √spend · ring colour = risk band · ring thickness = exposure)"
            )

        nodes.append({
            "id":    entity.id,
            "label": entity.name,
            "size":  size,
            "title": tooltip,
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
                "comp_drivers": score_obj.compliance.key_drivers if score_obj else [],
                "geo_drivers": score_obj.geopolitical.key_drivers if score_obj else [],

                # C1: data gaps per dimension (surfaced explicitly, never silent default)
                "fin_gaps": score_obj.financial.data_gaps if score_obj else [],
                "ops_gaps": score_obj.operational.data_gaps if score_obj else [],
                "comp_gaps": score_obj.compliance.data_gaps if score_obj else [],
                "geo_gaps": score_obj.geopolitical.data_gaps if score_obj else [],

                # E3/F2/C2: structured, URL-bearing evidence per dimension
                "fin_evidence": [e.model_dump() for e in score_obj.financial.evidence] if score_obj else [],
                "comp_evidence": [e.model_dump() for e in score_obj.compliance.evidence] if score_obj else [],
                "geo_evidence": [e.model_dump() for e in score_obj.geopolitical.evidence] if score_obj else [],

                # G2: ranked, justified pre-vetted alternative vendors
                "backups": score_obj.backups if score_obj else [],

                # C1: dimension weights for the inspector breakdown
                "weights": {"fin": 30, "ops": 30, "comp": 20, "geo": 20},

                "annual_spend_usd": round(spend, 2),
                "importance":  entity.importance_score,
                "is_target":   is_target,
                "is_spof":     is_spof,
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
    # Build node meta once (not per-row) and index by id.
    meta_by_id = {n["id"]: n["meta"] for n in _build_vis_nodes(ps)}

    # B4: dependence = vendor spend / total vendor portfolio spend. Use the same
    # vendor-only portfolio total computed in cascading_risk.py (target excluded).
    total_portfolio_spend = (
        ps.graph_metrics.total_portfolio_value_usd
        if ps.graph_metrics and ps.graph_metrics.total_portfolio_value_usd
        else 0.0
    )

    rows = []
    for entity in ps.entities:
        score_obj = ps.risk_scores.get(entity.id)
        if not score_obj:
            continue
        var_val = 0.0
        if ps.graph_metrics and entity.id in ps.graph_metrics.node_metrics:
            var_val = ps.graph_metrics.node_metrics[entity.id].value_at_risk_usd

        spend = _spend_for_entity(entity)
        dependence_pct = round(spend / total_portfolio_spend * 100.0, 1) if total_portfolio_spend else 0.0

        rows.append({
            "entity_id": entity.id,
            "name":      entity.name,
            "ticker":    entity.ticker or "—",
            "type":      entity.entity_type.value.title(),
            "country":   entity.hq_country or "—",
            "industry":  entity.industry or "General Operations",
            "score":     round(score_obj.composite_score, 1),
            "level":     score_obj.risk_level.value,
            "fin":       round(score_obj.financial.score, 1),
            "ops":       round(score_obj.operational.score, 1),
            "comp":      round(score_obj.compliance.score, 1),
            "geo":       round(score_obj.geopolitical.score, 1),
            "annual_spend_usd": round(spend, 2),
            "dependence_pct": dependence_pct,
            "value_at_risk": var_val,
            "meta":      meta_by_id.get(entity.id, {})
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def _abbrev_usd(value: float) -> str:
    """Abbreviated currency for stat cards ($84.2B, $26.3B, $850.0M).
    Full precision is preserved in the analytics table — this is header-only."""
    v = float(value or 0.0)
    if abs(v) >= 1e9:
        return f"${v / 1e9:.1f}B"
    if abs(v) >= 1e6:
        return f"${v / 1e6:.1f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:,.0f}"


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
        "geo_concentration_hhi": hhi_val,
        # Abbreviated header strings (full precision stays in the analytics table).
        "total_portfolio_value_abbr": _abbrev_usd(portfolio_val),
        "total_value_at_risk_abbr": _abbrev_usd(var_total),
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

    # ── Per-source retrieval timestamps (latest retrieved_at per source group) ──
    # DriverEvidence carries no source_name, so group by the anchor-key prefix
    # ("financials", "sec", "news", "geo") which maps cleanly to a data source.
    _SOURCE_LABELS = {
        "financials": "Yahoo Finance (financials)",
        "sec": "SEC EDGAR (filings)",
        "news": "GDELT / News wire",
        "geo": "GDELT (geopolitical events)",
    }
    latest_by_source: dict[str, datetime] = {}
    for fp in ps.footprint_data.values():
        for key, anchor in fp.provenance_anchors.items():
            prefix = key.split(".")[0]
            source = _SOURCE_LABELS.get(prefix, prefix.title())
            ts = anchor.retrieved_at
            if source not in latest_by_source or ts > latest_by_source[source]:
                latest_by_source[source] = ts

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


# Any vendor at or above this composite is worth a playbook even without a
# structural flag (SPOF / single-source / financial distress).
_PLAYBOOK_COMPOSITE_FLOOR = 35.0


def _build_playbooks(ps: PipelineState) -> list[dict]:
    """
    Generate one action playbook for EVERY vendor with discernible risk, ranked by
    severity (composite desc, then VaR desc). A vendor qualifies if its composite is
    at/above the floor, OR it is a single point of failure, OR it is single-source,
    OR it carries a financial-distress driver. The playbook flavour is chosen from
    the dominant risk: single-source → dual-sourcing; SPOF → resilience; otherwise
    continuity outreach. Each body cites that vendor's own top drivers.
    """
    candidates: list[dict] = []

    for eid, s in ps.risk_scores.items():
        e = ps.entity_by_id(eid)
        if not e or e.depth_level <= 0:
            continue

        # Structural flags pulled from footprint + graph metrics.
        fp = ps.footprint_data.get(eid)
        single_source = bool(fp and fp.internal_record and fp.internal_record.single_source)
        is_spof = False
        var_val = 0.0
        if ps.graph_metrics and eid in ps.graph_metrics.node_metrics:
            nm = ps.graph_metrics.node_metrics[eid]
            is_spof = nm.is_single_point_of_failure
            var_val = nm.value_at_risk_usd
        distress = any("distress" in d.lower() for d in s.financial.key_drivers)

        if not (s.composite_score >= _PLAYBOOK_COMPOSITE_FLOOR
                or is_spof or single_source or distress):
            continue

        candidates.append({
            "eid": eid, "entity": e, "score": s,
            "single_source": single_source, "is_spof": is_spof,
            "var": var_val, "distress": distress,
        })

    # Most severe first: composite, then exposure as the tie-breaker.
    candidates.sort(key=lambda c: (c["score"].composite_score, c["var"]), reverse=True)

    playbooks: list[dict] = []
    for idx, c in enumerate(candidates, start=1):
        e, s = c["entity"], c["score"]
        drivers = (s.financial.key_drivers + s.operational.key_drivers
                   + s.compliance.key_drivers + s.geopolitical.key_drivers)
        drivers = [d for d in drivers if d][:3]
        driver_text = "; ".join(drivers) if drivers else "elevated composite risk score"

        if c["single_source"]:
            title = f"Initiate dual-sourcing for {e.name} (single-source dependency)"
            subject = f"Dual-sourcing & resilience plan — {e.name}"
            body = (
                f"Dear Sourcing Operations Team,\n\n"
                f"{e.name} is a single-source vendor with no approved alternate "
                f"(composite risk {s.composite_score:.0f}/100, {s.risk_level.value}). "
                f"Key drivers: {driver_text}.\n\n"
                f"Please initiate qualification of at least one alternate supplier and report "
                f"a target qualification timeline within 10 business days.\n\n"
                f"Regards,\nProcurement Resilience Office"
            )
        elif c["is_spof"]:
            title = f"Resilience review — {e.name} (single point of failure)"
            subject = f"SPOF mitigation & contingency — {e.name}"
            body = (
                f"Dear Sourcing Operations Team,\n\n"
                f"{e.name} is a single point of failure in the supply network "
                f"(composite risk {s.composite_score:.0f}/100, {s.risk_level.value}). "
                f"Key drivers: {driver_text}.\n\n"
                f"Please document downstream dependents and stand up a contingency plan, "
                f"with an updated business continuity plan requested within 10 business days.\n\n"
                f"Regards,\nProcurement Resilience Office"
            )
        else:
            title = f"Mitigate {e.name} exposure (composite {s.composite_score:.0f}/100)"
            subject = f"Risk review & continuity assurance — {e.name}"
            body = (
                f"Dear {e.name} Account Team,\n\n"
                f"Our third-party risk monitoring has flagged {e.name} at a composite "
                f"risk score of {s.composite_score:.0f}/100 ({s.risk_level.value}). "
                f"Key drivers: {driver_text}.\n\n"
                f"To support continuity planning, please provide your current business continuity "
                f"plan and any updated financial disclosures within 5 business days.\n\n"
                f"Regards,\nStrategic Procurement & Risk Operations"
            )

        playbooks.append({"id": idx, "title": title, "subject": subject, "body": body})

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

    _t0 = time.perf_counter()

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

    add_latency("dashboard_generator", time.perf_counter() - _t0)
    return ps.model_dump()