"""
Cascading risk analysis.
Uses NetworkX centrality metrics to identify critical nodes,
quantifies the financial Value-at-Risk (VaR), and measures regional concentration indices.
"""

import logging
import networkx as nx
from src.models import (
    MathematicalLineage, NodeMetrics, GraphMetrics, RiskScore, FootprintData,
)

logger = logging.getLogger(__name__)

# B1 severity model: a node with no internal record gets the neutral 0.6 default.
SEVERITY_BASE          = 0.5
SEVERITY_SINGLE_SOURCE = 0.3
SEVERITY_NO_ALTERNATE  = 0.2
SEVERITY_DEFAULT       = 0.6   # 0.5 + 0.3*0 ... applied when no internal record exists


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _disruption_flags(
    node_id: str,
    footprint_data: dict[str, FootprintData] | None,
) -> tuple[bool, bool, bool]:
    """
    Resolve (single_source, no_alternate_available, has_internal_record) for a node
    from its internal vendor record. Returns has_record=False when no record exists,
    so the caller can fall back to the neutral severity default.
    """
    if footprint_data and node_id in footprint_data:
        rec = footprint_data[node_id].internal_record
        if rec is not None:
            return bool(rec.single_source), (not rec.alternate_vendor_available), True
    return False, False, False


def compute_node_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
    footprint_data: dict[str, FootprintData] | None = None,
) -> dict[str, NodeMetrics]:
    """
    Compute per-node cascade risk metrics with a risk-scaled Value-at-Risk model (B1).

    VaR_total = direct_VaR + cascade_VaR, where
      p_disruption = composite_score / 100
      severity     = clamp(0.5 + 0.3*single_source + 0.2*no_alternate, 0, 1)  (0.6 if no record)
      direct_VaR   = direct_spend * p_disruption * severity   (hard invariant: <= direct_spend)
      cascade_VaR  = Sum over upstream dependencies of
                       child_direct_spend * (child_composite/100) * dependency_strength

    A node's "dependencies" are its graph predecessors: edges point from a supplier
    into the entity that depends on it (`L2_supplier -> L1_supplier -> target`), and the
    incoming edge carries the `dependency_strength` of that dependence.
    """
    if G.number_of_nodes() == 0:
        return {}

    betweenness = nx.betweenness_centrality(G, normalized=True, weight="dependency_strength")
    node_metrics: dict[str, NodeMetrics] = {}
    total_nodes = G.number_of_nodes()

    def _direct_spend(nid: str) -> float:
        attrs = G.nodes[nid]
        importance = attrs.get("importance_score", 5.0)
        return float(attrs.get("annual_spend_usd", importance * 1_850_000))

    for node_id in G.nodes():
        in_deg  = G.in_degree(node_id)
        out_deg = G.out_degree(node_id)
        bc      = betweenness.get(node_id, 0.0)

        try: descendants = nx.descendants(G, node_id)
        except Exception: descendants = set()
        blast_radius = round(len(descendants) / max(total_nodes - 1, 1) * 100, 1)

        # Fix 5: betweenness alone misses real bottlenecks. TSMC sits at
        # betweenness 0.0625 — below the 0.15 bar — yet it is single-source with
        # three downstream dependents, which makes it a single point of failure by
        # definition. A single-source vendor that anything depends on IS a SPOF,
        # regardless of centrality. Resolve the single-source flag from the
        # internal vendor record (same source the severity model uses).
        node_single_source, _, _ = _disruption_flags(node_id, footprint_data)
        is_spof = (
            (bc > 0.15 and in_deg <= 1)
            or blast_radius > 40
            or (node_single_source and len(descendants) >= 1)
        )
        base_score = risk_scores[node_id].composite_score if node_id in risk_scores else 50.0

        centrality_factor = 1.0 + (bc * cascade_multiplier)
        cascade_score = min(100.0, round(base_score * centrality_factor, 1))

        direct_spend = _direct_spend(node_id)

        # ── B1: risk-scaled direct VaR ──────────────────────────────────────
        p_disruption = base_score / 100.0
        single_source, no_alternate, has_record = _disruption_flags(node_id, footprint_data)
        if has_record:
            severity = _clamp(
                SEVERITY_BASE
                + SEVERITY_SINGLE_SOURCE * single_source
                + SEVERITY_NO_ALTERNATE * no_alternate,
                0.0, 1.0,
            )
        else:
            severity = SEVERITY_DEFAULT
        # Invariant: p_disruption <= 1 and severity <= 1, so direct_var <= direct_spend.
        direct_var = direct_spend * p_disruption * severity

        # ── B1: cascade VaR from genuine upstream dependencies ──────────────
        # Each dependency contributes its own spend, scaled by its own risk and by
        # how strongly this node depends on it (the incoming edge weight).
        cascade_var = 0.0
        for dep_id in G.predecessors(node_id):
            child_spend = _direct_spend(dep_id)
            child_score = (
                risk_scores[dep_id].composite_score if dep_id in risk_scores else 50.0
            )
            dep_strength = G.edges[dep_id, node_id].get("dependency_strength", 0.5)
            cascade_var += child_spend * (child_score / 100.0) * dep_strength

        # Fix 1: cascade VaR is, by definition, exposure propagated to *downstream
        # dependents*. A node with no downstream dependents (a leaf — e.g. the
        # customer nodes AT&T / Amazon, whose only graph neighbour is the target
        # they hang off of) cannot cascade risk to anyone. Gate it hard to zero so
        # an incoming predecessor edge can never manufacture phantom cascade VaR.
        if len(descendants) == 0:
            cascade_var = 0.0

        var_total = direct_var + cascade_var

        # Retain a structural propagated-exposure figure for the audit trail.
        ASSET_SCALING_FACTOR = 450000.0
        propagated_exposure = sum(
            float(G.nodes[d].get("importance_score", 5.0) * ASSET_SCALING_FACTOR)
            for d in descendants
        )

        lineage = MathematicalLineage(
            direct_spend_usd=round(direct_spend, 2),
            betweenness_centrality=round(bc, 4),
            cascade_multiplier=cascade_multiplier,
            calculated_centrality_factor=round(centrality_factor, 4),
            downstream_dependent_nodes_count=len(descendants),
            raw_propagated_exposure_usd=round(propagated_exposure, 2),
            composite_risk_weight=round(p_disruption, 4),
            p_disruption=round(p_disruption, 4),
            severity=round(severity, 4),
            single_source=single_source,
            no_alternate_available=no_alternate,
            direct_var_usd=round(direct_var, 2),
            cascade_var_usd=round(cascade_var, 2),
            final_value_at_risk_usd=round(var_total, 2),
        )

        node_metrics[node_id] = NodeMetrics(
            entity_id=node_id,
            betweenness_centrality=round(bc, 4),
            in_degree=in_deg,
            out_degree=out_deg,
            is_single_point_of_failure=is_spof,
            blast_radius_pct=blast_radius,
            cascade_risk_score=cascade_score,
            direct_spend_usd=round(direct_spend, 2),
            value_at_risk_usd=round(var_total, 2),
            alternative_suppliers=G.nodes[node_id].get("backups", ["Alternative Provider Alpha"]),
            mathematical_lineage=lineage,
        )

    return node_metrics


def _is_target(G: nx.DiGraph, node_id: str) -> bool:
    """Target entity (e.g. Apple) — excluded from vendor-portfolio aggregations (B3)."""
    attrs = G.nodes[node_id]
    return attrs.get("entity_type") == "target" or attrs.get("depth_level", 1) == 0


def compute_graph_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
    footprint_data: dict[str, FootprintData] | None = None,
) -> GraphMetrics:
    """
    Compute graph-level cascade metrics including portfolio metrics and HHI Geopolitical indices.
    """
    node_metrics = compute_node_metrics(G, risk_scores, cascade_multiplier, footprint_data)

    critical_path: list[str] = []
    try:
        if nx.is_directed_acyclic_graph(G):
            critical_path = nx.dag_longest_path(G, weight="dependency_strength")
    except Exception:
        pass

    spofs = [nid for nid, nm in node_metrics.items() if nm.is_single_point_of_failure]
    top_cascade = sorted(node_metrics.keys(), key=lambda x: node_metrics[x].cascade_risk_score, reverse=True)[:5]
    density = nx.density(G)

    max_depth = 0
    try:
        max_depth = max((G.nodes[n].get("depth_level", 0) for n in G.nodes()), default=0)
    except Exception:
        pass

    # ── CALCULATE FINANCIAL SHOCK & GEOPOLITICAL CONCENTRATION VALUES ──
    # B3: vendor-only aggregations. The target entity (Apple) is excluded — its own
    # spend would inflate the portfolio total, and its geo=50.0 default must never feed
    # any score-derived portfolio figure. Total VaR is the straight sum of per-node
    # VaR_total from the B1 model (already risk-scaled — no second risk multiplier here).
    vendor_metrics = {
        nid: nm for nid, nm in node_metrics.items() if not _is_target(G, nid)
    }
    total_portfolio_value = sum(nm.direct_spend_usd for nm in vendor_metrics.values())
    total_var = sum(nm.value_at_risk_usd for nm in vendor_metrics.values())

    # Country distribution maps (vendors only — target excluded from HHI)
    country_spend_map = {}
    for node_id, metrics in vendor_metrics.items():
        country = G.nodes[node_id].get("hq_country", "US")
        country_spend_map[country] = country_spend_map.get(country, 0.0) + metrics.direct_spend_usd

    # Compute Herfindahl-Hirschman Concentration Index (HHI)
    hhi_score = 0.0
    country_distribution = {}
    if total_portfolio_value > 0:
        for country, spend in country_spend_map.items():
            percentage = (spend / total_portfolio_value) * 100.0
            country_distribution[country] = round(percentage, 1)
            hhi_score += percentage ** 2

    metrics = GraphMetrics(
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        max_depth=max_depth,
        density=round(density, 4),
        node_metrics=node_metrics,
        critical_path=critical_path,
        single_points_of_failure=spofs,
        top_cascade_risks=top_cascade,
        total_portfolio_value_usd=round(total_portfolio_value, 2),
        total_value_at_risk_usd=round(total_var, 2),
        geo_concentration_hhi=round(hhi_score, 2),
        country_spend_distribution=country_distribution
    )

    logger.info(f"Financial Portfolio VaR Engine Calculated. HHI Concentration: {hhi_score:.2f}")
    return metrics