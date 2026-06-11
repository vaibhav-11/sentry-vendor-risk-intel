"""
Cascading risk analysis.
Uses NetworkX centrality metrics to identify critical nodes,
quantifies the financial Value-at-Risk (VaR), and measures regional concentration indices.
"""

import logging
import networkx as nx
from src.models import MathematicalLineage, NodeMetrics, GraphMetrics, RiskScore

logger = logging.getLogger(__name__)


def compute_node_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
) -> dict[str, NodeMetrics]:
    """
    Compute per-node cascade risk metrics, mapping financial spend value exposures and alternative backups.
    """
    if G.number_of_nodes() == 0:
        return {}

    betweenness = nx.betweenness_centrality(G, normalized=True, weight="dependency_strength")
    node_metrics: dict[str, NodeMetrics] = {}
    total_nodes = G.number_of_nodes()

    # Pre-calculate fallback alternative market options for dashboard playbook visualization
    mock_backups = {
        "semiconductor": ["Intel Foundry Services (US)", "Samsung Electronics (KR)", "GlobalFoundries (US)"],
        "software": ["AWS Compliance Cloud (US)", "Microsoft Azure Gov (US)"],
        "logistics": ["FedEx Custom Critical (US)", "DHL Global Forwarding (DE)"]
    }

    # Refactor loop inside compute_node_metrics in src/graph/cascading_risk.py

    for node_id in G.nodes():
        in_deg  = G.in_degree(node_id)
        out_deg = G.out_degree(node_id)
        bc      = betweenness.get(node_id, 0.0)

        try: descendants = nx.descendants(G, node_id)
        except Exception: descendants = set()
        blast_radius = round(len(descendants) / max(total_nodes - 1, 1) * 100, 1)

        is_spof = (bc > 0.15 and in_deg <= 1) or blast_radius > 40
        base_score = risk_scores[node_id].composite_score if node_id in risk_scores else 50.0
        
        centrality_factor = 1.0 + (bc * cascade_multiplier)
        cascade_score = min(100.0, round(base_score * centrality_factor, 1))

        node_attrs = G.nodes[node_id]
        importance = node_attrs.get("importance_score", 5.0)
        direct_spend = node_attrs.get("annual_spend_usd", float(importance * 1_850_000))
        
        # Compute downstream dependent value exposure using a structural model
        ASSET_SCALING_FACTOR = 450000.0
        propagated_exposure = sum([float(G.nodes[d].get("importance_score", 5.0) * ASSET_SCALING_FACTOR) for d in descendants])
        risk_weight_ratio = base_score / 100.0
        calculated_var = direct_spend + (propagated_exposure * risk_weight_ratio)

        # Instantiate the complete mathematical audit trail
        lineage = MathematicalLineage(
            direct_spend_usd=round(direct_spend, 2),
            betweenness_centrality=round(bc, 4),
            cascade_multiplier=cascade_multiplier,
            calculated_centrality_factor=round(centrality_factor, 4),
            downstream_dependent_nodes_count=len(descendants),
            raw_propagated_exposure_usd=round(propagated_exposure, 2),
            composite_risk_weight=round(risk_weight_ratio, 4),
            final_value_at_risk_usd=round(calculated_var, 2)
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
            value_at_risk_usd=round(calculated_var, 2),
            alternative_suppliers=node_attrs.get("backups", ["Alternative Provider Alpha"]),
            mathematical_lineage=lineage # Attach the tracking lineage to the pipeline state
        )

    # for node_id in G.nodes():
    #     in_deg  = G.in_degree(node_id)
    #     out_deg = G.out_degree(node_id)
    #     bc      = betweenness.get(node_id, 0.0)

    #     try:
    #         descendants = nx.descendants(G, node_id)
    #     except Exception:
    #         descendants = set()
    #     blast_radius = round(len(descendants) / max(total_nodes - 1, 1) * 100, 1)

    #     is_spof = (bc > 0.15 and in_deg <= 1) or blast_radius > 40

    #     base_score = risk_scores[node_id].composite_score if node_id in risk_scores else 50.0
    #     centrality_factor = 1.0 + (bc * cascade_multiplier)
    #     cascade_score = min(100.0, round(base_score * centrality_factor, 1))

    #     # ── EXTRACT OR DERIVE PROCUREMENT STRATEGIC SPEND DATA ──
    #     node_attrs = G.nodes[node_id]
    #     importance = node_attrs.get("importance_score", 5.0)
        
    #     # Pull mock financial value if internal record isn't fully linked
    #     direct_spend = node_attrs.get("annual_spend_usd", float(importance * 1_850_000))
        
    #     # Value at Risk calculation = direct contract spend + downstream dependent impact exposures
    #     propagated_exposure = sum([float(G.nodes[d].get("importance_score", 5.0) * 450_000) for d in descendants])
    #     calculated_var = direct_spend + (propagated_exposure * (base_score / 100.0))

    #     # Map smart alternative backup arrays based on node properties
    #     industry_lower = node_attrs.get("industry", "manufacturing").lower()
    #     backups = mock_backups.get("semiconductor")
    #     if "software" in industry_lower or "cloud" in industry_lower:
    #         backups = mock_backups.get("software")
    #     elif "logistics" in industry_lower or "shipping" in industry_lower:
    #         backups = mock_backups.get("logistics")

    #     node_metrics[node_id] = NodeMetrics(
    #         entity_id=node_id,
    #         betweenness_centrality=round(bc, 4),
    #         in_degree=in_deg,
    #         out_degree=out_deg,
    #         is_single_point_of_failure=is_spof,
    #         blast_radius_pct=blast_radius,
    #         cascade_risk_score=cascade_score,
    #         direct_spend_usd=round(direct_spend, 2),
    #         value_at_risk_usd=round(calculated_var, 2),
    #         alternative_suppliers=backups or ["Alternative Sourcing Option A", "Alternative Sourcing Option B"]
    #     )

    return node_metrics


def compute_graph_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
) -> GraphMetrics:
    """
    Compute graph-level cascade metrics including portfolio metrics and HHI Geopolitical indices.
    """
    node_metrics = compute_node_metrics(G, risk_scores, cascade_multiplier)

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
    total_portfolio_value = sum([nm.direct_spend_usd for nm in node_metrics.values()])
    total_var = sum([nm.value_at_risk_usd * (risk_scores[nid].composite_score / 100.0) if nid in risk_scores else 0 for nid, nm in node_metrics.items()])

    # Country distribution maps
    country_spend_map = {}
    for node_id, metrics in node_metrics.items():
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