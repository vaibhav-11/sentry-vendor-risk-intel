"""
Cascading risk analysis.
Uses NetworkX centrality metrics to identify critical nodes
and quantify the blast radius of individual entity failures.
"""

import logging
from src.models import NodeMetrics, GraphMetrics, RiskScore
import networkx as nx

logger = logging.getLogger(__name__)


def compute_node_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
) -> dict[str, NodeMetrics]:
    """
    Compute per-node cascade risk metrics.

    cascade_risk_score = composite_risk_score × centrality_multiplier
    blast_radius = % of graph nodes reachable downstream if node is removed
    """
    if G.number_of_nodes() == 0:
        return {}

    # Betweenness centrality — normalised 0-1
    betweenness = nx.betweenness_centrality(G, normalized=True, weight="dependency_strength")

    node_metrics: dict[str, NodeMetrics] = {}
    total_nodes = G.number_of_nodes()

    for node_id in G.nodes():
        in_deg  = G.in_degree(node_id)
        out_deg = G.out_degree(node_id)
        bc      = betweenness.get(node_id, 0.0)

        # Blast radius: how many nodes are reachable downstream if this node fails
        # = descendants + 1 (the node itself) / total nodes
        try:
            descendants = nx.descendants(G, node_id)
        except Exception:
            descendants = set()
        blast_radius = round(len(descendants) / max(total_nodes - 1, 1) * 100, 1)

        # Single point of failure: high centrality + no parallel paths
        is_spof = (bc > 0.15 and in_deg <= 1) or blast_radius > 40

        # Cascade score amplifies the entity's risk score by its centrality
        base_score = risk_scores[node_id].composite_score if node_id in risk_scores else 50.0
        centrality_factor = 1.0 + (bc * cascade_multiplier)
        cascade_score = min(100.0, round(base_score * centrality_factor, 1))

        node_metrics[node_id] = NodeMetrics(
            entity_id=node_id,
            betweenness_centrality=round(bc, 4),
            in_degree=in_deg,
            out_degree=out_deg,
            is_single_point_of_failure=is_spof,
            blast_radius_pct=blast_radius,
            cascade_risk_score=cascade_score,
        )

    return node_metrics


def compute_graph_metrics(
    G: nx.DiGraph,
    risk_scores: dict[str, RiskScore],
    cascade_multiplier: float = 1.4,
) -> GraphMetrics:
    """
    Compute graph-level cascade risk metrics including critical path and SPOFs.
    """
    node_metrics = compute_node_metrics(G, risk_scores, cascade_multiplier)

    # Identify critical path (longest path by dependency strength if DAG)
    critical_path: list[str] = []
    try:
        if nx.is_directed_acyclic_graph(G):
            def edge_weight(u, v, d): return 1 - d.get("dependency_strength", 0.5)
            critical_path = nx.dag_longest_path(G, weight="dependency_strength")
    except Exception:
        pass

    # SPOFs
    spofs = [nid for nid, nm in node_metrics.items() if nm.is_single_point_of_failure]

    # Top cascade risks (sorted by cascade_risk_score)
    top_cascade = sorted(
        node_metrics.keys(),
        key=lambda x: node_metrics[x].cascade_risk_score,
        reverse=True,
    )[:5]

    density = nx.density(G)

    max_depth = 0
    try:
        max_depth = max((G.nodes[n].get("depth_level", 0) for n in G.nodes()), default=0)
    except Exception:
        pass

    metrics = GraphMetrics(
        total_nodes=G.number_of_nodes(),
        total_edges=G.number_of_edges(),
        max_depth=max_depth,
        density=round(density, 4),
        node_metrics=node_metrics,
        critical_path=critical_path,
        single_points_of_failure=spofs,
        top_cascade_risks=top_cascade,
    )

    logger.info(
        f"Graph metrics: {metrics.total_nodes} nodes, "
        f"{len(spofs)} SPOFs, density={density:.3f}"
    )
    return metrics
