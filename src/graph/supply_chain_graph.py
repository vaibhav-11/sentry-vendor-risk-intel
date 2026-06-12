"""
Builds and manages the supply chain graph using NetworkX.
Entities are nodes; relationships are directed edges.
"""

import json
import logging
from pathlib import Path

import networkx as nx

from src.models import Entity, EntityRelationship, RiskScore

logger = logging.getLogger(__name__)


def build_graph(
    entities: list[Entity],
    relationships: list[EntityRelationship],
) -> nx.DiGraph:
    """
    Build a directed NetworkX graph from entities and relationships.
    Node attributes include entity metadata and risk scores (populated later).
    """
    G = nx.DiGraph()

    for entity in entities:
        attrs = {
            "name": entity.name,
            "ticker": entity.ticker or "",
            "entity_type": entity.entity_type.value,
            "depth_level": entity.depth_level,
            "importance_score": entity.importance_score,
            "industry": entity.industry,
            "hq_country": entity.hq_country,
            "is_public": entity.is_public,
            # Risk fields (populated after scoring)
            "composite_score": 0.0,
            "risk_level": "unknown",
            "financial_score": 0.0,
            "operational_score": 0.0,
            "compliance_score": 0.0,
            "geopolitical_score": 0.0,
        }
        # Only set spend when we actually know it, so cascade analysis can fall
        # back to its importance proxy for entities with no procurement record.
        if entity.annual_spend_usd is not None:
            attrs["annual_spend_usd"] = entity.annual_spend_usd
        G.add_node(entity.id, **attrs)

    for rel in relationships:
        if rel.source_id in G and rel.target_id in G:
            G.add_edge(rel.source_id, rel.target_id, **{
                "relationship_type": rel.relationship_type,
                "dependency_strength": rel.dependency_strength,
                "annual_value_usd": rel.annual_value_usd or 0,
                "is_single_source": rel.is_single_source,
            })

    logger.info(f"Built graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def attach_risk_scores(G: nx.DiGraph, risk_scores: dict[str, RiskScore]) -> nx.DiGraph:
    """Annotate graph nodes with computed risk scores."""
    for entity_id, score in risk_scores.items():
        if entity_id in G:
            G.nodes[entity_id].update({
                "composite_score":    score.composite_score,
                "risk_level":         score.risk_level.value,
                "financial_score":    score.financial.score,
                "operational_score":  score.operational.score,
                "compliance_score":   score.compliance.score,
                "geopolitical_score": score.geopolitical.score,
                "risk_narrative":     score.narrative,
            })
    return G


def entities_to_relationships(entities: list[Entity]) -> list[EntityRelationship]:
    """
    Derive relationships from entity parent_id fields.
    Used when relationships are not explicitly provided by the watchlist agent.
    """
    rels: list[EntityRelationship] = []
    entity_map = {e.id: e for e in entities}

    for entity in entities:
        if entity.parent_id and entity.parent_id in entity_map:
            parent = entity_map[entity.parent_id]
            rels.append(EntityRelationship(
                source_id=parent.id,
                target_id=entity.id,
                relationship_type=entity.entity_type.value,
                dependency_strength=entity.importance_score / 10.0,
            ))
    return rels


def save_graph(G: nx.DiGraph, path: Path) -> None:
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Graph saved to {path}")


def load_graph(path: Path) -> nx.DiGraph:
    with open(path) as f:
        data = json.load(f)
    return nx.node_link_graph(data, directed=True)


def get_subgraph_for_entity(G: nx.DiGraph, entity_id: str, hops: int = 2) -> nx.DiGraph:
    """Return a subgraph centred on a given entity within N hops."""
    nodes = {entity_id}
    for _ in range(hops):
        neighbours = set()
        for n in nodes:
            neighbours |= set(G.predecessors(n)) | set(G.successors(n))
        nodes |= neighbours
    return G.subgraph(nodes).copy()
