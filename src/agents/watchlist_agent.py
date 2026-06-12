"""
Watchlist Agent — LangGraph node.
Given a target company name, calls the LLM to generate a structured
supply chain entity graph (2-3 levels deep).
"""

import json
import logging
import re
import uuid
from typing import Any

from src.models import Entity, EntityRelationship, EntityType, PipelineState
from src.llm.interface import get_llm_client
from src.data_sources.ticker_resolver import resolve_entity_tickers
from src.data_sources.aggregator import load_vendor_registry
from config.prompts import WATCHLIST_PROMPT, SYSTEM_RISK_ANALYST
from config.settings import settings

logger = logging.getLogger(__name__)


def _slug(name: str, suffix: str = "") -> str:
    """Generate a URL-safe entity ID from company name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{suffix}" if suffix else slug


def _attach_internal_spend(entities: list[Entity]) -> None:
    """
    Populate entity.annual_spend_usd from the internal vendor registry where a
    record exists (matched by name). Leaves it None for entities we have no
    procurement record for — cascade analysis falls back to an importance proxy.
    """
    registry = load_vendor_registry()
    matched = 0
    for entity in entities:
        record = registry.get(entity.name.lower())
        if record and record.annual_spend_usd:
            entity.annual_spend_usd = record.annual_spend_usd
            matched += 1
    logger.info(f"Attached internal spend to {matched}/{len(entities)} entities")


def _parse_entities(raw_json: str, target_id: str) -> tuple[list[Entity], list[EntityRelationship]]:
    """Parse LLM JSON output into Entity and EntityRelationship objects."""
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?", "", raw_json).strip()
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse watchlist JSON: {e}\nRaw: {raw_json[:300]}")
        return [], []

    entities: list[Entity] = []
    relationships: list[EntityRelationship] = []

    # Build a name→id map for relationship construction
    name_to_id: dict[str, str] = {}

    for item in data.get("entities", []):
        name        = item.get("name", "Unknown")
        country     = item.get("hq_country", "")
        entity_id   = _slug(name, country.lower() if country else "")
        parent_name = item.get("relationship_to_parent", "")

        try:
            etype = EntityType(item.get("entity_type", "supplier"))
        except ValueError:
            etype = EntityType.SUPPLIER

        entity = Entity(
            id=entity_id,
            name=name,
            ticker=item.get("ticker"),
            entity_type=etype,
            relationship_to_parent=parent_name,
            parent_id=name_to_id.get(parent_name.lower()),
            depth_level=item.get("depth_level", 1),
            importance_score=float(item.get("importance_score", 5)),
            industry=item.get("industry", ""),
            hq_country=country,
        )
        entities.append(entity)
        name_to_id[name.lower()] = entity_id

    # Resolve parent IDs now that all entities are created.
    # Edge direction encodes goods/services flow:
    #   Supplier chain:  deeper_supplier → parent_supplier  (flows TOWARD target)
    #   Customer chain:  parent_customer → deeper_customer  (flows AWAY from target)
    _UPSTREAM = (EntityType.SUPPLIER, EntityType.LOGISTICS, EntityType.PARTNER)

    for entity in entities:
        if entity.relationship_to_parent:
            parent_id = name_to_id.get(entity.relationship_to_parent.lower())
            if parent_id:
                entity.parent_id = parent_id
                if entity.entity_type in _UPSTREAM:
                    src, tgt = entity.id, parent_id   # L2_supplier → L1_supplier
                else:
                    src, tgt = parent_id, entity.id   # L1_customer → L2_customer
                relationships.append(EntityRelationship(
                    source_id=src,
                    target_id=tgt,
                    relationship_type=entity.entity_type.value,
                    dependency_strength=entity.importance_score / 10.0,
                ))

    # Add target ↔ Level-1 entity relationships with correct flow direction
    for entity in entities:
        if entity.depth_level == 1:
            if entity.entity_type in _UPSTREAM:
                src, tgt = entity.id, target_id   # supplier/partner → target
            else:
                src, tgt = target_id, entity.id   # target → customer/financial
            relationships.append(EntityRelationship(
                source_id=src,
                target_id=tgt,
                relationship_type=entity.entity_type.value,
                dependency_strength=entity.importance_score / 10.0,
            ))

    logger.info(f"Parsed {len(entities)} entities, {len(relationships)} relationships")
    return entities, relationships


async def watchlist_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: generate watchlist from target company name.
    Mutates state dict in place and returns it.
    """
    ps = PipelineState(**state)
    ps.stage = "watchlist"
    logger.info(f"[Watchlist] Generating supply chain for: {ps.target_company}")

    # Create the target entity (depth 0)
    target_id = _slug(ps.target_company)
    target_entity = Entity(
        id=target_id,
        name=ps.target_company,
        ticker=ps.target_ticker,
        entity_type=EntityType.TARGET,
        depth_level=0,
        importance_score=10.0,
    )

    # Build prompt and call LLM
    prompt = WATCHLIST_PROMPT.format(
        company_name=ps.target_company,
        max_depth=settings.max_depth,
        max_children=settings.max_children_per_node,
    )

    llm = get_llm_client(ps.llm_backend)
    try:
        raw = await llm.generate_json(prompt, system=SYSTEM_RISK_ANALYST)
        entities, relationships = _parse_entities(raw, target_id)
    except Exception as e:
        ps.add_error(f"Watchlist LLM call failed: {e}")
        logger.error(f"[Watchlist] Error: {e}")
        entities, relationships = [], []

    # Enforce node cap
    if len(entities) > settings.max_entities:
        logger.warning(
            f"Capping entities from {len(entities)} to {settings.max_entities}"
        )
        entities = entities[:settings.max_entities]

    # Ground tickers to real market symbols — the LLM's tickers are unreliable and
    # null/wrong tickers starve every downstream financial/SEC/news fetch.
    try:
        await resolve_entity_tickers(entities)
    except Exception as e:
        ps.add_error(f"Ticker resolution failed: {e}")
        logger.warning(f"[Watchlist] Ticker resolution error: {e}")

    # Attach real internal procurement spend where we have a registry record.
    _attach_internal_spend(entities)

    ps.entities    = [target_entity] + entities
    ps.relationships = relationships

    logger.info(
        f"[Watchlist] Complete: {len(ps.entities)} entities, "
        f"{len(ps.relationships)} relationships"
    )
    return ps.model_dump()