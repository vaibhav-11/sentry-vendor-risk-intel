"""
Watchlist Agent — LangGraph node.
Given a target company name, calls the LLM to generate a structured
supply chain entity graph (2-3 levels deep).
"""

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any

from src.models import Entity, EntityRelationship, EntityType, PipelineState
from src.risk.scorer import normalize_country
import time

from src.llm.interface import get_llm_client
from src.llm.metrics import add_latency
from src.data_sources.ticker_resolver import resolve_entity_tickers
from src.data_sources.aggregator import load_vendor_registry
from config.prompts import WATCHLIST_PROMPT, TIER2_EXPANSION_PROMPT, SYSTEM_RISK_ANALYST
from config.settings import settings

logger = logging.getLogger(__name__)

# Curated tier-1 seeds, backend-independent. The watchlist LLM (any backend) is
# unreliable about completeness — a vLLM run returned only 8 nodes. When a seed
# exists for the target, tier-1 is loaded deterministically from disk and the LLM
# is used only for tier-2 upstream expansion. Non-seeded targets fall back to the
# LLM-only path unchanged.
_SEED_DIR = Path(__file__).resolve().parents[2] / "data" / "seed"
_SEED_REGISTRY: dict[str, str] = {
    "apple-inc": "apple_network.json",
}

# How many of the most material tier-1 nodes to expand into tier-2.
_TIER2_EXPAND_TOP_N = 3


def _load_tier1_seed(target_company: str) -> list[dict] | None:
    """
    Return the curated tier-1 entity dicts for a seeded target, or None when no
    seed exists (caller falls back to the LLM watchlist path).
    """
    filename = _SEED_REGISTRY.get(_slug(target_company))
    if not filename:
        return None
    seed_path = _SEED_DIR / filename
    if not seed_path.exists():
        logger.warning(f"[Watchlist] Seed registered but missing on disk: {seed_path}")
        return None
    try:
        data = json.loads(seed_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[Watchlist] Failed to read seed {seed_path}: {e}")
        return None
    entities = data.get("entities", [])
    if not entities:
        return None
    logger.info(f"[Watchlist] Loaded {len(entities)} tier-1 entities from seed {filename}")
    return entities


async def _expand_tier2(
    tier1_raw: list[dict],
    target_company: str,
    llm,
) -> list[dict]:
    """
    Agentic tier-2 expansion (A5): pick the most material tier-1 nodes by
    importance_score and ask the LLM for each one's top upstream suppliers. Returns
    the parsed tier-2 entity dicts (depth_level forced to 2, parent wired). Failures
    for an individual parent are logged and skipped — never fatal.
    """
    tier1_only = [e for e in tier1_raw if e.get("depth_level", 1) == 1]
    top = sorted(
        tier1_only,
        key=lambda e: e.get("importance_score", 5),
        reverse=True,
    )[:_TIER2_EXPAND_TOP_N]

    tier2_raw: list[dict] = []
    seen_names = {e.get("name", "").lower() for e in tier1_raw}
    for parent in top:
        parent_name = parent.get("name", "")
        prompt = TIER2_EXPANSION_PROMPT.format(
            target_company=target_company,
            parent_name=parent_name,
            parent_industry=parent.get("industry", "Unknown"),
            parent_country=parent.get("hq_country", ""),
        )
        try:
            raw = await llm.generate_json(prompt, system=SYSTEM_RISK_ANALYST,
                                          label="watchlist_agent")
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            items = json.loads(clean).get("entities", [])
        except Exception as e:
            logger.warning(f"[Watchlist] Tier-2 expansion failed for {parent_name}: {e}")
            continue

        for item in items:
            name = item.get("name", "")
            if not name or name.lower() in seen_names:
                continue   # skip blanks and duplicates (avoid colliding node IDs)
            item["depth_level"] = 2
            item["relationship_to_parent"] = parent_name
            tier2_raw.append(item)
            seen_names.add(name.lower())

    logger.info(
        f"[Watchlist] Tier-2 expansion added {len(tier2_raw)} entities "
        f"across {len(top)} parent nodes"
    )
    return tier2_raw


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

    # Tier-1 and tier-2 expansions are concatenated upstream, so the same company
    # can legitimately appear twice (e.g. a vendor that supplies both the target
    # and a tier-1 supplier). Their slugs collide, producing duplicate entity ids.
    # A duplicate id is fatal downstream: vis.js DataSet.add() silently aborts the
    # entire batch on a repeated id, leaving a blank graph, and the id-keyed
    # score/footprint dicts would clobber each other. So we keep one canonical
    # Entity per id (first occurrence — the tier closest to the target) while still
    # capturing every occurrence's parent link below, so a shared vendor keeps all
    # of its edges.
    seen_ids: set[str] = set()
    # (entity_id, entity_type, parent_name, depth_level, importance_score) for
    # every parsed item, duplicates included — drives relationship construction.
    link_specs: list[tuple[str, EntityType, str, int, float]] = []

    for item in data.get("entities", []):
        name        = item.get("name", "Unknown")
        country     = normalize_country(item.get("hq_country", ""))
        entity_id   = _slug(name, country.lower() if country else "")
        parent_name = item.get("relationship_to_parent", "")

        try:
            etype = EntityType(item.get("entity_type", "supplier"))
        except ValueError:
            etype = EntityType.SUPPLIER

        # Customers are out of scope: the target (Apple) is the terminal/rightmost
        # node, so downstream channel entities are dropped here. The seed carries
        # none; this guard stops the LLM fallback path from reintroducing them.
        if etype == EntityType.CUSTOMER:
            continue

        depth      = item.get("depth_level", 1)
        importance = float(item.get("importance_score", 5))

        # Resolve names to ids regardless of dedup so cross-tier parents wire up.
        name_to_id[name.lower()] = entity_id
        link_specs.append((entity_id, etype, parent_name, depth, importance))

        if entity_id in seen_ids:
            logger.warning(
                f"Duplicate entity id '{entity_id}' (name='{name}') from LLM output — "
                f"merging into first occurrence, edges preserved"
            )
            continue
        seen_ids.add(entity_id)

        entity = Entity(
            id=entity_id,
            name=name,
            ticker=item.get("ticker"),
            entity_type=etype,
            relationship_to_parent=parent_name,
            parent_id=name_to_id.get(parent_name.lower()),
            depth_level=depth,
            importance_score=importance,
            industry=item.get("industry", ""),
            hq_country=country,
        )
        entities.append(entity)

    # Resolve parent IDs now that all entities are created.
    # Edge direction encodes goods/services flow:
    #   Supplier chain:  deeper_supplier → parent_supplier  (flows TOWARD target)
    #   Customer chain:  parent_customer → deeper_customer  (flows AWAY from target)
    _UPSTREAM = (EntityType.SUPPLIER, EntityType.LOGISTICS, EntityType.PARTNER)

    # Re-resolve each entity's parent_id now that every name is known — a parent
    # listed after its child would have resolved to None at construction time.
    # supply_chain_graph derives edges from this field, so it must be populated.
    for entity in entities:
        if entity.relationship_to_parent:
            resolved = name_to_id.get(entity.relationship_to_parent.lower())
            if resolved:
                entity.parent_id = resolved

    # Edges are built from link_specs (every occurrence) rather than the deduped
    # entity list, so a shared vendor keeps an edge to each parent it was listed
    # under. A seen-set drops identical edges that a literal duplicate would emit.
    seen_edges: set[tuple[str, str, str]] = set()

    def _add_edge(src: str, tgt: str, rel_type: str, strength: float) -> None:
        if src == tgt:
            return
        key = (src, tgt, rel_type)
        if key in seen_edges:
            return
        seen_edges.add(key)
        relationships.append(EntityRelationship(
            source_id=src,
            target_id=tgt,
            relationship_type=rel_type,
            dependency_strength=strength,
        ))

    for entity_id, etype, parent_name, _depth, importance in link_specs:
        if parent_name:
            parent_id = name_to_id.get(parent_name.lower())
            if parent_id:
                if etype in _UPSTREAM:
                    src, tgt = entity_id, parent_id   # L2_supplier → L1_supplier
                else:
                    src, tgt = parent_id, entity_id   # L1_customer → L2_customer
                _add_edge(src, tgt, etype.value, importance / 10.0)

    # Add target ↔ Level-1 entity relationships with correct flow direction
    for entity_id, etype, _parent_name, depth, importance in link_specs:
        if depth == 1:
            if etype in _UPSTREAM:
                src, tgt = entity_id, target_id   # supplier/partner → target
            else:
                src, tgt = target_id, entity_id   # target → customer/financial
            _add_edge(src, tgt, etype.value, importance / 10.0)

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

    _t0 = time.perf_counter()

    # Create the target entity (depth 0)
    target_id = _slug(ps.target_company)
    target_entity = Entity(
        id=target_id,
        name=ps.target_company,
        ticker=ps.target_ticker,
        entity_type=EntityType.TARGET,
        depth_level=0,
        importance_score=10.0,
        # Target HQ defaults to US (the worked example is Apple Inc). Leaving this
        # blank creates an empty "" bucket in the geo HHI concentration chart.
        hq_country="US",
    )

    llm = get_llm_client(ps.llm_backend)

    # ── Tier-1: seed-first, LLM fallback ────────────────────────────────────
    # A curated seed guarantees a complete, verified tier-1 set regardless of
    # backend. Only non-seeded targets fall through to the LLM watchlist prompt.
    tier1_raw = _load_tier1_seed(ps.target_company)
    if tier1_raw is None:
        prompt = WATCHLIST_PROMPT.format(
            company_name=ps.target_company,
            max_depth=settings.max_depth,
            max_children=settings.max_children_per_node,
        )
        try:
            raw = await llm.generate_json(prompt, system=SYSTEM_RISK_ANALYST,
                                          label="watchlist_agent")
            clean = re.sub(r"```(?:json)?", "", raw).strip()
            tier1_raw = json.loads(clean).get("entities", [])
        except Exception as e:
            ps.add_error(f"Watchlist LLM call failed: {e}")
            logger.error(f"[Watchlist] Error: {e}")
            tier1_raw = []

    # ── Tier-2: agentic upstream expansion on the most material nodes ───────
    tier2_raw: list[dict] = []
    if tier1_raw:
        try:
            tier2_raw = await _expand_tier2(tier1_raw, ps.target_company, llm)
        except Exception as e:
            ps.add_error(f"Tier-2 expansion failed: {e}")
            logger.warning(f"[Watchlist] Tier-2 expansion error: {e}")

    # Combine and build the entity/relationship graph in a single pass so the
    # parent-wiring logic stays consistent across tiers.
    combined_raw = tier1_raw + tier2_raw
    entities, relationships = _parse_entities(
        json.dumps({"target": ps.target_company, "entities": combined_raw}),
        target_id,
    )

    # Enforce node cap (applies after expansion — invariant D)
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

    add_latency("watchlist_agent", time.perf_counter() - _t0)
    return ps.model_dump()