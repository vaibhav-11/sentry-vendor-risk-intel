"""
Pre-vetted alternatives ranking (G2).

Loads industry-keyed candidate alternative vendors from data/alternatives_seed.yaml,
then for each scored entity asks the LLM to rank the candidates and justify each in
one sentence given that entity's specific risk drivers. The ranked, justified list is
stored on RiskScore.backups, which the dashboard surfaces as node.meta.backups.
"""

import json
import logging
import re

import yaml

from src.models import PipelineState
from src.llm.interface import BaseLLMClient
from config.prompts import ALTERNATIVES_PROMPT, SYSTEM_RISK_ANALYST
from config.settings import settings

logger = logging.getLogger(__name__)

_seed_cache: dict[str, list[str]] | None = None


def load_alternatives_seed() -> dict[str, list[str]]:
    """Load industry → candidate-list mapping (cached)."""
    global _seed_cache
    if _seed_cache is not None:
        return _seed_cache
    path = settings.alternatives_seed_path
    if not path.exists():
        logger.warning(f"Alternatives seed not found at {path}")
        _seed_cache = {}
        return _seed_cache
    with open(path) as f:
        _seed_cache = yaml.safe_load(f) or {}
    logger.info(f"Loaded alternatives seed for {len(_seed_cache)} industries")
    return _seed_cache


def _parse_ranked(raw: str, candidates: list[str]) -> list[dict]:
    """Parse the LLM JSON ranking; fall back to seed order if parsing fails."""
    try:
        clean = re.sub(r"```(?:json)?", "", raw).strip()
        data = json.loads(clean)
        out = []
        for item in data:
            if isinstance(item, dict) and item.get("name"):
                out.append({
                    "name": str(item["name"]),
                    "justification": str(item.get("justification", "")).strip(),
                })
        if out:
            return out
    except Exception as e:
        logger.debug(f"Alternatives parse fell back to seed order: {e}")
    # Fallback: seed order with a generic justification so the list is never empty.
    return [
        {"name": c, "justification": "Pre-vetted same-industry alternative supplier."}
        for c in candidates
    ]


async def attach_alternatives(ps: PipelineState, llm: BaseLLMClient) -> None:
    """
    For every scored entity that has a seed entry for its industry, rank the
    candidate alternatives via the LLM and store them on RiskScore.backups.
    """
    seed = load_alternatives_seed()
    if not seed:
        return

    prompts: list[str] = []
    target_ids: list[tuple[str, list[str]]] = []

    for eid, score in ps.risk_scores.items():
        entity = ps.entity_by_id(eid)
        if entity is None:
            continue
        # Exact match first; then case-insensitive; then a partial/substring match.
        # The mock industries are hand-aligned to the seed keys, but real LLM
        # backends emit free-form labels ("semiconductor foundry", lowercase, etc.)
        # that miss an exact key — which silently emptied every node's backups.
        ind = (entity.industry or "").lower()
        candidates = seed.get(entity.industry) or next(
            (v for k, v in seed.items() if k.lower() == ind), None
        ) or next(
            (v for k, v in seed.items()
             if ind and (k.lower() in ind or ind in k.lower())), None
        )
        logger.debug(
            f"Alternatives seed lookup: {entity.name!r} "
            f"industry={entity.industry!r} -> {bool(candidates)}"
        )
        if not candidates:
            continue
        # Exclude the entity itself if it appears among its own industry candidates.
        candidates = [c for c in candidates if c.lower() != entity.name.lower()]
        if not candidates:
            continue

        drivers = (
            score.financial.key_drivers[:2]
            + score.operational.key_drivers[:1]
            + score.geopolitical.key_drivers[:1]
        )
        prompt = ALTERNATIVES_PROMPT.format(
            entity_name=entity.name,
            industry=entity.industry,
            hq_country=entity.hq_country or "—",
            risk_score=f"{score.composite_score:.0f}",
            risk_drivers="\n".join(f"- {d}" for d in drivers) or "- elevated composite risk",
            candidates="\n".join(f"- {c}" for c in candidates),
        )
        prompts.append(prompt)
        target_ids.append((eid, candidates))

    if not prompts:
        return

    responses = await llm.generate_batch(
        prompts, system=SYSTEM_RISK_ANALYST, temperature=0.2
    )
    for (eid, candidates), raw in zip(target_ids, responses):
        ps.risk_scores[eid].backups = _parse_ranked(raw, candidates)

    logger.info(f"Attached ranked alternatives to {len(target_ids)} entities")
