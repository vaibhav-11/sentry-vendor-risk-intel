"""
Footprint Agent — LangGraph node.
Aggregates all external and internal data for every entity in the watchlist.
Runs all fetches with bounded concurrency.
"""

import logging
from typing import Any

from src.models import PipelineState
from src.data_sources.aggregator import aggregate_all_entities
from config.settings import settings

logger = logging.getLogger(__name__)


async def footprint_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: collect digital footprint data for all entities.
    Skips the target company itself (depth_level == 0).
    """
    ps = PipelineState(**state)
    ps.stage = "footprint"

    # Only fetch for supply chain entities, not the target itself
    entities_to_fetch = [e for e in ps.entities if e.depth_level > 0]
    logger.info(f"[Footprint] Fetching data for {len(entities_to_fetch)} entities")

    try:
        footprints = await aggregate_all_entities(
            entities=entities_to_fetch,
            newsapi_key=settings.news_api_key,
            sec_user_agent=settings.sec_user_agent,
            concurrency=5,
        )
        ps.footprint_data = footprints
    except Exception as e:
        ps.add_error(f"Footprint aggregation failed: {e}")
        logger.error(f"[Footprint] Error: {e}")

    logger.info(
        f"[Footprint] Complete: {len(ps.footprint_data)}/{len(entities_to_fetch)} entities fetched"
    )
    return ps.model_dump()
