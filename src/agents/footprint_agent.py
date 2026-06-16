"""
Footprint Agent — LangGraph node.
Aggregates all external and internal data for every entity in the watchlist.
Runs all fetches with bounded concurrency.
"""

import logging
import time
from typing import Any

from src.models import PipelineState
from src.data_sources.aggregator import aggregate_all_entities
from src.llm.metrics import add_latency
from config.settings import settings

logger = logging.getLogger(__name__)


async def footprint_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node: collect digital footprint data for all entities.
    Skips the target company itself (depth_level == 0).
    """
    ps = PipelineState(**state)
    ps.stage = "footprint"

    _t0 = time.perf_counter()

    # Only fetch for supply chain entities, not the target itself
    entities_to_fetch = [e for e in ps.entities if e.depth_level > 0]
    logger.info(f"[Footprint] Fetching data for {len(entities_to_fetch)} entities")

    try:
        footprints = await aggregate_all_entities(
            entities=entities_to_fetch,
            newsapi_key=settings.news_api_key,
            sec_user_agent=settings.sec_user_agent,
            concurrency=5,
            llm_backend=ps.llm_backend,
        )
        ps.footprint_data = footprints
    except Exception as e:
        ps.add_error(f"Footprint aggregation failed: {e}")
        logger.error(f"[Footprint] Error: {e}")

    logger.info(
        f"[Footprint] Complete: {len(ps.footprint_data)}/{len(entities_to_fetch)} entities fetched"
    )

    add_latency("footprint_agent", time.perf_counter() - _t0)
    return ps.model_dump()
