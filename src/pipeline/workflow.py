"""
LangGraph pipeline orchestrator.
Wires all agent nodes into a directed StateGraph:

  watchlist → footprint → risk_scoring → report → dashboard → END
"""

import asyncio
import logging
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, END

from src.models import PipelineState
from src.agents.watchlist_agent  import watchlist_node
from src.agents.footprint_agent  import footprint_node
from src.agents.risk_agent       import risk_node
from src.agents.report_agent     import report_node
from src.dashboard.html_generator import dashboard_node

logger = logging.getLogger(__name__)


# ── Build the graph ────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    workflow = StateGraph(dict)   # state is a plain dict (PipelineState serialised)

    workflow.add_node("watchlist",   watchlist_node)
    workflow.add_node("footprint",   footprint_node)
    workflow.add_node("risk_scoring", risk_node)
    workflow.add_node("report",      report_node)
    workflow.add_node("dashboard",   dashboard_node)

    workflow.set_entry_point("watchlist")
    workflow.add_edge("watchlist",    "footprint")
    workflow.add_edge("footprint",    "risk_scoring")
    workflow.add_edge("risk_scoring", "report")
    workflow.add_edge("report",       "dashboard")
    workflow.add_edge("dashboard",    END)

    return workflow.compile()


# ── Public runner ─────────────────────────────────────────────────────────────

async def run_pipeline(
    target_company: str,
    target_ticker: str | None = None,
    llm_backend: str = "mock",
) -> PipelineState:
    """
    Run the full vendor risk pipeline for a target company.
    Returns a completed PipelineState with dashboard_html populated.
    """
    run_id = str(uuid.uuid4())[:8]
    logger.info(
        f"[Pipeline] Starting run {run_id} | "
        f"company='{target_company}' | backend={llm_backend}"
    )

    initial_state = PipelineState(
        target_company=target_company,
        target_ticker=target_ticker,
        llm_backend=llm_backend,
        run_id=run_id,
    ).model_dump()

    pipeline = build_pipeline()

    try:
        final_state_dict = await pipeline.ainvoke(initial_state)
        final_state = PipelineState(**final_state_dict)
        final_state.completed_at = datetime.utcnow()

        elapsed = (final_state.completed_at - final_state.started_at).total_seconds()
        logger.info(
            f"[Pipeline] Run {run_id} complete in {elapsed:.1f}s | "
            f"entities={len(final_state.entities)} | "
            f"alerts={len(final_state.alerts)} | "
            f"errors={len(final_state.errors)}"
        )
        if final_state.errors:
            logger.warning(f"[Pipeline] Errors: {final_state.errors}")

        return final_state

    except Exception as e:
        logger.error(f"[Pipeline] Fatal error in run {run_id}: {e}", exc_info=True)
        raise


def run_pipeline_sync(
    target_company: str,
    target_ticker: str | None = None,
    llm_backend: str = "mock",
) -> PipelineState:
    """Synchronous wrapper — use in Jupyter notebooks or scripts."""
    return asyncio.run(
        run_pipeline(target_company, target_ticker, llm_backend)
    )
