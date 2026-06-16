"""
Run-level instrumentation for the Sentry pipeline.

Two module-level ledgers accumulate across a single process run:

  * TOKEN_LEDGER   — per-caller LLM token usage (prompt / completion / total),
                     populated by the vLLM and Ollama clients after each
                     successful response. The mock backend records zeros so the
                     instrumentation stays inert on the GPU-free path.
  * LATENCY_LEDGER — per-agent wall-clock seconds, populated by the agent layer
                     via the `record_latency` context manager.

`print_run_metrics()` renders both ledgers to stdout and appends a timestamped
snapshot to logs/run_metrics.json so successive runs accumulate without
overwriting.

Nothing here changes pipeline behaviour or output files — it is pure
side-channel measurement.
"""

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Repo root: src/llm/metrics.py → parents[2]
_LOG_DIR  = Path(__file__).resolve().parents[2] / "logs"
_LOG_FILE = _LOG_DIR / "run_metrics.json"

# Canonical agent ordering for the report. The first four also key TOKEN_LEDGER;
# dashboard_generator is latency-only (no LLM call).
_AGENT_ORDER = [
    "watchlist_agent",
    "footprint_agent",
    "risk_agent",
    "report_agent",
    "dashboard_generator",
]
# Agents that issue LLM calls (and therefore appear in the token table).
_TOKEN_AGENTS = _AGENT_ORDER[:4]


# ── Module-level ledgers ──────────────────────────────────────────────────────

# label -> {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
TOKEN_LEDGER: dict[str, dict[str, int]] = {}

# agent name -> elapsed wall-clock seconds (summed across calls)
LATENCY_LEDGER: dict[str, float] = {}

# Total pipeline wall-clock time, set from the generate_demo.py entry point.
PIPELINE_WALL_SECONDS: float = 0.0


def reset_ledgers() -> None:
    """Clear all accumulated metrics. Useful for tests or repeated in-process runs."""
    global PIPELINE_WALL_SECONDS
    TOKEN_LEDGER.clear()
    LATENCY_LEDGER.clear()
    PIPELINE_WALL_SECONDS = 0.0


# ── Token accounting (called by the LLM clients) ──────────────────────────────

def record_tokens(
    label: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """
    Accumulate a single response's usage into TOKEN_LEDGER under `label`.

    Called by the vLLM/Ollama clients after each successful generation; the mock
    client calls this with zeros so the ledger key exists but stays inert.
    """
    bucket = TOKEN_LEDGER.setdefault(
        label or "unknown",
        {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    )
    bucket["prompt_tokens"]     += int(prompt_tokens or 0)
    bucket["completion_tokens"] += int(completion_tokens or 0)
    bucket["total_tokens"]      += int(total_tokens or 0)


# ── Latency accounting (called by the agent layer) ────────────────────────────

@contextmanager
def record_latency(agent_name: str):
    """
    Context manager that times its block with perf_counter() and adds the elapsed
    seconds to LATENCY_LEDGER[agent_name] (accumulating if entered more than once).
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        LATENCY_LEDGER[agent_name] = LATENCY_LEDGER.get(agent_name, 0.0) + elapsed


def add_latency(agent_name: str, seconds: float) -> None:
    """
    Add `seconds` to LATENCY_LEDGER[agent_name] (accumulating across calls).

    Equivalent to the `record_latency` context manager but usable when timing
    with an explicit perf_counter() pair is more convenient than wrapping a block.
    """
    LATENCY_LEDGER[agent_name] = LATENCY_LEDGER.get(agent_name, 0.0) + float(seconds)


def set_pipeline_wall_seconds(seconds: float) -> None:
    """Record total pipeline wall-clock time (from the generate_demo.py entry point)."""
    global PIPELINE_WALL_SECONDS
    PIPELINE_WALL_SECONDS = float(seconds)


# ── Reporter ──────────────────────────────────────────────────────────────────

def _pipeline_token_totals() -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for bucket in TOKEN_LEDGER.values():
        totals["prompt_tokens"]     += bucket["prompt_tokens"]
        totals["completion_tokens"] += bucket["completion_tokens"]
        totals["total_tokens"]      += bucket["total_tokens"]
    return totals


def print_run_metrics() -> None:
    """
    Print a structured latency + token-usage summary to stdout, and append a
    timestamped snapshot to logs/run_metrics.json (created if absent).
    """
    bar = "═" * 43
    rule = "  " + "─" * 41

    lat_total = (
        PIPELINE_WALL_SECONDS
        if PIPELINE_WALL_SECONDS > 0
        else sum(LATENCY_LEDGER.values())
    )

    lines: list[str] = []
    lines.append(bar)
    lines.append("SENTRY RUN METRICS")
    lines.append(bar)
    lines.append("LATENCY")
    for agent in _AGENT_ORDER:
        secs = LATENCY_LEDGER.get(agent, 0.0)
        lines.append(f"  {agent:<22} {secs:.1f}s")
    lines.append(rule)
    lines.append(f"  {'TOTAL (wall clock)':<22} {lat_total:.1f}s")
    lines.append("")
    lines.append("TOKEN USAGE")
    lines.append(f"  {'Agent':<22} {'Prompt':<9} {'Completion':<11} {'Total'}")
    for agent in _TOKEN_AGENTS:
        b = TOKEN_LEDGER.get(
            agent, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        lines.append(
            f"  {agent:<22} "
            f"{b['prompt_tokens']:<9} {b['completion_tokens']:<11} {b['total_tokens']}"
        )
    lines.append(rule)
    totals = _pipeline_token_totals()
    lines.append(
        f"  {'PIPELINE TOTAL':<22} "
        f"{totals['prompt_tokens']:<9} {totals['completion_tokens']:<11} {totals['total_tokens']}"
    )
    lines.append(bar)

    print("\n".join(lines))

    _append_json_snapshot(lat_total, totals)


def _append_json_snapshot(lat_total: float, token_totals: dict[str, int]) -> None:
    """Append this run's metrics to logs/run_metrics.json as one entry in a list."""
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_seconds": {
            **{a: round(LATENCY_LEDGER.get(a, 0.0), 3) for a in _AGENT_ORDER},
            "total_wall_clock": round(lat_total, 3),
        },
        "token_usage": {
            **{
                a: TOKEN_LEDGER.get(
                    a,
                    {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                )
                for a in _TOKEN_AGENTS
            },
            "pipeline_total": token_totals,
        },
    }

    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        if _LOG_FILE.exists():
            try:
                existing = json.loads(_LOG_FILE.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except (json.JSONDecodeError, OSError):
                existing = []
        else:
            existing = []
        existing.append(snapshot)
        _LOG_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError as e:
        logger.warning(f"Could not write run metrics to {_LOG_FILE}: {e}")
