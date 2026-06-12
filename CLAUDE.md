# CLAUDE.md — Sentry Vendor Risk Intelligence

> Repo-root context file, auto-loaded by Claude Code each session.
> The active engineering plan lives in `docs/ROADMAP.md`.

## What this is
Sentry is an agentic vendor / third-party risk platform. Given a target company, it maps that
company's supply-chain network, aggregates financial, operational, compliance, and geopolitical
risk signals from public sources, scores each entity across four weighted dimensions, and renders a
**single self-contained interactive HTML dashboard** with LLM-written risk narratives. The worked
example throughout the codebase uses Apple Inc. as the target company.

## Design priorities
Two principles govern every change to this system:

1. **Grounding** — every score, figure, and claim on the dashboard must trace back to a real, cited source.
2. **Explainability** — a reader must be able to see *why* a score is what it is, not just the final number.

Prefer changes that strengthen these over changes that add surface area. The litmus test for any
feature: *does the output let a user see that the analysis is real and verify where it came from?*

## Stack
LangGraph (5-node `StateGraph`) · NetworkX · Pydantic v2 · Jinja2 · vis.js · Plotly · ChromaDB ·
yfinance · GDELT · SEC EDGAR. LLM inference sits behind a single backend-agnostic interface; the
intended production backend is **vLLM on ROCm (AMD MI300X)** running Qwen2.5-14B-Instruct-GPTQ-Int4,
with a fully functional mock backend for GPU-free local development.

## Architecture
Five-stage pipeline: **Watchlist → Footprint → Risk (scoring + cascade) → Report → Dashboard.**

```
config/prompts.py            # all LLM prompt templates
config/settings.py           # reads .env (pydantic-settings)
config/risk_weights.yaml     # dimension weights (fin 30 / ops 30 / comp 20 / geo 20)
src/models.py                # all Pydantic schemas — note SourceProvenanceAnchor, mathematical_lineage
src/agents/{watchlist,footprint,risk,report}_agent.py
src/data_sources/{yfinance_client,news_client,sec_edgar,wikipedia_client,aggregator}.py
src/risk/scorer.py           # four-dimension scoring engine
src/graph/{supply_chain_graph,cascading_risk}.py   # NetworkX + VaR / HHI / SPOF
src/dashboard/html_generator.py + templates/dashboard.html.j2
scripts/{run_pipeline.py, generate_demo.py, amd_start.py}
data/outputs/<company>_<runid>.html   # generated dashboard artifact
```

## LLM backend toggle (load-bearing)
`LLM_BACKEND` switches `mock | ollama | vllm`. The mock client (`src/llm/mock_client.py`) is fully
functional and lets the entire pipeline run with **no GPU** — the most important portability decision
in the project. **Never break the mock path.** Per-run override: `--backend`.

## Conventions
- Complete, production-quality implementations. No placeholders or stub functions.
- Prefer **targeted `str_replace`** over full-file rewrites.
- **Inspect the live file before editing** — do not trust remembered line numbers or prior context.
- **Audit before declaring done:** after a change, re-read the touched code and trace the data all the
  way through to the rendered dashboard. Self-check; never assume correctness.
- Work **one plan task at a time**. Don't batch unrelated edits.
- After any data-model or scorer change, re-run `python scripts/generate_demo.py` (mock) and open the
  HTML to verify the output actually changed as intended.

## Known gotchas (don't relearn these)
- **pydantic-settings reads `.env` at import time**, before CLI flags apply. Handle backend switching explicitly.
- **vis.js `DataSet.update()` is a silent no-op without an `id`** on the node/edge object — not an error, just nothing happens.
- **Country-code mismatch (active bug):** `COUNTRY_RISK` in `scorer.py` is keyed by ISO-2 (`"CN"`, `"TW"`),
  but the watchlist LLM emits full names (`"China"`, `"Taiwan"`). Every lookup misses → `geo_score` flatlines
  at the `DEFAULT_COUNTRY_RISK` of 40. Geo is 20% of the composite and is currently dead. Normalize to ISO-2.
- **Graph layout fights itself:** `renderLeftToRightNetwork` sets manual `node.x/node.y` and then leaves
  `physics: true`, so the engine drags the hierarchy back into a blob. Use fixed positions or hierarchical
  LR layout with physics disabled for placed nodes.
- **`provenance_anchors` is in the schema but never populated** by the footprint agent. That empty `{}` is
  the central grounding gap the current work addresses.
- `mathematical_lineage` on each node **is** populated and real — it's the seed of the explainability story.
  Grow it, don't replace it.

## Experimental / out of scope
`src/analysis/blast_radius.py` is an experimental what-if simulator that is **not wired into** the pipeline
or dashboard. It is left in place but should not be extended or integrated without explicit intent.