---
name: remediation-progress
description: Status of the grounding/explainability remediation plan (docs/REMEDIATION_PLAN.md)
metadata:
  type: project
---

Working through docs/REMEDIATION_PLAN.md one task at a time. CLAUDE.md + the plan are the source of truth. Demo target is **Apple Inc** (do NOT switch to AMD). Deadline ~Monday.

**A1 — DONE (2026-06-12).** Chose the *hybrid* approach over the plan's curated seed (user's call): LLM still generates the network; a new deterministic ticker resolver grounds it.
- `src/data_sources/ticker_resolver.py` (new): validate-first — keep a real LLM ticker, else Yahoo-search the company name and pick the primary listing (exchange-rank to avoid BDRs like TSMC34.SA). Caches to data/cache/ticker_resolutions.json (deterministic after first run). Unresolved → kept, marked is_public=False (honest, not dropped).
- `Entity` gained `annual_spend_usd` (from internal vendor_registry by name) and `is_public`.
- `watchlist_agent` runs resolver + spend attach after parsing; `build_graph` sets annual_spend_usd node attr (only when known, so cascade's importance fallback still works).
- Result: 16/16 entities now have REAL financials (was 1/8). Tickers 100%, correct primary symbols.
- Side effect to address later: with real financials these healthy mega-caps all score Low → 0 alerts. Scoring calibration / VaR (B1) / tier-2 expansion (A5) will restore meaningful risk signal.

**A2 — DONE (2026-06-13).** `normalize_country()` in scorer.py + applied in watchlist_agent; target set to US. Geo dimension live and varies by country. See [[country-normalization]].

**A3 — DONE (2026-06-13).** `provenance_anchors` populated by each data source (`build_*_anchor` helpers) and collected in the aggregator via `_collect_provenance`; threads `llm_backend` so mock mode returns stub data+anchors. Flows into node meta + rendered as clickable cited sources in the inspector. See [[mock-footprint-path]].

**A4 — DONE (2026-06-13).** Dashboard honesty pass: removed fake 4-step refresh animation → static data-freshness panel (per-source `retrieved_at` + per-node composite deltas cached to `data/cache/<company>_last_run.json`, "Baseline" on first run); Data Room relabeled ILLUSTRATIVE with RAG badges removed and placeholder docs mapped to real node IDs; playbooks generated from highest-composite + single-source entities. Removed `uploaded_documents` field from PipelineState.

**NEXT: P1 — B1** (risk-scaled VaR) → B3 (portfolio aggregation) per session sequence, or C2 (provenance already half-rendered in inspector).

See [[known-issues-and-gaps]], [[project-overview]], [[country-normalization]], [[mock-footprint-path]].
