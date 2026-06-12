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

**NEXT: A2** — normalize hq_country to ISO-2 everywhere (revive geo for the LLM path; geo already works for ISO-2 mock data) + set target's country (Apple = US) to kill the blank HHI bucket.

See [[known-issues-and-gaps]] and [[project-overview]].
