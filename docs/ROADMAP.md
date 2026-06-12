# Engineering Roadmap — Grounding & Explainability

> Place at `docs/ROADMAP.md`. Execute **one task at a time**, top to bottom within each
> priority tier. After each task: run `python scripts/generate_demo.py`, open the HTML,
> and confirm the output changed as intended before moving on.

## Problem statement
The dashboard runs end-to-end but is **data-starved**. Most entities reach the scorer with
`financials = None` because the watchlist LLM emitted `ticker: null` for them, so every
financial/SEC/news fetch no-ops and scores collapse to defaults (`fin 50, comp 40, geo 40`,
empty drivers). The narratives hedge ("limited financial data and recent news") because the
model is honestly describing an empty plate. Compounding this: the geo dimension is silently
dead (ISO-2 key mismatch), the graph is a flat one-tier supplier-only star, VaR is mis-defined
(direct spend is added to rather than scaled by risk), node size ignores spend,
`provenance_anchors` is never populated, and several dashboard tabs are decorative mocks
presented as live features. **Fix the data foundation first; most cosmetic symptoms resolve
downstream.**

## Definition of done
- Apple centered; **real tier-1 suppliers to the left, real customers/channel partners to the
  right**, with tier-2 expansion on the most material nodes.
- Each entity carries **real** financial data (real ticker → real yfinance/SEC/news) wherever
  the company is public; private entities are explicitly labeled as such, not silently defaulted.
- Clicking a node shows the **four dimension scores with their drivers and data gaps**, plus
  **clickable source links** (SEC filing URL, news URL, financial data provider). Visible provenance.
- VaR is **risk-scaled**: a healthy vendor's VaR is a small fraction of spend; a distressed
  single-source vendor's approaches full exposure. `mathematical_lineage` shows the breakdown.
- Node size is **proportional to contract spend**. No tab presents a hardcoded mock as live.

---

# P0 — Foundation (credibility-critical)

## ✅ A1 · Curated tier-1 seed with real tickers (both directions)
**Status: Complete.**
Real tier-1 Apple network with verified tickers and ISO-2 countries is seeded in
`data/seed/` (or `mock_client.py` `MOCK_WATCHLISTS`), covering suppliers (upstream/left),
customers, and partners (downstream/right) with tier-2 children under the most material nodes.
`watchlist_agent.py` loads the seed when a seed exists for the target; LLM path remains as
fallback for non-seeded targets. `vendor_registry.json` carries real spend figures,
single-source flags, and audit scores.

## A2 · Normalize country to ISO-2 everywhere (revive geo score)
**Files:** `src/risk/scorer.py`, `src/agents/watchlist_agent.py`, anywhere `hq_country` is
set or read.
**Do:**
- Add a `normalize_country(value) -> str` helper that maps full country names and common aliases
  to ISO-2 (`"China" → "CN"`, `"Taiwan" → "TW"`, `"South Korea" → "KR"`, etc.) and passes
  through values that are already valid ISO-2. Apply at entity construction so `hq_country` is
  always ISO-2 by the time it reaches the scorer.
- Fix the **empty-country target bug**: the target node (Apple) currently has `hq_country = ""`
  which creates a blank bucket in the geo HHI chart. Set it to `"US"`.
**Why this matters:** `COUNTRY_RISK` in `scorer.py` is keyed by ISO-2 but the LLM emits full
names, so every lookup misses and `geo_score` flatlines at `DEFAULT_COUNTRY_RISK = 40`. The
geopolitical dimension (20% of composite) is currently dead. This is a one-function fix with
outsized scoring impact.
**Acceptance:** `geo_score` varies meaningfully by country (CN ~65, TW ~45, US ~10); the
regional-concentration chart has no blank bucket; geo contributes real differentiation to
composite scores.

## A3 · Populate `provenance_anchors` (core grounding fix)
**Files:** `src/data_sources/{yfinance_client,sec_edgar,news_client}.py`,
`src/agents/footprint_agent.py`, `src/models.py` (confirm `SourceProvenanceAnchor` fields),
`src/dashboard/html_generator.py` + template.
**Do:**
- Every data source, when it returns a value, also returns a `SourceProvenanceAnchor` with:
  `source_name`, `url` (the actual SEC filing/document URL from EDGAR, the news article URL, or
  the yfinance provider reference), `retrieved_at` (datetime), and a short `claim` string
  describing what the anchor supports.
- In the footprint agent, collect anchors into `fp.provenance_anchors` keyed by field
  (e.g. `"financials.altman_z"`, `"news.headline_0"`, `"sec.10K"`). Carry them through into
  node `meta` in the dashboard payload.
**Acceptance:** `provenance_anchors` is non-empty on every node that had any real data fetch;
each anchor carries a non-empty, working URL.

## A4 · Dashboard honesty pass
**Files:** `src/dashboard/templates/dashboard.html.j2`, `src/dashboard/html_generator.py`.
**Do:**
- **Data Room:** `VAULT_DOCUMENTS` references node IDs that don't exist in the live graph
  (`tsmc-tw`, `asml-nl`, `foxconn-hi`). Either drive the list from real node metadata, or
  clearly label the tab "Illustrative — document ingestion roadmap." Remove "Vectorized & Synced
  via RAG" badges from data that isn't RAG-processed.
- **Refresh button:** remove the fake 4-step animation entirely. Replace per C5 below.
- **Playbooks:** replace the two hardcoded generic emails with playbooks generated from the
  actual highest-risk entities in this run (highest composite score / single-source flags).
**Acceptance:** every interactive element does what it claims or is explicitly labeled
illustrative; no element simulates work that isn't happening.

---

# P1 — Substance & correctness

## B1 · Redefine Value-at-Risk (risk-scaled)
**Files:** `src/graph/cascading_risk.py` (VaR computation); mirror the formula in the
manual-add JS block in the dashboard template.
**Formula:**
```
p_disruption = composite_score / 100
severity     = clamp(0.5 + 0.3 * single_source + 0.2 * no_alternate_available, 0, 1)
               # defaults to 0.6 when no internal record exists
direct_VaR   = direct_spend * p_disruption * severity      # always ≤ direct_spend
cascade_VaR  = Σ (child_direct_spend * child_composite/100 * dependency_strength)
VaR_total    = direct_VaR + cascade_VaR
```
**Invariants:** `direct_VaR ≤ direct_spend` always. `VaR_total` may exceed `direct_spend` only
via genuine cascade. `mathematical_lineage` must record `direct_VaR` and `cascade_VaR`
separately so the decomposition is visible.
**Acceptance:** a low-composite vendor's VaR is a small fraction of its spend; a high-composite
single-source vendor's direct VaR approaches its spend; two equal-spend vendors with different
risk scores produce clearly different VaR values.

## B2 · Node size proportional to contract spend
**Files:** `src/dashboard/html_generator.py` (where node `size` is assigned).
**Do:** `size = clamp(min_size + k * sqrt(annual_spend_usd), min_size, max_size)`. Square-root
scaling prevents large spends from visually dominating. Target node remains largest by design.
Surface the mapping in the node tooltip.
**Acceptance:** node radius visibly tracks spend; tooltip explains the size encoding.

## B3 · Fix portfolio aggregation
**Files:** `src/graph/cascading_risk.py`.
**Do:** Total Portfolio Value should sum vendor spend only, excluding the target entity's own
`direct_spend` (Apple's spend is currently included, inflating the header figure). Re-derive
total VaR as the sum of per-node `VaR_total` from the B1 model.
**Acceptance:** portfolio value = Σ vendor spend, no target included; total VaR is consistent
with the sum of per-node values.

## B4 · Exposure-weighting and dependence ranking
**Why:** dependence should be driven by contract value, not vendor brand size. Keep two axes
separate — contract value is **impact**, vendor health is **probability** — or the largest
healthy supplier perversely reads as the highest risk.
**Files:** `src/graph/cascading_risk.py`, `src/dashboard/html_generator.py` + template.
**Do:**
- `annual_spend_usd` is the impact axis; `composite_score / 100` is the probability axis. B1's
  VaR already multiplies them — this task makes the decomposition *visible*.
- Analytics table: add sortable `Contract Value` and `Dependence %` columns so a high-contract
  lesser-known vendor can rank above a low-contract major brand.
- Add an **impact × probability scatter**: x = vendor composite risk, y = contract spend, bubble
  size = VaR. This is the legible "why vendor X outranks Samsung" picture.
- Verify that total portfolio VaR shifts more when a high-contract vendor's score changes than
  when a low-contract vendor's does (falls out of B1 — confirm and surface).
**Acceptance:** table is rankable by spend independently of composite score; the scatter plot
separates impact from probability; sensitivity is demonstrably spend-proportional.

## C1 · Inspector shows the full dimension breakdown
**Files:** template inspector panel + `html_generator.py`.
**Do:** on node click, render all four dimensions with their `key_drivers` and `data_gaps` (the
scorer already produces these), the weight each contributes to the composite, and the resulting
composite score. Show "no data available" explicitly where `data_gaps` is non-empty rather than
silently showing a default.
**Acceptance:** every node's inspector explains *why* its composite is what it is, dimension by
dimension, with gaps surfaced honestly.

## C2 · Render provenance in the UI
**Files:** template (inspector panel + risk cards). Depends on A3.
**Do:** surface the A3 provenance anchors as clickable source links next to the data they
support — SEC filing, news article, financial data provider. Each cited figure should be
followable to its source.
**Acceptance:** a reader can click from any cited score driver to the underlying source document
or data feed.

## C5 · Data-freshness & change panel (replaces the refresh button)
**Files:** `src/dashboard/templates/dashboard.html.j2`, `src/dashboard/html_generator.py`.
Depends on A3 (`retrieved_at` on provenance anchors).
**Why:** a static self-contained HTML file has no server to re-fetch against, so a live refresh
button is not implementable without a backend. The real value behind that concept — "is this
current, and what changed?" — is deliverable statically and reinforces grounding.
**Do:**
- Replace the refresh button with a **data-freshness panel** showing per-source retrieval
  timestamps pulled from provenance anchor `retrieved_at` fields (e.g. "10-K retrieved 3 days
  ago", "financials as of [date]").
- Show **change-since-last-run** using the existing `score_delta_7d` field on `RiskScore`: cache
  the previous run's per-node composites to `data/cache/<company>_last_run.json`, diff on the
  next run, and display movement per node ("TSMC composite ▲ 4.2 since last run"). Absent a
  prior run, display "Baseline — no prior run cached."
**Acceptance:** the panel shows real retrieval timestamps per source and real deltas versus the
last cached run; nothing on the dashboard simulates work that is not happening.

## D1 · README rewrite
**Files:** `README.md`.
**Do:** disclose curated seed data vs. LLM-generated content (architecture transparency).
Update the Mermaid diagram to reflect the grounded pipeline. Write the project framing around
the core engineering values (grounding, explainability, right-tool-per-stage) and include a
learnings and future-work section.
**Acceptance:** disclosure is explicit; README reads as a portfolio project, not just setup
notes; the architecture described matches the code.

---

# P2 — Stretch goals

## A5 · LLM tier-2 expansion
**Files:** `src/agents/watchlist_agent.py`, `config/prompts.py`.
**Do:** a recursive second pass that expands the most material tier-1 nodes into tier-2 (e.g.
TSMC → ASML, Applied Materials, Lam Research). This is where the agentic story lives: the
system reasons about which nodes are material enough to expand, then autonomously maps their
upstream dependencies. Only pursue after P0/P1 are solid.
**Acceptance:** ≥2 tier-1 nodes have real tier-2 children; SPOF detection finds non-trivial
bottleneck nodes; the graph is visibly multi-layered.

## C3 · Working stat cards (filtered views)
**Do:** top-of-page cards carry context through to the tab they open — e.g. "Portfolio VaR"
lands on the analytics tab pre-sorted by VaR; "Supply Chain Nodes" focuses the graph on the
selected tier. No dead or contextless clicks.

## C4 · Lock the graph layout
**Do:** switch to vis.js hierarchical LR layout with physics disabled for placed nodes. Apple
center, suppliers left, customers/partners right, tiers separated by depth level. The current
approach sets manual `x/y` then leaves `physics: true`, which causes the engine to collapse the
hierarchy back into a blob.

## D2 · End-to-end vLLM run + demo recording
**Do:** full pipeline run on the ROCm/vLLM backend with the grounded pipeline. Capture recorded
demo video and slides.

---

## Out of scope (do not start without explicit re-scoping)

- **Live footprint refresh via a backend.** True on-demand re-fetch and recompute requires a
  service layer (e.g. FastAPI) that converts the project from a portable self-contained artifact
  into a running web service, sacrificing the no-dependencies property and introducing
  live-network fragility. C5 delivers the real value without this cost. Revisit only as a
  deliberate separately-scoped track.
- **Full document-discovery RAG** (extracting new supply-chain entities from arbitrary uploaded
  documents). Open-ended entity discovery is a separate effort from the bounded single-node
  contract upload described in the Data Room scope.

---

## Session sequence

1. **Session 1 — Foundation (A1 done ✅):** Continue with A2. Get the geo dimension live and
   country normalization clean before anything else touches scores.
2. **Session 2 — Grounding:** A3 → C2. Provenance recorded at fetch time and rendered in the UI.
3. **Session 3 — Honesty + math:** A4 → B1 → B3. Remove theater, fix VaR, fix aggregation.
4. **Session 4 — Explainability + exposure:** C1 → B4 → C5 → B2. Dimension breakdown, impact
   matrix, freshness panel, spend-sized nodes.
5. **Session 5 — Stretch:** A5 → C3 → C4 → D1 → D2 as time allows.

If time is short: **P0 (A2–A4) + B1 + D1** alone move the two largest quality levers.