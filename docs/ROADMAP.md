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

## A3 · Populate `provenance_anchors` at fetch time
**Files:** `src/data_sources/{yfinance_client,sec_edgar,news_client}.py`,
`src/agents/footprint_agent.py`, `src/models.py`.
**Note:** `DriverEvidence` (defined in E1) is the canonical provenance type used here.
If A3 is implemented before E1, use a forward-compatible placeholder and migrate in the E1 commit.
**Do:**
- Each data source returns a `DriverEvidence` alongside its data: `label` (short claim string),
  `source_url` (actual SEC filing URL, news article URL, or yfinance provider reference),
  and `retrieved_at`. No separate pass — provenance is captured inline at fetch time.
- Footprint agent collects into `fp.provenance_anchors` keyed by field
  (e.g. `"financials.altman_z"`, `"news.headline_0"`, `"sec.10K"`). Carries through to node `meta`.
- Mock client returns realistic stub anchors so the mock path exercises the full schema.
**Acceptance:** `provenance_anchors` is non-empty on every node with a real data fetch; each
anchor has a non-empty URL; mock path produces stub anchors for every node.

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

# P1 (continued) — Dimension grounding & operational completeness

## E1 · Unified `DriverEvidence` model — consolidates A3's `SourceProvenanceAnchor`
**Files:** `src/models.py`.
**Context:** A3 introduced `SourceProvenanceAnchor` on `FootprintData` as a general fetch-time
provenance carrier. E and F need a typed, scorer-level evidence model. Rather than two parallel
schemas with overlapping fields, **`DriverEvidence` replaces `SourceProvenanceAnchor`** as the
single provenance primitive used everywhere — at fetch time (footprint agent), at score time
(scorer dimensions), and at render time (dashboard). Migrate any existing `SourceProvenanceAnchor`
usages to `DriverEvidence` in the same commit.
```python
class DriverEvidence(BaseModel):
    label: str                    # human-readable driver string rendered in the UI
    source_url: str               # direct link: SEC filing, GDELT article, yfinance ref
    retrieved_at: datetime
    value: Optional[str] = None   # raw value that produced the label
```
`DimensionScore` gains `evidence: list[DriverEvidence] = []` alongside the existing `key_drivers`
list. `key_drivers` is kept for backwards compatibility (plain strings); `evidence` is the
structured, URL-bearing form consumed by the UI.
`FootprintData.provenance_anchors` type changes from `dict[str, SourceProvenanceAnchor]` to
`dict[str, DriverEvidence]` — same dict shape, unified type.
**Acceptance:** `SourceProvenanceAnchor` is removed; `DriverEvidence` is the only provenance
type; all existing tests pass; no other files broken by the rename.

## E2 · Compliance grounding — live SEC EDGAR fetch
**Files:** `src/data_sources/sec_edgar.py`, `src/risk/scorer.py` (`_score_compliance`),
`src/llm/mock_client.py`, `config/prompts.py`.
**Spec:**
- `sec_edgar.py` fetches the entity's most recent filings via the EDGAR full-text search API.
  For each filing found, capture: form type (10-K, 10-Q, 8-K), filing date, and the direct
  EDGAR document URL. Return a list of `DriverEvidence` objects populated inline — do not defer
  provenance to a later pass.
- **Foreign-entity handling (TSMC, Samsung, ASML, Shin-Etsu, Pegatron, Foxconn, SoftBank,
  ARM — non-US-listed):** attempt the lookup; on zero results emit exactly one `DriverEvidence`
  with `label = "No SEC filings — non-US-listed entity"`, `source_url` pointing to the EDGAR
  search URL that was tried, and `retrieved_at` set. Never leave compliance drivers blank for
  these entities.
- `_score_compliance` in `scorer.py` replaces its current hardcoded placeholder logic with
  deterministic scoring driven by the EDGAR results:
  - Recent 10-K present → compliance baseline score reduced (good signal)
  - 8-K filings in last 90 days → flag count drives score upward
  - No filings found → score set to a defined `NO_FILINGS_DEFAULT` constant, not a magic number
  - Each branch appends a `DriverEvidence` to the dimension's `evidence` list
- The LLM compliance narrative (`config/prompts.py`) is rewritten to receive the driver strings
  and evidence labels as structured input, not an empty context. The model narrates *from* the
  drivers; it does not invent them.
- **Mock parity:** `mock_client.py` must return a realistic stub `DriverEvidence` list for
  compliance — at minimum one 10-K entry for US-listed entities and the `no-US-filings` entry
  for foreign ones. The mock path must exercise the full schema end-to-end.
**Acceptance:** `comp_score` varies across entities and is derivable from the evidence list;
every node has at least one `DriverEvidence` in compliance; foreign nodes carry the explicit
no-US-filings entry; compliance narrative references the actual filing data.

## E3 · Compliance evidence rendered in inspector UI
**Files:** `src/dashboard/templates/dashboard.html.j2`, `src/dashboard/html_generator.py`.
**Do:** In the node inspector panel, render the compliance `evidence` list as clickable source
links (filing type + date + link to EDGAR document). This is the visible payoff of E2 — a reader
must be able to click from a compliance score to the filing that drove it.
**Acceptance:** clicking a compliance-flagged node shows at least one linked filing or an
explicit "no filings" note; no compliance score is unattributed.

## F1 · Geopolitical grounding — live GDELT fetch + HHI explanation
**Files:** `src/data_sources/news_client.py` (or a new `src/data_sources/gdelt_client.py` if
cleaner), `src/risk/scorer.py` (`_score_geopolitical`), `src/llm/mock_client.py`,
`config/prompts.py`.
**Spec:**
- Fetch recent GDELT events for the entity's `hq_country` and industry. For each material event
  (conflict, trade restriction, sanctions signal), return a `DriverEvidence` with `label`,
  `source_url` (the GDELT article URL), and `retrieved_at`.
- `_score_geopolitical` replaces its current score-only output with deterministic driver
  construction:
  - The existing `COUNTRY_RISK` index score always emits a `DriverEvidence` explaining the
    baseline: e.g. `"Taiwan country risk index: 45/100 (elevated geopolitical exposure)"`.
  - The existing HHI figure (already computed in `cascading_risk.py`) is currently unexplained
    in the inspector. Pipe it into the geopolitical evidence as a driver:
    `"Portfolio geo-concentration HHI: 4112 — Taiwan accounts for 55.6% of supply spend"`.
  - GDELT events append additional `DriverEvidence` entries.
- The LLM geopolitical narrative receives the driver strings as structured input.
- **Mock parity:** stub `DriverEvidence` list covering: one country-risk baseline entry, one
  HHI entry, and one GDELT headline per entity.
**Acceptance:** `geo_score` is traceable to at least a country-risk baseline driver and an HHI
contribution for every node; the HHI figure visible in the stat card is now also explained in
the per-node inspector; geo narrative references actual evidence.

## F2 · Geopolitical evidence rendered in inspector UI
**Files:** `src/dashboard/templates/dashboard.html.j2`, `src/dashboard/html_generator.py`.
**Do:** same pattern as E3 — render geo `evidence` as clickable links in the inspector panel.
The HHI driver in particular should cross-link to the regional-concentration chart in the
analytics tab, making the stat card and the per-node inspector mutually reinforcing.
**Acceptance:** every node's geo inspector section shows at least the country-risk baseline and
HHI contribution as attributed, linked entries.

## G1 · Operational completeness — always emit at least one `ops_driver`
**Files:** `src/risk/scorer.py` (`_score_operational`).
**Do:** The current scorer emits an empty `ops_drivers` list when no internal record flags are
triggered (e.g. Samsung Electronics, which has no spend-concentration flag and no single-source
flag). A clean bill of health is itself evidence and should be stated:
- If `internal_record` exists and no flags fired: emit `"No operational flags — vendor meets
  all monitored thresholds"` as the driver.
- If `internal_record` is `None`: emit `"No internal vendor record — operational score set to
  default"` as the driver, making the gap explicit rather than silent.
- Any existing flag-based drivers (spend concentration, single-source, BCP absent, audit score)
  are unchanged.
**Acceptance:** `ops_drivers` is non-empty on every scored node; clean nodes carry an explicit
positive confirmation driver; nodes without internal records carry an explicit gap driver.

## G2 · Pre-vetted alternatives seed + LLM ranking into `node.meta.backups`
**Files:** new `data/alternatives_seed.yaml`, `src/agents/risk_agent.py` (or `report_agent.py`
— whichever currently constructs node meta), `config/prompts.py`, `src/llm/mock_client.py`.
**Spec:**
- Create `data/alternatives_seed.yaml` keyed by industry. Must cover all 15 distinct industry
  values present in the current graph — **every industry must have an entry or some nodes stay
  empty**:
  ```
  Semiconductor Fabrication: [Samsung Foundry, GlobalFoundries, Intel Foundry, SMIC]
  Contract Electronics Manufacturing: [Pegatron, Wistron, Compal, Flex Ltd]
  Contract Manufacturing: [Pegatron, Wistron, Compal, Flex Ltd]
  Memory & Display: [Micron, SK Hynix, Kioxia, Japan Display]
  Wireless Chips: [Qualcomm, MediaTek, Marvell, Intel]
  Lithography Equipment: [Nikon, Canon, Ultratech]
  Semiconductor Equipment: [Lam Research, KLA Corporation, Tokyo Electron]
  Silicon Wafers: [Sumco, Siltronic, SK Siltron, GlobalWafers]
  Specialty Glass: [AGC Inc, Nippon Electric Glass, Schott]
  IP Licensing: [MIPS Technologies, Imagination Technologies, Tensilica]
  Telecommunications: [Verizon, T-Mobile, Deutsche Telekom, Vodafone]
  E-Commerce / Cloud: [Microsoft Azure, Google Cloud, Alibaba Cloud]
  Search & Advertising: [Microsoft Bing Ads, Meta Audience Network]
  Investment / Telecommunications: [SoftBank Vision Fund alternatives: KKR, Sequoia]
  Industrial Gases: [Air Liquide, Messer Group, Air Products]
  ```
  These are starting candidates — Claude Code may revise/extend but must not leave any industry
  unkeyed.
- In the pipeline (risk or report agent), after scoring: for each node, load candidates from the
  seed by `entity.industry`, pass them plus the entity's risk drivers to the LLM with a prompt
  that asks it to rank the candidates and justify each in 1 sentence given the specific risk
  context. Store the ranked, justified list into `node.meta.backups`.
- **Mock parity:** mock client returns a realistic stub ranked list for any alternatives prompt.
**Acceptance:** `node.meta.backups` is non-empty on every node that has a seed entry for its
industry; each backup carries a name and a 1-sentence justification; the inspector's
alternatives section is no longer blank.

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

1. **Session 1 — Foundation (A1 done ✅):** A2 → A3 → A4. Country normalization, provenance
   anchors, dashboard honesty pass. Single HTML output at the end.
2. **Session 2 — Math + exposure:** B1 → B3 → B4 → B2. Risk-scaled VaR, portfolio aggregation,
   impact×probability matrix, spend-sized nodes. Single HTML output at the end.
3. **Session 3 — Compliance grounding:** E1 → E2 → E3. `DriverEvidence` model, SEC EDGAR
   fetch, compliance rendered in inspector. Single HTML output at the end.
4. **Session 4 — Geo grounding + ops completeness:** F1 → F2 → G1 → G2. GDELT fetch, geo
   evidence in inspector, ops driver completeness, pre-vetted alternatives seeded and ranked.
   Single HTML output at the end.
5. **Session 5 — Explainability UI + polish:** C1 → C2 → C5. Full dimension breakdown in
   inspector, all provenance links rendered, freshness panel.
6. **Session 6 — Stretch:** A5 → C3 → C4 → D1 → D2 as time allows.

If time is short: **A2–A4 + B1 + E1–E2 + G1** deliver grounded, differentiated scores across
all four dimensions with visible evidence — the highest-leverage subset.
