---
name: mock-footprint-path
description: How mock mode produces footprint data + provenance without hitting the network
metadata:
  type: project
---

In mock mode the LLM is mocked but the data sources (yfinance/SEC/GDELT) are NOT — they
would still hit the live network unless explicitly short-circuited. As of A3, `llm_backend`
is threaded `footprint_node → aggregate_all_entities → aggregate_entity_footprint`, and when
`llm_backend == "mock"` the aggregator returns `_build_mock_footprint(entity)` instead of
fetching. That builder pulls deterministic per-ticker stubs from `_MOCK_FINANCIALS` (keyed by
ISO ticker, upper-cased) in `src/data_sources/aggregator.py` and synthesizes a 10-K filing +
one news headline + matching `SourceProvenanceAnchor`s.

**Why:** the offline demo must exercise the full grounding schema without network flakiness.
If you add a tier-1/tier-2 entity to `MOCK_WATCHLISTS`, also add its ticker to `_MOCK_FINANCIALS`
or it will only get a news anchor (e.g. Pegatron's resolved ticker `4938.TW` isn't in the dict,
so it shows 1 anchor instead of 4).

Real-fetch path collects the same anchors via `_collect_provenance(...)`. Both paths set
`fp.provenance_anchors`, which flows into node `meta` in `html_generator._build_vis_nodes`.
The target company (depth 0) is never fetched or scored, so it legitimately has 0 anchors and
its dashboard `geo_score`/dimension scores are html_generator fallbacks (50.0), not real scores.
See [[country-normalization]].
