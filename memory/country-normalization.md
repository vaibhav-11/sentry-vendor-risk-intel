---
name: country-normalization
description: hq_country is normalized to ISO-2 via normalize_country() before scoring
metadata:
  type: project
---

`normalize_country(value)` lives in `src/risk/scorer.py` (it owns `COUNTRY_RISK`). It maps full
names/aliases ("China"→"CN", "UK"→"GB") to ISO-2 and passes valid ISO-2 through. The alias lookup
runs BEFORE the 2-letter passthrough so "UK"→"GB" isn't short-circuited to "UK".

Applied in `watchlist_agent._parse_entities` (entity construction, before the id-slug suffix) and
defensively in `_score_geopolitical`. The target entity is constructed with `hq_country="US"` to
avoid an empty `""` bucket in the geo HHI concentration chart.

This revived the geo dimension (was flatlined at DEFAULT_COUNTRY_RISK=40 because the LLM emitted
full names that missed the ISO-2-keyed COUNTRY_RISK). Verified live: TW=60 (45+15 cross-strait),
KR=33 (25+8 semi export-control), US=10, NL=10, GB=12. See [[mock-footprint-path]].
