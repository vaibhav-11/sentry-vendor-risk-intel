---
name: known-issues-and-gaps
description: Concrete bugs, dead code, and feature gaps found in the first repo study (2026-06-12)
metadata:
  type: project
---

Found during initial study (2026-06-12), unfixed as of then:

1. **LLM output invisible in dashboard** (biggest): per-entity narratives (node.meta.narrative), alerts (alerts_json), and executive report (report_html) are all generated but the dashboard template never renders them. README advertises Alerts + Executive Report tabs that don't exist.
2. `src/risk/scorer.py`: `_score_financial` defined 6x (lines ~43-241); only last wins, rest is dead code. Also large commented-out block in `src/graph/cascading_risk.py:89-136`.
3. Node inspector "Contingency Options" always empty: template reads node.meta.backups which is never set; NodeMetrics.alternative_suppliers is computed then discarded.
4. VaR uses synthetic spend (`importance * $1.85M`) instead of real `internal_record.annual_spend_usd` from the vendor registry; graph nodes never get annual_spend_usd attached.
5. provenance_anchors always empty; uploaded_documents hardcoded in PipelineState default — Data Room + provenance UI are facades.
6. Doc drift: README/settings default to Qwen2.5-14B but actual AMD launch cmd uses Qwen2.5-3B at 0.25 GPU util.
7. vLLM generate_json only sets a system instruction (no guided/json mode) — risk of malformed JSON from small models.

See [[project-overview]].
