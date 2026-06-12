---
name: project-overview
description: What sentry-vendor-risk-intel is, the dev/deploy workflow, and the AMD hackathon goal
metadata:
  type: project
---

Vendor risk management & intelligence product aimed at the procurement function. Built for the AMD Developer Cloud hackathon submission: runs on AMD ROCm via vLLM with an open-source LLM (Qwen2.5), with ~25GB persistent storage budget.

**Workflow:** Vaibhav develops + mock-tests locally on Mac, pushes to this git repo, then clones on AMD hardware and runs with the `vllm` backend. So mock parity and determinism matter.

5-stage LangGraph pipeline: watchlist → footprint → risk_scoring → report → dashboard. LLM-agnostic factory (mock/ollama/vllm) — only the LLM swaps. Output is a single self-contained HTML dashboard.

Note: "mock" backend only mocks the LLM; data-source fetches (yfinance/GDELT/SEC/wiki) still hit live networks, making mock runs slow (~60s) and flaky. See [[known-issues-and-gaps]].
