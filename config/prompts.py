"""
All LLM prompt templates centralised here.
Edit prompts without touching agent logic.
"""

SYSTEM_RISK_ANALYST = """You are a senior enterprise risk analyst specialising in third-party and
vendor risk management. You have deep expertise in supply chain analysis, financial risk assessment,
and geopolitical risk. You respond with precise, structured, and actionable analysis.
Always ground your analysis in the data provided. Do not hallucinate metrics."""

# ── Watchlist Generation ──────────────────────────────────────────────────────

WATCHLIST_PROMPT = """You are building a vendor risk watchlist for: {company_name}

Generate a structured JSON list of the company's key third-party relationships across
{max_depth} levels. Include: direct suppliers, major customers, technology partners,
logistics providers, and financial counterparties.

Rules:
- Maximum {max_children} relationships per entity
- Focus on relationships material to business continuity
- Include the relationship type and estimated importance (1-10)

Return ONLY valid JSON matching this exact schema:
{{
  "target": "{company_name}",
  "entities": [
    {{
      "name": "string",
      "ticker": "string or null",
      "entity_type": "supplier|customer|partner|logistics|financial",
      "relationship_to_parent": "string (parent entity name)",
      "depth_level": 1,
      "importance_score": 8,
      "industry": "string",
      "hq_country": "string (ISO-2 code)"
    }}
  ]
}}"""

# ── Entity Risk Narrative ─────────────────────────────────────────────────────

ENTITY_NARRATIVE_PROMPT = """Analyse the following third-party entity for vendor risk:

Entity: {entity_name} ({entity_type})
Relationship: {relationship_description}
Industry: {industry} | HQ Country: {hq_country}

Financial Data:
{financial_summary}

Recent News (last 30 days):
{news_summary}

Internal Vendor Data:
{internal_summary}

Risk Score: {risk_score}/100 (Financial: {fin_score}, Operational: {ops_score},
Compliance: {comp_score}, Geopolitical: {geo_score})

Write a concise 3-paragraph risk narrative (max 200 words total):
1. Current risk status and primary drivers
2. Key vulnerabilities and early warning signals observed
3. Recommended mitigation actions

Be specific. Reference actual data points provided above."""

# ── Alert Generation ──────────────────────────────────────────────────────────

ALERT_PROMPT = """Based on the risk analysis below, generate a structured risk alert.

Entity: {entity_name}
Risk Score: {risk_score}/100 (threshold breached: {threshold_name})
Key Risk Drivers: {risk_drivers}
Recent Signal: {triggering_signal}

Return ONLY valid JSON:
{{
  "alert_title": "concise title under 10 words",
  "severity": "critical|high|medium|low",
  "summary": "2-sentence summary of the risk",
  "recommended_action": "single most important action to take",
  "escalate_to": "CISO|CPO|CFO|Legal|Procurement",
  "time_sensitivity": "immediate|24h|1-week|monitoring"
}}"""

# ── Executive Report ──────────────────────────────────────────────────────────

EXECUTIVE_REPORT_PROMPT = """You are preparing an executive risk briefing for the Chief Procurement
Officer and Chief Risk Officer.

Target Company: {target_company}
Analysis Date: {analysis_date}
Total Entities Monitored: {total_entities}
Critical Risk Entities: {critical_count}
High Risk Entities: {high_count}

Top 5 Entities by Risk Score:
{top_entities_summary}

Key Supply Chain Vulnerabilities Identified:
{vulnerabilities_summary}

Cascade Risk Analysis:
{cascade_summary}

Write a professional executive briefing report with these sections:
1. Executive Summary (3-4 sentences, board-level language)
2. Critical Risk Findings (bullet points, max 5)
3. Supply Chain Concentration Risks
4. Recommended Immediate Actions (numbered, max 5)
5. 30-Day Monitoring Priorities

Format in clean HTML using <h2>, <p>, <ul>, <li>, <strong> tags only.
Total length: 400-600 words."""

# ── Cascade Risk Summary ──────────────────────────────────────────────────────

CASCADE_SUMMARY_PROMPT = """Analyse the following supply chain cascade risk data:

Target Company: {target_company}
High-Centrality Nodes: {central_nodes}
Single Points of Failure: {spof_nodes}
Estimated blast radius if top node fails: {blast_radius}%

Write 2 paragraphs explaining:
1. The nature and severity of cascade risks in this supply chain
2. Which specific failure scenarios pose the greatest systemic threat

Be concrete and reference the specific entities named above."""
