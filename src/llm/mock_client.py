"""
Mock LLM client for local development and testing.
Returns realistic pre-templated responses — no model, no GPU.
The full pipeline runs end-to-end in mock mode.
"""

import json
import asyncio
import logging
from src.llm.interface import BaseLLMClient

logger = logging.getLogger(__name__)

# ── Pre-built watchlists for known demo companies ─────────────────────────────

MOCK_WATCHLISTS: dict[str, list[dict]] = {
    "apple inc": [
        # Level 1 — Suppliers
        {"name": "TSMC", "ticker": "TSM", "entity_type": "supplier",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 10, "industry": "Semiconductor Fabrication",
         "hq_country": "TW"},
        {"name": "Foxconn", "ticker": "HNHPF", "entity_type": "supplier",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 9, "industry": "Contract Electronics Manufacturing",
         "hq_country": "TW"},
        {"name": "Samsung Electronics", "ticker": "SSNLF", "entity_type": "supplier",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 8, "industry": "Memory & Display",
         "hq_country": "KR"},
        {"name": "Broadcom", "ticker": "AVGO", "entity_type": "supplier",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 8, "industry": "Wireless Chips",
         "hq_country": "US"},
        {"name": "Corning", "ticker": "GLW", "entity_type": "supplier",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 7, "industry": "Specialty Glass",
         "hq_country": "US"},
        # Level 1 — Customers
        {"name": "AT&T", "ticker": "T", "entity_type": "customer",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 7, "industry": "Telecommunications",
         "hq_country": "US"},
        {"name": "Amazon", "ticker": "AMZN", "entity_type": "customer",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 7, "industry": "E-Commerce / Cloud",
         "hq_country": "US"},
        # Level 1 — Partners
        {"name": "Alphabet", "ticker": "GOOGL", "entity_type": "partner",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 9, "industry": "Search & Advertising",
         "hq_country": "US"},
        {"name": "ARM Holdings", "ticker": "ARM", "entity_type": "partner",
         "relationship_to_parent": "Apple Inc", "depth_level": 1,
         "importance_score": 9, "industry": "IP Licensing",
         "hq_country": "GB"},
        # Level 2 — TSMC suppliers
        {"name": "ASML", "ticker": "ASML", "entity_type": "supplier",
         "relationship_to_parent": "TSMC", "depth_level": 2,
         "importance_score": 9, "industry": "Lithography Equipment",
         "hq_country": "NL"},
        {"name": "Applied Materials", "ticker": "AMAT", "entity_type": "supplier",
         "relationship_to_parent": "TSMC", "depth_level": 2,
         "importance_score": 8, "industry": "Semiconductor Equipment",
         "hq_country": "US"},
        {"name": "Shin-Etsu Chemical", "ticker": "SHECY", "entity_type": "supplier",
         "relationship_to_parent": "TSMC", "depth_level": 2,
         "importance_score": 7, "industry": "Silicon Wafers",
         "hq_country": "JP"},
        # Level 2 — Foxconn suppliers
        {"name": "Pegatron", "ticker": "4938.TW", "entity_type": "supplier",
         "relationship_to_parent": "Foxconn", "depth_level": 2,
         "importance_score": 6, "industry": "Contract Manufacturing",
         "hq_country": "TW"},
        # Level 2 — ARM suppliers
        {"name": "SoftBank Group", "ticker": "SFTBY", "entity_type": "partner",
         "relationship_to_parent": "ARM Holdings", "depth_level": 2,
         "importance_score": 7, "industry": "Investment / Telecommunications",
         "hq_country": "JP"},
        # Level 3
        {"name": "Air Products & Chemicals", "ticker": "APD", "entity_type": "supplier",
         "relationship_to_parent": "ASML", "depth_level": 3,
         "importance_score": 5, "industry": "Industrial Gases",
         "hq_country": "US"},
        {"name": "Linde", "ticker": "LIN", "entity_type": "supplier",
         "relationship_to_parent": "Applied Materials", "depth_level": 3,
         "importance_score": 5, "industry": "Industrial Gases",
         "hq_country": "IE"},
    ],
    "microsoft": [
        {"name": "Intel", "ticker": "INTC", "entity_type": "supplier",
         "relationship_to_parent": "Microsoft", "depth_level": 1,
         "importance_score": 8, "industry": "Processors",
         "hq_country": "US"},
        {"name": "AMD", "ticker": "AMD", "entity_type": "supplier",
         "relationship_to_parent": "Microsoft", "depth_level": 1,
         "importance_score": 8, "industry": "Processors / GPUs",
         "hq_country": "US"},
        {"name": "Nvidia", "ticker": "NVDA", "entity_type": "supplier",
         "relationship_to_parent": "Microsoft", "depth_level": 1,
         "importance_score": 9, "industry": "AI GPUs",
         "hq_country": "US"},
        {"name": "OpenAI", "ticker": None, "entity_type": "partner",
         "relationship_to_parent": "Microsoft", "depth_level": 1,
         "importance_score": 9, "industry": "AI Research",
         "hq_country": "US"},
        {"name": "SAP", "ticker": "SAP", "entity_type": "partner",
         "relationship_to_parent": "Microsoft", "depth_level": 1,
         "importance_score": 7, "industry": "Enterprise Software",
         "hq_country": "DE"},
    ],
}

# ── Pre-built tier-2 upstream expansions (A5) ─────────────────────────────────
# Keyed by lowercased tier-1 parent name. The mock returns these when the
# watchlist agent issues a TIER2_EXPANSION_PROMPT, so the mock path exercises the
# same seed-tier-1 + LLM-tier-2 flow the vLLM backend uses.

MOCK_TIER2: dict[str, list[dict]] = {
    "tsmc": [
        {"name": "ASML", "ticker": "ASML", "industry": "Lithography Equipment",
         "importance_score": 9, "hq_country": "NL"},
        {"name": "Applied Materials", "ticker": "AMAT", "industry": "Semiconductor Equipment",
         "importance_score": 8, "hq_country": "US"},
        {"name": "Shin-Etsu Chemical", "ticker": "SHECY", "industry": "Silicon Wafers",
         "importance_score": 7, "hq_country": "JP"},
    ],
    "foxconn": [
        {"name": "Pegatron", "ticker": "4938.TW", "industry": "Contract Manufacturing",
         "importance_score": 6, "hq_country": "TW"},
        {"name": "Wistron", "ticker": "3231.TW", "industry": "Contract Manufacturing",
         "importance_score": 6, "hq_country": "TW"},
        {"name": "Compal Electronics", "ticker": "2324.TW", "industry": "Contract Manufacturing",
         "importance_score": 5, "hq_country": "TW"},
    ],
    "alphabet": [
        {"name": "Arista Networks", "ticker": "ANET", "industry": "Networking Hardware",
         "importance_score": 7, "hq_country": "US"},
        {"name": "Celestica", "ticker": "CLS", "industry": "Hardware Manufacturing",
         "importance_score": 6, "hq_country": "CA"},
        {"name": "Quanta Computer", "ticker": "2382.TW", "industry": "Server Manufacturing",
         "importance_score": 6, "hq_country": "TW"},
    ],
    "arm holdings": [
        {"name": "Synopsys", "ticker": "SNPS", "industry": "EDA Software",
         "importance_score": 8, "hq_country": "US"},
        {"name": "Cadence Design Systems", "ticker": "CDNS", "industry": "EDA Software",
         "importance_score": 7, "hq_country": "US"},
        {"name": "SoftBank Group", "ticker": "SFTBY", "industry": "Investment / Telecommunications",
         "importance_score": 7, "hq_country": "JP"},
    ],
    "samsung electronics": [
        {"name": "Tokyo Electron", "ticker": "TOELY", "industry": "Semiconductor Equipment",
         "importance_score": 8, "hq_country": "JP"},
        {"name": "Sumco", "ticker": "3436.T", "industry": "Silicon Wafers",
         "importance_score": 7, "hq_country": "JP"},
        {"name": "Lam Research", "ticker": "LRCX", "industry": "Semiconductor Equipment",
         "importance_score": 7, "hq_country": "US"},
    ],
}


def _mock_tier2_json(prompt: str) -> str:
    """
    Mock TIER2_EXPANSION_PROMPT response (A5). Detects the tier-1 parent from the
    prompt's 'Tier-1 entity:' line and returns 3 plausible upstream tier-2 suppliers
    with the parent wired as relationship_to_parent. Falls back to a generic trio.
    """
    parent_name = "Unknown Parent"
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("tier-1 entity:"):
            parent_name = stripped.split(":", 1)[1].strip()
            break

    stubs = MOCK_TIER2.get(parent_name.lower())
    if stubs is None:
        stubs = [
            {"name": f"{parent_name} Components Co", "ticker": None,
             "industry": "Industrial Components", "importance_score": 6, "hq_country": "US"},
            {"name": f"{parent_name} Materials Ltd", "ticker": None,
             "industry": "Raw Materials", "importance_score": 5, "hq_country": "DE"},
            {"name": f"{parent_name} Logistics Partners", "ticker": None,
             "industry": "Freight & Logistics", "importance_score": 5, "hq_country": "SG"},
        ]

    entities = [
        {
            "name": s["name"],
            "ticker": s.get("ticker"),
            "entity_type": "supplier",
            "relationship_to_parent": parent_name,
            "depth_level": 2,
            "importance_score": s["importance_score"],
            "industry": s["industry"],
            "hq_country": s["hq_country"],
        }
        for s in stubs
    ]
    return json.dumps({"entities": entities}, indent=2)


# ── Mock narrative templates ───────────────────────────────────────────────────

def _parse_driver_block(prompt: str, header_prefix: str) -> list[str]:
    """
    Extract the bullet lines that follow a labelled driver block in the narrative
    prompt (e.g. "Financial Drivers (...):" then "- driver one"). Stops at the
    first blank line after the block begins. Returns the driver strings (no "- ").
    """
    drivers: list[str] = []
    in_block = False
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith(header_prefix.lower()):
            in_block = True
            continue
        if in_block:
            if stripped.startswith("-"):
                drivers.append(stripped.lstrip("- ").strip())
            elif stripped == "":
                break
    # Drop only the explicit "No <dimension> drivers recorded" placeholder lines
    # (keep genuine drivers that happen to start with "No", e.g. "No alternate
    # vendor currently available").
    return [d for d in drivers if d and not d.lower().endswith("drivers recorded")]


def _mock_narrative(
    entity_name: str,
    risk_score: float,
    fin_drivers: list[str],
    ops_drivers: list[str],
    comp_drivers: list[str],
    geo_drivers: list[str],
) -> str:
    """
    Build a vendor-specific narrative by interpolating the *actual* driver strings
    received in the prompt — not a canned generic line. Mirrors the crisp plain-text
    format the real prompt requests: a one-line Status, 2-3 Key-risk bullets, and a
    one-line Action. No markdown, so the dashboard renders it cleanly either way.
    """
    if risk_score >= 75:
        level = "elevated"
        action = ("Engage procurement leadership now to evaluate contingency suppliers "
                  "and request updated continuity documentation.")
    elif risk_score >= 50:
        level = "moderate"
        action = ("Schedule a quarterly business review and request updated continuity "
                  "and financial disclosures.")
    else:
        level = "manageable"
        action = "Maintain standard monitoring with an annual vendor assessment."

    # Pick the single strongest signal per dimension, quoted verbatim so acronyms
    # ("Altman Z-Score") and country codes are never case-mangled.
    bullets: list[str] = []
    if fin_drivers:
        bullets.append(fin_drivers[0])
    if ops_drivers:
        bullets.append(ops_drivers[0])
    # Prefer the most specific geopolitical driver (cross-strait / export-control)
    # over the generic country-risk baseline; fall back to compliance.
    geo = None
    if geo_drivers:
        specific = [d for d in geo_drivers if "country risk index" not in d.lower()]
        geo = specific[0] if specific else geo_drivers[0]
    third = geo or (comp_drivers[0] if comp_drivers else None)
    if third:
        bullets.append(third)

    if not bullets:
        bullets = ["No material risk flags raised this cycle."]

    status = (
        f"Status: {entity_name} presents {level} third-party risk "
        f"(composite {risk_score:.0f}/100)."
    )
    key_risks = "Key risks:\n" + "\n".join(f"- {b}" for b in bullets[:3])
    return f"{status}\n{key_risks}\nAction: {action}"


def _mock_alert_json(entity_name: str, score: float) -> str:
    severity = "critical" if score >= 80 else "high" if score >= 65 else "medium"
    return json.dumps({
        "alert_title": f"{entity_name} risk threshold exceeded",
        "severity": severity,
        "summary": (f"{entity_name} has breached the {severity} risk threshold with a "
                    f"composite score of {score:.0f}/100. Primary drivers include "
                    f"financial stress and geopolitical exposure."),
        "recommended_action": "Initiate contingency supplier identification process",
        "escalate_to": "CPO",
        "time_sensitivity": "24h" if score >= 80 else "1-week",
    })


def _mock_alternatives_json(prompt: str) -> str:
    """
    Mock ranked-alternatives response (G2). Extracts the candidate list from the
    prompt and returns a realistic ranked, justified JSON array so the mock path
    populates node.meta.backups end-to-end.
    """
    candidates: list[str] = []
    in_block = False
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("candidate alternative vendors"):
            in_block = True
            continue
        if in_block:
            if stripped.startswith("-"):
                candidates.append(stripped.lstrip("- ").strip())
            elif stripped == "" and candidates:
                break

    justifications = [
        "Strongest capacity and qualification overlap for rapid contingency switch.",
        "Credible second source with comparable technical scope but longer ramp time.",
        "Viable fallback; geographic diversification reduces correlated regional risk.",
        "Backup option — qualification effort higher, hold as tertiary contingency.",
    ]
    ranked = [
        {"name": name, "justification": justifications[min(i, len(justifications) - 1)]}
        for i, name in enumerate(candidates)
    ]
    return json.dumps(ranked, indent=2)


def _mock_report_html(target: str, entity_count: int, critical_count: int) -> str:
    return f"""<h2>Executive Risk Briefing — {target}</h2>
<p>This briefing summarises the third-party risk posture across <strong>{entity_count} monitored
entities</strong> in {target}'s extended supply chain. The analysis identifies
<strong>{critical_count} entities</strong> at critical or high risk levels requiring
immediate attention.</p>

<h2>Critical Risk Findings</h2>
<ul>
<li><strong>Geographic concentration:</strong> Over 60% of tier-1 suppliers are headquartered
in high-geopolitical-risk regions, creating systemic vulnerability to trade disruptions.</li>
<li><strong>Single-source dependencies:</strong> Three tier-1 suppliers have no approved
alternates, representing an unacceptable concentration of operational risk.</li>
<li><strong>Contract expiry exposure:</strong> Multiple high-criticality vendor contracts
expire within 12 months without confirmed renewal terms.</li>
<li><strong>Financial stress signals:</strong> Two suppliers show deteriorating Altman Z-scores
approaching the distress threshold, warranting enhanced financial monitoring.</li>
</ul>

<h2>Supply Chain Concentration Risks</h2>
<p>Cascade analysis identifies three nodes with blast-radius impact exceeding 40% of total
supply chain throughput. The removal of any single critical node would trigger second-order
disruptions across multiple product lines, with estimated recovery timelines of 6–18 months
given current lead times and qualification processes.</p>

<h2>Recommended Immediate Actions</h2>
<ol>
<li>Engage CPO to initiate dual-sourcing programme for the three identified single-source
critical vendors within 30 days.</li>
<li>Commission financial health deep-dive for vendors with Z-score below 1.8.</li>
<li>Accelerate contract renewal negotiations for agreements expiring in Q3/Q4.</li>
<li>Request updated business continuity plans from all Tier-1 critical suppliers.</li>
<li>Establish weekly monitoring cadence for entities currently rated Critical or High.</li>
</ol>

<h2>30-Day Monitoring Priorities</h2>
<ul>
<li>Weekly financial signal review for high-risk entities</li>
<li>Geopolitical developments in TW, CN, KR supplier regions</li>
<li>Contract negotiation progress for expiring agreements</li>
<li>Alternate supplier qualification pipeline status</li>
</ul>"""


# ── Mock Client ───────────────────────────────────────────────────────────────

class MockLLMClient(BaseLLMClient):
    """
    Fully functional mock — returns realistic responses without any model.
    Routes requests based on keyword detection in the prompt.
    """

    async def _generate(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        await asyncio.sleep(0.05)   # simulate small latency
        prompt_lower = prompt.lower()

        # ── Alternatives ranking (G2) ──────────────────────────────────────
        if "rank the candidates" in prompt_lower or "alternative vendors" in prompt_lower:
            return _mock_alternatives_json(prompt)

        # ── Tier-2 upstream expansion (A5) ─────────────────────────────────
        # Must be checked BEFORE the watchlist branch: the expansion prompt also
        # mentions "supply-chain" and "json", which would otherwise return the
        # full tier-1 watchlist instead of a tier-2 trio.
        if "tier-2" in prompt_lower or "upstream tier" in prompt_lower:
            return _mock_tier2_json(prompt)

        # ── Watchlist ──────────────────────────────────────────────────────
        if "watchlist" in prompt_lower or "supply chain" in prompt_lower and "json" in prompt_lower:
            # Find matching company
            for key, entities in MOCK_WATCHLISTS.items():
                if key in prompt_lower:
                    return json.dumps({"target": key.title(), "entities": entities}, indent=2)
            # Generic fallback
            return json.dumps({
                "target": "Unknown Company",
                "entities": [
                    {"name": "Acme Supplies Co", "ticker": None, "entity_type": "supplier",
                     "relationship_to_parent": "Unknown Company", "depth_level": 1,
                     "importance_score": 7, "industry": "Industrial Components",
                     "hq_country": "US"},
                    {"name": "Global Logistics Inc", "ticker": None, "entity_type": "logistics",
                     "relationship_to_parent": "Unknown Company", "depth_level": 1,
                     "importance_score": 6, "industry": "Freight & Logistics",
                     "hq_country": "SG"},
                ]
            }, indent=2)

        # ── Alert ──────────────────────────────────────────────────────────
        if "alert" in prompt_lower and "json" in prompt_lower:
            # Extract entity name heuristically
            lines = prompt.split("\n")
            entity_name = "Unknown Entity"
            score = 72.0
            for line in lines:
                if line.startswith("Entity:"):
                    entity_name = line.replace("Entity:", "").strip()
                if "Risk Score:" in line:
                    try:
                        score = float(line.split(":")[1].split("/")[0].strip())
                    except Exception:
                        pass
            return _mock_alert_json(entity_name, score)

        # ── Cascade summary ────────────────────────────────────────────────
        if "cascade" in prompt_lower:
            return (
                "The supply chain exhibits significant cascade risk concentration, "
                "with three nodes demonstrating betweenness centrality scores placing "
                "them as critical bridges in the network topology. A failure at any of "
                "these nodes would propagate disruption across multiple downstream entities "
                "within 48-72 hours, affecting an estimated 35-55% of total supply capacity. "
                "The TSMC node represents the most acute single point of failure given its "
                "role as sole qualified fabricator for leading-edge silicon components, "
                "with no credible alternate capable of absorbing volume within a 12-month horizon."
            )

        # ── Executive report ───────────────────────────────────────────────
        if "executive" in prompt_lower or "briefing" in prompt_lower or "report" in prompt_lower:
            # Extract target company
            target = "Target Company"
            for line in prompt.split("\n"):
                if "Target Company:" in line:
                    target = line.split(":", 1)[1].strip()
                    break
            entity_count = 0
            critical_count = 0
            for line in prompt.split("\n"):
                if "Total Entities" in line:
                    try:
                        entity_count = int(line.split(":")[-1].strip())
                    except Exception:
                        pass
                if "Critical Risk" in line and ":" in line:
                    try:
                        critical_count = int(line.split(":")[-1].strip())
                    except Exception:
                        pass
            return _mock_report_html(target, entity_count or 16, critical_count or 3)

        # ── Risk narrative (default) ───────────────────────────────────────
        entity_name = "This entity"
        risk_score = 55.0
        for line in prompt.split("\n"):
            if line.startswith("Entity:"):
                entity_name = line.replace("Entity:", "").strip().split("(")[0].strip()
            if "Risk Score:" in line:
                try:
                    risk_score = float(line.split(":")[1].split("/")[0].strip())
                except Exception:
                    pass
        # Fix 3: interpolate the *actual* drivers the prompt carries so each
        # narrative is specific to its vendor rather than a canned generic string.
        fin_drivers  = _parse_driver_block(prompt, "Financial Drivers")
        ops_drivers  = _parse_driver_block(prompt, "Operational Drivers")
        comp_drivers = _parse_driver_block(prompt, "Compliance Drivers")
        geo_drivers  = _parse_driver_block(prompt, "Geopolitical Drivers")
        return _mock_narrative(
            entity_name, risk_score,
            fin_drivers, ops_drivers, comp_drivers, geo_drivers,
        )

    async def generate_batch(
        self,
        prompts: list[str],
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Process batch sequentially (mock — no concurrency benefit)."""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, system=system,
                                         temperature=temperature, max_tokens=max_tokens)
            results.append(result)
        return results
