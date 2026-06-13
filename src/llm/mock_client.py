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
    received in the prompt — not a canned generic line. Each narrative references
    the entity name, composite score, and at least one financial, operational, and
    geopolitical/compliance driver, so every entity's narrative is distinguishable.
    """
    if risk_score >= 75:
        level = "elevated"
        action = ("Immediate engagement with procurement leadership to evaluate "
                  "contingency suppliers and request updated continuity documentation")
    elif risk_score >= 50:
        level = "moderate"
        action = ("Schedule a quarterly business review and request updated business "
                  "continuity and financial disclosures")
    else:
        level = "manageable"
        action = "Maintain standard monitoring cadence with an annual vendor assessment"

    # Surface up to the two strongest financial signals (e.g. Z-score + revenue
    # growth) verbatim so acronyms like "Altman Z-Score" / country codes are never
    # case-mangled. Drivers are quoted as-is, joined with semicolons.
    fin = "; ".join(fin_drivers[:2]) if fin_drivers else None
    ops = ops_drivers[0] if ops_drivers else None
    # Prefer the most *specific* geopolitical driver (cross-strait / export-control
    # exposure) over the generic country-risk baseline; fall back to compliance.
    geo = None
    if geo_drivers:
        specific = [d for d in geo_drivers if "country risk index" not in d.lower()]
        geo = specific[0] if specific else geo_drivers[0]
    third = geo or (comp_drivers[0] if comp_drivers else None)

    # Opening: status + the financial signals we actually have.
    if fin:
        opening = (
            f"{entity_name} presents {level} third-party risk (composite {risk_score:.0f}/100). "
            f"Financial signals — {fin}."
        )
    else:
        opening = (
            f"{entity_name} presents {level} third-party risk (composite {risk_score:.0f}/100), "
            f"with limited financial telemetry available for this cycle."
        )

    # Middle: operational and geo/compliance specifics, quoted verbatim.
    middle_bits: list[str] = []
    if ops:
        middle_bits.append(f"operationally, {ops}")
    if third:
        middle_bits.append(third)
    middle = (
        ("On the supply side: " + "; ".join(middle_bits) + ".")
        if middle_bits else
        "No additional operational or geopolitical flags were raised this cycle."
    )

    return (
        f"{opening} {middle} "
        f"These signals are specific to {entity_name} and warrant proportionate attention. "
        f"Recommendation: {action}."
    )


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

    async def generate(
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
