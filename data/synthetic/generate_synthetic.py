"""
Synthetic vendor registry generator.
Run this to regenerate or extend the synthetic internal data for a new target company.

Usage:
    python data/synthetic/generate_synthetic.py --company "Microsoft" --output data/synthetic/msft_registry.json
"""

import json
import random
import argparse
from datetime import datetime, timedelta
from pathlib import Path


def random_date(years_ahead_min=0, years_ahead_max=3) -> str:
    days = random.randint(years_ahead_min * 365, years_ahead_max * 365)
    return (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d")


def random_past_date(months_back=18) -> str:
    days = random.randint(0, months_back * 30)
    return (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")


CERT_POOL = ["ISO27001", "SOC2", "ISO9001", "ISO14001", "PCI-DSS", "EICC", "GDPR-DPO"]
COUNTRY_POOL = ["US", "TW", "KR", "JP", "DE", "GB", "NL", "SG", "IN", "IE"]
CATEGORY_POOL = [
    "Cloud Infrastructure", "Software Licensing", "Hardware Components",
    "Professional Services", "Logistics", "Marketing & Advertising",
    "Contract Manufacturing", "R&D Services", "Facilities Management",
]


def generate_vendor(vendor_id: int, company_name: str) -> dict:
    spend = random.randint(500_000, 5_000_000_000)
    return {
        "vendor_id": f"V{vendor_id:03d}",
        "vendor_name": f"Vendor {vendor_id} for {company_name}",
        "annual_spend_usd": spend,
        "spend_percentage": round(random.uniform(0.1, 15.0), 2),
        "contract_expiry": random_date(0, 3),
        "single_source": random.random() < 0.25,
        "criticality_tier": random.randint(1, 3),
        "compliance_certifications": random.sample(CERT_POOL, k=random.randint(1, 4)),
        "geographic_risk_country": random.choice(COUNTRY_POOL),
        "business_continuity_plan": random.random() < 0.7,
        "last_audit_date": random_past_date(18),
        "audit_score": round(random.uniform(60, 100), 1),
        "incidents_last_12m": random.randint(0, 4),
        "payment_terms_days": random.choice([30, 45, 60, 90]),
        "alternate_vendor_available": random.random() < 0.6,
        "service_categories": random.sample(CATEGORY_POOL, k=random.randint(1, 3)),
        "contract_auto_renew": random.random() < 0.3,
        "gdpr_dpa_signed": random.random() < 0.75,
    }


def generate_registry(company_name: str, vendor_count: int = 20) -> dict:
    return {
        "company": company_name,
        "reporting_date": datetime.utcnow().strftime("%Y-%m-%d"),
        "total_external_spend_usd": random.randint(1_000_000_000, 200_000_000_000),
        "vendor_records": [
            generate_vendor(i + 1, company_name) for i in range(vendor_count)
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic vendor registry")
    parser.add_argument("--company", default="Target Company", help="Company name")
    parser.add_argument("--count", type=int, default=20, help="Number of vendors")
    parser.add_argument("--output", default="data/synthetic/vendor_registry.json")
    args = parser.parse_args()

    registry = generate_registry(args.company, args.count)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"Generated {args.count} vendor records for '{args.company}' → {out_path}")
