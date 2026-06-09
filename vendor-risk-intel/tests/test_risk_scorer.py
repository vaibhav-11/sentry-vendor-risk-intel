"""
Unit tests for the risk scoring engine.
Run with: pytest tests/ -v
"""

import pytest
from src.models import (
    Entity, EntityType, FootprintData, FinancialMetrics,
    InternalVendorRecord, RiskLevel
)
from src.risk.scorer import (
    score_entity, _score_financial, _score_operational,
    _score_compliance, _score_geopolitical, _risk_level
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_entity():
    return Entity(
        id="test-us", name="Test Corp", entity_type=EntityType.SUPPLIER,
        hq_country="US", industry="Software", importance_score=5.0
    )

@pytest.fixture
def high_risk_entity():
    return Entity(
        id="test-cn", name="China Supplier Co", entity_type=EntityType.SUPPLIER,
        hq_country="CN", industry="Electronics", importance_score=9.0
    )

@pytest.fixture
def empty_footprint(healthy_entity):
    return FootprintData(entity_id=healthy_entity.id, entity_name=healthy_entity.name)

@pytest.fixture
def rich_footprint(healthy_entity):
    fin = FinancialMetrics(
        entity_id=healthy_entity.id,
        market_cap_usd=50_000_000_000,
        revenue_ttm_usd=10_000_000_000,
        revenue_growth_yoy_pct=8.0,
        debt_to_equity=30.0,
        current_ratio=2.5,
        altman_z_score=3.5,
        data_quality=0.9,
    )
    internal = InternalVendorRecord(
        vendor_id="V001", vendor_name=healthy_entity.name,
        annual_spend_usd=500_000_000, spend_percentage=5.0,
        single_source=False, criticality_tier=2,
        compliance_certifications=["ISO27001", "SOC2"],
        geographic_risk_country="US",
        business_continuity_plan=True,
        audit_score=90.0, incidents_last_12m=0,
        alternate_vendor_available=True, gdpr_dpa_signed=True,
    )
    return FootprintData(
        entity_id=healthy_entity.id, entity_name=healthy_entity.name,
        financials=fin, internal_record=internal,
    )

@pytest.fixture
def distressed_footprint(high_risk_entity):
    fin = FinancialMetrics(
        entity_id=high_risk_entity.id,
        revenue_growth_yoy_pct=-20.0,
        debt_to_equity=250.0,
        current_ratio=0.8,
        altman_z_score=1.2,
        data_quality=0.7,
    )
    internal = InternalVendorRecord(
        vendor_id="V002", vendor_name=high_risk_entity.name,
        annual_spend_usd=5_000_000_000, spend_percentage=20.0,
        single_source=True, criticality_tier=1,
        compliance_certifications=[],
        geographic_risk_country="CN",
        business_continuity_plan=False,
        audit_score=55.0, incidents_last_12m=4,
        alternate_vendor_available=False, gdpr_dpa_signed=False,
    )
    return FootprintData(
        entity_id=high_risk_entity.id, entity_name=high_risk_entity.name,
        financials=fin, internal_record=internal,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_critical_threshold(self):
        assert _risk_level(85) == RiskLevel.CRITICAL
        assert _risk_level(80) == RiskLevel.CRITICAL

    def test_high_threshold(self):
        assert _risk_level(70) == RiskLevel.HIGH
        assert _risk_level(65) == RiskLevel.HIGH

    def test_medium_threshold(self):
        assert _risk_level(50) == RiskLevel.MEDIUM
        assert _risk_level(45) == RiskLevel.MEDIUM

    def test_low_threshold(self):
        assert _risk_level(44) == RiskLevel.LOW
        assert _risk_level(0)  == RiskLevel.LOW


class TestFinancialScoring:
    def test_healthy_financials_give_low_score(self, healthy_entity, rich_footprint):
        dim = _score_financial(rich_footprint)
        assert dim.score < 40, "Healthy company should have low financial risk"
        assert dim.confidence > 0.5

    def test_distressed_financials_give_high_score(self, high_risk_entity, distressed_footprint):
        dim = _score_financial(distressed_footprint)
        assert dim.score > 60, "Distressed company should have high financial risk"

    def test_no_financials_returns_neutral(self, healthy_entity, empty_footprint):
        dim = _score_financial(empty_footprint)
        assert dim.score == 50.0
        assert dim.confidence < 0.5

    def test_altman_z_distress_zone(self, healthy_entity, rich_footprint):
        rich_footprint.financials.altman_z_score = 1.5
        dim = _score_financial(rich_footprint)
        assert any("distress" in d.lower() for d in dim.key_drivers)


class TestOperationalScoring:
    def test_single_source_raises_score(self, high_risk_entity, distressed_footprint):
        dim = _score_operational(high_risk_entity, distressed_footprint)
        assert dim.score > 70
        assert any("single-source" in d.lower() for d in dim.key_drivers)

    def test_high_spend_concentration_flagged(self, high_risk_entity, distressed_footprint):
        dim = _score_operational(high_risk_entity, distressed_footprint)
        assert any("concentration" in d.lower() for d in dim.key_drivers)

    def test_low_risk_operational(self, healthy_entity, rich_footprint):
        dim = _score_operational(healthy_entity, rich_footprint)
        assert dim.score < 40


class TestComplianceScoring:
    def test_sanctioned_country_triggers_high_score(self, healthy_entity, empty_footprint):
        healthy_entity.hq_country = "RU"
        dim = _score_compliance(healthy_entity, empty_footprint)
        assert dim.score > 80

    def test_missing_certs_flagged(self, high_risk_entity, distressed_footprint):
        dim = _score_compliance(high_risk_entity, distressed_footprint)
        assert any("certification" in d.lower() for d in dim.key_drivers)


class TestGeopoliticalScoring:
    def test_tw_higher_than_us(self, healthy_entity, empty_footprint):
        healthy_entity.hq_country = "US"
        us_score = _score_geopolitical(healthy_entity, empty_footprint).score
        healthy_entity.hq_country = "TW"
        tw_score = _score_geopolitical(healthy_entity, empty_footprint).score
        assert tw_score > us_score

    def test_cn_flagged(self, high_risk_entity, distressed_footprint):
        dim = _score_geopolitical(high_risk_entity, distressed_footprint)
        assert dim.score > 60


class TestCompositeScoring:
    def test_healthy_entity_low_composite(self, healthy_entity, rich_footprint):
        score = score_entity(healthy_entity, rich_footprint)
        assert score.composite_score < 50
        assert score.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

    def test_distressed_entity_high_composite(self, high_risk_entity, distressed_footprint):
        score = score_entity(high_risk_entity, distressed_footprint)
        assert score.composite_score > 65
        assert score.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)

    def test_score_in_range(self, healthy_entity, empty_footprint):
        score = score_entity(healthy_entity, empty_footprint)
        assert 0 <= score.composite_score <= 100

    def test_score_fields_populated(self, healthy_entity, rich_footprint):
        score = score_entity(healthy_entity, rich_footprint)
        assert score.entity_id == healthy_entity.id
        assert score.entity_name == healthy_entity.name
        assert score.financial is not None
        assert score.operational is not None
        assert score.compliance is not None
        assert score.geopolitical is not None
