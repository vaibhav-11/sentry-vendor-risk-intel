"""
Central data models for the Vendor Risk Intelligence platform.
All pipeline modules share these schemas — change here, change everywhere.
"""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class EntityType(str, Enum):
    SUPPLIER    = "supplier"
    CUSTOMER    = "customer"
    PARTNER     = "partner"
    LOGISTICS   = "logistics"
    FINANCIAL   = "financial"
    TARGET      = "target"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class AlertSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH     = "high"
    MEDIUM   = "medium"
    LOW      = "low"


class EscalateTo(str, Enum):
    CISO        = "CISO"
    CPO         = "CPO"
    CFO         = "CFO"
    LEGAL       = "Legal"
    PROCUREMENT = "Procurement"


# ── Core Entity ───────────────────────────────────────────────────────────────

class Entity(BaseModel):
    id: str                                         # slug: "tsmc-tw"
    name: str
    ticker: Optional[str] = None
    entity_type: EntityType
    relationship_to_parent: str = ""               # e.g. "Primary chip fab"
    parent_id: Optional[str] = None
    depth_level: int = 1                           # 0 = target, 1,2,3 = supply chain
    importance_score: float = Field(ge=0, le=10, default=5.0)
    industry: str = ""
    hq_country: str = ""                           # ISO-2 code, e.g. "TW"
    description: str = ""
    wikipedia_url: Optional[str] = None


class EntityRelationship(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str                         # "manufactures_for", "supplies_to" etc.
    dependency_strength: float = Field(ge=0, le=1, default=0.5)
    annual_value_usd: Optional[float] = None
    is_single_source: bool = False


# ── Financial & News Data ─────────────────────────────────────────────────────

class FinancialMetrics(BaseModel):
    entity_id: str
    fetch_date: datetime = Field(default_factory=datetime.utcnow)
    market_cap_usd: Optional[float] = None
    stock_price: Optional[float] = None
    price_change_30d_pct: Optional[float] = None
    revenue_ttm_usd: Optional[float] = None
    revenue_growth_yoy_pct: Optional[float] = None
    gross_margin_pct: Optional[float] = None
    net_income_ttm_usd: Optional[float] = None
    total_debt_usd: Optional[float] = None
    cash_usd: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    altman_z_score: Optional[float] = None        
    data_quality: float = 1.0                     


class NewsItem(BaseModel):
    entity_id: str
    title: str
    source: str
    published_at: datetime
    url: str = ""
    sentiment_score: float = Field(ge=-1, le=1, default=0.0)
    risk_relevant: bool = False
    summary: str = ""


class SECFiling(BaseModel):
    entity_id: str
    form_type: str           
    filed_at: datetime
    accession_number: str
    description: str = ""
    risk_flags: list[str] = Field(default_factory=list)
    url: Optional[str] = None


# ── Internal Vendor Registry (synthetic data) ─────────────────────────────────

class InternalVendorRecord(BaseModel):
    vendor_id: str
    vendor_name: str
    annual_spend_usd: float = 0.0
    spend_percentage: float = 0.0          
    contract_expiry: Optional[str] = None  
    single_source: bool = False
    criticality_tier: int = Field(ge=1, le=3, default=2)   
    compliance_certifications: list[str] = Field(default_factory=list)
    geographic_risk_country: str = ""
    business_continuity_plan: bool = False
    last_audit_date: Optional[str] = None
    audit_score: Optional[float] = None   
    incidents_last_12m: int = 0
    payment_terms_days: int = 30
    alternate_vendor_available: bool = True
    service_categories: list[str] = Field(default_factory=list)
    contract_auto_renew: bool = False
    gdpr_dpa_signed: bool = True


# ── Aggregated Footprint ──────────────────────────────────────────────────────

class FootprintData(BaseModel):
    entity_id: str
    entity_name: str
    fetch_date: datetime = Field(default_factory=datetime.utcnow)
    financials: Optional[FinancialMetrics] = None
    news_items: list[NewsItem] = Field(default_factory=list)
    sec_filings: list[SECFiling] = Field(default_factory=list)
    internal_record: Optional[InternalVendorRecord] = None
    description: str = ""
    news_sentiment_avg: float = 0.0
    negative_news_count: int = 0
    risk_news_headlines: list[str] = Field(default_factory=list)


# ── Risk Scoring ──────────────────────────────────────────────────────────────

class DimensionScore(BaseModel):
    score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1, default=0.8)
    key_drivers: list[str] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)


class RiskScore(BaseModel):
    entity_id: str
    entity_name: str
    composite_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    financial: DimensionScore
    operational: DimensionScore
    compliance: DimensionScore
    geopolitical: DimensionScore
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    score_delta_7d: Optional[float] = None        
    narrative: str = ""                           


# ── Graph / Cascade Risk ──────────────────────────────────────────────────────

class NodeMetrics(BaseModel):
    entity_id: str
    betweenness_centrality: float = 0.0
    in_degree: int = 0
    out_degree: int = 0
    is_single_point_of_failure: bool = False
    blast_radius_pct: float = 0.0   
    cascade_risk_score: float = 0.0  
    # NEW PROCUREMENT FIELDS
    direct_spend_usd: float = 0.0
    value_at_risk_usd: float = 0.0
    alternative_suppliers: list[str] = Field(default_factory=list)


class GraphMetrics(BaseModel):
    total_nodes: int
    total_edges: int
    max_depth: int
    density: float
    node_metrics: dict[str, NodeMetrics] = Field(default_factory=dict)
    critical_path: list[str] = Field(default_factory=list)    
    single_points_of_failure: list[str] = Field(default_factory=list)
    top_cascade_risks: list[str] = Field(default_factory=list)  
    # NEW STRATEGIC CONCENTRATION METRICS
    total_portfolio_value_usd: float = 0.0
    total_value_at_risk_usd: float = 0.0
    geo_concentration_hhi: float = 0.0
    country_spend_distribution: dict[str, float] = Field(default_factory=dict)


# ── Alerts ────────────────────────────────────────────────────────────────────

class RiskAlert(BaseModel):
    alert_id: str
    entity_id: str
    entity_name: str
    alert_title: str
    severity: AlertSeverity
    summary: str
    recommended_action: str
    escalate_to: str
    time_sensitivity: str
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    triggering_score: float
    triggering_dimension: str = ""
    acknowledged: bool = False


# ── Pipeline State (LangGraph) ────────────────────────────────────────────────

class PipelineState(BaseModel):
    """Mutable state object that flows through the LangGraph pipeline."""
    target_company: str
    target_ticker: Optional[str] = None
    llm_backend: str = "mock"
    run_id: str = ""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[EntityRelationship] = Field(default_factory=list)
    footprint_data: dict[str, FootprintData] = Field(default_factory=dict)
    risk_scores: dict[str, RiskScore] = Field(default_factory=dict)
    graph_metrics: Optional[GraphMetrics] = None
    alerts: list[RiskAlert] = Field(default_factory=list)
    report_html: str = ""
    dashboard_html: str = ""

    # PROVISIONED CUSTOMER DATA ROOM (FOR DEMONSTRATING ENTERPRISE RAG CAPABILITY)
    uploaded_documents: list[dict] = Field(default_factory=lambda: [
        {"name": "SOW_Core_Infrastructure_v3.pdf", "type": "Statement of Work", "size": "412 KB", "status": "Vectorized & Synced via RAG", "linked_nodes": ["tsmc-tw", "asml-nl"]},
        {"name": "Master_Sourcing_Agreement_2025.pdf", "type": "MSA Terms", "size": "1.8 MB", "status": "Vectorized & Synced via RAG", "linked_nodes": ["foxconn-hi"]},
        {"name": "Enterprise_CRM_Sourcing_Mappings.csv", "type": "ERP/CRM Direct Integration", "size": "84 KB", "status": "Live Feed Mapping Active", "linked_nodes": ["all"]}
    ])

    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    errors: list[str] = Field(default_factory=list)
    stage: str = "initialised"

    def add_error(self, msg: str) -> None:
        self.errors.append(f"[{self.stage}] {msg}")

    def entity_by_id(self, entity_id: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.id == entity_id), None)


# ── Report ────────────────────────────────────────────────────────────────────

class RiskReport(BaseModel):
    target_company: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    executive_summary: str
    critical_findings: list[str]
    supply_chain_risks: str
    recommended_actions: list[str]
    monitoring_priorities: list[str]
    full_html: str = ""