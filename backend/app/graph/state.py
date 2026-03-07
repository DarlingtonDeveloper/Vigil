from typing import TypedDict


class VigilState(TypedDict):
    # Input
    session_id: str
    user_id: str
    description: str
    jurisdictions: list[str]
    sector: str | None

    # Intake output
    deployment_profile: dict | None

    # Knowledge graph context (fetched from SurrealDB for agents)
    applicable_doctrines: list[dict]
    applicable_regulations: list[dict]
    risk_factors: list[dict]
    available_mitigations: list[dict]
    mitigation_edges: list[dict]

    # Agent outputs
    legal_analysis: dict | None
    technical_analysis: dict | None
    mitigation_analysis: dict | None
    risk_price: dict | None

    # Opik quality scores
    legal_quality_score: float
    technical_quality_score: float
    pricing_quality_score: float

    # Meta
    current_step: str
    error: str | None
