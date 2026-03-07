"""
Legal Analyst Agent — Assesses legal exposure by querying the knowledge graph
for applicable doctrines, regulations, and precedent.

This agent receives:
- The structured deployment profile from Intake
- Applicable legal doctrines from SurrealDB
- Applicable regulations from SurrealDB

And produces:
- A legal exposure score
- Specific doctrine-based risk assessments
- Regulatory compliance gaps
- Key uncertainties and worst-case scenarios
"""
import json
import logging
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.tracing.cost_tracker import track_llm_cost

logger = logging.getLogger(__name__)


class DoctrineAssessment(BaseModel):
    doctrine_name: str
    applies: bool
    exposure_level: str          # "high", "medium", "low", "uncertain"
    reasoning: str
    worst_case: str


class RegulatoryGap(BaseModel):
    regulation: str
    requirement: str
    status: str                  # "compliant", "partial", "non_compliant", "unknown"
    risk_if_non_compliant: str


class LegalAnalysis(BaseModel):
    legal_exposure_score: float  # 0.0-1.0
    doctrine_assessments: list[DoctrineAssessment]
    regulatory_gaps: list[RegulatoryGap]
    contract_formation_risk: str  # assessment of whether agent can form binding contracts
    tort_exposure: str            # assessment of negligence/misrepresentation risk
    key_uncertainties: list[str]
    confidence: float


LEGAL_SYSTEM_PROMPT = """You are the Legal Analyst Agent for Vigil, an AI risk pricing engine.

You assess the legal exposure of an agentic AI deployment. You have access to:
1. A structured deployment profile (what the agent does, its tools, data access, oversight)
2. Relevant legal doctrines from Vigil's knowledge graph
3. Applicable regulations

For each legal doctrine provided, assess:
- Does it apply to this specific deployment? (not all doctrines apply to all deployments)
- What is the exposure level? (high/medium/low/uncertain)
- Why? (specific reasoning grounded in the deployment's characteristics)
- What is the worst-case scenario if this risk materialises?

For each regulation, assess compliance gaps:
- Which requirements apply?
- Is the deployment compliant, partially compliant, or non-compliant?
- What is the risk if non-compliant?

Key principles:
- LOW precedent clarity INCREASES risk (uncertainty favours claimants in novel litigation)
- The direction of travel is toward MORE liability, not less
- Missing information should be treated conservatively (assume worst case)
- An agent that can communicate externally has higher contract/tort exposure than an internal-only agent
- Autonomy level directly correlates with liability — fully autonomous = maximum exposure
- Multi-jurisdiction deployments compound risk (must meet strictest standard)

Respond with valid JSON matching the LegalAnalysis schema.
Be specific. Cite the doctrines and regulations by name. Do not be generic."""


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


async def run_legal_analysis(
    profile: dict,
    applicable_doctrines: list[dict],
    applicable_regulations: list[dict],
    session_id: str,
) -> LegalAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=8192,
        api_key=settings.anthropic_api_key,
    )

    context = json.dumps({
        "deployment_profile": profile,
        "applicable_doctrines": applicable_doctrines,
        "applicable_regulations": applicable_regulations,
    }, indent=2, default=str)

    user_msg = f"Assess the legal exposure of this agentic deployment:\n\n{context}"

    response = await llm.ainvoke([
        SystemMessage(content=LEGAL_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Track cost
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        await track_llm_cost(
            session_id, "legal", "claude-sonnet-4-20250514",
            usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        )

    raw = _extract_json(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Legal agent returned invalid JSON: %s", e)
        raise ValueError(f"Legal agent returned invalid JSON: {e}") from e

    try:
        return LegalAnalysis(**parsed)
    except ValidationError as e:
        logger.error("Legal agent output failed validation: %s", e)
        raise ValueError(f"Legal agent output failed validation: {e}") from e
