"""
Mitigation Scorer Agent — Evaluates the deployment's existing mitigations
and recommends additional ones.

Assesses the four mitigation axes:
1. Legal conformity (threshold)
2. Human oversight (who, authority, cognitive load)
3. Architectural controls (guardrails)
4. Evidentiary position (audit, documentation)
"""
import json
import logging
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings

logger = logging.getLogger(__name__)


class MitigationScore(BaseModel):
    mitigation_name: str
    present: bool
    effectiveness_if_present: float  # 0.0-1.0
    notes: str


class MitigationAxisScore(BaseModel):
    axis: str                    # legal_conformity, human_oversight, architectural, evidentiary
    score: float                 # 0.0-1.0
    present_mitigations: list[str]
    missing_mitigations: list[str]
    critical_gaps: list[str]


class MitigationAnalysis(BaseModel):
    overall_mitigation_score: float  # 0.0-1.0
    axis_scores: list[MitigationAxisScore]
    recommendations: list[dict]      # [{name, priority, impact, cost, reasoning}]
    quick_wins: list[str]            # mitigations that are trivial to implement
    confidence: float


MITIGATION_SYSTEM_PROMPT = """You are the Mitigation Scorer Agent for FaultLine, an AI risk pricing engine.

You assess what risk mitigations are in place for an agentic deployment, score their effectiveness, and recommend additional ones.

You receive:
1. The deployment profile (existing guardrails, oversight model, etc.)
2. The full mitigation catalog from FaultLine's knowledge graph
3. The mitigation-to-risk-factor reduction edges (which mitigations reduce which risks and by how much)

Score each of the four mitigation axes:

**Legal Conformity** (threshold — are you compliant or not?)
- EU AI Act classification and conformity
- GDPR/DPIA compliance
- Sector-specific regulation

**Human Oversight** (gradient — how meaningful is the oversight?)
- Consider: WHO reviews (domain expert vs general operator), their AUTHORITY (override vs observe), COGNITIVE LOAD (decisions per hour), and whether override rates are tracked
- Rubber-stamping scores worse than no oversight (false security)

**Architectural Controls** (deterministic risk reduction)
- Input validation, output schemas, tool scoping, confidence thresholds, adversarial testing, scope constraints

**Evidentiary Position** (post-incident severity reduction)
- Audit trails, prompt versioning, incident response, vendor contract review

For recommendations, include:
- name: specific mitigation
- priority: critical / high / medium / low
- impact: estimated risk reduction
- cost: trivial / moderate / significant / major
- reasoning: why this matters for THIS specific deployment

Identify quick wins — mitigations that are trivial to implement but meaningful.

Respond with valid JSON matching the MitigationAnalysis schema."""


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


async def run_mitigation_analysis(
    profile: dict,
    available_mitigations: list[dict],
    mitigation_edges: list[dict],
    session_id: str,
) -> MitigationAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )

    context = json.dumps({
        "deployment_profile": profile,
        "mitigation_catalog": available_mitigations,
        "mitigation_risk_reductions": mitigation_edges,
    }, indent=2, default=str)

    user_msg = f"Score this deployment's mitigations and recommend improvements:\n\n{context}"

    response = await llm.ainvoke([
        SystemMessage(content=MITIGATION_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    raw = _extract_json(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Mitigation agent returned invalid JSON: %s", e)
        raise ValueError(f"Mitigation agent returned invalid JSON: {e}") from e

    try:
        return MitigationAnalysis(**parsed)
    except ValidationError as e:
        logger.error("Mitigation agent output failed validation: %s", e)
        raise ValueError(f"Mitigation agent output failed validation: {e}") from e
