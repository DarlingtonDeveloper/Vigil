"""
Mitigation Scorer Agent — Evaluates the deployment's existing mitigations
and recommends additional ones.
"""
import json
import logging
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.llm import invoke_structured
from app.prompts.manager import get_active_prompt

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
    quick_wins: list[str]
    confidence: float


MITIGATION_SYSTEM_PROMPT = """You are the Mitigation Scorer Agent for Vigil, an AI risk pricing engine.

You assess what risk mitigations are in place for an agentic deployment, score their effectiveness, and recommend additional ones.

You receive:
1. The deployment profile (existing guardrails, oversight model, etc.)
2. The full mitigation catalog from Vigil's knowledge graph
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


async def run_mitigation_analysis(
    profile: dict,
    available_mitigations: list[dict],
    mitigation_edges: list[dict],
    session_id: str,
) -> MitigationAnalysis:
    system_prompt = await get_active_prompt("mitigation") or MITIGATION_SYSTEM_PROMPT

    context = json.dumps({
        "deployment_profile": profile,
        "mitigation_catalog": available_mitigations,
        "mitigation_risk_reductions": mitigation_edges,
    }, indent=2, default=str)

    user_msg = f"Score this deployment's mitigations and recommend improvements:\n\n{context}"

    return await invoke_structured(
        MitigationAnalysis,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)],
        "Mitigation agent",
    )
