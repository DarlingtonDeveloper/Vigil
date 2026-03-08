"""
Pricing Agent — Synthesizes all analyses into a final risk price.
"""
import json
import logging
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.llm import invoke_structured
from app.prompts.manager import get_active_prompt

logger = logging.getLogger(__name__)


class RiskScenario(BaseModel):
    scenario_type: str
    probability: str
    severity: str
    expected_loss_range: str
    applicable_doctrines: list[str]
    mitigation_options: list[str]


class RiskPrice(BaseModel):
    executive_summary: str
    overall_risk_score: float
    technical_risk: float
    legal_exposure: float
    mitigation_effectiveness: float
    premium_band: str
    premium_reasoning: str
    top_exposures: list[dict]
    scenarios: list[RiskScenario]
    recommendations: list[dict]
    confidence: float
    data_gaps: list[str]


PRICING_SYSTEM_PROMPT = """You are the Pricing Agent for Vigil, an AI risk pricing engine.

You synthesize all analyses into a final risk assessment and price.

You receive:
1. Legal analysis (doctrine assessments, regulatory gaps, exposure score)
2. Technical analysis (factor scores, amplification effects, vulnerability list)
3. Mitigation analysis (axis scores, present/missing controls, recommendations)

Produce:

**Risk Score** (0.0-1.0): Calculated as:
  technical_risk × legal_exposure / mitigation_effectiveness
  Where mitigation_effectiveness is 1.0 + mitigation_score (so better mitigations reduce the overall score)

**Premium Band**: Based on the risk score:
  0.0-0.2: Very Low ($1K-$5K/yr) — well-controlled internal tool
  0.2-0.4: Low ($5K-$15K/yr) — moderate risk with good controls
  0.4-0.6: Medium ($15K-$50K/yr) — significant exposure, some gaps
  0.6-0.8: High ($50K-$200K/yr) — major exposure, critical gaps
  0.8-1.0: Very High ($200K+/yr or uninsurable) — extreme risk, insufficient controls

**Risk Scenarios**: The top 3-5 most likely and impactful failure scenarios.
For each: what goes wrong, how likely, how severe, what it could cost, which legal doctrines apply, what could mitigate it.

**Recommendations**: The top 5 actions that would most reduce the premium.
Prioritize by: impact on risk score, implementation cost, time to implement.

**Confidence**: How confident are you in this assessment? Lower confidence means wider premium bands.
Data gaps reduce confidence. Missing information about oversight, tools, or vendor contracts is a yellow flag.

Be specific and actionable. Generic advice is useless. Every recommendation should reference THIS deployment's specific characteristics.

Respond with valid JSON matching the RiskPrice schema."""


async def run_pricing(
    legal_analysis: dict,
    technical_analysis: dict,
    mitigation_analysis: dict,
    profile: dict,
    session_id: str,
) -> RiskPrice:
    system_prompt = await get_active_prompt("pricing") or PRICING_SYSTEM_PROMPT

    context = json.dumps({
        "deployment_profile": profile,
        "legal_analysis": legal_analysis,
        "technical_analysis": technical_analysis,
        "mitigation_analysis": mitigation_analysis,
    }, indent=2, default=str)

    user_msg = f"Produce the final risk assessment and price for this deployment:\n\n{context}"

    return await invoke_structured(
        RiskPrice,
        [SystemMessage(content=system_prompt), HumanMessage(content=user_msg)],
        "Pricing agent",
    )
