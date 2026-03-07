"""
Pricing Agent — Synthesizes all analyses into a final risk price.

Takes: legal analysis, technical analysis, mitigation analysis
Produces: overall risk score, premium band, scenario simulations, executive summary
"""
import json
import logging
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.tracing.cost_tracker import track_llm_cost

logger = logging.getLogger(__name__)


class RiskScenario(BaseModel):
    scenario_type: str           # what goes wrong
    probability: str             # "likely", "possible", "unlikely", "rare"
    severity: str                # "catastrophic", "major", "moderate", "minor"
    expected_loss_range: str     # "$10K-$50K", "$100K-$1M", etc.
    applicable_doctrines: list[str]
    mitigation_options: list[str]


class RiskPrice(BaseModel):
    executive_summary: str
    overall_risk_score: float    # 0.0-1.0
    technical_risk: float
    legal_exposure: float
    mitigation_effectiveness: float
    premium_band: str            # "low ($5K-$15K/yr)", "medium ($15K-$50K/yr)", etc.
    premium_reasoning: str
    top_exposures: list[dict]    # [{exposure, severity, mitigation_available}]
    scenarios: list[RiskScenario]
    recommendations: list[dict]  # [{action, priority, impact, reasoning}]
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


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


async def run_pricing(
    legal_analysis: dict,
    technical_analysis: dict,
    mitigation_analysis: dict,
    profile: dict,
    session_id: str,
) -> RiskPrice:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=8192,
        api_key=settings.anthropic_api_key,
    )

    context = json.dumps({
        "deployment_profile": profile,
        "legal_analysis": legal_analysis,
        "technical_analysis": technical_analysis,
        "mitigation_analysis": mitigation_analysis,
    }, indent=2, default=str)

    user_msg = f"Produce the final risk assessment and price for this deployment:\n\n{context}"

    response = await llm.ainvoke([
        SystemMessage(content=PRICING_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Track cost
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        await track_llm_cost(
            session_id, "pricing", "claude-sonnet-4-20250514",
            usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        )

    raw = _extract_json(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Pricing agent returned invalid JSON: %s", e)
        raise ValueError(f"Pricing agent returned invalid JSON: {e}") from e

    try:
        return RiskPrice(**parsed)
    except ValidationError as e:
        logger.error("Pricing agent output failed validation: %s", e)
        raise ValueError(f"Pricing agent output failed validation: {e}") from e
