from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


class RiskScenario(BaseModel):
    scenario_type: str = ""
    probability: str = ""
    severity: str = ""
    expected_loss_range: str = ""
    applicable_doctrines: list[str] = Field(default_factory=list)
    mitigation_options: list[str] = Field(default_factory=list)


class PricingResult(BaseModel):
    technical_risk: float = 0.0
    legal_exposure: float = 0.0
    overall_risk_score: float = 0.0
    premium_band: str = ""
    confidence: float = 0.0
    executive_summary: str = ""
    top_exposures: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    scenarios: list[RiskScenario] = Field(default_factory=list)


PRICING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a pricing actuary for FaultLine AI liability insurance. "
        "Synthesise the legal analysis, technical analysis, and mitigation analysis "
        "into a final risk score and premium band. "
        "Premium bands: very_low, low, medium, high, very_high, uninsurable. "
        "Generate realistic risk scenarios with probability/severity assessments. "
        "Respond with valid JSON matching the PricingResult schema."
    )),
    ("human", (
        "Legal analysis:\n{legal}\n\n"
        "Technical analysis:\n{technical}\n\n"
        "Mitigation analysis:\n{mitigation}\n\n"
        "Deployment profile:\n{profile}"
    )),
])


async def run_pricing(
    legal_analysis: dict,
    technical_analysis: dict,
    mitigation_analysis: dict,
    deployment_profile: dict,
    session_id: str,
) -> PricingResult:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    structured_llm = llm.with_structured_output(PricingResult)

    chain = PRICING_PROMPT | structured_llm
    result = await chain.ainvoke({
        "legal": str(legal_analysis),
        "technical": str(technical_analysis),
        "mitigation": str(mitigation_analysis),
        "profile": str(deployment_profile),
    })
    return result
