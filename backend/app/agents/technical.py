from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


class TechnicalRisk(BaseModel):
    factor: str = ""
    score: float = 0.0
    reasoning: str = ""


class TechnicalAnalysis(BaseModel):
    technical_risk_score: float = 0.0
    risks: list[TechnicalRisk] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    data_risks: list[str] = Field(default_factory=list)
    summary: str = ""


TECHNICAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a technical risk analyst for FaultLine AI liability insurance. "
        "Analyse the AI deployment profile against known risk factors. "
        "Score technical risk from 0.0 (minimal) to 1.0 (extreme). "
        "Consider failure modes, data risks, and autonomy-related risks. "
        "Respond with valid JSON matching the TechnicalAnalysis schema."
    )),
    ("human", (
        "Deployment profile:\n{profile}\n\n"
        "Known risk factors:\n{risk_factors}"
    )),
])


async def run_technical_analysis(
    deployment_profile: dict,
    risk_factors: list[dict],
    session_id: str,
) -> TechnicalAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    structured_llm = llm.with_structured_output(TechnicalAnalysis)

    chain = TECHNICAL_PROMPT | structured_llm
    result = await chain.ainvoke({
        "profile": str(deployment_profile),
        "risk_factors": str(risk_factors),
    })
    return result
