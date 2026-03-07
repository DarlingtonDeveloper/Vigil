from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


class LegalExposure(BaseModel):
    doctrine: str = ""
    risk_level: str = ""
    reasoning: str = ""


class LegalAnalysis(BaseModel):
    legal_exposure_score: float = 0.0
    exposures: list[LegalExposure] = Field(default_factory=list)
    applicable_regulations: list[str] = Field(default_factory=list)
    compliance_gaps: list[str] = Field(default_factory=list)
    summary: str = ""


LEGAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a legal analyst for FaultLine AI liability insurance. "
        "Analyse the AI deployment profile against applicable legal doctrines and regulations. "
        "Score legal exposure from 0.0 (minimal) to 1.0 (extreme). "
        "Reference specific doctrines from the knowledge graph data provided. "
        "Respond with valid JSON matching the LegalAnalysis schema."
    )),
    ("human", (
        "Deployment profile:\n{profile}\n\n"
        "Applicable doctrines:\n{doctrines}\n\n"
        "Applicable regulations:\n{regulations}"
    )),
])


async def run_legal_analysis(
    deployment_profile: dict,
    applicable_doctrines: list[dict],
    applicable_regulations: list[dict],
    session_id: str,
) -> LegalAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    structured_llm = llm.with_structured_output(LegalAnalysis)

    chain = LEGAL_PROMPT | structured_llm
    result = await chain.ainvoke({
        "profile": str(deployment_profile),
        "doctrines": str(applicable_doctrines),
        "regulations": str(applicable_regulations),
    })
    return result
