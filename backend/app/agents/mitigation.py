from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


class MitigationRecommendation(BaseModel):
    name: str = ""
    effectiveness: float = 0.0
    implementation_effort: str = ""
    reasoning: str = ""


class MitigationAnalysis(BaseModel):
    overall_mitigation_score: float = 0.0
    recommended_mitigations: list[MitigationRecommendation] = Field(default_factory=list)
    existing_coverage: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""


MITIGATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a mitigation strategy analyst for FaultLine AI liability insurance. "
        "Analyse the deployment profile and available mitigations to recommend a mitigation strategy. "
        "Score overall mitigation effectiveness from 0.0 (no mitigation) to 1.0 (fully mitigated). "
        "Use the mitigation edges to understand which mitigations reduce which risk factors. "
        "Respond with valid JSON matching the MitigationAnalysis schema."
    )),
    ("human", (
        "Deployment profile:\n{profile}\n\n"
        "Available mitigations:\n{mitigations}\n\n"
        "Mitigation-to-risk edges:\n{edges}"
    )),
])


async def run_mitigation_analysis(
    deployment_profile: dict,
    available_mitigations: list[dict],
    mitigation_edges: list[dict],
    session_id: str,
) -> MitigationAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    structured_llm = llm.with_structured_output(MitigationAnalysis)

    chain = MITIGATION_PROMPT | structured_llm
    result = await chain.ainvoke({
        "profile": str(deployment_profile),
        "mitigations": str(available_mitigations),
        "edges": str(mitigation_edges),
    })
    return result
