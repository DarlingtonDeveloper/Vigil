from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

from app.config import settings


class ToolSpec(BaseModel):
    name: str
    action_type: str = ""
    description: str = ""


class DeploymentProfile(BaseModel):
    agent_description: str = ""
    tools: list[ToolSpec] = Field(default_factory=list)
    data_access: list[str] = Field(default_factory=list)
    autonomy_level: str = ""
    output_reach: str = ""
    sector: str = ""
    jurisdictions: list[str] = Field(default_factory=list)
    human_oversight_model: str | None = None
    existing_guardrails: list[str] = Field(default_factory=list)


INTAKE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an AI deployment intake analyst for FaultLine, an AI liability insurance platform. "
        "Parse the deployment description into a structured profile. "
        "Identify tools, data access patterns, autonomy level (supervised/semi-autonomous/autonomous), "
        "output reach (internal/external/public), and any existing guardrails. "
        "Sector should be one of: financial, healthcare, legal, technology, retail, other. "
        "Respond with valid JSON matching the DeploymentProfile schema."
    )),
    ("human", (
        "Deployment description: {description}\n"
        "Jurisdictions: {jurisdictions}\n"
        "Sector hint: {sector}"
    )),
])


async def run_intake(
    description: str,
    jurisdictions: list[str],
    sector: str | None,
    session_id: str,
) -> DeploymentProfile:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        api_key=settings.anthropic_api_key,
        max_tokens=2048,
    )
    structured_llm = llm.with_structured_output(DeploymentProfile)

    chain = INTAKE_PROMPT | structured_llm
    result = await chain.ainvoke({
        "description": description,
        "jurisdictions": ", ".join(jurisdictions),
        "sector": sector or "not specified",
    })
    return result
