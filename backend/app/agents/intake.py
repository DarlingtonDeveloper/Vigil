"""
Intake Agent — Parses a free-text description of an agentic deployment
and extracts structured data about:
- What the agent does
- What tools/actions it has access to
- What data it can access
- Its autonomy level
- Its output reach (who sees/relies on outputs)
- Which jurisdictions it operates in
- What human oversight exists
- What guardrails are in place
- Vendor/model information
"""
import json
import logging
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings
from app.tracing.cost_tracker import track_llm_cost

logger = logging.getLogger(__name__)


class ToolDescription(BaseModel):
    name: str
    action_type: str         # "read", "write", "communicate", "transact"
    description: str
    risk_note: str | None = None


class VendorInfo(BaseModel):
    provider: str            # "Anthropic", "OpenAI", etc.
    model: str               # "claude-sonnet-4-20250514", "gpt-4o"
    indemnification: str     # "full", "partial", "none", "unknown"
    contract_reviewed: bool


class DeploymentProfile(BaseModel):
    agent_description: str
    tools: list[ToolDescription]
    data_access: list[str]                  # ["PII", "financial", "health", "internal", "public"]
    autonomy_level: str                     # "human_in_command", "human_in_the_loop", "human_on_the_loop", "fully_autonomous"
    output_reach: str                       # "internal_only", "customer_facing", "public_facing", "legally_binding"
    sector: str                             # "financial", "healthcare", "legal", "education", "general", "employment"
    jurisdictions: list[str]
    human_oversight_model: str | None       # description of who reviews what
    reviewer_qualification: str | None      # "domain_expert", "general_operator", "none"
    existing_guardrails: list[str]
    vendor_info: VendorInfo | None
    key_risks_identified: list[str]         # agent's initial observations


INTAKE_SYSTEM_PROMPT = """You are the Intake Agent for FaultLine, an AI risk pricing engine.

A company is describing an agentic AI system they want to deploy (or have deployed). Your job is to extract structured data about the deployment that will feed into legal and technical risk assessment.

From the description, extract:

1. **agent_description**: Concise summary of what the agent does
2. **tools**: List of tools/actions the agent can perform. For each:
   - name, action_type (read/write/communicate/transact), description
   - risk_note: any concern about this tool's risk (e.g., "can send emails to customers" = external communication risk)
3. **data_access**: What types of data can the agent access? Classify as: public, internal, PII, financial, health, legal_privileged
4. **autonomy_level**: One of:
   - human_in_command: human makes all decisions, agent only suggests
   - human_in_the_loop: agent proposes, human approves each action
   - human_on_the_loop: agent acts, human monitors and can intervene
   - fully_autonomous: agent acts without oversight
5. **output_reach**: One of: internal_only, customer_facing, public_facing, legally_binding
6. **sector**: financial, healthcare, legal, education, employment, general
7. **jurisdictions**: Where does this operate? UK, EU, US, or combinations
8. **human_oversight_model**: Description of who reviews agent output and how
9. **reviewer_qualification**: Are reviewers domain experts, general operators, or nobody?
10. **existing_guardrails**: What safety measures are already in place?
11. **vendor_info**: If mentioned — which AI provider, model, contract terms
12. **key_risks_identified**: Your initial observations about major risk areas

Be thorough in extraction. If information is not provided, note it as missing — missing information is itself a risk signal. If the description is vague about autonomy or oversight, classify conservatively (assume higher risk).

Respond with valid JSON matching the DeploymentProfile schema."""


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


async def run_intake(
    description: str,
    jurisdictions: list[str],
    sector: str | None,
    session_id: str,
) -> DeploymentProfile:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )
    user_msg = f"""Parse this agentic deployment description and extract the structured risk profile:

DEPLOYMENT DESCRIPTION:
{description}

STATED JURISDICTIONS: {', '.join(jurisdictions)}
STATED SECTOR: {sector or 'not specified'}"""

    response = await llm.ainvoke([
        SystemMessage(content=INTAKE_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    # Track cost
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        await track_llm_cost(
            session_id, "intake", "claude-sonnet-4-20250514",
            usage.get("input_tokens", 0), usage.get("output_tokens", 0),
        )

    raw = _extract_json(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Intake agent returned invalid JSON: %s", e)
        raise ValueError(f"Intake agent returned invalid JSON: {e}") from e

    try:
        return DeploymentProfile(**parsed)
    except ValidationError as e:
        logger.error("Intake agent output failed validation: %s", e)
        raise ValueError(f"Intake agent output failed validation: {e}") from e
