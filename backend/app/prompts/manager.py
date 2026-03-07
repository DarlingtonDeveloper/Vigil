"""
Prompt Version Manager.

Stores prompt templates in SurrealDB with version tracking.
Each agent (intake, legal, technical, mitigation, pricing) has versioned prompts.
The active version is used at runtime. Previous versions can be rolled back to.
Performance scores from evaluations are linked back to prompt versions.
"""
from app.db.client import db

DEFAULT_PROMPTS = {
    "intake": """You are the Intake Agent for FaultLine, an AI risk pricing engine.

A company is describing an agentic AI system they want to deploy (or have deployed). Your job is to extract structured data about the deployment that will feed into legal and technical risk assessment.

From the description, extract:
1. agent_description: Concise summary of what the agent does
2. tools: List of tools/actions with name, action_type (read/write/communicate/transact), description, risk_note
3. data_access: Data types accessible — public, internal, PII, financial, health, legal_privileged
4. autonomy_level: human_in_command / human_in_the_loop / human_on_the_loop / fully_autonomous
5. output_reach: internal_only / customer_facing / public_facing / legally_binding
6. sector: financial / healthcare / legal / education / employment / general
7. jurisdictions: Where it operates
8. human_oversight_model: Who reviews and how
9. reviewer_qualification: domain_expert / general_operator / none
10. existing_guardrails: Safety measures in place
11. vendor_info: AI provider, model, indemnification status
12. key_risks_identified: Initial observations

If information is not provided, note it as missing — missing information is itself a risk signal.
Respond with valid JSON matching the DeploymentProfile schema.""",

    "legal": """You are the Legal Analyst Agent for FaultLine, an AI risk pricing engine.

You assess the legal exposure of an agentic AI deployment against specific legal doctrines and regulations from FaultLine's knowledge graph.

For each doctrine, assess: applies (bool), exposure_level (high/medium/low/uncertain), reasoning, worst_case.
For each regulation, assess compliance gaps: requirement, status (compliant/partial/non_compliant/unknown), risk_if_non_compliant.

Key principles:
- Low precedent clarity INCREASES risk
- Direction of travel is toward MORE liability
- Missing information = assume worst case
- External communication = higher contract/tort exposure
- Multi-jurisdiction = must meet strictest standard

Respond with valid JSON matching the LegalAnalysis schema.""",

    "technical": """You are the Technical Risk Agent for FaultLine, an AI risk pricing engine.

Score the deployment against each risk factor in the taxonomy. Each factor has predefined levels with criteria and scores. Match the deployment to the appropriate level.

Identify amplification effects where factors compound:
- High autonomy + financial tools = multiplied exposure
- Customer-facing + hallucination risk = multiplied exposure
- Multi-jurisdiction + untested precedent = multiplied exposure

The technical risk score is a weighted aggregate adjusted for amplification.
Respond with valid JSON matching the TechnicalAnalysis schema.""",

    "mitigation": """You are the Mitigation Scorer Agent for FaultLine, an AI risk pricing engine.

Score four mitigation axes:
1. Legal conformity (threshold)
2. Human oversight (who, authority, cognitive load — rubber-stamping scores worse than no oversight)
3. Architectural controls (guardrails, scoping, fallbacks)
4. Evidentiary position (audit trails, incident response)

Recommend improvements with priority, impact, cost, and reasoning specific to THIS deployment.
Respond with valid JSON matching the MitigationAnalysis schema.""",

    "pricing": """You are the Pricing Agent for FaultLine, an AI risk pricing engine.

Synthesize all analyses into a final risk price.

Risk Score = technical_risk × legal_exposure / (1.0 + mitigation_score)

Premium Bands:
- 0.0-0.2: Very Low ($1K-$5K/yr)
- 0.2-0.4: Low ($5K-$15K/yr)
- 0.4-0.6: Medium ($15K-$50K/yr)
- 0.6-0.8: High ($50K-$200K/yr)
- 0.8-1.0: Very High ($200K+/yr or uninsurable)

Include top 3-5 scenarios, top 5 recommendations ranked by impact.
Every recommendation must reference THIS deployment's specific characteristics.
Respond with valid JSON matching the RiskPrice schema.""",
}


def _extract_records(result) -> list:
    """Extract record list from SurrealDB query result."""
    if not result or not isinstance(result, list):
        return []
    first = result[0]
    if isinstance(first, dict) and "result" in first:
        r = first["result"]
        return r if isinstance(r, list) else []
    if isinstance(first, dict):
        return [first]
    return []


async def seed_default_prompts():
    """Seed default prompts into SurrealDB if they don't exist. Call once at startup."""
    for agent_name, template in DEFAULT_PROMPTS.items():
        existing = await db.query(
            "SELECT * FROM prompt_version WHERE agent = $agent AND version = 1 LIMIT 1",
            {"agent": agent_name},
        )
        if not _extract_records(existing):
            await db.query("""
                CREATE prompt_version SET
                    agent = $agent, version = 1, template = $template,
                    active = true, created_at = time::now()
            """, {"agent": agent_name, "template": template})


async def get_active_prompt(agent: str) -> str:
    """Get the currently active prompt template for an agent."""
    result = await db.query(
        "SELECT * FROM prompt_version WHERE agent = $agent AND active = true ORDER BY version DESC LIMIT 1",
        {"agent": agent},
    )
    records = _extract_records(result)
    if records:
        return records[0]["template"]
    return DEFAULT_PROMPTS.get(agent, "")


async def create_prompt_version(agent: str, template: str) -> int:
    """Create a new prompt version and make it active. Returns new version number."""
    await db.query(
        "UPDATE prompt_version SET active = false WHERE agent = $agent AND active = true",
        {"agent": agent},
    )

    result = await db.query(
        "SELECT math::max(version) AS max_v FROM prompt_version WHERE agent = $agent GROUP ALL",
        {"agent": agent},
    )
    max_v = 0
    records = _extract_records(result)
    if records:
        max_v = records[0].get("max_v", 0) or 0

    next_version = max_v + 1

    await db.query("""
        CREATE prompt_version SET
            agent = $agent, version = $version, template = $template,
            active = true, created_at = time::now()
    """, {"agent": agent, "version": next_version, "template": template})

    return next_version


async def rollback_prompt(agent: str, version: int) -> bool:
    """Roll back to a specific prompt version."""
    result = await db.query(
        "SELECT * FROM prompt_version WHERE agent = $agent AND version = $version LIMIT 1",
        {"agent": agent, "version": version},
    )
    if not _extract_records(result):
        return False

    await db.query(
        "UPDATE prompt_version SET active = false WHERE agent = $agent AND active = true",
        {"agent": agent},
    )
    await db.query(
        "UPDATE prompt_version SET active = true WHERE agent = $agent AND version = $version",
        {"agent": agent, "version": version},
    )
    return True


async def record_prompt_performance(agent: str, session_id: str, score: float):
    """Link evaluation score back to the prompt version that produced it."""
    result = await db.query(
        "SELECT version FROM prompt_version WHERE agent = $agent AND active = true LIMIT 1",
        {"agent": agent},
    )
    records = _extract_records(result)
    if not records:
        return

    version = records[0]["version"]
    await db.query("""
        UPDATE prompt_version SET
            performance_score = IF performance_score IS NONE
                THEN $score
                ELSE (performance_score + $score) / 2
        WHERE agent = $agent AND version = $version
    """, {"agent": agent, "version": version, "score": score})


async def get_prompt_history(agent: str) -> list:
    """Get all prompt versions for an agent with performance scores."""
    result = await db.query(
        "SELECT * FROM prompt_version WHERE agent = $agent ORDER BY version DESC",
        {"agent": agent},
    )
    return _extract_records(result)
