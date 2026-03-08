"""
Prompt Version Manager.

Stores prompt templates in SurrealDB with version tracking.
Syncs to Opik prompt library for dashboard visibility and A/B testing.
Each agent (intake, legal, technical, mitigation, pricing) has versioned prompts.
The active version is used at runtime. Previous versions can be rolled back to.
Performance scores from evaluations are linked back to prompt versions.
"""
import logging

import opik

from app.db.client import db

logger = logging.getLogger(__name__)


def _get_default_prompts() -> dict[str, str]:
    """Import the canonical system prompts from each agent module."""
    from app.agents.intake import INTAKE_SYSTEM_PROMPT
    from app.agents.legal import LEGAL_SYSTEM_PROMPT
    from app.agents.technical import TECHNICAL_SYSTEM_PROMPT
    from app.agents.mitigation import MITIGATION_SYSTEM_PROMPT
    from app.agents.pricing import PRICING_SYSTEM_PROMPT

    return {
        "intake": INTAKE_SYSTEM_PROMPT,
        "legal": LEGAL_SYSTEM_PROMPT,
        "technical": TECHNICAL_SYSTEM_PROMPT,
        "mitigation": MITIGATION_SYSTEM_PROMPT,
        "pricing": PRICING_SYSTEM_PROMPT,
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


def _sync_prompts_to_opik():
    """Sync all agent prompts to Opik prompt library for versioning and dashboard visibility."""
    try:
        client = opik.Opik()
        for agent_name, template in _get_default_prompts().items():
            prompt_name = f"vigil-{agent_name}"
            try:
                existing = client.get_prompt(name=prompt_name)
                if existing and existing.prompt == template:
                    continue  # Already up to date
            except Exception:
                pass  # Prompt doesn't exist yet
            client.create_prompt(
                name=prompt_name,
                prompt=template,
                metadata={"agent": agent_name, "project": "vigil"},
            )
            logger.info("Synced prompt to Opik: %s", prompt_name)
    except Exception as e:
        logger.warning("Failed to sync prompts to Opik (non-blocking): %s", e)


async def seed_default_prompts():
    """Seed default prompts into SurrealDB and Opik prompt library."""
    for agent_name, template in _get_default_prompts().items():
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

    # Sync all prompts to Opik prompt library
    _sync_prompts_to_opik()


async def get_active_prompt(agent: str) -> str:
    """Get the currently active prompt template for an agent."""
    result = await db.query(
        "SELECT * FROM prompt_version WHERE agent = $agent AND active = true ORDER BY version DESC LIMIT 1",
        {"agent": agent},
    )
    records = _extract_records(result)
    if records:
        return records[0]["template"]
    return _get_default_prompts().get(agent, "")


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

    # Sync new version to Opik
    try:
        client = opik.Opik()
        client.create_prompt(
            name=f"vigil-{agent}",
            prompt=template,
            metadata={"agent": agent, "version": next_version, "project": "vigil"},
        )
    except Exception as e:
        logger.warning("Failed to sync prompt v%d to Opik: %s", next_version, e)

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
