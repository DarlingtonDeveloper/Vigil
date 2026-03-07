"""
Technical Risk Agent — Evaluates the deployment's technical risk factors
against the risk factor taxonomy in the knowledge graph.
"""
import json
import logging
from pydantic import BaseModel, ValidationError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from app.config import settings

logger = logging.getLogger(__name__)


class FactorScore(BaseModel):
    factor_name: str
    level: str                   # which level from the taxonomy
    score: float                 # 0.0-1.0
    reasoning: str
    missing_info: str | None = None  # what information would improve this assessment


class TechnicalAnalysis(BaseModel):
    technical_risk_score: float  # 0.0-1.0 weighted aggregate
    factor_scores: list[FactorScore]
    amplification_effects: list[str]  # where risk factors compound each other
    key_vulnerabilities: list[str]
    confidence: float


TECHNICAL_SYSTEM_PROMPT = """You are the Technical Risk Agent for FaultLine, an AI risk pricing engine.

You assess the technical risk of an agentic AI deployment by scoring it against a taxonomy of risk factors. Each factor has defined levels with criteria and scores.

For each risk factor provided, determine:
- Which level best matches this deployment (match against the criteria)
- The score (use the predefined score for that level)
- Your reasoning (why this level, not another)
- What information is missing that would improve accuracy

Also identify amplification effects — where risk factors compound:
- High autonomy + financial tools = multiplied exposure (scope for large autonomous transactions)
- Customer-facing output + hallucination risk = multiplied exposure (false claims to customers)
- Multi-jurisdiction + untested precedent = multiplied exposure (can't predict legal outcome in any jurisdiction)
- Broad scope + no confidence thresholds = scope creep risk compounds all other factors

The technical risk score is a weighted aggregate of all factor scores, adjusted for amplification.

Respond with valid JSON matching the TechnicalAnalysis schema."""


def _extract_json(content: str) -> str:
    """Extract JSON from LLM response, handling markdown code fences."""
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return content.strip()


async def run_technical_analysis(
    profile: dict,
    risk_factors: list[dict],
    session_id: str,
) -> TechnicalAnalysis:
    llm = ChatAnthropic(
        model="claude-sonnet-4-20250514",
        temperature=0,
        max_tokens=4096,
        api_key=settings.anthropic_api_key,
    )

    context = json.dumps({
        "deployment_profile": profile,
        "risk_factor_taxonomy": risk_factors,
    }, indent=2, default=str)

    user_msg = f"Score this deployment against the technical risk taxonomy:\n\n{context}"

    response = await llm.ainvoke([
        SystemMessage(content=TECHNICAL_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])

    raw = _extract_json(response.content)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Technical agent returned invalid JSON: %s", e)
        raise ValueError(f"Technical agent returned invalid JSON: {e}") from e

    try:
        return TechnicalAnalysis(**parsed)
    except ValidationError as e:
        logger.error("Technical agent output failed validation: %s", e)
        raise ValueError(f"Technical agent output failed validation: {e}") from e
