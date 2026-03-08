"""
Opik Quality Gate — Evaluates agent outputs with both heuristic checks
and LLM-as-Judge metrics, then logs feedback scores to Opik traces.

Three layers:
1. Heuristic gates — fast inline checks (scores in range, schema valid, coverage)
2. LLM judges — Hallucination detection + custom Legal Precision eval
3. Feedback scoring — all scores logged to Opik traces for dashboard visibility
"""
import json
import logging
from opik import track, opik_context
from opik.evaluation.metrics import Hallucination, GEval
from pydantic import BaseModel

from app.db.client import db
logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.5

# ── LLM-as-Judge metrics (use Anthropic via LiteLLM) ──

_hallucination_metric = Hallucination(
    name="vigil_hallucination", track=False,
)

_legal_precision_metric = GEval(
    name="legal_precision",
    task_introduction=(
        "You are evaluating a legal risk analysis for an AI agent deployment. "
        "The analysis should cite specific legal doctrines, regulations, and case law "
        "rather than making vague or generic statements about legal risk."
    ),
    evaluation_criteria=(
        "Score the legal analysis on precision and specificity:\n"
        "- Does it cite specific doctrines by name (e.g., 'apparent authority', 'vicarious liability')?\n"
        "- Does it reference specific regulations (e.g., 'UK GDPR Article 6', 'EU AI Act Article 6')?\n"
        "- Are exposure scores justified with concrete reasoning?\n"
        "- Does it identify jurisdiction-specific risks rather than generic statements?\n"
        "- Are the regulatory gap findings actionable and specific?\n"
        "A score of 1.0 means highly precise and specific. A score of 0.0 means vague and generic."
    ),
    track=False,
)

_assessment_quality_metric = GEval(
    name="assessment_quality",
    task_introduction=(
        "You are evaluating the overall quality of an AI risk assessment. "
        "The assessment prices the legal and operational risk of deploying an autonomous AI agent."
    ),
    evaluation_criteria=(
        "Score the assessment on completeness and actionability:\n"
        "- Does the executive summary clearly state the risk level and key concerns?\n"
        "- Are the top exposures specific to this deployment (not generic)?\n"
        "- Do recommendations have clear, actionable steps?\n"
        "- Are risk scenarios realistic and well-reasoned?\n"
        "- Is the premium band justified by the analysis?\n"
        "A score of 1.0 means excellent quality. A score of 0.0 means poor quality."
    ),
    track=False,
)


class EvalResult(BaseModel):
    stage: str
    passed: bool
    score: float
    reasons: list[str]


# ── Feedback score helper ──

def _log_feedback(name: str, value: float, reason: str = ""):
    """Log a feedback score to the current Opik trace (if active)."""
    try:
        opik_context.update_current_trace(
            feedback_scores=[{"name": name, "value": value, "reason": reason}]
        )
    except Exception:
        pass  # No active trace — scores still go to SurrealDB


# ── Heuristic quality gates ──

@track(name="validate_legal_analysis")
async def validate_legal_analysis(
    analysis: dict,
    profile: dict,
    known_doctrines: list[str],
    session_id: str,
) -> EvalResult:
    score = 1.0
    reasons = []

    for da in analysis.get("doctrine_assessments", []):
        if da.get("doctrine_name") not in known_doctrines:
            score -= 0.3
            reasons.append(f"Unknown doctrine cited: {da.get('doctrine_name')}")

    exposure = analysis.get("legal_exposure_score", 0)
    if exposure < 0 or exposure > 1:
        score -= 0.5
        reasons.append(f"Exposure score out of range: {exposure}")

    confidence = analysis.get("confidence", 0)
    if confidence > 0.95:
        score -= 0.2
        reasons.append("Suspiciously high confidence — likely overconfident given legal uncertainty")

    tools = profile.get("tools", [])
    has_external_comms = any(
        t.get("action_type") in ["communicate", "transact"] for t in tools
    )
    if has_external_comms:
        doctrine_names = [
            da.get("doctrine_name") for da in analysis.get("doctrine_assessments", [])
        ]
        if "apparent_authority" not in doctrine_names:
            score -= 0.2
            reasons.append("Agent has external communication but apparent authority not assessed")

    data_access = profile.get("data_access", [])
    if any(d in ["PII", "pii", "health"] for d in data_access):
        has_gdpr = any(
            g.get("regulation") in ["GDPR", "General Data Protection Regulation"]
            for g in analysis.get("regulatory_gaps", [])
        )
        if not has_gdpr:
            score -= 0.15
            reasons.append("Agent accesses PII but GDPR compliance not assessed")

    passed = score >= QUALITY_THRESHOLD

    # Log to SurrealDB
    await db.query("""
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_legal', timestamp = time::now()
    """, {"sid": session_id, "score": score,
          "detail": "; ".join(reasons) if reasons else "Passed all checks"})

    # Log to Opik trace
    _log_feedback("legal_gate", score, "; ".join(reasons) if reasons else "Passed")

    return EvalResult(stage="legal_analysis", passed=passed, score=score,
                      reasons=reasons if reasons else ["Passed all checks"])


@track(name="validate_technical_analysis")
async def validate_technical_analysis(
    analysis: dict,
    risk_factors: list[dict],
    session_id: str,
) -> EvalResult:
    score = 1.0
    reasons = []

    known_factors = {rf.get("name") for rf in risk_factors}
    scored_factors = {fs.get("factor_name") for fs in analysis.get("factor_scores", [])}

    missing = known_factors - scored_factors
    if missing:
        penalty = min(len(missing) * 0.1, 0.4)
        score -= penalty
        reasons.append(f"Missing scores for: {', '.join(str(f) for f in list(missing)[:5])}")

    for fs in analysis.get("factor_scores", []):
        if fs.get("score", 0) < 0 or fs.get("score", 0) > 1:
            score -= 0.2
            reasons.append(f"Score out of range for {fs.get('factor_name')}: {fs.get('score')}")

    overall = analysis.get("technical_risk_score", 0)
    if overall < 0 or overall > 1:
        score -= 0.5
        reasons.append(f"Overall technical risk score out of range: {overall}")

    for amp in analysis.get("amplification_effects", []):
        if not any(fn in amp for fn in known_factors):
            score -= 0.1
            reasons.append(f"Amplification effect doesn't reference known factors: {amp[:80]}")

    passed = score >= QUALITY_THRESHOLD

    await db.query("""
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_technical', timestamp = time::now()
    """, {"sid": session_id, "score": score,
          "detail": "; ".join(reasons) if reasons else "Passed all checks"})

    _log_feedback("technical_gate", score, "; ".join(reasons) if reasons else "Passed")

    return EvalResult(stage="technical_analysis", passed=passed, score=score,
                      reasons=reasons if reasons else ["Passed all checks"])


@track(name="validate_pricing")
async def validate_pricing(
    pricing: dict,
    legal_score: float,
    technical_score: float,
    mitigation_score: float,
    session_id: str,
) -> EvalResult:
    score = 1.0
    reasons = []

    overall = pricing.get("overall_risk_score", 0)
    if overall < 0 or overall > 1:
        score -= 0.5
        reasons.append(f"Overall risk score out of range: {overall}")

    premium = pricing.get("premium_band", "")
    if overall < 0.3 and "High" in premium:
        score -= 0.3
        reasons.append("Low risk score but high premium band — inconsistent")
    if overall > 0.7 and "Low" in premium:
        score -= 0.3
        reasons.append("High risk score but low premium band — inconsistent")

    scenarios = pricing.get("scenarios", [])
    if not scenarios:
        score -= 0.2
        reasons.append("No risk scenarios provided")

    recommendations = pricing.get("recommendations", [])
    if not recommendations:
        score -= 0.2
        reasons.append("No recommendations provided")

    confidence = pricing.get("confidence", 0)
    if confidence > 0.95:
        score -= 0.15
        reasons.append("Overconfident pricing — novel legal domain warrants lower confidence")

    passed = score >= QUALITY_THRESHOLD

    await db.query("""
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_pricing', timestamp = time::now()
    """, {"sid": session_id, "score": score,
          "detail": "; ".join(reasons) if reasons else "Passed all checks"})

    _log_feedback("pricing_gate", score, "; ".join(reasons) if reasons else "Passed")

    return EvalResult(stage="pricing", passed=passed, score=score,
                      reasons=reasons if reasons else ["Passed all checks"])


# ── LLM-as-Judge evaluators ──

@track(name="judge_hallucination")
async def judge_hallucination(
    deployment_description: str,
    executive_summary: str,
    session_id: str,
) -> float:
    """
    Use Opik's Hallucination metric to check if the executive summary
    contains claims not grounded in the deployment description.
    Returns 0.0 (no hallucination) or 1.0 (hallucinated).
    """
    try:
        result = await _hallucination_metric.ascore(
            input=deployment_description,
            output=executive_summary,
            context=[deployment_description],
        )
        score = result.value
        _log_feedback("hallucination", score,
                      result.reason if hasattr(result, "reason") and result.reason else "")

        await db.query("""
            CREATE evaluation SET
                session_id = $sid, score = $score,
                feedback_type = 'llm_judge', feedback_detail = $detail,
                evaluator = 'opik_hallucination', timestamp = time::now()
        """, {"sid": session_id, "score": score,
              "detail": result.reason if hasattr(result, "reason") and result.reason else ""})

        return score
    except Exception as e:
        logger.warning("Hallucination judge failed: %s", e)
        return -1.0


@track(name="judge_legal_precision")
async def judge_legal_precision(
    legal_analysis: dict,
    session_id: str,
) -> float:
    """
    Custom GEval judge: does the legal analysis cite specific doctrines
    and regulations rather than being vague?
    Returns 0.0-1.0 (higher = more precise).
    """
    try:
        analysis_text = json.dumps(legal_analysis, indent=2, default=str)
        result = await _legal_precision_metric.ascore(
            input="Evaluate this legal risk analysis for precision and specificity.",
            output=analysis_text,
        )
        score = result.value
        _log_feedback("legal_precision", score,
                      result.reason if hasattr(result, "reason") and result.reason else "")

        await db.query("""
            CREATE evaluation SET
                session_id = $sid, score = $score,
                feedback_type = 'llm_judge', feedback_detail = $detail,
                evaluator = 'opik_legal_precision', timestamp = time::now()
        """, {"sid": session_id, "score": score,
              "detail": result.reason if hasattr(result, "reason") and result.reason else ""})

        return score
    except Exception as e:
        logger.warning("Legal precision judge failed: %s", e)
        return -1.0


@track(name="judge_assessment_quality")
async def judge_assessment_quality(
    pricing: dict,
    session_id: str,
) -> float:
    """
    Custom GEval judge: is the overall assessment complete, actionable,
    and well-reasoned?
    Returns 0.0-1.0 (higher = better quality).
    """
    try:
        pricing_text = json.dumps(pricing, indent=2, default=str)
        result = await _assessment_quality_metric.ascore(
            input="Evaluate this AI risk assessment for completeness and actionability.",
            output=pricing_text,
        )
        score = result.value
        _log_feedback("assessment_quality", score,
                      result.reason if hasattr(result, "reason") and result.reason else "")

        await db.query("""
            CREATE evaluation SET
                session_id = $sid, score = $score,
                feedback_type = 'llm_judge', feedback_detail = $detail,
                evaluator = 'opik_assessment_quality', timestamp = time::now()
        """, {"sid": session_id, "score": score,
              "detail": result.reason if hasattr(result, "reason") and result.reason else ""})

        return score
    except Exception as e:
        logger.warning("Assessment quality judge failed: %s", e)
        return -1.0


# ── Session-level auto-evaluation ──

@track(name="auto_evaluate_session")
async def auto_evaluate_session(session_id: str, assessment: dict) -> dict:
    """
    Auto-evaluate a completed assessment session.
    Runs heuristic completeness check + LLM judges in parallel-ish fashion.
    Returns a summary dict with all scores.
    """
    # Heuristic completeness scores
    completeness = {
        "has_legal": 1.0 if assessment.get("legal_analysis") else 0.0,
        "has_technical": 1.0 if assessment.get("technical_analysis") else 0.0,
        "has_mitigation": 1.0 if assessment.get("mitigation_analysis") else 0.0,
        "has_pricing": 1.0 if assessment.get("risk_price") else 0.0,
        "has_scenarios": 1.0 if assessment.get("risk_price", {}).get("scenarios") else 0.0,
        "has_recommendations": 1.0 if assessment.get("risk_price", {}).get("recommendations") else 0.0,
    }
    completeness_score = sum(completeness.values()) / len(completeness)

    # LLM judges (run on final output)
    pricing = assessment.get("risk_price", {})
    description = assessment.get("description", "")
    legal = assessment.get("legal_analysis", {})

    hallucination_score = await judge_hallucination(
        description, pricing.get("executive_summary", ""), session_id
    )
    legal_precision_score = await judge_legal_precision(legal, session_id)
    quality_score = await judge_assessment_quality(pricing, session_id)

    summary = {
        "completeness": completeness_score,
        "hallucination": hallucination_score,
        "legal_precision": legal_precision_score,
        "assessment_quality": quality_score,
        "heuristic_gates": {
            "legal": assessment.get("legal_quality_score", 0),
            "technical": assessment.get("technical_quality_score", 0),
            "pricing": assessment.get("pricing_quality_score", 0),
        },
    }

    # Overall composite (exclude failed judges)
    valid_scores = [completeness_score]
    if hallucination_score >= 0:
        valid_scores.append(1.0 - hallucination_score)  # Invert: 0=good for hallucination
    if legal_precision_score >= 0:
        valid_scores.append(legal_precision_score)
    if quality_score >= 0:
        valid_scores.append(quality_score)
    overall = sum(valid_scores) / len(valid_scores)

    _log_feedback("session_overall", overall,
                  f"completeness={completeness_score:.2f}, hallucination={hallucination_score:.2f}, "
                  f"legal_precision={legal_precision_score:.2f}, quality={quality_score:.2f}")

    await db.query("""
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'auto_session', timestamp = time::now()
    """, {"sid": session_id, "score": overall, "detail": json.dumps(summary, default=str)})

    return summary
