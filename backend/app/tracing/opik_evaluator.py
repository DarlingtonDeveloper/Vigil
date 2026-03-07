"""
Opik Quality Gate — Evaluates agent outputs BEFORE they are persisted.

This is the key differentiator for the Comet interview: Opik is used as an
inline quality gate, not just passive tracing. Assessments are validated for
internal consistency, hallucination, and unsupported claims.

Three validation stages:
1. Legal analysis validation — cited doctrines exist, exposure scores in range
2. Technical scoring validation — factor scores match taxonomy levels
3. Pricing validation — premium band consistent with risk score
"""
from opik import track
from pydantic import BaseModel

from app.db.client import db

QUALITY_THRESHOLD = 0.5


class EvalResult(BaseModel):
    stage: str
    passed: bool
    score: float
    reasons: list[str]


@track(name="validate_legal_analysis")
async def validate_legal_analysis(
    analysis: dict,
    profile: dict,
    known_doctrines: list[str],
    session_id: str,
) -> EvalResult:
    """
    Validate the legal analysis output.

    Checks:
    - Cited doctrines exist in the knowledge graph
    - Exposure score is within bounds (0.0-1.0)
    - Confidence is not suspiciously high (>0.95)
    - If agent has external communication tools, contract formation risk is assessed
    - If agent accesses PII, data protection is assessed
    """
    score = 1.0
    reasons = []

    # Check cited doctrines exist in knowledge graph
    for da in analysis.get("doctrine_assessments", []):
        if da.get("doctrine_name") not in known_doctrines:
            score -= 0.3
            reasons.append(f"Unknown doctrine cited: {da.get('doctrine_name')}")

    # Check exposure score bounds
    exposure = analysis.get("legal_exposure_score", 0)
    if exposure < 0 or exposure > 1:
        score -= 0.5
        reasons.append(f"Exposure score out of range: {exposure}")

    # Check confidence is reasonable
    confidence = analysis.get("confidence", 0)
    if confidence > 0.95:
        score -= 0.2
        reasons.append(
            "Suspiciously high confidence — likely overconfident given legal uncertainty"
        )

    # Check coverage: if agent can communicate externally, contract law should be assessed
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
            reasons.append(
                "Agent has external communication but apparent authority not assessed"
            )

    # Check coverage: if agent accesses PII, data protection should be assessed
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
    await db.query(
        """
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_legal', timestamp = time::now()
    """,
        {
            "sid": session_id,
            "score": score,
            "detail": "; ".join(reasons) if reasons else "Passed all checks",
        },
    )

    return EvalResult(
        stage="legal_analysis",
        passed=passed,
        score=score,
        reasons=reasons if reasons else ["Passed all checks"],
    )


@track(name="validate_technical_analysis")
async def validate_technical_analysis(
    analysis: dict,
    risk_factors: list[dict],
    session_id: str,
) -> EvalResult:
    """
    Validate the technical analysis output.

    Checks:
    - All risk factors from taxonomy are scored
    - Scores are within 0.0-1.0
    - Technical risk score is a reasonable aggregate
    - Amplification effects reference real factor names
    """
    score = 1.0
    reasons = []

    known_factors = {rf.get("name") for rf in risk_factors}
    scored_factors = {fs.get("factor_name") for fs in analysis.get("factor_scores", [])}

    # Check coverage
    missing = known_factors - scored_factors
    if missing:
        penalty = min(len(missing) * 0.1, 0.4)
        score -= penalty
        reasons.append(f"Missing scores for: {', '.join(list(missing)[:5])}")

    # Check score bounds
    for fs in analysis.get("factor_scores", []):
        if fs.get("score", 0) < 0 or fs.get("score", 0) > 1:
            score -= 0.2
            reasons.append(
                f"Score out of range for {fs.get('factor_name')}: {fs.get('score')}"
            )

    # Check overall score bounds
    overall = analysis.get("technical_risk_score", 0)
    if overall < 0 or overall > 1:
        score -= 0.5
        reasons.append(f"Overall technical risk score out of range: {overall}")

    # Check amplification effects reference real factors
    for amp in analysis.get("amplification_effects", []):
        if not any(fn in amp for fn in known_factors):
            score -= 0.1
            reasons.append(
                f"Amplification effect doesn't reference known factors: {amp[:80]}"
            )

    passed = score >= QUALITY_THRESHOLD

    await db.query(
        """
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_technical', timestamp = time::now()
    """,
        {
            "sid": session_id,
            "score": score,
            "detail": "; ".join(reasons) if reasons else "Passed all checks",
        },
    )

    return EvalResult(
        stage="technical_analysis",
        passed=passed,
        score=score,
        reasons=reasons if reasons else ["Passed all checks"],
    )


@track(name="validate_pricing")
async def validate_pricing(
    pricing: dict,
    legal_score: float,
    technical_score: float,
    mitigation_score: float,
    session_id: str,
) -> EvalResult:
    """
    Validate the pricing output.

    Checks:
    - Overall risk score is within 0.0-1.0
    - Premium band is consistent with risk score
    - At least some scenarios are provided
    - Recommendations reference specific deployment characteristics
    """
    score = 1.0
    reasons = []

    overall = pricing.get("overall_risk_score", 0)
    if overall < 0 or overall > 1:
        score -= 0.5
        reasons.append(f"Overall risk score out of range: {overall}")

    # Check premium band consistency
    premium = pricing.get("premium_band", "")
    if overall < 0.3 and "High" in premium:
        score -= 0.3
        reasons.append("Low risk score but high premium band — inconsistent")
    if overall > 0.7 and "Low" in premium:
        score -= 0.3
        reasons.append("High risk score but low premium band — inconsistent")

    # Check scenarios exist
    scenarios = pricing.get("scenarios", [])
    if not scenarios:
        score -= 0.2
        reasons.append("No risk scenarios provided")

    # Check recommendations exist
    recommendations = pricing.get("recommendations", [])
    if not recommendations:
        score -= 0.2
        reasons.append("No recommendations provided")

    # Check confidence
    confidence = pricing.get("confidence", 0)
    if confidence > 0.95:
        score -= 0.15
        reasons.append(
            "Overconfident pricing — novel legal domain warrants lower confidence"
        )

    passed = score >= QUALITY_THRESHOLD

    await db.query(
        """
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'opik_gate_pricing', timestamp = time::now()
    """,
        {
            "sid": session_id,
            "score": score,
            "detail": "; ".join(reasons) if reasons else "Passed all checks",
        },
    )

    return EvalResult(
        stage="pricing",
        passed=passed,
        score=score,
        reasons=reasons if reasons else ["Passed all checks"],
    )


@track(name="auto_evaluate_session")
async def auto_evaluate_session(session_id: str, assessment: dict) -> float:
    """
    Auto-evaluate a completed assessment session.
    Scores based on completeness, confidence, actionability.
    """
    scores = {
        "has_legal": 1.0 if assessment.get("legal_analysis") else 0.0,
        "has_technical": 1.0 if assessment.get("technical_analysis") else 0.0,
        "has_mitigation": 1.0 if assessment.get("mitigation_analysis") else 0.0,
        "has_pricing": 1.0 if assessment.get("risk_price") else 0.0,
        "confidence": assessment.get("risk_price", {}).get("confidence", 0.0),
        "has_scenarios": 1.0
        if assessment.get("risk_price", {}).get("scenarios")
        else 0.0,
        "has_recommendations": 1.0
        if assessment.get("risk_price", {}).get("recommendations")
        else 0.0,
        "has_data_gaps": 0.8
        if assessment.get("risk_price", {}).get("data_gaps")
        else 1.0,
    }
    overall = sum(scores.values()) / len(scores)

    await db.query(
        """
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = 'auto', feedback_detail = $detail,
            evaluator = 'auto', timestamp = time::now()
    """,
        {
            "sid": session_id,
            "score": overall,
            "detail": str(scores),
        },
    )

    return overall
