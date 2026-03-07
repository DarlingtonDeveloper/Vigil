"""
Assessment endpoint. Triggers the full LangGraph workflow.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.middleware.rate_limit import check_rate_limit
from app.middleware.validation import AssessDeploymentRequest
from app.graph.workflow import run_assessment

router = APIRouter(prefix="/api", tags=["assessment"])


@router.post("/assess")
async def assess_deployment(
    request: AssessDeploymentRequest,
    user: dict = Depends(check_rate_limit),
):
    """
    Assess an agentic deployment's legal and operational risk.

    Triggers the full LangGraph pipeline:
    Intake -> Fetch Knowledge -> Legal -> Technical -> Mitigation -> Pricing

    Returns the complete analysis including risk score, premium band,
    top exposures, scenarios, and recommendations.
    """
    try:
        result = await run_assessment(
            description=request.description,
            jurisdictions=request.jurisdictions,
            sector=request.sector,
            user_id=user["user_id"],
        )

        return {
            "session_id": result.get("session_id"),
            "status": "completed",
            "risk_price": result.get("risk_price"),
            "deployment_profile": result.get("deployment_profile"),
            "legal_analysis": result.get("legal_analysis"),
            "technical_analysis": result.get("technical_analysis"),
            "mitigation_analysis": result.get("mitigation_analysis"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")
