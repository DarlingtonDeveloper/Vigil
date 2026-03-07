"""
Scenario endpoint. Run ad-hoc risk scenarios against an existing assessment.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.middleware.rate_limit import check_rate_limit
from app.middleware.validation import RunRiskScenarioRequest
from app.db.client import db

router = APIRouter(prefix="/api", tags=["scenarios"])


@router.post("/scenario")
async def run_risk_scenario(
    request: RunRiskScenarioRequest,
    user: dict = Depends(check_rate_limit),
):
    """
    Run a specific risk scenario against an existing assessment.
    E.g., "What if the agent hallucinates a contract term?"
    """
    profile_result = await db.query(
        "SELECT * FROM deployment_profile WHERE session_id = $sid LIMIT 1",
        {"sid": request.session_id},
    )
    risk_result = await db.query(
        "SELECT * FROM risk_score WHERE session_id = $sid LIMIT 1",
        {"sid": request.session_id},
    )

    if not profile_result or not risk_result:
        raise HTTPException(
            status_code=400,
            detail="Assessment not found. Run /api/assess first.",
        )

    scenarios = await db.query(
        "SELECT * FROM risk_scenario WHERE session_id = $sid AND scenario_type = $type",
        {"sid": request.session_id, "type": request.scenario_type},
    )

    scenario_list = []
    if scenarios and isinstance(scenarios, list):
        first = scenarios[0]
        if isinstance(first, dict) and "result" in first:
            scenario_list = first["result"] if isinstance(first["result"], list) else []
        elif isinstance(first, dict) and "scenario_type" in first:
            scenario_list = scenarios

    return {
        "session_id": request.session_id,
        "scenario_type": request.scenario_type,
        "results": scenario_list,
    }
