"""
Knowledge graph query endpoints.
Returns legal doctrine, regulations, risk factors, mitigations for the frontend.
"""
from fastapi import APIRouter, Depends, Query

from app.middleware.auth import get_current_user
from app.db.client import db
from app.db.queries import get_knowledge_stats

router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.get("/full")
async def get_full_knowledge_graph(user: dict = Depends(get_current_user)):
    """Return the complete knowledge graph for visualization."""
    return await db.get_knowledge_graph_full()


@router.get("/stats")
async def get_stats(user: dict = Depends(get_current_user)):
    """Summary statistics for the knowledge graph."""
    return await get_knowledge_stats()


@router.get("/doctrines")
async def get_doctrines(
    jurisdiction: str = Query(default="UK"),
    user: dict = Depends(get_current_user),
):
    """Get legal doctrines for a jurisdiction."""
    return await db.get_applicable_doctrines(
        [jurisdiction, "global"],
        [
            "contract_law",
            "tort_law",
            "data_protection",
            "regulatory_compliance",
            "intellectual_property",
            "employment_law",
        ],
    )


@router.get("/regulations")
async def get_regulations(
    jurisdiction: str = Query(default="UK"),
    user: dict = Depends(get_current_user),
):
    """Get regulations in force for a jurisdiction."""
    return await db.get_applicable_regulations([jurisdiction, "global"])


@router.get("/risk-factors")
async def get_risk_factors(
    category: str = Query(default=None),
    user: dict = Depends(get_current_user),
):
    """Get risk factor taxonomy, optionally filtered by category."""
    if category:
        return await db.get_risk_factors_by_category(category)
    return await db.query("SELECT * FROM risk_factor ORDER BY weight DESC")


@router.get("/mitigations/{risk_factor}")
async def get_mitigations_for_risk(
    risk_factor: str,
    user: dict = Depends(get_current_user),
):
    """Get all mitigations that reduce a specific risk factor."""
    return await db.get_mitigations_for_risk(risk_factor)


@router.get("/doctrine/{doctrine_name}/relationships")
async def get_doctrine_relationships(
    doctrine_name: str,
    user: dict = Depends(get_current_user),
):
    """Get doctrines related to a specific doctrine."""
    return await db.get_doctrine_relationships(doctrine_name)


@router.get("/audit/{session_id}")
async def get_audit_trail(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Return the audit trail for a session — every decision the agent made."""
    result = await db.query(
        "SELECT * FROM audit_log WHERE session_id = $sid ORDER BY timestamp ASC",
        {"sid": session_id},
    )
    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            return first["result"]
    return result or []
