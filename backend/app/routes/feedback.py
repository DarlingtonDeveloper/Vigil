"""
Feedback endpoint. Users submit feedback on assessment quality.
"""
from fastapi import APIRouter, Depends

from app.middleware.auth import get_current_user
from app.middleware.validation import FeedbackRequest
from app.db.client import db

router = APIRouter(prefix="/api", tags=["feedback"])


@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    user: dict = Depends(get_current_user),
):
    """Submit feedback on an assessment session."""
    score = request.score
    if score is None:
        score = 1.0 if request.feedback_type == "thumbs_up" else 0.0

    await db.query(
        """
        CREATE evaluation SET
            session_id = $sid, score = $score,
            feedback_type = $type, feedback_detail = $detail,
            evaluator = 'user', timestamp = time::now()
    """,
        {
            "sid": request.session_id,
            "score": score,
            "type": request.feedback_type,
            "detail": request.detail,
        },
    )

    return {"status": "ok", "session_id": request.session_id}


@router.get("/feedback/{session_id}")
async def get_feedback(
    session_id: str,
    user: dict = Depends(get_current_user),
):
    """Get all feedback for a session."""
    result = await db.query(
        "SELECT * FROM evaluation WHERE session_id = $sid ORDER BY timestamp DESC",
        {"sid": session_id},
    )
    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            return first["result"]
    return result or []
