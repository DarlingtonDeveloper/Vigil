from app.db.client import db


async def log_audit(
    session_id: str,
    step: str,
    status: str,
    output_data: dict | None = None,
) -> None:
    """Write an audit trail entry for a workflow node."""
    await db.query(
        """
        CREATE audit_log SET
            session_id = $sid,
            step = $step,
            status = $status,
            output_data = $data,
            created_at = time::now()
        """,
        {
            "sid": session_id,
            "step": step,
            "status": status,
            "data": output_data,
        },
    )
