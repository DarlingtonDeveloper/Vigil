"""
Database query helpers.

Provides reusable query functions for audit logging and other common operations.
"""
from app.db.client import db


async def log_audit(
    session_id: str,
    agent: str,
    action: str,
    token_usage: dict | None = None,
    cost_usd: float = 0.0,
    details: dict | None = None,
):
    """Write an entry to the audit_log table."""
    await db.query("""
        CREATE audit_log SET
            session_id = $session_id,
            agent = $agent,
            action = $action,
            token_usage = $token_usage,
            cost_usd = $cost_usd,
            details = $details,
            created_at = time::now()
    """, {
        "session_id": session_id,
        "agent": agent,
        "action": action,
        "token_usage": token_usage or {},
        "cost_usd": cost_usd,
        "details": details or {},
    })
