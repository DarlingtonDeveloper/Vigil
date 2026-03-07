"""
Cost Tracking (Dojo Gap 6).

Tracks token usage and cost per LLM invocation. Stores in SurrealDB.
Reports per-session cost and aggregate daily cost.

Claude Sonnet pricing (as of March 2025):
- Input: $3.00 / 1M tokens
- Output: $15.00 / 1M tokens
"""
from datetime import datetime, timezone

from app.db.client import db
from app.db.queries import log_audit

PRICING = {
    "claude-sonnet-4-20250514": {
        "input_per_token": 3.0 / 1_000_000,
        "output_per_token": 15.0 / 1_000_000,
    },
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a single LLM call."""
    prices = PRICING.get(model, PRICING["claude-sonnet-4-20250514"])
    return (
        input_tokens * prices["input_per_token"]
        + output_tokens * prices["output_per_token"]
    )


async def track_llm_cost(
    session_id: str,
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    user_id: str = "demo-user",
):
    """Record cost for a single LLM invocation. Updates audit log and daily summary."""
    cost = calculate_cost(model, input_tokens, output_tokens)
    total_tokens = input_tokens + output_tokens

    await log_audit(
        session_id=session_id,
        agent=agent,
        action="llm_call",
        token_usage={"input": input_tokens, "output": output_tokens, "total": total_tokens},
        cost_usd=cost,
    )

    period = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await db.query("""
        UPSERT cost_summary SET
            user_id = $uid, period = $period,
            total_tokens += $tokens, total_cost_usd += $cost,
            invocation_count += 1
        WHERE user_id = $uid AND period = $period
    """, {
        "uid": user_id,
        "period": period,
        "tokens": total_tokens,
        "cost": cost,
    })


async def get_session_cost(session_id: str) -> dict:
    """Get total cost for a specific assessment session."""
    result = await db.query("""
        SELECT
            math::sum(cost_usd) AS total_cost,
            math::sum(token_usage.total) AS total_tokens,
            count() AS llm_calls
        FROM audit_log
        WHERE session_id = $sid AND action = 'llm_call'
        GROUP ALL
    """, {"sid": session_id})

    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            records = first["result"]
            if isinstance(records, list) and records:
                return records[0]
    return {"total_cost": 0, "total_tokens": 0, "llm_calls": 0}


async def get_daily_cost(user_id: str, days: int = 7) -> list:
    """Get daily cost summary for last N days."""
    result = await db.query("""
        SELECT * FROM cost_summary
        WHERE user_id = $uid
        ORDER BY period DESC
        LIMIT $days
    """, {"uid": user_id, "days": days})

    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            return first["result"] if isinstance(first["result"], list) else []
    return result or []


async def get_average_cost_per_run(user_id: str) -> float:
    """Calculate average cost per assessment run. For ROI justification."""
    result = await db.query("""
        SELECT
            math::mean(total_cost_usd / invocation_count) AS avg_cost_per_run
        FROM cost_summary
        WHERE user_id = $uid
        GROUP ALL
    """, {"uid": user_id})

    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            records = first["result"]
            if isinstance(records, list) and records:
                return records[0].get("avg_cost_per_run", 0)
    return 0
