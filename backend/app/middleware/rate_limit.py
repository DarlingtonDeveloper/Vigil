"""
Layer 4: Rate Limiting & Abuse Prevention.
Per-user rate limits backed by SurrealDB. Tracks request count and token budget per hour.
"""
from datetime import datetime, timedelta, timezone
from fastapi import HTTPException, Depends

from app.db.client import db
from app.config import settings
from app.middleware.auth import get_current_user


async def check_rate_limit(user: dict = Depends(get_current_user)) -> dict:
    """Check and increment rate limit. Returns user if within limits."""
    user_id = user["user_id"]
    window = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    window_iso = window.isoformat()

    result = await db.query(
        "SELECT * FROM rate_limit WHERE user_id = $uid AND window_start = $window LIMIT 1",
        {"uid": user_id, "window": window_iso},
    )

    records = []
    if result and isinstance(result, list):
        first = result[0]
        if isinstance(first, dict) and "result" in first:
            records = (
                first["result"] if isinstance(first["result"], list) else []
            )
        elif isinstance(first, dict) and "user_id" in first:
            records = [first]

    if not records:
        await db.query(
            """
            CREATE rate_limit SET
                user_id = $uid,
                window_start = $window,
                request_count = 1,
                token_budget_used = 0,
                token_budget_max = $max
            """,
            {
                "uid": user_id,
                "window": window_iso,
                "max": settings.rate_limit_tokens_per_hour,
            },
        )
        return user

    rl = records[0]

    if rl.get("request_count", 0) >= settings.rate_limit_requests_per_hour:
        reset_at = (window + timedelta(hours=1)).isoformat()
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "requests_used": rl["request_count"],
                "limit": settings.rate_limit_requests_per_hour,
                "resets_at": reset_at,
            },
        )

    if rl.get("token_budget_used", 0) >= rl.get(
        "token_budget_max", settings.rate_limit_tokens_per_hour
    ):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Token budget exceeded",
                "tokens_used": rl["token_budget_used"],
                "budget": rl["token_budget_max"],
            },
        )

    await db.query(
        "UPDATE rate_limit SET request_count += 1 WHERE user_id = $uid AND window_start = $window",
        {"uid": user_id, "window": window_iso},
    )

    return user


async def increment_token_usage(user_id: str, tokens: int):
    """Call after LLM response to track token consumption."""
    window = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    window_iso = window.isoformat()
    await db.query(
        "UPDATE rate_limit SET token_budget_used += $tokens WHERE user_id = $uid AND window_start = $window",
        {"uid": user_id, "tokens": tokens, "window": window_iso},
    )
