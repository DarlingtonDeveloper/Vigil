"""
Layer 8: Audit Logging.
Wraps agent actions with audit trail. Every decision logged to SurrealDB.
"""
import time
from functools import wraps
from typing import Callable

from app.db.queries import log_audit


async def audit_wrap(
    session_id: str,
    agent: str,
    action: str,
    func: Callable,
    *args,
    **kwargs,
) -> any:
    """Execute a function and log input/output/latency to audit_log."""
    start = time.time()
    input_data = {
        "args": str(args)[:500],
        "kwargs": {k: str(v)[:200] for k, v in kwargs.items()},
    }

    try:
        result = await func(*args, **kwargs)
        latency = int((time.time() - start) * 1000)

        await log_audit(
            session_id=session_id,
            agent=agent,
            action=action,
            input_data=input_data,
            output_data={"result": str(result)[:1000]},
            latency_ms=latency,
        )
        return result
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        await log_audit(
            session_id=session_id,
            agent=agent,
            action="error",
            input_data=input_data,
            output_data={"error": str(e)[:500]},
            latency_ms=latency,
        )
        raise


def audited(agent: str, action: str):
    """Decorator for agent functions that auto-logs to audit trail."""

    def decorator(func):
        @wraps(func)
        async def wrapper(session_id: str, *args, **kwargs):
            return await audit_wrap(
                session_id, agent, action, func, session_id, *args, **kwargs
            )

        return wrapper

    return decorator
