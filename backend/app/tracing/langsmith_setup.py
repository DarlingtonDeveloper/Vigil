"""
LangSmith integration for execution tracing.

LangSmith traces are enabled purely via environment variables.
LangChain auto-detects LANGCHAIN_TRACING_V2=true and sends traces.
No code changes needed in agents — it's automatic.

This module provides:
1. Verification that LangSmith is configured
2. A helper to add run metadata (session_id, agent name)
3. A function to generate trace URLs for the demo
"""

import os
from typing import Optional


def verify_langsmith_config() -> dict:
    """
    Check that LangSmith env vars are set.
    Call at startup to warn if tracing won't work.
    """
    tracing = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    api_key = bool(os.getenv("LANGCHAIN_API_KEY"))
    project = os.getenv("LANGCHAIN_PROJECT", "default")

    status = {
        "tracing_enabled": tracing,
        "api_key_set": api_key,
        "project": project,
        "ready": tracing and api_key,
    }

    if not status["ready"]:
        print(
            "WARNING: LangSmith tracing not fully configured. "
            "Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable."
        )

    return status


def get_langsmith_run_config(session_id: str, agent_name: str = "") -> dict:
    """
    Generate a LangChain run config with metadata for LangSmith.

    Pass this as the `config` parameter when invoking LangGraph:
        result = await graph.ainvoke(state, config=get_langsmith_run_config(session_id))

    Tags the trace with session_id and agent name for filtering in LangSmith.
    """
    return {
        "metadata": {
            "session_id": session_id,
            "agent": agent_name,
            "project": "vigil",
        },
        "tags": ["vigil", agent_name] if agent_name else ["vigil"],
        "run_name": f"vigil-{agent_name or 'workflow'}-{session_id[:8]}",
    }


def get_trace_url(run_id: str) -> Optional[str]:
    """
    Generate a LangSmith trace URL for a given run.
    Useful for including in API responses so the demo can link to traces.
    """
    project = os.getenv("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{run_id}"
