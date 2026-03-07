from app.config import settings


def get_langsmith_run_config(session_id: str, run_name: str) -> dict:
    """Build a LangSmith-compatible run config dict for LangGraph invocations."""
    return {
        "run_name": f"{run_name}:{session_id[:8]}",
        "metadata": {
            "session_id": session_id,
            "project": settings.langchain_project,
        },
        "tags": [run_name, "vigil"],
    }
