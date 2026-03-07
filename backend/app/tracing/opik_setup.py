from opik.integrations.langchain import OpikTracer

from app.config import settings


def get_opik_tracer(session_id: str, tags: list[str] | None = None) -> OpikTracer:
    """Create an Opik tracer callback for LangGraph runs."""
    return OpikTracer(
        project_name=settings.opik_project_name,
        tags=tags or [],
        metadata={"session_id": session_id},
    )
