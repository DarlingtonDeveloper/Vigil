"""
Opik initialization and LangGraph integration.

Two integration points:
1. track_langgraph — wraps the compiled graph for automatic trace capture
2. @opik.track — decorator for custom functions (SurrealDB writes, evals)

Opik is configured to use the Comet hosted platform.
"""
import opik
from opik.integrations.langchain import OpikTracer

from app.config import settings


def init_opik():
    """Initialize Opik. Call once at app startup."""
    opik.configure(
        api_key=settings.opik_api_key,
        workspace=settings.opik_workspace,
        project_name=settings.opik_project_name,
    )


def get_opik_tracer(session_id: str = "", tags: list[str] | None = None) -> OpikTracer:
    """
    Create an OpikTracer callback for LangChain/LangGraph.
    Pass this as a callback when invoking the graph or individual chains.
    """
    return OpikTracer(
        tags=tags or ["vigil"],
        metadata={
            "session_id": session_id,
            "project": "vigil",
        },
    )


def track_graph(compiled_graph):
    """
    Wrap a compiled LangGraph with Opik tracing using track_langgraph.
    All subsequent invocations are automatically traced.
    """
    from opik.integrations.langgraph import track_langgraph
    return track_langgraph(compiled_graph)
