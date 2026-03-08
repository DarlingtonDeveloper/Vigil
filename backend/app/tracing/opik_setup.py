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
    if not settings.opik_api_key:
        raise ValueError("OPIK_API_KEY not set — skipping Opik initialization")
    opik.configure(
        api_key=settings.opik_api_key,
        workspace=settings.opik_workspace,
        force=True,
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
    Wrap a compiled LangGraph with Opik tracing.
    Returns the graph unchanged — tracing is handled via OpikTracer callbacks
    passed at invocation time rather than by wrapping the graph.
    """
    return compiled_graph
