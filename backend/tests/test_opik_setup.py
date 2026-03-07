"""Tests for Opik setup and tracer creation."""
from unittest.mock import patch, MagicMock

from app.tracing.opik_setup import init_opik, get_opik_tracer, track_graph


def test_init_opik_calls_configure():
    with patch("app.tracing.opik_setup.opik") as mock_opik:
        init_opik()
        mock_opik.configure.assert_called_once()
        call_kwargs = mock_opik.configure.call_args[1]
        assert "api_key" in call_kwargs
        assert "workspace" in call_kwargs
        assert "project_name" in call_kwargs


def test_init_opik_uses_settings_values():
    with patch("app.tracing.opik_setup.opik") as mock_opik, \
         patch("app.tracing.opik_setup.settings") as mock_settings:
        mock_settings.opik_api_key = "test-key"
        mock_settings.opik_workspace = "test-ws"
        mock_settings.opik_project_name = "test-proj"
        init_opik()
        mock_opik.configure.assert_called_once_with(
            api_key="test-key",
            workspace="test-ws",
            project_name="test-proj",
        )


def test_get_opik_tracer_returns_tracer():
    with patch("app.tracing.opik_setup.OpikTracer") as MockTracer:
        MockTracer.return_value = MagicMock()
        tracer = get_opik_tracer(session_id="sess-1")
        assert tracer is not None
        MockTracer.assert_called_once()


def test_get_opik_tracer_default_tags():
    with patch("app.tracing.opik_setup.OpikTracer") as MockTracer:
        MockTracer.return_value = MagicMock()
        get_opik_tracer()
        call_kwargs = MockTracer.call_args[1]
        assert call_kwargs["tags"] == ["faultline"]


def test_get_opik_tracer_custom_tags():
    with patch("app.tracing.opik_setup.OpikTracer") as MockTracer:
        MockTracer.return_value = MagicMock()
        get_opik_tracer(tags=["custom", "tags"])
        call_kwargs = MockTracer.call_args[1]
        assert call_kwargs["tags"] == ["custom", "tags"]


def test_get_opik_tracer_metadata_includes_session():
    with patch("app.tracing.opik_setup.OpikTracer") as MockTracer:
        MockTracer.return_value = MagicMock()
        get_opik_tracer(session_id="abc-123")
        call_kwargs = MockTracer.call_args[1]
        assert call_kwargs["metadata"]["session_id"] == "abc-123"
        assert call_kwargs["metadata"]["project"] == "faultline"


def test_track_graph_wraps_compiled_graph():
    mock_graph = MagicMock()
    mock_wrapped = MagicMock()
    with patch(
        "app.tracing.opik_setup.track_langgraph",
        create=True,
    ) as mock_track, patch.dict(
        "sys.modules",
        {"opik.integrations.langgraph": MagicMock(track_langgraph=None)},
    ):
        # Re-import to pick up the patched module
        with patch("opik.integrations.langgraph.track_langgraph", mock_track, create=True):
            mock_track.return_value = mock_wrapped
            # Call via a fresh import path
            from importlib import reload
            import app.tracing.opik_setup as mod
            # Patch the lazy import inside track_graph
            with patch.object(mod, "track_graph", wraps=mod.track_graph):
                with patch("opik.integrations.langgraph.track_langgraph", return_value=mock_wrapped):
                    result = mod.track_graph(mock_graph)
                    assert result is mock_wrapped
