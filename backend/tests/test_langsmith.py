from app.tracing.langsmith_setup import (
    get_langsmith_run_config,
    get_trace_url,
    verify_langsmith_config,
)


class TestVerifyLangsmithConfig:
    def test_ready_when_configured(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-test-key")
        monkeypatch.setenv("LANGCHAIN_PROJECT", "vigil")
        status = verify_langsmith_config()
        assert status["ready"] is True
        assert status["tracing_enabled"] is True
        assert status["api_key_set"] is True
        assert status["project"] == "vigil"

    def test_not_ready_without_api_key(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        status = verify_langsmith_config()
        assert status["ready"] is False
        assert status["api_key_set"] is False

    def test_not_ready_without_tracing(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        monkeypatch.setenv("LANGCHAIN_API_KEY", "ls-test-key")
        status = verify_langsmith_config()
        assert status["ready"] is False
        assert status["tracing_enabled"] is False

    def test_default_project_when_unset(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        status = verify_langsmith_config()
        assert status["project"] == "default"

    def test_warning_printed_when_not_ready(self, monkeypatch, capsys):
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
        verify_langsmith_config()
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "LangSmith tracing not fully configured" in captured.out


class TestGetLangsmithRunConfig:
    def test_returns_valid_config(self):
        config = get_langsmith_run_config("abc12345-session-id")
        assert "metadata" in config
        assert "tags" in config
        assert "run_name" in config

    def test_metadata_contains_session_id(self):
        config = get_langsmith_run_config("sess-1234-abcd")
        assert config["metadata"]["session_id"] == "sess-1234-abcd"
        assert config["metadata"]["project"] == "vigil"

    def test_agent_name_in_config(self):
        config = get_langsmith_run_config("sess-1234", agent_name="enricher")
        assert config["metadata"]["agent"] == "enricher"
        assert "enricher" in config["tags"]
        assert "vigil" in config["tags"]
        assert "enricher" in config["run_name"]

    def test_default_agent_name_empty(self):
        config = get_langsmith_run_config("sess-1234")
        assert config["metadata"]["agent"] == ""
        assert config["tags"] == ["vigil"]
        assert "workflow" in config["run_name"]

    def test_run_name_uses_session_prefix(self):
        config = get_langsmith_run_config("abcdefgh-1234-5678", agent_name="legal")
        assert config["run_name"] == "vigil-legal-abcdefgh"


class TestGetTraceUrl:
    def test_generates_url(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_PROJECT", "vigil")
        url = get_trace_url("run-123")
        assert url == "https://smith.langchain.com/o/default/projects/p/vigil/r/run-123"

    def test_uses_default_project(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_PROJECT", raising=False)
        url = get_trace_url("run-456")
        assert url == "https://smith.langchain.com/o/default/projects/p/default/r/run-456"


class TestStartupIntegration:
    def test_health_with_lifespan(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
