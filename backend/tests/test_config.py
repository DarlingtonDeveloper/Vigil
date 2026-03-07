from app.config import Settings


def test_default_surreal_settings():
    s = Settings()
    assert s.surreal_url == "ws://localhost:8000/rpc"
    assert s.surreal_user == "root"
    assert s.surreal_pass == "root"
    assert s.surreal_ns == "faultline"
    assert s.surreal_db == "faultline"


def test_default_rate_limits():
    s = Settings()
    assert s.rate_limit_requests_per_hour == 20
    assert s.rate_limit_tokens_per_hour == 100000


def test_env_override(monkeypatch):
    monkeypatch.setenv("SURREAL_URL", "ws://custom:9999/rpc")
    monkeypatch.setenv("SURREAL_USER", "admin")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS_PER_HOUR", "50")
    s = Settings()
    assert s.surreal_url == "ws://custom:9999/rpc"
    assert s.surreal_user == "admin"
    assert s.rate_limit_requests_per_hour == 50


def test_anthropic_key_default_empty():
    s = Settings()
    assert s.anthropic_api_key == ""


def test_langchain_tracing_default_true():
    s = Settings()
    assert s.langchain_tracing_v2 is True
