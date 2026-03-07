from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    # SurrealDB
    surreal_url: str = "ws://localhost:8000/rpc"
    surreal_user: str = "root"
    surreal_pass: str = "root"
    surreal_ns: str = "portfoliograph"
    surreal_db: str = "portfoliograph"

    # Anthropic
    anthropic_api_key: str = ""

    # Opik
    opik_api_key: str = ""
    opik_workspace: str = ""
    opik_project_name: str = "portfoliograph"

    # LangSmith
    langchain_tracing_v2: bool = True
    langchain_api_key: str = ""
    langchain_project: str = "portfoliograph"

    # Rate limiting
    rate_limit_requests_per_hour: int = 20
    rate_limit_tokens_per_hour: int = 100000


settings = Settings()
