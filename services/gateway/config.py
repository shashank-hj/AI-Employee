from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_employee"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "changeme"
    ORCHESTRATOR_URL: str = "http://orchestrator:8001"
    TOOL_REGISTRY_URL: str = "http://tool-registry:8002"
    MEMORY_URL: str = "http://memory:8003"
    RAG_URL: str = "http://rag:8004"
    WORKFLOW_URL: str = "http://workflow:8005"
    SPEECH_URL: str = "http://speech:8006"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "gateway"

    # ── Edge guardrails (O4) ──
    GUARDRAILS_ENABLED: bool = True
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_LIMIT: int = 30
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # ── Channel event recording (dashboard) ──
    CHANNEL_EVENTS_ENABLED: bool = True

    # ── Channel forwarding ──
    CHANNEL_TIMEOUT_SECONDS: float = 300.0

    # ── Integrated dashboard (chat lives inside the dashboard now) ──
    # Browser-reachable URL for the dashboard; defaults to the localhost
    # orchestrator port (which is also what dashboard.html hardcodes).
    DASHBOARD_PUBLIC_URL: str = "http://localhost:8001/dashboard"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
