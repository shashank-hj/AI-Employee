from pydantic_settings import BaseSettings
from functools import lru_cache


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

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
