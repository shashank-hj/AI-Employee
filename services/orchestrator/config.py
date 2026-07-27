from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8001
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_employee"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "changeme"
    GATEWAY_URL: str = "http://gateway:8000"
    TOOL_REGISTRY_URL: str = "http://tool-registry:8002"
    MEMORY_URL: str = "http://memory:8003"
    RAG_URL: str = "http://localhost:8004"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "orchestrator"
    APP_VERSION: str = "0.1.0"

    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_MODEL: str = "sarvam-105b"
    SARVAM_TIMEOUT: float = 30.0
    SARVAM_MAX_RETRIES: int = 3
    LLM_FALLBACK_INTENT: str = "general"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_CLASSIFY_MAX_TOKENS: int = 512

    RAG_QUERY_PATH: str = "/api/v1/documents/query"
    RAG_HEALTH_PATH: str = "/health"
    RAG_TIMEOUT: float = 5.0

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
