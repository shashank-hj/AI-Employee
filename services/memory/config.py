from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8003
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_employee"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "changeme"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "memory"

    EMBEDDING_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://host.docker.internal:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    EMBEDDING_TIMEOUT: float = 30.0

    # ── Session lifecycle ──
    MEMORY_SESSION_TTL_SECONDS: int = 86400

    # ── Session summarization ──
    MEMORY_SUMMARY_MODE: str = "mock"
    MEMORY_SUMMARY_MODEL: str = "qwen3:8b"
    MEMORY_SUMMARY_TIMEOUT: float = 30.0
    SARVAM_API_KEY: str = ""
    SARVAM_MODEL: str = "sarvam-105b"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
