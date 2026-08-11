from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8004
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_employee"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "changeme"
    VECTOR_DB_URL: str = "redis://localhost:6379/1"
    EMBEDDING_DIMENSION: int = 768
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "rag"

    EMBEDDING_PROVIDER: str = "mock"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"
    EMBEDDING_TIMEOUT: float = 30.0

    # ── Agentic query refinement ──
    RAG_REFINE_ENABLED: bool = False
    RAG_REFINE_LLM: str = "mock"
    RAG_REFINE_MODEL: str = "qwen3:8b"
    RAG_REFINE_TIMEOUT: float = 30.0

    # ── Cross-lingual retrieval ──
    RAG_TRANSLATE_URL: str = ""
    RAG_TRANSLATE_API_KEY: str = ""
    RAG_TRANSLATE_TIMEOUT: float = 10.0

    # ── Usage / Cost tracking ──
    USAGE_ENABLED: bool = True
    USAGE_PRICING: str = "{}"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
