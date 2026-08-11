from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8006
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "speech"
    APP_VERSION: str = "0.1.0"

    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_TIMEOUT: float = 60.0
    SARVAM_STT_MODEL: str = "saaras:v3"
    SARVAM_TTS_SPEAKER: str = "anushka"
    SARVAM_TTS_LANGUAGE: str = "en-IN"

    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    SECRET_KEY: str = "changeme"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_employee"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Usage / Cost tracking ──
    USAGE_ENABLED: bool = True
    USAGE_PRICING: str = "{}"

    model_config = {"env_file": ".env", "case_sensitive": True, "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
