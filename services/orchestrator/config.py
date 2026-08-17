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
    SPEECH_URL: str = "http://speech:8006"
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://otel-collector:4317"
    ENVIRONMENT: str = "development"
    SERVICE_NAME: str = "orchestrator"
    APP_VERSION: str = "0.1.0"

    LLM_PROVIDER: str = ""
    SARVAM_API_KEY: str = ""
    SARVAM_BASE_URL: str = "https://api.sarvam.ai"
    SARVAM_MODEL: str = "sarvam-105b"
    SARVAM_TIMEOUT: float = 30.0
    SARVAM_MAX_RETRIES: int = 3
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_TIMEOUT: float = 120.0
    LLM_FALLBACK_INTENT: str = "general"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 2048
    LLM_CLASSIFY_MAX_TOKENS: int = 512

    # ── opencode (opencode serve HTTP backend) ──
    OPENCODE_BASE_URL: str = "http://localhost:4096"
    OPENCODE_MODEL: str = ""
    OPENCODE_AGENT: str = "general"
    OPENCODE_PASSWORD: str = ""
    OPENCODE_USERNAME: str = "opencode"
    OPENCODE_TIMEOUT: float = 120.0
    OPENCODE_MAX_TOKENS: int = 2048

    RAG_QUERY_PATH: str = "/api/query"
    RAG_HEALTH_PATH: str = "/health"
    RAG_TIMEOUT: float = 5.0

    # ── Memory Writer (M5) ──
    MEMORY_WRITER_ENABLED: bool = True
    MEMORY_WRITER_QUEUE_KEY: str = "memory_writer:queue"
    MEMORY_WRITER_POLL_TIMEOUT: int = 5

    # ── Human-in-the-loop approval (C4) ──
    HITL_ENABLED: bool = True
    HITL_APPROVAL_TOOLS: str = '["send_email"]'

    # ── Usage / Cost tracking ──
    USAGE_ENABLED: bool = True
    USAGE_PRICING: str = "{}"

    # ── Email (SMTP + IMAP) ──
    EMAIL_ENABLED: bool = False
    EMAIL_ADDRESS: str = ""
    EMAIL_PASSWORD: str = ""
    EMAIL_SMTP_HOST: str = "smtp.gmail.com"
    EMAIL_SMTP_PORT: int = 587
    EMAIL_IMAP_HOST: str = "imap.gmail.com"
    EMAIL_IMAP_PORT: int = 993
    EMAIL_DISPLAY_NAME: str = "AI Employee"

    # ── Calendar (Google Calendar API + ICS fallback) ──
    # provider: auto | google | ics. auto picks Google when valid OAuth creds
    # are present, otherwise falls back to .ics invites via the Gmail SMTP stack.
    CALENDAR_ENABLED: bool = False
    CALENDAR_PROVIDER: str = "auto"
    CALENDAR_TIMEZONE: str = "Asia/Kolkata"
    CALENDAR_ICS_INVITES_ENABLED: bool = True
    CALENDAR_DUPLICATE_WINDOW_MINUTES: int = 15
    GOOGLE_CALENDAR_ID: str = "primary"
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REFRESH_TOKEN: str = ""
    GOOGLE_CALENDAR_SCOPES: str = "https://www.googleapis.com/auth/calendar"

    model_config = {
        "env_file": (".env", ".env-opencode"),
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
