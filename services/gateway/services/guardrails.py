"""Gateway-side wiring of edge guardrails (O4)."""

from functools import lru_cache

import redis.asyncio as aioredis
import structlog

from gateway.config import settings
from shared.guardrails import (
    ContentFilter,
    GuardrailsService,
    InputSanitizer,
    PIIRedactor,
    RedisRateLimiter,
)

logger = structlog.get_logger(__name__)


@lru_cache
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2.0,
        socket_timeout=2.0,
    )


@lru_cache
def get_guardrails_service() -> GuardrailsService:
    if not settings.GUARDRAILS_ENABLED:
        logger.info("guardrails_disabled")
        return GuardrailsService(enabled=False)
    return GuardrailsService(
        sanitizer=InputSanitizer(),
        redactor=PIIRedactor(),
        content_filter=ContentFilter(),
    )


@lru_cache
def get_rate_limiter() -> RedisRateLimiter:
    return RedisRateLimiter(
        redis=get_redis_client(),
        default_limit=settings.RATE_LIMIT_LIMIT,
        window_seconds=settings.RATE_LIMIT_WINDOW_SECONDS,
        enabled=settings.RATE_LIMIT_ENABLED,
    )
