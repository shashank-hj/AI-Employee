"""Redis-backed rate limiter (O4)."""

import structlog

logger = structlog.get_logger(__name__)


class RedisLike:
    async def incr(self, key: str) -> int: ...

    async def expire(self, key: str, seconds: int) -> bool: ...

    async def get(self, key: str) -> str | None: ...


class RedisRateLimiter:
    """Fixed-window rate limiter using INCR + EXPIRE.

    A single counter per ``scope`` per window. ``allowed()`` returns True while the
    current window counter is at or below ``limit``. Failures degrade open (allow)
    so a Redis outage never blocks traffic.
    """

    def __init__(
        self,
        redis: RedisLike,
        default_limit: int = 30,
        window_seconds: int = 60,
        enabled: bool = True,
        prefix: str = "ratelimit",
    ) -> None:
        self._redis = redis
        self._default_limit = default_limit
        self._window = window_seconds
        self._enabled = enabled
        self._prefix = prefix

    def _window_key(self, scope: str, window_seconds: int) -> str:
        import time

        return f"{self._prefix}:{scope}:{int(time.time()) // window_seconds}"

    async def allowed(
        self,
        scope: str,
        limit: int | None = None,
        window_seconds: int | None = None,
    ) -> bool:
        if not self._enabled:
            return True
        limit = limit or self._default_limit
        window = window_seconds or self._window
        key = self._window_key(scope, window)
        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, window)
            return count <= limit
        except Exception as exc:
            logger.warning("rate_limiter_error", error=str(exc), scope=scope)
            return True

    async def current(self, scope: str, window_seconds: int | None = None) -> int:
        if not self._enabled:
            return 0
        window = window_seconds or self._window
        key = self._window_key(scope, window)
        try:
            value = await self._redis.get(key)
            return int(value) if value else 0
        except Exception:
            return 0
