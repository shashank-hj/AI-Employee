"""Generic Redis-backed task queue (shared).

A durable FIFO queue backed by a Redis list (``RPUSH`` to enqueue, ``LPOP`` to
consume). Jobs are opaque dicts wrapped in a small envelope:
``{"job_id": ..., "task": <name>, "payload": {...}, "enqueued_at": ...}``.
"""

import json
import time
import uuid
from typing import Any, Protocol

import structlog

logger = structlog.get_logger(__name__)


class RedisLike(Protocol):
    async def rpush(self, key: str, value: str) -> int: ...

    async def lpop(self, key: str) -> str | None: ...

    async def llen(self, key: str) -> int: ...


class RedisTaskQueue:
    """FIFO job queue over a single Redis list."""

    def __init__(self, redis: RedisLike, queue_key: str = "task:queue") -> None:
        self._redis = redis
        self._queue_key = queue_key

    @property
    def queue_key(self) -> str:
        return self._queue_key

    async def enqueue(self, task: str, payload: dict[str, Any]) -> bool:
        envelope = {
            "job_id": str(uuid.uuid4()),
            "task": task,
            "payload": payload,
            "enqueued_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        try:
            await self._redis.rpush(self._queue_key, json.dumps(envelope, default=str))
            return True
        except Exception as exc:
            logger.error("task_enqueue_failed", error=str(exc), task=task, queue=self._queue_key)
            return False

    async def poll(self) -> dict[str, Any] | None:
        """Pop the oldest job, or return None when the queue is empty."""
        raw = await self._redis.lpop(self._queue_key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error(
                "task_bad_payload",
                error=str(exc),
                queue=self._queue_key,
                payload=raw[:200],
            )
            return None

    async def length(self) -> int:
        try:
            return await self._redis.llen(self._queue_key)
        except Exception:
            return 0
