import json
import math
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Optional

import redis.asyncio as aioredis

from memory.schemas.session import SessionCreate, SessionResponse, SessionMessage


SESSION_KEY_PREFIX = "memory:session:"
DEFAULT_SESSION_TTL = 86400


class SessionStore:
    def __init__(self, redis_client: aioredis.Redis, ttl: int = DEFAULT_SESSION_TTL) -> None:
        self._redis = redis_client
        self._ttl = ttl

    def _key(self, session_id: str) -> str:
        return f"{SESSION_KEY_PREFIX}{session_id}"

    async def upsert(self, data: SessionCreate) -> SessionResponse:
        import uuid

        session_id = data.session_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        key = self._key(session_id)

        existing_raw = await self._redis.get(key)
        existing = json.loads(existing_raw) if existing_raw else {}
        created_at = existing.get("created_at") or now.isoformat()

        payload = {
            "session_id": session_id,
            "user_id": data.user_id,
            "messages": [m.model_dump(mode="json") for m in data.messages],
            "context": data.context,
            "metadata": data.metadata,
            "message_count": len(data.messages),
            "created_at": created_at,
            "updated_at": now.isoformat(),
        }

        await self._redis.set(key, json.dumps(payload), ex=self._ttl)

        return SessionResponse(**payload, ttl_seconds=self._ttl)

    async def get(self, session_id: str) -> Optional[SessionResponse]:
        key = self._key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        ttl = await self._redis.ttl(key)
        data["ttl_seconds"] = max(0, ttl)
        return SessionResponse(**data)

    async def delete(self, session_id: str) -> bool:
        key = self._key(session_id)
        deleted = await self._redis.delete(key)
        return deleted > 0

    async def add_message(self, session_id: str, message: SessionMessage) -> Optional[SessionResponse]:
        key = self._key(session_id)
        raw = await self._redis.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        msg_dict = message.model_dump(mode="json")
        msg_dict["timestamp"] = msg_dict.get("timestamp") or datetime.now(timezone.utc).isoformat()
        data["messages"].append(msg_dict)
        data["message_count"] = len(data["messages"])
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        await self._redis.set(key, json.dumps(data), ex=self._ttl)
        ttl = await self._redis.ttl(key)
        data["ttl_seconds"] = max(0, ttl)
        return SessionResponse(**data)


class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class MockEmbeddingService(BaseEmbeddingService):
    DIMENSION = 1536

    async def embed(self, text: str) -> list[float]:
        random.seed(hash(text) & 0xFFFFFFFF)
        vec = [random.gauss(0, 1) for _ in range(self.DIMENSION)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else [0.0] * self.DIMENSION
