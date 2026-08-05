import json
import random
import math
from abc import ABC, abstractmethod

import structlog
from redis.asyncio import Redis

from memory.schemas.session import SessionCreate, SessionMessage, SessionResponse

logger = structlog.get_logger(__name__)


class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed(self, text: str) -> list[float]: ...


class MockEmbeddingService(BaseEmbeddingService):
    DIMENSION = 768

    async def embed(self, text: str) -> list[float]:
        random.seed(hash(text) & 0xFFFFFFFF)
        vec = [random.gauss(0, 1) for _ in range(self.DIMENSION)]
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else [0.0] * self.DIMENSION


class OllamaEmbeddingService(BaseEmbeddingService):
    def __init__(
        self,
        base_url: str = "http://host.docker.internal:11434",
        model: str = "nomic-embed-text",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout = timeout
        self._provider = None

    async def embed(self, text: str) -> list[float]:
        if self._provider is None:
            from shared.llm.embedding_provider import OllamaEmbeddingProvider
            self._provider = OllamaEmbeddingProvider(
                base_url=self._base_url,
                model=self._model,
                timeout=self._timeout,
            )
        vectors = await self._provider.embed([text])
        return vectors[0] if vectors else [0.0] * 768


class SessionStore:
    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def upsert(self, session: SessionCreate) -> SessionResponse:
        key = f"session:{session.session_id}"
        data = {
            "session_id": session.session_id,
            "user_id": session.user_id or "anonymous",
            "messages": [],
        }
        await self._redis.hset(key, mapping={"data": json.dumps(data)})
        return SessionResponse(**data)

    async def get_session(self, session_id: str) -> SessionResponse | None:
        key = f"session:{session_id}"
        raw = await self._redis.hget(key, "data")
        if raw is None:
            return None
        data = json.loads(raw)
        return SessionResponse(**data)

    async def add_message(self, session_id: str, message: SessionMessage) -> SessionResponse | None:
        session = await self.get_session(session_id)
        if session is None:
            return None
        messages_raw = [m.model_dump(mode="json") if hasattr(m, "model_dump") else m for m in session.messages]
        messages_raw.append(message.model_dump(mode="json"))
        key = f"session:{session_id}"
        payload = {"session_id": session.session_id, "user_id": session.user_id, "messages": messages_raw}
        await self._redis.hset(key, mapping={"data": json.dumps(payload)})
        return session

    async def delete_session(self, session_id: str) -> None:
        await self._redis.delete(f"session:{session_id}")

    async def set_ttl(self, session_id: str, ttl: int = 3600) -> bool:
        key = f"session:{session_id}"
        return await self._redis.expire(key, ttl) if ttl > 0 else True
