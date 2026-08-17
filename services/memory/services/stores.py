from __future__ import annotations

import json
import math
import random
import uuid
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta

import structlog
from redis.asyncio import Redis

from memory.schemas.session import SessionCreate, SessionResponse

logger = structlog.get_logger(__name__)


def _merge_dicts(base: dict, update: dict) -> dict:
    """Shallow-merge ``update`` into a copy of ``base``."""
    merged = dict(base or {})
    merged.update(update or {})
    return merged


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


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


_KEY_PREFIX = "session:"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


class SessionStore:
    """Redis-backed session state store.

    Sessions hold *state only* (user_id, context, metadata, message_count, TTL).
    Conversation message bodies live in PostgreSQL (``conversation_messages``);
    ``message_count`` here is a cheap counter that the service keeps in sync.
    """

    def __init__(self, redis: Redis, ttl_seconds: int = 86400) -> None:
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    async def upsert(
        self,
        session: SessionCreate,
        message_count: int | None = None,
        ttl_seconds: int | None = None,
    ) -> SessionResponse:
        session_id = session.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
        key = f"{_KEY_PREFIX}{session_id}"

        existing = await self._read(key)
        now = _now_iso()
        if existing is not None:
            data = {
                "session_id": session_id,
                "user_id": session.user_id or existing.get("user_id"),
                "context": _merge_dicts(existing.get("context") or {}, session.context),
                "metadata": _merge_dicts(existing.get("metadata") or {}, session.metadata),
                "message_count": (
                    message_count if message_count is not None else existing.get("message_count", 0)
                ),
                "created_at": existing.get("created_at"),
                "updated_at": now,
                "ttl_seconds": existing.get("ttl_seconds") or ttl_seconds or self._ttl_seconds,
            }
        else:
            data = {
                "session_id": session_id,
                "user_id": session.user_id or "anonymous",
                "context": session.context,
                "metadata": session.metadata,
                "message_count": message_count or 0,
                "created_at": now,
                "updated_at": now,
                "ttl_seconds": ttl_seconds or self._ttl_seconds,
            }

        await self._write(key, data)
        return self._to_response(data)

    async def get(self, session_id: str) -> SessionResponse | None:
        data = await self._read(f"{_KEY_PREFIX}{session_id}")
        return self._to_response(data) if data is not None else None

    async def delete(self, session_id: str) -> bool:
        deleted = await self._redis.delete(f"{_KEY_PREFIX}{session_id}")
        return bool(deleted)

    async def add_message(self, session_id: str) -> SessionResponse | None:
        """Bump a session's message counter by one (message body lands in PG)."""
        key = f"{_KEY_PREFIX}{session_id}"
        data = await self._read(key)
        if data is None:
            return None
        data["message_count"] = data.get("message_count", 0) + 1
        data["updated_at"] = _now_iso()
        await self._write(key, data)
        return self._to_response(data)

    async def clear_messages(self, session_id: str) -> SessionResponse | None:
        key = f"{_KEY_PREFIX}{session_id}"
        data = await self._read(key)
        if data is None:
            return None
        data["message_count"] = 0
        data["updated_at"] = _now_iso()
        await self._write(key, data)
        return self._to_response(data)

    async def update_state(
        self,
        session_id: str,
        context: dict | None = None,
        metadata: dict | None = None,
    ) -> SessionResponse | None:
        key = f"{_KEY_PREFIX}{session_id}"
        data = await self._read(key)
        if data is None:
            return None
        if context is not None:
            data["context"] = _merge_dicts(data.get("context", {}), context)
        if metadata is not None:
            data["metadata"] = _merge_dicts(data.get("metadata", {}), metadata)
        data["updated_at"] = _now_iso()
        await self._write(key, data)
        return self._to_response(data)

    async def touch(self, session_id: str) -> bool:
        key = f"{_KEY_PREFIX}{session_id}"
        data = await self._read(key)
        if data is None:
            return False
        data["updated_at"] = _now_iso()
        await self._write(key, data)
        return True

    async def list(
        self,
        user_id: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[SessionResponse], int]:
        sessions: list[dict] = []
        async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            data = await self._read(key)
            if data is None:
                continue
            if user_id is not None and data.get("user_id") != user_id:
                continue
            sessions.append(data)

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        total = len(sessions)
        start = (page - 1) * page_size
        items = []
        for s in sessions[start : start + page_size]:
            sid = s.get("session_id")
            if not sid:
                continue
            items.append(self._to_response(s))
        return items, total

    async def list_ids(self) -> list[str]:
        ids: list[str] = []
        async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            prefix_len = len(_KEY_PREFIX)
            ids.append(key[prefix_len:])
        return ids

    async def set_message_count(self, session_id: str, count: int) -> SessionResponse | None:
        key = f"{_KEY_PREFIX}{session_id}"
        data = await self._read(key)
        if data is None:
            return None
        data["message_count"] = count
        data["updated_at"] = _now_iso()
        await self._write(key, data)
        return self._to_response(data)

    async def _read(self, key: str) -> dict | None:
        raw = await self._redis.hget(key, "data")
        if raw is None:
            return None
        return json.loads(raw)

    async def _write(self, key: str, data: dict) -> None:
        await self._redis.hset(key, mapping={"data": json.dumps(data)})
        ttl = int(data.get("ttl_seconds") or self._ttl_seconds)
        if ttl > 0:
            await self._redis.expire(key, ttl)

    @staticmethod
    def _to_response(data: dict) -> SessionResponse:
        sid = data.get("session_id")
        if not sid:
            sid = "corrupted-" + str(uuid.uuid4())[:8]
        ttl = int(data.get("ttl_seconds") or 86400)
        updated_raw = data.get("updated_at")
        updated = _parse_iso(updated_raw) if updated_raw else None
        expires_at = updated + timedelta(seconds=ttl) if updated is not None else None
        return SessionResponse(
            session_id=sid,
            user_id=data.get("user_id"),
            context=data.get("context") or {},
            metadata=data.get("metadata") or {},
            message_count=int(data.get("message_count") or 0),
            created_at=_parse_iso(data.get("created_at")) if data.get("created_at") else None,
            updated_at=updated,
            ttl_seconds=ttl,
            expires_at=expires_at,
        )
