import asyncio
from datetime import UTC
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from memory.app import create_app
from memory.container import get_memory_service
from memory.services.memory_service import MemoryService
from memory.services.stores import MockEmbeddingService


class MockSessionStore:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self.upsert = AsyncMock(wraps=self._upsert)
        self.get = AsyncMock(wraps=self._get)
        self.delete = AsyncMock(wraps=self._delete)
        self.get_message_count = AsyncMock(wraps=self._get_message_count)
        self.add_message = AsyncMock(wraps=self._add_message)
        self.update_state = AsyncMock(wraps=self._update_state)
        self.clear_messages = AsyncMock(wraps=self._clear_messages)
        self.list = AsyncMock(wraps=self._list)

    def _serialize(self, resp):
        return resp.model_dump(mode="json")

    async def _upsert(self, data, message_count=None):
        import uuid

        from memory.schemas.session import SessionResponse
        sid = data.session_id or str(uuid.uuid4())
        if sid in self._store:
            prev = self._store[sid]
            messages = prev.get("messages", [])
            msg_count = message_count if message_count is not None else len(messages)
        else:
            messages = [
                m.model_dump(mode="json") if hasattr(m, "model_dump") else m
                for m in data.messages
            ]
            msg_count = message_count if message_count is not None else len(messages)
        resp = SessionResponse(
            session_id=sid,
            user_id=data.user_id,
            messages=messages,
            context=data.context,
            metadata=data.metadata,
            message_count=msg_count,
        )
        self._store[sid] = self._serialize(resp)
        return resp

    async def _get(self, session_id):
        from memory.schemas.session import SessionResponse
        raw = self._store.get(session_id)
        if raw is None:
            return None
        return SessionResponse(**raw)

    async def _delete(self, session_id):
        if session_id in self._store:
            del self._store[session_id]
            return True
        return False

    async def _get_message_count(self, session_id):
        raw = self._store.get(session_id)
        if raw is None:
            return 0
        return raw.get("message_count", len(raw.get("messages", [])))

    async def _add_message(self, session_id):
        from memory.schemas.session import SessionResponse
        raw = self._store.get(session_id)
        if raw is None:
            return None
        raw["message_count"] = raw.get("message_count", 0) + 1
        return SessionResponse(**raw)

    async def _update_state(self, session_id, context=None, metadata=None):
        from memory.schemas.session import SessionResponse
        raw = self._store.get(session_id)
        if raw is None:
            return None
        if context is not None:
            raw["context"] = context
        if metadata is not None:
            raw["metadata"] = metadata
        return SessionResponse(**raw)

    async def _clear_messages(self, session_id):
        from memory.schemas.session import SessionResponse
        raw = self._store.get(session_id)
        if raw is None:
            return None
        raw["messages"] = []
        raw["message_count"] = 0
        return SessionResponse(**raw)

    async def _list(self, user_id=None, page=1, page_size=20):
        from memory.schemas.session import SessionResponse
        items = [SessionResponse(**raw) for raw in self._store.values()]
        if user_id:
            items = [i for i in items if i.user_id == user_id]
        items.sort(key=lambda i: (i.updated_at or i.created_at) is not None, reverse=True)
        start = (page - 1) * page_size
        return items[start : start + page_size], len(items)


class MockLongTermRepo:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.create = AsyncMock(wraps=self._create)
        self.get_by_id = AsyncMock(wraps=self._get_by_id)
        self.delete = AsyncMock(wraps=self._delete)
        self.search_by_embedding = AsyncMock(wraps=self._search)
        self.list_by_user = AsyncMock(wraps=self._list)

    async def _create(self, memory):
        import uuid
        from datetime import datetime
        now = datetime.now(UTC)
        memory.id = memory.id or uuid.uuid4()
        memory.created_at = now
        memory.updated_at = now
        self._store[str(memory.id)] = memory
        return memory

    async def _get_by_id(self, memory_id):
        return self._store.get(memory_id)

    async def _delete(self, memory):
        self._store.pop(str(memory.id), None)

    async def _search(self, embedding, user_id=None, memory_type=None, top_k=10, min_score=0.0,
                      importance_min=None, importance_max=None, sort="score"):
        return [(m, 0.85) for m in list(self._store.values())[:top_k]]

    async def _list(self, user_id, memory_type=None, page=1, page_size=20):
        items = list(self._store.values())
        return items, len(items)


class MockConvRepo:
    def __init__(self):
        self._store: dict[str, list] = {}
        self.create = AsyncMock(wraps=self._create)
        self.get_by_session = AsyncMock(wraps=self._get)
        self.delete_by_session = AsyncMock(wraps=self._delete_by_session)

    async def _create(self, msg):
        import uuid
        from datetime import datetime
        now = datetime.now(UTC)
        msg.id = msg.id or uuid.uuid4()
        msg.created_at = now
        msg.updated_at = now
        self._store.setdefault(msg.session_id, []).append(msg)
        return msg

    async def _get(self, session_id):
        return self._store.get(session_id, [])

    async def _delete_by_session(self, session_id):
        self._store.pop(session_id, None)


class MockProfileRepo:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.get_by_user_id = AsyncMock(wraps=self._get)
        self.upsert = AsyncMock(wraps=self._upsert)

    async def _get(self, user_id):
        return self._store.get(user_id)

    async def _upsert(self, profile):
        import uuid
        from datetime import datetime
        now = datetime.now(UTC)
        profile.id = profile.id or uuid.uuid4()
        profile.created_at = profile.created_at or now
        profile.updated_at = now
        self._store[profile.user_id] = profile
        return profile


class MockSummarizer:
    def __init__(self):
        self.summarize = AsyncMock(return_value="Test summary of the conversation.")

    async def _summarize(self, transcript):
        return "Test summary of the conversation."


def _make_mock_service():
    return MemoryService(
        session_store=MockSessionStore(),
        long_term_repo=MockLongTermRepo(),
        conversation_repo=MockConvRepo(),
        profile_repo=MockProfileRepo(),
        embedding_service=MockEmbeddingService(),
        summarizer=MockSummarizer(),
    )


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_service():
    return _make_mock_service()


@pytest.fixture
def app(mock_service):
    app = create_app()

    async def override():
        return mock_service

    app.dependency_overrides[get_memory_service] = override
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
