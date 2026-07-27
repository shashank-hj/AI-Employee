import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from memory.app import create_app
from memory.container import get_memory_service
from memory.services.memory_service import MemoryService
from memory.services.stores import SessionStore, MockEmbeddingService
from memory.repositories.long_term import LongTermMemoryRepository
from memory.repositories.conversation import ConversationRepository
from memory.repositories.profile import ProfileRepository


class MockSessionStore:
    def __init__(self):
        self._store: dict[str, dict] = {}
        self.upsert = AsyncMock(wraps=self._upsert)
        self.get = AsyncMock(wraps=self._get)
        self.delete = AsyncMock(wraps=self._delete)
        self.add_message = AsyncMock(wraps=self._add_message)

    def _serialize(self, resp):
        return resp.model_dump(mode="json")

    async def _upsert(self, data):
        import uuid
        from memory.schemas.session import SessionResponse
        sid = data.session_id or str(uuid.uuid4())
        resp = SessionResponse(
            session_id=sid,
            user_id=data.user_id,
            messages=data.messages,
            context=data.context,
            metadata=data.metadata,
            message_count=len(data.messages),
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

    async def _add_message(self, session_id, message):
        from memory.schemas.session import SessionResponse
        raw = self._store.get(session_id)
        if raw is None:
            return None
        msg_dict = message.model_dump(mode="json")
        raw["messages"].append(msg_dict)
        raw["message_count"] = len(raw["messages"])
        return SessionResponse(**raw)


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
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        memory.id = memory.id or uuid.uuid4()
        memory.created_at = now
        memory.updated_at = now
        self._store[str(memory.id)] = memory
        return memory

    async def _get_by_id(self, memory_id):
        return self._store.get(memory_id)

    async def _delete(self, memory):
        self._store.pop(str(memory.id), None)

    async def _search(self, embedding, user_id=None, memory_type=None, top_k=10, min_score=0.0):
        return [(m, 0.85) for m in list(self._store.values())[:top_k]]

    async def _list(self, user_id, memory_type=None, page=1, page_size=20):
        items = list(self._store.values())
        return items, len(items)


class MockConvRepo:
    def __init__(self):
        self._store: dict[str, list] = {}
        self.create = AsyncMock(wraps=self._create)
        self.get_by_session = AsyncMock(wraps=self._get)

    async def _create(self, msg):
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        msg.id = msg.id or uuid.uuid4()
        msg.created_at = now
        msg.updated_at = now
        self._store.setdefault(msg.session_id, []).append(msg)
        return msg

    async def _get(self, session_id):
        return self._store.get(session_id, [])


class MockProfileRepo:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.get_by_user_id = AsyncMock(wraps=self._get)
        self.upsert = AsyncMock(wraps=self._upsert)

    async def _get(self, user_id):
        return self._store.get(user_id)

    async def _upsert(self, profile):
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        profile.id = profile.id or uuid.uuid4()
        profile.created_at = profile.created_at or now
        profile.updated_at = now
        self._store[profile.user_id] = profile
        return profile


def _make_mock_service():
    return MemoryService(
        session_store=MockSessionStore(),
        long_term_repo=MockLongTermRepo(),
        conversation_repo=MockConvRepo(),
        profile_repo=MockProfileRepo(),
        embedding_service=MockEmbeddingService(),
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
