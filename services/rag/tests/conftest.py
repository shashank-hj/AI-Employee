import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from rag.app import create_app
from rag.container import get_rag_service
from rag.services.rag_service import RAGService
from rag.services.pipeline import MockEmbeddingProvider, TextChunker, DocumentIngester
from rag.schemas.documents import SearchResult


class MockDocRepo:
    def __init__(self):
        self._store: dict[str, object] = {}
        self.create = AsyncMock(wraps=self._create)
        self.get_by_id = AsyncMock(wraps=self._get_by_id)
        self.update_status = AsyncMock(wraps=self._update_status)
        self.list_all = AsyncMock(wraps=self._list_all)
        self.delete = AsyncMock(wraps=self._delete)

    async def _create(self, doc):
        import uuid
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        doc.id = doc.id or uuid.uuid4()
        doc.created_at = now
        doc.updated_at = now
        self._store[str(doc.id)] = doc
        return doc

    async def _get_by_id(self, doc_id):
        return self._store.get(doc_id)

    async def _update_status(self, doc, status, chunks_count=0):
        from datetime import datetime, timezone
        doc.status = status
        doc.chunks_count = chunks_count
        doc.updated_at = datetime.now(timezone.utc)
        return doc

    async def _list_all(self, page=1, page_size=20):
        items = list(self._store.values())
        return items, len(items)

    async def _delete(self, doc):
        self._store.pop(str(doc.id), None)


class MockVectorStore:
    def __init__(self):
        self._store: list[object] = []
        self.store_chunks = AsyncMock(wraps=self._store_chunks)
        self.search = AsyncMock(wraps=self._search)
        self.delete_by_document = AsyncMock(wraps=self._del_by_doc)

    async def _store_chunks(self, chunks):
        for c in chunks:
            self._store.append(c)
        return chunks

    async def _search(self, embedding, top_k=5, min_score=0.0):
        results = []
        for i, c in enumerate(self._store[:top_k]):
            results.append((c, str(c.document_id), f"Doc-{i}", 0.85 - i * 0.05))
        return results

    async def _del_by_doc(self, document_id):
        before = len(self._store)
        self._store = [c for c in self._store if str(c.document_id) != document_id]
        return before - len(self._store)


class MockRanker:
    def rank(self, results, query):
        return sorted(results, key=lambda r: r.score, reverse=True)


def _make_mock_service():
    doc_repo = MockDocRepo()
    vector_store = MockVectorStore()
    embedder = MockEmbeddingProvider()
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    ingester = DocumentIngester(chunker)

    from rag.services.rag_service import Retriever
    retriever = Retriever(vector_store, embedder)
    ranker = MockRanker()

    return RAGService(
        doc_repo=doc_repo,
        vector_store=vector_store,
        ingester=ingester,
        embedder=embedder,
        retriever=retriever,
        ranker=ranker,
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
    app.dependency_overrides[get_rag_service] = override
    return app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


DOC_PAYLOAD = {
    "title": "Test Document",
    "content": "This is paragraph one.\n\nThis is paragraph two.\n\nThis is paragraph three.",
    "source": "test-source",
    "content_type": "text/plain",
}
