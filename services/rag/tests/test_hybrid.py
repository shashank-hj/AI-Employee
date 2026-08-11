"""Tests for hybrid retrieval (RRF fusion), reranking, and citations (C3)."""

from types import SimpleNamespace

import pytest

from rag.schemas.documents import SearchResult
from rag.services.rag_service import HybridRetriever, Ranker


def _chunk(chunk_id: str, index: int = 0, content: str = "some content"):
    return SimpleNamespace(
        id=chunk_id,
        chunk_index=index,
        content=content,
        metadata_={"k": "v"},
        created_at=None,
        updated_at=None,
    )


def _vector_result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        document_title=f"Doc {chunk_id}",
        chunk_index=0,
        content="some content about " + chunk_id,
        score=score,
    )


class FakeVectorRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        return self._results[:top_k]


class FakeLexicalStore:
    def __init__(self, results: list) -> None:
        self._results = results

    async def search_lexical(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list:
        return self._results[:top_k]


class TestHybridRetriever:
    @pytest.mark.asyncio
    async def test_overlap_ranks_first(self):
        vector = [_vector_result("A", 0.9), _vector_result("B", 0.8), _vector_result("C", 0.7)]
        lexical = [
            (_chunk("B"), "doc-B", "Doc B", 0.6),
            (_chunk("D"), "doc-D", "Doc D", 0.5),
            (_chunk("E"), "doc-E", "Doc E", 0.4),
        ]
        hybrid = HybridRetriever(FakeVectorRetriever(vector), FakeLexicalStore(lexical))
        results = await hybrid.retrieve("test", top_k=3)

        # "B" appears in both lists -> highest RRF -> first
        assert results[0].chunk_id == "B"
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_respects_top_k(self):
        vector = [_vector_result(str(i), 0.9) for i in range(10)]
        lexical = [(_chunk(str(i)), f"doc-{i}", f"Doc {i}", 0.5) for i in range(10)]
        hybrid = HybridRetriever(FakeVectorRetriever(vector), FakeLexicalStore(lexical))
        results = await hybrid.retrieve("test", top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_scores_in_range(self):
        vector = [_vector_result("A", 0.9), _vector_result("B", 0.8)]
        lexical = [(_chunk("A"), "doc-A", "Doc A", 0.7)]
        hybrid = HybridRetriever(FakeVectorRetriever(vector), FakeLexicalStore(lexical))
        results = await hybrid.retrieve("test", top_k=5)
        for result in results:
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_single_source_works(self):
        vector = [_vector_result("X", 0.9)]
        hybrid = HybridRetriever(FakeVectorRetriever(vector), FakeLexicalStore([]))
        results = await hybrid.retrieve("test", top_k=5)
        assert [r.chunk_id for r in results] == ["X"]


class TestRanker:
    def test_sorts_by_score(self):
        low = SearchResult(chunk_id="1", document_id="d", document_title="t",
                           chunk_index=0, content="plain", score=0.2)
        high = SearchResult(chunk_id="2", document_id="d", document_title="t",
                            chunk_index=0, content="plain", score=0.8)
        ranked = Ranker().rank([low, high], "pricing")
        assert [r.chunk_id for r in ranked] == ["2", "1"]

    def test_keyword_overlap_boost(self):
        docs = [
            SearchResult(chunk_id="a", document_id="d", document_title="Unrelated",
                         chunk_index=0, content="nothing relevant here", score=0.55),
            SearchResult(chunk_id="b", document_id="d", document_title="Pricing Guide",
                         chunk_index=0, content="our enterprise pricing tiers", score=0.45),
        ]
        ranked = Ranker().rank(docs, "enterprise pricing guide")
        # "b" matches several query terms -> boosted above "a" despite lower vector score
        assert ranked[0].chunk_id == "b"

    def test_no_terms_preserves_order(self):
        docs = [
            SearchResult(chunk_id="a", document_id="d", document_title="t",
                         chunk_index=0, content="x", score=0.9),
            SearchResult(chunk_id="b", document_id="d", document_title="t",
                         chunk_index=0, content="y", score=0.1),
        ]
        ranked = Ranker().rank(docs, "the")
        assert [r.chunk_id for r in ranked] == ["a", "b"]


class TestCitations:
    @pytest.mark.asyncio
    async def test_query_returns_citations(self, client):
        await client.post("/api/documents", json={
            "title": "Pricing Doc",
            "content": "Enterprise pricing starts at 999 dollars per month with premium support.",
        })
        response = await client.post("/api/query", json={"query": "enterprise pricing"})
        assert response.status_code == 200
        data = response.json()
        assert "citations" in data
        assert len(data["citations"]) >= 1
        citation = data["citations"][0]
        first_result = data["results"][0]
        assert citation["document_id"] == first_result["document_id"]
        assert citation["document_title"] == first_result["document_title"]
        assert citation["chunk_index"] == first_result["chunk_index"]
        assert "content" in citation
        assert "score" in citation

    @pytest.mark.asyncio
    async def test_empty_query_has_no_citations(self, client):
        response = await client.post("/api/query", json={"query": "zzz nothing matches"})
        assert response.status_code == 200
        assert response.json()["citations"] == []
