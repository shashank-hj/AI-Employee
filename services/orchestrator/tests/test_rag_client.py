import pytest

from orchestrator.tools.rag_client import MockRAGClient, HttpRAGClient, RAGClient


class TestMockRAGClient:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        client = MockRAGClient()
        results = await client.search("remote work")
        assert len(results) > 0
        assert all("title" in r for r in results)
        assert all("snippet" in r for r in results)
        assert all("score" in r for r in results)

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        client = MockRAGClient()
        results = await client.search("policy", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_search_scores_normalized(self):
        client = MockRAGClient()
        results = await client.search("remote work")
        scores = [r["score"] for r in results]
        assert all(0 <= s <= 1 for s in scores)

    @pytest.mark.asyncio
    async def test_search_empty_query(self):
        client = MockRAGClient()
        results = await client.search("")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_is_abstract(self):
        with pytest.raises(TypeError):
            RAGClient()  # type: ignore


class TestHttpRAGClient:
    @pytest.mark.asyncio
    async def test_falls_back_to_mock_on_connection_error(self):
        client = HttpRAGClient(
            base_url="http://nonexistent:9999",
            timeout=0.1,
            fallback_client=MockRAGClient(),
        )
        results = await client.search("test query")
        assert len(results) > 0
        assert all("title" in r for r in results)

    @pytest.mark.asyncio
    async def test_default_fallback_is_mock(self):
        client = HttpRAGClient(
            base_url="http://nonexistent:9999",
            timeout=0.1,
            fallback_client=MockRAGClient(),
        )
        results = await client.search("test query")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_no_fallback_returns_friendly_message(self):
        client = HttpRAGClient(
            base_url="http://nonexistent:9999",
            timeout=0.1,
        )
        results = await client.search("test query")
        assert len(results) == 1
        assert results[0]["title"] == "Search Unavailable"

    @pytest.mark.asyncio
    async def test_health_check_false_when_unreachable(self):
        client = HttpRAGClient(
            base_url="http://nonexistent:9999",
            timeout=0.1,
        )
        healthy = await client.health_check()
        assert healthy is False

    @pytest.mark.asyncio
    async def test_mock_health_check_always_true(self):
        client = MockRAGClient()
        healthy = await client.health_check()
        assert healthy is True
