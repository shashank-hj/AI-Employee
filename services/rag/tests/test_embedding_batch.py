import json

import httpx
import pytest

from shared.llm.embedding_provider import OllamaEmbeddingProvider


def _provider(handler, batch_size=64):
    transport = httpx.MockTransport(handler)
    return OllamaEmbeddingProvider(
        base_url="http://ollama:11434",
        model="nomic-embed-text",
        timeout=10.0,
        batch_size=batch_size,
        transport=transport,
    )


class TestOllamaEmbeddingProviderBatch:
    @pytest.mark.asyncio
    async def test_embed_batches_via_api_embed(self):
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            assert request.url.path == "/api/embed"
            inputs = body["input"]
            return httpx.Response(200, json={"embeddings": [[1.0, 2.0]] * len(inputs)})

        provider = _provider(handler, batch_size=2)
        vectors = await provider.embed(["a", "b", "c", "d", "e"])

        assert len(vectors) == 5
        assert len(requests) == 3  # 5 items in batches of 2 -> 3 calls
        assert requests[0]["input"] == ["a", "b"]
        assert requests[1]["input"] == ["c", "d"]
        assert requests[2]["input"] == ["e"]
        assert provider.dimension == 2

    @pytest.mark.asyncio
    async def test_falls_back_to_legacy_endpoint_on_batch_failure(self):
        seen = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.url.path)
            if request.url.path == "/api/embed":
                return httpx.Response(500)
            # legacy single endpoint
            return httpx.Response(200, json={"embedding": [0.5, 0.5]})

        provider = _provider(handler, batch_size=2)
        vectors = await provider.embed(["a", "b", "c"])

        assert len(vectors) == 3
        # batches: [a,b] and [c] -> one /api/embed call per batch
        assert seen.count("/api/embed") == 2
        # each batch falls back to per-text /api/embeddings (2 + 1 calls)
        assert seen.count("/api/embeddings") == 3
        assert all(v == [0.5, 0.5] for v in vectors)

    @pytest.mark.asyncio
    async def test_returns_zero_vectors_when_everything_fails(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        provider = _provider(handler, batch_size=2)
        vectors = await provider.embed(["a", "b"])

        assert len(vectors) == 2
        assert all(v == [0.0] * 768 for v in vectors)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"embeddings": []})

        provider = _provider(handler)
        assert await provider.embed([]) == []
