from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag.services.translator import NoopQueryTranslator, SpeechQueryTranslator


class _FakeLLM:
    async def generate(self, system_prompt, user_message):
        return MagicMock(content="refined: " + user_message, model="fake")


class TestQueryRefiner:
    @pytest.mark.asyncio
    async def test_refine_uses_llm(self):
        from rag.services.query_refiner import QueryRefiner

        refiner = QueryRefiner(_FakeLLM())
        assert await refiner.refine("loan terms") == "refined: loan terms"

    @pytest.mark.asyncio
    async def test_refine_falls_back_on_failure(self):
        from rag.services.query_refiner import QueryRefiner

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("boom"))
        refiner = QueryRefiner(llm)
        assert await refiner.refine("loan terms") == "loan terms"

    @pytest.mark.asyncio
    async def test_refine_empty_returns_empty(self):
        from rag.services.query_refiner import QueryRefiner

        refiner = QueryRefiner(_FakeLLM())
        assert await refiner.refine("   ") == ""


class TestQueryTranslator:
    @pytest.mark.asyncio
    async def test_noop_returns_same_text(self):
        translator = NoopQueryTranslator()
        assert await translator.translate("hello") == "hello"

    @pytest.mark.asyncio
    async def test_speech_translator_calls_translate_endpoint(self):
        translator = SpeechQueryTranslator(base_url="https://mock.local", api_key="key")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"translated_text": "नमस्ते"}
        mock_response.raise_for_status = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await translator.translate("hello", target_language_code="hi-IN")

        assert result == "नमस्ते"
        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["target_language_code"] == "hi-IN"

    @pytest.mark.asyncio
    async def test_speech_translator_falls_back_on_error(self):
        translator = SpeechQueryTranslator(base_url="https://mock.local")
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("boom"))

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await translator.translate("hello")

        assert result == "hello"


class TestRAGQueryIntegration:
    @pytest.mark.asyncio
    async def test_query_runs_refiner_and_translator(self, mock_service, client):
        from rag.services.query_refiner import QueryRefiner
        from rag.services.translator import NoopQueryTranslator

        refiner = QueryRefiner(_FakeLLM())
        mock_service._refiner = refiner
        mock_service._translator = NoopQueryTranslator()

        await client.post("/api/documents", json={
            "title": "Doc",
            "content": "Content about machine learning.",
        })
        response = await client.post("/api/query", json={
            "query": "What is ML?",
            "language": "en-IN",
        })

        assert response.status_code == 200
        data = response.json()
        assert data["refined_query"] == "refined: What is ML?"
        assert data["language"] == "en-IN"

    @pytest.mark.asyncio
    async def test_query_without_configured_extras_is_unchanged(self, mock_service, client):
        await client.post("/api/documents", json={
            "title": "Doc",
            "content": "Some content.",
        })
        response = await client.post("/api/query", json={
            "query": "some content",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["refined_query"] is None
        assert data["translated_query"] is None
        assert data["language"] is None
