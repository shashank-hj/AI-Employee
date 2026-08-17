import json

import httpx
import pytest
from shared.llm import OllamaProvider, OpencodeProvider, SarvamProvider, LLMProvider
from shared.llm.openai_compatible_provider import OpenAICompatibleProvider


class TestOpencodeProvider:
    def test_default_config(self):
        provider = OpencodeProvider()
        assert provider._base_url == "http://localhost:4096"
        assert provider._agent == "general"

    def test_custom_config(self):
        provider = OpencodeProvider(
            base_url="http://localhost:9999",
            model="openai/gpt-4o",
            agent="build",
            password="secret",
            timeout=10.0,
        )
        assert provider._base_url == "http://localhost:9999"
        assert provider._model == "openai/gpt-4o"
        assert provider._agent == "build"
        assert provider._client.headers.get("Authorization") is not None

    def test_is_llm_provider(self):
        provider = OpencodeProvider()
        assert isinstance(provider, LLMProvider)
        assert not isinstance(provider, OpenAICompatibleProvider)

    def test_no_auth_header_without_password(self):
        provider = OpencodeProvider()
        assert "Authorization" not in provider._client.headers

    @pytest.mark.asyncio
    async def test_health_check(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"healthy": True, "version": "1.18.18"})

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        assert await provider.health_check() is True
        await provider.close()

    @pytest.mark.asyncio
    async def test_health_check_down(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        assert await provider.health_check() is False
        await provider.close()

    @pytest.mark.asyncio
    async def test_generate_extracts_text(self):
        created = {"session": "created"}

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/session":
                return httpx.Response(200, json={"id": "s1"})
            if request.method == "DELETE":
                return httpx.Response(200, json=True)
            return httpx.Response(
                200,
                json={
                    "info": {"id": "m1", "modelID": "gpt-4o", "role": "assistant"},
                    "parts": [{"type": "text", "text": "Hello from opencode"}],
                },
            )

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        result = await provider.generate("system", "user msg")
        assert result.content == "Hello from opencode"
        assert result.model == "opencode"
        await provider.close()

    @pytest.mark.asyncio
    async def test_generate_sends_no_tools_and_no_format(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/session":
                return httpx.Response(200, json={"id": "s1"})
            if request.method == "DELETE":
                return httpx.Response(200, json=True)
            body = json.loads(request.content)
            assert "tools" not in body
            assert "format" not in body
            return httpx.Response(
                200,
                json={
                    "info": {"structured_output": {"answer": "42"}},
                    "parts": [],
                },
            )

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        result = await provider.generate("system", "q")
        assert json.loads(result.content) == {"answer": "42"}
        await provider.close()

    @pytest.mark.asyncio
    async def test_classify_intent_parses_json(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/session":
                return httpx.Response(200, json={"id": "s1"})
            if request.method == "DELETE":
                return httpx.Response(200, json=True)
            return httpx.Response(
                200,
                json={
                    "info": {"id": "m1", "role": "assistant"},
                    "parts": [
                        {
                            "type": "text",
                            "text": json.dumps({
                                "intent": "booking",
                                "confidence": 0.95,
                                "requires_human": False,
                                "reason": "user wants to schedule",
                                "entities": [],
                                "suggested_tools": ["calendar"],
                            }),
                        }
                    ],
                },
            )

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        result = await provider.classify_intent("Book a demo tomorrow")
        assert result.intent == "booking"
        assert result.confidence == 0.95
        assert result.suggested_tools == ["calendar"]
        await provider.close()

    @pytest.mark.asyncio
    async def test_classify_intent_falls_back_to_general_on_error(self):
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        provider = OpencodeProvider(transport=httpx.MockTransport(handler))
        result = await provider.classify_intent("hello")
        assert result.intent == "general"
        await provider.close()


class TestOllamaProvider:
    def test_default_config(self):
        provider = OllamaProvider()
        assert provider._model == "qwen3:8b"
        assert provider._base_url == "http://localhost:11434"

    def test_custom_config(self):
        provider = OllamaProvider(
            base_url="http://localhost:9999",
            model="mistral",
            timeout=10.0,
            temperature=0.5,
            max_tokens=512,
        )
        assert provider._model == "mistral"
        assert provider._base_url == "http://localhost:9999"
        assert provider._temperature == 0.5
        assert provider._max_tokens == 512

    def test_is_llm_provider(self):
        provider = OllamaProvider()
        assert isinstance(provider, LLMProvider)
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_no_auth_header(self):
        provider = OllamaProvider()
        assert "Authorization" not in provider._client.headers


class TestSarvamProvider:
    def test_default_config(self):
        provider = SarvamProvider(api_key="test-key")
        assert provider._model == "sarvam-105b"
        assert provider._base_url == "https://api.sarvam.ai"

    def test_is_llm_provider(self):
        provider = SarvamProvider(api_key="test-key")
        assert isinstance(provider, LLMProvider)

    def test_has_auth_header(self):
        provider = SarvamProvider(api_key="test-key")
        assert "Authorization" in provider._client.headers
        assert provider._client.headers["Authorization"] == "Bearer test-key"


class TestProviderSwitch:
    """Verify that the provider factory picks the right provider."""
    def test_factory_returns_ollama(self, monkeypatch):
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
        provider = _resolve_provider()
        assert isinstance(provider, OllamaProvider)
        assert provider._model == settings.OLLAMA_MODEL

    def test_factory_returns_sarvam(self, monkeypatch):
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "sarvam")
        monkeypatch.setattr(settings, "SARVAM_API_KEY", "test-key")
        provider = _resolve_provider()
        assert isinstance(provider, SarvamProvider)

    def test_factory_returns_opencode(self, monkeypatch):
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "opencode")
        provider = _resolve_provider()
        assert isinstance(provider, OpencodeProvider)
        assert provider._base_url == settings.OPENCODE_BASE_URL

    def test_factory_returns_none_when_empty(self, monkeypatch):
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "")
        monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
        provider = _resolve_provider()
        assert provider is None

    def test_sarvam_not_used_implicitly(self, monkeypatch):
        """Sarvam must only be used when explicitly selected via LLM_PROVIDER."""
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "")
        monkeypatch.setattr(settings, "SARVAM_API_KEY", "test-key")
        provider = _resolve_provider()
        assert provider is None
