import pytest
from shared.llm import OllamaProvider, SarvamProvider, LLMProvider
from shared.llm.openai_compatible_provider import OpenAICompatibleProvider


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

    def test_factory_returns_none_when_empty(self, monkeypatch):
        from orchestrator.container import _resolve_provider
        from orchestrator.config import settings

        monkeypatch.setattr(settings, "LLM_PROVIDER", "")
        monkeypatch.setattr(settings, "SARVAM_API_KEY", "")
        provider = _resolve_provider()
        assert provider is None
