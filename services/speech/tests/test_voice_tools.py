import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _mock_httpx_post(return_value):
    mock_response = MagicMock()
    mock_response.json.return_value = return_value
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


class TestTranslation:
    @pytest.mark.asyncio
    async def test_translate_no_api_key_returns_informative_message(self, client, monkeypatch):
        from speech.container import get_translation_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_translation_provider.cache_clear()

        response = await client.post(
            "/api/translate-text",
            json={"text": "hola", "source_language_code": "auto", "target_language_code": "en-IN"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "not configured" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_translate_success(self):
        from speech.providers.translation import SarvamTranslationProvider

        provider = SarvamTranslationProvider(api_key="test-key", base_url="https://mock.local")
        mock_client = _mock_httpx_post({"translated_text": "Hello", "source_language_code": "es-ES"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.translate_text("hola", source_language_code="auto", target_language_code="en-IN")
            assert result["translated_text"] == "Hello"
            assert result["source_language_code"] == "es-ES"


class TestLanguageDetection:
    @pytest.mark.asyncio
    async def test_detect_language_no_api_key_returns_unknown(self, client, monkeypatch):
        from speech.container import get_language_detection_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_language_detection_provider.cache_clear()

        response = await client.post(
            "/api/detect-language",
            json={"text": "hola"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["language_code"] == "unknown"
        assert "not configured" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_detect_language_success(self):
        from speech.providers.language_detection import SarvamLanguageDetectionProvider

        provider = SarvamLanguageDetectionProvider(api_key="test-key", base_url="https://mock.local")
        mock_client = _mock_httpx_post({"language_code": "es-ES", "script_code": "Latn"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.detect("hola")
            assert result["language_code"] == "es-ES"
            assert result["script_code"] == "Latn"


class TestTransliteration:
    @pytest.mark.asyncio
    async def test_transliterate_no_api_key_returns_informative_message(self, client, monkeypatch):
        from speech.container import get_transliteration_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_transliteration_provider.cache_clear()

        response = await client.post(
            "/api/transliterate",
            json={"text": "नमस्ते", "source_language_code": "hi-IN", "target_language_code": "en-IN"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "not configured" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_transliterate_success(self):
        from speech.providers.transliteration import SarvamTransliterationProvider

        provider = SarvamTransliterationProvider(api_key="test-key", base_url="https://mock.local")
        mock_client = _mock_httpx_post({"transliterated_text": "namaste", "source_language_code": "hi-IN"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.transliterate("नमस्ते", source_language_code="hi-IN", target_language_code="en-IN")
            assert result["transliterated_text"] == "namaste"


class TestSTTTranslate:
    @pytest.mark.asyncio
    async def test_stt_translate_no_api_key_returns_informative_message(self, client, monkeypatch):
        from speech.container import get_translation_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "")
        get_translation_provider.cache_clear()

        files = {"file": ("test.webm", b"fake audio", "audio/webm")}
        response = await client.post(
            "/api/speech-to-text-translate",
            data={"language_code": "unknown"},
            files=files,
        )
        assert response.status_code == 200
        data = response.json()
        assert "not configured" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_stt_translate_success(self):
        from speech.providers.translation import SarvamTranslationProvider

        provider = SarvamTranslationProvider(api_key="test-key", base_url="https://mock.local")
        mock_client = _mock_httpx_post({"transcript": "Hello", "language_code": "es-ES"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            result = await provider.translate_speech(b"fake audio", language_code=None)
            assert result["transcript"] == "Hello"
