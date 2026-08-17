import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestVoiceModelsCatalog:
    @pytest.mark.asyncio
    async def test_voice_models_endpoint_returns_catalog(self, client):
        response = await client.get("/api/voice/models")
        assert response.status_code == 200
        data = response.json()
        assert "stt" in data and "tts" in data

        assert "saaras:v3" in data["stt"]["models"]
        assert data["stt"]["default"] == "saaras:v3"
        assert "transcribe" in data["stt"]["modes"]
        assert "en-IN" in data["stt"]["languages"]

        assert "bulbul:v2" in data["tts"]["models"]
        assert "bulbul:v3" in data["tts"]["models"]
        assert data["tts"]["default"] == "bulbul:v2"
        assert "anushka" in data["tts"]["speakers"]["bulbul:v2"]
        assert "shubh" in data["tts"]["speakers"]["bulbul:v3"]
        assert data["tts"]["default_speaker"]["bulbul:v3"] == "shubh"
        assert "en-IN" in data["tts"]["languages"]


class TestTTSSpeakerModelSelection:
    @pytest.mark.asyncio
    async def test_tts_endpoint_passes_model_and_speaker(
        self, client, monkeypatch
    ):
        from speech.container import get_tts_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        monkeypatch.setattr("speech.routers.tts.get_usage_recorder", lambda: AsyncMock())
        get_tts_provider.cache_clear()

        test_audio = b"fake wav"
        encoded = base64.b64encode(test_audio).decode("utf-8")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"audios": [encoded]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/text-to-speech",
                json={
                    "text": "Hello",
                    "model": "bulbul:v3",
                    "speaker": "shubh",
                    "language_code": "en-IN",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["audio_bytes"] == len(test_audio)
        assert data["speaker"] == "shubh"

        payload = mock_client.post.call_args.kwargs["json"]
        assert payload["model"] == "bulbul:v3"
        assert payload["speaker"] == "shubh"
        assert payload["target_language_code"] == "en-IN"

    @pytest.mark.asyncio
    async def test_tts_provider_model_override_defaults_speaker(self):
        from speech.providers.tts import SarvamTTSProvider

        provider = SarvamTTSProvider(api_key="test-key", base_url="https://mock.local")
        speaker, lang, _ = provider._resolve_voice(model="bulbul:v3")
        assert speaker == "shubh"
        assert lang == "en-IN"


class TestSTTSpeakerModelSelection:
    @pytest.mark.asyncio
    async def test_stt_endpoint_accepts_model_param(
        self, client, monkeypatch
    ):
        from speech.container import get_stt_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        monkeypatch.setattr("speech.routers.stt.get_usage_recorder", lambda: AsyncMock())
        get_stt_provider.cache_clear()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"transcript": "Hello", "language_code": "en-IN"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/speech-to-text",
                files={"file": ("test.webm", b"fake audio", "audio/webm")},
                data={"model": "saaras:v4", "mode": "transcribe"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["transcript"] == "Hello"

        post_kwargs = mock_client.post.call_args.kwargs
        assert post_kwargs["data"]["model"] == "saaras:v4"
        assert post_kwargs["data"]["mode"] == "transcribe"
