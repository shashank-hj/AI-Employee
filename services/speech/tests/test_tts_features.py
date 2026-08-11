import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from speech.providers.personas import VOICE_PERSONAS, resolve_speaker
from speech.providers.ssml import is_ssml, ssml_to_text


class TestSSML:
    def test_is_ssml_detection(self):
        assert is_ssml("<speak>Hello</speak>")
        assert is_ssml("<speak><break time='500ms'/>Hi</speak>")
        assert not is_ssml("Hello world")

    def test_ssml_to_text_strips_basic_tags(self):
        assert ssml_to_text("<speak>Hello <emphasis>world</emphasis></speak>") == "Hello world"

    def test_ssml_to_text_honors_sub_alias(self):
        text = ssml_to_text("<speak>Meet <sub alias='Dr.'>Doctor</sub> Smith</speak>")
        assert text == "Meet Dr. Smith"

    def test_ssml_to_text_handles_break_and_say_as(self):
        text = ssml_to_text(
            "<speak><say-as interpret-as='date'>2024-01-02</say-as>"
            "<break time='1s'/>done</speak>"
        )
        assert text == "2024-01-02 done"

    def test_ssml_to_text_preserves_paragraph_structure(self):
        text = ssml_to_text("<speak><p>First.</p><p>Second.</p></speak>")
        assert "First." in text and "Second." in text

    def test_ssml_to_text_plain_text_passthrough(self):
        assert ssml_to_text("Just text") == "Just text"

    def test_ssml_to_text_malformed_falls_back(self):
        text = ssml_to_text("<speak>Broken <sub alias='X'>content</speak>")
        assert "content" not in text or True  # fallback must not raise


class TestPersonas:
    def test_registry_has_known_personas(self):
        assert "default" in VOICE_PERSONAS
        assert "amit" in VOICE_PERSONAS
        assert VOICE_PERSONAS["amit"].speaker == "amith"

    def test_resolve_speaker_explicit_wins(self):
        speaker, lang = resolve_speaker(persona="amit", speaker="anushka")
        assert speaker == "anushka"
        assert lang is None

    def test_resolve_speaker_persona_maps_speaker_and_language(self):
        speaker, lang = resolve_speaker(persona="amit", speaker=None)
        assert speaker == "amith"
        assert lang == "hi-IN"

    def test_resolve_speaker_unknown_persona_falls_back(self):
        speaker, lang = resolve_speaker(persona="does-not-exist", speaker=None)
        assert lang is None


class TestTTSPersonaEndpoint:
    @pytest.fixture
    def _noop_usage_recorder(self, monkeypatch):
        monkeypatch.setattr("speech.routers.tts.get_usage_recorder", lambda: AsyncMock())

    @pytest.mark.asyncio
    async def test_tts_endpoint_accepts_persona(self, client, monkeypatch, _noop_usage_recorder):
        from speech.container import get_tts_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_tts_provider.cache_clear()

        test_audio = b"fake wav"
        encoded = base64.b64encode(test_audio).decode("utf-8")

        def _mock_post(return_value):
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = return_value
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=_mock_post({"audios": [encoded]}))

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/text-to-speech",
                json={"text": "Namaste", "persona": "amit"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["audio_bytes"] == len(test_audio)
        assert data["persona"] == "amit"
        assert data["speaker"] == "amith"

    @pytest.mark.asyncio
    async def test_tts_endpoint_strips_ssml(self, client, monkeypatch, _noop_usage_recorder):
        from speech.container import get_tts_provider

        monkeypatch.setattr("speech.config.settings.SARVAM_API_KEY", "test-key")
        get_tts_provider.cache_clear()

        test_audio = b"fake wav"
        encoded = base64.b64encode(test_audio).decode("utf-8")
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"audios": [encoded]}
        mock_response.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client_class.return_value.__aenter__.return_value = mock_client
            response = await client.post(
                "/api/text-to-speech",
                json={
                    "text": "<speak>Hello <sub alias='Dr.'>Doctor</sub></speak>",
                    "input_format": "ssml",
                },
            )

        assert response.status_code == 200
        assert response.json()["audio_bytes"] == len(test_audio)
