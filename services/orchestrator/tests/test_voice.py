import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSpeechClient:
    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        from orchestrator.services.speech_client import SpeechClient

        client = SpeechClient(base_url="https://mock.local", timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"transcript": "Hello", "language_code": "en-IN"}
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.transcribe(b"audio")
        assert result["transcript"] == "Hello"
        assert result["language_code"] == "en-IN"

    @pytest.mark.asyncio
    async def test_transcribe_error_returns_empty(self):
        from orchestrator.services.speech_client import SpeechClient

        client = SpeechClient(base_url="https://mock.local", timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.raise_for_status = MagicMock(side_effect=Exception("boom"))
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.transcribe(b"audio")
        assert result == {"transcript": "", "language_code": "unknown"}

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        from orchestrator.services.speech_client import SpeechClient

        client = SpeechClient(base_url="https://mock.local", timeout=5.0)
        wav = b"\x00\x01\x00\x01"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "audio_base64": base64.b64encode(wav).decode("utf-8"),
            "audio_bytes": len(wav),
            "format": "wav",
        }
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.synthesize("Hello", language_code="en-IN")
        assert result == wav

    @pytest.mark.asyncio
    async def test_synthesize_empty_returns_bytes_empty(self):
        from orchestrator.services.speech_client import SpeechClient

        client = SpeechClient(base_url="https://mock.local", timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "audio_base64": "",
            "audio_bytes": 0,
            "error": "no audio",
        }
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.synthesize("Hello")
        assert result == b""

    @pytest.mark.asyncio
    async def test_detect_language(self):
        from orchestrator.services.speech_client import SpeechClient

        client = SpeechClient(base_url="https://mock.local", timeout=5.0)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"language_code": "hi-IN", "script_code": "Deva"}
        mock_response.raise_for_status = MagicMock()
        client._client.post = AsyncMock(return_value=mock_response)

        result = await client.detect_language("नमस्ते")
        assert result["language_code"] == "hi-IN"
        assert result["script_code"] == "Deva"


class TestVoiceService:
    def _make_service(self):
        from orchestrator.services.agent_service import AgentService
        from orchestrator.services.speech_client import SpeechClient
        from orchestrator.services.voice_service import VoiceService

        speech = SpeechClient(base_url="https://mock.local", timeout=5.0)
        agent = MagicMock(spec=AgentService)
        return VoiceService(agent_service=agent, speech_client=speech), speech, agent

    @pytest.mark.asyncio
    async def test_audio_turn_full_pipeline(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTurnRequest

        service, speech, agent = self._make_service()
        wav = b"\x00\x01" * 100
        speech.transcribe = AsyncMock(return_value={
            "transcript": "What's the price?",
            "language_code": "en-IN",
        })
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r1",
            user_input="What's the price?",
            final_response="The enterprise plan costs $99.",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        response = await service.process_audio_turn(VoiceTurnRequest(
            audio_base64=base64.b64encode(b"audio").decode("utf-8"),
        ))
        assert response.transcript == "What's the price?"
        assert response.detected_language == "en-IN"
        assert "enterprise plan" in response.reply_text.lower()
        assert base64.b64decode(response.audio_base64) == wav

    @pytest.mark.asyncio
    async def test_audio_turn_empty_transcript_replies_default(self):
        from orchestrator.schemas.voice import VoiceTurnRequest

        service, speech, agent = self._make_service()
        speech.transcribe = AsyncMock(return_value={"transcript": "", "language_code": "unknown"})
        speech.synthesize = AsyncMock(return_value=b"")

        response = await service.process_audio_turn(VoiceTurnRequest(
            audio_base64=base64.b64encode(b"audio").decode("utf-8"),
        ))
        assert response.transcript == ""
        assert "couldn't understand" in response.reply_text.lower()
        agent.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_audio_turn_invalid_base64(self):
        from orchestrator.schemas.voice import VoiceTurnRequest

        service, speech, agent = self._make_service()
        response = await service.process_audio_turn(VoiceTurnRequest(audio_base64="not!!base64"))
        assert response.error is not None
        agent.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_text_turn_uses_detected_language(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTextTurnRequest

        service, speech, agent = self._make_service()
        wav = b"\x00\x01" * 50
        speech.detect_language = AsyncMock(return_value={
            "language_code": "hi-IN",
            "script_code": "Deva",
        })
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r2",
            user_input="नमस्ते",
            final_response="Hello!",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        response = await service.process_text_turn(VoiceTextTurnRequest(text="नमस्ते"))
        assert response.detected_language == "hi-IN"
        speech.synthesize.assert_awaited_once_with("Hello!", language_code="hi-IN")


class TestVoiceEndpoints:
    @pytest.mark.asyncio
    async def test_status_endpoint(self, client):
        from orchestrator.container import get_speech_client

        with patch.object(get_speech_client(), "health_check", new=AsyncMock(return_value=True)):
            response = await client.get("/api/voice/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["services"]["speech"] == "up"

    @pytest.mark.asyncio
    async def test_text_turn_endpoint(self, client):
        from orchestrator.container import get_speech_client

        wav = b"\x00\x01" * 50
        speech = get_speech_client()
        with patch.object(
            speech,
            "detect_language",
            new=AsyncMock(return_value={"language_code": "en-IN", "script_code": "Latn"}),
        ), patch.object(speech, "synthesize", new=AsyncMock(return_value=wav)):
            response = await client.post(
                "/api/voice/turn/text",
                json={"text": "Hello", "language_code": "en-IN"},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["request_id"]
        assert data["reply_text"]
        assert data["audio_base64"]
