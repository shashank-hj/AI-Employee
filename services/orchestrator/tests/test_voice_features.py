import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.services.language_utils import detect_code_switch


class TestCodeSwitch:
    def test_pure_latin_no_code_switch(self):
        code_switch, primary = detect_code_switch("Hello, how are you?", "en-IN")
        assert code_switch is False
        assert primary == "en-IN"

    def test_pure_devanagari_no_code_switch(self):
        code_switch, primary = detect_code_switch("नमस्ते दुनिया", "hi-IN")
        assert code_switch is False
        assert primary == "hi-IN"

    def test_mixed_scripts_detected(self):
        code_switch, primary = detect_code_switch("मुझे loan चाहिए", "hi-IN")
        assert code_switch is True
        assert primary == "hi-IN"

    def test_mixed_scripts_maps_dominant_to_language(self):
        code_switch, primary = detect_code_switch("मुझे loan चाहिए")
        assert code_switch is True
        assert primary == "hi-IN"

    def test_mostly_latin_with_one_word_not_flagged(self):
        code_switch, primary = detect_code_switch("I need a loan sir")
        assert code_switch is False
        assert primary == "en-IN"

    def test_empty_text(self):
        assert detect_code_switch("") == (False, None)


class TestVoiceServiceSessionLanguage:
    def _make_service(self, memory=None):
        from orchestrator.services.agent_service import AgentService
        from orchestrator.services.speech_client import SpeechClient
        from orchestrator.services.voice_service import VoiceService

        speech = SpeechClient(base_url="https://mock.local", timeout=5.0)
        agent = MagicMock(spec=AgentService)
        return (
            VoiceService(agent_service=agent, speech_client=speech, memory_client=memory),
            speech,
            agent,
        )

    @pytest.mark.asyncio
    async def test_text_turn_persists_session_language(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTextTurnRequest

        memory = AsyncMock()
        memory.get_session_language = AsyncMock(return_value=None)
        service, speech, agent = self._make_service(memory=memory)
        wav = b"\x00\x01" * 50
        speech.detect_language = AsyncMock(
            return_value={"language_code": "hi-IN", "script_code": "Deva"}
        )
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r1",
            user_input="नमस्ते",
            final_response="Hello!",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        response = await service.process_text_turn(VoiceTextTurnRequest(
            text="नमस्ते",
            session_id="sess-1",
        ))

        assert response.detected_language == "hi-IN"
        assert response.session_language is None
        memory.set_session_language.assert_awaited_once_with(
            session_id="sess-1",
            language_code="hi-IN",
            user_id=None,
        )

    @pytest.mark.asyncio
    async def test_text_turn_resolves_session_language_hint(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTextTurnRequest

        memory = AsyncMock()
        memory.get_session_language = AsyncMock(return_value="hi-IN")
        service, speech, agent = self._make_service(memory=memory)
        wav = b"\x00\x01" * 50
        speech.detect_language = AsyncMock(
            return_value={"language_code": "en-IN", "script_code": "Latn"}
        )
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r2",
            user_input="hello",
            final_response="Hi!",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        response = await service.process_text_turn(VoiceTextTurnRequest(
            text="hello",
            session_id="sess-1",
        ))

        assert response.session_language == "hi-IN"
        memory.get_session_language.assert_awaited_once_with("sess-1")

    @pytest.mark.asyncio
    async def test_audio_turn_reports_code_switch(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTurnRequest

        service, speech, agent = self._make_service(memory=AsyncMock())
        service._memory.get_session_language = AsyncMock(return_value=None)
        wav = b"\x00\x01" * 100
        speech.transcribe = AsyncMock(return_value={
            "transcript": "मुझे loan चाहिए",
            "language_code": "hi-IN",
        })
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r3",
            user_input="मुझे loan चाहिए",
            final_response="ठीक है",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        response = await service.process_audio_turn(VoiceTurnRequest(
            audio_base64=base64.b64encode(b"audio").decode("utf-8"),
            session_id="sess-1",
        ))

        assert response.code_switch is True
        assert response.primary_language == "hi-IN"
        assert response.detected_language == "hi-IN"

    @pytest.mark.asyncio
    async def test_audio_turn_uses_session_language_hint_for_stt(self):
        from orchestrator.schemas.agent import AgentResponse
        from orchestrator.schemas.voice import VoiceTurnRequest

        memory = AsyncMock()
        memory.get_session_language = AsyncMock(return_value="hi-IN")
        service, speech, agent = self._make_service(memory=memory)
        wav = b"\x00\x01" * 100
        speech.transcribe = AsyncMock(return_value={
            "transcript": "नमस्ते",
            "language_code": "hi-IN",
        })
        speech.synthesize = AsyncMock(return_value=wav)
        agent.run = AsyncMock(return_value=AgentResponse(
            request_id="r4",
            user_input="नमस्ते",
            final_response="नमस्ते!",
            steps=[],
            execution_log=[],
            completed_at="2026-08-10T00:00:00Z",
            duration_ms=1.0,
        ))

        await service.process_audio_turn(VoiceTurnRequest(
            audio_base64=base64.b64encode(b"audio").decode("utf-8"),
            session_id="sess-1",
        ))

        speech.transcribe.assert_awaited_once()
        assert speech.transcribe.await_args.kwargs["language_code"] == "hi-IN"
