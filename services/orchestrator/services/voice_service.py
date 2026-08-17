"""Voice-to-orchestrator bridge service (Layer 1 <-> Layer 2).

Handles a full voice turn: STT -> agent run -> TTS reply, replying natively in
the user's detected language (Sarvam is Indic-capable). Outbound translation is
available as a safety net only.
"""

import base64
import time
import uuid

import structlog

from orchestrator.schemas.agent import AgentRequest
from orchestrator.schemas.voice import (
    VoiceStatusResponse,
    VoiceTextTurnRequest,
    VoiceTextTurnResponse,
    VoiceTurnRequest,
    VoiceTurnResponse,
)
from orchestrator.services.agent_service import AgentService
from orchestrator.services.language_utils import detect_code_switch
from orchestrator.services.memory_client import MemoryClient
from orchestrator.services.speech_client import SpeechClient
from shared.schemas.channels import ChannelType

logger = structlog.get_logger(__name__)

_DEFAULT_REPLY = "Sorry, I couldn't understand the audio. Please try again."


class VoiceService:
    def __init__(
        self,
        agent_service: AgentService,
        speech_client: SpeechClient,
        memory_client: MemoryClient | None = None,
    ) -> None:
        self._agent_service = agent_service
        self._speech = speech_client
        self._memory = memory_client

    async def _resolve_session_language(
        self, session_id: str | None, explicit: str | None
    ) -> str | None:
        """Return the effective session language: explicit hint, else persisted."""
        if explicit:
            return explicit
        if session_id and self._memory:
            return await self._memory.get_session_language(session_id)
        return None

    async def _persist_session_language(
        self, session_id: str | None, user_id: str | None, language: str
    ) -> None:
        """Best-effort: persist the detected language on the session."""
        if not session_id or not language or language == "unknown" or not self._memory:
            return
        await self._memory.set_session_language(
            session_id=session_id,
            language_code=language,
            user_id=user_id,
        )

    async def process_audio_turn(self, request: VoiceTurnRequest) -> VoiceTurnResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        session_language = await self._resolve_session_language(
            request.session_id, request.language_code
        )

        try:
            audio_bytes = base64.b64decode(request.audio_base64)
        except Exception:
            return VoiceTurnResponse(
                request_id=request_id,
                error="Invalid base64 audio payload.",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        if not audio_bytes:
            return VoiceTurnResponse(
                request_id=request_id,
                error="Empty audio payload.",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # ── V1 STT (hinted by persisted session language when available) ──
        stt = await self._speech.transcribe(
            audio_bytes,
            language_code=session_language or request.language_code,
            model=request.stt_model,
        )
        transcript = stt.get("transcript", "").strip()
        detected_language = stt.get("language_code", request.language_code or "unknown")
        code_switch, primary_language = detect_code_switch(transcript, detected_language)
        await self._persist_session_language(request.session_id, request.user_id, detected_language)
        logger.info(
            "voice_turn_stt",
            request_id=request_id,
            transcript=transcript[:100],
            language=detected_language,
            code_switch=code_switch,
            session_language=session_language,
        )

        if not transcript:
            reply_text = _DEFAULT_REPLY
            reply_audio = await self._speech.synthesize(
                reply_text,
                language_code=session_language or request.language_code,
                model=request.tts_model,
                speaker=request.speaker,
            )
            return VoiceTurnResponse(
                request_id=request_id,
                transcript="",
                detected_language=detected_language,
                session_language=session_language,
                code_switch=code_switch,
                primary_language=primary_language,
                reply_text=reply_text,
                audio_base64=base64.b64encode(reply_audio).decode("utf-8") if reply_audio else "",
                audio_format="wav",
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        # ── Layer 2: run agent ──
        agent_response = await self._agent_service.run(AgentRequest(
            user_input=transcript,
            user_id=request.user_id,
            session_id=request.session_id,
            channel=ChannelType.VOICE,
            channel_message_id=request.channel_message_id,
            tenant_id=request.tenant_id,
            contact=request.contact,
            metadata=request.metadata,
        ))
        reply_text = agent_response.final_response or _DEFAULT_REPLY

        # ── V2 TTS: reply natively in the detected language ──
        reply_audio = await self._speech.synthesize(
            reply_text,
            language_code=detected_language,
            model=request.tts_model,
            speaker=request.speaker,
        )
        if not reply_audio:
            logger.warning(
                "voice_turn_tts_empty",
                request_id=request_id,
                language=detected_language,
            )

        return VoiceTurnResponse(
            request_id=request_id,
            transcript=transcript,
            detected_language=detected_language,
            session_language=session_language,
            code_switch=code_switch,
            primary_language=primary_language,
            reply_text=reply_text,
            audio_base64=base64.b64encode(reply_audio).decode("utf-8") if reply_audio else "",
            audio_format="wav",
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def process_text_turn(self, request: VoiceTextTurnRequest) -> VoiceTextTurnResponse:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())

        session_language = await self._resolve_session_language(
            request.session_id, request.language_code
        )

        # ── V4 language detection on the inbound text ──
        detected = await self._speech.detect_language(request.text)
        detected_language = request.language_code or detected.get("language_code", "unknown")
        code_switch, primary_language = detect_code_switch(request.text, detected_language)
        await self._persist_session_language(request.session_id, request.user_id, detected_language)

        agent_response = await self._agent_service.run(AgentRequest(
            user_input=request.text,
            user_id=request.user_id,
            session_id=request.session_id,
            channel=ChannelType.VOICE,
            channel_message_id=request.channel_message_id,
            tenant_id=request.tenant_id,
            contact=request.contact,
            metadata=request.metadata,
        ))
        reply_text = agent_response.final_response or _DEFAULT_REPLY

        reply_audio = await self._speech.synthesize(
            reply_text,
            language_code=detected_language,
            model=request.tts_model,
            speaker=request.speaker,
        )
        return VoiceTextTurnResponse(
            request_id=request_id,
            reply_text=reply_text,
            detected_language=detected_language,
            session_language=session_language,
            code_switch=code_switch,
            primary_language=primary_language,
            audio_base64=base64.b64encode(reply_audio).decode("utf-8") if reply_audio else "",
            audio_format="wav",
            duration_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    async def status(self) -> VoiceStatusResponse:
        speech_ok = await self._speech.health_check()
        return VoiceStatusResponse(
            status="ok" if speech_ok else "degraded",
            services={"speech": "up" if speech_ok else "down"},
        )
