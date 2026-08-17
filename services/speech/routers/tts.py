import base64
import json
import time

import structlog
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from shared.usage.records import UsageRecord
from speech.container import get_tts_provider, get_usage_recorder
from speech.providers.errors import SarvamAPIError
from speech.providers.tts import SarvamTTSProvider
from speech.schemas.tts import TextToSpeechRequest, TextToSpeechResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})


@router.post("/text-to-speech", response_model=TextToSpeechResponse)
async def text_to_speech(
    request: TextToSpeechRequest,
    provider: SarvamTTSProvider = Depends(get_tts_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error="TTS not configured. Set SARVAM_API_KEY.",
        )

    logger.info(
        "tts_request_received",
        text_length=len(request.text),
        input_format=request.input_format,
        persona=request.persona,
        speaker=request.speaker,
        model=request.model,
    )
    chars = len(request.text)
    start = time.perf_counter()
    effective_model = request.model or provider._model

    try:
        audio_bytes = await provider.synthesize(
            request.text,
            language_code=request.language_code,
            persona=request.persona,
            speaker=request.speaker,
            input_format=request.input_format,
            model=request.model,
        )
    except SarvamAPIError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("tts_sarve_error", error=exc.message, status=exc.status_code)
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="tts",
            model=effective_model,
            unit="characters",
            input_units=chars,
            status="error",
            error=exc.message[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error=exc.message,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        error_msg = str(exc) or type(exc).__name__
        logger.error(
            "tts_synthesis_failed",
            error=error_msg,
            exc_type=type(exc).__name__,
            text_length=len(request.text),
        )
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="tts",
            model=effective_model,
            unit="characters",
            input_units=chars,
            status="error",
            error=error_msg[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error=error_msg,
        )

    duration_ms = (time.perf_counter() - start) * 1000
    if not audio_bytes:
        logger.error("tts_returned_empty", text_length=len(request.text))
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="tts",
            model=effective_model,
            unit="characters",
            input_units=chars,
            status="error",
            error="Sarvam returned no audio for this text. Try shorter text.",
            duration_ms=round(duration_ms, 2),
        ))
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error="Sarvam returned no audio for this text. Try shorter text.",
        )

    await get_usage_recorder().record(UsageRecord(
        service="speech",
        category="speech",
        operation="tts",
        model=effective_model,
        unit="characters",
        input_units=chars,
        duration_ms=round(duration_ms, 2),
    ))
    effective_speaker, _, resolved_persona = provider._resolve_voice(
        request.persona,
        request.speaker,
        request.language_code,
        request.model,
    )
    return TextToSpeechResponse(
        audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_bytes=len(audio_bytes),
        format="wav",
        persona=resolved_persona,
        speaker=effective_speaker,
    )


@router.post("/text-to-speech/stream")
async def text_to_speech_stream(
    request: TextToSpeechRequest,
    provider: SarvamTTSProvider = Depends(get_tts_provider),
):
    """Server-Sent-Events streaming TTS.

    Yields one ``data: {json}`` event per synthesized audio chunk followed by a
    terminal ``data: [DONE]`` event.
    """
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return StreamingResponse(
            iter([
                _sse(json.dumps({"error": "TTS not configured. Set SARVAM_API_KEY."})),
                _sse("[DONE]"),
            ]),
            media_type="text/event-stream",
        )

    chars = len(request.text)
    start = time.perf_counter()
    effective_model = request.model or provider._model

    async def event_generator():
        chunks_sent = 0
        try:
            async for chunk in provider.synthesize_stream(
                request.text,
                language_code=request.language_code,
                persona=request.persona,
                speaker=request.speaker,
                input_format=request.input_format,
                model=request.model,
            ):
                chunks_sent += 1
                yield _sse(json.dumps(chunk))
            yield _sse("[DONE]")
            duration_ms = (time.perf_counter() - start) * 1000
            await get_usage_recorder().record(UsageRecord(
                service="speech",
                category="speech",
                operation="tts_stream",
                model=effective_model,
                unit="characters",
                input_units=chars,
                output_units=chunks_sent,
                duration_ms=round(duration_ms, 2),
            ))
        except SarvamAPIError as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            logger.error("tts_stream_sarvam_error", error=exc.message, status=exc.status_code)
            await get_usage_recorder().record(UsageRecord(
                service="speech",
                category="speech",
                operation="tts_stream",
                model=effective_model,
                unit="characters",
                input_units=chars,
                status="error",
                error=exc.message[:500],
                duration_ms=round(duration_ms, 2),
            ))
            yield _sse(json.dumps({"error": exc.message}))
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            error_msg = str(exc) or type(exc).__name__
            logger.error("tts_stream_failed", error=error_msg)
            await get_usage_recorder().record(UsageRecord(
                service="speech",
                category="speech",
                operation="tts_stream",
                model=effective_model,
                unit="characters",
                input_units=chars,
                status="error",
                error=error_msg[:500],
                duration_ms=round(duration_ms, 2),
            ))
            yield _sse(json.dumps({"error": error_msg}))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.websocket("/voice/tts/ws")
async def text_to_speech_websocket(websocket: WebSocket):
    """WebSocket TTS for turn-based voice pipelines.

    Client sends ``{"text": str, "language_code": str?}`` once. The server then
    streams one JSON message per synthesized chunk and closes with ``{done: true}``.
    """
    await websocket.accept()
    provider = get_tts_provider()
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        await websocket.send_json({"error": "TTS not configured. Set SARVAM_API_KEY."})
        await websocket.close()
        return

    try:
        data = await websocket.receive_json()
        text = data.get("text", "")
        language_code = data.get("language_code")
        model = data.get("model")
        persona = data.get("persona")
        speaker = data.get("speaker")
        input_format = data.get("input_format", "text")

        if not text.strip():
            await websocket.send_json({"error": "text is required"})
            await websocket.close()
            return

        async for chunk in provider.synthesize_stream(
            text,
            language_code=language_code,
            model=model,
            persona=persona,
            speaker=speaker,
            input_format=input_format,
        ):
            await websocket.send_json(chunk)
        await websocket.send_json({"done": True})
    except WebSocketDisconnect:
        logger.info("tts_ws_disconnected")
    except Exception as exc:
        logger.error("tts_ws_error", error=str(exc))
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
        await websocket.close()


def _sse(data: str) -> str:
    return f"data: {data}\n\n"

