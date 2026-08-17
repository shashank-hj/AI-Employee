import time

import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile, WebSocket, WebSocketDisconnect

from shared.usage.pricing import estimate_audio_seconds
from shared.usage.records import UsageRecord
from speech.container import get_stt_provider, get_usage_recorder
from speech.providers.stt import SarvamSTTProvider
from speech.providers.word_timestamps import estimate_word_timestamps
from speech.schemas.stt import SpeechToTextResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")


_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    language_code: str = Form(default="unknown"),
    mode: str = Form(default="transcribe"),
    model: str = Form(default=None),
    with_word_timestamps: bool = Form(default=False),
    provider: SarvamSTTProvider = Depends(get_stt_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return SpeechToTextResponse(
            transcript="Speech-to-text is not configured. Set SARVAM_API_KEY.",
            language_code="unknown",
        )

    audio_bytes = await file.read()
    logger.info(
        "stt_request_received",
        file_size=len(audio_bytes),
        filename=file.filename,
        mode=mode,
        model=model,
        with_word_timestamps=with_word_timestamps,
    )
    seconds = estimate_audio_seconds(audio_bytes)
    start = time.perf_counter()
    effective_model = model or provider._model

    try:
        result = await provider.transcribe(
            audio_bytes,
            language_code if language_code != "unknown" else None,
            mode=mode,
            model=model,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="stt",
            model=effective_model,
            unit="audio_seconds",
            input_units=seconds,
            output_units=0,
            duration_ms=round(duration_ms, 2),
        ))
        word_timestamps = None
        if with_word_timestamps:
            word_timestamps = estimate_word_timestamps(result["transcript"], seconds)
        return SpeechToTextResponse(
            transcript=result["transcript"],
            language_code=result["language_code"],
            word_timestamps=word_timestamps,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("stt_transcription_failed", error=str(exc))
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="stt",
            model=effective_model,
            unit="audio_seconds",
            input_units=seconds,
            status="error",
            error=str(exc)[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return SpeechToTextResponse(
            transcript="Sorry, I couldn't understand the audio. Please try again.",
            language_code="unknown",
        )


@router.websocket("/voice/stt/ws")
async def speech_to_text_websocket(websocket: WebSocket):
    """WebSocket STT for turn-based voice pipelines.

    The client first sends a JSON config message
    ``{"language_code": str?, "mode": str?, "with_word_timestamps": bool?}``,
    then one binary message per audio utterance. Each utterance is transcribed
    (batch Sarvam call) and the transcript is returned as a JSON message.
    """
    await websocket.accept()
    provider = get_stt_provider()
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        await websocket.send_json({"error": "STT not configured. Set SARVAM_API_KEY."})
        await websocket.close()
        return

    language_code: str | None = None
    mode = "transcribe"
    model: str | None = None
    with_word_timestamps = False

    try:
        config = await websocket.receive_json()
        language_code = config.get("language_code") or None
        mode = config.get("mode", "transcribe")
        model = config.get("model") or None
        with_word_timestamps = bool(config.get("with_word_timestamps", False))

        while True:
            message = await websocket.receive()
            message_type = message.get("type")
            if message_type == "websocket.disconnect":
                break
            if message_type != "websocket.receive_bytes":
                continue

            audio_bytes = message.get("bytes", b"")
            if not audio_bytes:
                await websocket.send_json({"transcript": "", "language_code": "unknown"})
                continue

            start = time.perf_counter()
            try:
                result = await provider.transcribe(
                    audio_bytes, language_code=language_code, mode=mode, model=model
                )
                duration_ms = (time.perf_counter() - start) * 1000
                await get_usage_recorder().record(UsageRecord(
                    service="speech",
                    category="speech",
                    operation="stt",
                    model=model or provider._model,
                    unit="audio_seconds",
                    input_units=estimate_audio_seconds(audio_bytes),
                    duration_ms=round(duration_ms, 2),
                ))
                payload = dict(result)
                if with_word_timestamps:
                    payload["word_timestamps"] = [
                        ts.model_dump()
                        for ts in estimate_word_timestamps(
                            result["transcript"], estimate_audio_seconds(audio_bytes)
                        )
                    ]
                await websocket.send_json(payload)
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error("stt_ws_transcription_failed", error=str(exc))
                await get_usage_recorder().record(UsageRecord(
                    service="speech",
                    category="speech",
                    operation="stt",
                    model=model or provider._model,
                    unit="audio_seconds",
                    input_units=estimate_audio_seconds(audio_bytes),
                    status="error",
                    error=str(exc)[:500],
                    duration_ms=round(duration_ms, 2),
                ))
                await websocket.send_json({
                    "transcript": "Sorry, I couldn't understand the audio. Please try again.",
                    "language_code": "unknown",
                })
    except WebSocketDisconnect:
        logger.info("stt_ws_disconnected")
    except Exception as exc:
        logger.error("stt_ws_error", error=str(exc))
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass
        await websocket.close()
