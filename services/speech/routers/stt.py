import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile

from speech.container import get_stt_provider
from speech.providers.stt import SarvamSTTProvider
from speech.schemas.stt import SpeechToTextResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")


_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})


@router.post("/speech-to-text", response_model=SpeechToTextResponse)
async def speech_to_text(
    file: UploadFile = File(...),
    language_code: str = Form(default="unknown"),
    mode: str = Form(default="transcribe"),
    provider: SarvamSTTProvider = Depends(get_stt_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return SpeechToTextResponse(
            transcript="Speech-to-text is not configured. Set SARVAM_API_KEY.",
            language_code="unknown",
        )

    audio_bytes = await file.read()
    logger.info("stt_request_received", file_size=len(audio_bytes), filename=file.filename, mode=mode)

    try:
        result = await provider.transcribe(
            audio_bytes,
            language_code if language_code != "unknown" else None,
            mode=mode,
        )
        return SpeechToTextResponse(
            transcript=result["transcript"],
            language_code=result["language_code"],
        )
    except Exception as exc:
        logger.error("stt_transcription_failed", error=str(exc))
        return SpeechToTextResponse(
            transcript="Sorry, I couldn't understand the audio. Please try again.",
            language_code="unknown",
        )
