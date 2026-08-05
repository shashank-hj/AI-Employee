import base64
import structlog
from fastapi import APIRouter, Depends

from speech.container import get_tts_provider
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

    logger.info("tts_request_received", text_length=len(request.text))

    try:
        audio_bytes = await provider.synthesize(
            request.text,
            language_code=request.language_code,
        )
    except SarvamAPIError as exc:
        logger.error("tts_sarve_error", error=exc.message, status=exc.status_code)
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error=exc.message,
        )
    except Exception as exc:
        error_msg = str(exc) or type(exc).__name__
        logger.error("tts_synthesis_failed", error=error_msg, exc_type=type(exc).__name__, text_length=len(request.text))
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error=error_msg,
        )

    if not audio_bytes:
        logger.error("tts_returned_empty", text_length=len(request.text))
        return TextToSpeechResponse(
            audio_base64="", audio_bytes=0, format="wav",
            error="Sarvam returned no audio for this text. Try shorter text.",
        )

    return TextToSpeechResponse(
        audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
        audio_bytes=len(audio_bytes),
        format="wav",
    )
