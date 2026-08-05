import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile

from speech.container import get_translation_provider
from speech.providers.errors import SarvamAPIError
from speech.providers.translation import SarvamTranslationProvider
from speech.schemas.translation import (
    TextTranslateRequest,
    TextTranslateResponse,
    SpeechTranslateResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})


@router.post("/translate-text", response_model=TextTranslateResponse)
async def translate_text(
    request: TextTranslateRequest,
    provider: SarvamTranslationProvider = Depends(get_translation_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return TextTranslateResponse(
            translated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error="Translation is not configured. Set SARVAM_API_KEY.",
        )

    logger.info("translate_text_request", target=request.target_language_code)
    try:
        result = await provider.translate_text(
            request.text,
            target_language_code=request.target_language_code,
            source_language_code=request.source_language_code,
        )
        return TextTranslateResponse(
            translated_text=result["translated_text"],
            source_language_code=result["source_language_code"],
            target_language_code=request.target_language_code,
        )
    except SarvamAPIError as exc:
        logger.error("translate_text_sarve_error", error=exc.message, status=exc.status_code)
        return TextTranslateResponse(
            translated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error=exc.message,
        )
    except Exception as exc:
        logger.error("translate_text_failed", error=str(exc))
        return TextTranslateResponse(
            translated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error="Translation failed. Please try again.",
        )


@router.post("/speech-to-text-translate", response_model=SpeechTranslateResponse)
async def speech_to_text_translate(
    file: UploadFile = File(...),
    language_code: str = Form(default="unknown"),
    provider: SarvamTranslationProvider = Depends(get_translation_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return SpeechTranslateResponse(
            transcript="",
            language_code="unknown",
            error="Translation is not configured. Set SARVAM_API_KEY.",
        )

    audio_bytes = await file.read()
    logger.info("speech_translate_request", file_size=len(audio_bytes), filename=file.filename)

    try:
        result = await provider.translate_speech(
            audio_bytes,
            language_code if language_code != "unknown" else None,
        )
        return SpeechTranslateResponse(
            transcript=result["transcript"],
            language_code=result["language_code"],
        )
    except SarvamAPIError as exc:
        logger.error("speech_translate_sarve_error", error=exc.message, status=exc.status_code)
        return SpeechTranslateResponse(
            transcript="",
            language_code="unknown",
            error=exc.message,
        )
    except Exception as exc:
        logger.error("speech_translate_failed", error=str(exc))
        return SpeechTranslateResponse(
            transcript="",
            language_code="unknown",
            error="Speech translation failed. Please try again.",
        )
