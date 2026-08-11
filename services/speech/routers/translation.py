import time

import structlog
from fastapi import APIRouter, Depends, File, Form, UploadFile

from shared.usage.pricing import estimate_audio_seconds
from shared.usage.records import UsageRecord
from speech.container import get_translation_provider, get_usage_recorder
from speech.providers.errors import SarvamAPIError
from speech.providers.translation import SarvamTranslationProvider
from speech.schemas.translation import (
    SpeechTranslateResponse,
    TextTranslateRequest,
    TextTranslateResponse,
)

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})

TRANSLATE_MODEL = "sarvam-translation"


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

    logger.info(
        "translate_text_request",
        target=request.target_language_code,
        pipeline_mode=request.pipeline_mode,
        with_glossary=bool(request.glossary),
        with_confidence=request.with_confidence,
    )
    chars = len(request.text)
    start = time.perf_counter()
    try:
        result = await provider.translate_text(
            request.text,
            target_language_code=request.target_language_code,
            source_language_code=request.source_language_code,
            pipeline_mode=request.pipeline_mode,
            glossary=request.glossary,
            with_confidence=request.with_confidence,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="translate_text",
            model=TRANSLATE_MODEL,
            unit="characters",
            input_units=chars,
            duration_ms=round(duration_ms, 2),
        ))
        return TextTranslateResponse(
            translated_text=result["translated_text"],
            source_language_code=result["source_language_code"],
            target_language_code=request.target_language_code,
            pipeline_mode=request.pipeline_mode,
            glossary_matches=result.get("glossary_matches") or None,
            confidence=result.get("confidence"),
        )
    except SarvamAPIError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("translate_text_sarve_error", error=exc.message, status=exc.status_code)
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="translate_text",
            model=TRANSLATE_MODEL,
            unit="characters",
            input_units=chars,
            status="error",
            error=exc.message[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return TextTranslateResponse(
            translated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error=exc.message,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("translate_text_failed", error=str(exc))
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="translate_text",
            model=TRANSLATE_MODEL,
            unit="characters",
            input_units=chars,
            status="error",
            error=str(exc)[:500],
            duration_ms=round(duration_ms, 2),
        ))
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
    seconds = estimate_audio_seconds(audio_bytes)
    start = time.perf_counter()

    try:
        result = await provider.translate_speech(
            audio_bytes,
            language_code if language_code != "unknown" else None,
        )
        duration_ms = (time.perf_counter() - start) * 1000
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="speech_to_text_translate",
            model=TRANSLATE_MODEL,
            unit="audio_seconds",
            input_units=seconds,
            output_units=len(result.get("transcript", "")),
            duration_ms=round(duration_ms, 2),
        ))
        return SpeechTranslateResponse(
            transcript=result["transcript"],
            language_code=result["language_code"],
        )
    except SarvamAPIError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("speech_translate_sarve_error", error=exc.message, status=exc.status_code)
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="speech_to_text_translate",
            model=TRANSLATE_MODEL,
            unit="audio_seconds",
            input_units=seconds,
            status="error",
            error=exc.message[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return SpeechTranslateResponse(
            transcript="",
            language_code="unknown",
            error=exc.message,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("speech_translate_failed", error=str(exc))
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="speech_to_text_translate",
            model=TRANSLATE_MODEL,
            unit="audio_seconds",
            input_units=seconds,
            status="error",
            error=str(exc)[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return SpeechTranslateResponse(
            transcript="",
            language_code="unknown",
            error="Speech translation failed. Please try again.",
        )
