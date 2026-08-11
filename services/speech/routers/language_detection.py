import time

import structlog
from fastapi import APIRouter, Depends

from speech.container import get_language_detection_provider, get_usage_recorder
from speech.providers.errors import SarvamAPIError
from speech.providers.language_detection import SarvamLanguageDetectionProvider
from speech.schemas.language_detection import (
    LanguageDetectionRequest,
    LanguageDetectionResponse,
)
from shared.usage.records import UsageRecord

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})

DETECT_MODEL = "sarvam-language-detection"


@router.post("/detect-language", response_model=LanguageDetectionResponse)
async def detect_language(
    request: LanguageDetectionRequest,
    provider: SarvamLanguageDetectionProvider = Depends(get_language_detection_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return LanguageDetectionResponse(
            language_code="unknown",
            script_code="unknown",
            error="Language detection is not configured. Set SARVAM_API_KEY.",
        )

    logger.info("language_detection_request", text_length=len(request.text))
    chars = len(request.text)
    start = time.perf_counter()

    try:
        result = await provider.detect(request.text)
        duration_ms = (time.perf_counter() - start) * 1000
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="detect_language",
            model=DETECT_MODEL,
            unit="characters",
            input_units=chars,
            duration_ms=round(duration_ms, 2),
        ))
        return LanguageDetectionResponse(
            language_code=result["language_code"],
            script_code=result["script_code"],
        )
    except SarvamAPIError as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("language_detection_sarve_error", error=exc.message, status=exc.status_code)
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="detect_language",
            model=DETECT_MODEL,
            unit="characters",
            input_units=chars,
            status="error",
            error=exc.message[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return LanguageDetectionResponse(
            language_code="unknown",
            script_code="unknown",
            error=exc.message,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error("language_detection_failed", error=str(exc))
        await get_usage_recorder().record(UsageRecord(
            service="speech",
            category="speech",
            operation="detect_language",
            model=DETECT_MODEL,
            unit="characters",
            input_units=chars,
            status="error",
            error=str(exc)[:500],
            duration_ms=round(duration_ms, 2),
        ))
        return LanguageDetectionResponse(
            language_code="unknown",
            script_code="unknown",
            error="Language detection failed. Please try again.",
        )
