import structlog
from fastapi import APIRouter, Depends

from speech.container import get_transliteration_provider
from speech.providers.errors import SarvamAPIError
from speech.providers.transliteration import SarvamTransliterationProvider
from speech.schemas.transliteration import TransliterationRequest, TransliterationResponse

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api")

_PLACEHOLDER_API_KEYS = frozenset({"", "your_sarvam_api_key_here"})


@router.post("/transliterate", response_model=TransliterationResponse)
async def transliterate(
    request: TransliterationRequest,
    provider: SarvamTransliterationProvider = Depends(get_transliteration_provider),
):
    if not provider._api_key or provider._api_key in _PLACEHOLDER_API_KEYS:
        return TransliterationResponse(
            transliterated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error="Transliteration is not configured. Set SARVAM_API_KEY.",
        )

    logger.info("transliteration_request", source=request.source_language_code, target=request.target_language_code)

    try:
        result = await provider.transliterate(
            request.text,
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            spoken_form=request.spoken_form,
            numerals_format=request.numerals_format,
        )
        return TransliterationResponse(
            transliterated_text=result["transliterated_text"],
            source_language_code=result["source_language_code"],
            target_language_code=request.target_language_code,
        )
    except SarvamAPIError as exc:
        logger.error("transliteration_sarve_error", error=exc.message, status=exc.status_code)
        return TransliterationResponse(
            transliterated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error=exc.message,
        )
    except Exception as exc:
        logger.error("transliteration_failed", error=str(exc))
        return TransliterationResponse(
            transliterated_text="",
            source_language_code=request.source_language_code,
            target_language_code=request.target_language_code,
            error="Transliteration failed. Please try again.",
        )
