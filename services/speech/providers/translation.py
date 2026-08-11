import httpx
import structlog

from speech.config import settings
from speech.providers.errors import SarvamAPIError, extract_sarve_error
from speech.providers.translation_utils import apply_glossary, compute_translation_confidence

logger = structlog.get_logger(__name__)


class SarvamTranslationProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._timeout = timeout or settings.SARVAM_TIMEOUT

    async def translate_text(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str = "auto",
        pipeline_mode: str = "direct",
        glossary: dict[str, str] | None = None,
        with_confidence: bool = False,
    ) -> dict[str, str]:
        source_text = text[:2000]
        source_text, glossary_matches = apply_glossary(source_text, glossary)

        payload: dict = {
            "input": source_text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
        }
        if pipeline_mode == "pipeline":
            payload["pipeline"] = True

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/translate",
                json=payload,
                headers={"api-subscription-key": self._api_key},
            )
            if response.status_code != 200:
                error_msg = extract_sarve_error(response)
                logger.error(
                    "translation_api_error",
                    status=response.status_code,
                    error=error_msg,
                )
                raise SarvamAPIError(
                    message=error_msg,
                    status_code=response.status_code,
                )
            result = response.json()

        translated_text = result.get("translated_text", "")
        detected_source = result.get("source_language_code", source_language_code)

        confidence = None
        if with_confidence:
            confidence = compute_translation_confidence(
                source_text,
                translated_text,
                detected_source,
                target_language_code,
                glossary_matches,
            )

        logger.info(
            "translation_success",
            source=detected_source,
            target=target_language_code,
            pipeline_mode=pipeline_mode,
            glossary_matches=len(glossary_matches),
        )
        return {
            "translated_text": translated_text,
            "source_language_code": detected_source,
            "glossary_matches": glossary_matches,
            "confidence": confidence,
        }

    async def translate_speech(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
    ) -> dict[str, str]:
        files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
        data: dict[str, str] = {"model": "saaras:v3", "mode": "translate"}
        if language_code:
            data["language_code"] = language_code

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/speech-to-text",
                files=files,
                data=data,
                headers={"api-subscription-key": self._api_key},
            )
            if response.status_code != 200:
                error_msg = extract_sarve_error(response)
                logger.error(
                    "speech_translation_api_error",
                    status=response.status_code,
                    error=error_msg,
                )
                raise SarvamAPIError(
                    message=error_msg,
                    status_code=response.status_code,
                )
            result = response.json()

        logger.info(
            "speech_translation_success",
            language_code=result.get("language_code"),
        )
        return {
            "transcript": result.get("transcript", ""),
            "language_code": result.get("language_code", "unknown"),
        }
