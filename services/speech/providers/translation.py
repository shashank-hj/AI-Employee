import httpx
import structlog

from speech.config import settings
from speech.providers.errors import SarvamAPIError, extract_sarve_error

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
    ) -> dict[str, str]:
        payload = {
            "input": text[:2000],
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
        }

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

        logger.info(
            "translation_success",
            source=result.get("source_language_code"),
            target=target_language_code,
        )
        return {
            "translated_text": result.get("translated_text", ""),
            "source_language_code": result.get("source_language_code", source_language_code),
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
