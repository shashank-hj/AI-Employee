import httpx
import structlog

from speech.config import settings
from speech.providers.errors import SarvamAPIError, extract_sarve_error

logger = structlog.get_logger(__name__)


class SarvamTransliterationProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._timeout = timeout or settings.SARVAM_TIMEOUT

    async def transliterate(
        self,
        text: str,
        source_language_code: str,
        target_language_code: str,
        spoken_form: bool = False,
        numerals_format: str = "international",
    ) -> dict[str, str]:
        payload = {
            "input": text[:1000],
            "source_language_code": source_language_code,
            "target_language_code": target_language_code,
            "spoken_form": spoken_form,
            "numerals_format": numerals_format,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/transliterate",
                json=payload,
                headers={"api-subscription-key": self._api_key},
            )
            if response.status_code != 200:
                error_msg = extract_sarve_error(response)
                logger.error(
                    "transliteration_api_error",
                    status=response.status_code,
                    error=error_msg,
                )
                raise SarvamAPIError(
                    message=error_msg,
                    status_code=response.status_code,
                )
            result = response.json()

        logger.info(
            "transliteration_success",
            source=source_language_code,
            target=target_language_code,
        )
        return {
            "transliterated_text": result.get("transliterated_text", ""),
            "source_language_code": result.get("source_language_code", source_language_code),
        }
