import httpx
import structlog

from speech.config import settings
from speech.providers.errors import SarvamAPIError, extract_sarve_error

logger = structlog.get_logger(__name__)


class SarvamLanguageDetectionProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._timeout = timeout or settings.SARVAM_TIMEOUT

    async def detect(self, text: str) -> dict[str, str]:
        payload = {"input": text[:1000]}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/text-lid",
                json=payload,
                headers={"api-subscription-key": self._api_key},
            )
            if response.status_code != 200:
                error_msg = extract_sarve_error(response)
                logger.error(
                    "language_detection_api_error",
                    status=response.status_code,
                    error=error_msg,
                )
                raise SarvamAPIError(
                    message=error_msg,
                    status_code=response.status_code,
                )
            result = response.json()

        logger.info(
            "language_detection_success",
            language_code=result.get("language_code"),
            script_code=result.get("script_code"),
        )
        return {
            "language_code": result.get("language_code", "unknown"),
            "script_code": result.get("script_code", "unknown"),
        }
