import httpx
import structlog

from speech.config import settings

logger = structlog.get_logger(__name__)


class SarvamQuotaError(Exception):
    """Raised when Sarvam rejects a request due to exhausted credits."""


class SarvamSTTProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._model = model or settings.SARVAM_STT_MODEL
        self._timeout = timeout or settings.SARVAM_TIMEOUT

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
        mode: str = "transcribe",
        model: str | None = None,
    ) -> dict[str, str]:
        files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
        data: dict[str, str] = {"model": model or self._model, "mode": mode}
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
                logger.error(
                    "stt_api_error",
                    status=response.status_code,
                    body=response.text[:500],
                )
                if response.status_code == 402:
                    raise SarvamQuotaError(
                        "Sarvam credits exhausted (402). Add credits at "
                        "dashboard.sarvam.ai/billing."
                    )
                response.raise_for_status()
            result = response.json()

        logger.info(
            "stt_transcribe_success",
            language_code=result.get("language_code"),
            transcript_length=len(result.get("transcript", "")),
            mode=mode,
        )
        return {
            "transcript": result.get("transcript", ""),
            "language_code": result.get("language_code", "unknown"),
        }

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._base_url}/speech-to-text",
                    headers={"api-subscription-key": self._api_key},
                )
            return response.status_code < 500
        except Exception:
            return False
