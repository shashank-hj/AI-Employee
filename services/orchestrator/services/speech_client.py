"""HTTP client for the Speech Service (V1-V5 APIs)."""

from typing import Any

import httpx
import structlog

from orchestrator.config import settings

logger = structlog.get_logger(__name__)


class SpeechClient:
    """Async HTTP client for the Speech micro-service."""

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = (base_url or settings.SPEECH_URL).rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout),
        )

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: str | None = None,
        mode: str = "transcribe",
    ) -> dict[str, str]:
        """STT: transcribe audio to text via the speech service."""
        files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
        data: dict[str, str] = {"mode": mode}
        if language_code:
            data["language_code"] = language_code
        try:
            response = await self._client.post("/api/speech-to-text", files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return {
                "transcript": result.get("transcript", ""),
                "language_code": result.get("language_code", "unknown"),
            }
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "speech_client_stt_failed",
                status=exc.response.status_code,
                detail=exc.response.text[:200],
            )
            return {"transcript": "", "language_code": "unknown"}
        except Exception as exc:
            logger.error("speech_client_stt_error", error=str(exc))
            return {"transcript": "", "language_code": "unknown"}

    async def synthesize(
        self,
        text: str,
        language_code: str | None = None,
    ) -> bytes:
        """TTS: synthesize text to wav audio bytes via the speech service."""
        payload: dict[str, Any] = {"text": text}
        if language_code:
            payload["language_code"] = language_code
        try:
            response = await self._client.post("/api/text-to-speech", json=payload)
            response.raise_for_status()
            result = response.json()
            if not result.get("audio_base64"):
                logger.warning("speech_client_tts_empty", detail=result.get("error"))
                return b""
            import base64
            return base64.b64decode(result["audio_base64"])
        except Exception as exc:
            logger.error("speech_client_tts_error", error=str(exc))
            return b""

    async def detect_language(self, text: str) -> dict[str, str]:
        """Language detection (V4)."""
        try:
            response = await self._client.post("/api/detect-language", json={"text": text[:1000]})
            response.raise_for_status()
            result = response.json()
            return {
                "language_code": result.get("language_code", "unknown"),
                "script_code": result.get("script_code", "unknown"),
            }
        except Exception as exc:
            logger.error("speech_client_language_detect_error", error=str(exc))
            return {"language_code": "unknown", "script_code": "unknown"}

    async def translate_text(
        self,
        text: str,
        target_language_code: str,
        source_language_code: str = "auto",
    ) -> str:
        """Translation (V3) — used as an outbound safety net when needed."""
        try:
            response = await self._client.post(
                "/api/translate-text",
                json={
                    "text": text[:2000],
                    "source_language_code": source_language_code,
                    "target_language_code": target_language_code,
                },
            )
            response.raise_for_status()
            result = response.json()
            return result.get("translated_text", text)
        except Exception as exc:
            logger.error("speech_client_translate_error", error=str(exc))
            return text

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/health", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._client.aclose()
