"""HTTP client for the Speech Service (V1-V5 APIs)."""

from typing import Any

import httpx
import structlog

from orchestrator.config import settings

logger = structlog.get_logger(__name__)

# Mirrors the speech service /api/voice/models catalog so the orchestrator can
# still answer model-discovery requests when the speech service is unreachable.
_FALLBACK_VOICE_MODELS: dict[str, Any] = {
    "stt": {
        "models": ["saaras:v3", "saaras:v4", "sarvam-1", "sarvam-1-20x-hi-en-2025-03-04"],
        "default": "saaras:v3",
        "modes": ["transcribe", "translate", "verbatim", "translit", "codemix"],
        "languages": [
            "unknown", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
            "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN", "ur-IN", "ne-IN", "kok-IN",
            "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN",
        ],
    },
    "tts": {
        "models": ["bulbul:v2", "bulbul:v3"],
        "default": "bulbul:v2",
        "speakers": {
            "bulbul:v2": ["anushka", "manisha", "vidya", "arya", "abhilash", "karun", "hitesh"],
            "bulbul:v3": [
                "shubh", "aditya", "ritu", "priya", "neha", "rahul", "pooja", "rohan",
                "simran", "kavya", "amit", "dev", "ishita", "shreya", "ratan", "varun",
                "manan", "sumit", "roopa", "kabir", "aayan", "ashutosh", "advait",
                "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay", "shruti",
                "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
            ],
        },
        "default_speaker": {"bulbul:v2": "anushka", "bulbul:v3": "shubh"},
        "languages": [
            "bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN",
            "pa-IN", "ta-IN", "te-IN",
        ],
        "default_language": "en-IN",
    },
}


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
        model: str | None = None,
    ) -> dict[str, str]:
        """STT: transcribe audio to text via the speech service."""
        files = {"file": ("audio.webm", audio_bytes, "audio/webm")}
        data: dict[str, str] = {"mode": mode}
        if model:
            data["model"] = model
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
        model: str | None = None,
        speaker: str | None = None,
    ) -> bytes:
        """TTS: synthesize text to wav audio bytes via the speech service."""
        payload: dict[str, Any] = {"text": text}
        if language_code:
            payload["language_code"] = language_code
        if model:
            payload["model"] = model
        if speaker:
            payload["speaker"] = speaker
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

    async def list_models(self) -> dict[str, Any]:
        """Fetch the selectable Sarvam STT/TTS models catalog from the speech service."""
        try:
            response = await self._client.get("/api/voice/models", timeout=10.0)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.warning("speech_client_models_fetch_failed", error=str(exc))
            return _FALLBACK_VOICE_MODELS

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
