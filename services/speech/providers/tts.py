import base64
import io
import re
import wave

import httpx
import structlog

from speech.config import settings
from speech.providers.errors import SarvamAPIError, extract_sarve_error

logger = structlog.get_logger(__name__)

_TTS_CHUNK_SIZE = 200
_TTS_DEFAULT_TIMEOUT = 60.0


def _split_text_into_chunks(text: str, max_chars: int = _TTS_CHUNK_SIZE) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        s = sentence.strip()
        if not s or not re.search(r"[^\W_]", s):
            continue
        if current_len + len(s) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(s)
        current_len += len(s) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]


def _concatenate_wav_bytes(audio_segments: list[bytes]) -> bytes:
    if len(audio_segments) == 1:
        return audio_segments[0]
    if not audio_segments:
        return b""

    params = None
    all_frames: list[bytes] = []

    for segment in audio_segments:
        with wave.open(io.BytesIO(segment), "rb") as wf:
            if params is None:
                params = wf.getparams()
            all_frames.append(wf.readframes(wf.getnframes()))

    if params is None:
        return b""

    output = io.BytesIO()
    nchannels, sampwidth, framerate, _, comptype, compname = params
    total_frames = sum(len(f) // (nchannels * sampwidth) for f in all_frames)

    with wave.open(output, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.setnframes(total_frames)
        wf.setcomptype(comptype, compname)
        for frames in all_frames:
            wf.writeframes(frames)

    return output.getvalue()


class SarvamTTSProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        speaker: str | None = None,
        target_language_code: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.SARVAM_API_KEY
        self._base_url = (base_url or settings.SARVAM_BASE_URL).rstrip("/")
        self._speaker = speaker or settings.SARVAM_TTS_SPEAKER
        self._lang = target_language_code or settings.SARVAM_TTS_LANGUAGE
        self._timeout = timeout or _TTS_DEFAULT_TIMEOUT

    async def _synthesize_single(self, text: str, language_code: str) -> bytes:
        payload = {
            "text": text,
            "speaker": self._speaker,
            "target_language_code": language_code,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/text-to-speech",
                json=payload,
                headers={"api-subscription-key": self._api_key},
            )
            if response.status_code != 200:
                error_msg = extract_sarve_error(response)
                logger.error("tts_api_error", status=response.status_code, error=error_msg)
                raise SarvamAPIError(message=error_msg, status_code=response.status_code)

            result = response.json()

        audios = result.get("audios", [])
        if not audios:
            logger.warning("tts_no_audio_for_chunk", text_preview=text[:80], response_keys=list(result.keys()))
            return b""

        return base64.b64decode(audios[0])

    async def synthesize(
        self,
        text: str,
        language_code: str | None = None,
    ) -> bytes:
        lang = language_code or self._lang

        if not text.strip():
            return b""

        chunks = _split_text_into_chunks(text, _TTS_CHUNK_SIZE)
        if len(chunks) > 1:
            logger.info("tts_chunking", text_length=len(text), num_chunks=len(chunks))

        audio_segments: list[bytes] = []
        for i, chunk in enumerate(chunks):
            try:
                segment = await self._synthesize_single(chunk, lang)
                if segment:
                    audio_segments.append(segment)
            except Exception as exc:
                logger.warning("tts_chunk_failed", chunk_index=i, error=str(exc) or type(exc).__name__)

        combined = _concatenate_wav_bytes(audio_segments)
        logger.info(
            "tts_synthesize_success",
            text_length=len(text),
            chunks=len(chunks),
            successful=len(audio_segments),
            audio_bytes=len(combined),
        )
        return combined

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._base_url}/text-to-speech",
                    headers={"api-subscription-key": self._api_key},
                )
            return response.status_code < 500
        except Exception:
            return False
