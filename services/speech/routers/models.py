from fastapi import APIRouter

from speech.providers.models import (
    STT_DEFAULT_MODEL,
    STT_LANGUAGES,
    STT_MODELS,
    STT_MODES,
    TTS_DEFAULT_LANGUAGE,
    TTS_DEFAULT_MODEL,
    TTS_DEFAULT_SPEAKER,
    TTS_LANGUAGES,
    TTS_MODELS,
    TTS_SPEAKERS,
)
from speech.schemas.models import VoiceModelsResponse

router = APIRouter(prefix="/api", tags=["Voice Models"])


@router.get("/voice/models", response_model=VoiceModelsResponse)
async def voice_models() -> VoiceModelsResponse:
    """Return the catalog of selectable Sarvam STT/TTS models, speakers, and languages."""
    return VoiceModelsResponse(
        stt={
            "models": STT_MODELS,
            "default": STT_DEFAULT_MODEL,
            "modes": STT_MODES,
            "languages": STT_LANGUAGES,
        },
        tts={
            "models": TTS_MODELS,
            "default": TTS_DEFAULT_MODEL,
            "speakers": TTS_SPEAKERS,
            "default_speaker": TTS_DEFAULT_SPEAKER,
            "languages": TTS_LANGUAGES,
            "default_language": TTS_DEFAULT_LANGUAGE,
        },
    )
