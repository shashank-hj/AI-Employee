import os
from functools import lru_cache

from speech.config import settings
from speech.providers.stt import SarvamSTTProvider
from speech.providers.tts import SarvamTTSProvider
from speech.providers.translation import SarvamTranslationProvider
from speech.providers.language_detection import SarvamLanguageDetectionProvider
from speech.providers.transliteration import SarvamTransliterationProvider
from shared.usage import UsageRecorder


@lru_cache()
def get_stt_provider() -> SarvamSTTProvider:
    return SarvamSTTProvider()


@lru_cache()
def get_tts_provider() -> SarvamTTSProvider:
    return SarvamTTSProvider()


@lru_cache()
def get_translation_provider() -> SarvamTranslationProvider:
    return SarvamTranslationProvider()


@lru_cache()
def get_language_detection_provider() -> SarvamLanguageDetectionProvider:
    return SarvamLanguageDetectionProvider()


@lru_cache()
def get_transliteration_provider() -> SarvamTransliterationProvider:
    return SarvamTransliterationProvider()


@lru_cache()
def get_usage_recorder() -> UsageRecorder:
    """Recorder that persists speech usage rows into the shared usage_events table."""
    if settings.USAGE_PRICING:
        os.environ["USAGE_PRICING"] = settings.USAGE_PRICING
    from speech.database.session import async_session

    return UsageRecorder(
        session_factory=async_session,
        service="speech",
        enabled=settings.USAGE_ENABLED,
    )
