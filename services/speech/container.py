from functools import lru_cache

from speech.config import settings
from speech.providers.stt import SarvamSTTProvider
from speech.providers.tts import SarvamTTSProvider
from speech.providers.translation import SarvamTranslationProvider
from speech.providers.language_detection import SarvamLanguageDetectionProvider
from speech.providers.transliteration import SarvamTransliterationProvider


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
