from pydantic import BaseModel

from speech.config import settings


class VoicePersona(BaseModel):
    """Named voice persona mapping to a Sarvam speaker and default language."""

    name: str
    speaker: str
    language_code: str
    description: str = ""


VOICE_PERSONAS: dict[str, VoicePersona] = {
    persona.name: persona
    for persona in [
        VoicePersona(
            name="default",
            speaker="anushka",
            language_code="en-IN",
            description="Default female assistant (en-IN)",
        ),
        VoicePersona(
            name="assistant",
            speaker="anushka",
            language_code="en-IN",
            description="Alias for the default assistant voice",
        ),
        VoicePersona(
            name="amit",
            speaker="amith",
            language_code="hi-IN",
            description="Male Hindi assistant",
        ),
        VoicePersona(
            name="neha",
            speaker="reva",
            language_code="hi-IN",
            description="Female Hindi assistant",
        ),
        VoicePersona(
            name="ramya",
            speaker="ramya",
            language_code="ta-IN",
            description="Female Tamil assistant",
        ),
        VoicePersona(
            name="nisha",
            speaker="meera",
            language_code="ta-IN",
            description="Female Tamil assistant",
        ),
        VoicePersona(
            name="gagan",
            speaker="gagan",
            language_code="kn-IN",
            description="Male Kannada assistant",
        ),
        VoicePersona(
            name="vivek",
            speaker="nikhil",
            language_code="en-IN",
            description="Male English assistant",
        ),
        VoicePersona(
            name="deepa",
            speaker="deepa",
            language_code="mr-IN",
            description="Female Marathi assistant",
        ),
    ]
}


def resolve_speaker(persona: str | None, speaker: str | None) -> tuple[str, str | None]:
    """Resolve a persona name to a (speaker, language_code_override) pair.

    An explicit ``speaker`` argument always wins. If ``persona`` is given and
    known, its speaker + default language are used. Otherwise the configured
    defaults are kept.
    """
    if speaker:
        return speaker, None

    if persona:
        match = VOICE_PERSONAS.get(persona)
        if match:
            return match.speaker, match.language_code

    return settings.SARVAM_TTS_SPEAKER, None
