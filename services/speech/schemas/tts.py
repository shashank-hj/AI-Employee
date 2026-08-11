from typing import Literal

from pydantic import BaseModel, Field


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language_code: str = "en-IN"
    input_format: Literal["text", "ssml"] = "text"
    persona: str | None = Field(
        default=None, description="Named voice persona (see personas registry)."
    )
    speaker: str | None = Field(default=None, description="Raw Sarvam speaker override.")


class TextToSpeechResponse(BaseModel):
    audio_base64: str
    audio_bytes: int
    format: str = "wav"
    persona: str | None = None
    speaker: str | None = None
    error: str | None = None
