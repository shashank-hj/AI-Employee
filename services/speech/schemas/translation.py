from typing import Literal

from pydantic import BaseModel, Field


class TextTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_language_code: str = "auto"
    target_language_code: str
    pipeline_mode: Literal["direct", "pipeline"] = "direct"
    glossary: dict[str, str] | None = Field(
        default=None,
        description="Client-side whole-word term substitutions applied before translation.",
    )
    with_confidence: bool = False


class TextTranslateResponse(BaseModel):
    translated_text: str
    source_language_code: str
    target_language_code: str
    pipeline_mode: str = "direct"
    glossary_matches: list[str] | None = None
    confidence: float | None = None
    error: str | None = None


class SpeechTranslateResponse(BaseModel):
    transcript: str
    language_code: str
    error: str | None = None
