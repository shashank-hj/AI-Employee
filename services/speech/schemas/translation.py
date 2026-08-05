from typing import Optional
from pydantic import BaseModel, Field


class TextTranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source_language_code: str = "auto"
    target_language_code: str


class TextTranslateResponse(BaseModel):
    translated_text: str
    source_language_code: str
    target_language_code: str
    error: Optional[str] = None


class SpeechTranslateResponse(BaseModel):
    transcript: str
    language_code: str
    error: Optional[str] = None
