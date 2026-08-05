import base64

from typing import Optional

from pydantic import BaseModel, Field


class TextToSpeechRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language_code: str = "en-IN"


class TextToSpeechResponse(BaseModel):
    audio_base64: str
    audio_bytes: int
    format: str = "wav"
    error: Optional[str] = None
