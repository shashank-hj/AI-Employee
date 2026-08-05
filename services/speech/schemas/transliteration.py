from typing import Literal, Optional
from pydantic import BaseModel, Field


class TransliterationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    source_language_code: str
    target_language_code: str
    spoken_form: bool = False
    numerals_format: Literal["international", "native"] = "international"


class TransliterationResponse(BaseModel):
    transliterated_text: str
    source_language_code: str
    target_language_code: str
    error: Optional[str] = None
