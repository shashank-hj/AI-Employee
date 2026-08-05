from typing import Optional
from pydantic import BaseModel, Field


class LanguageDetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)


class LanguageDetectionResponse(BaseModel):
    language_code: str
    script_code: str
    error: Optional[str] = None
