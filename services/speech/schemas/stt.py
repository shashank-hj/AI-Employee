from pydantic import BaseModel, Field


class SpeechToTextResponse(BaseModel):
    transcript: str
    language_code: str
