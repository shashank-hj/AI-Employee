from typing import Any

from pydantic import BaseModel


class VoiceModelsResponse(BaseModel):
    stt: dict[str, Any]
    tts: dict[str, Any]
