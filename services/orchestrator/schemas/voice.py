from typing import Any

from pydantic import BaseModel, Field

from shared.schemas.channels import ChannelContact, ChannelType


class VoiceTurnRequest(BaseModel):
    """A single voice turn: audio in, audio + text out.

    ``audio_base64`` is the raw audio bytes (webm/wav) from the caller.
    ``language_code`` is optional; when absent the speech service auto-detects.
    """

    audio_base64: str = Field(..., description="Base64-encoded inbound audio (webm/wav)")
    user_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    language_code: str | None = Field(
        default=None, description="Hint for STT/TTS; auto-detected when omitted"
    )
    channel_message_id: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    contact: ChannelContact | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class VoiceTurnResponse(BaseModel):
    request_id: str
    transcript: str = ""
    detected_language: str = "unknown"
    session_language: str | None = None
    code_switch: bool = False
    primary_language: str | None = None
    reply_text: str = ""
    audio_base64: str = ""
    audio_format: str = "wav"
    duration_ms: float = 0.0
    error: str | None = None


class VoiceTextTurnRequest(BaseModel):
    """Text-only voice turn for testing/debugging the bridge without audio."""

    text: str = Field(..., min_length=1, max_length=10000)
    user_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    language_code: str | None = Field(default=None)
    channel_message_id: str | None = Field(default=None)
    tenant_id: str | None = Field(default=None)
    contact: ChannelContact | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)


class VoiceTextTurnResponse(BaseModel):
    request_id: str
    reply_text: str = ""
    detected_language: str = "unknown"
    session_language: str | None = None
    code_switch: bool = False
    primary_language: str | None = None
    audio_base64: str = ""
    audio_format: str = "wav"
    duration_ms: float = 0.0
    error: str | None = None


class VoiceStatusResponse(BaseModel):
    status: str
    channel: ChannelType = ChannelType.VOICE
    services: dict[str, str] = Field(default_factory=dict)
