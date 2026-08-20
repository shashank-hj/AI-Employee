"""Canonical channel schemas (CH1).

These define the platform's common representation for inbound/outbound messages
regardless of the transport (web, WhatsApp, email, CRM, API, SMS). Every channel
connector normalizes its native payload into a :class:`ChannelMessage` before it
reaches the orchestrator's agent entrypoint.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ChannelType(StrEnum):
    """Transport a message arrives on. Used as the canonical channel key."""

    WEB = "web"
    WHATSAPP = "whatsapp"
    EMAIL = "email"
    CRM = "crm"
    API = "api"
    SMS = "sms"
    VOICE = "voice"
    SAMVAAD = "samvaad"


class ChannelContact(BaseModel):
    """End-user identity as known by the channel (best-effort, channel-dependent)."""

    external_id: str | None = Field(
        default=None,
        description="Channel-native user id (WhatsApp number, CRM contact id)",
    )
    name: str | None = Field(default=None, max_length=255)
    email: str | None = Field(default=None, max_length=320)
    phone: str | None = Field(default=None, max_length=32)
    user_id: str | None = Field(
        default=None,
        description="Platform user id when the contact is a known user",
    )


class ChannelMessage(BaseModel):
    """Normalized inbound message. Adapters convert channel payloads into this shape."""

    message_id: str | None = Field(
        default=None,
        description="Channel-native message id for deduplication/echo",
    )
    channel: ChannelType = ChannelType.API
    sender: ChannelContact = Field(default_factory=ChannelContact)
    tenant_id: str | None = Field(default=None, max_length=128)
    text: str = Field(..., min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] | None = Field(default=None)

    @property
    def canonical_user_id(self) -> str | None:
        """Best available platform user id for this message."""
        return self.sender.user_id or self.sender.external_id


class ChannelResponse(BaseModel):
    """Response envelope returned to a channel after processing an inbound message."""

    message_id: str | None = None
    channel: ChannelType = ChannelType.API
    reply_to: str | None = Field(
        default=None,
        description="The inbound message id this response answers",
    )
    final_response: str
    request_id: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] | None = None
