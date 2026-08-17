from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SessionMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str
    timestamp: datetime | None = None


class SessionCreate(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Existing session ID to update, or omit to create new",
    )
    user_id: str | None = None
    messages: list[SessionMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionUpdate(BaseModel):
    """Patch a session's mutable state (messages live in PostgreSQL)."""

    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class SummaryRequest(BaseModel):
    message_limit: int = Field(
        default=20,
        ge=1,
        le=200,
        description="How many recent messages to summarize",
    )
    store: bool = Field(
        default=True,
        description="Persist the summary as session context + long-term memory",
    )


class SessionResponse(BaseModel):
    session_id: str
    user_id: str | None = None
    messages: list[SessionMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    ttl_seconds: int = 86400
    expires_at: datetime | None = None


class SessionSummaryResponse(BaseModel):
    session_id: str
    summary: str
    message_count: int
