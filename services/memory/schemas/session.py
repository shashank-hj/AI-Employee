from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str
    timestamp: Optional[datetime] = None


class SessionCreate(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Existing session ID to update, or omit to create new")
    user_id: Optional[str] = None
    messages: list[SessionMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[SessionMessage] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    ttl_seconds: int = 86400
