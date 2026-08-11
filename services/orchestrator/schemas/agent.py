from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

from shared.schemas.channels import ChannelContact, ChannelType


class ToolResult(BaseModel):
    tool_name: str
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class ExecutionStep(BaseModel):
    step_index: int
    tool_name: str
    parameters: dict[str, Any]
    result: Optional[ToolResult] = None


class AgentRequest(BaseModel):
    user_input: str = Field(..., min_length=1, max_length=10000, description="The user's natural language request")
    user_id: Optional[str] = Field(default=None, description="Authenticated user identifier")
    session_id: Optional[str] = Field(default=None, description="Conversation session identifier for memory context")
    channel: ChannelType = Field(
        default=ChannelType.API,
        description="Channel the message arrived on (web, whatsapp, email, crm, api, sms)",
    )
    channel_message_id: Optional[str] = Field(default=None, description="Channel-native message id for echo/dedup")
    tenant_id: Optional[str] = Field(default=None, description="Tenant/workspace the request belongs to")
    contact: Optional[ChannelContact] = Field(default=None, description="Sender identity as known by the channel")
    metadata: Optional[dict[str, Any]] = Field(default=None, description="Channel-provided extra context")


class AgentResponse(BaseModel):
    request_id: str
    user_input: str
    final_response: str
    steps: list[ExecutionStep]
    execution_log: list[dict[str, Any]]
    completed_at: datetime
    duration_ms: float
    channel: ChannelType = Field(default=ChannelType.API, description="Echo of the inbound channel")
    channel_message_id: Optional[str] = Field(default=None, description="Echo of the inbound message id")
    tenant_id: Optional[str] = Field(default=None)

    model_config = {"from_attributes": True}
