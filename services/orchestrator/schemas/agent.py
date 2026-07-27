from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


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


class AgentResponse(BaseModel):
    request_id: str
    user_input: str
    final_response: str
    steps: list[ExecutionStep]
    execution_log: list[dict[str, Any]]
    completed_at: datetime
    duration_ms: float

    model_config = {"from_attributes": True}
