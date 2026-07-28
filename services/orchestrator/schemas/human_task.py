from datetime import datetime
from pydantic import BaseModel, Field


class HumanTaskCreate(BaseModel):
    user_input: str = Field(..., min_length=1)
    intent: str = Field(default="escalate")
    reason: str | None = None
    priority: str = Field(default="NORMAL")
    context: dict = Field(default_factory=dict)
    request_id: str | None = None


class HumanTaskResponse(BaseModel):
    id: str
    request_id: str | None
    user_input: str
    intent: str
    reason: str | None
    priority: str
    status: str
    context: dict
    assigned_to: str | None
    resolution_note: str | None
    created_at: datetime
    resolved_at: datetime | None

    model_config = {"from_attributes": True}


class HumanTaskResolve(BaseModel):
    resolution_note: str | None = None
    assigned_to: str | None = None
