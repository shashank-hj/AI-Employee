from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    workflow_type: str = "default"
    input_data: dict[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str | None = None
    workflow_type: str = "default"
    input_data: dict[str, Any] | None = None
    steps: list[dict[str, Any]] | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_step: str | None = None
    output_data: dict[str, Any] | None = None
    error_message: str | None = None
    pending_approval: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime | None = None


class WorkflowHistoryEntry(BaseModel):
    step_name: str
    status: WorkflowStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_data: dict[str, Any] | None = None
    output_data: dict[str, Any] | None = None
    error_message: str | None = None


class WorkflowRunRequest(BaseModel):
    input_data: dict[str, Any] | None = None
    stream: bool = False
    timeout_seconds: float | None = Field(default=300, gt=0)


class WorkflowRunResponse(BaseModel):
    workflow: WorkflowResponse
    interrupted: bool = False
    current_step: str | None = None
    outputs: dict[str, Any] | None = None
    history: list[dict[str, Any]] | None = None


class WorkflowResumeRequest(BaseModel):
    payload: dict[str, Any]
