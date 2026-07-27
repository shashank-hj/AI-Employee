from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    UTILITY = "utility"
    DATA = "data"
    COMMUNICATION = "communication"
    CODE_EXECUTION = "code_execution"
    BROWSER = "browser"
    FILE = "file"
    CUSTOM = "custom"


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=3, ge=0)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    category: ToolCategory
    permissions: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    tags: list[str] = Field(default_factory=list)


class ToolUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    description: Optional[str] = None
    version: Optional[str] = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    category: Optional[ToolCategory] = None
    permissions: Optional[list[str]] = None
    input_schema: Optional[dict[str, Any]] = None
    output_schema: Optional[dict[str, Any]] = None
    timeout_seconds: Optional[float] = Field(default=None, gt=0.0)
    retry_policy: Optional[RetryPolicy] = None
    tags: Optional[list[str]] = None
    is_active: Optional[bool] = None


class ToolResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    version: str
    category: ToolCategory
    permissions: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: float
    retry_policy: RetryPolicy
    tags: list[str]
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ToolListParams(BaseModel):
    category: Optional[ToolCategory] = None
    is_active: Optional[bool] = None
    tags: Optional[str] = Field(default=None, description="Comma-separated tag filter")
    search: Optional[str] = Field(default=None, description="Search in name and description")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
