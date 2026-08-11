from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ToolCategory(StrEnum):
    UTILITY = "utility"
    DATA = "data"
    COMMUNICATION = "communication"
    CODE_EXECUTION = "code_execution"
    BROWSER = "browser"
    FILE = "file"
    CUSTOM = "custom"


class ToolExecutionType(StrEnum):
    """How a registered tool is executed at invoke time.

    - ``native``: run a built-in handler bundled with the tool-registry.
    - ``http``: call a remote HTTP endpoint described by ``execution_config``.
    - ``mcp``: invoke a tool on an external MCP server via JSON-RPC.
    """

    NATIVE = "native"
    HTTP = "http"
    MCP = "mcp"


class RetryPolicy(BaseModel):
    max_retries: int = Field(default=3, ge=0)
    delay_seconds: float = Field(default=1.0, ge=0.0)
    backoff_multiplier: float = Field(default=2.0, ge=1.0)


class ExecutionConfig(BaseModel):
    """Runtime invocation details for ``http``/``mcp`` tools."""

    url: str | None = Field(default=None, description="Endpoint URL for http execution")
    method: str = Field(default="POST", description="HTTP method for http execution")
    headers: dict[str, str] = Field(default_factory=dict)
    mcp_server_url: str | None = Field(default=None, description="MCP server URL for mcp execution")
    mcp_server_name: str | None = Field(default=None, description="Registered MCP server name")


class ToolCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    version: str = Field(default="1.0.0", pattern=r"^\d+\.\d+\.\d+$")
    category: ToolCategory
    permissions: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    tags: list[str] = Field(default_factory=list)
    execution_type: ToolExecutionType = ToolExecutionType.NATIVE
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)


class ToolUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    version: str | None = Field(default=None, pattern=r"^\d+\.\d+\.\d+$")
    category: ToolCategory | None = None
    permissions: list[str] | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    timeout_seconds: float | None = Field(default=None, gt=0.0)
    retry_policy: RetryPolicy | None = None
    tags: list[str] | None = None
    is_active: bool | None = None
    execution_type: ToolExecutionType | None = None
    execution_config: ExecutionConfig | None = None


class ToolResponse(BaseModel):
    id: str
    name: str
    description: str | None
    version: str
    category: ToolCategory
    permissions: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    timeout_seconds: float
    retry_policy: RetryPolicy
    tags: list[str]
    is_active: bool
    execution_type: ToolExecutionType = ToolExecutionType.NATIVE
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ToolListParams(BaseModel):
    category: ToolCategory | None = None
    is_active: bool | None = None
    tags: str | None = Field(default=None, description="Comma-separated tag filter")
    search: str | None = Field(default=None, description="Search in name and description")
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class ToolInvokeRequest(BaseModel):
    parameters: dict[str, Any] = Field(default_factory=dict)
    context: dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    duration_ms: float = 0.0
    tool_id: str | None = None
    tool_name: str | None = None
