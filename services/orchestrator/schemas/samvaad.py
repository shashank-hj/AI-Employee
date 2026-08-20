from typing import Any, Literal

from pydantic import BaseModel, Field

SamvaadMode = Literal["call", "chat"]


class SamvaadStatusResponse(BaseModel):
    status: str
    enabled: bool = False
    agent_id: str = ""
    reason: str | None = None
    active_sessions: int = 0
    tools: dict[str, list[str]] = Field(
        default_factory=lambda: {"allowed": [], "blocked": []},
        description="Webhook tools gate state (SAMVAAD_TOOLS_ALLOWLIST).",
    )


class SamvaadSessionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=128)
    session_id: str | None = Field(default=None, max_length=256)
    mode: SamvaadMode = "chat"
    language: str | None = Field(
        default=None, description="Starting language name, e.g. 'English'"
    )
    agent_variables: dict[str, Any] | None = Field(default=None)


class SamvaadSessionResponse(BaseModel):
    session_id: str
    interaction_id: str | None = None
    mode: str
    status: str


class SamvaadSessionInfo(BaseModel):
    session_id: str
    mode: str
    connected: bool
    interaction_id: str | None = None


class SamvaadTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class SamvaadAudioRequest(BaseModel):
    audio_base64: str = Field(
        ..., description="Base64-encoded raw 16-bit PCM mono audio"
    )


class SamvaadMessagesResponse(BaseModel):
    session_id: str
    interaction_id: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)


class SamvaadSessionCost(BaseModel):
    interaction_id: str | None = None
    start_datetime: str | None = None
    turns: int = 0
    duration_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    stt_rs: float = 0.0
    cost_105b_rs: float = 0.0
    cost_glm_rs: float = 0.0


class SamvaadUsageResponse(BaseModel):
    available: bool = False
    reason: str | None = None
    total_105b_rs: float = 0.0
    total_glm_rs: float = 0.0
    stt_rs: float = 0.0
    session_count: int = 0
    sessions: list[SamvaadSessionCost] = Field(default_factory=list)
    days: int = 14
    spend_alert_rs: float = 25.0
    ts: int = 0
