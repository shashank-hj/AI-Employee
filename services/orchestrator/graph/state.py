from typing import Annotated, Any, TypedDict

import operator


class PlanStep(TypedDict):
    tool_name: str
    parameters: dict[str, Any]
    reason: str


class AgentState(TypedDict):
    request_id: str
    user_input: str
    user_id: str | None
    session_id: str | None
    channel: str | None
    channel_message_id: str | None
    tenant_id: str | None
    contact: dict[str, Any] | None
    request_metadata: dict[str, Any] | None

    memory_context: list[dict[str, Any]]
    document_context: list[dict[str, Any]]
    user_preferences: dict[str, Any]

    plan: list[PlanStep]
    current_step_index: int

    tool_results: Annotated[list[dict[str, Any]], operator.add]
    execution_log: Annotated[list[dict[str, Any]], operator.add]

    awaiting_approval: bool
    approval_task_id: str | None

    final_response: str | None
    error: str | None
