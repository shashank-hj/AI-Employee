from typing import Annotated, Any, TypedDict


class PlanStep(TypedDict):
    tool_name: str
    parameters: dict[str, Any]
    reason: str


_RESET_TURN = "__RESET_AGENT_STATE_TURN__"


def _list_or_reset(current: list[Any], incoming: list[Any]) -> list[Any]:
    if incoming and isinstance(incoming[0], str) and incoming[0] == _RESET_TURN:
        return list(incoming[1:])
    return current + incoming


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

    tool_results: Annotated[list[dict[str, Any]], _list_or_reset]
    execution_log: Annotated[list[dict[str, Any]], _list_or_reset]

    awaiting_approval: bool
    approval_task_id: str | None

    final_response: str | None
    error: str | None
