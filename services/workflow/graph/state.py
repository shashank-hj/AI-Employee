import operator
from typing import Annotated, Any, TypedDict


class WorkflowState(TypedDict, total=False):
    """Shared state flowing through a workflow graph.

    ``outputs`` accumulates step results keyed by step name (merged via
    ``operator.or_`` so nodes can emit partial updates). ``history`` is an
    append-only log of step executions (``operator.add``).
    """

    workflow_id: str
    input_data: dict[str, Any]
    outputs: Annotated[dict[str, Any], operator.or_]
    history: Annotated[list[dict[str, Any]], operator.add]
    current_step: str | None
    error: str | None
