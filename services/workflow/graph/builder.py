"""Turn a declarative ``steps`` list into a compiled LangGraph workflow.

Supported step types:
  * ``task``     — run a single handler (optional ``requires_approval`` gates the
                   side effect behind a human-in-the-loop interrupt).
  * ``fan_out``  — run several handlers in parallel, results collected under the
                   step name keyed by handler name.
  * ``branch``   — a no-op routing node; ``field`` selects a dot-path into state
                   and ``branches`` maps values to the next step.

Static ``next`` can be a single step name or a list (parallel fan-out to
multiple downstream steps). Omitted ``next`` terminates the workflow.
"""

import asyncio
import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from workflow.graph.handlers import APPROVAL_HANDLERS, Handler
from workflow.graph.state import WorkflowState

_REFERENCE = re.compile(r"^\$(input|outputs)\.")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def resolve_value(template: Any, input_data: dict, outputs: dict) -> Any:
    """Resolve ``$input.x`` / ``$outputs.x.y`` references inside params."""
    if isinstance(template, str) and _REFERENCE.match(template):
        path = template[1:].split(".")
        obj = input_data if path[0] == "input" else outputs
        for part in path[1:]:
            if isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return None
        return obj
    if isinstance(template, dict):
        return {k: resolve_value(v, input_data, outputs) for k, v in template.items()}
    if isinstance(template, list):
        return [resolve_value(v, input_data, outputs) for v in template]
    return template


def get_path(state: dict, path_str: str) -> Any:
    # "$input.X" / "$outputs.X" references map onto the state's input_data/outputs.
    if path_str.startswith("$input."):
        path_str = "input_data." + path_str[len("$input."):]
    elif path_str.startswith("$outputs."):
        path_str = "outputs." + path_str[len("$outputs."):]
    obj: Any = state
    for part in path_str.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            return None
    return obj


def make_task_node(step: dict, handlers: dict[str, Handler]) -> Callable:
    name = step["name"]
    handler_name = step.get("handler")
    requires_approval = bool(step.get("requires_approval")) or handler_name in APPROVAL_HANDLERS
    if handler_name not in handlers:
        raise ValueError(f"Unknown handler '{handler_name}' for step '{name}'")
    handler = handlers[handler_name]

    async def node(state: WorkflowState) -> dict[str, Any]:
        params = resolve_value(
            step.get("params", {}), state.get("input_data", {}), state.get("outputs", {}),
        )
        if requires_approval:
            decision = interrupt({
                "type": "human_approval",
                "workflow_step": name,
                "handler": handler_name,
                "params": params,
                "reason": step.get("approval_reason", f"'{name}' requires human approval"),
            })
            if not (isinstance(decision, dict) and decision.get("approved")):
                reason = decision.get("reason") if isinstance(decision, dict) else "not approved"
                entry = {
                    "step": name,
                    "status": "skipped",
                    "output": {"status": "rejected", "reason": reason},
                    "timestamp": _now(),
                }
                return {
                    "outputs": {name: {"status": "rejected", "reason": reason}},
                    "history": [entry],
                    "current_step": name,
                }
        try:
            result = await handler(params)
            status: str = "completed"
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status = "failed"
        return {
            "outputs": {name: result},
            "history": [{"step": name, "status": status, "output": result, "timestamp": _now()}],
            "current_step": name,
        }

    return node


def make_fanout_node(step: dict, handlers: dict[str, Handler]) -> Callable:
    name = step["name"]
    handler_names = step.get("handlers", [])
    for h in handler_names:
        if h not in handlers:
            raise ValueError(f"Unknown handler '{h}' for fan_out step '{name}'")

    async def node(state: WorkflowState) -> dict[str, Any]:
        params = resolve_value(
            step.get("params", {}), state.get("input_data", {}), state.get("outputs", {}),
        )
        results = await asyncio.gather(
            *(handlers[h](params) for h in handler_names), return_exceptions=True,
        )
        outputs: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        for handler_name, result in zip(handler_names, results, strict=True):
            if isinstance(result, BaseException):
                outputs[handler_name] = {"error": str(result)}
                history.append({
                    "step": name, "sub": handler_name, "status": "failed", "timestamp": _now(),
                })
            else:
                outputs[handler_name] = result
                history.append({
                    "step": name, "sub": handler_name, "status": "completed", "timestamp": _now(),
                })
        return {"outputs": {name: outputs}, "history": history, "current_step": name}

    return node


def make_branch_node(step: dict) -> Callable:
    async def node(state: WorkflowState) -> dict[str, Any]:
        return {"current_step": step["name"]}

    return node


def build_workflow_graph(
    steps: list[dict[str, Any]],
    handlers: dict[str, Handler] | None = None,
    checkpointer=None,
):
    from workflow.graph.handlers import HANDLERS

    if not steps:
        raise ValueError("Workflow must define at least one step")
    handlers = handlers or HANDLERS

    names = [s.get("name") for s in steps]
    if any(not n for n in names):
        raise ValueError("Every step must have a name")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate step names are not allowed")

    builder = StateGraph(WorkflowState)
    for step in steps:
        step_type = step.get("type", "task")
        if step_type == "task":
            builder.add_node(step["name"], make_task_node(step, handlers))
        elif step_type == "fan_out":
            builder.add_node(step["name"], make_fanout_node(step, handlers))
        elif step_type == "branch":
            builder.add_node(step["name"], make_branch_node(step))
        else:
            raise ValueError(f"Unknown step type '{step_type}' for step '{step['name']}'")

    name_to_step = {s["name"]: s for s in steps}
    builder.add_edge(START, steps[0]["name"])

    for step in steps:
        step_type = step.get("type", "task")
        next_targets = step.get("next")
        if step_type == "branch":
            targets = list(step.get("branches", {}).values())
            if step.get("default"):
                targets.append(step["default"])
            targets = [t for t in dict.fromkeys(targets) if t in name_to_step]
            for t in targets:
                if t not in name_to_step:
                    raise ValueError(f"Branch target '{t}' not found in steps")

            def route(state: WorkflowState, _step: dict = step) -> str:
                value = get_path(state, _step["field"])
                key = str(value).lower() if isinstance(value, bool) else str(value)
                return _step.get("branches", {}).get(key, _step.get("default", END))

            path_map = {t: t for t in targets}
            path_map[END] = END
            builder.add_conditional_edges(step["name"], route, path_map)
        elif next_targets:
            targets = [next_targets] if isinstance(next_targets, str) else next_targets
            for target in targets:
                if target not in name_to_step:
                    raise ValueError(f"Step '{step['name']}' points at unknown step '{target}'")
                builder.add_edge(step["name"], target)
        else:
            builder.add_edge(step["name"], END)

    return builder.compile(checkpointer=checkpointer)
