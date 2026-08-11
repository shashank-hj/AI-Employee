"""Built-in reference tool implementations for ``native`` execution.

These small, self-contained handlers let the tool-registry actually execute
registered tools end-to-end without an external runtime. New tools can register
a handler by name here, or use ``http``/``mcp`` execution types.
"""

import math
import re
from datetime import UTC, datetime


def _echo(parameters: dict) -> dict:
    return {"echo": parameters.get("text", "")}


def _calculator(parameters: dict) -> dict:
    expression = str(parameters.get("expression", ""))
    # Allow only safe math characters.
    if not re.fullmatch(r"[0-9+\-*/().\s%]+", expression):
        raise ValueError("Unsupported characters in expression")
    try:
        result = eval(expression, {"__builtins__": {}}, {"math": math})  # noqa: S307
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}") from exc
    return {"result": result}


def _uppercase(parameters: dict) -> dict:
    return {"uppercased": str(parameters.get("text", "")).upper()}


def _get_current_time(parameters: dict) -> dict:
    return {
        "utc_iso": datetime.now(UTC).isoformat(),
        "timezone": parameters.get("timezone", "UTC"),
    }


NATIVE_HANDLERS: dict[str, callable] = {
    "echo": _echo,
    "calculator": _calculator,
    "uppercase": _uppercase,
    "get_current_time": _get_current_time,
}


def resolve_native_handler(name: str) -> callable | None:
    return NATIVE_HANDLERS.get(name)
