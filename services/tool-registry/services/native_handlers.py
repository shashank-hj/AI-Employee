"""Built-in reference tool implementations for ``native`` execution.

These small, self-contained handlers let the tool-registry actually execute
registered tools end-to-end without an external runtime. New tools can register
a handler by name here, or use ``http``/``mcp`` execution types.
"""

from collections.abc import Callable
import math
from datetime import UTC, datetime

from shared.utils.safe_eval import ExpressionError, evaluate_expression


def _echo(parameters: dict) -> dict:
    return {"echo": parameters.get("text", "")}


_CALC_FUNCS: dict[str, Callable] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
}

_CALC_CONSTS: dict = {"pi": math.pi, "e": math.e}


def _calculator(parameters: dict) -> dict:
    expression = str(parameters.get("expression", ""))
    try:
        result = evaluate_expression(expression, functions=_CALC_FUNCS, constants=_CALC_CONSTS)
    except ExpressionError as exc:
        raise ValueError(f"Invalid expression: {exc}") from None
    except Exception as exc:
        raise ValueError(f"Invalid expression: {exc}") from None
    return {"result": result}


def _uppercase(parameters: dict) -> dict:
    return {"uppercased": str(parameters.get("text", "")).upper()}


def _get_current_time(parameters: dict) -> dict:
    return {
        "utc_iso": datetime.now(UTC).isoformat(),
        "timezone": parameters.get("timezone", "UTC"),
    }


NATIVE_HANDLERS: dict[str, Callable] = {
    "echo": _echo,
    "calculator": _calculator,
    "uppercase": _uppercase,
    "get_current_time": _get_current_time,
}


def resolve_native_handler(name: str) -> Callable | None:
    return NATIVE_HANDLERS.get(name)
