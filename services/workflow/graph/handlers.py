"""Built-in workflow step handlers.

A handler is an ``async def`` taking a single ``params`` dict and returning a
JSON-serialisable result dict. Handlers are the leaf work of the workflow
engine — new business capabilities plug in by registering a new handler here.
"""

import asyncio
import math
import random
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from shared.utils.safe_eval import ExpressionError, evaluate_expression

Handler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def echo(params: dict[str, Any]) -> dict[str, Any]:
    return {"echo": params}


_CALC_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
}

_CALC_CONSTS: dict[str, Any] = {"pi": math.pi, "e": math.e}


async def calculator(params: dict[str, Any]) -> dict[str, Any]:
    expression = str(params.get("expression", ""))
    try:
        result = evaluate_expression(expression, functions=_CALC_FUNCS, constants=_CALC_CONSTS)
        return {"expression": expression, "result": result}
    except ExpressionError as exc:
        return {"expression": expression, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001 - e.g. ZeroDivisionError, OverflowError
        return {"expression": expression, "error": str(exc)}


async def sleep(params: dict[str, Any]) -> dict[str, Any]:
    seconds = float(params.get("seconds", 0.0))
    await asyncio.sleep(seconds)
    return {"slept_seconds": seconds, "at": _now()}


async def validate_refund(params: dict[str, Any]) -> dict[str, Any]:
    amount = float(params.get("amount", 0.0))
    order_id = params.get("order_id", "")
    return {
        "order_id": order_id,
        "amount": amount,
        "eligible": amount > 0,
        "validated": amount > 0,
    }


async def process_refund(params: dict[str, Any]) -> dict[str, Any]:
    """Side-effecting refund execution. Gated by human approval at the graph
    level (see APPROVAL_HANDLERS) before this handler runs."""
    return {
        "refund_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
        "order_id": params.get("order_id", ""),
        "amount": params.get("amount", 0),
        "status": "processed",
        "timestamp": _now(),
    }


async def send_email(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "message_id": f"MSG-{uuid.uuid4().hex[:8].upper()}",
        "to": params.get("to", ""),
        "subject": params.get("subject", ""),
        "status": "sent",
        "timestamp": _now(),
    }


async def check_inventory(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "sku": params.get("sku", ""),
        "in_stock": random.random() > 0.2,
        "qty": random.randint(0, 20),
    }


async def check_pricing(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "sku": params.get("sku", ""),
        "price_inr": round(random.uniform(100, 9999), 2),
    }


HANDLERS: dict[str, Handler] = {
    "echo": echo,
    "calculator": calculator,
    "sleep": sleep,
    "validate_refund": validate_refund,
    "process_refund": process_refund,
    "send_email": send_email,
    "check_inventory": check_inventory,
    "check_pricing": check_pricing,
}

#: Handlers that always require human approval before their side effect runs.
APPROVAL_HANDLERS: frozenset[str] = frozenset({"process_refund"})
