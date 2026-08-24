"""Safe arithmetic expression evaluation via AST allowlisting.

Evaluates user-supplied arithmetic expressions without ``eval``. Only numeric
literals, arithmetic operators, comparisons, boolean logic, and an explicit
allowlist of functions/constants are permitted. Attribute access, subscripting,
comprehensions, and arbitrary names are rejected, so a crafted expression
cannot escape into Python object internals.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

# Cap on expression length and on the magnitude of ``**`` results to prevent
# CPU/memory exhaustion (e.g. ``9**9**9**9``).
_MAX_EXPR_LENGTH = 200
_MAX_POW_RESULT = 1_000_000_000_000_000_000


class ExpressionError(ValueError):
    pass


def _eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ExpressionError("Only numeric literals are allowed")
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        raise ExpressionError(f"Name '{node.id}' is not allowed")
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left, env)
        right = _eval_expr(node.right, env)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            # Guard the exponent *before* computing to avoid a CPU blow-up on
            # expressions like ``9 ** 9 ** 9`` (which evaluates a 370M-digit int).
            if isinstance(right, (int, float)) and abs(right) > 512:
                raise ExpressionError("Exponent is too large")
            result = left ** right
            if isinstance(result, (int, float)) and abs(result) > _MAX_POW_RESULT:
                raise ExpressionError("Result is too large")
            return result
        raise ExpressionError(f"Unsupported operator: {type(op).__name__}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise ExpressionError("Unsupported unary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in env:
            raise ExpressionError("Only whitelisted functions may be called")
        if any(kw.arg is None for kw in node.keywords):
            raise ExpressionError("Keyword unpacking is not allowed")
        args = [_eval_expr(a, env) for a in node.args]
        kwargs = {kw.arg: _eval_expr(kw.value, env) for kw in node.keywords}
        return env[node.func.id](*args, **kwargs)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = _eval_expr(value, env)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for value in node.values:
                result = _eval_expr(value, env)
                if result:
                    return result
            return result
        raise ExpressionError("Unsupported boolean operator")
    if isinstance(node, ast.Compare):
        left = _eval_expr(node.left, env)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval_expr(comparator, env)
            if isinstance(op, ast.Eq):
                if not (left == right):
                    return False
            elif isinstance(op, ast.NotEq):
                if not (left != right):
                    return False
            elif isinstance(op, ast.Lt):
                if not (left < right):
                    return False
            elif isinstance(op, ast.LtE):
                if not (left <= right):
                    return False
            elif isinstance(op, ast.Gt):
                if not (left > right):
                    return False
            elif isinstance(op, ast.GtE):
                if not (left >= right):
                    return False
            else:
                raise ExpressionError(f"Unsupported comparison: {type(op).__name__}")
            left = right
        return True
    raise ExpressionError(f"Unsupported expression element: {type(node).__name__}")


def evaluate_expression(
    expression: str,
    functions: dict[str, Callable[..., Any]] | None = None,
    constants: dict[str, Any] | None = None,
) -> Any:
    """Evaluate an arithmetic expression safely.

    ``functions`` and ``constants`` provide the only names available inside the
    expression. Raises :class:`ExpressionError` on disallowed constructs.
    """
    if len(expression) > _MAX_EXPR_LENGTH:
        raise ExpressionError("Expression is too long")
    env: dict[str, Any] = {}
    env.update(constants or {})
    env.update(functions or {})
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise ExpressionError("Invalid expression") from None
    return _eval_expr(tree, env)
