import math

import pytest

from shared.utils.safe_eval import ExpressionError, evaluate_expression

FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow,
    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
}
CONSTS = {"pi": math.pi, "e": math.e}


def _eval(expr):
    return evaluate_expression(expr, functions=FUNCS, constants=CONSTS)


class TestSafeEval:
    def test_arithmetic(self):
        assert _eval("2 + 3") == 5
        assert _eval("2 ** 3 + 4 * 5") == 28
        assert _eval("(1 + 2) * 3") == 9

    def test_whitelisted_functions(self):
        assert _eval("sqrt(16)") == 4.0
        assert _eval("max(3, 5, 2)") == 5
        assert _eval("round(pi, 2)") == 3.14

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            _eval("1 / 0")

    def test_blocks_attribute_access(self):
        with pytest.raises(ExpressionError):
            _eval("().__class__.__base__.__subclasses__()")

    def test_blocks_unknown_names(self):
        with pytest.raises(ExpressionError):
            _eval("__import__('os')")
        with pytest.raises(ExpressionError):
            _eval("open('/etc/passwd')")
        with pytest.raises(ExpressionError):
            _eval("getattr")

    def test_blocks_imports_and_call_on_literals(self):
        with pytest.raises(ExpressionError):
            _eval("[].__class__")
        with pytest.raises(ExpressionError):
            _eval("{}")

    def test_blocks_huge_powers(self):
        with pytest.raises(ExpressionError):
            _eval("9 ** 9 ** 9")

    def test_blocks_too_long_expression(self):
        with pytest.raises(ExpressionError):
            _eval("1 + 1 + " * 200 + "1")

    def test_empty_expression_raises(self):
        with pytest.raises(ExpressionError):
            _eval("")
