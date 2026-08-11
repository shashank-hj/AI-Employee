"""Minimal JSON Schema validation for tool input.

Implements the common JSON Schema subset (``type``, ``required``,
``properties``, ``items``, ``enum``, ``properties: {}``) used by the registry
without pulling in a full JSON Schema dependency.
"""

from typing import Any


class ValidationError(Exception):
    pass


_TYPE_CHECKS: dict[str, callable] = {
    "string": lambda v: isinstance(v, str),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> None:
    expected = schema.get("type")
    if expected:
        check = _TYPE_CHECKS.get(expected)
        if check is None:
            return  # unknown type — skip (permissive)
        if not check(value):
            raise ValidationError(
                f"'{path}' must be of type '{expected}', got {type(value).__name__}"
            )

    if (
        isinstance(value, str)
        and schema.get("minLength") is not None
        and len(value) < schema["minLength"]
    ):
        raise ValidationError(f"'{path}' must be at least {schema['minLength']} characters")

    if (
        isinstance(value, (int, float))
        and schema.get("minimum") is not None
        and value < schema["minimum"]
    ):
        raise ValidationError(f"'{path}' must be >= {schema['minimum']}")

    if (
        isinstance(value, (int, float))
        and schema.get("maximum") is not None
        and value > schema["maximum"]
    ):
        raise ValidationError(f"'{path}' must be <= {schema['maximum']}")

    enum = schema.get("enum")
    if enum is not None and value not in enum:
        raise ValidationError(f"'{path}' must be one of {enum}")

    if expected == "object":
        _validate_object(value, schema, path)

    if expected == "array" and schema.get("items"):
        items_schema = schema["items"]
        for i, item in enumerate(value):
            _validate_value(item, items_schema, f"{path}[{i}]")


def _validate_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> None:
    properties = schema.get("properties") or {}
    for prop, prop_schema in properties.items():
        if prop in value:
            _validate_value(value[prop], prop_schema, f"{path}.{prop}" if path else prop)

    for required in schema.get("required", []):
        if required not in value:
            raise ValidationError(f"'{required}' is required")


def validate_parameters(parameters: dict[str, Any], input_schema: dict[str, Any]) -> list[str]:
    """Validate ``parameters`` against ``input_schema``.

    Returns a list of validation error messages (empty when valid).
    """
    if not input_schema:
        return []
    errors: list[str] = []
    try:
        _validate_object(parameters, input_schema, "")
    except ValidationError as exc:
        errors.append(str(exc))
    return errors
