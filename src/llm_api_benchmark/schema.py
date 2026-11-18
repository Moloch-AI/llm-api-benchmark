"""A small JSON Schema validator for tool-call arguments.

Tool definitions in the OpenAI format describe their parameters with JSON Schema. This module
checks a model's arguments against that schema without adding a dependency. It supports the
keywords that tool definitions use in practice and refuses any other keyword when a dataset is
loaded, so a constraint is never skipped silently:

``type``, ``properties``, ``required``, ``additionalProperties``, ``items``, ``enum``, ``const``,
``minimum``, ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``minLength``,
``maxLength``, ``pattern``, ``minItems``, ``maxItems``, ``anyOf`` and ``oneOf``.
``description``, ``title``, ``default``, ``examples`` and ``format`` are annotations and are not
checked. ``$ref`` is not supported.
"""

from __future__ import annotations

import re
from typing import Any

VALIDATED = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "anyOf",
        "oneOf",
    }
)
ANNOTATIONS = frozenset({"description", "title", "default", "examples", "format"})
TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})


def check_schema(schema: Any, path: str) -> list[str]:
    """Problems that make ``schema`` unusable here (unknown keywords, wrong shapes)."""
    if not isinstance(schema, dict):
        return [f"{path}: a JSON Schema is a mapping"]
    problems: list[str] = []
    for key, value in schema.items():
        where = f"{path}.{key}"
        if key in ANNOTATIONS:
            continue
        if key not in VALIDATED:
            problems.append(
                f"{where}: keyword not supported by the argument validator "
                f"(supported: {', '.join(sorted(VALIDATED))})"
            )
        elif key == "type":
            names = value if isinstance(value, list) else [value]
            if not names or any(n not in TYPES for n in names):
                problems.append(f"{where}: one of {', '.join(sorted(TYPES))}, or a list of them")
        elif key == "properties":
            if not isinstance(value, dict):
                problems.append(f"{where}: a mapping of property schemas")
            else:
                for name, sub in value.items():
                    problems.extend(check_schema(sub, f"{where}.{name}"))
        elif key == "required":
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                problems.append(f"{where}: a list of property names")
        elif key == "additionalProperties":
            if not isinstance(value, bool):
                problems.extend(check_schema(value, where))
        elif key == "items":
            problems.extend(check_schema(value, where))
        elif key in ("anyOf", "oneOf"):
            if not isinstance(value, list) or not value:
                problems.append(f"{where}: a non-empty list of schemas")
            else:
                for i, sub in enumerate(value):
                    problems.extend(check_schema(sub, f"{where}[{i}]"))
        elif key == "enum":
            if not isinstance(value, list) or not value:
                problems.append(f"{where}: a non-empty list")
        elif key == "pattern":
            try:
                re.compile(str(value))
            except re.error as err:
                problems.append(f"{where}: invalid regular expression ({err})")
        elif key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
            if not _is_number(value):
                problems.append(f"{where}: a number")
        elif key in ("minLength", "maxLength", "minItems", "maxItems") and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            problems.append(f"{where}: a whole number")
    return problems


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Every way ``value`` breaks ``schema``; an empty list means it is valid."""
    errors: list[str] = []
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, n) for n in names):
            return [f"{path}: expected {' or '.join(names)}, got {_type_name(value)}"]
    if "const" in schema and not _equal(value, schema["const"]):
        errors.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and not any(_equal(value, v) for v in schema["enum"]):
        errors.append(f"{path}: {value!r} is not one of {schema['enum']!r}")
    if _is_number(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} must be below {schema['exclusiveMaximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not re.search(str(schema["pattern"]), value):
            errors.append(f"{path}: does not match {schema['pattern']!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than {schema['maxItems']} items")
        if isinstance(schema.get("items"), dict):
            for i, item in enumerate(value):
                errors.extend(validate(item, schema["items"], f"{path}[{i}]"))
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                errors.append(f"{path}: missing required property {name!r}")
        extra = schema.get("additionalProperties", True)
        for name, item in value.items():
            if name in props:
                errors.extend(validate(item, props[name], f"{path}.{name}"))
            elif extra is False:
                errors.append(f"{path}: unexpected property {name!r}")
            elif isinstance(extra, dict):
                errors.extend(validate(item, extra, f"{path}.{name}"))
    if "anyOf" in schema and not any(not validate(value, s, path) for s in schema["anyOf"]):
        errors.append(f"{path}: matches none of anyOf")
    if "oneOf" in schema:
        matches = sum(1 for s in schema["oneOf"] if not validate(value, s, path))
        if matches != 1:
            errors.append(f"{path}: matches {matches} of oneOf, expected exactly 1")
    return errors


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _is_type(value: Any, name: str) -> bool:
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "number":
        return _is_number(value)
    if name == "integer":  # JSON Schema: a number with no fractional part, 1.0 included
        return _is_number(value) and float(value).is_integer()
    return False


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    return bool(a == b)
