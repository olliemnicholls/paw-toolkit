"""Pydantic schema to regular expression compiler for constrained JSON decoding."""

import enum
import re
import types
from typing import Any, Dict, List, Literal, Optional, Type, Union, get_args, get_origin
from pydantic import BaseModel
from pydantic.fields import FieldInfo

# Atomic regex fragments for JSON primitives
JSON_WHITESPACE = r"[ \t\n\r]*"
JSON_STRING = r'"([^"\\\x00-\x1f\x7f-\x9f]|\\.)*"'
JSON_INTEGER = r"(-?(0|[1-9][0-9]*))"
JSON_FLOAT = r"(-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)"
JSON_BOOLEAN = r"(true|false)"
JSON_NULL = r"null"


def _json_collection_regex(open_lit: str, close_lit: str, entry_regex: str) -> str:
    """Build a regex matching a bracketed, comma-separated, optionally-empty JSON collection body."""
    comma_sep = rf"{JSON_WHITESPACE},{JSON_WHITESPACE}{entry_regex}"
    return (
        rf"{open_lit}{JSON_WHITESPACE}(?:"
        rf"{entry_regex}(?:{comma_sep})*"
        rf")?{JSON_WHITESPACE}{close_lit}"
    )


def _type_to_regex(annotation: Any) -> str:
    """Recursively convert a Python type annotation into a JSON-matching regex string."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # 1. Handle Union / Optional types (e.g. Union[str, None], Optional[int], str | None)
    if origin in (Union, types.UnionType):
        branches = [_type_to_regex(arg) for arg in args]
        return f"(?:{'|'.join(branches)})"

    # 2. Handle Literal types (e.g. Literal["low", "med", "high"])
    if origin is Literal:
        literal_branches = []
        for val in args:
            if isinstance(val, str):
                literal_branches.append(f'"{re.escape(val)}"')
            elif isinstance(val, bool):
                literal_branches.append("true" if val else "false")
            elif isinstance(val, (int, float)):
                literal_branches.append(re.escape(str(val)))
            elif val is None:
                literal_branches.append(JSON_NULL)
            else:
                literal_branches.append(re.escape(str(val)))
        return f"(?:{'|'.join(literal_branches)})"

    # 3. Handle Enums
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        enum_branches = []
        for item in annotation:
            val = item.value
            if isinstance(val, str):
                enum_branches.append(f'"{re.escape(val)}"')
            elif isinstance(val, bool):
                enum_branches.append("true" if val else "false")
            elif isinstance(val, (int, float)):
                enum_branches.append(re.escape(str(val)))
            else:
                enum_branches.append(f'"{re.escape(str(val))}"')
        return f"(?:{'|'.join(enum_branches)})"

    # 4. Handle List / list[T]
    if origin in (list, List):
        item_type = args[0] if args else Any
        item_regex = _type_to_regex(item_type)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 5. Handle Nested Pydantic BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return pydantic_to_regex(annotation, anchors=False)

    # 6. Handle Dict / dict[K, V] (bare `dict` has no origin, only matches by identity)
    if origin is dict or annotation is dict:
        value_type = args[1] if len(args) > 1 else Any
        value_regex = _type_to_regex(value_type)
        entry = rf"{JSON_STRING}{JSON_WHITESPACE}:{JSON_WHITESPACE}{value_regex}"
        return _json_collection_regex(r"\{", r"\}", entry)

    # 7. Primitive types
    if annotation is str:
        return JSON_STRING
    if annotation is int:
        return JSON_INTEGER
    if annotation is float:
        return JSON_FLOAT
    if annotation is bool:
        return JSON_BOOLEAN
    if annotation is type(None) or annotation is None:
        return JSON_NULL
    if annotation is Any:
        return rf"(?:{JSON_STRING}|{JSON_FLOAT}|{JSON_BOOLEAN}|{JSON_NULL})"

    # Fallback to JSON string
    return JSON_STRING


def pydantic_to_regex(model: Type[BaseModel], anchors: bool = False) -> str:
    """Compile a Pydantic BaseModel class into a strict regex matching compliant JSON.

    Args:
        model: A Pydantic BaseModel subclass.
        anchors: Whether to wrap with ^ and $ anchors (useful for python re.match, omitted for interegular FSM).

    Returns:
        A regular expression pattern enforcing field names, types, commas, and braces.
    """
    fields: Dict[str, FieldInfo] = model.model_fields
    if not fields:
        pattern = r"\{" + JSON_WHITESPACE + r"\}"
        return f"^{pattern}$" if anchors else pattern

    field_patterns: List[str] = []
    for field_name, field_info in fields.items():
        field_key = f'"{re.escape(field_name)}"'
        value_regex = _type_to_regex(field_info.annotation)
        field_pattern = f"{field_key}{JSON_WHITESPACE}:{JSON_WHITESPACE}{value_regex}"
        field_patterns.append(field_pattern)

    combined_body = f"{JSON_WHITESPACE},{JSON_WHITESPACE}".join(field_patterns)
    pattern = r"\{" + JSON_WHITESPACE + combined_body + JSON_WHITESPACE + r"\}"
    return f"^{pattern}$" if anchors else pattern
