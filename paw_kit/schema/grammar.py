"""Pydantic schema to regular expression compiler for constrained JSON decoding."""

import datetime as dt
import enum
from functools import lru_cache
import re
import types
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)
import uuid
from decimal import Decimal
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from paw_kit.schema.exceptions import PAWSchemaError

# Atomic regex fragments for JSON primitives
JSON_WHITESPACE = r"[ \t\n\r]*"
JSON_STRING = r'"([^"\\\x00-\x1f\x7f-\x9f]|\\.)*"'
JSON_INTEGER = r"(-?(0|[1-9][0-9]*))"
JSON_FLOAT = r"(-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?)"
JSON_BOOLEAN = r"(true|false)"
JSON_NULL = r"null"

# Specialized type regex fragments
JSON_UUID = r'"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"'
JSON_DATE = r'"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"'
JSON_DATETIME = (
    r'"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])'
    r"[T ](?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r'(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"'
)

# Maximum recursion depth for nested BaseModel resolution
_MAX_RECURSION_DEPTH = 10


def _json_collection_regex(open_lit: str, close_lit: str, entry_regex: str) -> str:
    """Build a regex matching a bracketed, comma-separated, optionally-empty JSON collection body."""
    comma_sep = rf"{JSON_WHITESPACE},{JSON_WHITESPACE}{entry_regex}"
    return (
        rf"{open_lit}{JSON_WHITESPACE}(?:"
        rf"{entry_regex}(?:{comma_sep})*"
        rf")?{JSON_WHITESPACE}{close_lit}"
    )


def _extract_pattern_from_field(field_info: FieldInfo) -> Optional[str]:
    """Extract pattern constraint from Pydantic v2 field metadata, if present."""
    for meta in field_info.metadata:
        if hasattr(meta, "pattern") and meta.pattern is not None:
            return str(meta.pattern)
    return None


def _type_to_regex(
    annotation: Any,
    *,
    seen: Optional[frozenset] = None,
    depth: int = 0,
) -> str:
    """Recursively convert a Python type annotation into a JSON-matching regex string.

    Args:
        annotation: A Python type annotation.
        seen: Set of BaseModel types already visited (cycle detection).
        depth: Current recursion depth.

    Returns:
        A regex string matching valid JSON representations of the type.

    Raises:
        PAWSchemaError: If a recursive model cycle or excessive recursion depth is detected.
    """
    if seen is None:
        seen = frozenset()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # 1. Handle Union / Optional types (e.g. Union[str, None], Optional[int], str | None)
    if origin in (Union, types.UnionType):
        branches = [_type_to_regex(arg, seen=seen, depth=depth) for arg in args]
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
        item_regex = _type_to_regex(item_type, seen=seen, depth=depth)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 5. Handle Tuple / tuple[A, B] / tuple[T, ...]
    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            # Variadic: tuple[str, ...] -> same as list[str]
            item_regex = _type_to_regex(args[0], seen=seen, depth=depth)
            return _json_collection_regex(r"\[", r"\]", item_regex)
        elif args:
            # Fixed-length: tuple[str, int, bool] -> [str, int, bool] exact positions
            elem_regexes = [_type_to_regex(a, seen=seen, depth=depth) for a in args]
            inner = f"{JSON_WHITESPACE},{JSON_WHITESPACE}".join(elem_regexes)
            return rf"\[{JSON_WHITESPACE}{inner}{JSON_WHITESPACE}\]"
        else:
            # Bare tuple[()] -> empty array
            any_regex = _type_to_regex(Any, seen=seen, depth=depth)
            return _json_collection_regex(r"\[", r"\]", any_regex)

    # 6. Handle Set / set[T] / FrozenSet / frozenset[T]
    if origin in (set, Set, frozenset, FrozenSet):
        item_type = args[0] if args else Any
        item_regex = _type_to_regex(item_type, seen=seen, depth=depth)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 7. Handle Nested Pydantic BaseModel (with cycle detection)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in seen or depth > _MAX_RECURSION_DEPTH:
            raise PAWSchemaError(
                f"Recursive model detected: {annotation.__name__} at depth {depth}. "
                f"Grammar-constrained decoding cannot express infinite recursion. "
                f"Consider flattening the schema or limiting nesting depth."
            )
        return _pydantic_to_regex_impl(
            annotation,
            anchors=False,
            _seen=seen | {annotation},
            _depth=depth + 1,
        )

    # 8. Handle Dict / dict[K, V] (bare `dict` has no origin, only matches by identity)
    if origin is dict or annotation is dict:
        value_type = args[1] if len(args) > 1 else Any
        value_regex = _type_to_regex(value_type, seen=seen, depth=depth)
        entry = rf"{JSON_STRING}{JSON_WHITESPACE}:{JSON_WHITESPACE}{value_regex}"
        return _json_collection_regex(r"\{", r"\}", entry)

    # 9. Bare collection identity checks (no generic args -> get_origin returns None)
    if annotation is list:
        any_regex = _type_to_regex(Any, seen=seen, depth=depth)
        return _json_collection_regex(r"\[", r"\]", any_regex)
    if annotation is tuple:
        any_regex = _type_to_regex(Any, seen=seen, depth=depth)
        return _json_collection_regex(r"\[", r"\]", any_regex)
    if annotation is set or annotation is frozenset:
        any_regex = _type_to_regex(Any, seen=seen, depth=depth)
        return _json_collection_regex(r"\[", r"\]", any_regex)

    # 10. Specialized types
    if annotation is uuid.UUID:
        return JSON_UUID
    if annotation is dt.datetime:
        return JSON_DATETIME
    if annotation is dt.date:
        return JSON_DATE
    if annotation is Decimal:
        return JSON_STRING

    # 11. Primitive types
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


def _pydantic_to_regex_impl(
    model: Type[BaseModel],
    *,
    anchors: bool,
    _seen: frozenset,
    _depth: int,
) -> str:
    """Internal implementation of pydantic_to_regex with cycle detection state.

    Args:
        model: A Pydantic BaseModel subclass.
        anchors: Whether to wrap with ^ and $ anchors.
        _seen: Set of BaseModel types already visited (cycle detection).
        _depth: Current recursion depth.

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
        # Check for Field(pattern=...) constraint
        pattern_override = _extract_pattern_from_field(field_info)
        if pattern_override is not None:
            clean_pattern = pattern_override.lstrip("^").rstrip("$")
            value_regex = f'"{clean_pattern}"'
        else:
            value_regex = _type_to_regex(field_info.annotation, seen=_seen, depth=_depth)
        field_pattern = f"{field_key}{JSON_WHITESPACE}:{JSON_WHITESPACE}{value_regex}"
        field_patterns.append(field_pattern)

    combined_body = f"{JSON_WHITESPACE},{JSON_WHITESPACE}".join(field_patterns)
    pattern = r"\{" + JSON_WHITESPACE + combined_body + JSON_WHITESPACE + r"\}"
    return f"^{pattern}$" if anchors else pattern


@lru_cache(maxsize=128)
def pydantic_to_regex(model: Type[BaseModel], anchors: bool = False) -> str:
    """Compile a Pydantic BaseModel class into a strict regex matching compliant JSON.

    Results are cached (LRU, maxsize=128) to avoid redundant regex compilation
    and downstream FSM construction costs.

    Args:
        model: A Pydantic BaseModel subclass.
        anchors: Whether to wrap with ^ and $ anchors (useful for python re.match,
                 omitted for interegular FSM).

    Returns:
        A regular expression pattern enforcing field names, types, commas, and braces.

    Raises:
        PAWSchemaError: If the model contains recursive references that cannot be
                        expressed as a finite regex.
    """
    return _pydantic_to_regex_impl(model, anchors=anchors, _seen=frozenset(), _depth=0)
