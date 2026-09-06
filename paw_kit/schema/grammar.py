"""Pydantic schema to regular expression compiler for constrained JSON decoding."""

import datetime as dt
import enum
from functools import lru_cache
import json
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

# PAW-SCHEMA-02: maximum nesting depth for generic collections (List/Tuple/Set/Dict
# wrapping each other, e.g. List[List[List[int]]]), tracked as a budget separate from
# _MAX_RECURSION_DEPTH above. _json_collection_regex embeds its entry_regex twice
# (once for the first element, once for each repeated element), so regex length grows
# roughly geometrically with collection nesting depth; sharing one counter with
# BaseModel nesting would either let that blowup through uncapped or start rejecting
# realistic schemas that legally nest BaseModels several levels deep. This budget
# resets at each BaseModel boundary (a model's own fields start a fresh nesting
# context), so e.g. six BaseModels nested inside each other, each with its own
# List[Dict[str, ...]] field, stays well within it.
_MAX_COLLECTION_DEPTH = 10


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


def _json_string_literal_regex(val: str) -> str:
    """Build a regex matching the exact JSON-encoded form of a literal string value.

    PAW-SCHEMA-01: `re.escape(val)` alone is not enough for a literal string value
    embedded in a hand-built `f'"{...}"'` template -- `re.escape` has not escaped `"`
    since Python 3.7 (it isn't a regex metacharacter), so a value containing a quote
    (e.g. `Literal['say "hi"']`, or a str-valued Enum member) breaks out of the
    surrounding JSON string boundary and lets the rest of the value be interpreted as
    new JSON structure. `json.dumps` produces the correct JSON-escaped representation
    (quotes, backslashes, control characters -- all of it, including the surrounding
    quote marks); `re.escape` on top of that makes the backslashes json.dumps
    introduced (and any other regex metacharacters) safe as a literal regex fragment.
    """
    return re.escape(json.dumps(val))


def _sanitize_field_pattern(pattern: str) -> str:
    """Validate a `Field(pattern=...)` regex constraint for safe JSON-string embedding.

    PAW-SCHEMA-01: the pattern is inserted verbatim as regex source between JSON
    quote marks (`f'"{clean_pattern}"'`) since it constrains what the *string value*
    may contain -- unlike a Literal/Enum value, it cannot simply be JSON-escaped
    without changing its regex semantics. Any `"` in the pattern source is rejected
    outright, including one the schema author intended as an "escaped" quote.

    A naive "reject unescaped quotes, allow backslash-escaped ones" check (the
    audit's own suggested fix) is not actually safe here: in a *regex*, a single
    backslash before `"` (`r'a\\"b'`, one backslash) does not require a backslash in
    the matched text at all -- `re.compile(r'a\\"b').fullmatch('a"b')` matches, since
    `\"` isn't a recognized escape and Python's `re` simply drops the backslash and
    matches the literal `"`. Requiring a backslash in the matched *output* text needs
    *two* source backslashes (`r'a\\\\"b'`), which no schema author would intuitively
    write, and a heuristic that tries to tell these apart by counting backslash parity
    is exactly the kind of subtle-and-wrong check that reintroduces the vulnerability
    for anyone who writes the "obvious" single-backslash escape. So: no quote
    character is permitted in a pattern constraint at all, escaped or not.
    """
    clean = pattern.lstrip("^").rstrip("$")
    if '"' in clean:
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: double quote characters "
            "are forbidden in Field(pattern=...) constraints entirely -- a compiled "
            "regex referencing a quote, escaped or not, can be made to match a "
            "literal JSON-string-terminating quote in the constrained decoder's "
            "output. Remove the quote from the pattern."
        )
    return clean


def _check_collection_depth(collection_depth: int) -> None:
    """Raise if entering another generic-collection nesting level exceeds the budget.

    PAW-SCHEMA-02: `_json_collection_regex` embeds its entry_regex twice (once for the
    first element, once per repeated element), so regex length grows roughly
    geometrically with how many List/Tuple/Set/Dict wrap each other -- and depth was
    previously never incremented for these branches at all, so nothing bounded it.
    """
    if collection_depth >= _MAX_COLLECTION_DEPTH:
        raise PAWSchemaError(
            f"Collection nesting exceeds maximum depth of {_MAX_COLLECTION_DEPTH} "
            f"(List/Tuple/Set/Dict nested within each other, e.g. List[List[List[...]]])"
            f". Grammar-constrained decoding cannot safely express this due to "
            f"regex size growing geometrically with nesting depth. Consider "
            f"flattening the schema."
        )


def _type_to_regex(
    annotation: Any,
    *,
    seen: Optional[frozenset] = None,
    depth: int = 0,
    collection_depth: int = 0,
) -> str:
    """Recursively convert a Python type annotation into a JSON-matching regex string.

    Args:
        annotation: A Python type annotation.
        seen: Set of BaseModel types already visited (cycle detection).
        depth: Current BaseModel-nesting recursion depth.
        collection_depth: Current generic-collection (List/Tuple/Set/Dict) nesting
            depth. Tracked separately from `depth` (PAW-SCHEMA-02) so the two budgets
            don't interfere with each other; it resets to 0 whenever recursion enters
            a nested BaseModel's own fields, since those form a fresh nesting context.

    Returns:
        A regex string matching valid JSON representations of the type.

    Raises:
        PAWSchemaError: If a recursive model cycle, excessive BaseModel recursion
            depth, or excessive collection nesting depth is detected.
    """
    if seen is None:
        seen = frozenset()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # 1. Handle Union / Optional types (e.g. Union[str, None], Optional[int], str | None)
    if origin in (Union, types.UnionType):
        branches = [
            _type_to_regex(arg, seen=seen, depth=depth, collection_depth=collection_depth) for arg in args
        ]
        return f"(?:{'|'.join(branches)})"

    # 2. Handle Literal types (e.g. Literal["low", "med", "high"])
    if origin is Literal:
        literal_branches = []
        for val in args:
            if isinstance(val, str):
                # PAW-SCHEMA-01: JSON-escape the value itself, not just regex-escape it
                # -- see _json_string_literal_regex docstring for why re.escape alone
                # lets a quote inside the literal break out of the JSON string boundary.
                literal_branches.append(_json_string_literal_regex(val))
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
                enum_branches.append(_json_string_literal_regex(val))  # PAW-SCHEMA-01
            elif isinstance(val, bool):
                enum_branches.append("true" if val else "false")
            elif isinstance(val, (int, float)):
                enum_branches.append(re.escape(str(val)))
            else:
                enum_branches.append(_json_string_literal_regex(str(val)))  # PAW-SCHEMA-01
        return f"(?:{'|'.join(enum_branches)})"

    # 4. Handle List / list[T]
    if origin in (list, List):
        _check_collection_depth(collection_depth)
        item_type = args[0] if args else Any
        item_regex = _type_to_regex(item_type, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 5. Handle Tuple / tuple[A, B] / tuple[T, ...]
    if origin in (tuple, Tuple):
        _check_collection_depth(collection_depth)
        if len(args) == 2 and args[1] is Ellipsis:
            # Variadic: tuple[str, ...] -> same as list[str]
            item_regex = _type_to_regex(args[0], seen=seen, depth=depth, collection_depth=collection_depth + 1)
            return _json_collection_regex(r"\[", r"\]", item_regex)
        elif args:
            # Fixed-length: tuple[str, int, bool] -> [str, int, bool] exact positions
            elem_regexes = [
                _type_to_regex(a, seen=seen, depth=depth, collection_depth=collection_depth + 1) for a in args
            ]
            inner = f"{JSON_WHITESPACE},{JSON_WHITESPACE}".join(elem_regexes)
            return rf"\[{JSON_WHITESPACE}{inner}{JSON_WHITESPACE}\]"
        else:
            # Bare tuple[()] -> empty array
            any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
            return _json_collection_regex(r"\[", r"\]", any_regex)

    # 6. Handle Set / set[T] / FrozenSet / frozenset[T]
    if origin in (set, Set, frozenset, FrozenSet):
        _check_collection_depth(collection_depth)
        item_type = args[0] if args else Any
        item_regex = _type_to_regex(item_type, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 7. Handle Nested Pydantic BaseModel (with cycle detection)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if annotation in seen or depth > _MAX_RECURSION_DEPTH:
            raise PAWSchemaError(
                f"Recursive model detected: {annotation.__name__} at depth {depth}. "
                f"Grammar-constrained decoding cannot express infinite recursion. "
                f"Consider flattening the schema or limiting nesting depth."
            )
        # collection_depth is deliberately not threaded through here: the nested
        # model's own fields (via _pydantic_to_regex_impl -> _type_to_regex) start a
        # fresh collection-nesting context, tracked only against depth+1 above.
        return _pydantic_to_regex_impl(
            annotation,
            anchors=False,
            _seen=seen | {annotation},
            _depth=depth + 1,
        )

    # 8. Handle Dict / dict[K, V] (bare `dict` has no origin, only matches by identity)
    if origin is dict or annotation is dict:
        _check_collection_depth(collection_depth)
        value_type = args[1] if len(args) > 1 else Any
        value_regex = _type_to_regex(value_type, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        entry = rf"{JSON_STRING}{JSON_WHITESPACE}:{JSON_WHITESPACE}{value_regex}"
        return _json_collection_regex(r"\{", r"\}", entry)

    # 9. Bare collection identity checks (no generic args -> get_origin returns None)
    if annotation is list:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", any_regex)
    if annotation is tuple:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", any_regex)
    if annotation is set or annotation is frozenset:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
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
            clean_pattern = _sanitize_field_pattern(pattern_override)  # PAW-SCHEMA-01
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
