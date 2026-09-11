"""Pydantic schema to regular expression compiler for constrained JSON decoding."""

from collections import namedtuple, OrderedDict
import datetime as dt
import enum
import json
import re
import threading
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

# PAW-SCHEMA-06: caps the digit run of a JSON number's integer part (in both
# JSON_INTEGER and JSON_FLOAT -- a bare integer with no decimal point or exponent
# full-matches JSON_FLOAT too). Python's int<->str conversion refuses more than
# `sys.get_int_max_str_digits()` digits (4300 by default) and raises ValueError, so an
# uncapped `[0-9]*` run lets a decoder emit a single absurdly-long digit string that
# crashes `int()`/`json.loads` on the consuming side -- reachable through every `int`
# field and, since JSON_FLOAT's integer part is just as unbounded, every `float` field
# and every `Any`-typed field too. 100 digits is far beyond any realistic integer
# while staying nowhere near the 4300-digit failure point.
_MAX_NUMBER_DIGITS = 100

JSON_WHITESPACE = r"[ \t\n\r]*"

# S-2: the escape sequences JSON actually permits after a backslash. The previous
# spelling of this was `\\.`, which permits ANY character after a backslash, so the
# grammar for the simplest possible schema accepted `{"text":"a\qb"}` and
# `{"text":"\u12"}` -- strings that satisfy the grammar and fail `json.loads`. Driven
# end to end through `RegexLogitsProcessor` every one of those characters was in the
# allowed mask at its step and EOS was allowed at the end, so the headline claim that
# structural validity is a property of the FSM did not hold as written.
#
# The `\uXXXX` form deliberately EXCLUDES the surrogate range D800-DFFF rather than
# accepting all four hex digits. `json.loads` tolerates a lone surrogate
# (`json.loads('"\\ud800"')` returns an unpaired code point) but pydantic's Rust JSON
# parser does not -- `model_validate_json` rejects it with "unexpected end of hex
# escape". Admitting it would therefore leave the grammar WIDER than the validator it
# exists to guarantee, which is the defect this whole change is about. Excluding it is
# the narrowing direction and costs nothing reachable: a non-BMP character is still
# emittable verbatim as raw UTF-8 (see `_json_string_literal_regex`'s
# `ensure_ascii=False` note), which is the form a decoder actually produces.
#
# Verified exhaustively: over all 234,256 four-hex-digit spellings (both letter cases)
# this alternation matches exactly those whose value is outside D800-DFFF, and every
# escape it permits satisfies both `json.loads` and `model_validate_json`.
_JSON_HEX_ESCAPE = r"u(?:[0-9a-cA-Ce-fE-F][0-9a-fA-F]|[dD][0-7])[0-9a-fA-F]{2}"
_JSON_ESCAPE = rf'\\(["\\/bfnrt]|{_JSON_HEX_ESCAPE})'
JSON_STRING = rf'"([^"\\\x00-\x1f\x7f-\x9f]|{_JSON_ESCAPE})*"'
JSON_INTEGER = rf"(-?(0|[1-9][0-9]{{0,{_MAX_NUMBER_DIGITS - 1}}}))"
JSON_FLOAT = rf"(-?(0|[1-9][0-9]{{0,{_MAX_NUMBER_DIGITS - 1}}})(\.[0-9]+)?([eE][+-]?[0-9]+)?)"
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

    `ensure_ascii=False` is deliberate. json.dumps' default (`ensure_ascii=True`)
    rewrites every non-ASCII character as a `\\uXXXX` escape, which would make
    `Literal["café"]` compile to a grammar accepting only `"caf\\u00e9"` and *rejecting*
    the raw UTF-8 `"café"` a decoder actually emits -- a silent behavioural change to
    every non-ASCII Literal/Enum that compiles today (Schema Determinism,
    `decisions.md` #2). Turning it off keeps non-ASCII characters verbatim, exactly as
    the pre-fix `re.escape(val)` did, while still escaping the only characters
    PAW-SCHEMA-01 is about: quotes, backslashes and control characters.
    """
    return re.escape(json.dumps(val, ensure_ascii=False))


def _is_escaped(pattern: str, index: int) -> bool:
    """Return True if the character at `index` is preceded by an odd run of backslashes."""
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and pattern[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _strip_anchors(pattern: str) -> str:
    """Remove leading `^` and trailing `$` anchors, honouring backslash escapes (S-7).

    The previous spelling was `pattern.lstrip("^").rstrip("$")`, which is
    *character-wise*: it removes every trailing `$` regardless of what precedes it. So
    `r"a\$"` -- a perfectly ordinary currency pattern meaning "a followed by a literal
    dollar sign" -- became `a\`, and the stray trailing backslash then escaped the
    closing quote of the JSON string the pattern was spliced into. The grammar ended up
    accepting `{"x": "a"}` (which pydantic rejects) and rejecting `{"x": "a$"}` (which
    pydantic accepts) -- silently wrong in both directions.

    A leading `^` at index 0 is always an anchor (there is nothing in front of it to
    escape it), so a run of them is stripped outright. A trailing `$` is an anchor only
    when it is not itself escaped, which is an odd/even backslash-run question:
    `r"a\$"` (one backslash) is a literal dollar and stays, while `r"a\\$"` (an
    escaped backslash, then the anchor) loses its `$`.

    An anchor anywhere else -- `(?:a$)b` -- is left in place deliberately, and the
    translation that follows refuses it by name rather than guessing at its intent.
    """
    start = 0
    while start < len(pattern) and pattern[start] == "^":
        start += 1
    end = len(pattern)
    while end > start and pattern[end - 1] == "$" and not _is_escaped(pattern, end - 1):
        end -= 1
    return pattern[start:end]


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
    clean = _strip_anchors(pattern)
    if '"' in clean:
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: double quote characters "
            "are forbidden in Field(pattern=...) constraints entirely -- a compiled "
            "regex referencing a quote, escaped or not, can be made to match a "
            "literal JSON-string-terminating quote in the constrained decoder's "
            "output. Remove the quote from the pattern."
        )
    return clean


def _check_model_recursion(annotation: Type[BaseModel], seen: frozenset, depth: int) -> None:
    """Shared cycle/depth guard for nested BaseModel resolution.

    Used by both the regex compiler (`_type_to_regex`) and the cache-key fingerprint
    builder (`_fingerprint_annotation`, PAW-SCHEMA-07) so the two can never drift
    apart on this specific check (Phase 0 Round 3, N-16 flagged exactly this class of
    drift risk for the parallel dispatch the fingerprint builder necessarily is).
    """
    if annotation in seen or depth > _MAX_RECURSION_DEPTH:
        raise PAWSchemaError(
            f"Recursive model detected: {annotation.__name__} at depth {depth}. "
            f"Grammar-constrained decoding cannot express infinite recursion. "
            f"Consider flattening the schema or limiting nesting depth."
        )


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
        elif annotation is Tuple:
            # PAW-SCHEMA-05: bare, unsubscripted `typing.Tuple` legitimately means
            # "an array of anything" -- `Tuple[()]`, `tuple[()]` and bare `typing.Tuple`
            # are otherwise indistinguishable via get_origin/get_args (all three give
            # origin=tuple, args=()), so this identity check is the only way to keep
            # this case permissive while making the two empty-tuple spellings below
            # strict.
            any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
            return _json_collection_regex(r"\[", r"\]", any_regex)
        else:
            # PAW-SCHEMA-05: `Tuple[()]` / `tuple[()]` mean "must be an empty array" --
            # the previous code routed both through the same permissive any_regex
            # collection regex as the bare-Tuple case above, so the compiled grammar
            # accepted arbitrary non-empty arrays for an annotation whose only valid
            # value is `[]` (verified: the old regex matched `{"x": [1,2,3]}` while
            # `M.model_validate({"x": [1]})` raises ValidationError -- a direct breach
            # of the "syntax compliance guaranteed mathematically" invariant, not
            # merely an over-permissive constraint).
            return rf"\[{JSON_WHITESPACE}\]"

    # 6. Handle Set / set[T] / FrozenSet / frozenset[T]
    if origin in (set, Set, frozenset, FrozenSet):
        _check_collection_depth(collection_depth)
        item_type = args[0] if args else Any
        item_regex = _type_to_regex(item_type, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", item_regex)

    # 7. Handle Nested Pydantic BaseModel (with cycle detection)
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _check_model_recursion(annotation, seen, depth)
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


def _fingerprint_annotation(
    annotation: Any,
    *,
    seen: Optional[frozenset] = None,
    depth: int = 0,
    collection_depth: int = 0,
) -> Any:
    """Recursively compute a hashable cache-key fragment for `annotation` (PAW-SCHEMA-07).

    This is a *parallel* dispatch to `_type_to_regex` -- it returns a hashable key
    rather than a regex string, so it cannot simply call or reuse that function's
    branches directly (Phase 0 Round 3, N-16). Its branches are numbered to match
    `_type_to_regex`'s 1:1, to make the two easy to keep in sync by inspection, and it
    shares that function's `_check_model_recursion`/`_check_collection_depth` guards
    outright rather than re-implementing them, so those two specific checks cannot
    drift even though the surrounding dispatch necessarily is duplicated.

    It must raise the exact same `PAWSchemaError` `_type_to_regex` would for a
    recursive or over-deep model, since `pydantic_to_regex` calls this *before*
    attempting to compile a regex at all -- this is the first code to see such a model.

    The correctness property this function exists to provide (verified by
    construction against `_pydantic_to_regex_impl`'s actual input set, Phase 0 Round
    4): for any two annotations `a`, `b`, `_fingerprint_annotation(a) ==
    _fingerprint_annotation(b)` implies `_type_to_regex(a) == _type_to_regex(b)`. The
    converse need not hold -- a fingerprint finer than the regex it keys only costs a
    cache hit, never correctness.
    """
    if seen is None:
        seen = frozenset()

    origin = get_origin(annotation)
    args = get_args(annotation)

    # 1. Union / Optional
    if origin in (Union, types.UnionType):
        return (
            "union",
            tuple(
                _fingerprint_annotation(a, seen=seen, depth=depth, collection_depth=collection_depth)
                for a in args
            ),
        )

    # 2. Literal -- args are already hashable primitive values (str/int/float/bool/None).
    if origin is Literal:
        return ("literal", args)

    # 3. Enum -- member *values* only, matching _type_to_regex's branch (which reads
    # only item.value, never the class itself), so two differently-named Enum classes
    # with the same ordered member values correctly share a fingerprint: they are
    # guaranteed to compile to the same regex.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return ("enum", tuple(item.value for item in annotation))

    # 4. List / list[T]
    if origin in (list, List):
        _check_collection_depth(collection_depth)
        item_type = args[0] if args else Any
        return (
            "list",
            _fingerprint_annotation(item_type, seen=seen, depth=depth, collection_depth=collection_depth + 1),
        )

    # 5. Tuple / tuple[A, B] / tuple[T, ...] / Tuple[()] / bare typing.Tuple
    if origin in (tuple, Tuple):
        _check_collection_depth(collection_depth)
        if len(args) == 2 and args[1] is Ellipsis:
            return (
                "tuple_variadic",
                _fingerprint_annotation(args[0], seen=seen, depth=depth, collection_depth=collection_depth + 1),
            )
        elif args:
            return (
                "tuple_fixed",
                tuple(
                    _fingerprint_annotation(a, seen=seen, depth=depth, collection_depth=collection_depth + 1)
                    for a in args
                ),
            )
        elif annotation is Tuple:
            # PAW-SCHEMA-05: mirrors _type_to_regex's identity check distinguishing
            # bare typing.Tuple (permissive) from Tuple[()]/tuple[()] (strict empty).
            return ("tuple_bare_any",)
        else:
            return ("tuple_empty",)

    # 6. Set / FrozenSet
    if origin in (set, Set, frozenset, FrozenSet):
        _check_collection_depth(collection_depth)
        item_type = args[0] if args else Any
        return (
            "set",
            _fingerprint_annotation(item_type, seen=seen, depth=depth, collection_depth=collection_depth + 1),
        )

    # 7. Nested Pydantic BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        _check_model_recursion(annotation, seen, depth)
        return ("model", _fingerprint_model_fields(annotation, seen=seen | {annotation}, depth=depth + 1))

    # 8. Dict / dict[K, V]
    if origin is dict or annotation is dict:
        _check_collection_depth(collection_depth)
        value_type = args[1] if len(args) > 1 else Any
        return (
            "dict",
            _fingerprint_annotation(value_type, seen=seen, depth=depth, collection_depth=collection_depth + 1),
        )

    # 9. Bare collection identity checks. Deliberately opaque, distinct tags rather
    # than delegating to branch 4/5/6's fingerprint of Any (e.g. `("list",
    # fingerprint(Any))`) -- bare `list` and `List[Any]` do compile to identical
    # regex, so sharing a fingerprint would be *more* correct, not less, but it is
    # not required for correctness (the invariant is one-directional) and a distinct
    # tag is simpler to keep visibly in sync with _type_to_regex's own branch 9.
    if annotation is list:
        _check_collection_depth(collection_depth)
        return ("bare_list",)
    if annotation is tuple:
        _check_collection_depth(collection_depth)
        return ("bare_tuple",)
    if annotation is set or annotation is frozenset:
        _check_collection_depth(collection_depth)
        return ("bare_set",)

    # 10 & 11 (+ fallback). Specialized types, primitive types, and the JSON_STRING
    # fallback are all leaves whose regex output depends on nothing but which branch
    # they are (never on any further recursion), so the annotation object itself --
    # stable and hashable for every type reaching this point -- is a safe, precise key.
    return ("leaf", annotation)


def _fingerprint_model_fields(
    model: Type[BaseModel], *, seen: frozenset, depth: int
) -> Tuple[Tuple[str, Any, Optional[str]], ...]:
    """Ordered per-field cache-key tuple over `model.model_fields` (PAW-SCHEMA-07).

    Captures exactly what `_pydantic_to_regex_impl` reads to build its regex for each
    field -- name (in declaration order, since `model_fields` is itself
    order-preserving and `_pydantic_to_regex_impl` iterates it directly), a recursive
    annotation fingerprint, and any raw `Field(pattern=...)` constraint (verified at
    Phase 0 Round 4 to be `_pydantic_to_regex_impl`'s *entire* input set: field name,
    `_extract_pattern_from_field(field_info)`, and `field_info.annotation` -- never
    `is_required()` or a field's default).

    Built from *extracted values*, never from `FieldInfo` objects directly:
    `FieldInfo` inherits `object`'s identity `__hash__`/`__eq__`, so hashing
    `tuple(model.model_fields.items())` verbatim would still give two structurally
    identical dynamically-created models different keys -- reproducing the exact
    0%-cache-hit-rate bug this finding exists to fix (Phase 0 Round 2, N-12).
    """
    return tuple(
        (
            field_name,
            _fingerprint_annotation(field_info.annotation, seen=seen, depth=depth),
            _extract_pattern_from_field(field_info),
        )
        for field_name, field_info in model.model_fields.items()
    )


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


_REGEX_CACHE_MAXSIZE = 128
_RegexCacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])

# PAW-SCHEMA-07: keyed on a content fingerprint (see _fingerprint_model_fields), not
# on `model` itself. The previous `@lru_cache(maxsize=128)` on this function keyed on
# the class object's identity, so every dynamically-created model (e.g. one built via
# `pydantic.create_model` per-request) was a guaranteed cache miss no matter how many
# structurally identical models had already been compiled and cached -- a 0% hit rate
# for exactly the population this cache exists to help, plus unbounded retention of
# every distinct class object ever passed in (each one worth a full cache slot for its
# own lifetime, since maxsize=128 evicts by recency, not by how many *equivalent*
# entries already exist).
#
# A manual OrderedDict-based LRU replaces functools.lru_cache here because lru_cache
# has no way to key on anything other than its own call arguments -- the fingerprint
# has to be computed from `model` first, then used as the lookup key instead of
# `model` itself. Locking mirrors (not exactly reproduces) functools.lru_cache's own
# concurrency behavior: the lock is held only across the dict lookup/insert, not
# across regex compilation, so two threads racing on an identical uncached model may
# both compute once and each return their own (content-equal, possibly
# non-identical-object) result -- the same trade-off CPython's own lru_cache makes.
_regex_cache: "OrderedDict[Any, str]" = OrderedDict()
_regex_cache_lock = threading.Lock()
_regex_cache_hits = 0
_regex_cache_misses = 0


def pydantic_to_regex(model: Type[BaseModel], anchors: bool = False) -> str:
    """Compile a Pydantic BaseModel class into a strict regex matching compliant JSON.

    Results are cached (LRU, maxsize=128) by content fingerprint -- not by the model
    class object's identity -- to avoid redundant regex compilation and downstream FSM
    construction costs, including for structurally-identical models created
    dynamically via `pydantic.create_model` (PAW-SCHEMA-07).

    Args:
        model: A Pydantic BaseModel subclass.
        anchors: Whether to wrap with ^ and $ anchors (useful for python re.match,
                 omitted for interegular FSM).

    Returns:
        A regular expression pattern enforcing field names, types, commas, and braces.

    Raises:
        PAWSchemaError: If the model contains recursive references that cannot be
                        expressed as a finite regex, or exceeds the maximum
                        BaseModel/collection nesting depth.
    """
    global _regex_cache_hits, _regex_cache_misses

    cache_key = (anchors, _fingerprint_model_fields(model, seen=frozenset(), depth=0))

    with _regex_cache_lock:
        cached = _regex_cache.get(cache_key)
        if cached is not None:
            _regex_cache_hits += 1
            _regex_cache.move_to_end(cache_key)
            return cached
        _regex_cache_misses += 1

    result = _pydantic_to_regex_impl(model, anchors=anchors, _seen=frozenset(), _depth=0)

    with _regex_cache_lock:
        # Don't clobber an entry another thread already inserted for this exact key
        # while the lock was released above (see the module-level comment).
        if cache_key not in _regex_cache:
            _regex_cache[cache_key] = result
            while len(_regex_cache) > _REGEX_CACHE_MAXSIZE:
                _regex_cache.popitem(last=False)
        _regex_cache.move_to_end(cache_key)
    return result


def _pydantic_to_regex_cache_info() -> _RegexCacheInfo:
    with _regex_cache_lock:
        return _RegexCacheInfo(_regex_cache_hits, _regex_cache_misses, _REGEX_CACHE_MAXSIZE, len(_regex_cache))


def _pydantic_to_regex_cache_clear() -> None:
    global _regex_cache_hits, _regex_cache_misses
    with _regex_cache_lock:
        _regex_cache.clear()
        _regex_cache_hits = 0
        _regex_cache_misses = 0


pydantic_to_regex.cache_info = _pydantic_to_regex_cache_info
pydantic_to_regex.cache_clear = _pydantic_to_regex_cache_clear
