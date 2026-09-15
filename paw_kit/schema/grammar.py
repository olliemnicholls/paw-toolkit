"""Pydantic schema to regular expression compiler for constrained JSON decoding."""

from collections import namedtuple, OrderedDict
import datetime as dt
import enum
import json
import re
import threading
import types
import warnings
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
import annotated_types
from pydantic import AliasChoices, AliasPath, BaseModel
from pydantic.fields import FieldInfo

# S-3: `Field(pattern=...)` constraints are translated through interegular's AST rather
# than spliced as regex source (see `_translate_field_pattern`). interegular exports
# only `parse_pattern`, `Pattern`, `Unsupported`, `InvalidSyntax` and `REFlags`, so the
# node types and the parser itself are reached through private names. That coupling is
# deliberate and is pinned by `test_interegular_ast_surface_is_still_what_the_translator_expects`:
# it is a far smaller risk than writing a regex parser of our own, which is what every
# alternative to the AST approach reduces to.
from interegular.patterns import (  # noqa: E402 -- grouped with the other third-party imports
    _CHAR_GROUPS,
    _CharGroup,
    _Concatenation,
    _EMPTY,
    _DOT,
    _NonCapturing,
    _ParsePattern,
    _Repeated,
    InvalidSyntax,
    Pattern as _InteregularPattern,
    REFlags,
    Unsupported,
)

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

# S-19: JSON_FLOAT's exponent used to be `[0-9]+`, unbounded. `1e400` satisfies the
# grammar (and pydantic accepts it, since `json.loads("1e400")` returns `inf`, a
# `float` a `float` field is allowed to hold), but `inf` is not valid JSON to
# re-serialise (`json.dumps(float("inf"))` raises `ValueError`) -- a decoder emitting
# it hands the caller a value the grammar itself calls valid and the JSON spec does
# not. Two digits caps the exponent at 99, which combined with the 100-digit mantissa
# this module already bounds (`_MAX_NUMBER_DIGITS`) stays at or below ~1e198 in the
# worst case -- safely inside float64's ~1.8e308 range, so no reachable combination of
# mantissa and exponent can overflow to inf.
_MAX_EXPONENT_DIGITS = 2

JSON_WHITESPACE = r"[ \t\n\r]*"

# S-2: the escape sequences JSON actually permits after a backslash. The previous
# spelling of this was `\\.`, which permits ANY character after a backslash, so the
# grammar for the simplest possible schema accepted `{"text":"a\qb"}` and
# `{"text":"\u12"}` -- strings that satisfy the grammar and fail `json.loads`. Driven
# end to end through the project's masking engine of the time (a character-level FSM,
# since deleted and replaced by `paw_kit.schema.constraint`'s byte-level engine) every
# one of those characters was in the allowed mask at its step and EOS was allowed at
# the end, so the headline claim that structural validity is a property of the mask did
# not hold as written.
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
JSON_FLOAT = rf"(-?(0|[1-9][0-9]{{0,{_MAX_NUMBER_DIGITS - 1}}})(\.[0-9]+)?([eE][+-]?[0-9]{{1,{_MAX_EXPONENT_DIGITS}}})?)"
JSON_BOOLEAN = r"(true|false)"
JSON_NULL = r"null"

# S-10: `Decimal` compiled to "any JSON string", so the grammar accepted
# `{"x": "hello"}` and rejected `{"x": 1.5}` and `{"x": 42}` -- the natural output for a
# money field, and the whole reason to annotate one as Decimal. pydantic accepts BOTH a
# bare JSON number and a quoted numeric string (executed against pydantic 2.13.5: `1.5`,
# `"1.5"`, `42`, `"42"`, `1e5` and `"1E+5"` all validate), and `model_dump_json()` emits
# the QUOTED form -- `Decimal("1.5")` round-trips as `{"x":"1.5"}` -- so the quoted form
# is not optional: the PAW-SCHEMA-07 kitchen-sink corpus asserts it, and dropping it
# would make every Decimal model fail its own round trip.
#
# The body is JSON's own number grammar rather than everything `Decimal(str)` will
# swallow. pydantic also accepts `"0007"`, `"+1"`, `".5"` and `"1. "`; the grammar
# refuses them, which is the narrowing direction and keeps one spelling per value for a
# decoder to find.
# The two halves are NOT the same grammar, and the difference is load bearing. pydantic
# parses a BARE JSON number for a Decimal field through an f64 first, so an exponent
# that overflows the double range makes it infinite and pydantic then rejects it
# ("Input should be a finite number"): `{"x": 8E383}` would satisfy the grammar and fail
# validation. Caught by the structural fuzz, not by inspection. So the bare form carries
# no exponent at all -- with the integer part capped at _MAX_NUMBER_DIGITS it can then
# never exceed 1e100 and can never overflow. The QUOTED form is parsed as a string by
# `Decimal` itself, exactly and without an f64 in the middle, so it keeps its exponent;
# its digits are capped because pydantic refuses an exponent that does not fit in its
# own integer type (executed: `"1E+99999999999"` validates, `"1E+9999999999999999999"`
# does not), and six digits is far inside that while being far beyond any real schema.
# Losing the bare `1e5` spelling is a narrowing, which rule 1 permits, and costs
# nothing reachable: `model_dump_json()` emits the quoted form.
_DECIMAL_COEFFICIENT = rf"-?(?:0|[1-9][0-9]{{0,{_MAX_NUMBER_DIGITS - 1}}})(?:\.[0-9]+)?"
_DECIMAL_QUOTED_BODY = rf"{_DECIMAL_COEFFICIENT}(?:[eE][+-]?[0-9]{{1,6}})?"
JSON_DECIMAL = rf'(?:{_DECIMAL_COEFFICIENT}|"{_DECIMAL_QUOTED_BODY}")'

# Specialized type regex fragments
JSON_UUID =r'"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"'
# S-11: the day alternation used to be month-INDEPENDENT -- `[0-9]{4}-(?:0[1-9]|1[0-2])
# -(?:0[1-9]|[12][0-9]|3[01])` -- so `"2026-02-30"` and `"2026-04-31"` full-matched
# while pydantic rejects both. `2026-02-30` is a classic LLM output, and a date field is
# one of the main reasons to reach for constrained decoding in the first place.
#
# Four cases, because a calendar has four: 31-day months, 30-day months, February up to
# the 28th in any year, and February the 29th in a leap year only. Accepting 29 February
# unconditionally would leave the grammar wider than the validator for three years in
# four, which is the defect this is fixing rather than a smaller version of it.
#
# `_LEAP_YEAR`'s two branches are "divisible by 4 but not by 100" (the last two digits
# are a non-zero multiple of four) and "divisible by 400" (a century year whose first
# two digits are a multiple of four). `_YEAR` excludes 0000 because `datetime.date`
# does: its minimum year is 1, so `"0000-01-01"` is a string the old `[0-9]{4}` accepted
# and pydantic refuses. Both branches of `_LEAP_YEAR` already imply a year of at least
# 4, so they need no such exclusion.
_LEAP_YEAR = (
    r"(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)"
)
_YEAR = r"(?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})"
_CALENDAR_DATE = (
    rf"(?:{_YEAR}-(?:"
    r"(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01])"  # 31-day months
    r"|(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30)"  # 30-day months
    r"|02-(?:0[1-9]|1[0-9]|2[0-8])"  # February, any year
    rf")|{_LEAP_YEAR}-02-29)"  # February 29th, leap years only
)
# `[T ]` and the optional fraction/offset are unchanged; only the date half moved.
_ISO_TIME = (
    r"[T ](?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"(?:\.[0-9]+)?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])?"
)
JSON_DATE = rf'"{_CALENDAR_DATE}"'
JSON_DATETIME = rf'"{_CALENDAR_DATE}{_ISO_TIME}"'

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


def _json_collection_regex(
    open_lit: str,
    close_lit: str,
    entry_regex: str,
    bounds: Optional[Tuple[int, Optional[int]]] = None,
) -> str:
    """Build a regex matching a bracketed, comma-separated JSON collection body.

    `bounds` is S-9's `(min_items, max_items)`; `None` keeps the previous
    "zero or more" shape byte for byte. Element counts are one more than separator
    counts, which is why the quantifier below is written against `low - 1` / `high - 1`.
    """
    comma_sep = rf"{JSON_WHITESPACE},{JSON_WHITESPACE}{entry_regex}"
    if bounds is None:
        body = rf"(?:{entry_regex}(?:{comma_sep})*)?"
    else:
        low, high = bounds
        if high == 0:
            # The only legal value is the empty collection; a single whitespace run.
            return rf"{open_lit}{JSON_WHITESPACE}{close_lit}"
        tail_high = None if high is None else high - 1
        if low <= 0:
            body = rf"(?:{entry_regex}(?:{comma_sep}){_render_quantifier(0, tail_high)})?"
        else:
            body = rf"{entry_regex}(?:{comma_sep}){_render_quantifier(low - 1, tail_high)}"
    return rf"{open_lit}{JSON_WHITESPACE}{body}{JSON_WHITESPACE}{close_lit}"


def _json_string_regex(bounds: Optional[Tuple[int, Optional[int]]] = None) -> str:
    """`JSON_STRING`, optionally length-bounded (S-9).

    The quantifier goes on the character alternation rather than on the whole string,
    and that is exactly right for pydantic's notion of length: each repetition matches
    one *decoded* character, whether it arrives raw or as a `\\uXXXX` escape, and
    pydantic measures the decoded value.
    """
    if bounds is None:
        return JSON_STRING
    return (
        r'"(?:[^"\\\x00-\x1f\x7f-\x9f]|'
        + _JSON_ESCAPE
        + ")"
        + _render_quantifier(max(bounds[0], 0), bounds[1])
        + '"'
    )


def _extract_pattern_from_field(field_info: FieldInfo) -> Optional[Tuple[str, int]]:
    """Extract a `Field(pattern=...)` constraint as `(source, re_flags)`, if present.

    S-6: this used to return `str(meta.pattern)`. pydantic accepts a *precompiled*
    `re.Pattern` here as readily as a string, and `str(re.compile('[a-z]+'))` is
    `"re.compile('[a-z]+')"` -- so the grammar compiled the pattern's **repr**. Because
    that repr happens to contain lowercase letters, pydantic's search semantics then
    *accepted* the nonsense value, and it flowed through `paw.load` as a successful
    result rather than as a visible error. Silent garbage, not a broken build.

    The flags are returned alongside the source rather than discarded, for two reasons.
    They are semantically load bearing -- `re.compile("abc", re.I)` and
    `re.compile("abc")` are different constraints and pydantic honours the difference
    (executed against pydantic 2.13.5: the first validates `"ABC"`, the second does
    not). And they are part of the cache key: `_fingerprint_model_fields` stores
    whatever this function returns, so before this change the two spellings were told
    apart only accidentally, by the flags showing up in the repr. Returning the source
    alone would have made them collide while compiling differently -- S-4's bug,
    re-introduced through a different door. A plain string pattern carries flags 0.
    """
    for meta in field_info.metadata:
        if hasattr(meta, "pattern") and meta.pattern is not None:
            pattern = meta.pattern
            if isinstance(pattern, re.Pattern):
                return pattern.pattern, pattern.flags
            return str(pattern), 0
    return None


# --- S-9: numeric and length constraints -------------------------------------------
#
# `Field(ge=..., le=...)` and `min_length`/`max_length` were read by nothing at all, so
# `Field(ge=0, le=10)` compiled to a grammar that happily emits `{"x": 99}` and
# `min_length`/`max_length` on a string or a list were dropped on the floor. Several of
# them are straightforwardly expressible as a regex, so that was a gap rather than a
# limit -- and the ones that are not expressible were dropped *silently*, which is the
# report's Pattern 1: a constraint the caller believes is being enforced, and is not.
#
# An integer range is rendered by enumerating it. That is only reasonable for a small
# range; beyond this many values the alternation is longer than it is useful, so a wider
# range is warned about instead. 256 covers the realistic cases -- a percentage, a
# rating, a small enum-like code, a byte.
#
# Re-derived 2026-09-15 against the engine that actually walks this grammar
# (`paw_kit.schema.constraint`), which is lazy, builds no DFA, and bounds construction
# by `INITIAL_LEXER_FUEL` (10,000). Measured on the real 151,936-token vocabulary, as
# the minimum `initial_lexer_fuel` at which `LLMatcher` construction plus its initial
# mask succeeds:
#
#   one int field, ge=0 le=n-1, through this compiler        raw alternation alone
#     n=2      76        n=128   1,322                        n=256     2,550  (9.96/member)
#     n=16    182        n=256   2,612                        n=1,024  10,242 (10.00/member)
#     n=64    674        n=257     102 <- bound dropped        n=4,096  40,985 (10.01/member)
#
# So an integer enumeration costs about **10 fuel per member** -- slightly more than a
# string `Literal` of the same arity, which the same run put at 8.3-8.5 per member
# asymptotically (400 members 3,387; 2,000 members 16,667), matching the figure
# `constraint.py`'s module docstring quotes. One field at the cap costs
# **2,612 -- 26% of the whole grammar's budget**. The cap is doing its job at n=257: the
# bound is dropped and warned about, and the field falls back to the unbounded integer
# rendering at 102 fuel.
#
# **The budget is for the WHOLE grammar, and this constant is not safe against that.**
# Measured, same day, on a model whose only fields are int ranges at the cap:
#
#   1 field   2,612      3 fields   7,760  <- last one that fits
#   2 fields  5,186      4 fields  10,334  <- REFUSED at INITIAL_LEXER_FUEL=10,000
#                        5 fields  12,908
#
# Four `Field(ge=0, le=255)` int fields in one model is an ordinary schema, and
# `build_constraint` refuses it with `PAWSchemaError` at construction. That is a
# fail-open, not invalid output -- `infer()` propagates it unwrapped, `paw.load` routes
# to the fallback, `get_local_fallback_count()` counts it and S-14 warns once -- but it
# is a fail-open on *every* call for that model, which is the money-leak class
# (`decisions.md` §1). **This value is left at 256 deliberately and not tuned here**:
# lowering it silently narrows what this compiler will enforce, and the number that is
# actually mis-set may be `INITIAL_LEXER_FUEL` rather than this one. Neither is a change
# a comment re-derivation gets to make; it needs its own track, with the two constants
# argued together.
_MAX_ENUMERATED_INT_RANGE = 256

# ... and the same question for a LENGTH bound, where the answer used to be much
# sharper and, re-derived 2026-09-15, is no longer the same question at all.
#
# The old argument: a DFA has no loop for `X{m,n}`, so it unrolls, costing roughly
# n * |DFA(X)| states under a character-level DFA engine -- invisible in the rendered
# regex, since `{0,19}` and `*` are the same three characters. That is why these three
# constants exist, and it was true of the engine they were measured against
# (`logits_processor.RegexLogitsProcessor`, capped at `_MAX_FSM_STATES` = 10,000 states
# and a 3 s compile timeout), which `constrained-decoding-real-backend` deleted.
#
# **Under the engine that replaced it, a length bound costs nothing.**
# `paw_kit.schema.constraint` is lazy: it builds no DFA, and `{0,n}` stays a repetition
# in the lexer instead of being unrolled. Measured on the real 151,936-token vocabulary
# as minimum `initial_lexer_fuel`, with this compiler's own budget lifted so the bounds
# were actually rendered rather than warned about:
#
#   str, max_length=32 / 512 / 100,000       643 fuel each -- and 643 unbounded
#   List[nested], max_items=8 / 20 / 1,000   988 fuel each -- and 988 unbounded
#   five `str` fields, max_length=32         963 fuel      -- and 963 unbounded
#   the realistic invoice model below        3,900 unrolled vs 3,902 with the bounds
#                                            dropped (2.56x headroom either way)
#
# The number does not move. Whatever these constants are protecting today, it is not
# the decoder's construction budget, and the state-count table they were derived from
# describes an engine that no longer exists. For the cost shape that *does* bind under
# this engine, see `_MAX_ENUMERATED_INT_RANGE` above: an enumeration costs about 10 fuel
# per member and four 256-value int fields already exceed `INITIAL_LEXER_FUEL`. A length
# bound is free; an enumerated alternation is not.
#
# **What these constants are now.** A rendering policy, not a decoder budget: past them
# this compiler declines to render the bound and warns, naming the budget, exactly like
# a constraint that is inexpressible for semantic reasons. That behaviour is shipped,
# tested and user-visible (the warning text names the numbers), and the fuel measurement
# above is a reason the limits could be *raised*, never a reason they are unsafe. Raising
# them changes what `pydantic_to_regex` enforces for callers who set a bound today, which
# is a semantics change and needs its own track -- so the values stand, and this comment
# no longer claims a decoder cost it cannot demonstrate.
#
# Historical measurement, against the deleted character-level engine, kept as the record
# of why the numbers are what they are:
#
#   str, max_length=64               653 states   0.12 s
#   str, max_length=256            2,573 states   1.45 s
#   str, max_length=512            5,133 states   5.71 s   <- past the compile timeout
#   List[int], max_items=8           852 states   0.11 s
#   List[str], max_items=8           132 states   0.02 s
#   List[nested model], max_items=8  2,116 states 1.51 s
#   List[nested model], max_items=20 7,496 states 19.07 s  <- past both budgets
#   five `str` fields, max_length=64 3,262 states 3.12 s   <- past the compile timeout
#   five `str` fields, max_length=32 1,662 states 0.85 s   <- hence 32
#
# (A string cost about ten states per character rather than one under that engine,
# because each position carried the S-2 escape alternation `\\(["\\/bfnrt]|uXXXX)`.
# Dropping escapes from inside a length-bounded string would have bought an order of
# magnitude there; under the fuel engine there is nothing to buy, so the option lapses
# rather than being taken.)
_MAX_UNROLLED_STRING_LENGTH = 32
_MAX_UNROLLED_COLLECTION_ITEMS = 8
# Note for a mutation-gate reader: `len(unbounded_regex) >= _MAX_UNROLLED_COLLECTION_CHARS`
# is an EQUIVALENT mutant of the `>` below, and provably so rather than by inspection. A
# collection's rendering is its entry regex twice inside a fixed frame, so its length is
# always odd (executed across List/Set/Dict/variadic-Tuple/nested-List families: 69, 71,
# 73, ... and 317, 319, 321, ...). An even budget is therefore never hit exactly and the
# two comparisons agree on every input this compiler can produce.
#
# The item count was the dominant lever under the deleted engine -- at eight items every
# collection this compiler can render measured between 122 and 2,122 states and under a
# second, including a list of nested models. This second limit is the backstop for an
# element whose rendering is enormous in its own right (a deeply nested model), where
# eight copies would be ruinous however few they are. The collection's unbounded
# rendering embeds its entry regex twice, so it is roughly "an entry of 500 characters
# or less". Under the fuel engine neither lever costs anything (see the re-derivation
# above: `List[nested]` needs 988 fuel at 8, 20 and 1,000 items alike), so this limit is
# likewise a rendering policy today rather than a decoder bound.
_MAX_UNROLLED_COLLECTION_CHARS = 1000

_FieldConstraints = namedtuple(
    "_FieldConstraints", ["min_len", "max_len", "ge", "gt", "le", "lt", "other"]
)


def _field_constraints(field_info: FieldInfo) -> _FieldConstraints:
    """Collect the `annotated_types` constraint metadata pydantic records for a field.

    Hashable by construction, because `_fingerprint_model_fields` stores it: a
    constraint the compiler reads has to be a constraint the cache key separates on
    (the track's fingerprint invariant), or `Field(le=10)` and `Field(le=99)` would
    share a grammar.

    `other` holds the reprs of the constraints this compiler cannot express at all
    (`multiple_of`, Decimal's `max_digits`/`decimal_places`, `Predicate`, ...). They are
    kept rather than discarded so they can be named in the warning -- and so that two
    models differing only in one of them still get different cache keys.
    """
    min_len = max_len = ge = gt = le = lt = None
    other: List[str] = []
    for meta in field_info.metadata:
        if isinstance(meta, annotated_types.MinLen):
            # Three separate types, NOT a hierarchy: `MinLen` and `MaxLen` derive from
            # `BaseMetadata` while `Len` is a `GroupedMetadata` protocol carrying both
            # ends. pydantic normally expands `Len` into the other two before this sees
            # it, but all three are handled so a directly-annotated `Len` cannot fall
            # through into `other` and be reported as inexpressible.
            min_len = meta.min_length
        elif isinstance(meta, annotated_types.MaxLen):
            max_len = meta.max_length
        elif isinstance(meta, annotated_types.Len):
            min_len = meta.min_length
            if meta.max_length is not None:
                max_len = meta.max_length
        elif isinstance(meta, annotated_types.Ge):
            ge = meta.ge
        elif isinstance(meta, annotated_types.Gt):
            gt = meta.gt
        elif isinstance(meta, annotated_types.Le):
            le = meta.le
        elif isinstance(meta, annotated_types.Lt):
            lt = meta.lt
        elif hasattr(meta, "pattern") and meta.pattern is not None:
            pass  # handled by `_extract_pattern_from_field` / S-3's translation
        else:
            other.append(repr(meta))
    return _FieldConstraints(min_len, max_len, ge, gt, le, lt, tuple(other))


def _length_constraint_kind(annotation: Any) -> Optional[str]:
    """Which rendering, if any, a `MinLen`/`MaxLen` on this annotation can bound.

    `"string"` -> the quantifier goes on `JSON_STRING`'s character alternation;
    `"collection"` -> it goes on the element count of a `_json_collection_regex` body;
    `None` -> this compiler cannot express it, so it is warned about rather than
    dropped.

    Deliberately conservative. `Optional[str]` is `None` even though the intent is
    obvious: the rendered regex is an alternation over the union's branches and the
    bound belongs to only one of them, so applying it to the whole is wrong and
    applying it to one branch means rewriting the union. Fixed-length and empty tuple
    spellings are `None` too -- their element count is already exact, so a length bound
    is either redundant or contradictory, and neither renders through
    `_json_collection_regex`.

    This mirrors a subset of `_type_to_regex`'s dispatch, which is a drift risk of the
    same shape `_fingerprint_annotation`'s docstring describes. It is pinned by
    `test_every_length_bounded_annotation_actually_changes_the_regex_S_9`, which
    asserts that for every annotation this function claims, the compiled regex really
    does change when a bound is added.
    """
    if annotation is str:
        return "string"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, List, set, Set, frozenset, FrozenSet):
        return "collection"
    if origin is dict or annotation is dict:
        return "collection"
    if origin in (tuple, Tuple) and len(args) == 2 and args[1] is Ellipsis:
        return "collection"
    if annotation in (list, set, frozenset):
        return "collection"
    return None


def _integer_range(constraints: _FieldConstraints) -> Optional[Tuple[int, int]]:
    """Resolve `ge`/`gt`/`le`/`lt` into a closed integer interval, if that is possible.

    Returns `None` -- meaning "warn, do not render" -- when either end is open, when a
    bound is not an integer (a float bound on an int field does not pick out an integer
    interval the way an enumeration needs), or when the span is wider than
    `_MAX_ENUMERATED_INT_RANGE`.
    """

    lows: List[int] = []
    highs: List[int] = []
    for value, offset, into in (
        (constraints.ge, 0, lows), (constraints.gt, 1, lows),
        (constraints.le, 0, highs), (constraints.lt, -1, highs),
    ):
        if value is None:
            continue
        # `bool` is an `int` subclass; `Field(ge=True)` is not an integer bound.
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        into.append(value + offset)
    if not lows or not highs:
        return None
    low, high = max(lows), min(highs)
    if low > high or high - low + 1 > _MAX_ENUMERATED_INT_RANGE:
        return None
    return low, high


def _bounded_integer_regex(low: int, high: int) -> str:
    """Render a closed integer interval as an alternation of its literal spellings.

    Enumeration and not a hand-built digit-range regex: the digit-range construction for
    an arbitrary interval (`-12..307`) is where off-by-one errors live, and the whole
    point of the exercise is that the grammar and the validator agree exactly. The
    interval is capped at `_MAX_ENUMERATED_INT_RANGE` precisely so enumeration stays
    affordable.
    """
    return "(?:" + "|".join(re.escape(str(v)) for v in range(low, high + 1)) + ")"


def _resolve_length_bounds(
    annotation: Any, constraints: _FieldConstraints, unbounded_regex: str,
    field_name: str = "<field>",
) -> Tuple[Optional[Tuple[int, Optional[int]]], Optional[str]]:
    """Decide whether a `MinLen`/`MaxLen` can be rendered, and say why not (S-9).

    Returns `(bounds, None)` when the bound will be applied, or `(None, reason)` when it
    will not -- either because this annotation's rendering cannot carry one, or because
    unrolling it would blow the decoder's state budget (see the measurements above).

    Raises `PAWSchemaError` when the bounds are incoherent -- `min_length > max_length`,
    or `max_length == 0` with `min_length > 0` -- rather than silently emitting a broken
    grammar. This is not a budget question and does not belong in the returned `reason`
    string: an incoherent bound describes a schema with NO legal value, which is exactly
    the case the module's "raise rather than render" invariant (rule 4, see `S-3`) covers
    -- it is unrelated to whether unrolling the bound would be affordable. Left
    unchecked, `_render_quantifier(low, high)` with `low > high` emits `{5,2}`, a string
    `re.compile` refuses outright (`re.error: min repeat greater than max repeat`) and
    that then reaches a caller of `pydantic_to_regex` (e.g.
    `paw_kit.schema.constraint.build_constraint`) as a bare, unwrapped exception -- S-1's
    and S-15's exact failure shapes, reopened through this door. Found by Phase F review.
    """
    kind = _length_constraint_kind(annotation)
    if kind is None:
        return None, f"min_length/max_length on {annotation!r}"
    low = constraints.min_len or 0
    high = constraints.max_len
    if high is not None and low > high:
        raise PAWSchemaError(
            f"{field_name!r} has min_length={low} greater than max_length={high}: no "
            "string or collection can satisfy both, so there is no legal value this "
            "field could ever take."
        )
    # An open-ended `{m,}` still unrolls its m mandatory repetitions, so the cost
    # question is about whichever end is actually pinned.
    repetitions = high if high is not None else low
    if kind == "string" and repetitions > _MAX_UNROLLED_STRING_LENGTH:
        return None, (
            f"min_length/max_length on {annotation!r} (a bound of {repetitions} "
            f"characters exceeds this compiler's unrolling budget of "
            f"{_MAX_UNROLLED_STRING_LENGTH}: a regex has no loop for `{{m,n}}`, so the "
            "decoder's DFA would grow by roughly ten states per permitted character "
            "and could no longer be compiled within its state and time limits)"
        )
    if kind == "collection" and (
        repetitions > _MAX_UNROLLED_COLLECTION_ITEMS
        or len(unbounded_regex) > _MAX_UNROLLED_COLLECTION_CHARS
    ):
        return None, (
            f"min_length/max_length on {annotation!r} (a bound of {repetitions} items "
            f"on an element rendering {len(unbounded_regex)} characters long exceeds "
            f"this compiler's unrolling budget of {_MAX_UNROLLED_COLLECTION_ITEMS} "
            f"items and {_MAX_UNROLLED_COLLECTION_CHARS} characters: a regex has no "
            "loop for `{m,n}`, so the decoder's DFA would grow by one whole copy of "
            "the element per permitted item and could no longer be compiled within its "
            "state and time limits)"
        )
    return (low, high), None


def _describe_dropped_constraints(
    annotation: Any,
    constraints: _FieldConstraints,
    has_pattern: bool,
    length_reason: Optional[str],
) -> List[str]:
    """List, in words, the constraints the compiled grammar will NOT enforce (S-9)."""
    dropped: List[str] = []
    has_length = constraints.min_len is not None or constraints.max_len is not None
    if has_length:
        if has_pattern:
            dropped.append(
                "min_length/max_length alongside a pattern= constraint (the pattern "
                "already determines the accepted language; intersecting the two is not "
                "expressible here)"
            )
        elif length_reason is not None:
            dropped.append(length_reason)
    numeric = [
        name
        for name, value in (
            ("ge", constraints.ge), ("gt", constraints.gt),
            ("le", constraints.le), ("lt", constraints.lt),
        )
        if value is not None
    ]
    if numeric and (annotation is not int or _integer_range(constraints) is None):
        dropped.append(f"{'/'.join(numeric)} on {annotation!r}")
    dropped.extend(constraints.other)
    return dropped


def _warn_dropped_constraints(
    model: Type[BaseModel], field_name: str, dropped: List[str]
) -> None:
    """Warn ONCE per field about constraints the grammar cannot enforce (S-9).

    One warning per field, not one per constraint: a field with four inexpressible
    constraints is one problem, and four lines would train the reader to filter them.

    In practice this fires at most once per *fingerprint*, per process, not once per
    field in any stronger sense: the compile is cached by fingerprint (S-4), and this
    call sits inside the cached path, so a second call with the identical model
    (compiled twice) is the only case genuinely deduplicated. A *different* model class
    that fingerprints identically -- the exact shape `pydantic.create_model` produces
    on every request, which the fingerprint cache exists to serve -- warns on whichever
    one compiles first and silently never again for the rest of the process, because
    the fingerprint carries no class identity. Phase F review, 2026-09-11: filed as an
    addendum finding (S-21b) rather than fixed here, since closing it means deciding
    whether the fingerprint should include a class identity at all, which is a cache
    invalidation policy question for the S-4 code, not a docstring fix.

    A warning and not a raise. Raising would refuse schemas that compile and work today
    -- `Field(multiple_of=3)` has never been enforced by the grammar -- and the parent
    track's fail-open invariant forbids a new raise on a path the caller depends on.
    What was wrong was the silence, not the narrowing.
    """
    if not dropped:
        return
    warnings.warn(
        f"paw_kit: {model.__name__}.{field_name}: the compiled grammar does not "
        f"enforce {', '.join(dropped)}. Grammar-constrained decoding therefore cannot "
        "guarantee this constraint -- a decoder can emit a value the grammar accepts "
        "and your model rejects, which shows up as a validation failure (and, through "
        "`paw.load`, as a fallback) rather than as constrained output. Express the "
        "constraint as a Field(pattern=...) if you need it enforced during decoding.",
        UserWarning,
        stacklevel=4,
    )


def _field_validation_keys(
    model: Type[BaseModel], field_name: str, field_info: FieldInfo
) -> Tuple[str, ...]:
    """Return the JSON object keys pydantic will *validate* for this field (S-5).

    The grammar previously always required the Python field name. With
    `full_name: str = Field(alias="fullName")` and pydantic's default configuration,
    pydantic accepts only `fullName` -- so output that perfectly satisfied the grammar
    raised `ValidationError`, output pydantic accepted was forbidden by the grammar,
    and through `paw.load` the local path failed **100% of the time** while every call
    silently ran the fallback. Any camelCase API schema produced a compiled function
    that was permanently all-fallback: all of the teacher cost, none of the benefit, no
    signal. (The *silence* is S-14 and belongs to Track C; this removes the cause, not
    the symptom.)

    Executed against pydantic 2.13.5, and the reason this targets the **validation**
    alias and never `serialization_alias`:

    * `validation_alias="vName", serialization_alias="sName"` -- pydantic validates
      `vName`, while `model_dump_json(by_alias=True)` emits `{"sName": ...}`, which
      pydantic then **rejects**. A grammar built from the serialization alias would be
      one the validator never accepts.
    * `validation_alias=AliasChoices("a", "b")` -- every choice validates, and
      `model_dump_json(by_alias=True)` emits `{"full_name": ...}`, also rejected. So
      the alternation comes from the choices.
    * `alias="fullName"` alone -- pydantic copies it into `validation_alias`, so
      reading the validation alias covers the plain `alias=` spelling too; the
      fallback to `.alias` below is belt and braces.
    * `alias_generator` needs no handling at all: it is resolved into each `FieldInfo`
      at class-build time, so it arrives here as an ordinary alias.

    Three config flags, not one. pydantic normalises them onto `model_config` at class
    build (executed: `populate_by_name=True` yields
    `{'populate_by_name': True, 'validate_by_alias': True, 'validate_by_name': True}`,
    and `validate_by_alias=False` yields `{'validate_by_alias': False,
    'validate_by_name': True}`), but all three are read so that the older
    `populate_by_name`-only spelling cannot be missed.

    Raises:
        PAWSchemaError: for an `AliasPath`, which addresses a value *inside* a nested
            object and is not expressible in a flat one-key-per-field grammar; and for
            a field with no validatable key at all.
    """
    config = model.model_config
    validate_by_alias = config.get("validate_by_alias", True)
    validate_by_name = bool(config.get("validate_by_name", False)) or bool(
        config.get("populate_by_name", False)
    )

    raw_alias = field_info.validation_alias
    if raw_alias is None:
        raw_alias = field_info.alias

    alias_keys: List[str] = []
    if isinstance(raw_alias, str):
        alias_keys.append(raw_alias)
    elif isinstance(raw_alias, AliasChoices):
        # A choice may itself be an AliasPath. Dropping those is the narrowing
        # direction and keeps the useful case working: pydantic accepts ANY of the
        # choices, so a grammar that emits only the flat ones emits something the
        # validator takes (executed: `AliasChoices("a", AliasPath("x", "y"))` validates
        # `{"a": "v"}`).
        alias_keys.extend(c for c in raw_alias.choices if isinstance(c, str))
        if not alias_keys:
            _raise_inexpressible_alias(model, field_name, raw_alias)
    elif isinstance(raw_alias, AliasPath):
        _raise_inexpressible_alias(model, field_name, raw_alias)
    elif raw_alias is not None:
        raise PAWSchemaError(
            f"Field {model.__name__}.{field_name} carries a validation alias of type "
            f"{type(raw_alias).__name__}, which this compiler does not know how to "
            "express as a JSON object key."
        )

    keys: List[str] = []
    if not alias_keys:
        # No alias at all: the field name is what pydantic validates, whatever
        # `validate_by_alias` says.
        keys.append(field_name)
    else:
        if validate_by_alias:
            keys.extend(alias_keys)
        if validate_by_name:
            keys.append(field_name)
    if not keys:
        raise PAWSchemaError(
            f"Field {model.__name__}.{field_name} has no key pydantic will validate: "
            "its model sets validate_by_alias=False and validate_by_name=False, so "
            "neither the alias nor the field name is accepted. There is no JSON object "
            "this field could appear in."
        )
    # Order-preserving dedupe: an alias equal to the field name must not double the
    # alternation, and the ORDER is part of the compiled artefact.
    return tuple(dict.fromkeys(keys))


def _raise_inexpressible_alias(model: Type[BaseModel], field_name: str, alias: Any) -> None:
    raise PAWSchemaError(
        f"Field {model.__name__}.{field_name} uses {alias!r}. An AliasPath addresses a "
        "value nested inside another object, and the compiled grammar is flat -- one "
        "key per field, at the top level -- so there is no JSON object it could emit "
        "that pydantic would validate through this path. Use a nested BaseModel, or a "
        "plain string alias (optionally via AliasChoices), instead."
    )


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
    r"""Remove leading `^` and trailing `$` anchors, honouring backslash escapes (S-7).

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


# --- S-3: JSON-safe AST translation of Field(pattern=...) ----------------------------
#
# The character set a JSON string can carry *raw*, i.e. the negated class inside
# JSON_STRING above. Every character class the translation renders is intersected with
# this, which is what makes splicing the result between two quote marks safe.
_JSON_UNSAFE_CHARS = frozenset(
    {'"', "\\"}
    | {chr(c) for c in range(0x00, 0x20)}
    | {chr(c) for c in range(0x7F, 0xA0)}
)
_ASCII_CHARS = frozenset(chr(c) for c in range(0x80))

# interegular's AST records a character class as a plain `frozenset` of characters and
# keeps no note of how it was spelled, so `\D` and `[^0-9]` arrive identical -- yet
# pydantic reads the first as "not a Unicode digit" (rejecting U+0663) and the second as
# "not one of these ten characters" (accepting it). Rendering the second reading for a
# pattern that meant the first is the one direction that is not allowed (rule 1), so the
# provenance has to be carried through the parse. It is carried as two private-use
# characters injected into the shorthand groups themselves, because that is the only
# thing `_combine_char_groups` propagates: it merges the positive and negated sides of a
# bracket group into one `chars` set, and a mark on the side it lands on survives. Two
# marks and not one, because a single mark cancels itself out: `[\S\d]` puts it on both
# sides and `neg - pos` then drops it, losing exactly the fact that the class is
# ASCII-under-approximated.
# Spelled as escapes, never as the characters themselves: they are invisible in a
# source file and an editor or a re-encoding could silently eat them.
_SHORTHAND_MARK_POSITIVE = "\ue000"  # injected into \d \w \s
_SHORTHAND_MARK_NEGATED = "\ue001"  # injected into \D \W \S
_SHORTHAND_MARKS = frozenset({_SHORTHAND_MARK_POSITIVE, _SHORTHAND_MARK_NEGATED})
_MARKED_SHORTHANDS = {
    "d": _SHORTHAND_MARK_POSITIVE,
    "w": _SHORTHAND_MARK_POSITIVE,
    "s": _SHORTHAND_MARK_POSITIVE,
    "D": _SHORTHAND_MARK_NEGATED,
    "W": _SHORTHAND_MARK_NEGATED,
    "S": _SHORTHAND_MARK_NEGATED,
}

_POSIX_BRACKET_CLASS = re.compile(
    r"\[:\^?(?:alnum|alpha|ascii|blank|cntrl|digit|graph|lower|print|punct|space"
    r"|upper|word|xdigit):\]"
)

# S-6: how a precompiled pattern's `re` flags are carried into the translation.
#
# EXPRESSIBLE -- rendered into the AST walk as an interegular flag, exactly as a leading
# `(?i)` / `(?s)` inline group already is.
_EXPRESSIBLE_RE_FLAGS: Tuple[Tuple[int, REFlags], ...] = (
    (re.IGNORECASE, REFlags.CASE_INSENSITIVE),
    (re.DOTALL, REFlags.SINGLE_LINE),
)
# IGNORABLE -- ignoring each of these either changes nothing or narrows the grammar,
# which rule 1 permits:
#   * UNICODE is set on every `re.compile` of a `str` pattern, so it carries no
#     information at all (`re.compile("a").flags` is 32).
#   * MULTILINE only changes what `^` and `$` mean, and by the time the AST is walked
#     there are none left: `_strip_anchors` removes the leading/trailing ones and
#     interegular refuses any other position outright. Under MULTILINE pydantic's
#     *search* would additionally accept a value with the match on an inner line
#     (`^a$` matching `"b\na"`); full-matching the stripped pattern rejects that, which
#     is the narrowing direction.
#   * ASCII narrows the shorthand classes to ASCII, and this translator already
#     ASCII-restricts every shorthand-derived class it renders in negated position
#     (rule 2), so honouring it could only ever remove characters the grammar has
#     already removed.
# Anything else -- VERBOSE above all, which changes how the pattern *source* is
# tokenised, and LOCALE, which has no meaning for a `str` pattern -- is refused rather
# than silently dropped, since dropping it would change what the pattern matches.
_IGNORABLE_RE_FLAGS = re.UNICODE | re.MULTILINE | re.ASCII


def _interegular_flags(pattern: str, flags: int) -> REFlags:
    """Translate a precompiled pattern's `re` flags into interegular's flag set (S-6)."""
    out = REFlags(0)
    known = _IGNORABLE_RE_FLAGS
    for re_flag, ie_flag in _EXPRESSIBLE_RE_FLAGS:
        known |= re_flag
        if flags & re_flag:
            out |= ie_flag
    leftover = flags & ~known
    if leftover:
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: it was supplied as a "
            f"precompiled regex carrying the flag(s) {re.RegexFlag(leftover)!r}, which "
            "this compiler cannot express in a decoding grammar. Only re.IGNORECASE "
            "and re.DOTALL are supported (re.MULTILINE, re.ASCII and re.UNICODE are "
            "accepted and have no effect on the compiled grammar). Rewrite the "
            "constraint without the flag, or spell it as a leading inline group."
        )
    return out


# `_ParsePattern.extension_group` is entered with the cursor just past the opening `(?`,
# so a flag group at the very start of the pattern -- the only position at which the
# three engines agree about what it means -- is the one entered at index 2.
_PATTERN_INITIAL_GROUP_INDEX = 2


class _ShorthandTrackingParser(_ParsePattern):
    """interegular's parser, with two things recorded that its AST throws away.

    `escaped` returns the module-level singleton out of `_CHAR_GROUPS` for a shorthand,
    which is what makes the identity check below exact: no other path produces those
    objects. `extension_group` is the only place `self.flags` is assigned, so noting the
    cursor there records where each inline flag group sat.
    """

    def __init__(self, data: str) -> None:
        super().__init__(data)
        self.inline_flag_positions: List[int] = []

    def escaped(self, inner: bool = False) -> Any:
        node = super().escaped(inner)
        for key, mark in _MARKED_SHORTHANDS.items():
            if node is _CHAR_GROUPS[key]:
                return _CharGroup(node.chars | {mark}, node.negated)
        return node

    def extension_group(self) -> Any:
        at = self.index
        before = self.flags
        node = super().extension_group()
        if self.flags is not before:
            self.inline_flag_positions.append(at)
        return node


def _char_set_intersect(
    a: Tuple[bool, FrozenSet[str]], b: Tuple[bool, FrozenSet[str]]
) -> Tuple[bool, FrozenSet[str]]:
    """Intersect two character sets, each `(is_positive, chars)`.

    `(True, s)` is exactly `s`; `(False, s)` is every character except `s`.
    """
    (a_pos, a_chars), (b_pos, b_chars) = a, b
    if a_pos and b_pos:
        return True, a_chars & b_chars
    if a_pos:
        return True, a_chars - b_chars
    if b_pos:
        return True, b_chars - a_chars
    return False, a_chars | b_chars


def _case_expand(chars: FrozenSet[str]) -> FrozenSet[str]:
    """Close `chars` over ASCII/simple case folding, as `(?i)` requires.

    Multi-character case mappings (`'ß'.upper() == 'SS'`) are dropped rather than
    spliced in as a string: they are not expressible as a member of a character class.
    Dropping them widens a *negated* class, which is why a negated class under `(?i)` is
    ASCII-restricted in `_render_char_group` -- within ASCII this closure is exact.
    """
    out: Set[str] = set()
    for char in chars:
        out.add(char)
        for variant in (char.lower(), char.upper(), char.casefold()):
            if len(variant) == 1:
                out.add(variant)
    return frozenset(out)


def _escape_in_class(char: str) -> str:
    """Render one character for use inside a `[...]` character class."""
    code = ord(char)
    if code < 0x20 or 0x7F <= code <= 0x9F:
        # `\xHH` and not `\uHHHH`: interegular's parser implements `\x` (exactly two hex
        # digits) and raises Unsupported for `\u`, so the escape has to be the one both
        # engines read. Every character that needs escaping here is below U+00A0.
        return f"\\x{code:02x}"
    if char in "\\]^-[":
        return "\\" + char
    return char


def _escape_literal(char: str) -> str:
    """Render one character for use outside a character class."""
    code = ord(char)
    if code < 0x20 or 0x7F <= code <= 0x9F:
        return f"\\x{code:02x}"
    if char in ".^$*+?{}[]()|\\":
        return "\\" + char
    return char


def _render_char_set(is_positive: bool, chars: FrozenSet[str]) -> str:
    """Render a character set as a regex atom, collapsing contiguous runs into ranges.

    Ranges and not enumerations (`[0-9]`, never `[0123456789]`): the JSON-safe negated
    classes this produces span thousands of code points, so writing them as ranges
    rather than enumerating every member keeps the compiled regex a size a reader can
    take in -- `examples/` tells users to read the compiled regex directly.
    """
    if is_positive and len(chars) == 1:
        return _escape_literal(next(iter(chars)))
    codes = sorted(ord(c) for c in chars)
    pieces: List[str] = []
    start = 0
    while start < len(codes):
        end = start
        while end + 1 < len(codes) and codes[end + 1] == codes[end] + 1:
            end += 1
        if end - start >= 2:
            pieces.append(_escape_in_class(chr(codes[start])) + "-" + _escape_in_class(chr(codes[end])))
        else:
            pieces.extend(_escape_in_class(chr(c)) for c in codes[start : end + 1])
        start = end + 1
    return ("[" if is_positive else "[^") + "".join(pieces) + "]"


def _render_quantifier(low: int, high: Optional[int]) -> str:
    # Second line of defense behind `_resolve_length_bounds`'s raise: every caller of
    # this function is expected to have already refused an incoherent bound, so this
    # assertion should never fire in production. It exists because `{5,2}` -- what an
    # unchecked `low > high` renders -- is exactly the kind of string that looks like a
    # regex, is accepted this far down the pipeline with no error, and only breaks on
    # `re.compile`, several call frames away from the field that caused it.
    assert high is None or low <= high, (
        f"_render_quantifier({low}, {high}): low must not exceed high -- the caller "
        "should have raised PAWSchemaError before reaching the renderer"
    )
    if high is None:
        if low == 0:
            return "*"
        if low == 1:
            return "+"
        return f"{{{low},}}"
    if low == 0 and high == 1:
        return "?"
    if low == high:
        return f"{{{low}}}"
    return f"{{{low},{high}}}"


def _raise_empty_intersection(source: str, described: str, lost: FrozenSet[str]) -> None:
    """Rule 4: refuse, rather than render a sub-expression that can match nothing."""
    detail = ""
    if '"' in lost or "\\" in lost:
        detail = (
            " Note that double quote characters are forbidden in Field(pattern=...) "
            "constraints entirely, escaped or not: in a regex a single backslash before "
            'a quote (`a\\"b`) does not require a backslash in the matched text at all, '
            "so a pattern that looks escaped in its source can still be made to match "
            "the literal quote that terminates the JSON string."
        )
    raise PAWSchemaError(
        f"Invalid field pattern constraint {source!r}: {described} can only match "
        f"characters a JSON string cannot carry ({''.join(sorted(lost))!r}), so it has "
        f"no JSON-safe form and the compiled grammar would accept nothing at that "
        f"position.{detail}"
    )


def _render_char_group(
    chars: FrozenSet[str], negated: bool, flags: REFlags, source: str
) -> str:
    """Translate one `_CharGroup` into a JSON-safe regex atom."""
    marks = _SHORTHAND_MARKS & chars
    chars = chars - _SHORTHAND_MARKS
    if flags & REFlags.CASE_INSENSITIVE:
        chars = _case_expand(chars)
    char_set: Tuple[bool, FrozenSet[str]] = (not negated, chars)

    # Rule 2, and the soundness condition on `_case_expand`. Only the *negated* case
    # needs it. A positive class is exactly the characters it lists, and a shorthand
    # contributes only ASCII ones, so it is already no wider than pydantic reads it. A
    # negated class excludes them instead, and under-excluding is the widening
    # direction: `\D` as interegular reads it admits U+0663, which pydantic rejects.
    if negated and (marks or flags & REFlags.CASE_INSENSITIVE):
        char_set = _char_set_intersect(char_set, (True, _ASCII_CHARS))

    safe = _char_set_intersect(char_set, (False, _JSON_UNSAFE_CHARS))
    if safe[0] and not safe[1]:
        # Only a positive set can empty out here (the complement of a finite set never
        # can), so `char_set[1]` is exactly the characters the sub-expression could have
        # matched -- all of which JSON forbids.
        described = (
            f"the literal {next(iter(char_set[1]))!r}"
            if len(char_set[1]) == 1
            else "a character class"
        )
        _raise_empty_intersection(source, described, char_set[1])
    return _render_char_set(*safe)


def _render_node(node: Any, flags: REFlags, source: str) -> Tuple[str, bool]:
    """Render one AST node, returning `(regex, is_atomic)`.

    `is_atomic` says whether a quantifier can be appended directly, which is what keeps
    the output free of gratuitous `(?:...)` wrapping.
    """
    # `__DotCls` and `__EmptyCls` are name-mangled inside `interegular.patterns` and are
    # not importable by name, so they are reached through the singletons the parser
    # actually returns. `isinstance` rather than `is`: the singletons are the only
    # instances the parser makes today, but a type check does not silently mistranslate
    # if that ever stops being true.
    if isinstance(node, type(_EMPTY)):
        return "", True

    if isinstance(node, type(_DOT)):
        excluded: FrozenSet[str] = frozenset() if flags & REFlags.SINGLE_LINE else frozenset("\n")
        return _render_char_set(*_char_set_intersect((False, excluded), (False, _JSON_UNSAFE_CHARS))), True

    if isinstance(node, _CharGroup):
        return _render_char_group(node.chars, node.negated, flags, source), True

    if isinstance(node, _Repeated):
        # A reversed quantifier bound written directly into a `Field(pattern=...)`
        # source regex (e.g. `a{5,2}`) is not currently reachable end-to-end --
        # pydantic itself rejects all such spellings at class-build time with its own
        # `SchemaError`, before a model carrying one could ever reach the compiler --
        # but `_render_quantifier`'s own assertion for this exact condition claims
        # "the caller should have raised PAWSchemaError before reaching the renderer",
        # which was false for this caller specifically: nothing upstream of it
        # validated a regex-embedded bound the way `_resolve_length_bounds` validates
        # a pydantic-constraint one. Raised here directly so the invariant holds for
        # both callers uniformly, rather than resting entirely on pydantic's own
        # unrelated rejection to keep this branch unreachable.
        if node.max is not None and node.min > node.max:
            raise PAWSchemaError(
                f"Pattern quantifier {{{node.min},{node.max}}} has a minimum greater "
                "than its maximum: no repetition count can satisfy both, so there is "
                "no legal value this pattern could ever match."
            )
        inner, atomic = _render_node(node.base, flags, source)
        if not inner:
            return "", True
        if not atomic:
            inner = f"(?:{inner})"
        # Reported as NOT atomic: a quantified atom cannot itself take a quantifier.
        # interegular parses `a**` as a repetition of a repetition (its `atom` consumes
        # the first `*` and its `obj` the second), so returning True here rendered
        # `a**` verbatim -- a string Python `re` refuses with "multiple repeat", which
        # is exactly the failure mode this whole translation exists to remove. The
        # wrapping is only ever added where a quantifier actually follows.
        return inner + _render_quantifier(node.min, node.max), False

    if isinstance(node, _Concatenation):
        parts: List[Tuple[str, bool]] = []
        for part in node.parts:
            if isinstance(part, _NonCapturing):
                raise PAWSchemaError(
                    f"Invalid field pattern constraint {source!r}: lookahead and "
                    "lookbehind are not expressible as a finite automaton, so they "
                    "cannot be compiled into a decoding grammar. Remove the "
                    "zero-width group."
                )
            rendered = _render_node(part, flags, source)
            if rendered[0]:
                parts.append(rendered)
        if len(parts) == 1:
            return parts[0]
        return "".join(text for text, _ in parts), not parts

    if isinstance(node, _InteregularPattern):
        scoped = (flags | node.added_flags) & ~node.removed_flags
        options = [_render_node(option, scoped, source) for option in node.options]
        if len(options) == 1:
            return options[0]
        return "(?:" + "|".join(text for text, _ in options) + ")", True

    raise PAWSchemaError(
        f"Invalid field pattern constraint {source!r}: the compiler does not know how "
        f"to translate a {type(node).__name__} node into a JSON-safe grammar."
    )


def _translate_field_pattern(pattern: str, flags: int = 0) -> str:
    """Translate a `Field(pattern=...)` constraint into a JSON-string-safe regex.

    `flags` are the `re` flags of a precompiled `re.Pattern` constraint (0 for a plain
    string one); see `_interegular_flags` for which are honoured and which are refused.

    S-8, DECIDED: the result is spliced into a position that is **full**-matched, even
    for an unanchored pattern. pydantic applies `pattern` with *search* semantics, so
    the grammar is narrower than the validator here -- which is the one direction the
    track's rule 1 permits, and cannot produce an unsound grammar. The alternative,
    emulating search as `(?:json-safe)*(?:pattern)(?:json-safe)*`, is equally sound and
    makes the constraint almost vacuous for a decoder, which is the opposite of the
    point. "Require an anchor and raise otherwise" was explicitly rejected: the
    `[0-9]{5}` zip-code constraint in `tests/test_schema.py` is unanchored and works
    today. Documented on `pydantic_to_regex` for users, and pinned by
    `test_unanchored_pattern_is_still_full_matched_S_8`.

    S-3, and the reason `_sanitize_field_pattern` no longer exists. The old approach
    spliced the user's regex *source* between JSON quote marks and defended the splice
    by refusing a literal `"` in the source. That defence is not sufficient and cannot
    be made sufficient: nothing stops a character *class* from matching a quote, so
    `Field(pattern=".*")` accepted a value consisting of a single bare quote (so the
    emitted object was `{"x": <quote><quote><quote>, "y": 1}`), and `\\S+`, `[^a]+`,
    `[\\w\\W]+` and `\\D+` all did the same -- strings that satisfy the grammar and are
    not JSON. The quote ban only blocked the one spelling nobody uses.

    Nor can it be fixed by rewriting the source: `[.]` is a literal dot, `\\.` is an
    escaped one, and `[^\\D]` means "digit", so textually substituting `\\D` yields
    `[^[^0-9...]]`, which Python `re` mis-parses rather than rejects. A safe source
    rewriter has to reimplement a regex parser.

    So the pattern is parsed into `interegular`'s AST -- the same parser the decoder
    uses -- and re-*rendered* as an explicit regex whose every character class has
    already been intersected with the set of characters a JSON string can carry. The
    output contains no shorthand class, no anchor, no capturing group and no inline
    flag, so Python `re`, `interegular` and pydantic's rust-regex all read it the same
    way. Four rules govern the translation:

    1. **Narrow, never widen.** Every transformation must shrink the accepted language
       or leave it alone. This is what makes the whole thing safe: the compiled grammar
       is allowed to reject something pydantic would accept (the grammar is already
       deliberately narrower), and is never allowed to accept something pydantic
       rejects.
    2. **A class derived from `\\d \\D \\w \\W \\s \\S` is additionally ASCII-restricted
       where it appears in negated position.** interegular's shorthands are ASCII;
       pydantic's (rust-regex) and Python `re`'s are Unicode. Executed:
       `parse_pattern(r"\\D").to_fsm().accepts("\u0663")` is True while `re.fullmatch` and
       pydantic both reject it -- so rendering interegular's reading verbatim would be
       *wider* than the validator (addendum S-3b). `.` and explicit classes are **not**
       ASCII-restricted: all three engines agree there, and `_json_string_literal_regex`
       deliberately keeps non-ASCII literals working (`ensure_ascii=False`).
    3. **Constructs the two engines read differently are refused.** Executed:
       `Field(pattern="[[:alpha:]]+")` -- pydantic accepts `"ahl]"` and `"[:a]"` on a
       POSIX reading while Python `re` full-match rejects them, so the grammar would be
       broader than the validator. Same for an inline flag group that is not at the very
       start: pydantic applies `(?i)` from that point onwards, `interegular` applies it
       to the whole pattern, and Python `re` refuses the pattern outright.
    4. **An empty intersection raises; it is never rendered as "match nothing".** A
       nomatch grammar hands the decoder an all-`-inf` mask at step 0, which is the S-12
       failure by a different door.

    Second accepted residual limitation, found while implementing rule 2 and not
    anticipated by the plan: a bracket group that mixes a shorthand with its own
    negation, `[\\w\\W]` being the idiomatic spelling of "any character at all",
    collapses in interegular's AST to "the complement of nothing" with only the
    shorthand provenance left behind. There is then no way to tell it apart from
    `[a-zA-Z0-9_\\W]`, which genuinely does exclude every non-ASCII word character, so
    both are ASCII-restricted. `[\\w\\W]` therefore stops matching non-ASCII input.
    That is the narrowing direction (rule 1), so it is safe; `.` is the spelling that
    keeps non-ASCII, and separating the two cases would need interval algebra over
    symbolic Unicode classes rather than interegular's plain frozensets.

    Accepted residual limitation, documented rather than fixed: after translation a
    `Field(pattern=".*")` value cannot contain a quote or a backslash *even escaped*, so
    `He said "hi"` is unreachable although pydantic accepts it. Admitting it is possible
    and would be sound (emit `\\\\"` and friends as alternatives beside the safe class,
    since the JSON escape decodes to the character `.` matched) but roughly doubles the
    rendered length for a case no schema author has yet asked for.

    Raises:
        PAWSchemaError: for a construct that cannot be expressed as a finite automaton
            (`\\b`, lookaround, `\\p{L}`), one the two engines disagree about, or one
            whose JSON-safe intersection is empty.
    """
    initial_flags = _interegular_flags(pattern, flags)
    if _SHORTHAND_MARKS & set(pattern):
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: it contains a Unicode "
            f"private-use character ({_SHORTHAND_MARKS!r}) that the pattern translator "
            "reserves for tracking where a shorthand class came from. Remove it."
        )
    posix = _POSIX_BRACKET_CLASS.search(pattern)
    if posix is not None:
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: the POSIX bracket class "
            f"{posix.group(0)!r} is not supported. pydantic's regex engine reads it as "
            "a named class while Python's `re` reads it as an ordinary set of "
            "characters, so a grammar built from it would accept values pydantic "
            "rejects. Spell the class out (e.g. `[a-zA-Z]` for `[:alpha:]`)."
        )

    clean = _strip_anchors(pattern)
    parser = _ShorthandTrackingParser(clean)
    try:
        node = parser.parse().simplify()
    except (Unsupported, InvalidSyntax) as exc:
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: "
            f"{type(exc).__name__}: {exc}. Grammar-constrained decoding needs a pattern "
            "expressible as a finite automaton; zero-width assertions (\\b, \\B), "
            "anchors other than a leading ^ / trailing $, lookaround, backreferences "
            "and Unicode property classes (\\p{...}) are not."
        ) from exc

    # An inline flag group is a *global* flag wherever it appears, for both interegular
    # (`_ParsePattern.start` applies it to the whole pattern) and Python `re` (which
    # refuses it outright anywhere but position 0). pydantic's rust-regex instead
    # applies it from that point onwards, so anywhere but the very start the three
    # engines mean three different things -- rule 3.
    if parser.inline_flag_positions not in ([], [_PATTERN_INITIAL_GROUP_INDEX]):
        raise PAWSchemaError(
            f"Invalid field pattern constraint {pattern!r}: an inline flag group such "
            "as `(?i)` is only supported at the very start of the pattern. Elsewhere "
            "pydantic applies it from that point onwards while this compiler (and "
            "`interegular`) would apply it to the whole pattern, and Python's `re` "
            "refuses the pattern outright. Move the flag to the start, or use a scoped "
            "group like `(?i:...)`."
        )

    rendered, _atomic = _render_node(node, initial_flags, pattern)
    return rendered


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
    length_bounds: Optional[Tuple[int, Optional[int]]] = None,
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
        length_bounds: S-9's `(min, max)` from a `Field(min_length=..., max_length=...)`
            on the field this annotation belongs to. Applied at the TOP level only --
            no recursive call passes it on, because the constraint belongs to the field
            and not to its element type. `_length_constraint_kind` decides in advance
            whether this annotation is one of the shapes below that can honour it; a
            bound reaching any other branch would be silently dropped, which is what
            that function and its test exist to prevent.

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
        return _json_collection_regex(r"\[", r"\]", item_regex, length_bounds)

    # 5. Handle Tuple / tuple[A, B] / tuple[T, ...]
    if origin in (tuple, Tuple):
        _check_collection_depth(collection_depth)
        if len(args) == 2 and args[1] is Ellipsis:
            # Variadic: tuple[str, ...] -> same as list[str]
            item_regex = _type_to_regex(args[0], seen=seen, depth=depth, collection_depth=collection_depth + 1)
            return _json_collection_regex(r"\[", r"\]", item_regex, length_bounds)
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
        return _json_collection_regex(r"\[", r"\]", item_regex, length_bounds)

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
        return _json_collection_regex(r"\{", r"\}", entry, length_bounds)

    # 9. Bare collection identity checks (no generic args -> get_origin returns None)
    if annotation is list:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", any_regex, length_bounds)
    if annotation is tuple:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", any_regex)
    if annotation is set or annotation is frozenset:
        _check_collection_depth(collection_depth)
        any_regex = _type_to_regex(Any, seen=seen, depth=depth, collection_depth=collection_depth + 1)
        return _json_collection_regex(r"\[", r"\]", any_regex, length_bounds)

    # 10. Specialized types
    if annotation is uuid.UUID:
        return JSON_UUID
    if annotation is dt.datetime:
        return JSON_DATETIME
    if annotation is dt.date:
        return JSON_DATE
    if annotation is Decimal:
        return JSON_DECIMAL  # S-10

    # 11. Primitive types
    if annotation is str:
        return _json_string_regex(length_bounds)
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
    #
    # S-4: the value's TYPE NAME is part of the key, not just the value. Fingerprints
    # are compared with `==`, and in Python `1 == True == 1.0` with equal hashes, so
    # `("literal", (1,))` and `("literal", (True,))` were the same cache key -- while
    # `_type_to_regex`'s branch 2 renders them as `1` and `true`, which are different
    # regexes. Compiling `Literal[1]` and then `Literal[True]` handed the second model
    # the first model's grammar: `B.model_dump_json()` is `{"x":true}`, which does not
    # match B's own compiled grammar, while `{"x":1}` does. Silent and order-dependent
    # -- whichever model was compiled first won for the rest of the process -- and the
    # docstring's invariant ("equal fingerprint implies equal regex") was simply false.
    if origin is Literal:
        return ("literal", tuple((type(v).__name__, v) for v in args))

    # 3. Enum -- member *values* only, matching _type_to_regex's branch (which reads
    # only item.value, never the class itself), so two differently-named Enum classes
    # with the same ordered member values correctly share a fingerprint: they are
    # guaranteed to compile to the same regex. Each value's type name is included for
    # the same reason as branch 2 (S-4): an int-valued and a bool-valued enum member
    # compare equal and compile differently.
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return ("enum", tuple((type(item.value).__name__, item.value) for item in annotation))

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

    **The fingerprint input set must track the compiler input set.** Any change that
    widens what `_pydantic_to_regex_impl` reads has to widen this in the same commit,
    or two models that compile differently share a cache entry (S-4 is one instance of
    that, not a one-off). Every widening so far is routed through the two helpers this
    function already calls, precisely so the two cannot drift:
    `_extract_pattern_from_field` returns the pattern's flags as well as its source
    (S-6), `_fingerprint_annotation` carries each Literal/Enum value's type name (S-4),
    and `_field_validation_keys` -- the same call `_pydantic_to_regex_impl` makes to
    decide which object key to emit -- contributes the resolved alias set (S-5). The
    alias set is NOT derivable from the field name: two models identical except for
    `Field(alias=...)`, or for `populate_by_name`, compile to different grammars. And
    `_field_constraints` -- again the same call the compiler makes -- contributes the
    length and range metadata (S-9), including the constraints the compiler can only
    warn about, since two models differing only in a `multiple_of` should not be told
    apart by luck.

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
            _field_validation_keys(model, field_name, field_info),
            _field_constraints(field_info),
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

    # S-5, found by the structural fuzz: two fields can resolve to the SAME JSON object
    # key once aliases are in play -- `Field(alias="x")` on two different fields, or an
    # alias equal to another field's name. pydantic builds such a model without
    # complaint and then reads the duplicate key last-wins, so the grammar would emit
    # `{"x": <for field a>, "x": <for field b>}`, which validates one field against the
    # other's value. Before aliases were read at all the keys were the Python field
    # names and could not collide, so this is a hazard the S-5 fix introduces and has to
    # close. Refuse, per the "never emit an unsound grammar in preference to an error"
    # invariant.
    claimed: Dict[str, str] = {}
    for field_name, field_info in fields.items():
        for key in _field_validation_keys(model, field_name, field_info):
            if key in claimed and claimed[key] != field_name:
                raise PAWSchemaError(
                    f"Fields {claimed[key]!r} and {field_name!r} of {model.__name__} "
                    f"both validate the JSON object key {key!r} (via Field(alias=...) "
                    "or validation_alias). A JSON object cannot carry the same key "
                    "twice unambiguously, so there is no grammar that feeds both fields "
                    "correctly. Give them distinct aliases."
                )
            claimed[key] = field_name

    field_patterns: List[str] = []
    # Deliberate: every field, `Optional`/defaulted or not, renders as a MANDATORY grammar
    # key -- never omittable, only nullable. `is_required()` and a field's default are not
    # part of this compiler's input at all (see `_fingerprint_model_fields`'s docstring
    # above for the full, deliberately-closed input set); the key is always present in the
    # object, and its value regex always accepts `null` when the field is `Optional`. This
    # gives the decoder a deterministic next key to emit at every position rather than a
    # choice of "emit or skip", and it means the compiled grammar is narrower than
    # pydantic's own acceptance (it never accepts less than what pydantic requires, but it
    # can reject a shorter, also-valid JSON form pydantic would accept) -- never wider.
    # Making optional keys actually omittable is a real but separate piece of work (roughly
    # O(n^2) regex growth to preserve a fixed key order across a variable set of present
    # keys, plus a corresponding widening of `_fingerprint_model_fields`'s input set); see
    # `grammar-omittable-optional-keys` if this ever needs revisiting.
    for field_name, field_info in fields.items():
        # S-5: the key(s) pydantic will VALIDATE, which is the field name only when the
        # field carries no alias. `_json_string_literal_regex` rather than
        # `f'"{re.escape(name)}"'`: an alias is an arbitrary string (pydantic accepts
        # `alias='fu"ll'`), so it needs the same PAW-SCHEMA-01 treatment as a Literal
        # value. For an ordinary identifier the two spellings are byte-identical.
        constraints = _field_constraints(field_info)
        keys = _field_validation_keys(model, field_name, field_info)
        rendered_keys = [_json_string_literal_regex(k) for k in keys]
        field_key = (
            rendered_keys[0] if len(rendered_keys) == 1 else "(?:" + "|".join(rendered_keys) + ")"
        )
        length_reason: Optional[str] = None
        # Check for Field(pattern=...) constraint
        pattern_override = _extract_pattern_from_field(field_info)
        if pattern_override is not None:
            # S-3 / PAW-SCHEMA-01: translated through interegular's AST and re-rendered
            # against the JSON-safe character set, never spliced as source. S-6: a
            # precompiled constraint contributes its source and its flags, never its
            # repr.
            value_regex = f'"{_translate_field_pattern(*pattern_override)}"'
        else:
            # S-9: honour the length and range constraints a regex can express, and
            # warn about the ones it cannot instead of dropping them in silence.
            annotation = field_info.annotation
            # The unbounded rendering is needed either way: it is the answer when no
            # bound applies, and its SIZE is what decides whether unrolling a
            # collection bound is affordable (see `_resolve_length_bounds`).
            value_regex = _type_to_regex(annotation, seen=_seen, depth=_depth)
            if constraints.min_len is not None or constraints.max_len is not None:
                bounds, length_reason = _resolve_length_bounds(
                    annotation, constraints, value_regex, field_name
                )
                if bounds is not None:
                    value_regex = _type_to_regex(
                        annotation, seen=_seen, depth=_depth, length_bounds=bounds
                    )
            if annotation is int:
                int_range = _integer_range(constraints)
                if int_range is not None:
                    value_regex = _bounded_integer_regex(*int_range)
        _warn_dropped_constraints(
            model,
            field_name,
            _describe_dropped_constraints(
                field_info.annotation, constraints, pattern_override is not None,
                length_reason,
            ),
        )
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

    **`Field(pattern=...)` is applied as a FULL match, including when the pattern is
    unanchored** (S-8). pydantic applies `pattern` with *search* semantics, so
    `Field(pattern=r"[0-9]+")` validates `"2026-01-02"` -- the compiled grammar does
    not, and will only emit a value the whole pattern matches. This is deliberate and
    is the narrowing direction (a full match is a subset of a search, so the grammar
    can never accept a value pydantic would reject). Emulating search -- wrapping the
    constraint as `(?:any)*(?:pattern)(?:any)*` -- would be equally sound and would make
    the constraint nearly vacuous for decoding, which is the opposite of the point of
    constraining a decoder at all. Anchor a pattern or not as you please; `^`/`$` at the
    ends are stripped and change nothing. If you want "contains", write it: `.*foo.*`.

    **`Field(alias=...)` decides the object key** (S-5). The grammar requires whichever
    key(s) pydantic will *validate* -- the validation alias, the field name, or both,
    according to the model's `populate_by_name` / `validate_by_name` /
    `validate_by_alias` settings. Note that with an alias and the default settings this
    is NOT what `model_dump_json()` emits, because that defaults to `by_alias=False`;
    pydantic will not read its own output back either, and the grammar follows the
    validator.

    **Length and range constraints are honoured where a regex can express them, and
    warned about where it cannot** (S-9). `min_length`/`max_length` become a `{m,n}`
    quantifier on a `str` field's characters or on a list/set/dict field's element
    count; a closed integer interval no wider than 256 values (`Field(ge=0, le=10)`)
    becomes an alternation over its members. Everything else -- an open-ended range, a
    float range, `multiple_of`, Decimal's `max_digits`, a length bound on
    `Optional[str]` or alongside a `pattern=` -- raises a `UserWarning` naming the
    field and the constraint, once per field. It is a warning and not an error because
    those schemas compile and work today; what was wrong was that the grammar quietly
    did not enforce them.

    **Caveat on `set`/`frozenset`/`dict` (S-21, not yet closed):** `max_length` on these
    is sound (a JSON array can only lose entries when read back as a set or dict, so an
    upper bound on rendered items stays an upper bound after that collapse) but
    `min_length` is currently **not enforced and not warned about**: the grammar bounds
    the rendered item *count*, and `{"x": [1, 1, 1]}` satisfies a `min_length=3` bound
    while pydantic's resulting `{1}` fails it. Prefer `list` with a `pattern=` if you
    need a real lower bound on a deduplicating container.

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
