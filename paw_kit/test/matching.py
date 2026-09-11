"""Shared output-equivalence normalisation, used by both `paw-test compare` and
`paw-test check`'s per-case `expected` comparison (see `paw_kit.test.runner`).

Lifted out of `paw_kit.test.compare` (where it was introduced to stop a whitespace-only
`json.dumps` formatting difference from reading as "these two programs completely
disagree", see that module's docstring) so `paw_kit.test.runner` can compare a case's
output against its suite-carried `expected` field with the *same* normalisation, rather
than a second, independently-drifting implementation of "are these two strings the same
answer". `runner.py` cannot import from `compare.py` directly -- `compare.py` already
imports `evaluate_assertion` from `runner.py`, and the reverse import would be circular.
"""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any, Tuple


def parse_json_or_none(text: str) -> Tuple[bool, Any]:
    """`(True, value)` if `text` parses as JSON, `(False, None)` otherwise."""
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        return False, None


def normalize_whitespace(text: str) -> str:
    """Unicode NFC normalize, then collapse all whitespace runs (including leading/
    trailing) to single spaces -- `str.split()` with no argument already does the
    collapsing half; NFC first so two visually-identical strings encoded differently
    (e.g. composed vs. decomposed accents) don't register as a difference either."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _json_kind(value: Any) -> str:
    """The JSON type of a `json.loads` result: one of `"null"`, `"bool"`, `"number"`,
    `"string"`, `"array"`, `"object"`, or `"unknown"`.

    `bool` is tested **first**, before `int`, because in Python `bool` is a subclass of
    `int` -- which is H-4's entire root cause. Naming the type once, here, is what lets
    `json_values_equal` say "different types are not equal" in a single place instead of
    as a chain of paired `isinstance(a, T) and isinstance(b, T)` guards. Those pairs are
    also untestable: weakened from `and` to `or` they raise or fall through to the same
    answer on every input, so `tools/mutate.py` reports them as survivors that no test
    can ever kill.
    """
    if isinstance(value, bool):
        return "bool"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def json_values_equal(a: Any, b: Any) -> bool:
    """Type-aware structural equality for two `json.loads` results (H-4).

    Python's `==` is the wrong comparison for JSON values in two specific ways, both
    of which `values_equivalent` inherited and both of which flatter an adapter:

    1. **`False == 0` and `True == 1`.** A hand-graded 51-pair table disagreed with the
       old `==` on `false`/`0`, `true`/`1`, `{"admin": true}`/`{"admin": 1}` and
       `[true, false]`/`[1, 0]` -- `paw-test compare` classified all four
       `match_kind="equivalent"` and listed them under "not a real disagreement". A
       JSON-emitting small model writing `1` where the schema says `true` is a classic
       failure mode, and this is the function whose docstring claims it "decides every
       published number".
    2. **Overflow to infinity.** `json.loads("1e400")` and `json.loads("1e500")` both
       return `inf`, so two different wrong numbers compared equal. Any non-finite
       float (`inf`, `-inf`, `nan`) is treated as *not* equal to anything here;
       `values_equivalent`'s byte-identical short-circuit still matches a token against
       itself, so only genuinely different tokens are affected.

    `int`/`float` of equal value (`1` vs `1.0`) are still equal -- that is a JSON
    encoding difference with no semantic content, unlike a bool/number confusion.
    """
    kind = _json_kind(a)
    if kind != _json_kind(b):
        return False

    if kind == "null":
        return True
    if kind == "bool":
        return a is b
    if kind == "number":
        # Each side checked separately, deliberately. Written as
        # `not (isfinite(a) and isfinite(b))` the guard is untestable: with one side
        # non-finite the subsequent `==` is False anyway, so weakening the `and` to
        # `or` changes no result and no test can pin it. Two statements, each of which
        # decides an outcome on its own, is the same rule with the same behaviour and
        # a boundary a test can actually reach.
        if not math.isfinite(a):
            return False
        if not math.isfinite(b):
            return False
        return a == b
    if kind == "string":
        return a == b
    if kind == "array":
        return len(a) == len(b) and all(json_values_equal(x, y) for x, y in zip(a, b))
    if kind == "object":
        return a.keys() == b.keys() and all(json_values_equal(a[k], b[k]) for k in a)
    return False


def values_equivalent(a: str, b: str) -> bool:
    """True if `a` and `b` are the same value: byte-identical, both parse as JSON to
    equal values (`json_values_equal`, which is type-aware -- see H-4 in its
    docstring), or (when they don't both parse as JSON) equal after
    `normalize_whitespace`. Mirrors `paw_kit.test.compare`'s `"equivalent"` match kind
    (byte-identical is a special case of equivalent, not a separate check)."""
    if a == b:
        return True
    a_is_json, parsed_a = parse_json_or_none(a)
    b_is_json, parsed_b = parse_json_or_none(b)
    if a_is_json and b_is_json:
        return json_values_equal(parsed_a, parsed_b)
    return normalize_whitespace(a) == normalize_whitespace(b)


def values_equivalent_unquoted(a: str, b: str) -> bool:
    """True under `values_equivalent`, or when unwrapping a JSON-string-quoted side
    makes the two equivalent -- e.g. `'"RG-M2"'` (a JSON string scalar) vs. the bare
    `RG-M2`. Measured directly: `measurements/README.md`, "Tool feedback" -- a lookup
    adapter that quoted every output scored 0/300 against `values_equivalent`, even
    though its unquoted answers were the right lookup value on a third of cases. The
    strict rule stays strict (a quoted output is a real defect for any consumer of the
    adapter); this is a *reporting* relaxation, not a change to what "matches" means.

    "Unwrapping" only fires when exactly one side parses as JSON at all, and that
    side's parsed value is a string -- deliberately narrower than "one side is a JSON
    string scalar and the other isn't". `'"5"'` vs. `5` must NOT be equivalent here:
    both sides parse as JSON (a string, a number) -- a real type mismatch, not a
    quoting artifact -- so it falls through to the (correctly negative) strict check.
    Only when the *other* side fails to parse as JSON at all is a quoted string scalar
    treated as "the same bare text, just quoted".
    """
    if values_equivalent(a, b):
        return True
    a_is_json, parsed_a = parse_json_or_none(a)
    b_is_json, parsed_b = parse_json_or_none(b)
    if a_is_json and isinstance(parsed_a, str) and not b_is_json:
        return values_equivalent(parsed_a, b)
    if b_is_json and isinstance(parsed_b, str) and not a_is_json:
        return values_equivalent(a, parsed_b)
    return False
