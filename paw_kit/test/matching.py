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


def values_equivalent(a: str, b: str) -> bool:
    """True if `a` and `b` are the same value: byte-identical, both parse as JSON to
    equal values, or (when they don't both parse as JSON) equal after
    `normalize_whitespace`. Mirrors `paw_kit.test.compare`'s `"equivalent"` match kind
    (byte-identical is a special case of equivalent, not a separate check)."""
    if a == b:
        return True
    a_is_json, parsed_a = parse_json_or_none(a)
    b_is_json, parsed_b = parse_json_or_none(b)
    if a_is_json and b_is_json:
        return parsed_a == parsed_b
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
