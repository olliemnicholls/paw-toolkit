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
