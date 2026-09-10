"""Agreement functions for shadow mode: does the compiled adapter's answer match the teacher's?

Track 14. Comparison runs on the *persisted serialized forms* of the two answers, not
on live caller objects: the teacher side is the same string `record_trace` writes, the
adapter side is what the adapter returned. When a `response_model` is in play both
sides are re-validated back into model instances inside the shadow worker before
comparison, so the `shadow_pairs` row is a faithful record of exactly what was compared
and the comparison never holds a reference to a caller object that may be mutated after
the call returns.

The default is deliberately biased toward reporting *disagreement*. A disagreement only
delays a hot-swap; a false agreement ships a wrong answer to production.
"""

import json
import math
import unicodedata
from typing import Any, Callable, Dict, Optional, Tuple

from pydantic import BaseModel

# Recursion budget for the structural comparison below. Exceeding it is a
# disagreement, not a RecursionError: an adapter that returns something pathologically
# deep has not demonstrated it agrees with the teacher.
_MAX_AGREEMENT_DEPTH = 8


def _normalize_str(value: str) -> str:
    """NFC-normalize and strip a string before comparison.

    Deliberately *not* casefolded -- see `default_agreement_fn`.
    """
    return unicodedata.normalize("NFC", value).strip()


def stringify_answer(value: Any) -> str:
    """Serialize a teacher/adapter answer to the one canonical string form.

    Track 14 finding 9: this used to be three near-identical near-duplicates --
    `agreement._stringify` (this function, used below for the str-vs-structured
    comparison rule), `shadow.serialize_answer` (the shadow worker's comparison-output
    persistence path) and an inline block in `decorator.py` (the trace-path
    persistence path) -- and only this one handled `tuple` and used `default=str`. Now
    there is exactly one implementation, imported by both other call sites, so a
    teacher or adapter that returns e.g. a tuple serializes identically no matter
    which path it went through.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, default=str, sort_keys=False)
        except Exception:
            return str(value)
    return str(value)


def _is_structured(value: Any) -> bool:
    return isinstance(value, (BaseModel, dict, list, tuple))


def _compare(a: Any, b: Any, depth: int) -> bool:
    """Structural comparison implementing the rule table in the track's design §1."""
    if depth > _MAX_AGREEMENT_DEPTH:
        return False

    # None -- both sides must be None. `None` vs `""` disagrees.
    if a is None or b is None:
        return a is None and b is None

    # bool BEFORE int: Python's bool is an int subclass, so a naive `==` would make
    # True and 1 agree silently. They must not.
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a is b

    if isinstance(a, str) and isinstance(b, str):
        return _normalize_str(a) == _normalize_str(b)

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        # float, and the int/float cross-type case. Explicitly NOT a domain
        # tolerance -- "urgency score within 1" is a task policy, not a library
        # default. Callers who want one use `field_tolerance_agreement`.
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)

    # Mixed str vs structured: the no-`response_model` path, where the adapter returns
    # raw text and the teacher may have returned a model/dict/list. Both sides are
    # serialized and compared by the str rule. A str vs a bare number is NOT covered
    # here -- that stays a type mismatch, i.e. a disagreement.
    if isinstance(a, str) and _is_structured(b):
        return _normalize_str(a) == _normalize_str(stringify_answer(b))
    if isinstance(b, str) and _is_structured(a):
        return _normalize_str(stringify_answer(a)) == _normalize_str(b)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_compare(x, y, depth + 1) for x, y in zip(a, b))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_compare(a[k], b[k], depth + 1) for k in a)

    if isinstance(a, BaseModel) and isinstance(b, BaseModel):
        if type(a) is not type(b):
            return False
        return _compare(a.model_dump(), b.model_dump(), depth + 1)

    # Any other type mismatch.
    return False


def default_agreement_fn(teacher: Any, adapter: Any) -> bool:
    """Conservative structural equality between a teacher answer and an adapter answer.

    Rules, applied recursively with a depth budget of `_MAX_AGREEMENT_DEPTH`:

    - `str`: NFC-normalized and stripped, then **case-sensitive** exact match.
      `"High"` vs `"high"` is a *disagreement*; a caller whose labels are
      case-insensitive says so with their own `agreement_fn`.
    - `bool`: identity, checked before `int` (`True` vs `1` disagrees).
    - `int`: exact. `float` and `int`/`float` cross-type: `math.isclose` at
      `rel_tol=1e-9`.
    - `None`: both sides must be `None`.
    - `list`/`tuple`: equal length, element-wise, order-sensitive.
    - `dict`: identical key sets, then value-wise.
    - `BaseModel`: same class, then field-wise over `model_dump()`. No partial credit.
    - str vs structured: both serialized to string, compared by the `str` rule.
    - anything else: disagree.

    Never raises: an exception anywhere in the comparison is a disagreement.
    """
    try:
        return _compare(teacher, adapter, 0)
    except Exception:
        return False


def field_tolerance_agreement(
    tolerances: Dict[str, float],
    *,
    base: Callable[[Any, Any], bool] = default_agreement_fn,
) -> Callable[[Any, Any], bool]:
    """Build an `agreement_fn` that allows a numeric tolerance on named fields.

    The documented recipe for the case the repo's own measurement hit:
    `agreement_fn=field_tolerance_agreement({"urgency_score": 1})` reproduces the
    measurement's "urgency within 1" scoring rule for `BaseModel` (or `dict`) outputs.

    Fields not named in `tolerances` are compared with `base` (the conservative
    default). Anything that is not a model/dict pair falls through to `base` entirely.
    """

    def _fields(value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, BaseModel):
            return value.model_dump()
        if isinstance(value, dict):
            return dict(value)
        return None

    def _fn(teacher: Any, adapter: Any) -> bool:
        t_fields = _fields(teacher)
        a_fields = _fields(adapter)
        if t_fields is None or a_fields is None:
            return base(teacher, adapter)
        if isinstance(teacher, BaseModel) and isinstance(adapter, BaseModel):
            if type(teacher) is not type(adapter):
                return False
        if set(t_fields.keys()) != set(a_fields.keys()):
            return False
        for key, t_value in t_fields.items():
            a_value = a_fields[key]
            if key in tolerances:
                if isinstance(t_value, bool) or isinstance(a_value, bool):
                    # bool before int, as everywhere else: a tolerance is meaningless
                    # for a flag.
                    if not base(t_value, a_value):
                        return False
                    continue
                if isinstance(t_value, (int, float)) and isinstance(a_value, (int, float)):
                    if abs(float(t_value) - float(a_value)) > float(tolerances[key]):
                        return False
                    continue
            if not base(t_value, a_value):
                return False
        return True

    return _fn


def safe_agreement(
    fn: Callable[[Any, Any], bool], teacher: Any, adapter: Any
) -> Tuple[bool, Optional[str]]:
    """Run any `agreement_fn` (including a caller's) inside a try/except.

    Returns `(agreed, error_type)`. An `agreement_fn` that raises is a *disagreement*
    recorded as `verdict='error'`; it never propagates and never reaches the caller.
    """
    try:
        return bool(fn(teacher, adapter)), None
    except Exception as exc:
        return False, type(exc).__name__
