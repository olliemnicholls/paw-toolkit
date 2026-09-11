"""Unit tests for paw_kit.test.matching, in particular the quoted-scalar follow-up
(measurements/README.md, "Tool feedback"): `values_equivalent_unquoted` reports the gap
between a strictly-scored adapter and one that simply wraps every answer in quotes,
without loosening what `values_equivalent` itself counts as a match.
"""

from paw_kit.test.matching import values_equivalent, values_equivalent_unquoted


# --------------------------------------------------------------- values_equivalent (strict, unchanged)


def test_strict_quoted_scalar_is_not_equivalent_to_bare_value() -> None:
    """The decision this whole feature exists to respect: a quoted output is a real
    defect, so the strict comparison must keep calling it different."""
    assert values_equivalent('"RG-M2"', "RG-M2") is False
    assert values_equivalent("RG-M2", '"RG-M2"') is False


# --------------------------------------------------------------- values_equivalent_unquoted


def test_unquoted_matches_everything_strict_does() -> None:
    """`values_equivalent_unquoted` is a strict relaxation, not a different rule --
    every strictly-equivalent pair is unquoted-equivalent too."""
    assert values_equivalent_unquoted("RG-M2", "RG-M2") is True
    assert values_equivalent_unquoted('{"a": 1}', '{"a":1}') is True
    assert values_equivalent_unquoted("the   quick fox", "the quick fox") is True


def test_unquoted_json_string_scalar_matches_bare_value() -> None:
    """The measured defect itself: the fast compiler's adapter answered `'"RG-M2"'`
    where the correct bare answer was `RG-M2`."""
    assert values_equivalent_unquoted('"RG-M2"', "RG-M2") is True
    assert values_equivalent_unquoted("RG-M2", '"RG-M2"') is True


def test_unquoted_json_string_versus_json_number_is_not_equivalent() -> None:
    """`'"5"'` vs. `5`: both sides parse as JSON (a string, a number) -- a genuine type
    mismatch, not a quoting artifact -- so unquoting must NOT paper over it."""
    assert values_equivalent_unquoted('"5"', "5") is False
    assert values_equivalent_unquoted("5", '"5"') is False


def test_unquoted_json_string_versus_different_bare_text_still_differs() -> None:
    """Unwrapping the quotes must not make an unrelated answer match."""
    assert values_equivalent_unquoted('"RG-M2"', "RG-M3") is False


def test_unquoted_two_different_quoted_json_strings_not_equivalent() -> None:
    """Both sides are JSON string scalars (not "exactly one side") with different
    values -- strict comparison already decides this correctly, unquoting adds
    nothing and must not force a match."""
    assert values_equivalent_unquoted('"RG-M2"', '"RG-M3"') is False


def test_unquoted_json_object_versus_non_json_text_not_equivalent() -> None:
    """Only a JSON *string scalar* unwraps -- a JSON object/array on one side against
    non-JSON text on the other is not this rule's business."""
    assert values_equivalent_unquoted('{"a": 1}', "not json at all") is False


# --------------------------------------------------------------- H-4: type-aware JSON equality

import pytest  # noqa: E402 -- appended section (bug-hunt-remediation Track B)

#: (a, b, expected) -- the hand-graded table from report section 6's H-4, plus the
#: neighbouring cases needed to show the fix does not over-reach. Every `False` row
#: here returned `True` before the fix; every `True` row must keep returning `True`.
_H4_TABLE = [
    # --- the eight disagreements H-4 names -------------------------------------
    ("false", "0", False),
    ("true", "1", False),
    ("0", "false", False),
    ("1", "true", False),
    ('{"admin": true}', '{"admin": 1}', False),
    ("[true, false]", "[1, 0]", False),
    ("1e400", "1e500", False),          # both overflow to inf
    ("-1e400", "-1e500", False),
    # --- bools still compare to bools ------------------------------------------
    ("true", "true", True),
    ("true", " true ", True),
    ("false", "false", True),
    ("true", "false", False),
    ('{"admin": true}', '{"admin":true}', True),
    ("[true, false]", "[true,false]", True),
    # --- numbers still compare to numbers --------------------------------------
    ("1", "1.0", True),                 # a JSON encoding difference, not a type error
    ("1", "1", True),
    ("1", "2", False),
    ("1e400", "1e400", True),           # byte-identical short-circuit
    # One side finite, one overflowed. Both orders, because the two finiteness checks
    # are separate statements and a single order only reaches the first of them.
    ("1", "1e400", False),
    ("1e400", "1", False),
    ("2.5", "-1e400", False),
    ("-1e400", "2.5", False),
    ("[1, 2]", "[1,2]", True),
    ('{"a": 1}', '{"a": 1.0}', True),
    # --- null ------------------------------------------------------------------
    ("null", "null", True),
    # Reaches `json_values_equal(None, None)` WITHOUT the byte-identical
    # short-circuit, which the row above takes. Mutation testing found the
    # `a is None and b is None` return was otherwise unasserted.
    ("null", " null ", True),
    ('{"a": null}', '{"a":null}', True),
    ("null", "0", False),
    ("null", "false", False),
    ('{"a": null}', '{"a": 0}', False),
    # --- a container against a non-container ------------------------------------
    # These pin the `isinstance(a, list) and isinstance(b, list)` / dict guards. With
    # `and` weakened to `or` the recursion reaches `len()` on an int or `.keys()` on a
    # list and raises, which no test noticed before these rows.
    ("0", "[]", False),
    ("[]", "0", False),
    ("0", "{}", False),
    ("{}", "0", False),
    ("[]", "{}", False),
    ("{}", "[]", False),
    ('"x"', "[]", False),
    ("[1]", '{"a": 1}', False),
    # --- nested ----------------------------------------------------------------
    ('{"a": [true]}', '{"a": [1]}', False),
    ('{"a": {"b": false}}', '{"a": {"b": 0}}', False),
    ('{"a": [1, {"b": "x"}]}', '{"a":[1,{"b":"x"}]}', True),
    ('{"a": 1}', '{"a": 1, "b": 2}', False),
    ("[1]", "[1, 2]", False),
    # --- non-JSON text is untouched by any of this -----------------------------
    ("RG-M2", "RG-M2", True),
    ("the   quick fox", "the quick fox", True),
    ("RG-M2", "RG-Q9", False),
]


@pytest.mark.parametrize("a,b,expected", _H4_TABLE)
def test_h4_values_equivalent_table(a: str, b: str, expected: bool) -> None:
    """H-4: `values_equivalent` treated JSON booleans as equal to 0/1 (Python's
    `False == 0`) and treated two different overflowing numbers as equal (both
    `json.loads` to `inf`). `compare` classified all of those `match_kind="equivalent"`
    and listed them under "not a real disagreement"."""
    assert values_equivalent(a, b) is expected


@pytest.mark.parametrize("a,b,expected", _H4_TABLE)
def test_h4_unquoted_relaxation_still_agrees_on_every_true_row(
    a: str, b: str, expected: bool
) -> None:
    """`values_equivalent_unquoted` is a strict relaxation: it must still say True
    everywhere the strict rule does. (It may say True on some strict-False rows -- the
    quoting cases -- so only the True direction is asserted.)"""
    if expected:
        assert values_equivalent_unquoted(a, b) is True


def test_h4_unquoted_does_not_reopen_the_bool_int_hole() -> None:
    """The relaxation must not become a back door: `'"true"'` vs `1` is neither a
    quoting artifact nor an equal value."""
    assert values_equivalent_unquoted("true", "1") is False
    assert values_equivalent_unquoted('{"admin": true}', '{"admin": 1}') is False


def test_h4_json_values_equal_rejects_a_type_json_never_produces() -> None:
    """The fallback for a value that is not a JSON type at all.

    `json_values_equal` is public and takes `Any`, so it can be handed a `set`, a
    `tuple`, or an arbitrary object by a caller that did not go through `json.loads`.
    Unequal-by-default is the safe answer -- the alternative is reporting two things
    this function cannot compare as "the same value", in the function whose docstring
    says it decides every published number.
    """
    from paw_kit.test.matching import json_values_equal

    assert json_values_equal(set(), set()) is False
    assert json_values_equal({1, 2}, {1, 2}) is False
    assert json_values_equal((1, 2), (1, 2)) is False
    assert json_values_equal(object(), object()) is False


def test_h4_json_values_equal_rejects_non_finite_directly() -> None:
    # Imported inside the test, not at module scope: a module-scope import of a symbol
    # that does not exist at `main` turns the whole file into a collection ERROR, which
    # makes the red-at-main gate pass for the wrong reason (an ImportError, not the
    # behaviour). Every table row above must fail at `main` on its own assertion.
    from paw_kit.test.matching import json_values_equal

    assert json_values_equal(float("inf"), float("inf")) is False
    assert json_values_equal(float("nan"), float("nan")) is False
    assert json_values_equal(float("-inf"), float("inf")) is False
