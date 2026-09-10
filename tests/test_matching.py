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
