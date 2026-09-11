"""Unit tests for `paw_kit.test.reporting` -- the shared "name what the denominator
excluded" helper (bug-hunt-remediation Track B, Phase B1).

These are red at `main` by construction: the module does not exist there.
"""

import pytest

from paw_kit.test.reporting import ScoredRate, denominator_note, scored_denominator


# ----------------------------------------------------------------- denominator_note


def test_note_names_every_non_zero_exclusion_in_caller_order() -> None:
    note = denominator_note(scored=7, total=10, excluded={"abstained": 2, "errored": 1})
    assert note == "7 of 10 scored; 2 abstained, 1 errored, not scored"


def test_note_drops_zero_buckets_but_keeps_the_rest() -> None:
    note = denominator_note(scored=9, total=10, excluded={"abstained": 0, "errored": 1})
    assert note == "9 of 10 scored; 1 errored, not scored"


def test_note_on_a_clean_run_still_says_so() -> None:
    """A rate that names its denominator only when something went wrong trains the
    reader to skim the clause exactly when it matters."""
    assert denominator_note(scored=10, total=10, excluded={}) == "all 10 scored"
    assert denominator_note(scored=10, total=10, excluded={"errored": 0}) == "all 10 scored"


def test_note_for_zero_scored_out_of_a_real_population_is_explicit() -> None:
    """H-1/H-2/H-6/H-7 all printed a confident percentage over a denominator of zero
    (or of nothing but their own failures). That case gets its own wording."""
    assert (
        denominator_note(scored=0, total=10, excluded={"errored": 10})
        == "none of 10 scored -- 10 errored"
    )
    assert denominator_note(scored=0, total=10, excluded={}) == "none of 10 scored"


def test_note_for_an_empty_population() -> None:
    assert denominator_note(scored=0, total=0, excluded={}) == "nothing to score"


# -------------------------------------------------------------------- ScoredRate


def test_rate_and_render() -> None:
    sr = scored_denominator(
        total=10, scored=8, excluded={"abstained": 1, "errored": 1}, label="pass"
    )
    assert isinstance(sr, ScoredRate)
    assert sr.rate(4) == 50.0
    assert sr.render(4) == "4/8 (50.0%) -- 8 of 10 scored; 1 abstained, 1 errored, not scored"


def test_rate_of_zero_scored_is_zero_not_a_crash() -> None:
    """Matches every existing rate property in the package, so moving a call site onto
    this helper cannot turn a no-cases run into a ZeroDivisionError."""
    sr = scored_denominator(total=3, scored=0, excluded={"errored": 3}, label="pass")
    assert sr.rate(0) == 0.0
    assert sr.render(0) == "0/0 (0.0%) -- none of 3 scored -- 3 errored"


def test_excluded_mapping_is_copied_not_aliased() -> None:
    """A caller mutating its own dict afterwards must not retroactively rewrite a
    note that has already been built."""
    buckets = {"errored": 1}
    sr = scored_denominator(total=2, scored=1, excluded=buckets, label="pass")
    buckets["errored"] = 99
    assert sr.note == "1 of 2 scored; 1 errored, not scored"


# ------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": -1, "scored": 0, "excluded": {}},
        {"total": 5, "scored": -1, "excluded": {}},
        {"total": 5, "scored": 6, "excluded": {}},
        {"total": 5, "scored": 5, "excluded": {"errored": -1}},
    ],
)
def test_impossible_arithmetic_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        scored_denominator(label="pass", **kwargs)


def test_non_partitioning_buckets_are_allowed() -> None:
    """Phase 0 found four of six candidate sites do not partition their total, so the
    helper must accept buckets that overlap their siblings -- `errored` cases that are
    also counted as failures, `unparseable` folded into `pass_count`."""
    sr = scored_denominator(total=10, scored=10, excluded={"errored": 3}, label="pass")
    assert sr.scored == 10
    assert sr.note == "10 of 10 scored; 3 errored, not scored"


def test_a_numerator_above_the_denominator_is_refused() -> None:
    """The one piece of bucket arithmetic that is NOT an identity at these call sites,
    and the shape a miscount would take: "11/10 (110.0%)" printed with a straight face.

    (The `partition=True` argument Phase 0 specified is deliberately absent -- at every
    call site `scored` is *derived* from the exclusions, so the identity it would check
    holds by construction. `tools/mutate.py` confirmed it: flipping the flag changed
    nothing observable. See the module docstring.)
    """
    sr = scored_denominator(
        total=10, scored=7, excluded={"abstained": 2, "errored": 1}, label="expected"
    )
    assert sr.rate(7) == 100.0
    with pytest.raises(ValueError, match="exceeds the denominator"):
        sr.rate(8)
    with pytest.raises(ValueError, match="must not be negative"):
        sr.rate(-1)
    with pytest.raises(ValueError, match="exceeds the denominator"):
        sr.render(8)
