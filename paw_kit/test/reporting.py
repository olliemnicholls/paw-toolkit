"""Shared vocabulary for printing a rate that has a denominator worth explaining.

WHY THIS EXISTS
---------------
The 2026-09-11 bug hunt's report calls this "Pattern 1": *a metric whose denominator
quietly excludes its failures*. Six findings in section 6 are instances of it -- an
adapter that raised on every case reporting `Pass rate: 100.0% (10/10)` (H-1), an
always-abstaining adapter reporting `Correct against expected: 10/10` (H-2), a judge
folding its own 30 crashed calls into a 60-case denominator and printing 50% (H-6),
`judge --diff` dividing by zero shared cases and printing `0.0% (0/0)` (H-7).

The campaign's exit criterion (report section 14) is that **every printed rate names
what it excluded**. That is what this module provides, and it is deliberately *not*
what the first draft of this track proposed. That draft was a single per-report
assertion, `numerator + known_failures + errored + skipped == total`, which requires
the buckets to partition the total. Phase 0 measured that against the six candidate
sites and found it arithmetically false at four of them:

* `TestRunReport` -- post-H-1 an errored case lands in `failed_cases` and post-H-2 an
  abstained case stays in `passed_cases`, so `passed + failed == total` already holds
  and adding `errored + abstained` on top double-counts them.
* `CompareReport.errored_count` -- documented as independent of
  `identical`/`pass_a`/`pass_b` (two adapters that both raise are `identical=True`).
* `JudgeReport.unparseable_count` -- documented as folded into `pass_count`.

Only `expected_total` vs `expected_matched` is a genuine partition. So the abstraction
here is one helper **called once per printed rate**, not once per report, and the
partition check is opt-in (`partition=True`) at the one kind of site where it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

__all__ = ["ScoredRate", "denominator_note", "scored_denominator"]


def _ordered_exclusions(excluded: Mapping[str, int]) -> Tuple[Tuple[str, int], ...]:
    """Non-zero exclusion buckets, in the caller's own key order.

    Insertion order is kept deliberately rather than sorted: the caller lists its
    buckets in the order it wants them read ("j abstained, e errored"), and a sort
    would silently reorder a sentence someone wrote on purpose. Zero-count buckets are
    dropped so a clean run does not print "0 abstained, 0 errored" noise.
    """
    return tuple((name, count) for name, count in excluded.items() if count)


def denominator_note(*, scored: int, total: int, excluded: Mapping[str, int]) -> str:
    """One human-readable clause naming what a rate's denominator left out.

    `scored` is the rate's actual denominator, `total` the population it was drawn
    from, and `excluded` the named reasons for the gap (`{"abstained": 2,
    "errored": 1}`). Returns text suitable for appending after a rate, never a
    trailing full stop -- the caller decides the punctuation around it.

    Examples:
        `scored=7, total=10, excluded={"abstained": 2, "errored": 1}`
            -> `"7 of 10 scored; 2 abstained, 1 errored, not scored"`
        `scored=10, total=10, excluded={"abstained": 0}`
            -> `"all 10 scored"`
        `scored=0, total=0, excluded={}`
            -> `"nothing to score"`

    A zero `scored` with a non-zero `total` is the case every one of this cluster's
    findings printed as a confident percentage, so it gets its own explicit wording
    rather than falling out of the general branch.
    """
    if total <= 0:
        return "nothing to score"

    parts = _ordered_exclusions(excluded)
    if scored <= 0:
        if not parts:
            return f"none of {total} scored"
        reasons = ", ".join(f"{count} {name}" for name, count in parts)
        return f"none of {total} scored -- {reasons}"

    if not parts:
        return f"all {total} scored" if scored == total else f"{scored} of {total} scored"

    reasons = ", ".join(f"{count} {name}" for name, count in parts)
    return f"{scored} of {total} scored; {reasons}, not scored"


@dataclass(frozen=True)
class ScoredRate:
    """A rate plus the denominator arithmetic behind it, built by `scored_denominator`.

    `label` names what is being rated ("pass", "correct against expected") and exists
    so a caller printing several of these cannot mix up which note belongs to which
    number.
    """

    label: str
    total: int
    scored: int
    excluded: Mapping[str, int]

    @property
    def note(self) -> str:
        """`denominator_note` for this rate -- see that function."""
        return denominator_note(scored=self.scored, total=self.total, excluded=self.excluded)

    def rate(self, numerator: int) -> float:
        """`numerator / scored` as a percentage, or **0.0** when nothing was scored.

        0.0 rather than a `ZeroDivisionError` matches every existing rate property in
        this package (`TestRunReport.pass_rate`, `JudgeReport.pass_rate`,
        `VerdictDiffReport.flip_rate`), so swapping a call site onto this helper cannot
        change a no-cases run from "0.0%" into a crash. The *reason* a 0.0 here is not
        misleading the way the findings' 0.0s were is that `note` is printed with it.
        """
        return (numerator / self.scored * 100.0) if self.scored > 0 else 0.0

    def render(self, numerator: int) -> str:
        """`"m/k (x%) -- <note>"`, the full printable form.

        The note is always present, including on a clean run ("all 10 scored"), because
        a rate that names its denominator only when something went wrong trains a
        reader to skim past the clause exactly when it matters.
        """
        return f"{numerator}/{self.scored} ({round(self.rate(numerator), 1)}%) -- {self.note}"


def scored_denominator(
    *,
    total: int,
    scored: int,
    excluded: Mapping[str, int],
    label: str,
    partition: bool = False,
) -> ScoredRate:
    """Build a `ScoredRate`, validating the arithmetic the caller claims.

    Args:
        total: the population the rate is drawn from.
        scored: the rate's denominator -- how many of `total` actually got a verdict.
        excluded: named counts for the gap, e.g. `{"abstained": 2, "errored": 1}`.
        label: what is being rated, for the caller's own disambiguation.
        partition: when True, require `scored + sum(excluded.values()) == total`.
            **Opt-in on purpose.** Phase 0 of this track found that four of the six
            sites that print a rate here do not partition their total (see the module
            docstring), so a mandatory check would have to be either wrong or disabled
            at most call sites. Pass it at the sites where the buckets genuinely are a
            partition -- `expected_total` split into matched/abstained/errored is the
            one such shape in this package -- and the arithmetic is then enforced
            rather than assumed.

    Raises:
        ValueError: on a negative count, a `scored` above `total`, or (with
            `partition=True`) buckets that do not add up.
    """
    if total < 0:
        raise ValueError(f"{label}: total must not be negative, got {total}")
    if scored < 0:
        raise ValueError(f"{label}: scored must not be negative, got {scored}")
    if scored > total:
        raise ValueError(
            f"{label}: scored ({scored}) exceeds total ({total}) -- a rate cannot have a "
            "denominator larger than the population it was drawn from"
        )
    for name, count in excluded.items():
        if count < 0:
            raise ValueError(f"{label}: excluded[{name!r}] must not be negative, got {count}")

    if partition:
        excluded_total = sum(excluded.values())
        if scored + excluded_total != total:
            raise ValueError(
                f"{label}: buckets do not partition the total -- scored ({scored}) + "
                f"excluded ({excluded_total}) != total ({total}). Either a case is being "
                "counted twice or one is falling through every bucket."
            )

    return ScoredRate(label=label, total=total, scored=scored, excluded=dict(excluded))
