"""Smoke tests for runnable examples."""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from paw_kit.cli import app, test_app as paw_test_app
import examples.triage_ticket.run as triage_demo
import examples.pii_scrubber.run as pii_demo
import examples.date_normalizer.run as date_demo

runner = CliRunner()


def test_triage_ticket_example_smoke() -> None:
    """Verify triage_ticket runnable example executes cleanly."""
    triage_demo.main()


def test_pii_scrubber_example_smoke() -> None:
    """Verify pii_scrubber runnable example executes cleanly."""
    pii_demo.main()


def test_date_normalizer_example_smoke() -> None:
    """Verify date_normalizer runnable example executes cleanly."""
    date_demo.main()


def test_date_normalizer_cli_invocation() -> None:
    """Verify paw-test check runs on date_normalizer suite.yaml via CLI, and reports
    the fuzz cases it cannot honestly repair instead of training over them.

    ASSERTIONS CHANGED by H-8(b) (bug-hunt-remediation Track B; see the track file's
    "Justified assertion changes"). This asserted `exit_code == 0` and
    `"All assertions passed" in result.output`. Both were true at `main`, and both were
    true *because of the defect*: the suite ran 78 generated fuzz mutations with no
    answer key, the CLI's two-branch demo teacher answered "2026-01-01" to nearly all
    of them, those labels were compiled into the mock adapter, and the adapter was then
    graded on them and scored 82/82. That is report M-1's finding shape -- the answer
    key in the training set -- in the shipped example, and report H-8's in the
    mechanism that put it there.

    The example's *repairable* cases (the three normalisation edge cases that always
    had known correct answers) were moved from `fuzzing.adversarial_probes` into
    `standard_cases` with those answers, which is the migration the fix requires of any
    suite. What remains failing is genuine: a mock adapter does not normalise
    right-to-left-override-corrupted dates, and now says so.

    The example needs a redesign to be a good demo again -- filed as an addendum
    finding on the parent track, not attempted here.

    **C-2 collateral, 2026-09-13.** This test writes to the example's real (non-tmp)
    `.paw_demo_dates/` directory, not a fixture, so its outcome used to be independent
    of whether an adapter already existed there -- `check`'s pre-C-2 guard exempted
    any adapter declaring `backend: "mock"`, which this one does, so recompilation
    always proceeded either way. After C-2 (any existing adapter, mock or not, blocks
    auto-recompile), running `test_date_normalizer_example_smoke` first in this same
    file leaves a real adapter on disk that this test then finds already present --
    deterministic under pytest's default (unrandomized) in-file ordering, not a flake
    -- and the run takes the read-only path instead of the active-learning path this
    test exists to pin. Clearing the directory first makes the test self-contained
    regardless of what ran before it in the same process, which is what its assertions
    below already assumed.
    """
    demo_dir = Path(__file__).parent.parent / "examples" / "date_normalizer" / ".paw_demo_dates"
    if demo_dir.exists():
        import shutil
        shutil.rmtree(demo_dir)

    suite_path = Path(__file__).parent.parent / "examples" / "date_normalizer" / "suite.yaml"
    result = runner.invoke(paw_test_app, ["check", str(suite_path)])

    assert result.exit_code == 1
    assert "All assertions passed" not in result.output
    # The seven cases that carry an answer key are repaired and scored.
    assert "Correct against expected: 7/7" in result.output
    # And the run says why it stopped, rather than leaving a bare [FAIL].
    collapsed = " ".join(result.output.split())
    assert "were not sent to the teacher" in collapsed
    assert "Stopped because: no_falsifiable_failures" in collapsed
