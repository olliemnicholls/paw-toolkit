"""Tests for the pure, offline-testable parts of `scripts/measure_*.py`.

Every number published in `docs/results.md`, the README and `measurements/README.md` was
produced by one of those scripts, and the 2026-09-11 bug hunt (report §5, findings
B-1..B-9) found nine of them wrong. The scripts themselves cannot be run here -- they need
API keys and spend money -- so what is pinned below is (a) the arithmetic that turns a
recorded run into a published number, exercised against the committed artifact it was
published from, and (b) the train/eval disjointness and summary-completeness properties
whose absence is what let the wrong numbers out.

The scripts are loaded by path, like `tests/test_fiscal_week.py` does: they live in
`scripts/`, which is not a package.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _ROOT / "scripts"
_MEASUREMENTS = _ROOT / "measurements"


def _load(name: str) -> ModuleType:
    """Import `scripts/<name>.py` under its own module name, once per session."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================  B-2 / B-8h
#
# The committed jit-speedup artifact's own `speedup_x` is 3.19 while every published page
# says ~11x, because the split was taken at the call index: the synchronous compile landed
# in "pre" and the adapter download landed in "post".

JIT_ARTIFACT = _MEASUREMENTS / "jit-speedup-3080-20260908-165914.json"


@pytest.fixture(scope="module")
def jit_run() -> dict:
    return json.loads(JIT_ARTIFACT.read_text())


def test_jit_split_reproduces_the_recorded_run(jit_run: dict) -> None:
    """The four-way split and `speedup_x`, against the 20 recorded latencies.

    These four means were re-derived independently twice (the bug hunt, and the Track A
    Phase 0 review) from `calls[]` of the artifact named above, which is untouched.
    """
    mjs = _load("measure_jit_speedup")
    latencies = [c["ms"] for c in jit_run["calls"]]
    assert len(latencies) == 20
    threshold = jit_run["threshold"]
    assert threshold == 5

    stats = mjs.phase_stats(latencies, threshold)
    phases = stats["phases"]

    assert phases["teacher_only"]["n"] == 4
    assert round(phases["teacher_only"]["mean_ms"], 2) == 987.33
    assert phases["compile_call"]["n"] == 1
    assert round(phases["compile_call"]["mean_ms"], 2) == 5449.66
    assert phases["cold_first_local"]["n"] == 1
    assert round(phases["cold_first_local"]["mean_ms"], 2) == 7599.32
    assert phases["steady_state"]["n"] == 14
    assert round(phases["steady_state"]["mean_ms"], 2) == 88.37

    # The published ~11x, now derivable from a committed file.
    assert round(stats["speedup_x"], 2) == 11.17
    assert stats["speedup_definition"] == "mean(teacher_only) / mean(steady_state)"

    # And the contaminated pair the artifact actually contains, to show the split is what
    # differs and not the data: 1879.8 / 589.1 = 3.19.
    assert round(jit_run["pre_threshold"]["mean_ms"], 1) == 1879.8
    assert round(jit_run["post_threshold"]["mean_ms"], 1) == 589.1
    assert round(jit_run["speedup_x"], 2) == 3.19


def test_jit_split_is_a_partition_and_covers_every_call() -> None:
    """No call may be counted twice or dropped, at any threshold the script accepts."""
    mjs = _load("measure_jit_speedup")
    latencies = [float(i) for i in range(1, 21)]
    for threshold in range(1, 19):
        split = mjs.split_latencies(latencies, threshold)
        assert list(split) == list(mjs.SPLIT_PHASES)
        flat = [x for name in mjs.SPLIT_PHASES for x in split[name]]
        assert flat == latencies, threshold
        assert len(split["compile_call"]) == 1
        assert len(split["cold_first_local"]) == 1
        # The compile is the threshold-th call; the first local call is the next one.
        assert split["compile_call"] == [float(threshold)]
        assert split["cold_first_local"] == [float(threshold + 1)]


def test_jit_split_rejects_a_threshold_below_one() -> None:
    mjs = _load("measure_jit_speedup")
    with pytest.raises(ValueError):
        mjs.split_latencies([1.0, 2.0, 3.0], 0)


def test_jit_run_records_what_served_each_call(jit_run: dict) -> None:
    """B-8h: `served_by` must come from the teacher counter, not the call index.

    Nothing asserted that the recorded run's post-threshold calls were served locally --
    a fail-open fallback would have been averaged in as local inference while the script
    printed "0 tokens -- they never left the machine". The script now records `served_by`
    per call and a `teacher_served_after_threshold` count, and conditions that sentence on
    it. The summariser is exercised here; the committed artifact predates the field.
    """
    mjs = _load("measure_jit_speedup")
    src = (_SCRIPTS / "measure_jit_speedup.py").read_text()
    assert '"served_by": served_by' in src
    assert 'served_by = "teacher" if len(teacher_usage) > billed_before else "adapter"' in src
    assert "teacher_served_after_threshold" in src
    # The claim is now conditional on the observation, not on the call index.
    assert "they never left the machine" in src
    assert "WARNING:" in src
    # The recorded run billed exactly `threshold` teacher calls, which is what makes the
    # published "post-threshold calls used 0 tokens" true of *that* run.
    assert jit_run["teacher_tokens_total"]["calls_billed"] == jit_run["threshold"]
    assert mjs.SPLIT_PHASES == (
        "teacher_only", "compile_call", "cold_first_local", "steady_state",
    )
