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


# ================================================================================  B-1
#
# The published 60% triage agreement folded 5 of the 20 tickets it scored. Those five
# carried their own answer inside the adapter's prompt, and agreed 5/5; the fifteen held
# out agreed 7/15 = 46.7%.

TRIAGE_ARTIFACT = _MEASUREMENTS / "triage-semantic-agreement-3080-20260909-002033.json"


@pytest.fixture(scope="module")
def triage_run() -> dict:
    return json.loads(TRIAGE_ARTIFACT.read_text())


def test_triage_folding_pool_is_disjoint_from_the_scored_set() -> None:
    """`set(folding_inputs) & set(eval_inputs) == set()` over the script's constants.

    This is the assertion report §5 asked for by name.
    """
    mts = _load("measure_triage_semantic_agreement")
    assert set(mts.FOLDING_TICKETS) & set(mts.TICKETS) == set()
    assert len(mts.FOLDING_TICKETS) == 5
    assert len(mts.TICKETS) == 20
    # And the runtime guard agrees, rather than the constants merely happening to be fine.
    mts.assert_folding_pool_disjoint(mts.FOLDING_TICKETS, mts.TICKETS)


def test_triage_disjointness_guard_actually_raises() -> None:
    """The guard must be falsifiable: re-creating B-1's exact arrangement must stop."""
    mts = _load("measure_triage_semantic_agreement")
    with pytest.raises(RuntimeError, match="overlaps the evaluation set"):
        mts.assert_folding_pool_disjoint(mts.TICKETS[:5], mts.TICKETS)


def test_triage_adapter_path_does_not_collide_with_the_shadow_mode_adapter() -> None:
    """The folded-pool change alters `full_spec`, hence the compile, hence the adapter.

    `measure_shadow_mode.py` pins the 2026-09-09 adapter path as `ADAPTER` and the
    committed shadow-mode artifact embeds that manifest as `adapter_manifest`, so a
    re-run writing to the same path would make that measurement irreproducible.
    """
    mts = _load("measure_triage_semantic_agreement")
    msm_src = (_SCRIPTS / "measure_shadow_mode.py").read_text()
    assert 'ADAPTER = "measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw"' in msm_src
    assert mts.ADAPTER_FILENAME == "triage_semantic_agreement_heldout-paw-4b-qwen3-0.6b.paw"
    assert mts.ADAPTER_FILENAME not in msm_src
    shadow_artifact = json.loads(
        (_MEASUREMENTS / "shadow-mode-3080-20260910-124735.json").read_text()
    )
    assert shadow_artifact["adapter_manifest"]["program_id"]


def test_triage_score_rows_reproduces_the_recorded_leak(triage_run: dict) -> None:
    """The split, against the untouched `cases[]` of the 2026-09-09 artifact.

    A named recomputation: `measure_triage_semantic_agreement.score_rows` over
    `measurements/triage-semantic-agreement-3080-20260909-002033.json`'s `cases[]`, with
    `folded_inputs = set(TICKETS[:5])` -- the tickets that run folded.
    """
    mts = _load("measure_triage_semantic_agreement")
    rows = triage_run["cases"]
    assert len(rows) == 20
    assert triage_run["full_agreement_rate"] == 60.0

    slices = mts.score_rows(rows, set(mts.TICKETS[:5]))

    assert slices["folded_into_spec"]["n"] == 5
    assert slices["folded_into_spec"]["full_agreement"] == 5
    assert slices["folded_into_spec"]["full_agreement_rate"] == 100.0

    assert slices["heldout"]["n"] == 15
    assert slices["heldout"]["full_agreement"] == 7
    assert round(slices["heldout"]["full_agreement_rate"], 1) == 46.7
    assert round(slices["heldout"]["urgency_within_1_rate"], 1) == 86.7

    assert slices["all_scored"]["n"] == 20
    assert slices["all_scored"]["full_agreement_rate"] == 60.0
    assert slices["all_scored"]["urgency_within_1_rate"] == 90.0


def test_triage_summary_carries_a_heldout_denominator(triage_run: dict) -> None:
    """Regression assert from report §5: a held-out denominator in the artifact itself.

    The defect was not only the folding: the summary reported one mixed rate with no way
    for a reader to separate the slices. `build_summary` is pure, so the shape of what
    would be written is checkable without an API key.
    """
    mts = _load("measure_triage_semantic_agreement")
    rows = [dict(r, folded_into_spec=False) for r in triage_run["cases"]]
    summary = mts.build_summary(
        label="test",
        adapter_path="measurements/" + mts.ADAPTER_FILENAME,
        manifest={"program_id": "deadbeef", "public": False,
                  "examples_folded_into_spec": 5, "folded_example_ids": ["a"]},
        rows=rows,
        folded_inputs=set(mts.FOLDING_TICKETS),
    )

    assert summary["n_heldout"] == 20
    assert summary["full_agreement_rate_heldout"] == 60.0
    assert summary["urgency_within_1_rate_heldout"] == 90.0
    assert summary["slices"]["heldout"]["n"] == 20
    assert summary["slices"]["folded_into_spec"]["n"] == 0
    assert summary["leak_flags"]["folding_pool_disjoint_from_eval"] is True
    assert summary["leak_flags"]["scored_rows_folded_into_spec"] == 0
    # Visibility and identity of the adapter are recorded, not assumed (A-2 / Phase 0 #9).
    assert summary["public"] is False
    assert summary["program_id"] == "deadbeef"

    # And the same summariser over B-1's arrangement reports the leak instead of hiding it.
    leaked_rows = [
        dict(r, folded_into_spec=r["ticket"] in set(mts.TICKETS[:5]))
        for r in triage_run["cases"]
    ]
    leaked = mts.build_summary(
        label="test", adapter_path="x.paw", manifest={}, rows=leaked_rows,
        folded_inputs=set(mts.TICKETS[:5]),
    )
    assert leaked["n_heldout"] == 15
    assert round(leaked["full_agreement_rate_heldout"], 1) == 46.7
    assert leaked["full_agreement_rate"] == 60.0
    assert leaked["leak_flags"]["scored_rows_folded_into_spec"] == 5


def test_triage_requests_a_private_compile_explicitly() -> None:
    """`public=False` is passed, not inherited from the backend default.

    The backend already defaults to `public=False`
    (`paw_kit/backend/programasweights.py`), but this script relied on that silently while
    a public compile would have published all five folded ticket bodies verbatim.
    """
    mts_src = (_SCRIPTS / "measure_triage_semantic_agreement.py").read_text()
    assert "public=False," in mts_src
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend
    import inspect
    assert inspect.signature(ProgramAsWeightsBackend).parameters["public"].default is False
