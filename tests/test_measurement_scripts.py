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
    # A-2 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    # `summary["public"]` became `summary["public_requested"]`, because the manifest key
    # it mirrors recorded the *request* under a name that read as the confirmed fact.
    # The fixture manifest above deliberately still uses the legacy `public` key, so this
    # also pins the legacy fallback: a summary built from a pre-rename manifest must
    # still report the value rather than silently None.
    assert summary["public_requested"] is False
    # Never checked is never "private": an old manifest carries no confirmation at all.
    assert summary["public_confirmed"] is None
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


# ================================================================================  B-3
#
# Arm D of the fiscal run was called with a hardcoded `max_tokens=400` that the artifact
# recorded nowhere, while the compiled arms ran unbounded. 39 of 300 answers were truncated
# and scored as failures with no label.

FISCAL_ARTIFACT = _MEASUREMENTS / "finetune-fiscal-3080-20260910-190137.json"


@pytest.fixture(scope="module")
def fiscal_run() -> dict:
    return json.loads(FISCAL_ARTIFACT.read_text())


def test_fiscal_reference_max_tokens_is_the_fiscal_value_not_the_lookup_one() -> None:
    """2000, and specifically not the lookup script's task-specific 30.

    `measure_finetune_lookup.py` has a constant of the same name set to 30 because its
    answer is six characters. Copying that value here would truncate essentially every
    arm-D answer rather than 13% of them, which is a worse version of the same bug.
    """
    mff = _load("measure_finetune_fiscal")
    mfl = _load("measure_finetune_lookup")
    assert mff.REFERENCE_MAX_TOKENS == 2000
    assert mfl.REFERENCE_MAX_TOKENS == 30
    src = (_SCRIPTS / "measure_finetune_fiscal.py").read_text()
    assert "max_tokens=REFERENCE_MAX_TOKENS" in src
    assert "max_tokens=400" not in src


def test_fiscal_summary_records_every_reference_sampling_parameter() -> None:
    """Report §5's named test for B-3: the summary must record all of them.

    The 2026-09-10 artifact records `reference_model` and `reference_temperature` and
    nothing else -- the token cap that decided 39 of its 300 results is absent. Asserted
    against the script's own keys, not against a re-run, since a re-run costs money.
    """
    mff = _load("measure_finetune_fiscal")
    src = (_SCRIPTS / "measure_finetune_fiscal.py").read_text()
    summary_block = src[src.index('summary = {\n        "label": args.label,'):]
    for key in ("reference_model", "reference_temperature", "reference_max_tokens",
                "reference_sampling"):
        assert f'"{key}"' in summary_block, key
    sampling = src[src.index('"reference_sampling": {'):]
    for field in ("model", "temperature", "max_tokens", "top_p", "top_k",
                  "stop_sequences", "system", "prompt_template"):
        assert f'"{field}"' in sampling, field
    # The cap that is recorded is the one the call actually uses.
    assert mff.REFERENCE_MAX_TOKENS == 2000


def test_fiscal_rates_separate_exact_from_exact_when_answered(fiscal_run: dict) -> None:
    """A named recomputation over the committed artifact's untouched `cases[]`.

    `measure_finetune_fiscal._rates` applied to arm D's 300 rows: 257 exact over 300
    (85.7%, the published figure) and 257 exact over the 261 that produced a label
    (98.5%). The truncation is the whole of the difference.
    """
    mff = _load("measure_finetune_fiscal")
    arms = {a["arm"]: a for a in fiscal_run["arms"]}
    d = mff._rates(arms["D"]["cases"])

    assert d["n"] == 300
    assert d["counts"]["exact"] == 257
    assert round(d["rates_pct"]["exact"], 1) == 85.7
    assert d["answered_n"] == 261
    assert d["unanswered_n"] == 39
    assert d["exact_when_answered"] == 257
    assert round(d["exact_when_answered_pct"], 1) == 98.5
    assert d["error_n"] == 0

    # The compiled arms answered every case, so the two rates coincide there -- which is
    # what makes the gap specific to the token-capped arm rather than a scoring change.
    for arm in ("A", "B", "C"):
        cut = mff._rates(arms[arm]["cases"])
        assert cut["answered_n"] == 300, arm
        assert cut["unanswered_n"] == 0, arm
        assert cut["exact_when_answered_pct"] == cut["rates_pct"]["exact"], arm


def test_fiscal_truncation_count_comes_from_the_api_stop_reason() -> None:
    """`truncated_n` must be an observation, not "no label was found".

    An arm can also fail to produce a label by answering in prose, so inferring truncation
    from an absent label would over-count. `_row` records `stop_reason` and `truncated`.
    """
    mff = _load("measure_finetune_fiscal")
    case = {"id": "x", "date": "2026-03-03", "format": "iso", "input": "2026-03-03",
            "expected": "FY2026-W05", "boundary": False, "fiscal_year": 2026, "week": 5}

    truncated = mff._row(case, "Let me work through this step by", None, 1.0,
                         stop_reason="max_tokens")
    assert truncated["truncated"] is True
    assert truncated["label_found"] is False

    chatty = mff._row(case, "I think the answer is probably around week five.", None, 1.0,
                      stop_reason="end_turn")
    assert chatty["truncated"] is False
    assert chatty["label_found"] is False

    clean = mff._row(case, "FY2026-W05", None, 1.0, stop_reason="end_turn")
    assert clean["truncated"] is False
    assert clean["exact"] is True

    # Adapter arms pass no stop reason at all and must not be counted as truncated.
    adapter = mff._row(case, "FY2026-W05", None, 1.0)
    assert adapter["stop_reason"] is None
    assert adapter["truncated"] is False

    rates = mff._rates([truncated, chatty, clean, adapter])
    assert rates["truncated_n"] == 1
    assert rates["unanswered_n"] == 2
    assert rates["exact_when_answered"] == 2


# ==============================================================================  B-6
#
# (a) `measure_semantic_correctness.py` folded `suite.standard_cases` and then scored the
# same cases. (b) The adapter was named `{task}-{compiler}.paw`, so the 0-example and
# 8-example runs overwrote each other -- which is how the constrained-upstream section came
# to be measured against an 8-example adapter.

_SPEC_DRAFTS = _MEASUREMENTS / "spec-drafts"


def _suite_cases(name: str) -> list:
    import yaml
    return yaml.safe_load((_SPEC_DRAFTS / name).read_text())["standard_cases"]


def test_semantic_adapter_path_carries_max_spec_examples() -> None:
    """Report §5's named test for B-6(b): two `max_spec_examples` must not collide.

    `measurements/phone_extractor-paw-4b-qwen3-0.6b.paw` has
    `examples_folded_into_spec: 8` because the 8-example run overwrote the 0-example one at
    the same path, and the (since-deleted) upstream-injection measurement script then
    read that path.
    """
    msc = _load("measure_semantic_correctness")
    zero = msc.adapter_filename("phone_extractor", "paw-4b-qwen3-0.6b", 0)
    eight = msc.adapter_filename("phone_extractor", "paw-4b-qwen3-0.6b", 8)
    assert zero != eight
    assert "0" in zero and "8" in eight
    assert zero.endswith(".paw") and eight.endswith(".paw")
    # The colliding name must be gone from the script entirely.
    src = (_SCRIPTS / "measure_semantic_correctness.py").read_text()
    assert '''f"{suite_dict['task_name']}-{args.compiler}.paw"''' not in src
    assert "adapter_filename(" in src
    # And the old colliding path is the one the committed artifact was produced at, which
    # is why it cannot simply be reused.
    assert msc.adapter_filename("phone_extractor", "paw-4b-qwen3-0.6b", 8) != \
        "phone_extractor-paw-4b-qwen3-0.6b.paw"


def test_semantic_folded_inputs_do_not_appear_in_scored_results() -> None:
    """Report §5's named test for B-6(a), over the real committed suites.

    Whatever is folded into the spec is removed from the suite before the runner sees it,
    so it cannot be in the scored denominator.
    """
    msc = _load("measure_semantic_correctness")
    for suite_file in ("spec-1-json-repair.yaml", "spec-2-phone-extractor.yaml",
                       "spec-3-review-sentiment.yaml"):
        cases = _suite_cases(suite_file)
        for k in range(1, len(cases) + 1):
            folded, scored = msc.split_fold_and_eval(cases, k)
            folded_inputs = {c["input"] for c in folded}
            scored_inputs = {c["input"] for c in scored}
            assert folded_inputs & scored_inputs == set(), (suite_file, k)
            # A partition: nothing invented, nothing lost.
            assert len(folded) + len(scored) == len(cases), (suite_file, k)
            assert len(folded) == min(k, sum(1 for c in cases if c.get("expected")))
            # Drawn from the tail, so the ids recorded in the artifact are exactly what the
            # backend folds (it takes `examples[:limit]`).
            assert folded == [c for c in cases if c in folded]
            assert msc.fold_case_ids(cases, folded)[0].startswith("standard_cases[")


def test_semantic_zero_examples_folds_nothing_and_scores_everything() -> None:
    """The committed terse-spec runs used `--max-spec-examples 0`; they must not move."""
    msc = _load("measure_semantic_correctness")
    cases = _suite_cases("spec-2-phone-extractor.yaml")
    for k in (0, -1):
        folded, scored = msc.split_fold_and_eval(cases, k)
        assert folded == []
        assert scored == cases


def test_semantic_fold_split_skips_cases_with_no_expected_output() -> None:
    """A case with no `expected` cannot be a few-shot example, so it is never folded."""
    msc = _load("measure_semantic_correctness")
    cases = [
        {"input": "a", "expected": "A"},
        {"input": "b"},
        {"input": "c", "expected": "C"},
    ]
    folded, scored = msc.split_fold_and_eval(cases, 2)
    assert [c["input"] for c in folded] == ["a", "c"]
    assert [c["input"] for c in scored] == ["b"]
    assert msc.fold_case_ids(cases, folded) == ["standard_cases[0]", "standard_cases[2]"]


def test_semantic_summary_records_the_adapter_identity_and_the_held_out_ids() -> None:
    """B-6's third clause: `program_id` and `examples_folded_into_spec` in every artifact.

    The committed artifacts record neither, which is why the constrained-upstream section
    could be attributed to the wrong adapter for two days without anything noticing.
    """
    src = (_SCRIPTS / "measure_semantic_correctness.py").read_text()
    summary_block = src[src.index('summary = {\n        "label": args.label,'):]
    for key in ("program_id", "examples_folded_into_spec", "folded_case_ids",
                "folded_inputs", "standard_cases_total", "standard_cases_scored",
                "scored_rows_folded_into_spec", "adapter_path"):
        assert f'"{key}"' in summary_block, key
    # The runtime guard that makes `scored_rows_folded_into_spec: 0` an assertion and not
    # just a hopeful field.
    assert "folded case(s) appear in the scored results" in src
    assert "which is the B-6(a) leak" in src
    committed = json.loads(
        (_MEASUREMENTS / "semantic-phone_extractor-3080-fewshot8-20260909-005717.json").read_text()
    )
    assert "program_id" not in committed
    assert "folded_case_ids" not in committed


# ================================================================================  B-7
#
# `measure_shadow_mode.py`'s audit arm broke on the call that completed the 20th audit
# sample, so `teacher_calls / served` was `20 / N` for a negative-binomial `N` -- published
# under the name `teacher_calls_per_served_call`, which reads as a measured rate.

SHADOW_ARTIFACT = _MEASUREMENTS / "shadow-mode-3080-20260910-124735.json"


@pytest.fixture(scope="module")
def shadow_run() -> dict:
    return json.loads(SHADOW_ARTIFACT.read_text())


def test_shadow_stopping_time_is_not_reported_as_a_rate(shadow_run: dict) -> None:
    """The recorded arm's numbers, through the new block: 20/621 is a stopping time.

    The committed artifact reports `teacher_calls_per_served_call: 0.0322` for an arm that
    stopped as soon as 20 audit samples existed, against a configured `audit_rate` of 0.05.
    """
    msm = _load("measure_shadow_mode")
    recorded = shadow_run["experiment_2_audit_rate_005"]
    assert recorded["served_calls_after_promotion"] == 621
    assert recorded["teacher_calls_after_promotion"] == 20
    assert round(recorded["teacher_calls_per_served_call"], 4) == 0.0322

    block = msm.audit_cost_block(
        teacher_calls=20, served=621, audit_rate=0.05, serve_calls_requested=1200,
        stopped_at_first_audit_window=True, calls_to_first_completed_window=621,
    )
    assert block["teacher_calls_per_served_call"] is None
    assert round(block["stopping_time_ratio"], 4) == 0.0322
    assert block["calls_to_first_completed_window"] == 621
    assert block["audit_rate_configured"] == 0.05
    assert "stopping time" in block["teacher_calls_per_served_call_note"]

    # The same arithmetic on the 0.5-rate arm gave 0.69, which is the other half of the
    # evidence that the quantity is not a rate.
    demote = shadow_run["experiment_2c_demotion"]
    assert round(demote["teacher_calls_per_served_call"], 4) == 0.6897
    assert msm.audit_cost_block(
        teacher_calls=20, served=29, audit_rate=0.5, serve_calls_requested=200,
        stopped_at_first_audit_window=True, calls_to_first_completed_window=29,
    )["teacher_calls_per_served_call"] is None


def test_shadow_fixed_n_arm_does_report_a_rate() -> None:
    """With the served-call count fixed in advance the ratio *is* an estimate of the rate."""
    msm = _load("measure_shadow_mode")
    block = msm.audit_cost_block(
        teacher_calls=61, served=1200, audit_rate=0.05, serve_calls_requested=1200,
        stopped_at_first_audit_window=False, calls_to_first_completed_window=400,
    )
    assert block["stopping_time_ratio"] is None
    assert round(block["teacher_calls_per_served_call"], 5) == round(61 / 1200, 5)
    assert "fixed before the run" in block["teacher_calls_per_served_call_note"]
    assert block["calls_to_first_completed_window"] == 400
    # Zero served calls must not divide by zero in either mode.
    for stopped in (True, False):
        zero = msm.audit_cost_block(
            teacher_calls=0, served=0, audit_rate=0.05, serve_calls_requested=0,
            stopped_at_first_audit_window=stopped, calls_to_first_completed_window=None,
        )
        assert zero["teacher_calls_per_served_call"] is None
        assert zero["stopping_time_ratio"] is None


def test_shadow_has_a_fixed_served_calls_arm() -> None:
    """Report §5's suggestion for B-7: run a fixed number of served calls and report from it."""
    src = (_SCRIPTS / "measure_shadow_mode.py").read_text()
    assert "experiment_2d_audit_rate_005_fixed_n" in src
    assert "stop_at_first_audit_window=False" in src
    assert "--fixed-serve-calls" in src
    # The loop only breaks when the arm asked to stop early.
    assert "if stop_at_first_audit_window:\n                    break" in src


def test_shadow_binomial_p_is_parameterised_and_exact(shadow_run: dict) -> None:
    """`p` was the literal 0.6 at the call site -- B-1's leaked all-rows rate.

    The arithmetic itself is exact and was independently re-verified; what was wrong was
    the `p` it was evaluated at. Both values are asserted so the A3 re-basing has a
    committed reference: at the held-out 0.467 the gate's conclusion gets *stronger*.
    """
    msm = _load("measure_shadow_mode")
    assert shadow_run["binomial_residual"]["p_agreement"] == 0.6
    assert shadow_run["recorded_full_agreement_rate"] == 60.0

    leaked = msm.binomial_residual(0.6, 20, 0.8, 5)
    assert leaked["successes_needed"] == 16
    assert round(leaked["p_one_window_clears"] * 100, 1) == 5.1
    assert round(leaked["p_promotes_within_stall"] * 100, 1) == 23.0

    corrected = msm.binomial_residual(0.467, 20, 0.8, 5)
    assert corrected["p_agreement"] == 0.467
    assert corrected["successes_needed"] == 16
    assert round(corrected["p_one_window_clears"] * 100, 2) == 0.25
    assert round(corrected["p_promotes_within_stall"] * 100, 2) == 1.23
    assert corrected["p_promotes_within_stall"] < leaked["p_promotes_within_stall"]

    src = (_SCRIPTS / "measure_shadow_mode.py").read_text()
    assert "binomial_residual(0.6, 20, 0.8, 5)" not in src
    assert "binomial_residual(binomial_p, 20, 0.8, 5)" in src
    assert "--binomial-p" in src
    assert '"binomial_p_source"' in src
    assert "full_agreement_rate_heldout" in src


# ===============================================================================  B-8a
#
# The lookup spec's worked example one renders byte-identical to evaluation case `r168`
# ("Ship this order to Porto, Portugal.") with its answer stated, so that case is in every
# arm's prompt. `build_fixture`'s overlap assertion checked folding-vs-eval only.

LOOKUP_ARTIFACT = _MEASUREMENTS / "finetune-lookup-3080-20260910-192159.json"
LOOKUP_FIXTURE = _MEASUREMENTS / "finetune-lookup-regions.json"


@pytest.fixture(scope="module")
def lookup_run() -> dict:
    return json.loads(LOOKUP_ARTIFACT.read_text())


def test_lookup_spec_eval_overlap_is_exactly_r168() -> None:
    """The reported-overlap assertion: `spec_eval_overlap_ids == ["r168"]`.

    Deliberately a *reported* overlap and not an absence assertion. The overlap exists
    now, and the only ways to clear it are changing `SPEC` -- which changes the upstream
    compile-cache key and invalidates the committed 33.0/29.0/97.7/100% table, costing
    three paid recompiles -- or changing the evaluation template. So it is computed,
    recorded, and pinned against growth.
    """
    mfl = _load("measure_finetune_lookup")
    committed = json.loads(LOOKUP_FIXTURE.read_text())
    assert mfl.KNOWN_SPEC_EVAL_OVERLAP == ["r168"]
    assert mfl.spec_eval_overlap(mfl.SPEC, committed["evaluation"]) == ["r168"]

    leaked = [c for c in committed["evaluation"] if c["id"] == "r168"][0]
    assert leaked["input"] == "Ship this order to Porto, Portugal."
    assert leaked["input"] in mfl.SPEC
    assert leaked["expected"] in mfl.SPEC

    # Every other evaluation case is absent from the spec, so the known overlap is the
    # whole of it rather than the first one someone happened to notice.
    assert sum(1 for c in committed["evaluation"] if c["input"] in mfl.SPEC) == 1


def test_lookup_spec_eval_overlap_growth_is_rejected() -> None:
    """The assertion must be falsifiable: a second spec-answered case must stop the run."""
    mfl = _load("measure_finetune_lookup")
    committed = json.loads(LOOKUP_FIXTURE.read_text())
    evaluation = list(committed["evaluation"])
    # Worked example two's sentence, as an extra evaluation case.
    smuggled = "Our Osaka office handled the call, but the customer is in Kenya."
    assert smuggled in mfl.SPEC
    evaluation.append({"id": "r999", "input": smuggled, "expected": "RG-K7"})
    assert mfl.spec_eval_overlap(mfl.SPEC, evaluation) == ["r168", "r999"]
    assert mfl.spec_eval_overlap(mfl.SPEC, evaluation) != mfl.KNOWN_SPEC_EVAL_OVERLAP

    src = (_SCRIPTS / "measure_finetune_lookup.py").read_text()
    assert "spec/eval overlap changed" in src
    # And the existing folding-vs-eval RuntimeError is deliberately left alone -- extending
    # it would raise on every invocation, since the overlap exists now.
    assert "folding pool overlaps the evaluation set" in src


def test_lookup_reports_exact_and_exact_excluding_the_spec_leak(lookup_run: dict) -> None:
    """A named recomputation over the untouched `cases[]` of the committed lookup run.

    `measure_finetune_lookup.score_arm(rows, folding_countries, ["r168"])` on arm A:
    99/300 = 33.0% overall, 98/299 = 32.7759% = **32.8%** to one decimal place with the
    spec-answered case removed (32.7 is the truncation, not the rounding -- corrected in
    the track file and the bug-hunt report during this track). Every arm answered `r168`
    correctly, so every arm's exact count drops by exactly one.
    """
    mfl = _load("measure_finetune_lookup")
    arms = {a["arm"]: a for a in lookup_run["arms"]}
    expected_overall = {"A": 99, "B": 87, "C": 293, "D": 300}

    for arm, exact in expected_overall.items():
        scores = mfl.score_arm(arms[arm]["cases"], lookup_run["folding_countries"], ["r168"])
        assert scores["overall"]["n"] == 300, arm
        assert scores["overall"]["counts"]["exact"] == exact, arm
        assert scores["excluding_spec_leak"]["n"] == 299, arm
        assert scores["excluding_spec_leak"]["counts"]["exact"] == exact - 1, arm
        assert scores["spec_eval_overlap_ids"] == ["r168"], arm
        leaked_row = [r for r in arms[arm]["cases"] if r["id"] == "r168"][0]
        assert leaked_row["exact"] is True, arm

    a = mfl.score_arm(arms["A"]["cases"], lookup_run["folding_countries"], ["r168"])
    assert round(a["overall"]["rates_pct"]["exact"], 1) == 33.0
    assert round(a["excluding_spec_leak"]["rates_pct"]["exact"], 4) == 32.7759
    assert round(a["excluding_spec_leak"]["rates_pct"]["exact"], 1) == 32.8

    # With no overlap passed, the two cuts coincide -- so the field cannot quietly become a
    # different denominator for a task that has no spec leak.
    none = mfl.score_arm(arms["A"]["cases"], lookup_run["folding_countries"])
    assert none["excluding_spec_leak"]["n"] == 300
    assert none["spec_eval_overlap_ids"] == []


def test_lookup_summary_and_fixture_record_the_overlap() -> None:
    """Recorded in both places, and recomputed in `main` rather than trusted.

    The committed fixture predates the field, so a `--skip-compile` re-run against it must
    still get a 299 denominator.
    """
    src = (_SCRIPTS / "measure_finetune_lookup.py").read_text()
    fixture_block = src[src.index('    fixture = {'):src.index("    out_path.write_text(json.dumps(fixture")]
    assert '"spec_eval_overlap_ids": spec_overlap' in fixture_block
    assert '"known_spec_eval_overlap"' in fixture_block
    summary_block = src[src.index('    summary = {\n        "label": args.label,'):]
    assert '"spec_eval_overlap_ids": spec_overlap_ids,' in summary_block
    assert "spec_overlap_ids = spec_eval_overlap(SPEC, evaluation)" in src
    committed = json.loads(LOOKUP_FIXTURE.read_text())
    assert "spec_eval_overlap_ids" not in committed


# ===============================================================================  B-8d
#
# `measure_real_backend.py`'s `_pct` indexed `min(n - 1, round(p * (n - 1)))`
# unconditionally, so at n=10 and p=0.99 it returned the maximum. Both published 3080
# `p99_ms` values are a single observation.


def test_real_backend_p99_over_ten_calls_is_not_the_maximum() -> None:
    """The exact arrangement of the two published 3080 rows: ten warm calls, p99.

    The old arithmetic returned index `min(9, round(0.99 * 9)) == 9`, the maximum.
    """
    mrb = _load("measure_real_backend")
    ten = [float(i) for i in range(10)]
    assert mrb._pct(ten, 0.99) is None
    # The old index arithmetic, shown explicitly so the regression is unmistakable.
    assert min(len(ten) - 1, round(0.99 * (len(ten) - 1))) == 9
    assert ten[9] == max(ten)

    # p50 and p90 are estimable at n=10 and must keep working: the "numbers checked and
    # found sound" list includes the 3080 and A100 p50 latencies.
    assert mrb._pct(ten, 0.50) == 4.0
    assert mrb._pct(ten, 0.90) == 8.0
    assert mrb._pct(ten, 0.90) != max(ten)

    # A hundred calls can support a p99.
    hundred = [float(i) for i in range(100)]
    assert mrb._pct(hundred, 0.99) == 98.0


def test_real_backend_minimum_n_per_percentile() -> None:
    """`n >= 1/(1-p)`: 2 for p50, 10 for p90, 100 for p99.

    The float-rounding guard matters here -- `1 / (1 - 0.9)` is 10.000000000000002, and
    ceiling that would rule out the p90 over 10 calls this script legitimately reports.
    """
    mrb = _load("measure_real_backend")
    assert mrb._min_n_for_percentile(0.50) == 2
    assert mrb._min_n_for_percentile(0.90) == 10
    assert mrb._min_n_for_percentile(0.95) == 20
    assert mrb._min_n_for_percentile(0.99) == 100
    assert mrb._min_n_for_percentile(0.999) == 1000
    for bad in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            mrb._min_n_for_percentile(bad)
    assert mrb._pct([], 0.50) is None
    assert mrb._pct([1.0], 0.50) is None


def test_real_backend_records_the_calls_it_ran_and_the_maximum() -> None:
    """B-8d's second half: `README.md` quotes `--calls 50` for rows that ran at 10.

    The artifact now carries `calls_requested` next to `warm_calls`, `max_ms` always, and
    `percentiles_unavailable` naming the n each missing percentile would need.
    """
    src = (_SCRIPTS / "measure_real_backend.py").read_text()
    lat_block = src[src.index("    lat = {"):src.index('    print(f"[infer] warm p50=')]
    for key in ("calls_requested", "warm_calls", "min_ms", "max_ms",
                "percentiles_unavailable", "percentile_note"):
        assert f'"{key}"' in lat_block, key
    assert '"calls": args.calls,' in src

    # The two published 3080 rows ran 10 warm calls and their p99 is the largest value
    # recorded for them -- which is what makes it one observation.
    for name in ("3080-cuda-paw-4b-qwen3-0.6b-20260908-155314.json",
                 "3080-paw-4b-qwen3-0.6b-20260908-152233.json"):
        lat = json.loads((_MEASUREMENTS / name).read_text())["latency"]
        assert lat["warm_calls"] == 10
        assert lat["p99_ms"] > lat["p90_ms"] > lat["p50_ms"]
        assert "calls_requested" not in lat
    # And the A100 row ran 50, which is still under the 100 a p99 needs: at n=50 the old
    # arithmetic indexed round(0.99 * 49) == 49, the maximum again. Not in the report's
    # wording, which named only the 3080 rows.
    a100 = json.loads(
        (_MEASUREMENTS / "a100-paw-4b-qwen3-0.6b-20260908-160856.json").read_text()
    )["latency"]
    assert a100["warm_calls"] == 50
    assert round(0.99 * (a100["warm_calls"] - 1)) == a100["warm_calls"] - 1
    mrb = _load("measure_real_backend")
    assert a100["warm_calls"] < mrb._min_n_for_percentile(0.99)


# ================================================================================  B-9
#
# SUSPECTED in the report; CONFIRMED here by inspection. `measure_fail_open.py`'s teacher
# returned one constant marker for every input, so the adapter's only training signal was
# three identical pairs mapping arbitrary inputs to that string -- and phases 3 and 4 then
# asserted `out == TEACHER_MARKER`, which a degenerate adapter emitting the memorised marker
# satisfies with no fallback having occurred. The script is NOT executed by this track: it
# needs a real PAW_API_KEY and performs a real paid compile.


class _FakeDecorated:
    """Stands in for a `compile_on_hit`-wrapped function: callable, with a fail-open count."""

    def __init__(self, behaviour, fail_open_increments: bool) -> None:
        self._behaviour = behaviour
        self._increments = fail_open_increments
        self._fail_open = 0

    def get_fail_open_count(self) -> int:
        return self._fail_open

    def __call__(self, text: str) -> str:
        if self._increments:
            self._fail_open += 1
        return self._behaviour(text)


def test_fail_open_teacher_output_is_unique_per_call() -> None:
    """The training signal must not be reproducible by memorisation."""
    mfo = _load("measure_fail_open")
    teacher = mfo.UniqueTeacher()
    outputs = [teacher(f"input {i}") for i in range(5)]
    assert len(set(outputs)) == 5
    assert teacher.calls == 5
    assert teacher.last == outputs[-1]
    assert teacher.history == outputs
    assert all(o.startswith(mfo.TEACHER_MARKER) for o in outputs)
    # And the old constant marker is never what the teacher returns.
    assert mfo.TEACHER_MARKER not in outputs


def test_fail_open_check_fails_for_a_backend_that_echoes_the_marker() -> None:
    """Report §5's named test for B-9: the check must be falsifiable.

    An adapter that memorised the marker and echoes it, with no fallback having happened,
    must make the fail-open check FAIL. Under the old `out == TEACHER_MARKER` assertion it
    passed.
    """
    mfo = _load("measure_fail_open")
    teacher = mfo.UniqueTeacher()
    # Three traced calls, all mapping to the constant-looking marker family -- the training
    # set the degenerate adapter would memorise from.
    for i in range(3):
        teacher(f"date input {i}")

    memoriser = _FakeDecorated(lambda _t: mfo.TEACHER_MARKER, fail_open_increments=False)
    verdict = mfo.probe_fallback(memoriser, teacher, "January 1, 2026")
    assert verdict["is_fallback"] is False
    assert verdict["returned_this_calls_teacher_output"] is False
    assert verdict["teacher_called_exactly_once"] is False
    assert verdict["fallback_counted"] is False
    # The old assertion, for contrast: this is what used to be checked, and it passes.
    assert verdict["output"] == mfo.TEACHER_MARKER

    # An adapter that echoes the most recent teacher output verbatim -- a harder fake, since
    # it defeats the text comparison -- is still caught by the two instrumented signals.
    echoer = _FakeDecorated(lambda _t: teacher.last, fail_open_increments=False)
    echo_verdict = mfo.probe_fallback(echoer, teacher, "January 1, 2026")
    assert echo_verdict["returned_this_calls_teacher_output"] is True
    assert echo_verdict["teacher_called_exactly_once"] is False
    assert echo_verdict["fallback_counted"] is False
    assert echo_verdict["is_fallback"] is False


def test_fail_open_check_passes_for_a_real_fallback() -> None:
    """The converse: a genuine fallback must still be recognised, or the check is useless."""
    mfo = _load("measure_fail_open")
    teacher = mfo.UniqueTeacher()
    for i in range(3):
        teacher(f"date input {i}")

    falling_open = _FakeDecorated(teacher, fail_open_increments=True)
    verdict = mfo.probe_fallback(falling_open, teacher, "January 1, 2026")
    assert verdict["is_fallback"] is True
    assert verdict["returned_this_calls_teacher_output"] is True
    assert verdict["teacher_called_exactly_once"] is True
    assert verdict["fallback_counted"] is True
    assert verdict["fail_open_after"] == verdict["fail_open_before"] + 1


def test_fail_open_served_locally_verdict_catches_memorised_output() -> None:
    """Phase 2's check: `output != teacher_output` was not enough on its own.

    `measurements/README.md`'s fail-open section records an identically-shaped compile
    degenerating into constant output. A local call echoing an *earlier* teacher output is
    that degeneration, and is reported separately from "the teacher was not called".
    """
    mfo = _load("measure_fail_open")
    teacher = mfo.UniqueTeacher()
    for i in range(3):
        teacher(f"date input {i}")

    healthy = mfo.served_locally_verdict("2026-01-01", 3, 3, teacher.history)
    assert healthy["served_locally"] is True
    assert healthy["output_is_not_any_teacher_output"] is True

    memorised = mfo.served_locally_verdict(teacher.history[0], 3, 3, teacher.history)
    assert memorised["served_locally"] is True
    assert memorised["output_is_not_any_teacher_output"] is False


def test_fail_open_writes_an_artifact_naming_every_check() -> None:
    """The script used to write nothing, so there was no record of which checks passed."""
    mfo = _load("measure_fail_open")
    log = mfo.CheckLog()
    assert log.any_failed is False
    log.record("a passing check", True, detail=1)
    assert log.any_failed is False
    log.record("a failing check", False, detail=2)
    assert log.any_failed is True
    assert [c["check"] for c in log.checks] == ["a passing check", "a failing check"]
    assert [c["passed"] for c in log.checks] == [True, False]
    assert log.checks[0]["evidence"] == {"detail": 1}

    src = (_SCRIPTS / "measure_fail_open.py").read_text()
    assert "out_path.write_text(json.dumps(artifact" in src
    assert '"checks": log.checks' in src
    assert '"checks_passed"' in src and '"all_passed"' in src
    assert 'f"fail-open-{args.label}-' in src
    # The old output-text-only assertions are gone.
    assert "result == TEACHER_MARKER" not in src
    assert "out != TEACHER_MARKER" not in src
    assert "live != TEACHER_MARKER" not in src


# ================================================================================  A-6
#
# `examples_folded_into_spec` was `min(len(examples), max_spec_examples)` while the spec
# renderer filtered to usable dicts first, so any malformed example inflated the count --
# and the count is mirrored into every published measurement artifact, which is where a
# wrong number stops being a library bug and becomes a published claim. Only one of the
# four scripts that mirror the count also mirrored `folded_example_ids`, so three of them
# reported *how many* examples were folded with no way to check *which*.


@pytest.mark.parametrize(
    "script",
    [
        "measure_finetune_triage",
        "measure_semantic_correctness",
        "measure_finetune_lookup",
        "measure_triage_semantic_agreement",
    ],
)
def test_summary_mirrors_folded_example_ids_beside_the_count_A_6(script: str) -> None:
    """Every script that publishes the folded *count* also publishes the folded *ids*.

    Additive -- it changes no published number. The ids are SHA-256 digests of
    input+output, so this names what was folded without republishing any traced text,
    which is the whole reason ids are the right thing to mirror here.
    """
    src = (_SCRIPTS / f"{script}.py").read_text()
    assert '"examples_folded_into_spec"' in src, f"{script} does not report the count at all"
    assert '"folded_example_ids"' in src, (
        f"{script} publishes `examples_folded_into_spec` but not `folded_example_ids`, so "
        "a reader of its artifact cannot check which examples the count refers to"
    )


# ==============================================================  constrained decoding
#
# `measure_constrained_decoding.py` is the one committed script for the shipped
# grammar-constrained decoding path. Its hardware body needs two gitignored `.paw`
# adapters and a GPU, so what is pinned here is the same thing as everywhere else in
# this file: the arithmetic that turns a recorded run into a published number. Three of
# these functions exist specifically because the near-miss version of the number was
# published in this project's own planning documents before Phase 0 caught it -- the
# full-exact/urgency-within-1 confusion (round 4, K-7), a masking statistic that reads
# as zero when the processor was never called at all (K-10), and a UTF-8 check that
# cannot fail (K-1).


@pytest.fixture(scope="module")
def cd_script() -> ModuleType:
    return _load("measure_constrained_decoding")


def test_constrained_artifact_name_follows_the_measurements_convention(cd_script) -> None:
    import datetime as _dt

    name = cd_script.artifact_name("3080", _dt.datetime(2026, 9, 15, 18, 50, 48))
    assert name == "constrained-decoding-3080-20260915-185048.json"


def test_pair_report_lists_every_discordant_pair_not_just_a_count(cd_script) -> None:
    """A null result here is the result, so the discordant list must be exhaustive:
    a count alone cannot be checked against anything."""
    report = cd_script.pair_report(
        ["a", "b", "c"], ["x", "y", "z"], ["x", "Y", "z"]
    )
    assert report["pairs"] == 3
    assert report["byte_identical_pairs"] == 2
    assert report["discordant_pairs"] == 1
    assert report["discordant"] == [
        {"input": "b", "constrained": "y", "unconstrained": "Y"}
    ]


def test_pair_report_refuses_unequal_arms(cd_script) -> None:
    with pytest.raises(ValueError, match="equal-length"):
        cd_script.pair_report(["a"], ["x", "y"], ["x", "y"])


def test_full_exact_agreement_is_full_exact_and_not_urgency_within_one(cd_script) -> None:
    """K-7: the neighbouring statistic ("priority and department exact, urgency within
    1") is a *larger* number, and the two were confused once already. A case that is off
    by one on `urgency_score` must not count here."""
    fields = ["priority", "department", "urgency_score"]
    labels = [
        {"priority": "high", "department": "billing", "urgency_score": 4},
        {"priority": "high", "department": "billing", "urgency_score": 4},
        {"priority": "low", "department": "sales", "urgency_score": 2},
    ]
    outputs = [
        '{"priority": "high", "department": "billing", "urgency_score": 4}',   # exact
        '{"priority": "high", "department": "billing", "urgency_score": 3}',   # within 1
        '{"priority": "high", "department": "sales", "urgency_score": 2}',     # priority wrong
    ]
    agreement = cd_script.full_exact_agreement(outputs, labels, fields)
    assert agreement["statistic"] == "full_exact"
    assert agreement["matched"] == 1
    assert agreement["n"] == 3
    assert agreement["unparsed"] == 0


def test_full_exact_agreement_counts_unparsed_separately_from_wrong(cd_script) -> None:
    agreement = cd_script.full_exact_agreement(
        ["not json at all"], [{"priority": "low"}], ["priority"]
    )
    assert agreement["matched"] == 0
    assert agreement["unparsed"] == 1


def test_full_exact_agreement_refuses_a_label_count_mismatch(cd_script) -> None:
    with pytest.raises(ValueError, match="one label per output"):
        cd_script.full_exact_agreement(["{}", "{}"], [{"priority": "low"}], ["priority"])


def test_masked_summary_reports_no_invocations_as_vacuous_not_as_zero(cd_script) -> None:
    """K-10: the failure this signal exists to catch -- `Llama.sample()` reusing an
    installed sampler and never invoking the processor -- leaves the per-step list
    *empty*. A "mean masked" field computed over that would be a statistic about
    nothing, so an empty list gets no statistics at all."""
    vacuous = cd_script.masked_summary([], 151936)
    assert vacuous["vacuous"] is True
    assert vacuous["steps"] == 0
    assert "min" not in vacuous and "mean" not in vacuous

    real = cd_script.masked_summary([151570, 151936, 151800], 151936)
    assert real["vacuous"] is False
    assert real["min"] == 151570
    assert real["steps_masking_zero"] == 0

    with_a_zero = cd_script.masked_summary([0, 151936], 151936)
    assert with_a_zero["steps_masking_zero"] == 1


def test_appliedness_relates_invocations_to_emitted_tokens_without_claiming_equality(
    cd_script,
) -> None:
    """The SDK samples EOS under the mask and then breaks without emitting it, so an
    EOS-terminated call has exactly one more invocation than emitted token. Asserting
    plain equality would fail on every normal call; asserting nothing would miss the
    case the signal exists for."""
    eos = cd_script.appliedness_check(21, 20)
    assert eos["ok"] is True and eos["terminated_by"] == "eos"

    budget = cd_script.appliedness_check(128, 128)
    assert budget["ok"] is True and budget["terminated_by"] == "token_budget_or_context"

    never_ran = cd_script.appliedness_check(0, 20)
    assert never_ran["ok"] is False
    assert never_ran["relation"] == "unexpected"


def test_values_inside_sets_separates_not_json_from_out_of_set(cd_script) -> None:
    """Three outcomes, not two: `None` is "not JSON at all", `False` is "JSON whose
    value is outside the closed set". Collapsing them would hide which failure
    happened."""
    sets = cd_script.literal_sets(cd_script.Triage)
    assert sets["urgency_score"] == [1, 2, 3, 4, 5]

    good = '{"priority": "low", "department": "billing", "urgency_score": 1}'
    assert cd_script.values_inside_sets(good, sets) is True
    bad = '{"priority": "URGENT", "department": "billing", "urgency_score": 1}'
    assert cd_script.values_inside_sets(bad, sets) is False
    missing = '{"priority": "low", "department": "billing"}'
    assert cd_script.values_inside_sets(missing, sets) is False
    assert cd_script.values_inside_sets("banana", sets) is None


def test_replacement_char_scan_names_the_arm_and_case_of_every_offender(cd_script) -> None:
    scan = cd_script.replacement_char_scan(
        [("15 TICKETS/constrained/0", "fine"), ("15 TICKETS/constrained/1", "b��d")]
    )
    assert scan["strings_scanned"] == 2
    assert scan["strings_containing_fffd"] == 1
    assert scan["offenders"][0]["name"] == "15 TICKETS/constrained/1"
    assert scan["offenders"][0]["count"] == 2


def test_the_artifact_says_the_fffd_scan_cannot_prove_utf8_wellformedness(cd_script) -> None:
    """K-1: the SDK decodes with `errors="replace"`, so an end-to-end UTF-8 round-trip
    on the returned `str` cannot fail. The artifact must say so rather than publish a
    number that cannot be wrong."""
    caveat = cd_script._BYTE_SCAN_CAVEAT
    assert "CANNOT FAIL" in caveat
    assert "errors='replace'" in caveat
    assert "byte-exact end-to-end check" in caveat


def test_the_artifact_says_the_agreement_figure_is_a_property_of_the_adapter(cd_script) -> None:
    caveat = cd_script._AGREEMENT_CAVEAT
    assert "property of the adapter" in caveat
    assert "byte-identical" in caveat
    assert "recorded ONCE" in caveat


def test_cost_summary_separates_processor_time_from_what_a_caller_pays(cd_script) -> None:
    """Two different numbers: time spent inside the processor, and the end-to-end
    overhead per generated token, which also carries the per-call matcher build and is
    what `roadmap.md`'s budget is about."""
    cost = cd_script.cost_summary(
        processor_ms=[0.6, 0.7, 0.8],
        constrained_call_ms=[132.0, 134.0],
        unconstrained_call_ms=[115.0, 117.0],
        generated_tokens_per_call=[20, 20],
    )
    assert cost["end_to_end"]["overhead_ms_per_generated_token"] == 0.85
    assert cost["budget"]["budget_ms_per_token"] == 2.0
    assert cost["budget"]["within_budget"] is True

    blown = cd_script.cost_summary([0.6], [377.1], [114.8], [20])
    assert blown["budget"]["within_budget"] is False
    assert blown["budget"]["measured_ms_per_token"] == 13.115


def test_cost_summary_refuses_an_empty_arm(cd_script) -> None:
    with pytest.raises(ValueError, match="non-empty sample"):
        cd_script.cost_summary([], [1.0], [1.0], [1])


def test_the_script_drives_the_shipped_load_path_not_a_private_hook(cd_script) -> None:
    """The two scripts this one replaces drove a deleted class, one of them through the
    SDK's private `_llm` attribute. This one must call the public backend entry point
    with the same regex `paw_kit.schema.load` builds."""
    src = (_SCRIPTS / "measure_constrained_decoding.py").read_text()
    assert "backend.infer(adapter, text, grammar_constraint=pattern)" in src
    assert "pydantic_to_regex(Triage, anchors=False)" in src
    # The deleted class may be *named* in the module docstring, which explains what this
    # script replaces; it must not be imported or constructed.
    assert "logits_processor import" not in src
    assert "RegexLogitsProcessor(" not in src
    assert "import torch" not in src
