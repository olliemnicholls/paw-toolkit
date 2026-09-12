"""Real fail-open safety test.

`decisions.md`'s #1 invariant: "local compiled functions are performance optimizations,
never single points of failure." Every existing test of this (test_jit.py etc.) mocks
the exception -- `raise Exception("boom")` from a fake backend. This script induces two
*real* failures against the real `ProgramAsWeightsBackend` and confirms the documented
fallback actually holds against whatever the real SDK/filesystem actually does, not
whatever a test double was written to do:

  1. A real compiled adapter file goes missing/corrupted after a successful compile
     (disk cleanup, race condition, bad deploy -- a realistic production scenario).
  2. A real, deliberately invalid `PAW_API_KEY` causes the real upstream service to
     reject a real compile request at threshold (not a mocked RuntimeError).

B-9 (bug hunt 2026-09-11), CONFIRMED by inspection: the teacher used to return one
constant marker string for every input, so the adapter's only training signal was three
identical pairs mapping arbitrary inputs to that string -- and phases 3 and 4 then asserted
`out == TEACHER_MARKER`. A degenerate adapter that had memorised the marker satisfies that
assertion with no fallback having occurred, and `measurements/README.md`'s fail-open section
independently documents exactly that degeneration on an identically-shaped compile (and a
cached `PawFunction` survives deleting the manifest, so phase 3 does not even need the
adapter to be reloaded). The script also wrote no artifact, so there was no record of which
checks passed. The claim was unsupported rather than wrong.

Three changes, so the assertion cannot be satisfied by memorisation:

  * the teacher's output is **unique per call** (`TEACHER_MARKER-0001`, `-0002`, ...), so
    no adapter trained on earlier calls can produce the value this call returned;
  * a fallback is asserted on three independent signals that must all agree -- the output
    equals the value the teacher produced on *this* call, the teacher call count went up by
    exactly one, and the decorator's own `get_fail_open_count()` went up. Output text alone
    is never sufficient;
  * every check and its outcome is written to a JSON artifact.

Uses a trivial deterministic Python function as the "teacher" rather than a real LLM -- the
backend's failure behavior is what's under test, not teacher quality, so no Anthropic key is
spent on this.

Prerequisites:
    export PAW_API_KEY=paw_sk_...   (must be a REAL, currently-valid key for phase 1-3;
                                      phase 4 deliberately overwrites it with garbage)

Usage:
    uv run python scripts/measure_fail_open.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from paw_kit import ProgramAsWeightsBackend, compile_on_hit

TEACHER_MARKER = "TEACHER-FALLBACK-1999-12-31"
SPEC = "Convert natural language date expressions into standard ISO-8601 YYYY-MM-DD format."


class UniqueTeacher:
    """Stands in for 'your existing Claude/OpenAI call'. Deterministic, obviously
    synthetic, and **different on every call**.

    The per-call suffix is the whole point (B-9). With a constant marker, an adapter that
    memorised its three identical training pairs returns the marker forever, and
    `out == TEACHER_MARKER` is then satisfied whether a fallback happened or not. With a
    per-call suffix no adapter can produce the value this call returned, so equality with
    `last` is evidence about *this* call rather than about the training set.
    """

    def __init__(self, marker: str = TEACHER_MARKER) -> None:
        self.marker = marker
        self.calls = 0
        self.last: Optional[str] = None
        self.history: List[str] = []

    def __call__(self, _text: str) -> str:
        self.calls += 1
        self.last = f"{self.marker}-{self.calls:04d}"
        self.history.append(self.last)
        return self.last


class CheckLog:
    """Every check, its outcome and its evidence, for the artifact."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def record(self, label: str, passed: bool, **evidence: Any) -> bool:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {label}")
        for key, value in evidence.items():
            print(f"         {key}={value!r}")
        self.checks.append({"check": label, "passed": bool(passed), "evidence": evidence})
        return bool(passed)

    @property
    def any_failed(self) -> bool:
        return any(not c["passed"] for c in self.checks)


def fallback_verdict(
    output: str,
    teacher_output: Optional[str],
    teacher_calls_before: int,
    teacher_calls_after: int,
    fail_open_before: int,
    fail_open_after: int,
) -> Dict[str, Any]:
    """Did this one call fall open to the teacher? Pure; all three signals must agree.

    * `returned_this_calls_teacher_output` -- the output is the value the teacher produced
      on *this* call, not a value it produced during training. A memorising adapter fails
      this because the per-call suffix is new.
    * `teacher_called_exactly_once` -- the teacher really was invoked.
    * `fallback_counted` -- the decorator's own fail-open counter moved. This is the
      instrumented signal the report asked for: it does not depend on output text at all.

    Reported field by field so the artifact says *which* signal disagreed when one does.
    """
    returned_this = teacher_output is not None and output == teacher_output
    called_once = teacher_calls_after == teacher_calls_before + 1
    counted = fail_open_after > fail_open_before
    return {
        "returned_this_calls_teacher_output": returned_this,
        "teacher_called_exactly_once": called_once,
        "fallback_counted": counted,
        "teacher_calls_before": teacher_calls_before,
        "teacher_calls_after": teacher_calls_after,
        "fail_open_before": fail_open_before,
        "fail_open_after": fail_open_after,
        "output": output,
        "teacher_output_this_call": teacher_output,
        "is_fallback": returned_this and called_once and counted,
    }


def probe_fallback(decorated: Callable[[str], str], teacher: UniqueTeacher,
                   text: str) -> Dict[str, Any]:
    """Call `decorated(text)` and return `fallback_verdict` for that call.

    `decorated` needs only to be callable and to expose `get_fail_open_count()`, which is
    what `compile_on_hit` attaches -- so a test double can stand in for it and prove the
    verdict is falsifiable.
    """
    calls_before = teacher.calls
    fail_open_before = decorated.get_fail_open_count()  # type: ignore[attr-defined]
    output = decorated(text)
    return fallback_verdict(
        output=output,
        teacher_output=teacher.last,
        teacher_calls_before=calls_before,
        teacher_calls_after=teacher.calls,
        fail_open_before=fail_open_before,
        fail_open_after=decorated.get_fail_open_count(),  # type: ignore[attr-defined]
    )


def served_locally_verdict(
    output: str,
    teacher_calls_before: int,
    teacher_calls_after: int,
    teacher_history: List[str],
) -> Dict[str, Any]:
    """The converse of `fallback_verdict`: this call was served by the adapter.

    The teacher not having been called is the load-bearing signal. The two output
    comparisons are reported as evidence, and the `any` one is the interesting one: an
    adapter emitting a string the teacher produced *earlier* is the memorisation the
    measurements README observed, which `served_locally` alone would not distinguish from
    healthy local inference.
    """
    not_called = teacher_calls_after == teacher_calls_before
    return {
        "teacher_not_called": not_called,
        "output_is_not_any_teacher_output": output not in set(teacher_history),
        "output": output,
        "teacher_outputs_so_far": list(teacher_history),
        "served_locally": not_called,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="measurements")
    ap.add_argument("--label", default="3080")
    args = ap.parse_args()

    log = CheckLog()
    started = time.time()
    phases: Dict[str, Any] = {}

    if not os.environ.get("PAW_API_KEY"):
        print("export a real PAW_API_KEY first (phases 1-3 need a working compile)", file=sys.stderr)
        return 2

    # ---- Phase 1: normal tracing + real compile, valid key -------------------------
    cache_dir = "./.paw_failopen_demo"
    shutil.rmtree(cache_dir, ignore_errors=True)
    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16,
                                      public=False)
    teacher = UniqueTeacher()
    threshold = 3
    decorated = compile_on_hit(
        spec=SPEC,
        threshold=threshold,
        backend=backend,
        cache_dir=cache_dir,
        sync_compile=True,
        # Track 14: this measurement is about the fail-open path *after* a hot-swap, so
        # it needs the swap to happen at the threshold exactly as the recorded run did.
        shadow_window=0,
    )(teacher)

    print("=== phase 1: trace to threshold, real synchronous compile ===")
    for i in range(threshold):
        out = decorated(f"date input {i}")
        print(f"  call {i + 1}: {out!r}")
    status = decorated.db.get_status(decorated.task_id)
    adapter_path = decorated.db.get_adapter_path(decorated.task_id)
    print(f"  status={status!r} adapter_path={adapter_path!r}")
    log.record("compile succeeded with a valid key",
               status == "ready" and adapter_path is not None,
               status=status, adapter_path=adapter_path)
    manifest = {}
    if adapter_path and Path(adapter_path).is_file():
        manifest = json.loads(Path(adapter_path).read_text())
    phases["phase_1_compile"] = {
        "status": status,
        "adapter_path": adapter_path,
        "teacher_calls": teacher.calls,
        "teacher_outputs": list(teacher.history),
        "program_id": manifest.get("program_id"),
        # A-2: the manifest key that records the *requested* visibility was renamed
        # `public` -> `public_requested`, and `public_confirmed` (three-state: True /
        # False / None-with-a-reason) now carries what the server actually reported.
        # Read the new name with a legacy fallback so a summary built from a manifest
        # written before the rename still reports the value instead of silently None.
        "public_requested": manifest.get("public_requested", manifest.get("public")),
        "public_confirmed": manifest.get("public_confirmed"),
        "public_confirmed_reason": manifest.get("public_confirmed_reason"),
        "examples_folded_into_spec": manifest.get("examples_folded_into_spec"),
    }

    print("\n=== phase 2: next call hot-swaps to the real local adapter ===")
    calls_before = teacher.calls
    live = decorated("January 1, 2026")
    served = served_locally_verdict(live, calls_before, teacher.calls, teacher.history)
    print(f"  result: {live!r}")
    log.record("hot-swapped call was served locally (the teacher was not called)",
               served["served_locally"], **served)
    log.record("hot-swapped call did not echo any earlier teacher output (not memorised)",
               served["output_is_not_any_teacher_output"],
               output=served["output"], teacher_outputs_so_far=served["teacher_outputs_so_far"])
    phases["phase_2_hot_swap"] = served

    print("\n=== phase 3: adapter file goes missing (real FileNotFoundError) ===")
    os.remove(adapter_path)
    print(f"  deleted {adapter_path}")
    verdict = probe_fallback(decorated, teacher, "January 1, 2026")
    log.record("fell back to the teacher, did not raise", verdict["is_fallback"], **verdict)
    # Spelt out separately so a partial result says which signal disagreed. The middle one
    # is B-9 itself: output text alone could be satisfied by a memorising adapter.
    log.record("phase 3 returned the value the teacher produced on THIS call",
               verdict["returned_this_calls_teacher_output"],
               output=verdict["output"], teacher_output=verdict["teacher_output_this_call"])
    log.record("phase 3 incremented the decorator's fail-open counter",
               verdict["fallback_counted"],
               before=verdict["fail_open_before"], after=verdict["fail_open_after"])
    phases["phase_3_missing_adapter"] = verdict

    # ---- Phase 4: a real invalid API key breaks a fresh compile --------------------
    print("\n=== phase 4: real invalid PAW_API_KEY, fresh task, repeated post-threshold calls ===")
    real_key = os.environ["PAW_API_KEY"]
    os.environ["PAW_API_KEY"] = "paw_sk_intentionally_invalid_for_fail_open_test"
    phase4: Dict[str, Any] = {"calls": []}
    try:
        cache_dir2 = "./.paw_failopen_demo2"
        shutil.rmtree(cache_dir2, ignore_errors=True)
        backend2 = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16,
                                           public=False)
        teacher2 = UniqueTeacher(marker=TEACHER_MARKER + "-PHASE4")
        decorated2 = compile_on_hit(
            spec=SPEC + " (fail-open test variant, invalid key)",
            threshold=threshold,
            backend=backend2,
            cache_dir=cache_dir2,
            sync_compile=True,
            shadow_window=0,  # see the note on the first decoration above
        )(teacher2)

        all_ok = True
        for i in range(6):  # 3 to cross threshold, 3 more to watch repeated-attempt behavior
            calls_before = teacher2.calls
            out = decorated2(f"broken-key input {i}")
            attempts = decorated2.db.get_compile_attempts(decorated2.task_id)
            st = decorated2.db.get_status(decorated2.task_id)
            # Pre-threshold calls are served by the teacher without any fallback having
            # occurred, so only the post-threshold ones are fallbacks. What must hold for
            # every call is that the caller got this call's teacher output and did not see
            # an exception.
            returned_this = out == teacher2.last
            called_once = teacher2.calls == calls_before + 1
            ok = returned_this and called_once
            all_ok = all_ok and ok
            print(f"  call {i + 1}: result={out!r} status={st!r} compile_attempts={attempts} "
                  f"returned_this_calls_teacher_output={returned_this}")
            phase4["calls"].append({
                "call": i + 1,
                "status": st,
                "compile_attempts": attempts,
                "output": out,
                "teacher_output_this_call": teacher2.last,
                "returned_this_calls_teacher_output": returned_this,
                "teacher_called_exactly_once": called_once,
                # decorator.py:517 triggers the compile attempt when
                # `call_count >= threshold`, and `call_count` is 1-indexed (the i-th 0-indexed
                # call makes it i+1) -- so the call that actually crosses the threshold and
                # experiences the (failing) compile attempt is `i == threshold - 1`, not
                # `i == threshold`. The off-by-one previously excluded exactly that call from
                # "post_threshold", which is the one call this label most needs to include.
                "post_threshold": i >= threshold - 1,
                "fail_open_count": decorated2.get_fail_open_count(),
            })
        log.record("every call returned THIS call's teacher result despite compile failing",
                   all_ok, calls=len(phase4["calls"]))
        post = [c for c in phase4["calls"] if c["post_threshold"]]
        # NOT a fallback in decorator.py's sense, and `fail_open_count` must stay 0 here.
        # `get_fail_open_count()` is incremented in exactly one place (decorator.py:448):
        # inside the `status == "ready"` branch, when a *compiled adapter's local inference*
        # raises. In this phase the compile itself never succeeds (the SDK is patched to
        # reject every attempt), so the task never reaches "ready" -- every call takes the
        # "adapter not serving: invoke wrapped function" branch (decorator.py:466), which is
        # not instrumented as a fail-open at all. A prior version of this check asserted
        # `fail_open_count > 0` here, which is false by construction and would fail every
        # real run: exactly the kind of claim this campaign exists to catch, caught by Phase F
        # before this script was ever executed. What phase 4 actually demonstrates is that a
        # broken compile degrades to "always call the teacher" with no exception and no
        # (mis-attributed) counter increment -- recorded as its own, correctly-named check.
        log.record("fail_open_count stays 0 (a compile failure is not a local-inference "
                   "fail-open; decorator.py counts the latter, not the former)",
                   bool(post) and post[-1]["fail_open_count"] == 0,
                   fail_open_count=post[-1]["fail_open_count"] if post else None)
        final_status = decorated2.db.get_status(decorated2.task_id)
        phase4["final_status"] = final_status
        phase4["final_compile_attempts"] = decorated2.db.get_compile_attempts(decorated2.task_id)
        phase4["teacher_outputs"] = list(teacher2.history)
        print(f"  final status: {final_status!r} (expect 'failed' after "
              f"{phase4['final_compile_attempts']} attempts, or repeated retries if sync "
              "bypasses the terminal-state guard)")
    finally:
        os.environ["PAW_API_KEY"] = real_key  # restore for anything after this script
    phases["phase_4_invalid_key"] = phase4

    artifact = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": time.time() - started,
        "spec": SPEC,
        "teacher": (
            "UniqueTeacher: a pure Python function returning "
            f"'{TEACHER_MARKER}-NNNN' where NNNN is the call number. Unique per call on "
            "purpose -- see report B-9. No LLM, no Anthropic key."
        ),
        "threshold": threshold,
        "fallback_evidence": (
            "A fallback is asserted on three signals that must all agree: the output equals "
            "the value the teacher produced on THIS call, the teacher call count rose by "
            "exactly one, and the decorator's get_fail_open_count() rose. Output text alone "
            "is never sufficient, because an adapter that memorised its training pairs can "
            "reproduce a constant marker with no fallback having occurred."
        ),
        "checks": log.checks,
        "checks_total": len(log.checks),
        "checks_passed": sum(1 for c in log.checks if c["passed"]),
        "all_passed": not log.any_failed,
        "phases": phases,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fail-open-{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\nwrote {out_path}")

    print()
    if log.any_failed:
        print("SOME FAIL-OPEN CHECKS FAILED -- see above")
        return 1
    print("ALL FAIL-OPEN CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
