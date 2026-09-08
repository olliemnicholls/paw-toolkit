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

Uses a trivial deterministic Python function as the "teacher" (returns a fixed marker
string) rather than a real LLM -- the backend's failure behavior is what's under test,
not teacher quality, so no Anthropic key is spent on this.

Prerequisites:
    export PAW_API_KEY=paw_sk_...   (must be a REAL, currently-valid key for phase 1-3;
                                      phase 4 deliberately overwrites it with garbage)

Usage:
    uv run python scripts/measure_fail_open.py
"""

from __future__ import annotations

import os
import shutil
import sys

from paw_kit import ProgramAsWeightsBackend, compile_on_hit

TEACHER_MARKER = "TEACHER-FALLBACK-1999-12-31"
SPEC = "Convert natural language date expressions into standard ISO-8601 YYYY-MM-DD format."


def teacher(_text: str) -> str:
    """Stands in for 'your existing Claude/OpenAI call'. Deterministic and obviously
    synthetic on purpose, so any test assertion below can tell teacher output apart
    from real adapter output at a glance."""
    return TEACHER_MARKER


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        global _any_failed
        _any_failed = True


_any_failed = False


def main() -> int:
    if not os.environ.get("PAW_API_KEY"):
        print("export a real PAW_API_KEY first (phases 1-3 need a working compile)", file=sys.stderr)
        return 2

    # ---- Phase 1: normal tracing + real compile, valid key -------------------------
    cache_dir = "./.paw_failopen_demo"
    shutil.rmtree(cache_dir, ignore_errors=True)
    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)
    threshold = 3
    decorated = compile_on_hit(
        spec=SPEC, threshold=threshold, backend=backend, cache_dir=cache_dir, sync_compile=True
    )(teacher)

    print("=== phase 1: trace to threshold, real synchronous compile ===")
    for i in range(threshold):
        out = decorated(f"date input {i}")
        print(f"  call {i + 1}: {out!r}")
    status = decorated.db.get_status(decorated.task_id)
    adapter_path = decorated.db.get_adapter_path(decorated.task_id)
    print(f"  status={status!r} adapter_path={adapter_path!r}")
    check("compile succeeded with a valid key", status == "ready" and adapter_path is not None)

    print("\n=== phase 2: next call hot-swaps to the real local adapter ===")
    live = decorated("January 1, 2026")
    print(f"  result: {live!r}")
    check("hot-swapped call did not return the teacher marker (used the real adapter)", live != TEACHER_MARKER)

    print("\n=== phase 3: adapter file goes missing (real FileNotFoundError) ===")
    os.remove(adapter_path)
    print(f"  deleted {adapter_path}")
    result = decorated("January 1, 2026")
    print(f"  result: {result!r}")
    check("fell back to the teacher, did not raise", result == TEACHER_MARKER)

    # ---- Phase 4: a real invalid API key breaks a fresh compile --------------------
    print("\n=== phase 4: real invalid PAW_API_KEY, fresh task, repeated post-threshold calls ===")
    real_key = os.environ["PAW_API_KEY"]
    os.environ["PAW_API_KEY"] = "paw_sk_intentionally_invalid_for_fail_open_test"
    try:
        cache_dir2 = "./.paw_failopen_demo2"
        shutil.rmtree(cache_dir2, ignore_errors=True)
        backend2 = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)
        decorated2 = compile_on_hit(
            spec=SPEC + " (fail-open test variant, invalid key)",
            threshold=threshold,
            backend=backend2,
            cache_dir=cache_dir2,
            sync_compile=True,
        )(teacher)

        all_ok = True
        for i in range(6):  # 3 to cross threshold, 3 more to watch repeated-attempt behavior
            out = decorated2(f"broken-key input {i}")
            attempts = decorated2.db.get_compile_attempts(decorated2.task_id)
            st = decorated2.db.get_status(decorated2.task_id)
            print(f"  call {i + 1}: result={out!r} status={st!r} compile_attempts={attempts}")
            if out != TEACHER_MARKER:
                all_ok = False
        check("every call returned the teacher's result despite compile failing every time", all_ok)
        final_status = decorated2.db.get_status(decorated2.task_id)
        print(f"  final status: {final_status!r} (expect 'failed' after {decorated2.db.get_compile_attempts(decorated2.task_id)} attempts, or repeated retries if sync bypasses the terminal-state guard)")
    finally:
        os.environ["PAW_API_KEY"] = real_key  # restore for anything after this script

    print()
    if _any_failed:
        print("SOME FAIL-OPEN CHECKS FAILED -- see above")
        return 1
    print("ALL FAIL-OPEN CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
