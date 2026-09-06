"""End-to-end measurement of ProgramAsWeightsBackend against the real upstream service.

This is the script that produces the first honest numbers for paw-kit. It has never been
run as of 2026-09-06; expect to fix things the first time. Run it on each machine you care
about (e.g. an RTX 3080 and an A100) and commit the JSON it writes under `measurements/`.

Prerequisites:
    pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
    export PAW_API_KEY=paw_sk_...

Usage:
    uv run python scripts/measure_real_backend.py examples/date_normalizer/suite.yaml \
        --compiler paw-4b-qwen3-0.6b --calls 50 --label rtx3080

    # the finetune compiler from the paper (queued on the service; minutes)
    uv run python scripts/measure_real_backend.py examples/date_normalizer/suite.yaml \
        --compiler paw-ft-bs48 --calls 50 --label a100

What it measures, in order:
  1. compile wall time (includes queueing; the service's own timing is in the manifest)
  2. cold first call (model load + generation) then N warm calls: p50 / p90 / p99 / mean
  3. `paw-test` pass rate on the suite's standard cases + fuzzed cases, with per-case output
     so you can read *what* it got wrong, not just how many
It writes `measurements/<label>-<compiler>-<timestamp>.json` and prints a summary. No
comparison column against a remote API is produced: measure that separately, with a real
API, if you want to publish one.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

from paw_kit import ProgramAsWeightsBackend, TestRunner, load_suite


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "none"
    except Exception:
        return "none"


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(round(p * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", help="path to a suite.yaml (spec + cases + assertions)")
    ap.add_argument("--compiler", default="paw-4b-qwen3-0.6b")
    ap.add_argument("--calls", type=int, default=50, help="warm inference calls to time")
    ap.add_argument("--label", default=platform.node(), help="machine label for the output file")
    ap.add_argument("--gpu-layers", type=int, default=None, help="0 forces CPU; default lets the SDK decide")
    ap.add_argument("--max-spec-examples", type=int, default=0,
                    help="how many standard cases to fold into the spec as few-shot examples (0 = pure spec)")
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    if not os.environ.get("PAW_API_KEY"):
        print("PAW_API_KEY is not set. Get one at https://programasweights.com/settings", file=sys.stderr)
        return 2

    suite = load_suite(args.suite)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    adapter_path = str(out_dir / f"{suite.task_name}-{args.compiler}.paw")

    backend = ProgramAsWeightsBackend(
        compiler=args.compiler, n_gpu_layers=args.gpu_layers, max_spec_examples=args.max_spec_examples,
    )
    if not backend.is_available():
        print("programasweights SDK not importable; see module docstring for install.", file=sys.stderr)
        return 2

    examples = [{"input": c.input, "output": c.expected} for c in suite.standard_cases if c.expected]

    # 1. compile
    print(f"[compile] compiler={args.compiler} spec={suite.spec!r} examples_folded={min(len(examples), args.max_spec_examples)}")
    t0 = time.perf_counter()
    backend.compile(suite.spec, examples, adapter_path)
    compile_s = time.perf_counter() - t0
    manifest = json.loads(Path(adapter_path).read_text())
    print(f"[compile] done in {compile_s:.1f}s wall; program_id={manifest['program_id']}")

    # 2. latency
    probe = suite.standard_cases[0].input if suite.standard_cases else "hello"
    t0 = time.perf_counter()
    first_out = backend.infer(adapter_path, probe)
    cold_ms = (time.perf_counter() - t0) * 1000
    print(f"[infer] cold call {cold_ms:.0f}ms -> {first_out!r}")

    warm: list[float] = []
    inputs = [c.input for c in suite.standard_cases] or [probe]
    for i in range(args.calls):
        text = inputs[i % len(inputs)]
        t0 = time.perf_counter()
        backend.infer(adapter_path, text)
        warm.append((time.perf_counter() - t0) * 1000)
    warm_sorted = sorted(warm)
    lat = {
        "cold_ms": cold_ms,
        "warm_calls": len(warm),
        "p50_ms": _pct(warm_sorted, 0.50),
        "p90_ms": _pct(warm_sorted, 0.90),
        "p99_ms": _pct(warm_sorted, 0.99),
        "mean_ms": statistics.fmean(warm) if warm else float("nan"),
    }
    print(f"[infer] warm p50={lat['p50_ms']:.0f}ms p90={lat['p90_ms']:.0f}ms p99={lat['p99_ms']:.0f}ms over {len(warm)} calls")

    # 3. suite pass rate (standard + fuzzed), using the adapter we just compiled
    config = suite.model_copy(update={"adapter_path": adapter_path})
    report = TestRunner(backend=backend).run(config)
    print(f"[paw-test] {report.passed_cases}/{report.total_cases} passed ({report.pass_rate:.1f}%)")
    for r in report.results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"   {mark} {r.input[:40]!r:44} -> {r.output[:40]!r}" + (f"  [{'; '.join(r.failed_rules)}]" if r.failed_rules else ""))

    result = {
        "label": args.label,
        "timestamp": stamp,
        "machine": {"host": platform.node(), "platform": platform.platform(), "gpu": _gpu_name(), "python": sys.version.split()[0]},
        "suite": args.suite,
        "task_name": suite.task_name,
        "compiler": args.compiler,
        "gpu_layers": args.gpu_layers,
        "max_spec_examples": args.max_spec_examples,
        "compile_wall_s": compile_s,
        "manifest": manifest,
        "latency": lat,
        "paw_test": {
            "total": report.total_cases,
            "passed": report.passed_cases,
            "pass_rate": report.pass_rate,
            "cases": [r.model_dump() for r in report.results],
        },
    }
    out_path = out_dir / f"{args.label}-{args.compiler}-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
