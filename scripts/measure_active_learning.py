"""Real active-learning self-healing loop against a live Claude teacher.

Previously `run_active_learning_loop` had only ever been exercised against
`MockPAWBackend` in unit tests. This runs it for real: the failing cases are the
genuine 11/82 fuzzer failures found by scripts/measure_real_backend.py against the real
compiled program, the teacher is an actual Claude call (not a fixture), and recompilation
goes through the real upstream service.

Prerequisites:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    export PAW_API_KEY=paw_sk_...
    An existing manifest from measure_real_backend.py (default path below).

Usage:
    uv run python scripts/measure_active_learning.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from paw_kit import ProgramAsWeightsBackend, load_suite
from paw_kit.test.active import run_active_learning_loop

try:
    import anthropic
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(2)

SUITE = "examples/date_normalizer/suite.yaml"
ADAPTER = "measurements/date_normalizer-paw-4b-qwen3-0.6b.paw"
TEACHER_MODEL = "claude-haiku-4-5-20251001"


def main() -> int:
    if not Path(ADAPTER).is_file():
        print(f"{ADAPTER} not found -- run scripts/measure_real_backend.py first", file=sys.stderr)
        return 2

    client = anthropic.Anthropic()

    def teacher_provider(framed_prompt: str) -> str:
        resp = client.messages.create(
            model=TEACHER_MODEL,
            max_tokens=50,
            messages=[{"role": "user", "content": framed_prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()

    config = load_suite(SUITE).model_copy(update={"adapter_path": ADAPTER})
    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)

    print(f"spec: {config.spec!r}")
    print(f"max_iterations={config.active_learning.max_iterations} "
          f"max_queries_per_iteration={config.active_learning.max_queries_per_iteration}")

    t0 = time.perf_counter()
    report = run_active_learning_loop(config, backend=backend, teacher_provider=teacher_provider)
    wall_s = time.perf_counter() - t0

    print(f"\niterations_run={report.iterations_run}")
    print(f"final_pass_rate={report.final_pass_rate:.1f}%")
    print(f"repaired_edge_cases={report.repaired_edge_cases}")
    print(f"is_success={report.is_success}")
    for i, ir in enumerate(report.iteration_reports, 1):
        print(f"  iteration {i}: {ir.passed_cases}/{ir.total_cases} passed ({ir.pass_rate:.1f}%)")
        for r in ir.results:
            if not r.passed:
                print(f"    still failing: {r.input[:40]!r:44} -> {r.output[:40]!r}  [{'; '.join(r.failed_rules)}]")

    out_path = Path("measurements") / f"active-learning-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps({"wall_s": wall_s, **report.model_dump()}, indent=2, default=str))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
