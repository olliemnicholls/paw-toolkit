"""Neural Hardening & Active Learning Example using paw.test.

Demonstrates:
1. Declarative testing with suite.yaml.
2. Adversarial fuzzing: Unicode corruption, whitespace floods, and domain edge cases.
3. The Active-Learning Self-Healing Loop:
   - Catch failing edge cases under adversarial evaluation.
   - Query teacher for gold labels.
   - Recompile adapter automatically.
   - Achieve 100% assertion compliance.
"""

import os
from pathlib import Path
import re
import shutil
from typing import Dict

from paw_kit import (
    AdversarialFuzzer,
    MockPAWBackend,
    load_suite,
    run_active_learning_loop,
)


SUITE_PATH = os.path.join(os.path.dirname(__file__), "suite.yaml")
CACHE_DIR = os.path.join(os.path.dirname(__file__), ".paw_demo_dates")


def date_teacher(input_text: str) -> str:
    """Simulates frontier teacher (Claude/GPT) resolving difficult date edge cases."""
    clean = input_text.strip()
    # Normalize unicode hyphens/dashes
    clean = clean.replace("—", "-").replace("–", "-")
    # Match standard YYYY-MM-DD
    m = re.search(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})", clean)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    if "January 15, 2026" in clean:
        return "2026-01-15"
    if "December 31" in clean:
        return "2025-12-31"
    return "2026-09-05"


def main():
    print("=" * 75)
    print("PAW-Kit Example: Neural Hardening & Active Learning with paw.test")
    print("=" * 75)
    print(f"Loading suite: {SUITE_PATH}")

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    suite = load_suite(SUITE_PATH)
    adapter_path = suite.adapter_path

    # Step 1: Initialize baseline adapter with standard cases only
    backend = MockPAWBackend()
    initial_examples = [
        {"input": tc.input, "output": tc.expected or "2026-01-01"}
        for tc in suite.standard_cases
    ]
    print(f"\n[1/3] Compiling baseline adapter with {len(initial_examples)} standard examples...")
    backend.compile(
        spec=suite.spec,
        examples=initial_examples,
        output_path=adapter_path,
    )

    # Step 2: Generate adversarial fuzzing mutations
    print("\n[2/3] Generating adversarial fuzzer probes...")
    sample_cases = [tc.input for tc in suite.standard_cases]
    fuzzed_inputs = AdversarialFuzzer.generate(suite.fuzzing, sample_cases)
    print(f"Generated {len(fuzzed_inputs)} synthetic mutations (Unicode, whitespace, domain probes).")
    print(f"Sample mutation: {repr(fuzzed_inputs[0])}")

    # Step 3: Run Active Learning Auto-Repair Loop
    print("\n[3/3] Running Active-Learning Self-Healing Loop:")
    print("-" * 75)

    report = run_active_learning_loop(
        config=suite,
        backend=backend,
        teacher_provider=date_teacher,
    )

    print(f"Active Learning Complete in {report.iterations_run} iteration(s).")
    print(f"Final Status:        {'[PASSED]' if report.is_success else '[FAILED]'}")
    print(f"Pass Rate:           {report.final_pass_rate:.1f}%")
    print(f"Total Repaired:      {report.repaired_edge_cases} edge cases resolved by teacher")
    print(f"Recompiled Adapter:  {report.recompiled}")
    print("-" * 75)
    print("\n[DONE] The loop converged -- on a mock adapter with a stub teacher, so this shows the")
    print("loop's mechanics only. Against a real adapter and a real teacher it repaired 0 of 11")
    print("failures on this same suite (see measurements/README.md, 'Active learning, for real').")
    print("Run via CLI anytime: `uv run paw-test check examples/date_normalizer/suite.yaml`\n")


if __name__ == "__main__":
    main()
