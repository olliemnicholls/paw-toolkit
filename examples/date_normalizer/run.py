"""Date normaliser example: paw-test suites, fuzzing and active learning.

Demonstrates:
1. Declarative testing with suite.yaml.
2. Adversarial fuzzing: unicode corruption, whitespace floods, and domain edge cases.
3. The active-learning loop: failing cases that carry an `expected` answer are sent
   to a teacher for labels, the labels are checked against the suite, and the adapter
   is recompiled with the ones that pass. Cases with no answer key are refused, so
   this run ends FAILED with nothing repaired; see README.md for why that is correct.
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
    print("PAW-Kit Example: paw-test suites, fuzzing and active learning")
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
    print(f"Generated {len(fuzzed_inputs)} synthetic mutations (unicode, whitespace).")
    print(f"Sample mutation: {repr(fuzzed_inputs[0])}")

    # Step 3: run the active-learning loop
    print("\n[3/3] Running the active-learning loop:")
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
    print("\n[DONE] FAILED with nothing repaired is the expected result: the fuzz-generated cases")
    print("have no answer key, so the loop refuses to send them to the teacher. This is a mock")
    print("adapter and a stub teacher; the example shows the loop's mechanics, not model quality.")
    print("Same suite from the CLI: `uv run paw-test check examples/date_normalizer/suite.yaml`\n")


if __name__ == "__main__":
    main()
