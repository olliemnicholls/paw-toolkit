# Example 3: Test-Driven Neural Hardening & Active Learning (`paw.test`)

This example demonstrates how `paw-kit` solves the core challenge of deploying small neural functions in production: **resilience against adversarial edge cases, strange formatting, and distribution drift**.

## What It Does

1. **Declarative Test Suite (`suite.yaml`)**:
   - Outlines expected behavior on standard in-distribution cases.
   - Sets strict invariant assertions (e.g. `regex_match`, `min_length`, `max_length`, `not_contains`).
   - Configures synthetic mutation generators (`fuzzing`).
2. **Adversarial Fuzzing (`AdversarialFuzzer`)**:
   - Generates synthetic test inputs containing zero-width spaces, RTL overrides, whitespace floods, and specialized domain probes (e.g. CJK date characters, em-dashes).
3. **Active-Learning Self-Healing Loop (`run_active_learning_loop`)**:
   - Evaluates the adapter against the adversarial inputs.
   - Flags assertion failures.
   - Automatically queries the frontier teacher model for gold labels on failing inputs.
   - Recompiles the adapter with the augmented dataset.
   - Repeats until every assertion passes or `max_iterations` is reached.

**Note:** in this example both the "teacher" and the adapter are deterministic Python stubs
(`MockPAWBackend`), so the loop always converges. That demonstrates the loop's mechanics, not
that active learning improves a real compiled function. Measuring the latter is the point of
`scripts/measure_real_backend.py` in the repo root.

## Running the Example

Run the Python script:
```bash
uv run python examples/date_normalizer/run.py
```

Run the suite directly from the CLI:
```bash
uv run paw-test check examples/date_normalizer/suite.yaml
```
