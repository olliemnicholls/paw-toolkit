# Example 3: Test-Driven Neural Hardening & Active Learning (`paw.test`)

This example shows the loop `paw-kit` uses to go after **adversarial edge cases, strange formatting, and distribution drift** in a small neural function: fuzz, find failures, ask a teacher for labels, recompile. Whether it helps is task-dependent — see the note at the end.

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
that active learning improves a real compiled function. It has been run for real — a real
compiled adapter, a live Claude teacher, this same suite — and repaired **0 of 11** failures,
correctly: every teacher label failed the suite's own assertions, because the suite has no
"not a date" case for whitespace-only input. See `measurements/README.md`, "Active learning,
for real". The loop is bounded and best-effort; it does not promise 100%.

## Running the Example

Run the Python script:
```bash
uv run python examples/date_normalizer/run.py
```

Run the suite directly from the CLI:
```bash
uv run paw-test check examples/date_normalizer/suite.yaml
```
