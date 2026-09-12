# Date normaliser: `paw-test` suites, fuzzing and active learning

Runs a declarative test suite against a compiled adapter, fuzzes it, and drives the
active-learning loop that sends repairable failures to a teacher for labels.

```bash
uv run python examples/date_normalizer/run.py                    # in code
uv run paw-test check examples/date_normalizer/suite.yaml         # same suite, from the CLI
```

## What it runs

`suite.yaml` holds seven standard cases, each with an `expected` answer, four assertions
on the output shape (a `YYYY-MM-DD` regex, exact length, no `ERROR`), and fuzzing
settings that mutate the standard inputs with unicode corruption and whitespace floods.

The script compiles a baseline adapter from the seven standard cases, generates the fuzz
cases, and calls `run_active_learning_loop` with a stub teacher. The loop evaluates the
adapter on every case, sends failing cases that carry an `expected` value to the teacher,
checks the returned labels against the suite's assertions and against `expected`, and
recompiles with any label that passes.

## What to expect from the output

The run ends `[FAILED]` at a pass rate around 5%, with nothing repaired and no
recompile. That is the correct result. The seven standard cases pass; the roughly 120
fuzz-generated cases fail, and none of them can be repaired because a fuzz mutation has
no answer key. A teacher label for it could not be checked against anything, so the loop
declines to send it and reports how many cases it declined. To make an edge case
repairable, add it to `standard_cases` with its `expected` value; the three unicode and
whitespace cases at the bottom of the standard list were added that way.

## What it does not show

The adapter is `MockPAWBackend`, a dictionary lookup, and the teacher is a few lines of
regex. The example demonstrates the loop's mechanics: what gets sent to the teacher, what
gets refused, and why. It says nothing about whether active learning improves a real
compiled function. On a real adapter with a live teacher, this suite repaired nothing
for the same reason: every failing input was whitespace-only garbage the suite gave no
legal answer for. Setting the suite's `abstain_value` is how to teach a model that "no
valid date" is the right answer there; see [testing](../../docs/testing.md).
