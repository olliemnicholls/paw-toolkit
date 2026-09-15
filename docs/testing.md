# Test a compiled function before trusting it

```yaml
# suite.yaml
task_name: date_normalizer
spec: "Convert natural language date expressions into ISO-8601 YYYY-MM-DD."
adapter_path: ".paw/date_normalizer.paw"

standard_cases:
  - input: "January 15, 2026"
    expected: "2026-01-15"

assertions:
  - rule: regex_match
    pattern: "^\\d{4}-\\d{2}-\\d{2}$"
  - rule: max_length
    value: 10

fuzzing:
  inject_unicode: true
  whitespace_flood: true
  adversarial_probes:
    - "2026年09月05日"

active_learning:
  auto_recompile: true
  max_iterations: 3
```

```bash
uv run paw-test check examples/date_normalizer/suite.yaml
```

The runner reports pass rate per case and per assertion. A case with an `expected` value
is also checked against it, and a mismatch counts as a failure; `check` prints a separate
"Correct against expected" line for these. Matching is strict about quoting: a JSON
string scalar never matches the bare value it wraps. When unquoting would have matched
more cases, a second "Correct after unquoting a JSON string" line says how many, so the
gap is visible rather than silently scored wrong. That hint fires only when exactly one
side parses as JSON, so it cannot diagnose a quoted object or a quoted number; those are
reported as plain differences. `--adapter PATH` runs the same suite against a different
compiled adapter.

With `auto_recompile: true` and a teacher callable supplied in code
(`run_active_learning_loop(..., teacher_provider=...)`), failing inputs are sent to the
teacher inside a delimited prompt, the returned labels are checked against the suite's
assertions and against the case's own `expected` value, and the adapter is recompiled
with the accepted labels folded in.

> **Only cases that carry an `expected` value are repairable.** A fuzz-generated case or
> an `adversarial_probes` entry has no answer key, so a teacher label for it cannot be
> checked against anything. Such inputs are never sent to the teacher; `check` reports
> how many it declined to guess at. To make an edge case repairable, promote it to a
> `standard_case` and write down its answer. To train a model to *abstain* on an input,
> set the case's `expected` to the suite's `abstain_value`.

## `paw-test check` and its stub teacher

**`paw-test check` supplies its own teacher**, and it is a demo stub: a two-branch lookup
that answers `"2026-01-01"` to almost any input, not a frontier model. Because recompilation
overwrites the adapter in place (and, on a real backend, costs a paid upstream compile), the
CLI runs **read-only against `--backend real`**: `auto_recompile` is forced off with a
printed notice, and passing `--auto-recompile` explicitly is refused rather than allowed,
because stub labels must never become training signal for a paid compile. Drive the loop
from code, with a real teacher, when you want it to actually repair something.

Against the mock backend (the default), `paw-test check` will compile a *fresh* adapter —
there is nothing at `adapter_path` yet to lose. But it **refuses to recompile any adapter
that already exists**, mock or real, printing a notice instead: `MockPAWBackend.compile()`
writes a real file, so overwriting one in place would replace it with a mock stub containing
the demo teacher's invented labels, and an existing adapter is as much a real artifact as any
other. If you want the stub teacher to repair an existing mock adapter, delete it first (the
same opt-out this guard uses for a first compile), or drive the loop from code with a real
teacher.

## Why the loop made no progress

`ActiveLearningReport.stuck_reason` says why a repair iteration made no progress. It is
one of
`"all_labels_rejected"` (the teacher answered, but every answer failed the suite's own
assertions), `"teacher_errors"` (the teacher call itself raised or returned nothing), or
`"no_failures"` (there was nothing to query). The report's `rejected_labels` lists each
rejected input with the teacher's (truncated) output and which rules it failed. A
recompile is skipped when an iteration added no new examples.

Sometimes `all_labels_rejected` means the suite, not the model, is wrong. If every failing
input has no legal answer (whitespace-only garbage with no valid date, say), set a
suite-level `abstain_value`: any assertion then passes automatically when the output
matches it exactly, so the loop can teach the model to admit "I don't know" instead of
being forced to hallucinate a shaped-but-wrong answer. `check` and `compare` both honour
it.

## `paw-test compare`

**`paw-test compare A.paw B.paw suite.yaml`** runs every case in a suite, fuzz cases
included, through two compiled adapters and diffs the results per case: differences
first, then a one-line summary (`--json out.json` for the full report; `--no-fuzz` for
standard cases only). Use it to decide between two compiles of the same spec, or between
upstream's two compilers: per-case differences are often more telling than the aggregate
pass rates, which can sit inside the judge's own noise (next section).

Three things to know. `--backend real` is read-only here too: `compare` never calls
`compile()`. Both adapters stay loaded for the whole run, two copies of the base model
under `--backend real`, and the first row's latencies include that cold load. A case
on which an adapter could not run at all is reported as an execution error, counted in
the summary and reflected in a non-zero exit, never as agreement between two adapters
that both failed. And "identical" is byte-level; the summary also reports how many
outputs are equivalent after parsing both as JSON (or normalising whitespace), and lists
whitespace-only differences under their own heading, since `json.dumps` spacing alone
can account for most of a diff.

## `paw-test judge`

**`paw-test judge report.json --spec "..."`** scores a `compare`/`check` report's
outputs with an independent LLM judge and persists a per-case verdict and reason, keyed
by a stable hash of (input, output) so two runs can be diffed later
(`paw-test judge --diff old.json new.json`, which also reads compare-shaped reports and
diffs each side). The judge is noisy: at the API's default temperature, re-judging
identical pairs flips some verdicts run to run (see [results](./results.md)). The shipped
`anthropic_judge` therefore pins `temperature=0.0`. That alone does not guarantee
bit-identical judging, and `--diff` is how you check whether it held for your own prompt
and judge model.

**This command sends data off your machine.** Each case's input and output, plus the
spec, go to the Anthropic API; nothing else is transmitted, and the key is read only from
`ANTHROPIC_API_KEY`. A compare report costs two calls per case. The report separates
three things a single pass rate would hide: cases where the judge disagrees with the
suite's own assertions (listed by case id), verdicts the judge phrased in a way that could
not be parsed, and calls that failed outright. A failed call is recorded and the run
continues, so a rate limit late in a long run does not discard the verdicts already paid
for. If every case errors the command exits non-zero and says the judge itself is
failing, so a broken SDK cannot read as an adapter failing every case. The reference judge
needs the `judge` extra; see [install](./install.md#the-judge-extra).

## `paw-kit lint-spec`

`paw-kit lint-spec "text"` or `--file spec.txt` runs static checks for common
spec-authoring mistakes: an unpinned output format, a forced choice with no way to
abstain, a schema with every field required, an over-long spec, and examples in a single
form. It is advice, not a gate.

An `Optional`/defaulted field gives the model a representable "not applicable" — it can
emit `null` — but under grammar-constrained decoding the key itself is still always
emitted; "optional" means nullable, not omittable.

## Running paw-kit's own suite in both engine configurations

paw-kit's own test suite (`.venv/bin/python -m pytest -q -p no:cacheprovider`) needs to
pass in two configurations, because grammar-constrained decoding's engine
(`llguidance`) is optional: with it importable, and with it absent (the backend then
degrades to post-hoc validation only, with one warning — see
[real-backend](./real-backend.md)). A checkout with `llguidance` already installed in
`.venv` — as this project's own dev environment has it, for local iteration on
`paw_kit.schema.constraint` — exercises the engine-present configuration by default,
so the engine-absent one has to be forced rather than assumed:

```bash
PYTHONPATH=tests/_no_llguidance .venv/bin/python -m pytest -q -p no:cacheprovider
```

`tests/_no_llguidance/sitecustomize.py` sets `sys.modules["llguidance"] = None` before
any test module imports, which is what a genuinely absent `llguidance` looks like to
every call site that checks for it (`import llguidance` and
`importlib.util.find_spec("llguidance")` both fail identically) — reproducing the
no-extras install CI's own engine-absent job runs, without uninstalling anything from
`.venv`. Tests that need the engine skip cleanly in this configuration
(`pytest.importorskip("llguidance")`); nothing should fail in either configuration.
