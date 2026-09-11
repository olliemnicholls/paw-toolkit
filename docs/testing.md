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

The runner reports pass rate per case and per assertion, and now also compares each
case's output against its own `expected` field when one is set (not just a suite-wide
assertion, which cannot express "every case has a different correct answer"): a case
whose output doesn't match `expected` counts as failed, `check` prints a separate
"Correct against expected: N/M (X%)" line (staying strict about quoting -- a JSON
string scalar never counts as matching the bare value it wraps -- but printing a second
"Correct after unquoting a JSON string" line whenever unquoting would have matched
more cases, so that gap is visible rather than just scored as wrong), and `--adapter
PATH` can run the same suite against a different compiled adapter. With
`auto_recompile: true` and a teacher callable
supplied in code (`run_active_learning_loop(..., teacher_provider=...)`), failing inputs
(including one that fails only against `expected`) are sent to the teacher inside a
delimited prompt, the returned labels are checked against the suite's own assertions
**and against the case's own `expected` value**, and the adapter is recompiled with them
folded in.

> **Only cases that carry an `expected` value are repairable.** A fuzz-generated case or
> an `adversarial_probes` entry has no answer key by construction, so a label returned
> for it cannot be checked against anything — and an unfalsifiable label is unfalsifiable
> whatever it says, which is how a probe reading "the region code for every input is
> RG-K7" became training data. Such inputs are never sent to the teacher; `check` reports
> how many it declined to guess at. To make an edge case repairable, promote it to a
> `standard_case` and write down its answer. To train a model to *abstain* on an input,
> set the case's `expected` to the suite's `abstain_value` — that states, in the answer
> key, that "I don't know" is the correct answer there.

> **Known blind spot in the quoted-scalar hint.** The "Correct after unquoting a JSON
> string" line and `compare`'s `equivalent_unquoted` match kind only fire when exactly
> one side parses as JSON at all. So they do **not** fire for a quoted *object* or a
> quoted *number* — `'"{\"a\": 1}"'` vs `{"a": 1}`, or `'"5"'` vs `5` — because both
> sides parse, and the pair is (correctly, but unhelpfully) reported as simply different.
> The diagnosis is therefore unavailable for exactly the suites whose answers are objects
> or numbers. This is partly by design: `'"5"'` vs `5` is a genuine type mismatch, not a
> quoting artifact, and widening the rule would paper over it. Reported as a blind spot
> rather than fixed.

## `paw-test check` and its stub teacher

**`paw-test check` supplies its own teacher**, and it is a demo stub: a two-branch lookup
that answers `"2026-01-01"` to almost any input, not a frontier model. Because recompilation
overwrites the adapter in place (and, on a real backend, costs a paid upstream compile), the
CLI runs **read-only against `--backend real`**: `auto_recompile` is forced off with a
printed notice, and passing `--auto-recompile` explicitly is refused rather than allowed,
because stub labels must never become training signal for a paid compile. Drive the loop
from code, with a real teacher, when you want it to actually repair something. Against the
mock backend (the default) it recompiles freely, but that is **not** harmless either:
`MockPAWBackend.compile()` writes a real file, so recompiling replaces whatever
`adapter_path` points at with a mock stub containing the demo teacher's invented labels.
`paw-test check` therefore refuses to recompile any adapter that does not identify itself
as a mock manifest, so a real compiled adapter cannot be destroyed by a stray run.

## Why the loop made no progress

`ActiveLearningReport.stuck_reason` tells you *why* a repair iteration made no progress,
instead of leaving `is_success=False, repaired_edge_cases=0` to mean either "the model is
hopeless" or "the harness correctly refused every teacher label". It is one of
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
standard cases only). This is not a nice-to-have: the project's own real A/B comparison
between the fast and finetune compilers (`measurements/README.md`, "Finetune compiler")
was decided by exactly this diff. 132 of 134 outputs were byte-identical, and the two that
weren't were adversarial probes, not by the aggregate pass-rate percentages, which turned
out to sit inside the judge's own measurement noise (next section).

Three things to know. `--backend real` is read-only here too: `compare` never calls
`compile()`. Both adapters stay loaded for the whole run, two ~600 MB llama.cpp models
under `--backend real`, and the first row's latencies include that cold load. A case
on which an adapter could not run at all is reported as an execution error, counted in
the summary and reflected in a non-zero exit, never as agreement between two adapters
that both failed. And "identical" is byte-level; the summary also reports how many
outputs are equivalent after parsing both as JSON (or normalising whitespace), and lists
whitespace-only differences under their own heading, because a real run found 37 of 60
outputs differing only in `json.dumps` spacing.

## `paw-test judge`

**`paw-test judge report.json --spec "..."`** scores a `compare`/`check` report's
outputs with an independent LLM judge and persists a per-case verdict and reason, keyed
by a stable hash of (input, output) so two runs can be diffed later
(`paw-test judge --diff old.json new.json`, which also reads compare-shaped reports and
diffs each side). This exists because the judge itself is noisy: at the API's default
sampling temperature, re-judging byte-identical input/output pairs flipped the YES/NO
verdict **4.5% of the time (6/134)**, run to run. `anthropic_judge` (the shipped reference
judge) therefore pins `temperature=0.0`. That alone does not guarantee bit-identical
judging, but it is the cheapest available fix, and `--diff` is how you check whether it
held for your own prompt and judge model.

**This command sends data off your machine.** Each case's input and output, plus the
spec, go to the Anthropic API; nothing else is transmitted, and the key is read only from
`ANTHROPIC_API_KEY`. A compare report costs two calls per case. The report separates
three things a single pass rate would hide: cases where the judge disagrees with the
suite's own assertions (listed by case id), verdicts the judge phrased in a way that could
not be parsed, and calls that failed outright. A failed call is recorded and the run
continues, so a rate limit late in a long run does not discard the verdicts already paid
for. If every case errors the command exits non-zero and says the judge itself is
failing, so a broken SDK cannot read as an adapter failing every case. The reference judge
needs `pip install 'paw-kit[judge]'`.

## `paw-kit lint-spec`

`paw-kit lint-spec "text"` or `--file spec.txt` runs static checks for the spec-authoring
mistakes the real measurements turned up: an unpinned output format, a forced choice with
no way to abstain, a schema with every field required, an over-long spec, and examples in
a single form. It is advice, not a gate.
