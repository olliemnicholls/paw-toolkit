# paw-kit

**A reliability and migration harness for Program-as-Weights (PAW) neural functions.**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04199-b31b1b.svg)](https://arxiv.org/abs/2609.04199)

> **Status: early alpha (v0.1), one person, one weekend.** The harness — tracing
> decorator, test runner, schema validation, HTTP server — is built and unit-tested, and
> its one real backend, `ProgramAsWeightsBackend`, has been run end to end against the
> upstream service on an RTX 3080 and an A100. What that showed, in one paragraph: a
> compiled date normaliser answers in ~65 ms on the 3080 (~6 s on CPU) and passes 71/82
> of its own suite; a compiled ticket-triage adapter replaced a live Claude teacher at
> ~11x lower latency and zero tokens billed, but agreed with a fresh teacher call on only
> 60% of tickets; semantic correctness across four tasks ranged from 60% to ~90%, with a
> ±2-point noise floor from the LLM judge itself. Everything that is not labelled
> "measured" runs on a deterministic mock — a dictionary lookup, not a model — so the
> workflow can be tried with no GPU and no API key. Several first-pass numbers were wrong
> and are corrected in place, visibly, in
> [`measurements/README.md`](./measurements/README.md); read it before repeating any
> figure from this repo.

---

## What PAW is, in three sentences

[Deng, Nie and Shieber (2026)](https://arxiv.org/abs/2609.04199) compile a natural-language
specification into a small LoRA adapter for a frozen 0.6B interpreter (Qwen3-0.6B). Teacher
models generate the training examples at compile time, so after roughly a minute you have a
"neural function" that runs locally with no teacher in the loop. The authors publish an
official Python SDK, [`programasweights`](https://github.com/programasweights/programasweights-python),
that compiles on their service and runs the result on your machine with llama.cpp.

## What paw-kit adds

The upstream SDK gives you `compile(spec)` and a callable. paw-kit is the layer around
that for people who want to swap a compiled function into a codebase that currently calls
a frontier API, and who want evidence before they trust it:

- **`@compile_on_hit` (paw.jit)**: wrap the function that currently calls your LLM. It
  keeps calling it, logs every input/output pair to a local SQLite trace database, and
  once a call-count threshold is reached it triggers compilation in a background thread
  and routes later calls to the compiled adapter. Any exception or schema violation on the
  local path falls back to the original function (fail-open).
- **`paw-test` (paw.test)**: a declarative `suite.yaml` of standard cases and assertions,
  an adversarial fuzzer (Unicode injection, whitespace floods, payload extremes, your own
  probes), and an active-learning loop that sends failing inputs to a teacher for labels
  and recompiles. Use it to find out what a compiled function gets wrong before you ship it.
- **`paw.load` (paw.schema)**: bind an adapter to a Pydantic model. Output is validated and,
  on failure, routed to a fallback. Includes a regex-to-FSM logits processor for
  token-level constrained decoding, but see the caveat below: no current backend can apply it.
- **`paw-serve`**: expose any adapter as a local HTTP service speaking the OpenAI Chat
  Completions and Anthropic Messages wire formats, so non-Python clients can call it.
  `paw-kit export docker` generates a Dockerfile and compose file for it.

### What is real and what is mocked

| Component | State | Exercised against |
|---|---|---|
| Tracing decorator, SQLite trace DB, background compile, hot-swap, fail-open | Implemented, unit-tested, **and run for real**: a live Claude teacher hot-swapped to a real compiled adapter (~11x lower steady-state latency, zero tokens billed afterwards; 60% full agreement with a fresh teacher call on the same tickets). One real induced failure (adapter file deleted) fell open cleanly; a second (bad API key) was inconclusive because the service accepted the key | `MockPAWBackend` for unit tests; real teacher + `ProgramAsWeightsBackend` for the numbers — [`measurements/`](./measurements) |
| `suite.yaml` runner, fuzzer, active-learning loop | Implemented, unit-tested, **and run for real** against 11 real fuzzer failures with a live Claude teacher: **0 repaired**, and correctly so — every teacher label failed the suite's own assertions (the suite had no "not a date" case), so the loop refused to train on them. The loop is bounded and best-effort, not a guarantee | Same as above |
| Pydantic-to-regex compiler and FSM logits processor | Implemented, unit-tested, **confirmed against a real model** (15/15 valid Pydantic parses vs 0/15 raw and 11/15 fence-stripped unconstrained; no latency cost once warm), and driven once against a real upstream adapter through a **private** SDK attribute (4/5 structurally valid; the constrained value of one field was constant regardless of input; ~1.7 s warm-up per new FSM state, so 47 s on the first call). **No shipped backend applies it.** `paw.load` validates after generation instead | `Qwen2.5-0.5B-Instruct` via transformers; one upstream adapter via a monkeypatched llama.cpp sampler. Both are measurement scripts, not shipped code — [`measurements/`](./measurements) |
| HTTP server, Docker export, dataset export, CLI | Implemented, unit-tested | `MockPAWBackend` only |
| `ProgramAsWeightsBackend` (official upstream SDK) | Implemented, unit-tested against a fake SDK, **and run end to end against the real service** with both upstream compilers | Real compile + inference on an RTX 3080 (CPU and CUDA) and an A100 — [`measurements/`](./measurements) |
| `MockPAWBackend` | A dictionary lookup that returns canned strings. It is a test double, not a model | n/a |

The demo command and the three examples all run on the mock. When they print "local
adapter" they mean the dictionary lookup. They demonstrate the *control flow* of the
harness, nothing about model quality or speed.

---

## Install

Not on PyPI yet. From source:

```bash
git clone https://github.com/olliemnicholls/paw-toolkit
cd paw-toolkit
uv sync --dev          # or: pip install -e .
uv run pytest -q       # 230 tests, no GPU, no network, no API key
```

For a real backend, install the `real` extra and get an API key from
[programasweights.com/settings](https://programasweights.com/settings):

```bash
uv sync --extra real   # or: pip install 'paw-kit[real]'
export PAW_API_KEY=paw_sk_...
```

`paw-kit[real]` pulls the official upstream SDK, which is what
`ProgramAsWeightsBackend` runs on. It resolves from PyPI directly — the
`--extra-index-url` in upstream's own README is not needed (verified 2026-09-09 against
`programasweights==0.4.4`). No API key is required to *run* an already-compiled program;
only to compile a new one.

Two things to know before the first real call: the `llama-cpp-python` wheel this pulls
from PyPI is **CPU-only** — on this repo's date normaliser that meant ~5.9 s per call
against ~65 ms once a CUDA build was in place (the build notes are at the end of
[`measurements/README.md`](./measurements/README.md#if-inference-is-unexpectedly-slow-seconds-not-milliseconds));
and the first call to any program downloads the ~600 MB base model into the SDK's cache.

`paw-kit[measure]` is a separate, optional extra pulling PyTorch/transformers. It exists
only to reproduce `scripts/measure_schema_real_model.py`, which backs the
constrained-decoding numbers in [`measurements/`](./measurements). It is **not** a backend
and buys you no inference. (It was called `[torch]`, and before that it was what `[real]`
installed — both were misleading, so it is now named for what it actually does.)

---

## Try the workflow with no hardware (mock backend)

```bash
uv run paw-kit demo                  # ticket triage: trace, threshold, compile, hot-swap
uv run paw-kit demo --scenario pii   # schema-validated extraction with fallback
uv run python examples/triage_ticket/run.py
uv run python examples/pii_scrubber/run.py
uv run python examples/date_normalizer/run.py
```

The same thing in code. Compilation triggers at the end of call 3 (`threshold=3`); from
call 4 the decorator routes to the adapter once the compile (asynchronous by default) has
finished. The status is printed explicitly rather than inferred from the returned value:
with `MockPAWBackend` a teacher call and a hot-swapped call return identical-looking
values whether or not the swap happened. (An earlier version of this block varied the
ticket text per call, which a literal-match mock can never match, so every call was
silently falling back to the teacher through the decorator's fail-open path — caught in
review 2026-09-08. `wrapper.get_fail_open_count()` now exists so you can check.)

```python
from pydantic import BaseModel
from paw_kit import MockPAWBackend, compile_on_hit

class SupportTriage(BaseModel):
    priority: str
    department: str
    urgency_score: int

@compile_on_hit(
    spec="Classify a customer inquiry into priority, department, and urgency score 1-5.",
    threshold=3,
    response_model=SupportTriage,
    backend=MockPAWBackend(),   # explicit. Omitting backend= also gives you the mock, with a warning.
    cache_dir="./.paw",
    sync_compile=True,  # blocks call 3 until compilation finishes, so this 5-call demo
                         # reaches "ready" deterministically. Compilation is asynchronous
                         # by default (the point of it, in production, is that threshold
                         # calls stay fast) -- drop this in real use.
)
def triage_ticket(ticket_body: str) -> SupportTriage:
    # In real use, this body is your existing Claude/OpenAI call.
    return SupportTriage(priority="high", department="billing", urgency_score=4)

ticket = "Invoice refund needed for charge #1!"  # same input every call, on purpose:
# MockPAWBackend only matches input it has seen verbatim. Vary the text and every
# post-threshold call falls open to `triage_ticket`'s own body, silently.
for i in range(1, 6):
    served_by = triage_ticket.db.get_status(triage_ticket.task_id)  # "tracing" / "compiling" / "ready"
    print(i, served_by, triage_ticket(ticket), triage_ticket.get_fail_open_count())
```

---

## Use a real model (ProgramAsWeightsBackend)

`ProgramAsWeightsBackend` implements paw-kit's backend interface on top of the official
SDK. Compilation goes to the upstream service; inference runs locally through the SDK's
llama.cpp runtime (GPU if available). The `.paw` file paw-kit writes is a small JSON
manifest pointing at the upstream program ID; the weights live in the SDK's cache.

```python
from pydantic import BaseModel
import paw_kit as paw
from paw_kit import ProgramAsWeightsBackend, compile_on_hit

backend = ProgramAsWeightsBackend(
    compiler="paw-4b-qwen3-0.6b",   # upstream default, seconds. "paw-ft-bs48" = the finetune
                                    # compiler from the paper, queued and polled (minutes).
    n_gpu_layers=None,              # SDK decides; 0 forces CPU
    max_spec_examples=16,           # how many traced examples to fold into the spec text
)

class Triage(BaseModel):
    priority: str
    department: str
    urgency_score: int

@compile_on_hit(spec="...", threshold=50, response_model=Triage, backend=backend, cache_dir="./.paw")
def triage_ticket(body: str) -> Triage:
    return call_your_llm(body)      # traced until threshold, then compiled and replaced
```

**Compiles are private by default.** Upstream `paw.compile`/`paw.compile_async` default to
`public=True`, which lists the compiled program on programasweights.com with its full spec
text readable by anyone, no login required. `ProgramAsWeightsBackend` passes `public=False`
unless you opt in. The spec that gets uploaded is not just what you wrote in `spec=` --
it has up to `max_spec_examples` traced input/output pairs folded into it, so a public
compile publishes a sample of real production traffic; think about `redact_trace=True`
on the decorator if you do set `public=True`. Note also that upstream's compile cache is
keyed on the spec text and ignores `public` on a cache hit, so recompiling a spec that was
previously compiled public will return that same public program regardless of what you pass
this time -- `compile()` warns when it detects this via `precheck_compile`, but it can't
change the existing program's visibility. Versions of paw-kit before this change compiled
publicly by default; if you compiled anything with an earlier version, check
programasweights.com for it.

What to expect, from the runs in [`measurements/`](./measurements) (one machine each, one
run each — indicative, not a benchmark):

- **Compile**: 1–5 s wall time with the default fast compiler; ~3 min with `paw-ft-bs48`.
  On one phone-extraction task the two produced byte-identical output on 132 of 134 inputs.
- **First call**: 2 s to ~110 s, depending on whether the base model and program are
  already in the SDK cache.
- **Steady state**: ~65 ms per call on an RTX 3080, ~89 ms on a shared A100, ~5.9 s on
  the CPU-only PyPI wheel. The model is small enough that GPU class barely matters; GPU
  versus CPU matters ~90x.
- **Quality**: task-dependent and the thing to test, not assume. Structural pass rates of
  0% to 100% on the same task depending on whether the spec pins down the output format;
  60% full agreement with a fresh teacher call on ticket triage; one clear fabricated
  answer (`1-800-FLOWERS` → invented digits) found by the fuzzer.

Two limitations come straight from the upstream API and are worth knowing before you
plan around them:

1. **The upstream compiler takes a spec, not a dataset.** It generates its own examples
   with teacher models. paw-kit's traced calls and active-learning labels can only reach it
   as few-shot demonstrations appended to the spec text (`max_spec_examples`). Whether that
   helps is exactly the kind of question `paw-test` is for. It is not assumed.
2. **No grammar-constrained decoding.** The SDK's callable has no grammar or logits hook, so
   the FSM logits processor in `paw.schema` cannot be applied. `paw.load` validates output
   after generation with Pydantic and falls back on failure. Whether the processor stays in
   this package at all is an open question, pending upstream: `llama_cpp.Llama.sample()`
   already accepts `grammar` and `logits_processor`, and the SDK's decode loop already
   calls it, so the ask is a passthrough rather than new machinery. See roadmap item 5.
   It is **not** being kept for a future in-process backend — there isn't going to be one.

**Bringing your own runtime.** `AbstractPAWBackend` is three methods — `compile`, `infer`,
`is_available`. Implement them and pass `backend=` to `paw.load` or `@compile_on_hit`, and
paw-kit will drive whatever runtime you like. There is no in-process PyTorch/PEFT backend in
this package and there is not going to be one: paw-kit is a toolkit around upstream PAW, not
a reimplementation of it. (A `RealPAWBackend` placeholder existed through v0.1 and raised
`NotImplementedError`; it was deleted rather than built, because shipping a class that looks
like a working backend and is not is the exact confusion this project spent a track
removing.)

---

## Test a compiled function before trusting it

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

The runner reports pass rate per case and per assertion. With `auto_recompile: true` and a
teacher callable supplied in code (`run_active_learning_loop(..., teacher_provider=...)`),
failing inputs are sent to the teacher inside a delimited prompt, the returned labels are
checked against the suite's own assertions before being trusted, and the adapter is
recompiled with them folded in.

**`paw-test check` supplies its own teacher**, and it is a demo stub — a two-branch lookup
that answers `"2026-01-01"` to almost any input, not a frontier model. Because recompilation
overwrites the adapter in place (and, on a real backend, costs a paid upstream compile), the
CLI runs **read-only against `--backend real`**: `auto_recompile` is forced off with a
printed notice, and passing `--auto-recompile` explicitly is refused rather than allowed,
because stub labels must never become training signal for a paid compile. Drive the loop
from code, with a real teacher, when you want it to actually repair something. Against the mock
backend (the default) it recompiles freely, but that is **not** harmless either:
`MockPAWBackend.compile()` writes a real file, so recompiling replaces whatever
`adapter_path` points at with a mock stub containing the demo teacher's invented labels.
`paw-test check` therefore refuses to recompile any adapter that does not identify itself
as a mock manifest, so a real compiled adapter cannot be destroyed by a stray run.

---

## Serve over HTTP

```bash
uv run paw-serve .paw/triage.paw --port 8000        # auth on by default; see --help
curl -X POST localhost:8000/invoke -H 'Content-Type: application/json' -d '{"input": "Outage in eu-west"}'
```

`POST /v1/chat/completions` (OpenAI shape) and `POST /v1/messages` (Anthropic shape) are
also exposed, so the official OpenAI and Anthropic client libraries work with
`baseURL` pointed at the server. Only `GET /health` is unauthenticated; `GET /metrics` and
every inference route require the bearer token.

```bash
uv run paw-kit export docker .paw/triage.paw --out-dir ./docker   # Dockerfile + compose
uv run paw-kit export dataset --db ./.paw/traces.db --out traces.jsonl
```

## CLI

```
paw-kit demo [--scenario pii]         mock-backend walkthroughs
paw-test check suite.yaml             run a suite (--backend real runs the upstream SDK, read-only)
paw-inspect adapter.paw               show an adapter manifest
paw-kit history adapter.paw           show every past compile of an adapter, oldest first
paw-clean [--dry-run]                 remove cached adapters and trace DB
paw-serve adapter.paw --port 8000     HTTP server
paw-kit export docker|dataset ...     scaffolding and trace export
paw-kit lint-spec "text"|--file f.txt static checks for spec-authoring mistakes measured against real adapters
```

---

## Roadmap

In order, and nothing gets announced until the first item is done:

1. ~~Run `ProgramAsWeightsBackend` end to end against the real service on an RTX 3080
   (11GB) and an A100. Publish actual compile time, per-call latency, and `paw-test`
   pass rates on the three example tasks, with the exact commands used.~~ **Done** —
   see [`measurements/`](./measurements), plus real JIT hot-swap, grammar-constrained
   decoding, fail-open, semantic-correctness, and finetune-compiler tests that went
   beyond the original scope of this item.
2. ~~Decide from those numbers whether folding traced examples into the spec helps at
   all.~~ **Done, and the honest answer is "it depends on the failure mode."** A real
   A/B test (`measurements/README.md#does-folding-examples-into-the-spec-text-actually-help-a-real-answer-on-the-second-try`)
   found folding examples in fixed format-ambiguity failures dramatically (one task
   went from 0% to 92.5% structural pass) and did nothing for failures unrelated to
   format (unicode-handling edge cases) — and introduced a new failure mode of its own
   (verbatim memorization of an example for out-of-distribution input). Not a flat
   yes/no; read the section before deciding whether to fold examples in for your task.
3. ~~Wire `--backend real` in the CLI to `ProgramAsWeightsBackend`.~~ **Done** — it
   resolves to the upstream SDK, announces any fallback to the mock, and refuses to
   recompile (a paid, destructive operation) unless asked explicitly. See `paw_kit/cli.py`.
4. ~~Decide `RealPAWBackend`'s fate.~~ **Done — deleted.** It was the placeholder for an
   in-process PEFT path, justified mainly as "the only place the logits processor could
   ever be applied", which [`measurements/`](./measurements) showed to be false:
   constrained decoding reaches the real upstream adapter through llama.cpp's own
   sampling loop. paw-kit wraps upstream PAW rather than reimplementing it, so the
   placeholder was removed instead of built.
5. Ask upstream for a supported `grammar` / `logits_processor` passthrough on
   `PawFunction.__call__`. `llama_cpp.Llama.sample()` already accepts both; the SDK's
   decode loop already calls it. Until there is an answer, **nothing in this package
   applies constrained decoding** — `paw.load` validates after generation and falls back
   on failure. Whether `RegexLogitsProcessor` stays here at all depends on that answer.

## Relationship to upstream

This is an independent project and is not affiliated with the paper's authors or
programasweights.com. It depends on their SDK and service for anything real. If you want
to compile and run PAW functions, start with
[their SDK](https://github.com/programasweights/programasweights-python); come back here
when you want tracing, testing, fallback, or an HTTP front.

## Citation

```bibtex
@article{deng2026compile,
  title={Compile by Training: Turning Natural-Language Specifications into Local Neural Functions},
  author={Deng, Yuntian and Nie, Pengyu and Shieber, Stuart},
  journal={arXiv preprint arXiv:2609.04199},
  year={2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
