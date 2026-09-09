# paw-kit

**A reliability and migration harness for Program-as-Weights (PAW) neural functions.**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04199-b31b1b.svg)](https://arxiv.org/abs/2609.04199)

> **Status: early alpha (v0.1), one person, one weekend.** The harness is built and tested.
> It runs a real model through the official upstream SDK via `ProgramAsWeightsBackend`,
> which has now been measured end to end on an RTX 3080 and an A100 (see
> [`measurements/`](./measurements)) — including a real teacher-vs-compiled-adapter
> comparison, a real grammar-constrained-decoding test against a live model, a real
> fail-open test, and a real end-to-end run of the upstream finetune compiler. A few
> numbers from the first passes of that testing were wrong (arithmetic slips, a
> cache-discarding bug in a measurement script, transcription errors) and have since
> been corrected in place, with each mistake left visible rather than quietly fixed —
> see `measurements/README.md` if you want the specifics before trusting any number in
> this repo. Everything else runs on a deterministic mock so you can try the workflow
> with no GPU and no API key. **Semantic correctness — whether the compiled adapter's
> outputs are actually right, not just fast and schema-shaped — has now been measured,
> and the honest answer is "it depends, and don't trust the exact percentage past ±2
> points":** structural-vs-semantic agreement ranged from 60% (a ticket-triage adapter
> re-scored against a fresh teacher call) to ~90% depending on task; real hallucinations
> and memorization failures were found by fuzzing, not by inspection; and the LLM judge
> used to score "semantic" itself flipped its verdict on 4.5% of cases given
> byte-identical input and output, so every semantic-pass number here carries that much
> unreported noise. Not a clean "yes it works" — see
> [`measurements/README.md`](./measurements/README.md#semantic-correctness-for-real-does-it-mean-the-right-thing-not-just-look-right)
> for the actual breakdown before repeating any percentage from this project as settled.

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
| Tracing decorator, SQLite trace DB, background compile, hot-swap, fail-open | Implemented, unit-tested, **and run for real**: a real Claude teacher + real compiled adapter (~11x steady-state latency, not yet checked for output *correctness*), plus two real induced failures (one confirmed clean fallback, one inconclusive) | `MockPAWBackend` for unit tests; real teacher + `ProgramAsWeightsBackend` for the numbers above — see [`measurements/`](./measurements) |
| `suite.yaml` runner, fuzzer, active-learning loop | Implemented, unit-tested, **and run for real** against the real 11/82 fuzzer failures below with a live Claude teacher: 0 repaired, correctly — the teacher declines to hallucinate labels the suite's own assertions would reject, surfacing a gap in the suite's assertions rather than the model | `MockPAWBackend` for unit tests; real teacher + `ProgramAsWeightsBackend` for the run above — see [`measurements/`](./measurements) |
| Pydantic-to-regex compiler and FSM logits processor | Implemented, unit-tested, **and confirmed against a real model**: 15/15 valid Pydantic parses (100%) both raw and fence-stripped, vs 0/15 raw / 11/15 (73%) fence-stripped unconstrained. **Not shipped wired into any `paw_kit` backend** — `RealPAWBackend.infer()` forwards `grammar_constraint` to a caller-supplied `runtime_executor` unmodified; applying it is the executor's job, demonstrated only in `scripts/measure_schema_real_model.py`, not in the library itself | Real generation on `Qwen2.5-0.5B-Instruct`; an initial ~13x latency-cost measurement was a caching bug in the test script (fixed) — properly measured, constrained decoding is roughly on par with unconstrained once warm; see [`measurements/`](./measurements) |
| HTTP server, Docker export, dataset export, CLI | Implemented, unit-tested | `MockPAWBackend` |
| `ProgramAsWeightsBackend` (official upstream SDK) | Implemented, unit-tested against a fake SDK, **and run end-to-end against the real service and a real model** | Real compile + inference on an RTX 3080 and an A100; see [`measurements/`](./measurements) |
| `RealPAWBackend` (in-process PyTorch/PEFT) | Stub. Raises `NotImplementedError`. | Nothing |
| `MockPAWBackend` | A dictionary lookup that returns canned strings. It is a test double, not a model. | n/a |

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
uv run pytest -q       # 217 tests, no GPU, no network, no API key
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

`paw-kit[torch]` is a separate, optional extra pulling PyTorch/transformers. It serves
`RealPAWBackend` (a stub) and the standalone measurement scripts — it is **not** what you
want for a working real backend, and it used to be what `[real]` installed.

---

## Try the workflow with no hardware (mock backend)

```bash
uv run paw-kit demo                  # ticket triage: trace, threshold, compile, hot-swap
uv run paw-kit demo --scenario pii   # schema-validated extraction with fallback
uv run python examples/triage_ticket/run.py
uv run python examples/pii_scrubber/run.py
uv run python examples/date_normalizer/run.py
```

The same thing in code. Compilation triggers at the end of call 3 (`threshold=3`); call 4
onward *may* route to the adapter, once the (asynchronous, by default) compile finishes —
matching what `paw-kit demo` itself prints (try it, it's the same threshold). The status
is printed explicitly below rather than left for you to infer from the returned value,
because with `MockPAWBackend` — a literal-input-match lookup, not a real model — a
teacher call and a hot-swapped call can return an identical-looking value for reasons
that have nothing to do with whether the hot-swap actually happened, and eyeballing
output values is exactly how that distinction silently went missing from this example
before (caught by review, 2026-09-08: this block previously varied the ticket text on
every call, which a literal-match mock can never generalize across, so it was silently
falling back to the teacher on *every* call via the decorator's own fail-open path — the
"hot-swap" that block claimed to demonstrate was never actually happening):

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

ticket = "Invoice refund needed for charge #1!"  # same input every call, on purpose --
# MockPAWBackend only ever matches input it has seen verbatim before; varying the text
# per call (as this example used to) means it can never match, and every call silently
# falls back to `triage_ticket`'s own body via fail-open, whether or not compilation
# actually finished. That's a real property of MockPAWBackend worth knowing, not a
# demo-only quirk -- your own inputs will vary at threshold, and MockPAWBackend does
# not generalize across them either.
for i in range(1, 6):
    served_by = triage_ticket.db.get_status(triage_ticket.task_id)  # "tracing" / "compiling" / "ready"
    print(i, served_by, triage_ticket(ticket))
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

Two limitations come straight from the upstream API and are worth knowing before you
plan around them:

1. **The upstream compiler takes a spec, not a dataset.** It generates its own examples
   with teacher models. paw-kit's traced calls and active-learning labels can only reach it
   as few-shot demonstrations appended to the spec text (`max_spec_examples`). Whether that
   helps is exactly the kind of question `paw-test` is for. It is not assumed.
2. **No grammar-constrained decoding.** The SDK's callable has no grammar or logits hook, so
   the FSM logits processor in `paw.schema` cannot be applied. `paw.load` validates output
   after generation with Pydantic and falls back on failure. The processor is kept for a
   future in-process backend (llama.cpp itself supports GBNF grammars; wiring that through
   the SDK is upstream work, not something this repo can do alone).

`RealPAWBackend` is the placeholder for an in-process HuggingFace/PEFT path. It raises
`NotImplementedError` today. It exists so the interface is settled, not because it works.

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
paw-test check suite.yaml             run a suite (--backend real currently falls back to mock; see roadmap)
paw-inspect adapter.paw               show an adapter manifest
paw-clean [--dry-run]                 remove cached adapters and trace DB
paw-serve adapter.paw --port 8000     HTTP server
paw-kit export docker|dataset ...     scaffolding and trace export
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
3. Wire `--backend real` in the CLI to `ProgramAsWeightsBackend`.
4. Either implement `RealPAWBackend` (in-process PEFT on Qwen3-0.6B, where the logits
   processor can actually be applied) or delete it.

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
