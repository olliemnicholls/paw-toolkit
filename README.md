# paw-kit

**A reliability and migration harness for Program-as-Weights (PAW) neural functions.**

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04199-b31b1b.svg)](https://arxiv.org/abs/2609.04199)

> **Status: early alpha (v0.1), one person, one weekend.** The harness is built and tested.
> It runs a real model only through the official upstream SDK via `ProgramAsWeightsBackend`,
> which has not yet been measured end to end. Everything else in this repo runs on a
> deterministic mock so you can try the workflow with no GPU and no API key. **No
> performance numbers are published here yet.** Real measurements on an RTX 3080 and an
> A100 are the next milestone. Until then, treat every latency or accuracy claim you might
> infer from the code as untested.

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
| Tracing decorator, SQLite trace DB, background compile, hot-swap, fail-open | Implemented, unit-tested | `MockPAWBackend` |
| `suite.yaml` runner, fuzzer, active-learning loop | Implemented, unit-tested | `MockPAWBackend` |
| Pydantic-to-regex compiler and FSM logits processor | Implemented, unit-tested | Synthetic token vocabularies only. **Never applied to a live model.** |
| HTTP server, Docker export, dataset export, CLI | Implemented, unit-tested | `MockPAWBackend` |
| `ProgramAsWeightsBackend` (official upstream SDK) | Implemented, unit-tested against a fake SDK | **Not yet run against the real service or a real model** |
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
uv run pytest -q       # ~200 tests, no GPU, no network
```

For a real backend, also install the upstream SDK and get an API key from
[programasweights.com/settings](https://programasweights.com/settings):

```bash
pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
export PAW_API_KEY=paw_sk_...
```

---

## Try the workflow with no hardware (mock backend)

```bash
uv run paw-kit demo                  # ticket triage: trace, threshold, compile, hot-swap
uv run paw-kit demo --scenario pii   # schema-validated extraction with fallback
uv run python examples/triage_ticket/run.py
uv run python examples/pii_scrubber/run.py
uv run python examples/date_normalizer/run.py
```

The same thing in code. Calls 1 to 3 run your function and are traced; call 4 triggers
compilation; call 5 onward routes to the adapter:

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
)
def triage_ticket(ticket_body: str) -> SupportTriage:
    # In real use, this body is your existing Claude/OpenAI call.
    return SupportTriage(priority="high", department="billing", urgency_score=4)

for i in range(1, 6):
    print(i, triage_ticket(f"Invoice refund needed for charge #{i}!"))
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

1. Run `ProgramAsWeightsBackend` end to end against the real service on an RTX 3080 (11GB)
   and an A100. Publish actual compile time, per-call latency, and `paw-test` pass rates on
   the three example tasks, with the exact commands used.
2. Decide from those numbers whether folding traced examples into the spec helps at all.
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
