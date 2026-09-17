# paw-kit

**PAW compiles a spec into a tiny local model. paw-kit tells you whether you can trust the model you just made.**

A reliability and migration harness for Program-as-Weights (PAW) neural functions.

[![CI](https://github.com/olliemnicholls/paw-toolkit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/olliemnicholls/paw-toolkit/actions/workflows/ci.yml?query=branch%3Amain)
[![PyPI](https://img.shields.io/pypi/v/paw-kit.svg)](https://pypi.org/project/paw-kit/)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2609.04199-b31b1b.svg)](https://arxiv.org/abs/2609.04199)

> **Status: early alpha, one maintainer.** The real backend has been run end
> to end against the upstream service. The demo and examples run on a deterministic mock,
> so the workflow can be tried with no GPU and no API key. Every measured number is on the
> [results page](./docs/results.md).

## What PAW is

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
  keeps calling it and logs every input/output pair to a local SQLite trace database.
  Once a call-count threshold is reached it compiles in a background thread and enters
  [*shadow mode*](./docs/shadow-mode.md): the adapter runs on every input off the request
  path and its answer is compared against the teacher's, but the teacher keeps serving.
  Only when agreement over a window of real inputs clears a threshold does the adapter
  take over; optionally, a sampled fraction of calls still runs the teacher afterwards so
  drift stays measurable. Any exception or schema violation on the local path falls back
  to the original function (fail-open).
- **`paw-test` (paw.test)**: a declarative `suite.yaml` of standard cases and assertions,
  an adversarial fuzzer, an active-learning loop that sends failing inputs to a teacher
  for labels and recompiles, a per-case diff of two adapters, and an LLM judge. Use it to 
  find out what a compiled function gets wrong before you ship it.
- **`paw.load` (paw.schema)**: bind an adapter to a Pydantic model. On the real backend
  every generation step is masked to your schema by default, so the output parses — a
  shape guarantee, not a correctness one, and `constrained_decoding=False` turns it off.
  Output is validated after generation either way and, on failure, routed to a fallback.
- **`paw-serve`**: expose any adapter as a local HTTP service speaking the OpenAI Chat
  Completions and Anthropic Messages wire formats. `paw-kit export docker` scaffolds a
  container for it.

The line between what is measured and what is mocked is drawn in
[`docs/what-is-real.md`](./docs/what-is-real.md).

## Install

```bash
pip install paw-kit      # or: uv add paw-kit
paw-kit demo             # no GPU, no network, no API key
```

For a real backend: `pip install 'paw-kit[real]'`, set `PAW_API_KEY`, and run `paw-kit doctor`.
The PyPI `llama-cpp-python` wheel is CPU-only, which costs ~90x in latency; the first call
downloads a ~600 MB base model. Details in [`docs/install.md`](./docs/install.md).

## Try the workflow with no hardware

```bash
paw-kit demo                  # ticket triage: trace, threshold, compile, hot-swap
paw-kit demo --scenario pii   # schema-validated extraction with fallback
```

The longer worked examples live in the repository rather than the package, so they need a
clone: `python examples/triage_ticket/run.py`.

The same thing in code. Compilation triggers at the end of call 3; from call 4 the
decorator routes to the adapter. The status is printed explicitly because, with the mock,
a teacher call and a swapped call return identical-looking values.

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
    sync_compile=True,  # blocks call 3 until compilation finishes so this 5-call demo
                        # reaches "ready" deterministically. Drop this in real use.
    shadow_window=0,    # shadow mode is on by default; five calls cannot fill an agreement
                        # window, so turn it off here to keep the swap deterministic.
)
def triage_ticket(ticket_body: str) -> SupportTriage:
    # In real use, this body is your existing Claude/OpenAI call.
    return SupportTriage(priority="high", department="billing", urgency_score=4)

ticket = "Invoice refund needed for charge #1!"  # same input every call, on purpose:
# MockPAWBackend only matches input it has seen verbatim. Vary the text and every
# post-threshold call falls open to `triage_ticket`'s own body, with a logged warning.
for i in range(1, 6):
    state = triage_ticket.db.get_status(triage_ticket.task_id)  # "tracing" / "compiling" / "shadow" / "ready" / "failed"
    print(i, state, triage_ticket(ticket), triage_ticket.get_fail_open_count())
```

## Going further

- [Use a real model](./docs/real-backend.md): `ProgramAsWeightsBackend`, private-by-default
  compiles, what a timeout means, what to expect, upstream limitations.
- [Shadow mode](./docs/shadow-mode.md): how the adapter earns production traffic, every
  parameter and default, what is stored, `paw-kit report`.
- [Test before trusting](./docs/testing.md): suites, fuzzing, active learning,
  `paw-test compare`, `paw-test judge`, `paw-kit lint-spec`.
- [Serve over HTTP](./docs/serving.md): `paw-serve`, `/ready`, Docker export.
- [Results](./docs/results.md): every measured number, one page.
- [What is real and what is mocked](./docs/what-is-real.md).

`scripts/` and `measurements/` hold the scripts and raw output behind the results page;
`tools/` is the project's own test tooling. None of it is needed to use paw-kit.

## CLI

```
paw-kit demo [--scenario pii]             mock-backend walkthroughs
paw-kit doctor [--adapter a.paw]          diagnose the local environment for --backend real
paw-kit report [--task id] [--json]       task state, shadow-mode agreement, disagreements
paw-kit history adapter.paw               every past compile of an adapter, oldest first
paw-kit lint-spec "text"|--file f.txt     static checks for spec-authoring mistakes
paw-kit export docker|dataset ...         container scaffold (--backend mock|real), trace export
paw-test check suite.yaml                 run a suite (--backend real is read-only)
paw-test compare A.paw B.paw suite.yaml   diff two adapters' outputs, per case
paw-test judge report.json --spec ".."    score a report with an LLM judge (sends data to Anthropic)
paw-inspect adapter.paw                   show an adapter manifest
paw-serve adapter.paw --port 8000         HTTP server (--warm to pay the cold load before binding)
paw-clean [--dry-run]                     remove cached adapters and trace DB
```

## Contributing

Changing paw-kit's own code? See [CONTRIBUTING.md](CONTRIBUTING.md) for dev-process
notes (e.g. running the test suite in both engine configurations).

## Relationship to upstream

This is an independent project and is not affiliated with the paper's authors or
programasweights.com. It depends on their SDK and service to compile and run PAW
functions. If you only want to compile and call one, start with
[their SDK](https://github.com/programasweights/programasweights-python); come back here
when you want tracing, testing, fallback, or an HTTP front.

## Citing the paper

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
