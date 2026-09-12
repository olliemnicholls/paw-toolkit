# Use a real model (`ProgramAsWeightsBackend`)

`ProgramAsWeightsBackend` implements paw-kit's backend interface on top of the official
SDK. Compilation goes to the upstream service; inference runs locally through the SDK's
llama.cpp runtime (GPU if available). The `.paw` file paw-kit writes is a small JSON
manifest pointing at the upstream program ID; the weights live in the SDK's cache.

Run `paw-kit doctor` first: most `--backend real` failures (a CPU-only `llama-cpp-python`
wheel, an un-downloaded base model, an upstream compile service that returns a healthy
`200` with no GPU workers behind it) are environment problems `doctor` catches up front,
with a one-line remedy, rather than a confusing failure deep inside `compile()`/`infer()`.

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
    return call_your_llm(body)      # traced until threshold, then compiled, shadowed against
                                    # the teacher, and only promoted once it agrees
```

With the defaults this block does **not** hot-swap at call 50. Call 50 triggers the
compile; the task then enters `shadow`, and the adapter takes over only after a further
window of calls (20 by default) on which it agrees with your function at least 80% of the
time. See [Shadow mode](./shadow-mode.md) for the parameters and how to watch it.

## Compiles are private by default

Upstream `paw.compile`/`paw.compile_async` default to `public=True`, which lists the
compiled program on programasweights.com with its full spec text readable by anyone, no
login required. `ProgramAsWeightsBackend` passes `public=False` unless you opt in. The spec
that gets uploaded is not just what you wrote in `spec=`: it has up to `max_spec_examples`
traced input/output pairs folded into it, so a public compile publishes a sample of real
production traffic. Think about `redact_trace=True` on the decorator if you do set
`public=True`. Note also that upstream's compile cache is keyed on the spec text and
ignores `public` on a cache hit, so recompiling a spec that was previously compiled public
returns that same public program regardless of what you pass this time. `compile()` warns
when it detects this, but it cannot change the existing program's visibility. The
manifest's `public` field records what was requested, not what the server confirmed; if
it matters, check the program on programasweights.com.

## Retries, and what a timeout means

A compile submission is a POST that is not idempotent: if the server received it, it is
compiling, and sending it again queues a second compile and spends a second unit of the
rate-limited quota. So `compile()` retries only when the connection provably never
reached the server (a connect error or connect timeout) or on a 5xx other than 504, up to
`compile_retries` times with a short backoff. A read timeout, a 504, or any 4xx is raised
immediately with the response body in the message. After a read timeout, re-running the
same spec hits upstream's compile cache once the first compile finishes, instead of
paying for a second one. Status polling for the finetune compiler is idempotent and does
tolerate transient failures; after five consecutive failures it raises with the `job_id`
so you can poll it later.

## What to expect

From the runs in [`measurements/`](../measurements/README.md), one machine each, one run
each. Indicative, not a benchmark.

- **Compile**: 1–5 s wall time with the default fast compiler; ~3 min with `paw-ft-bs48`.
  On easy tasks the two produce near-identical adapters; on tasks that need an arbitrary
  mapping from the spec, only the finetune compiler learns it.
- **First call**: 2 s to ~110 s, depending on whether the base model and program are
  already in the SDK cache.
- **Steady state**: ~65 ms per call on an RTX 3080, ~89 ms on a shared A100, ~5.9 s on
  the CPU-only PyPI wheel. The model is small enough that GPU class barely matters; GPU
  versus CPU matters ~90x.
- **Quality**: task-dependent and the thing to test, not assume. Structural pass rates of
  0% to 100% on the same task depending on whether the spec pins down the output format;
  under half of held-out tickets in full agreement with a fresh teacher call on ticket
  triage; one clear fabricated answer (`1-800-FLOWERS` → invented digits) found by the
  fuzzer. `paw-kit lint-spec` checks a spec for the authoring mistakes those runs turned
  up. The numbers are on the [results page](./results.md).

## Two upstream limitations

1. **The upstream compiler takes a spec, not a dataset.** It generates its own examples
   with teacher models. paw-kit's traced calls and active-learning labels can only reach it
   as few-shot demonstrations appended to the spec text (`max_spec_examples`). Whether that
   helps depends on the task, and `paw-test compare` is how to find out.
2. **No grammar-constrained decoding.** The SDK's callable has no grammar or logits hook,
   so paw-kit cannot constrain generation to a schema. `paw.load` validates output after
   generation with Pydantic and falls back on failure.

## Bringing your own runtime

`AbstractPAWBackend` is three methods: `compile`, `infer`, `is_available`. Implement them
and pass `backend=` to `paw.load` or `@compile_on_hit`, and paw-kit will drive whatever
runtime you like. paw-kit ships no in-process PyTorch backend of its own; it is a toolkit
around upstream PAW, not a reimplementation of it.

## Lineage

Every compile appends a line to `<adapter>.history.jsonl` beside the manifest, and the
manifest records the spec hash, the folded example ids and the parent program.
`paw-kit history a.paw` lists every past compile, oldest first; `paw-inspect a.paw` shows
the current manifest.
