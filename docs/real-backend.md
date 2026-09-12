# Use a real model (`ProgramAsWeightsBackend`)

`ProgramAsWeightsBackend` implements paw-kit's backend interface on top of the official
SDK. Compilation goes to the upstream service; inference runs locally through the SDK's
llama.cpp runtime (GPU if available). The `.paw` file paw-kit writes is a small JSON
manifest pointing at the upstream program ID; the weights live in the SDK's cache.

Run `paw-kit doctor` first. Most `--backend real` failures (a CPU-only `llama-cpp-python`
wheel, an un-downloaded base model, an upstream compile service that is up but has no GPU
workers behind it) are environment problems `doctor` catches up front, with a one-line
remedy.

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
when it detects this (naming the existing program, and distinctly from the case where the
check itself could not run), but it cannot change that program's visibility.

The manifest keeps the request and the fact apart. `public_requested` is what paw-kit
asked for. `public_confirmed` is what the server said when asked directly, and it is
three-state: `true`, `false`, or `null` with `public_confirmed_reason` saying why there is
no answer (not attempted, offline, no API key, the request failed, or the response carried
no visibility field). A question that was never asked reads as `null`, never as "private".
Pass `verify_visibility=True` to have `compile()` ask; it costs one extra authenticated GET
per compile, which is why it is opt-in, and no outcome of it can prevent the manifest being
written. On a cache hit the manifest also records `cached_program_id` — the existing program
you are being handed back.

Two smaller guarantees in the same area: only the literal `True` opts into a public compile
(`public=None` used to forward `null`, which upstream reads as public), and a backend built
with `offline=True` refuses to compile rather than contacting the service anyway — it is an
inference mode, so construct a second backend without it if you need to compile.

## Retries, and what a timeout means

Resending a compile the server already received queues a second compile and spends a
second unit of the rate-limited quota. So `compile()` retries only when the connection
provably never reached the server — a connect error or connect timeout — up to
`compile_retries` times with a short backoff. A read timeout, a 504, or any 4xx is raised
immediately with the response body in the message. After a read timeout, re-running the
same spec hits upstream's compile cache once the first compile finishes, instead of paying
for a second one.

A 5xx other than 504 is retried on the **synchronous** fast-compiler path only. It is
never retried for `paw-ft-bs48`: that submission goes through `compile_async`, where a
duplicate is minutes of paid GPU *and* discards the first attempt's `job_id`, leaving that
job unpollable and uncancellable. The sync retry is not free either — if the first POST
landed before its program was cached, the retry buys a second fast compile — but it is
bounded to seconds of work and there is no `job_id` to lose.

Status polling for the finetune compiler tolerates transient failures; after five
consecutive failures it gives up and raises with the `job_id` so you can poll it later. A
job that reports an error and no program is treated as terminal whatever its status string
is called, with the server's own message passed through, rather than being polled for the
full `compile_timeout_s`. If the poll loop does time out, `compile()` asks the service to
cancel the job.

## What to expect

The measured figures are on the [results page](./results.md). The shape of them:

- **Compile** takes seconds with the default fast compiler and minutes with
  `paw-ft-bs48`. On easy tasks the two produce near-identical adapters; on tasks that
  need an arbitrary mapping from the spec, only the finetune compiler learns it.
- **The first call** can take a couple of minutes if the base model and program are not
  yet in the SDK cache.
- **Steady-state latency** is tens of milliseconds on any CUDA GPU. The model is small
  enough that GPU class barely matters; GPU versus CPU is what matters.
- **Quality** is task-dependent and the thing to test, not assume. Whether the spec pins
  down the output format can swing structural pass rate from nothing to everything on
  the same task; a swapped-in adapter can disagree with its teacher on most inputs; and
  the fuzzer has found a fabricated answer. `paw-kit lint-spec` checks a spec for the
  common authoring mistakes.

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
runtime you like. paw-kit ships no in-process PyTorch backend of its own.

## Lineage

Every compile appends a line to `<adapter>.history.jsonl` beside the manifest, and the
manifest records the spec hash, the folded example ids and the parent program.
`paw-kit history a.paw` lists every past compile, oldest first; `paw-inspect a.paw` shows
the current manifest.
