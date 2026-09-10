# What is real and what is mocked

Everything in paw-kit that is not labelled "measured" runs on a deterministic mock: a
dictionary lookup, not a model. This page says which is which. The numbers come from
[`measurements/`](../measurements/README.md), one machine each, one run each; several
first-pass figures were wrong and are corrected in place there, so read it before
repeating any figure from this repo.

| Component | State | Exercised against |
|---|---|---|
| Tracing decorator, SQLite trace DB, background compile, shadow mode, hot-swap, fail-open | Implemented, unit-tested, **and run for real** (before shadow mode existed): a live Claude teacher hot-swapped to a real compiled adapter (~11x lower steady-state latency, zero tokens billed afterwards; 60% full agreement with a fresh teacher call on the same tickets). One real induced failure (adapter file deleted) fell open cleanly; a second (bad API key) was inconclusive because the service accepted the key. Shadow mode itself is unit-tested only; measuring it is a follow-up | `MockPAWBackend` for unit tests; real teacher + `ProgramAsWeightsBackend` for the numbers |
| `suite.yaml` runner, fuzzer, active-learning loop | Implemented, unit-tested, **and run for real** against 11 real fuzzer failures with a live Claude teacher: **0 repaired**, and correctly so: every teacher label failed the suite's own assertions (the suite had no "not a date" case), so the loop refused to train on them. The loop is bounded and best-effort, not a guarantee | Same as above |
| `paw-test compare` and `paw-test judge` | Implemented, unit-tested, and lifted from the measurement scripts that produced the finetune-compiler A/B and the judge-noise figures | Real adapters via the scripts; mock in tests |
| Pydantic-to-regex compiler and FSM logits processor | Implemented, unit-tested, **confirmed against a real model** (15/15 valid Pydantic parses vs 0/15 raw and 11/15 fence-stripped unconstrained; no latency cost once warm), and driven once against a real upstream adapter through a **private** SDK attribute (4/5 structurally valid; the constrained value of one field was constant regardless of input; ~1.7 s warm-up per new FSM state, so 47 s on the first call). **No shipped backend applies it.** `paw.load` validates after generation instead | `Qwen2.5-0.5B-Instruct` via transformers; one upstream adapter via a monkeypatched llama.cpp sampler. Both are measurement scripts, not shipped code |
| HTTP server, Docker export, dataset export, `doctor`, CLI | Implemented, unit-tested | `MockPAWBackend` only |
| `ProgramAsWeightsBackend` (official upstream SDK) | Implemented, unit-tested against a fake SDK, **and run end to end against the real service** with both upstream compilers | Real compile + inference on an RTX 3080 (CPU and CUDA) and an A100 |
| `MockPAWBackend` | A dictionary lookup that returns canned strings. It is a test double, not a model | n/a |

The demo command and the three examples all run on the mock. When they print "local
adapter" they mean the dictionary lookup. They demonstrate the *control flow* of the
harness, nothing about model quality or speed.

The tracing-decorator row deserves one more sentence. The 60%-agreement adapter it
mentions was hot-swapped into service, because at the time the decorator swapped as soon
as a compile finished. [Shadow mode](./shadow-mode.md) is the mechanism that now sits in
front of that swap: with the shipped defaults (`shadow_threshold=0.8`) that adapter would
have stayed in `shadow` and the teacher would have kept serving.
