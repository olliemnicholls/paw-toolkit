# What is real and what is mocked

Everything in paw-kit that is not labelled "measured" runs on a deterministic mock: a
dictionary lookup, not a model. This page says which is which. The numbers behind the
"measured" entries are on the [results page](./results.md); the raw runs and exact
commands are in [`measurements/`](../measurements/README.md).

| Component | Unit-tested | Run against a real model |
|---|---|---|
| Tracing decorator, SQLite trace DB, background compile, hot-swap, fail-open | Yes | Yes. A live Claude teacher was hot-swapped to a real compiled adapter, and an induced failure (adapter file deleted) fell open to the teacher cleanly |
| Shadow mode | Yes | Yes, against a real adapter and a replay of recorded teacher answers: promotion, stalling, audit cost and caller latency. Not yet against live traffic |
| `suite.yaml` runner, fuzzer, active-learning loop | Yes | Yes. The loop was run on real fuzzer failures with a live Claude teacher; it is bounded and best-effort, not a guarantee |
| `paw-test compare`, `paw-test judge` | Yes | Yes. Both were used to produce the compiler comparison and judge-noise figures |
| Pydantic-to-regex compiler and grammar-constrained decoding (byte-level, `llguidance`-backed) | Yes | Yes. Applied by `ProgramAsWeightsBackend` by default whenever `llguidance` is installed, and measured end to end on two real adapters: shape, cost, and that the mask runs on every generation step. `MockPAWBackend` still ignores it, and `paw.load`'s post-hoc validation still runs either way |
| `ProgramAsWeightsBackend` (official upstream SDK) | Yes, against a fake SDK | Yes. Real compile and inference on an RTX 3080 (CPU and CUDA) and an A100, with both upstream compilers |
| `paw-kit doctor` | Yes, against a fake SDK | Not measured; it is a diagnostic, not a model path |
| HTTP server, Docker export, dataset export, CLI | Yes | No. Exercised with `MockPAWBackend` only |
| `MockPAWBackend` | n/a | A dictionary lookup that returns canned strings. It is a test double, not a model |

The demo command and the three examples all run on the mock. When they print "local
adapter" they mean the dictionary lookup. They demonstrate the *control flow* of the
harness, nothing about model quality or speed.
