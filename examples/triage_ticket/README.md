# Ticket triage: `@compile_on_hit`

Wraps a function that stands in for a frontier API call, traces it, compiles at a call
threshold, and hot-swaps to the compiled adapter with fail-open fallback.

```bash
uv run python examples/triage_ticket/run.py
```

## What it runs

Ten support tickets go through `triage_ticket`, a function decorated with
`@compile_on_hit(threshold=5)`.

- **Calls 1 to 5** go to the wrapped function. Here that is `simulated_remote_llm`, a
  keyword lookup with a 200 ms sleep; in real use it is your Claude or OpenAI call. Each
  input/output pair is recorded in a SQLite trace database under `.paw_demo_triage/`.
- **Call 5** reaches the threshold and compiles the spec, with the traced pairs folded in
  as examples, into a `.paw` adapter in a background thread.
- **Calls 6 to 10** are routed to the adapter. If the adapter raised or returned
  something that failed the `TriageResult` schema, the call would fall back to the
  wrapped function.

The script prints which side served each call and how long it took.

## What it does not show

The adapter is `MockPAWBackend`, a dictionary lookup, not a model. The sub-millisecond
timings on calls 6 to 10 are the harness's own overhead; the measured latency of a real
compiled adapter is on the [results page](../../docs/results.md).

In real use a freshly compiled adapter does not take over at call 6. It first runs in
[shadow mode](../../docs/shadow-mode.md), where the teacher keeps serving until the
adapter agrees with it over a window of calls. This example passes `shadow_window=0` to
switch that off, because ten calls cannot fill a window.

## Inspect the adapter

```bash
uv run paw-inspect examples/triage_ticket/.paw_demo_triage/adapter_*.paw
```
