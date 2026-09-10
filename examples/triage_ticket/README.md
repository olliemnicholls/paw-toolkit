# Example 1: Customer Support Ticket Triage (`paw.jit`)

This example demonstrates how to use the `@compile_on_hit` decorator from `paw-kit` to trace a high-volume LLM function, trigger compilation at a call threshold, and hot-swap to the compiled adapter with fail-open fallback.

## What It Does

1. **Calls 1 to 5 (Remote Teacher Tracing)**: Requests go to the wrapped function, which in real use is your Claude/OpenAI call. In this example it is `simulated_remote_llm`: a `time.sleep(0.2)` plus keyword rules, no network. Inputs and outputs are captured in an embedded SQLite database (`.paw_demo_triage/traces.db`).
2. **Threshold Reached (Call 5)**: When the call count reaches the threshold (5 here; the decorator's default is 50), `paw-kit` compiles the specification, with the traced pairs folded in as examples, into a local `.paw` adapter in a background thread.
3. **Calls 6 to 10 (Hot-Swap to Local Neural Function)**: Once compiled, subsequent calls are automatically redirected to the local adapter. In this demo the "local adapter" is `MockPAWBackend` (a dictionary lookup, not a real model), so the timing shown is illustrative of the intended hot-swap mechanism, not a measured benchmark of real model inference.
4. **Fail-Open Fallback**: If local execution throws an error or schema validation fails, `paw-kit` transparently fails open back to the remote teacher.

## Running the Example

```bash
uv run python examples/triage_ticket/run.py
```

## Inspecting the Compiled Adapter

After running the script, inspect the generated adapter using the `paw-inspect` CLI:

```bash
uv run paw-inspect examples/triage_ticket/.paw_demo_triage/adapter_*.paw
```
