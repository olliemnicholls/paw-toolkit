# Example 1: Customer Support Ticket Triage (`paw.jit`)

This example demonstrates how to use the `@compile_on_hit` decorator from `paw-kit` to trace a high-volume LLM function, trigger compilation at a call threshold, and hot-swap to the compiled adapter with fail-open fallback.

## What It Does

1. **Calls 1 to 5 (Remote Teacher Tracing)**: Production requests are routed to the teacher LLM (e.g. Claude 3.5 Sonnet / GPT-4o). Inputs and teacher responses are captured transparently in an embedded SQLite database (`.paw/traces.db`).
2. **Threshold Reached (Call 5)**: When call volume reaches the designated threshold (5 calls in this demo, typically 50-100 in production), `paw-kit` asynchronously compiles the specification and collected dataset into a local `.paw` adapter.
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
