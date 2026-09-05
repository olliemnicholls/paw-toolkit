# Example 2: High-Throughput PII Scrubber (`paw.schema`)

This example demonstrates how `paw-kit` uses Pydantic schema compilation and grammar-constrained decoding to deliver high-throughput, structured neural inference with **0.0% JSON syntax errors**.

## What It Does

1. **Schema Enforcement**: Defines a nested Pydantic model (`PIIScrubResult` with `List[PIIEntity]`).
2. **Grammar Compilation**: Uses `paw.schema` to convert the Pydantic type annotations into an exact finite-state machine (FSM) regex pattern.
3. **Constrained Autoregressive Decoding**: Constrains neural generation at the token level, mathematically preventing invalid JSON syntax, missing braces, or corrupted keys.
4. **Air-Gapped Privacy**: Executes entirely on local hardware, ensuring customer PII (credit cards, SSNs, emails) never leaves your private infrastructure.
5. **Fail-Open Fallback**: If local execution throws or violates constraints, execution falls back immediately to the specified fallback callable.

## Running the Example

```bash
uv run python examples/pii_scrubber/run.py
```
