# Example 2: PII Scrubber (`paw.schema`)

This example shows the `paw.load` control flow: bind an adapter to a nested Pydantic model
(`PIIScrubResult` with `List[PIIEntity]`), validate every output against it, and fall back to
a deterministic callable when validation fails.

**What it does not show:** a model. The adapter is a `MockPAWBackend` subclass that runs the
same rule-based logic as the fallback. It ignores the grammar constraint, so nothing here
exercises token-level constrained decoding. The regex grammar is compiled from the schema
(you can see it via `pydantic_to_regex`), but no current backend applies it during decoding;
see the "What is real and what is mocked" table in the top-level README. Timings printed by
the script are the cost of a Python dictionary lookup, not a benchmark.

## Running the Example

```bash
uv run python examples/pii_scrubber/run.py
```
