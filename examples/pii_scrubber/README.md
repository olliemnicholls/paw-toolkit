# PII scrubber: `paw.load` with a Pydantic schema

Binds an adapter to a nested Pydantic model, validates every output against it, and
falls back to a plain Python function when validation fails.

```bash
uv run python examples/pii_scrubber/run.py
```

## What it runs

1. Compiles an adapter from a spec and one worked example.
2. Binds it with `paw.load(..., response_model=PIIScrubResult, fallback_provider=...)`.
   `PIIScrubResult` has a list of `PIIEntity` values inside it, so this is the
   nested-schema case.
3. Runs five strings through the bound function and prints the sanitised text and the
   entities found. Every output is parsed into `PIIScrubResult` before it is returned;
   an output that failed to parse would be replaced by the fallback's answer.

## What it does not show

There is no model here. The adapter is a `MockPAWBackend` subclass that runs the same
regex rules as the fallback, so the adapter and the fallback always agree. The timings
printed are the cost of a Python function call, not inference.

`paw.load` compiles the schema to a regex grammar, and you can see it with
`pydantic_to_regex`, but no shipped backend applies that grammar during decoding.
Validation happens after generation. See
[What is real and what is mocked](../../docs/what-is-real.md).
