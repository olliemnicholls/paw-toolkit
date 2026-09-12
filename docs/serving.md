# Serve over HTTP

```bash
uv run paw-serve .paw/triage.paw --port 8000
curl -X POST localhost:8000/invoke \
  -H "Authorization: Bearer $PAW_API_KEY" \
  -H 'Content-Type: application/json' -d '{"input": "Outage in eu-west"}'
```

Authentication is on by default. The bearer token is `--api-key` or the `PAW_API_KEY`
environment variable; if neither is set, the server generates one and prints it to
stderr at startup. `--allow-anonymous` turns authentication off.

`POST /v1/chat/completions` (OpenAI shape) and `POST /v1/messages` (Anthropic shape) are
also exposed, so the official OpenAI and Anthropic client libraries work with `baseURL`
pointed at the server. `GET /health` (liveness) and `GET /ready` (readiness) are the only
unauthenticated routes; `GET /metrics` and every inference route require the bearer token.

## Readiness

`/ready` returns 503 until the adapter has served one successful call, or until the
`--warm` pass has loaded the model, and 200 from then on. It is useful because a cold
model load is slow enough to matter (measured; see [results](./results.md)). Two things
to know:

- `--warm` exercises model load only, through the backend's inference call, not the
  schema-validating wrapper, so a strict `response_model` does not keep the server at 503.
- `/ready` **never de-asserts.** Once it has returned 200 it keeps doing so, even if the
  backend later breaks; after that point it carries no more signal than `/health`. Do not
  wire it as a Kubernetes readiness probe expecting eviction on failure.

## Docker

```bash
uv run paw-kit export docker .paw/triage.paw --out-dir ./docker                  # Dockerfile + compose
uv run paw-kit export docker .paw/triage.paw --out-dir ./docker --backend mock   # demo container
```

`--backend` (default `real`) decides what the generated container actually serves. With
`real`, the generated requirements include the SDK extra, the CMD passes `--warm`, and the
HEALTHCHECK polls `/ready` with a 180 s start period to cover the cold model download.
With `mock`, the container serves the dictionary-lookup test double, `--warm` is omitted
and the start period is short; the generated comments say so. The scaffold never
describes a backend the container cannot run.

## Exporting traces

```bash
uv run paw-kit export dataset --db ./.paw/traces.db --out traces.jsonl
```

The JSONL contains the traced inputs and teacher outputs as stored, so it holds whatever
your production inputs held. `redact_trace=True` on the decorator applies a best-effort
scrub of bearer tokens and password/secret/api-key values before rows are written; it is
not a guarantee. Treat the file accordingly.
