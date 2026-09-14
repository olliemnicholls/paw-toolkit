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

## Rate limiting, CORS, and reverse proxies

Each client is capped at 120 requests/minute by default (`PAW_RATE_LIMIT_PER_MINUTE`;
`0` disables it), plus a combined ceiling across every client of 20,000/minute
(`PAW_RATE_LIMIT_GLOBAL_PER_MINUTE`) so no amount of address-rotation by one caller can
starve everyone else. The client is identified by the connecting socket address, which is
correct for a direct connection but **the same for every caller** if the server sits behind
a reverse proxy or load balancer — set `PAW_TRUST_PROXY_HEADER=1` in that case so the
limiter reads the real client address from `X-Forwarded-For` instead. Only set it when a
proxy you control is actually the one setting that header: a caller directly attached to
the server can otherwise put anything it likes in `X-Forwarded-For` and evade the limiter
entirely.

No cross-origin browser access is allowed by default. Set `PAW_CORS_ORIGINS` to a
comma-separated allowlist of origins to permit it.

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

## Concurrency and slow adapters

A `paw-serve` process shares one small worker pool across every compiled function it
serves. If one adapter wedges or runs slowly, it can occupy every slot in that pool,
which makes calls to your *other*, healthy adapters fail open to their own teacher too
(correctly, not silently) for as long as the wedged one holds them. This is not a crash
and not data loss — a further call, once the pool is genuinely full, fails open
immediately rather than queueing behind the wedged one and paying its deadline too — but
it is a cost and availability surprise if you are not expecting it.

Two levers, in order of what to try first:

- Lower `adapter_timeout_s` (on `@compile_on_hit`) for adapters where a slow or wedged
  backend call should give up quickly, so a single bad call occupies a slot for less time.
- If several latency-sensitive tasks share one process and you want real isolation between
  them, run them in separate `paw-serve` processes instead of one process serving all of
  them — a process boundary is currently the only way to give one task's calls their own
  pool.

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
