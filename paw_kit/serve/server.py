"""FastAPI HTTP microservice serving compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints with post-generation Pydantic validation and telemetry.
"""

from collections import OrderedDict, deque
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.datastructures import Headers

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.schema.loader import load
from paw_kit.serve.models import (
    AnthropicContentBlock,
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    AnthropicUsage,
    ChatCompletionChoice,
    ChatCompletionChoiceMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
    MetricsResponse,
    UsageInfo,
)

logger = logging.getLogger("paw_kit.serve")


def backend_label(backend: object) -> str:
    """Describe a backend as "mock" or "real" for telemetry.

    Two deliberate choices. First, the test is `isinstance(..., MockPAWBackend)` rather
    than a whitelist of known real backends: the previous form was
    `"real" if isinstance(b, RealPAWBackend) else "mock"` (that class was deleted in
    Track 13), which mislabelled *every* other
    backend -- including `ProgramAsWeightsBackend`, the only one proven against a real
    model, and any third-party `AbstractPAWBackend` -- as "mock". Anything that is not the
    known test double is real.

    Second, this returns a fixed two-value vocabulary rather than `type(backend).__name__`.
    PAW-SERVE-06 deliberately stripped backend implementation detail out of the telemetry
    surface; a concrete class name is exactly what that finding removed, so it must not
    come back in through the label if `ServerState` is ever re-exposed. It also matches
    what `paw-serve` prints to the user.

    Factored out as a function so the logic is testable and has a single definition:
    `paw-serve` prints it to the user, and it was previously duplicated inline in the CLI
    with a different vocabulary (`backend_type.lower()`), so the two could disagree.
    """
    return "mock" if isinstance(backend, MockPAWBackend) else "real"


class ServerState:
    """Thread-safe runtime server telemetry state."""

    def __init__(self) -> None:
        # `adapter_path` and `backend_name` used to be stored here and were read
        # nowhere: PAW-SERVE-06 removed both from /health, leaving write-only state
        # that no test could observe and no endpoint could expose. Dead telemetry is
        # worse than none -- it reads as a live signal. `backend_label()` above is the
        # surviving, tested half; the CLI is its consumer.
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        self.latencies: deque[float] = deque(maxlen=10000)
        self._lock = threading.Lock()
        # `/ready` (readiness, distinct from `/health`'s liveness): flips permanently
        # once true. `threading.Event` rather than a plain bool under `_lock` so
        # `is_ready()` -- read on every request, including by the same liveness-style
        # probe pattern PAW-SERVE-07 keeps lock-free for `get_uptime()` -- never
        # contends with the request-recording lock above.
        self._ready = threading.Event()

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.error_count += 1
            self.latencies.append(latency_ms)

    def mark_ready(self) -> None:
        """Flip `/ready` to 200. Called after the first successful inference (either a
        real request or an explicit `--warm` warm-up); never unset afterward."""
        self._ready.set()

    def is_ready(self) -> bool:
        """True once `mark_ready()` has been called at least once."""
        return self._ready.is_set()

    def get_uptime(self) -> float:
        """PAW-SERVE-07: lock-free. `start_time` is set once in `__init__` and never
        mutated afterward, so reading it needs no synchronization — unlike
        `get_metrics()`, this is safe to call from a liveness probe under contention
        without ever waiting on the same lock request recording is fighting over."""
        return max(0.0, time.time() - self.start_time)

    def get_metrics(self) -> Dict[str, Any]:
        # PAW-SERVE-07: snapshot the cheap counters and copy the latency deque while
        # holding the lock, then release it before sorting. Sorting up to 10,000
        # latencies while holding the lock serializes every concurrent request behind
        # whichever caller is percentile-computing, including record_request(); a copy
        # is O(n) under the lock but the O(n log n) sort itself runs lock-free.
        with self._lock:
            total = self.total_requests
            errors = self.error_count
            latencies_snapshot = list(self.latencies)
        uptime = self.get_uptime()
        if not latencies_snapshot:
            return {
                "total_requests": total,
                "error_count": errors,
                "uptime_seconds": round(uptime, 2),
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
            }
        sorted_lat = sorted(latencies_snapshot)
        n = len(sorted_lat)
        p50 = sorted_lat[int(n * 0.50)]
        p95 = sorted_lat[min(int(n * 0.95), n - 1)]
        p99 = sorted_lat[min(int(n * 0.99), n - 1)]
        return {
            "total_requests": total,
            "error_count": errors,
            "uptime_seconds": round(uptime, 2),
            "p50_latency_ms": round(p50, 3),
            "p95_latency_ms": round(p95, 3),
            "p99_latency_ms": round(p99, 3),
        }


def _extract_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract flat string text from string or message content block list.

    PAW-SERVE-05: `item.get("text", "")` only falls back to `""` when the key is
    *absent*; a block shaped `{"type": "text", "text": None}` has the key present with
    an explicit `None` value, so `.get` returns `None` and the later `" ".join(parts)`
    raises `TypeError: sequence item N: expected str instance, NoneType found`. Guard
    the extracted value's type explicitly instead of trusting the default kwarg alone.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_val = item.get("text", "")
                parts.append(text_val if isinstance(text_val, str) else "")
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic for token count estimation.

    PAW-SERVE-08: `text.split()` materializes a full list of every whitespace-separated
    word before it can be counted, so a multi-megabyte input allocates a proportionally
    large list purely to estimate a number. `len(text) // 4` (the same char-per-token
    approximation OpenAI's own docs use) is O(1) space and does not walk a copy of the
    input to do it.
    """
    return max(1, len(text) // 4)


class PayloadSizeLimitMiddleware:
    """Pure ASGI middleware enforcing a maximum request body size.

    Deliberately *not* a `BaseHTTPMiddleware` subclass: Starlette's
    `BaseHTTPMiddleware.call_next` ignores a rebuilt `Request`'s receive closure, so
    calling `request.stream()` inside one and rebinding the body afterwards sends an
    empty body downstream (`_CachedRequest` only replays a body cached via
    `request.body()`). Wrapping `receive` directly at the ASGI layer avoids that class
    of bug, but a wrapper that merely counts bytes and *raises* once over the limit does
    not work either: FastAPI's own body-parsing (`Request.json()`/dependency resolution)
    catches any exception raised while reading the body and turns it into a generic 400
    ("There was an error parsing the body") before it ever reaches this middleware.

    So this buffers the body itself, bounded, before ever handing control to the
    downstream app: it reads `http.request` messages one at a time, aborting with its
    own 413 as soon as the running total exceeds the limit — so at most one chunk past
    the limit is ever held in memory, never the full oversized body — and only once a
    complete body within the limit has been collected does it replay those exact
    messages, in order, to the downstream app via a synthetic `receive`. The downstream
    app therefore sees a completely normal ASGI request; `request.body()`/`.json()`
    behave exactly as if no middleware were present.
    """

    def __init__(self, app: Any, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    response = Response(status_code=413, content="Payload Too Large (maximum 10MB)")
                    await response(scope, receive, send)
                    return
            except ValueError:
                response = Response(status_code=400, content="Invalid Content-Length header")
                await response(scope, receive, send)
                return

        # Defense in depth: bound the body ourselves too, in case Content-Length was
        # absent (chunked transfer) or understated.
        buffered_messages: List[Dict[str, Any]] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                buffered_messages.append(message)
                break
            buffered_messages.append(message)
            total_bytes += len(message.get("body", b"") or b"")
            if total_bytes > self.max_body_bytes:
                response = Response(status_code=413, content="Payload Too Large (maximum 10MB)")
                await response(scope, receive, send)
                return
            if not message.get("more_body", False):
                break

        buffered_iter = iter(buffered_messages)

        async def replay_receive() -> Dict[str, Any]:
            try:
                return next(buffered_iter)
            except StopIteration:
                return await receive()

        await self.app(scope, replay_receive, send)


_DEFAULT_RATE_LIMIT_PER_MINUTE = 120
_MAX_RATE_LIMIT_BUCKETS = 10_000
# `/ready` gets the same exemption as `/health` and for the same reason: it is polled on
# a fixed interval by container orchestration (Track 09's Dockerfile HEALTHCHECK now
# polls `/ready` instead of `/health`, see `docker.py`) and does no inference work of
# its own.
_RATE_LIMIT_EXEMPT_PATHS = frozenset({"/health", "/ready"})


class RateLimitMiddleware:
    """Pure ASGI token-bucket rate limiter, keyed per client address.

    PAW-SERVE-10: nothing previously bounded how many requests a single client could
    fire, on any route including the unauthenticated-by-default-deny inference
    endpoints, making brute-forcing the ephemeral bearer token (PAW-SERVE-01) or simply
    exhausting server capacity unthrottled. `/health` is exempt — Track 09's generated
    Dockerfile `HEALTHCHECK` and docker-compose healthcheck poll it on a fixed interval
    for the life of the container, and it does no inference work, so counting it against
    a client's budget would make container orchestration itself trip the limiter.

    Bucket storage is bounded (`_MAX_RATE_LIMIT_BUCKETS`, evicted oldest-first) so an
    attacker spraying requests from many distinct source addresses cannot turn the
    limiter's own bookkeeping into an unbounded-memory DoS — the defense this middleware
    exists to provide would otherwise become a new instance of the exact problem class.
    """

    def __init__(self, app: Any, requests_per_minute: int) -> None:
        self.app = app
        self.capacity = float(requests_per_minute)
        self.refill_per_second = requests_per_minute / 60.0
        self._buckets: "OrderedDict[str, tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()

    def _consume(self, client_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last_refill = self._buckets.pop(client_key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last_refill) * self.refill_per_second)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[client_key] = (tokens, now)
            self._buckets.move_to_end(client_key)
            while len(self._buckets) > _MAX_RATE_LIMIT_BUCKETS:
                self._buckets.popitem(last=False)
            return allowed

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope.get("path") in _RATE_LIMIT_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        client = scope.get("client")
        client_key = client[0] if client else "unknown"
        if not self._consume(client_key):
            response = Response(
                status_code=429,
                content="Rate limit exceeded, try again shortly",
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


_INFERENCE_SLOT_TIMEOUT_SECONDS = 30.0


def _execute_with_telemetry(
    work: Callable[[], Any],
    semaphore: threading.BoundedSemaphore,
    state: ServerState,
    route_name: str,
) -> "tuple[Any, float]":
    """Run `work()` under a bounded concurrency slot, recording telemetry on every
    outcome and converting any failure — timeout, parsing, or inference — into an
    `HTTPException` instead of letting it propagate as an unhandled 500 that bypasses
    `state.record_request(is_error=True)` (PAW-SERVE-05).

    PAW-SERVE-04: acquiring with a timeout instead of blocking forever bounds how long a
    request can occupy a threadpool worker waiting for the (single-slot, today) inference
    resource — an unbounded wait lets concurrent inference traffic pile up worker
    threads indefinitely. This does not touch `/health`/`/metrics`, which no longer
    route through this function at all (see PAW-SERVE-04's `async def` fix) and so
    cannot be starved by inference contention regardless of this timeout's value.
    """
    t0 = time.perf_counter()
    if not semaphore.acquire(timeout=_INFERENCE_SLOT_TIMEOUT_SECONDS):
        state.record_request((time.perf_counter() - t0) * 1000.0, is_error=True)
        raise HTTPException(
            status_code=503,
            detail="Server busy: too many concurrent inference requests, try again shortly",
        )
    try:
        res = work()
    except Exception as exc:
        latency = (time.perf_counter() - t0) * 1000.0
        state.record_request(latency, is_error=True)
        logger.exception("Inference failed in %s: %s", route_name, exc)
        raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        semaphore.release()
    latency = (time.perf_counter() - t0) * 1000.0
    state.record_request(latency, is_error=False)
    # A real request just proved the adapter works end to end -- that satisfies /ready's
    # contract (see ServerState.mark_ready) exactly as well as an explicit --warm
    # warm-up would have, so this is the only other place that needs to set it.
    state.mark_ready()
    return res, latency


# Fixed, deliberately short: the point of a warm-up call is to pay the one-time cost of
# loading the base model into memory (measured up to ~110s cold, see
# measurements/README.md), not to exercise the adapter's actual task.
_WARMUP_INPUT = "ping"


def _run_warmup(exec_fn: Callable[[str], Any], state: "ServerState") -> None:
    """Run one inference with a fixed short input so `/ready` is 200 before the server
    starts accepting real traffic, instead of 503 until the first request completes.

    A failure here is logged, not raised: `--warm` is a latency optimization, and a
    broken warm-up must not crash `paw-serve` on startup -- it just leaves `/ready` at
    503 until a real request succeeds (or never, if the backend is genuinely broken,
    which every other endpoint will also report).
    """
    t0 = time.perf_counter()
    try:
        exec_fn(_WARMUP_INPUT)
    except Exception as exc:
        logger.warning(
            "Warm-up inference failed after %.2fs: %s", time.perf_counter() - t0, exc
        )
        return
    elapsed = time.perf_counter() - t0
    logger.info("Warm-up inference completed in %.2fs", elapsed)
    state.mark_ready()


def create_app(
    adapter_path: Union[str, Path],
    backend: Optional[AbstractPAWBackend] = None,
    response_model: Optional[Type[BaseModel]] = None,
    task_name: Optional[str] = None,
    api_key: Optional[str] = None,
    allow_anonymous: bool = False,
    requests_per_minute: Optional[int] = None,
    warm: bool = False,
) -> FastAPI:
    """Factory creating configured FastAPI microservice for the given .paw adapter.

    `warm`, if True, runs one inference with a fixed short input before returning the
    app (i.e. before `serve_adapter` binds the port), so `/ready` reports 200 from the
    first request instead of 503 until a real inference succeeds. See `_run_warmup`.
    """
    path_obj = Path(adapter_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    selected_backend = backend or MockPAWBackend()
    # Return basename to avoid exposing host filesystem directory layout
    state = ServerState()
    configured_api_key = api_key or os.environ.get("PAW_API_KEY")
    if not configured_api_key and not allow_anonymous:
        # PAW-SERVE-01: default-deny. Without this, every inference endpoint is
        # unauthenticated by default (CWE-306). Generate a per-process ephemeral token
        # rather than silently running open; --allow-anonymous is the explicit opt-out.
        configured_api_key = secrets.token_urlsafe(32)
        print(
            f"[paw-serve] SECURITY NOTICE: no API key configured. Generated ephemeral "
            f"bearer token: {configured_api_key}\n"
            f"[paw-serve] Pass this as 'Authorization: Bearer <token>', or set PAW_API_KEY "
            f"/ --api-key for a stable key, or pass --allow-anonymous to disable auth.",
            file=sys.stderr,
        )
        logger.warning("No API key configured. Generated an ephemeral bearer token (see stderr).")
    # PAW-SERVE-04: a single-slot bounded semaphore instead of a plain `threading.Lock`
    # so a caller waiting for the inference slot can time out (503) rather than occupy a
    # threadpool worker indefinitely; see `_execute_with_telemetry`.
    inference_semaphore = threading.BoundedSemaphore(1)

    # Initialize execution function (with schema validation if model provided)
    if response_model is not None:
        exec_fn = load(
            adapter_path=str(path_obj),
            response_model=response_model,
            backend=selected_backend,
        )
    else:
        exec_fn = lambda inp: selected_backend.infer(str(path_obj), inp)

    if warm:
        _run_warmup(exec_fn, state)

    # PAW-SERVE-06: the default docs routes are disabled here and re-registered below,
    # behind the same `_verify_auth` gate as everything but `/health`.
    app = FastAPI(
        title="PAW-Kit Microservice",
        description=f"Zero-marginal-cost neural microservice serving {path_obj.name}",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # Middleware ordering (outermost first; add_middleware-registration order below
    # matches this, since Starlette runs the first-registered middleware outermost):
    #   CORS -> RateLimit -> PayloadSizeLimit -> routes.
    # CORS decides same-origin/allowlisted-origin handling first, cheaply, before any
    # other work. RateLimit runs next so an already-throttled client is turned away
    # before PayloadSizeLimitMiddleware spends any effort buffering its body.
    # PayloadSizeLimitMiddleware stays innermost, closest to the routes it protects.
    # Trade-off, noted rather than left implicit: a 429 response is produced above the
    # CORS layer's own middleware position but below CORSMiddleware in the *outer* wrap,
    # so it does carry CORS response headers when CORS is configured — a rate-limited
    # cross-origin browser caller can still read the 429 body.

    # PAW-SERVE-02: no wildcard CORS default. Cross-origin access is opt-in only, via a
    # comma-separated PAW_CORS_ORIGINS allowlist; absent that, no CORS middleware is
    # installed at all, so browsers deny cross-origin requests outright.
    cors_env = os.environ.get("PAW_CORS_ORIGINS", "")
    allowed_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Authorization"],
        )

    # PAW-SERVE-10: bound the request rate per client. `requests_per_minute=0` disables
    # the limiter explicitly (e.g. for trusted internal benchmarking); anything else,
    # including the parameter left unset, falls back to PAW_RATE_LIMIT_PER_MINUTE and
    # then the built-in default.
    if requests_per_minute is None:
        requests_per_minute = int(
            os.environ.get("PAW_RATE_LIMIT_PER_MINUTE", _DEFAULT_RATE_LIMIT_PER_MINUTE)
        )
    if requests_per_minute > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=requests_per_minute)

    # PAW-SERVE-03: enforce the 10MB body limit via pure ASGI middleware (see
    # PayloadSizeLimitMiddleware docstring for why BaseHTTPMiddleware cannot do this
    # safely without emptying the request body for every downstream handler).
    app.add_middleware(PayloadSizeLimitMiddleware, max_body_bytes=10 * 1024 * 1024)

    def _verify_auth(request: Request) -> None:
        """Verify bearer token if PAW_API_KEY is configured using constant-time comparison."""
        if not configured_api_key:
            return
        auth = request.headers.get("authorization")
        if not auth or not auth.startswith("Bearer "):
            # PAW-SERVE-09: RFC 6750 requires a WWW-Authenticate challenge on 401s from
            # a Bearer-protected resource; omitting it doesn't stop a knowledgeable
            # client but breaks generic HTTP clients that rely on the header to know
            # which auth scheme to retry with.
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Missing or malformed Bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth[7:].strip()
        if not hmac.compare_digest(token, configured_api_key):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized: Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    # PAW-SERVE-04: `/health` and `/metrics` are `async def` so FastAPI runs them
    # directly on the event loop instead of dispatching to the AnyIO worker threadpool
    # the three inference routes below block a slot in. An exhausted threadpool (a burst
    # of concurrent inference calls each waiting on `inference_semaphore`) can then no
    # longer starve the liveness probe or the metrics scrape alongside it. The three
    # inference routes stay synchronous `def` on purpose: `exec_fn` is a blocking call,
    # and converting them to `async def` would run inference on the event loop itself
    # and stall every other request the server is handling, which is the opposite of
    # this fix's intent.
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        # PAW-SERVE-06: status/uptime/version only — see HealthResponse's docstring.
        return HealthResponse(status="ok", uptime_seconds=state.get_uptime(), version="0.1.0")

    @app.get("/ready", include_in_schema=False)
    async def ready_check() -> JSONResponse:
        # Readiness, distinct from `/health`'s liveness: 200 only once the adapter has
        # completed one successful inference (a real request or an explicit `--warm`
        # warm-up, see ServerState.mark_ready) -- not merely once the process is up and
        # accepting connections. Same auth exemption (no `_verify_auth` call) and rate-
        # limit exemption (`_RATE_LIMIT_EXEMPT_PATHS`) as `/health`, for orchestration
        # to poll freely.
        if state.is_ready():
            return JSONResponse({"status": "ready"}, status_code=200)
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.get("/metrics", response_model=MetricsResponse)
    async def telemetry_metrics(request: Request) -> MetricsResponse:
        # PAW-SERVE-06: /metrics is not a liveness probe and discloses request-volume
        # and latency telemetry, so it follows the same auth policy as everything but
        # /health, unlike before this fix.
        _verify_auth(request)
        return MetricsResponse(**state.get_metrics())

    @app.get("/openapi.json", include_in_schema=False)
    def openapi_json(request: Request) -> JSONResponse:
        # PAW-SERVE-06: the generated OpenAPI schema can describe internal endpoint
        # shapes to an unauthenticated caller; gate it like /docs and /redoc.
        _verify_auth(request)
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    def swagger_docs(request: Request) -> HTMLResponse:
        _verify_auth(request)
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")

    @app.get("/redoc", include_in_schema=False)
    def redoc_docs(request: Request) -> HTMLResponse:
        _verify_auth(request)
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
        _verify_auth(request)
        res, latency = _execute_with_telemetry(
            lambda: exec_fn(req.input), inference_semaphore, state, "/invoke"
        )

        if isinstance(res, BaseModel):
            out = res.model_dump()
        elif isinstance(res, str):
            try:
                out = json.loads(res)
            except Exception:
                out = res
        else:
            out = res

        return InvokeResponse(
            output=out,
            latency_ms=round(latency, 2),
            adapter=path_obj.name,
            model=task_name or path_obj.stem,
        )

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
        _verify_auth(request)
        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming is not yet supported")
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages array cannot be empty")

        def _run() -> "tuple[Any, str]":
            # PAW-SERVE-05: content extraction now runs inside the telemetry-and-error
            # boundary below, so a malformed content block records the request as an
            # error instead of raising past `state.record_request` entirely.
            system_content = ""
            user_prompts: List[str] = []
            for msg in req.messages:
                text = _extract_content(msg.content)
                if msg.role == "system":
                    system_content = text
                elif msg.role in ("user", "tool"):
                    user_prompts.append(text)

            last_user = user_prompts[-1] if user_prompts else ""
            combined_input = (
                f"{system_content}\n\n{last_user}".strip()
                if system_content and last_user
                else (last_user or system_content)
            )
            return exec_fn(combined_input), combined_input

        (res, input_payload), latency = _execute_with_telemetry(
            _run, inference_semaphore, state, "/v1/chat/completions"
        )

        if isinstance(res, BaseModel):
            content_str = res.model_dump_json()
        elif isinstance(res, dict):
            content_str = json.dumps(res)
        else:
            content_str = str(res)

        p_tokens = _estimate_tokens(input_payload)
        c_tokens = _estimate_tokens(content_str)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            object="chat.completion",
            created=int(time.time()),
            model=req.model or path_obj.stem,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionChoiceMessage(role="assistant", content=content_str),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=p_tokens + c_tokens,
            ),
        )

    @app.post("/v1/messages", response_model=AnthropicMessageResponse)
    def anthropic_messages(req: AnthropicMessageRequest, request: Request) -> AnthropicMessageResponse:
        _verify_auth(request)
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages array cannot be empty")

        def _run() -> "tuple[Any, str]":
            # PAW-SERVE-05: see the matching comment in chat_completions' `_run`.
            system_text = _extract_content(req.system) if req.system else ""
            user_texts = []
            for msg in req.messages:
                if msg.role == "user":
                    user_texts.append(_extract_content(msg.content))

            last_user = user_texts[-1] if user_texts else ""
            combined_input = (
                f"{system_text}\n\n{last_user}".strip()
                if system_text and last_user
                else (last_user or system_text)
            )
            return exec_fn(combined_input), combined_input

        (res, input_payload), latency = _execute_with_telemetry(
            _run, inference_semaphore, state, "/v1/messages"
        )

        if isinstance(res, BaseModel):
            content_str = res.model_dump_json()
        elif isinstance(res, dict):
            content_str = json.dumps(res)
        else:
            content_str = str(res)

        p_tokens = _estimate_tokens(input_payload)
        c_tokens = _estimate_tokens(content_str)

        return AnthropicMessageResponse(
            id=f"msg_{uuid.uuid4().hex[:16]}",
            type="message",
            role="assistant",
            content=[AnthropicContentBlock(type="text", text=content_str)],
            model=req.model or path_obj.stem,
            stop_reason="end_turn",
            usage=AnthropicUsage(
                input_tokens=p_tokens,
                output_tokens=c_tokens,
            ),
        )

    return app


def serve_adapter(
    adapter_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: Optional[AbstractPAWBackend] = None,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
    allow_anonymous: bool = False,
    requests_per_minute: Optional[int] = None,
    warm: bool = False,
) -> None:
    """Start Uvicorn web server hosting the compiled adapter.

    `warm`: see `create_app`'s docstring -- runs a warm-up inference before binding.
    """
    import uvicorn

    app = create_app(
        adapter_path=adapter_path,
        backend=backend,
        response_model=response_model,
        api_key=api_key,
        allow_anonymous=allow_anonymous,
        requests_per_minute=requests_per_minute,
        warm=warm,
    )
    uvicorn.run(app, host=host, port=port)
