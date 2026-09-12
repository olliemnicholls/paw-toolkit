"""FastAPI HTTP microservice serving compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints with post-generation Pydantic validation and telemetry.
"""

from collections import OrderedDict, deque
from contextlib import asynccontextmanager
import asyncio
import hmac
import ipaddress
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

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.middleware import Middleware

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
        # X-9: an internal aggregate, deliberately *not* part of MetricsResponse
        # (paw_kit/serve/models.py is out of scope for this beyond X-2's own
        # narrowly-named field) -- failed-auth requests are counted via the existing
        # total_requests/error_count counters below (record_auth_failure), and this
        # extra counter exists only so the log line can report a running total
        # ("aggregated", per the finding) without echoing anything caller-supplied.
        self.auth_failure_count = 0

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.error_count += 1
            self.latencies.append(latency_ms)

    def record_auth_failure(self) -> None:
        """X-9: a failed-auth request previously vanished entirely -- neither logged
        nor counted, so /metrics read total_requests: 0 after 25 failed attempts.
        Counted through the same total_requests/error_count counters every other
        failure uses (so /metrics reflects it immediately), plus a dedicated
        aggregate for the log line in AuthMiddleware to report."""
        with self._lock:
            self.total_requests += 1
            self.error_count += 1
            self.auth_failure_count += 1

    def get_auth_failure_count(self) -> int:
        with self._lock:
            return self.auth_failure_count

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


# `/health` and `/ready` are exempt from both authentication and rate limiting: both
# are unauthenticated-by-design orchestration probes (container HEALTHCHECK/compose
# healthcheck poll them on a fixed interval, see docker.py) that do no inference work
# of their own, so gating either would make container orchestration itself trip the
# gate it is polling to avoid.
_ORCHESTRATION_EXEMPT_PATHS = frozenset({"/health", "/ready"})


def _format_max_body_message(max_body_bytes: int) -> str:
    """Render the 413 body text from the *actual* configured cap, rather than a
    hardcoded "10MB" that would silently go stale the moment the cap changes (X-2
    lowers the default) -- a message that describes a limit other than the one being
    enforced is exactly the class of bug this campaign exists to close."""
    mb = max_body_bytes / (1024 * 1024)
    if mb == int(mb):
        return f"Payload Too Large (maximum {int(mb)}MB)"
    return f"Payload Too Large (maximum {mb:.2f}MB)"


# X-3: total wall-clock deadline for reading a request body to completion, on top of
# the byte-count cap below. 15s is generous for any normal client (even a slow mobile
# connection uploading a payload under the cap) while still bounding a client that
# sends one chunk and stalls -- a legitimate slow chunked-transfer client or a
# deliberate slow-loris-style attacker -- to a fixed worst case instead of holding the
# connection (pre-auth, pre-rate-limit: this middleware buffers the body before either
# of those layers run, by design, see PayloadSizeLimitMiddleware's own docstring) open
# indefinitely.
#
# This bounds only *this* request's body read; it does not bound how many
# connections/requests can be open at once server-wide. An operator expecting many
# slow/stalled clients simultaneously (rather than one at a time) should also set
# Uvicorn's own `--limit-concurrency` -- `paw-serve` does not thread this through
# today, so pass it directly to `uvicorn` if running this ASGI app outside `paw-serve`.
_BODY_READ_TIMEOUT_SECONDS = 15.0


class PayloadSizeLimitMiddleware:
    """Pure ASGI middleware enforcing a maximum request body size and a total
    body-read deadline (X-3).

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

    This middleware is registered *inside* `AuthMiddleware` in the stack (see
    create_app's ordering comment) -- an unauthenticated caller is rejected before any
    of this buffering ever runs (X-2), not just before the buffered body is parsed.
    """

    def __init__(
        self,
        app: Any,
        max_body_bytes: int,
        body_read_timeout: float = _BODY_READ_TIMEOUT_SECONDS,
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.body_read_timeout = body_read_timeout
        self._too_large_message = _format_max_body_message(max_body_bytes)

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    response = Response(status_code=413, content=self._too_large_message)
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
        deadline = time.monotonic() + self.body_read_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                response = Response(status_code=408, content="Request body read timed out")
                await response(scope, receive, send)
                return
            try:
                message = await asyncio.wait_for(receive(), timeout=remaining)
            except asyncio.TimeoutError:
                response = Response(status_code=408, content="Request body read timed out")
                await response(scope, receive, send)
                return
            if message["type"] != "http.request":
                buffered_messages.append(message)
                break
            buffered_messages.append(message)
            total_bytes += len(message.get("body", b"") or b"")
            if total_bytes > self.max_body_bytes:
                response = Response(status_code=413, content=self._too_large_message)
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
# X-4: an aggregate ceiling across *every* client combined, closing the address-
# rotation evasion no amount of per-key granularity can close on its own: a caller
# with many distinct addresses (trivial with a routable IPv6 /64, or a residential
# proxy pool) gets a fresh full per-key bucket each time it switches key, no matter
# how that key is chosen. Tied to _MAX_RATE_LIMIT_BUCKETS (2x) so it comfortably
# exceeds anything the bucket-storage-bound tests below exercise while still being a
# real, finite ceiling; overridable via PAW_RATE_LIMIT_GLOBAL_PER_MINUTE for an
# operator who wants it tighter (or looser) than that default.
_DEFAULT_GLOBAL_RATE_LIMIT_PER_MINUTE = 2 * _MAX_RATE_LIMIT_BUCKETS


def _int_env_with_fallback(env_var: str, default: int, param_value: Optional[int] = None) -> int:
    """Resolve an integer config value (X-8). An explicit caller-supplied
    `param_value` wins as-is -- it comes from a programmatic caller, not raw
    environment text, so e.g. `requests_per_minute=0` (an intentional, documented
    disable) is trusted rather than second-guessed. Otherwise `env_var` is read and
    parsed: a value that isn't a valid integer falls back to `default` with a logged
    warning instead of crashing `paw-serve` at startup with a raw `ValueError`; a
    negative parsed value is clamped up to 0 (also logged) rather than handed to a
    token-bucket limiter, which has no defined behaviour for a negative capacity.
    """
    if param_value is not None:
        return param_value
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not a valid integer; falling back to the default of %d "
            "instead of crashing at startup.",
            env_var,
            raw,
            default,
        )
        return default
    if value < 0:
        logger.warning(
            "%s=%d is negative; treating it as 0 instead of passing a negative "
            "capacity through to the rate limiter.",
            env_var,
            value,
        )
        return 0
    return value


def _normalize_address_key(address: str) -> str:
    """X-4: key an IPv6 address on its /64 rather than the full 128-bit address --
    a single client can be handed a vast number of distinct addresses inside its own
    routed /64 and rotate through them, defeating a limiter keyed on the full
    address, in a way an IPv4 client (a whole /32 is comparatively scarce) cannot.
    IPv4 addresses and anything that doesn't parse as an IP at all (e.g. the
    ASGI-test-transport's literal "testclient") are keyed on the literal string,
    unchanged."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return address
    if isinstance(ip, ipaddress.IPv6Address):
        network = ipaddress.ip_network(f"{ip}/64", strict=False)
        return str(network.network_address)
    return str(ip)


def _resolve_client_key(scope: Dict[str, Any], trust_proxy_header: bool) -> "tuple[str, bool]":
    """Return `(key, client_was_none)`. X-4's Phase 0 resolution: key on the
    connecting IP address (via `/64` for IPv6, see `_normalize_address_key`)
    regardless of authentication state -- this server's auth is one shared secret,
    not per-client identity, so "key on the authenticated principal" would collapse
    every legitimate caller onto one bucket, reproducing this same finding's
    headline defect under a different name. `trust_proxy_header` (the
    PAW_TRUST_PROXY_HEADER opt-in, see create_app) is the *only* thing that changes
    which address is read: when set, the first hop of `X-Forwarded-For` is trusted;
    otherwise `scope["client"]` (the actual socket peer) is used, exactly as before.

    `client_was_none` is True when neither source yields an address at all --
    `scope["client"] is None` (the documented reverse-proxy deployment) with no
    trusted header configured. The caller logs this loudly (once) rather than
    silently collapsing every such client onto a shared "unknown" bucket.
    """
    if trust_proxy_header:
        forwarded = Headers(scope=scope).get("x-forwarded-for")
        if forwarded:
            candidate = forwarded.split(",")[0].strip()
            if candidate:
                return _normalize_address_key(candidate), False
    client = scope.get("client")
    if client:
        return _normalize_address_key(client[0]), False
    return "unknown", True


class RateLimitMiddleware:
    """Pure ASGI token-bucket rate limiter, keyed per client address (X-4), with a
    global ceiling across all clients combined (X-4) and eviction that never resets
    a still-throttled client's budget (X-5).

    PAW-SERVE-10: nothing previously bounded how many requests a single client could
    fire, on any route including the unauthenticated-by-default-deny inference
    endpoints, making brute-forcing the ephemeral bearer token (PAW-SERVE-01) or simply
    exhausting server capacity unthrottled. `/health`/`/ready` are exempt -- Track 09's
    generated Dockerfile `HEALTHCHECK` and docker-compose healthcheck poll them on a
    fixed interval for the life of the container, and neither does inference work, so
    counting them against a client's budget would make container orchestration itself
    trip the limiter.

    Bucket storage is bounded (`_MAX_RATE_LIMIT_BUCKETS`) so an attacker spraying
    requests from many distinct source addresses cannot turn the limiter's own
    bookkeeping into an unbounded-memory DoS -- the defense this middleware exists to
    provide would otherwise become a new instance of the exact problem class.
    """

    def __init__(
        self,
        app: Any,
        requests_per_minute: int,
        trust_proxy_header: bool = False,
        global_requests_per_minute: Optional[int] = None,
    ) -> None:
        self.app = app
        self.capacity = float(requests_per_minute)
        self.refill_per_second = requests_per_minute / 60.0
        self.trust_proxy_header = trust_proxy_header
        global_rpm = (
            global_requests_per_minute
            if global_requests_per_minute is not None
            else _DEFAULT_GLOBAL_RATE_LIMIT_PER_MINUTE
        )
        self._global_capacity = float(global_rpm)
        self._global_refill_per_second = global_rpm / 60.0
        self._global_tokens = self._global_capacity
        self._global_last_refill = time.monotonic()
        self._buckets: "OrderedDict[str, tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()
        # X-4: log the "scope['client'] is None, no trusted header" situation once
        # per middleware instance rather than on every such request -- it is a
        # startup/deployment-shape fact, not a per-request event, and a busy server
        # in that situation would otherwise flood its own logs.
        self._warned_no_client = False

    def _consume_global_locked(self, now: float) -> bool:
        """Caller must hold `self._lock`. X-4's global ceiling: consumed on every
        request regardless of per-key key/state, so no amount of address rotation
        (a fresh key, and therefore a fresh full per-key bucket, every time) can
        exceed it."""
        self._global_tokens = min(
            self._global_capacity,
            self._global_tokens + (now - self._global_last_refill) * self._global_refill_per_second,
        )
        self._global_last_refill = now
        if self._global_tokens < 1.0:
            return False
        self._global_tokens -= 1.0
        return True

    def _evict_one_full_bucket_locked(self, now: float) -> bool:
        """Caller must hold `self._lock`. X-5: eviction must never reset a
        *throttled* client's budget -- the previous `popitem(last=False)` evicted
        whichever bucket was oldest regardless of its token level, and a fresh key
        defaults to a full bucket (see the old `.pop(key, (capacity, now))`), so
        evicting a throttled client's entry was indistinguishable from handing it a
        brand new full bucket. Only a bucket that has refilled to (approximately)
        full capacity -- meaning that client has been idle for the entire window,
        equivalent to one that was never tracked at all -- is safe to evict for
        free. Returns False, evicting nothing, if every tracked bucket is currently
        below capacity (i.e. every existing key is actively throttled): the caller
        must shed load (deny the new key) rather than admit it by force-evicting a
        throttled client.
        """
        for key, (tokens, last_refill) in list(self._buckets.items()):
            projected = min(self.capacity, tokens + (now - last_refill) * self.refill_per_second)
            if projected >= self.capacity - 1e-9:
                del self._buckets[key]
                return True
        return False

    def _consume(self, client_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if not self._consume_global_locked(now):
                return False

            existing = self._buckets.get(client_key)
            if existing is None:
                if len(self._buckets) >= _MAX_RATE_LIMIT_BUCKETS:
                    if not self._evict_one_full_bucket_locked(now):
                        # X-5: shed load instead of admitting a new key by forcibly
                        # evicting a still-throttled one.
                        return False
                tokens, last_refill = self.capacity, now
            else:
                tokens, last_refill = existing

            tokens = min(self.capacity, tokens + (now - last_refill) * self.refill_per_second)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            self._buckets[client_key] = (tokens, now)
            self._buckets.move_to_end(client_key)
            return allowed

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or scope.get("path") in _ORCHESTRATION_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        client_key, client_was_none = _resolve_client_key(scope, self.trust_proxy_header)
        if client_was_none and not self._warned_no_client:
            logger.warning(
                "RateLimitMiddleware: scope['client'] is None and no trusted proxy "
                "header is configured (PAW_TRUST_PROXY_HEADER); every such client "
                "shares one rate-limit bucket instead of being limited "
                "individually. If this server is behind a reverse proxy you "
                "trust to set X-Forwarded-For honestly, set "
                "PAW_TRUST_PROXY_HEADER=1 so real client addresses are used instead."
            )
            self._warned_no_client = True

        if not self._consume(client_key):
            response = Response(
                status_code=429,
                content="Rate limit exceeded, try again shortly",
                headers={"Retry-After": "1"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


class AuthMiddleware:
    """Pure ASGI middleware performing bearer-token authentication ahead of body
    parsing (X-2).

    Positioned *inside* `CORSMiddleware` but *outside* `PayloadSizeLimitMiddleware`
    in the registered stack (see create_app's ordering comment):

    - Ahead of `PayloadSizeLimitMiddleware` so an unauthenticated caller is rejected
      before any body buffering, JSON parsing or Pydantic validation ever runs --
      the actual cost X-2 is about, none of which is avoided by authenticating
      merely "before the route body executes" the way the old per-route
      `_verify_auth()` calls did.
    - *Inside* `CORSMiddleware` (not the literal outermost layer) because
      Starlette's `CORSMiddleware` answers a valid cross-origin preflight `OPTIONS`
      request itself, without ever calling the wrapped app -- so this middleware
      never even sees a preflight request when CORS is configured. Placing auth
      *outside* CORS instead would reject every such preflight with 401, since
      browsers never attach `Authorization` to one, breaking cross-origin use of
      the API whenever both `PAW_CORS_ORIGINS` and an API key are configured (the
      expected production configuration, not an edge case).

    `/health` and `/ready` are exempt, matching `RateLimitMiddleware`.
    """

    def __init__(
        self,
        app: Any,
        configured_api_key: Optional[str],
        state: "ServerState",
    ) -> None:
        self.app = app
        self.configured_api_key = configured_api_key
        self.state = state

    async def __call__(self, scope: Dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http" or not self.configured_api_key:
            await self.app(scope, receive, send)
            return
        if scope.get("path") in _ORCHESTRATION_EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        auth = headers.get("authorization")
        failure_reason: Optional[str] = None
        if not auth or not auth.startswith("Bearer "):
            failure_reason = "missing or malformed Bearer token"
        else:
            token = auth[7:].strip()
            try:
                # X-6: compare bytes, never `str` -- `hmac.compare_digest` raises
                # `TypeError` on a `str` argument containing non-ASCII characters
                # (reachable here because ASGI header values are decoded via
                # latin-1, so an attacker-supplied byte >= 0x80 survives as a
                # non-ASCII `str` character), which previously escaped
                # `_verify_auth` as an unhandled 500 (a full traceback per attempt
                # via `ServerErrorMiddleware`) and was never recorded as an error.
                # The whole comparison is additionally wrapped so *any* failure --
                # not just this specific, known one -- becomes a 401, never a 500.
                token_ok = hmac.compare_digest(
                    token.encode("utf-8"), self.configured_api_key.encode("utf-8")
                )
            except Exception:
                token_ok = False
            if not token_ok:
                failure_reason = "invalid API key"

        if failure_reason is not None:
            # X-9: previously neither logged nor counted at all -- /metrics read
            # total_requests: 0 after 25 failed attempts. Counted through the same
            # counters every other failure uses; logged at WARNING, aggregated (a
            # running total, not per-attempt detail) and never including the
            # presented token or any other header content.
            self.state.record_auth_failure()
            logger.warning(
                "Authentication failed (%s) for %s %s (auth failures so far: %d)",
                failure_reason,
                scope.get("method"),
                scope.get("path"),
                self.state.get_auth_failure_count(),
            )
            response = Response(
                status_code=401,
                content=f"Unauthorized: {failure_reason[0].upper()}{failure_reason[1:]}",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


# X-7: previously this bounded the ADMISSION wait -- `semaphore.acquire(timeout=...)`
# -- which only started counting once FastAPI's sync-route dispatch had already
# granted a worker thread from AnyIO's own (separately bounded, separately
# unbounded-wait) threadpool. That thread-token wait was itself unbounded, so a
# request could occupy a worker thread for up to the old 30s *after* an unbounded
# queueing delay (50s+ end to end), starving the sync /docs, /redoc and
# /openapi.json routes of worker threads in the meantime (see PAW-SERVE-04).
#
# Admission is now immediate (see _execute_with_telemetry: a non-blocking check on
# the event loop, before any worker thread is ever requested) -- a caller either gets
# the single inference slot on the spot or is told 503 right away, consuming no
# worker thread either way. This constant now bounds something new instead: how long
# an *admitted* inference is allowed to run before the caller is failed and the slot
# is released, so one hung backend call cannot occupy the slot -- and thus fail every
# subsequent caller with an instant 503 -- forever. Reduced from 30.0 accordingly: a
# real inference finishing well under it (see measurements/README.md) makes 20s ample
# headroom without the old figure's tail-latency cost.
#
# Caveat, documented rather than solved: `asyncio.wait_for` cancels the *waiting*
# coroutine, not the underlying OS thread `run_in_threadpool` dispatched the blocking
# call to -- Python cannot forcibly stop a running thread. A hung `work()` call keeps
# running in the background after this fires; the slot is still released promptly (so
# new callers are not blocked on it), but the abandoned thread itself is not reclaimed
# until the call eventually returns on its own.
_INFERENCE_SLOT_TIMEOUT_SECONDS = 20.0


class InferenceSlot:
    """A single-permit gate whose acquisition never waits (X-7): exactly one caller
    holds it at a time, and every other caller is told immediately -- on the event
    loop, before any worker thread is requested -- that it is busy.

    Deliberately not `threading.BoundedSemaphore`/`asyncio.Semaphore`: both support a
    *blocking* acquire (with or without a timeout), and the whole point of X-7's fix
    is that admission never blocks at all. A single boolean held under the GIL is
    sufficient here (and trivially correct) because asyncio's event loop is
    single-threaded and cooperative: `try_acquire` contains no `await`, so it cannot
    be interleaved with another coroutine's `try_acquire`/`release` call.
    """

    def __init__(self) -> None:
        self._held = False

    def try_acquire(self) -> bool:
        if self._held:
            return False
        self._held = True
        return True

    def release(self) -> None:
        self._held = False


async def _execute_with_telemetry(
    work: Callable[[], Any],
    slot: InferenceSlot,
    state: ServerState,
    route_name: str,
) -> "tuple[Any, float]":
    """Run `work()` under the single bounded inference slot, recording telemetry on
    every outcome and converting any failure — admission overflow, execution timeout,
    or an inference exception — into an `HTTPException` instead of letting it
    propagate as an unhandled 500 that bypasses `state.record_request(is_error=True)`
    (PAW-SERVE-05).

    X-7: admission (`slot.try_acquire()`) is a non-blocking check run directly on the
    event loop, before `work()` is ever dispatched to a worker thread via
    `run_in_threadpool` -- see `InferenceSlot`'s docstring and
    `_INFERENCE_SLOT_TIMEOUT_SECONDS`'s comment for why this replaces the previous
    blocking-with-timeout `threading.BoundedSemaphore.acquire(timeout=...)`.

    This does not touch `/health`/`/metrics`, which no longer route through this
    function at all (see PAW-SERVE-04's `async def` fix) and so cannot be starved by
    inference contention regardless of this timeout's value.
    """
    t0 = time.perf_counter()
    if not slot.try_acquire():
        state.record_request((time.perf_counter() - t0) * 1000.0, is_error=True)
        raise HTTPException(
            status_code=503,
            detail="Server busy: too many concurrent inference requests, try again shortly",
        )
    try:
        try:
            res = await asyncio.wait_for(
                run_in_threadpool(work), timeout=_INFERENCE_SLOT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=True)
            logger.exception(
                "Inference exceeded its %.1fs execution bound in %s",
                _INFERENCE_SLOT_TIMEOUT_SECONDS,
                route_name,
            )
            raise HTTPException(
                status_code=503,
                detail="Server busy: inference exceeded its execution bound, try again shortly",
            ) from exc
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=True)
            logger.exception("Inference failed in %s: %s", route_name, exc)
            raise HTTPException(status_code=500, detail="Internal server error") from exc
    finally:
        slot.release()
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


def _run_warmup(backend: AbstractPAWBackend, adapter_path: str, state: "ServerState") -> None:
    """Run one inference with a fixed short input so `/ready` is 200 before the server
    starts accepting real traffic, instead of 503 until the first request completes.

    Calls `backend.infer(...)` directly -- *not* the schema-validating `load()`
    wrapper `exec_fn` may be built from when `response_model` is set. The point of a
    warm-up is to pay the one-time cost of loading the base model into memory; the
    fixed `_WARMUP_INPUT` ("ping") is not expected to produce output that parses as an
    arbitrary caller-supplied `response_model`, and warming through that wrapper made
    `--warm` fail (leaving `/ready` at 503 indefinitely) on essentially every adapter
    that set one.

    A failure here is logged, not raised: `--warm` is a latency optimization, and a
    broken warm-up must not crash `paw-serve` on startup -- it just leaves `/ready` at
    503 until a real request succeeds (or never, if the backend is genuinely broken,
    which every other endpoint will also report).
    """
    t0 = time.perf_counter()
    try:
        backend.infer(adapter_path, _WARMUP_INPUT)
    except Exception as exc:
        logger.warning(
            "Warm-up inference failed after %.2fs: %s", time.perf_counter() - t0, exc
        )
        return
    elapsed = time.perf_counter() - t0
    logger.info("Warm-up inference completed in %.2fs", elapsed)
    state.mark_ready()


# X-2: lowered from the old 10MB. Any legitimate /invoke, /v1/chat/completions or
# /v1/messages payload is text, and the new `_MAX_MESSAGES_PER_REQUEST` field
# constraint (paw_kit/serve/models.py) already bounds how many messages a request can
# carry; 2MB leaves generous headroom over that while sharply cutting the amount of
# work (buffering, then JSON parsing, then Pydantic validation) an oversized body can
# force before AuthMiddleware -- now ahead of all of it -- has even run.
_MAX_BODY_BYTES = 2 * 1024 * 1024


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
    if configured_api_key:
        # X-12: `_verify_auth`'s bearer-token comparison strips the *incoming*
        # token (RFC 6750's "Bearer <token>" -- see AuthMiddleware), but never
        # stripped the *configured* key, so a PAW_API_KEY/--api-key value with
        # stray leading/trailing whitespace can never be presented back
        # successfully by any client that sends exactly the value it was given --
        # the operator sees only "Invalid API key" with no indication why.
        if configured_api_key != configured_api_key.strip():
            logger.warning(
                "The configured API key has leading or trailing whitespace, which "
                "the incoming bearer token comparison strips but the configured "
                "key itself does not -- a client presenting the key exactly as "
                "configured will never authenticate. Remove the surrounding "
                "whitespace from PAW_API_KEY / --api-key."
            )
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
    # X-7: a single-permit, never-blocking gate instead of a `threading.BoundedSemaphore`
    # -- see InferenceSlot's docstring.
    inference_slot = InferenceSlot()

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
        _run_warmup(selected_backend, str(path_obj), state)

    # Middleware ordering (outermost first -- listed in the order the *list* passed
    # to FastAPI(middleware=...) below must have them, which is the order they
    # actually run in for an incoming request):
    #   CORS -> RateLimit -> Auth -> PayloadSizeLimit -> routes.
    #
    # X-1: the previous code built this stack with `app.add_middleware(...)` calls,
    # whose `insert(0, ...)` semantics make the *last*-registered middleware
    # outermost -- the exact inverse of the call order used, and the exact inverse
    # of what the comment at this location used to claim. `middleware=[...]` passed
    # directly to FastAPI's constructor has no such trap: the list is used in the
    # order given, so "the order below" and "the order above" cannot disagree again.
    #
    # Rationale for each position:
    # - CORS is outermost so it decides same-origin/allowlisted-origin handling
    #   first, cheaply, before any other work -- and so it can answer a cross-origin
    #   preflight OPTIONS request itself, without ever invoking anything below it.
    #   That second property is *why* Auth is not outermost instead (see
    #   AuthMiddleware's docstring and the Implementation overview this track was
    #   written against): browsers never attach Authorization to a preflight, so an
    #   auth layer outside CORS would 401 every one of them whenever both
    #   PAW_CORS_ORIGINS and an API key are configured.
    # - RateLimit runs next so an unauthenticated flood (e.g. brute-forcing the
    #   bearer token, see RateLimitMiddleware's own docstring) is throttled before
    #   Auth ever inspects it, not just before inference runs.
    # - Auth (X-2) runs ahead of PayloadSizeLimit specifically so an unauthenticated
    #   caller is rejected before any byte of the body is buffered, let alone JSON-
    #   parsed or Pydantic-validated -- the actual cost X-2 is about.
    # - PayloadSizeLimit stays innermost, closest to the routes it protects.
    middleware_stack: List[Middleware] = []

    # PAW-SERVE-02: no wildcard CORS default. Cross-origin access is opt-in only, via a
    # comma-separated PAW_CORS_ORIGINS allowlist; absent that, no CORS middleware is
    # installed at all, so browsers deny cross-origin requests outright.
    cors_env = os.environ.get("PAW_CORS_ORIGINS", "")
    allowed_origins = [origin.strip() for origin in cors_env.split(",") if origin.strip()]
    if allowed_origins:
        middleware_stack.append(
            Middleware(
                CORSMiddleware,
                allow_origins=allowed_origins,
                allow_credentials=False,
                allow_methods=["GET", "POST", "OPTIONS"],
                allow_headers=["Content-Type", "Authorization"],
            )
        )

    # PAW-SERVE-10 / X-8: bound the request rate per client. `requests_per_minute=0`
    # (explicitly, as a caller-supplied parameter) disables the limiter deliberately
    # (e.g. for trusted internal benchmarking); an unset parameter falls back to
    # PAW_RATE_LIMIT_PER_MINUTE and then the built-in default, with a bad env value
    # falling back cleanly instead of crashing (see _int_env_with_fallback).
    requests_per_minute = _int_env_with_fallback(
        "PAW_RATE_LIMIT_PER_MINUTE", _DEFAULT_RATE_LIMIT_PER_MINUTE, requests_per_minute
    )
    if requests_per_minute > 0:
        global_requests_per_minute = _int_env_with_fallback(
            "PAW_RATE_LIMIT_GLOBAL_PER_MINUTE", _DEFAULT_GLOBAL_RATE_LIMIT_PER_MINUTE
        )
        # X-4: PAW_TRUST_PROXY_HEADER is the new trusted-proxy config surface --
        # opt-in only, since trusting X-Forwarded-For from a client that is not
        # actually behind a trusted reverse proxy lets that client spoof any
        # address it likes and evade the limiter entirely.
        trust_proxy_header = os.environ.get("PAW_TRUST_PROXY_HEADER", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        middleware_stack.append(
            Middleware(
                RateLimitMiddleware,
                requests_per_minute=requests_per_minute,
                trust_proxy_header=trust_proxy_header,
                global_requests_per_minute=global_requests_per_minute,
            )
        )
    else:
        # X-8: previously silent -- an operator could end up with no rate limiting
        # at all (requests_per_minute<=0, from either an explicit 0 or a negative
        # env value now clamped to 0) with no indication anything was disabled.
        logger.warning(
            "Rate limiting is disabled (requests_per_minute=%d); this server has "
            "no per-client request throttle.",
            requests_per_minute,
        )

    # X-2: authenticate ahead of body parsing -- see AuthMiddleware's docstring for
    # why this sits inside CORS but outside PayloadSizeLimit specifically.
    middleware_stack.append(
        Middleware(AuthMiddleware, configured_api_key=configured_api_key, state=state)
    )

    # PAW-SERVE-03 / X-2 / X-3: enforce the body-size cap (now lowered, see
    # _MAX_BODY_BYTES) and the total body-read deadline via pure ASGI middleware
    # (see PayloadSizeLimitMiddleware's docstring for why BaseHTTPMiddleware cannot
    # do this safely without emptying the request body for every downstream handler).
    middleware_stack.append(
        Middleware(PayloadSizeLimitMiddleware, max_body_bytes=_MAX_BODY_BYTES)
    )

    # PAW-SERVE-06: the default docs routes are disabled here and re-registered below,
    # behind the same auth policy as everything but `/health`/`/ready`.
    app = FastAPI(
        title="PAW-Kit Microservice",
        description=f"Zero-marginal-cost neural microservice serving {path_obj.name}",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        middleware=middleware_stack,
    )

    # PAW-SERVE-04: `/health` and `/metrics` are `async def` so FastAPI runs them
    # directly on the event loop instead of dispatching to the AnyIO worker threadpool
    # the three inference routes below block a slot in. An exhausted threadpool (a burst
    # of concurrent inference calls each waiting on a worker thread) can then no
    # longer starve the liveness probe or the metrics scrape alongside it. The three
    # inference routes are `async def` too (X-7), but only to run admission -- a
    # non-blocking check, see InferenceSlot -- directly on the event loop; the actual
    # blocking `exec_fn` call is still dispatched to a worker thread explicitly via
    # `run_in_threadpool` inside `_execute_with_telemetry`, so inference itself still
    # never runs on the event loop and cannot stall other requests.
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        # PAW-SERVE-06: status/uptime/version only — see HealthResponse's docstring.
        return HealthResponse(status="ok", uptime_seconds=state.get_uptime(), version="0.1.0")

    @app.get("/ready")
    async def ready_check() -> JSONResponse:
        # Readiness, distinct from `/health`'s liveness: 200 only once the adapter has
        # completed one successful inference (a real request or an explicit `--warm`
        # warm-up, see ServerState.mark_ready) -- not merely once the process is up and
        # accepting connections. Same auth exemption and rate-limit exemption
        # (`_ORCHESTRATION_EXEMPT_PATHS`) as `/health`, for orchestration to poll freely.
        if state.is_ready():
            return JSONResponse({"status": "ready"}, status_code=200)
        return JSONResponse({"status": "loading"}, status_code=503)

    @app.get("/metrics", response_model=MetricsResponse)
    async def telemetry_metrics() -> MetricsResponse:
        # PAW-SERVE-06 / X-2: /metrics is not a liveness probe and discloses
        # request-volume and latency telemetry, so it follows the same auth policy
        # as everything but /health and /ready -- enforced by AuthMiddleware ahead
        # of this route running at all, not by a call inside it (see X-2).
        return MetricsResponse(**state.get_metrics())

    @app.get("/openapi.json", include_in_schema=False)
    def openapi_json() -> JSONResponse:
        # PAW-SERVE-06 / X-2: the generated OpenAPI schema can describe internal
        # endpoint shapes to an unauthenticated caller; gated like /docs and /redoc
        # by AuthMiddleware ahead of this route, same as /metrics above.
        return JSONResponse(app.openapi())

    @app.get("/docs", include_in_schema=False)
    def swagger_docs() -> HTMLResponse:
        return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Swagger UI")

    @app.get("/redoc", include_in_schema=False)
    def redoc_docs() -> HTMLResponse:
        return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

    @app.post("/invoke", response_model=InvokeResponse)
    async def invoke(req: InvokeRequest) -> InvokeResponse:
        res, latency = await _execute_with_telemetry(
            lambda: exec_fn(req.input), inference_slot, state, "/invoke"
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
    async def chat_completions(req: ChatCompletionRequest) -> ChatCompletionResponse:
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

        (res, input_payload), latency = await _execute_with_telemetry(
            _run, inference_slot, state, "/v1/chat/completions"
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
    async def anthropic_messages(req: AnthropicMessageRequest) -> AnthropicMessageResponse:
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

        (res, input_payload), latency = await _execute_with_telemetry(
            _run, inference_slot, state, "/v1/messages"
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

    X-3: this calls `uvicorn.run(app, host=host, port=port)` with no
    `limit_concurrency` -- Uvicorn's own server-wide cap on simultaneously accepted
    connections, distinct from `PayloadSizeLimitMiddleware`'s per-request body-read
    deadline (which bounds one stalled request, not how many can be open at once).
    `paw-serve` does not thread a flag through to it today; an operator who wants
    that ceiling should run `uvicorn` directly with `--limit-concurrency` instead of
    going through `paw-serve`/`serve_adapter`.
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
