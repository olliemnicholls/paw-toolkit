"""Unit and integration tests for paw_kit.serve: OpenAI and Anthropic HTTP serving layer and Docker exporter."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest
from typer.testing import CliRunner

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.cli import app
from paw_kit.jit.db import TraceDB
from paw_kit.serve.docker import export_docker_scaffold
from paw_kit.serve.server import ServerState, create_app


class TicketSchema(BaseModel):
    priority: str
    confidence: float


@pytest.fixture
def mock_adapter(tmp_path: Path) -> Path:
    """Create a temporary compiled mock adapter."""
    adapter_file = tmp_path / "triage.paw"
    backend = MockPAWBackend()
    backend.compile(
        spec="Classify ticket priority",
        examples=[
            {
                "input": "Urgent payment failure",
                "output": json.dumps({"priority": "high", "confidence": 0.98}),
            },
            {
                "input": "You are a customer support classifier.\n\nUrgent payment failure",
                "output": json.dumps({"priority": "high", "confidence": 0.98}),
            },
            {
                "input": "You are an automated triage agent.\n\nUrgent payment failure",
                "output": json.dumps({"priority": "high", "confidence": 0.98}),
            },
        ],
        output_path=str(adapter_file),
    )
    return adapter_file


def test_health_and_metrics_endpoints(mock_adapter: Path) -> None:
    """Verify /health and /metrics report proper uptime, status, and telemetry without leaking host paths."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0.0
    # PAW-SERVE-06: /health is stripped to status/uptime/version only — no adapter
    # path or backend implementation detail leaks through the unauthenticated route.
    assert "backend" not in data
    assert "adapter_path" not in data

    # Initial metrics
    m_res = client.get("/metrics")
    assert m_res.status_code == 200
    m_data = m_res.json()
    assert m_data["total_requests"] == 0
    assert m_data["error_count"] == 0
    assert m_data["p50_latency_ms"] == 0.0


def test_invoke_endpoint(mock_adapter: Path) -> None:
    """Verify direct RPC /invoke executes adapter and tracks request latency."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    res = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res.status_code == 200
    data = res.json()
    assert data["adapter"] == "triage.paw"
    assert data["latency_ms"] >= 0.0
    assert data["output"]["priority"] == "high"

    # Verify metrics incremented
    m_res = client.get("/metrics")
    assert m_res.json()["total_requests"] == 1


def test_openai_chat_completions(mock_adapter: Path) -> None:
    """Verify POST /v1/chat/completions adheres to OpenAI specification."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    payload = {
        "model": "paw-triage-v1",
        "messages": [
            {"role": "system", "content": "You are a customer support classifier."},
            {"role": "user", "content": "Urgent payment failure"},
        ],
        "temperature": 0.0,
    }

    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"].startswith("chatcmpl-")
    assert data["object"] == "chat.completion"
    assert data["model"] == "paw-triage-v1"
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    parsed_content = json.loads(choice["message"]["content"])
    assert parsed_content["priority"] == "high"
    assert choice["finish_reason"] == "stop"
    assert data["usage"]["total_tokens"] > 0


def test_anthropic_messages_endpoint(mock_adapter: Path) -> None:
    """Verify POST /v1/messages adheres to Anthropic Messages specification."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    payload = {
        "model": "paw-triage-claude",
        "system": "You are an automated triage agent.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Urgent payment failure"}],
            }
        ],
        "max_tokens": 512,
    }

    res = client.post("/v1/messages", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["id"].startswith("msg_")
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["model"] == "paw-triage-claude"
    assert data["stop_reason"] == "end_turn"
    assert len(data["content"]) == 1
    assert data["content"][0]["type"] == "text"
    parsed_text = json.loads(data["content"][0]["text"])
    assert parsed_text["priority"] == "high"
    assert data["usage"]["input_tokens"] > 0
    assert data["usage"]["output_tokens"] > 0


def test_schema_enforcement_in_server(mock_adapter: Path) -> None:
    """Verify typed response_model enforces schema decoding across all protocols."""
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter,
        backend=backend,
        response_model=TicketSchema,
        allow_anonymous=True,
    )
    client = TestClient(fastapi_app)

    # 1. /invoke returns validated dict
    res_inv = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res_inv.status_code == 200
    assert res_inv.json()["output"] == {"priority": "high", "confidence": 0.98}

    # 2. OpenAI returns JSON string conforming to TicketSchema
    res_oai = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Urgent payment failure"}]},
    )
    assert res_oai.status_code == 200
    oai_content = json.loads(res_oai.json()["choices"][0]["message"]["content"])
    assert TicketSchema.model_validate(oai_content)

    # 3. Anthropic returns text conforming to TicketSchema
    res_claude = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "Urgent payment failure"}]},
    )
    assert res_claude.status_code == 200
    claude_text = json.loads(res_claude.json()["content"][0]["text"])
    assert TicketSchema.model_validate(claude_text)


def test_error_handling_and_validation(mock_adapter: Path) -> None:
    """Verify proper error responses, generic 500 error sanitization, and metric tracking."""
    # Non-existent adapter file
    with pytest.raises(FileNotFoundError):
        create_app(Path("/non/existent/path.paw"))

    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    # Empty messages in OpenAI format
    res_oai = client.post("/v1/chat/completions", json={"messages": []})
    assert res_oai.status_code == 400

    # Empty messages in Anthropic format
    res_ant = client.post("/v1/messages", json={"messages": []})
    assert res_ant.status_code == 400

    # Runtime backend error — verify C-1: generic error message returned, no exception leakage
    def failing_backend(*args, **kwargs):
        raise RuntimeError("Secret internal database connection string: postgres://root:pass@db/internal")

    broken_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    backend.infer = failing_backend  # type: ignore
    broken_client = TestClient(broken_app)

    res_err = broken_client.post("/invoke", json={"input": "fail"})
    assert res_err.status_code == 500
    assert res_err.json()["detail"] == "Internal server error"
    assert "postgres" not in res_err.text

    m = broken_client.get("/metrics").json()
    assert m["error_count"] >= 1


def test_stream_rejected_explicitly(mock_adapter: Path) -> None:
    """Verify M-5: stream=True is rejected with clear 400 instead of silently returning non-streamed JSON."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    payload = {
        "messages": [{"role": "user", "content": "Urgent payment failure"}],
        "stream": True,
    }
    res = client.post("/v1/chat/completions", json=payload)
    assert res.status_code == 400
    assert "Streaming is not yet supported" in res.json()["detail"]


def test_api_key_authentication(mock_adapter: Path) -> None:
    """Verify H-2: Optional API key authentication guards all inference endpoints (S-1, S-6)."""
    backend = MockPAWBackend()
    auth_app = create_app(mock_adapter, backend=backend, api_key="secret-api-key-999")
    client = TestClient(auth_app)

    # 1. Health remains open for monitoring (Dockerfile HEALTHCHECK relies on this).
    assert client.get("/health").status_code == 200
    # 2. PAW-SERVE-06: /metrics is not a liveness probe and discloses request-volume
    # and latency telemetry, so it now follows the same auth policy as inference.
    assert client.get("/metrics").status_code == 401
    assert (
        client.get("/metrics", headers={"Authorization": "Bearer secret-api-key-999"}).status_code
        == 200
    )

    # 2. Missing authorization header on /invoke
    assert client.post("/invoke", json={"input": "test"}).status_code == 401

    # 3. Invalid token on /invoke
    assert client.post(
        "/invoke",
        json={"input": "test"},
        headers={"Authorization": "Bearer wrong-key"},
    ).status_code == 401

    # 4. Missing auth on /v1/chat/completions (S-6)
    assert client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
    ).status_code == 401

    # 5. Missing auth on /v1/messages (S-6)
    assert client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "test"}]},
    ).status_code == 401

    # 6. Valid token succeeds on all endpoints
    res_inv = client.post(
        "/invoke",
        json={"input": "Urgent payment failure"},
        headers={"Authorization": "Bearer secret-api-key-999"},
    )
    assert res_inv.status_code == 200

    res_oai = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Urgent payment failure"}]},
        headers={"Authorization": "Bearer secret-api-key-999"},
    )
    assert res_oai.status_code == 200

    res_msg = client.post(
        "/v1/messages",
        json={"messages": [{"role": "user", "content": "Urgent payment failure"}]},
        headers={"Authorization": "Bearer secret-api-key-999"},
    )
    assert res_msg.status_code == 200


def test_serve_requires_auth_by_default_PAW_SERVE_01(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify PAW-SERVE-01: with no --api-key/PAW_API_KEY and no --allow-anonymous,
    inference is denied by default and an ephemeral bearer token is generated and
    printed to stderr, usable to authenticate."""
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    backend = MockPAWBackend()
    app_default_deny = create_app(mock_adapter, backend=backend)
    client = TestClient(app_default_deny)

    captured = capsys.readouterr()
    assert "Generated ephemeral bearer token" in captured.err
    token = captured.err.split("bearer token: ")[1].split("\n")[0].strip()
    assert token

    # No credentials at all: denied.
    res_no_auth = client.post("/invoke", json={"input": "test"})
    assert res_no_auth.status_code == 401

    # Wrong credentials: still denied.
    res_wrong = client.post(
        "/invoke", json={"input": "test"}, headers={"Authorization": "Bearer not-the-token"}
    )
    assert res_wrong.status_code == 401

    # The printed ephemeral token itself authenticates successfully.
    res_ok = client.post(
        "/invoke",
        json={"input": "Urgent payment failure"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res_ok.status_code == 200


def test_serve_allow_anonymous_disables_default_deny_PAW_SERVE_01(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Verify --allow-anonymous opts back out of PAW-SERVE-01's default-deny behavior."""
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    backend = MockPAWBackend()
    app_anonymous = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(app_anonymous)

    captured = capsys.readouterr()
    assert "ephemeral bearer token" not in captured.err

    res = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res.status_code == 200


def test_serve_cors_default_denies_cross_origin_PAW_SERVE_02(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-SERVE-02: with no PAW_CORS_ORIGINS set, no CORS headers are issued
    at all, so browsers deny cross-origin access by default."""
    monkeypatch.delenv("PAW_CORS_ORIGINS", raising=False)
    backend = MockPAWBackend()
    app_no_cors = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(app_no_cors)

    res = client.get("/health", headers={"Origin": "https://evil.example"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers.keys()}


def test_serve_cors_allowlist_env_PAW_SERVE_02(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-SERVE-02: PAW_CORS_ORIGINS grants only the listed origin(s)."""
    monkeypatch.setenv("PAW_CORS_ORIGINS", "https://good.example")
    backend = MockPAWBackend()
    app_with_cors = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(app_with_cors)

    res_allowed = client.get("/health", headers={"Origin": "https://good.example"})
    assert res_allowed.headers.get("access-control-allow-origin") == "https://good.example"

    res_denied = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in {k.lower() for k in res_denied.headers.keys()}


def test_payload_size_limit_middleware(mock_adapter: Path) -> None:
    """Verify M-1 and S-2: Enforce request payload limits, invalid headers, and oversized bodies."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    # 1. Simulate 11MB Content-Length header
    res_header = client.post(
        "/invoke",
        headers={"Content-Length": str(11 * 1024 * 1024)},
        json={"input": "test"},
    )
    assert res_header.status_code == 413
    assert "Payload Too Large" in res_header.text

    # 2. Malformed non-numeric Content-Length header (S-2)
    res_malformed = client.post(
        "/invoke",
        headers={"Content-Length": "not-a-number"},
        json={"input": "test"},
    )
    assert res_malformed.status_code == 400
    assert "Invalid Content-Length" in res_malformed.text

    # 3. Oversized body without Content-Length header (chunked simulation)
    large_payload = b"a" * (11 * 1024 * 1024)
    res_body = client.post(
        "/invoke",
        content=large_payload,
    )
    assert res_body.status_code == 413
    assert "Payload Too Large" in res_body.text


def test_serve_payload_limit_asgi_streaming_PAW_SERVE_03(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-03: the ASGI-level rewrite of limit_payload_size correctly
    rejects a truly streamed oversized body with no Content-Length header (proving the
    running byte counter works, not just the header pre-check), while a normal-size
    streamed body still reaches the route handler intact — the exact regression a naive
    BaseHTTPMiddleware + request.stream() rewrite would fail (Phase 0 Round 1, N-1)."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    def oversized_chunks():
        chunk = b"a" * (1024 * 1024)  # 1MB per chunk
        for _ in range(12):  # 12MB total, no Content-Length known upfront
            yield chunk

    res_over = client.post("/invoke", content=oversized_chunks())
    assert "content-length" not in {k.lower() for k in res_over.request.headers.keys()}
    assert res_over.status_code == 413
    assert "Payload Too Large" in res_over.text

    def normal_chunks():
        body = json.dumps({"input": "Urgent payment failure"}).encode("utf-8")
        midpoint = len(body) // 2
        yield body[:midpoint]
        yield body[midpoint:]

    res_normal = client.post(
        "/invoke",
        content=normal_chunks(),
        headers={"Content-Type": "application/json"},
    )
    assert "content-length" not in {k.lower() for k in res_normal.request.headers.keys()}
    assert res_normal.status_code == 200
    assert res_normal.json()["output"]["priority"] == "high"


# ---------------------------------------------------------------- X-2 / X-1 / X-3: Group 1
# (auth/middleware-ordering/buffering reordering -- see the track's Implementation
# overview for why these three land together.)


def test_serve_auth_before_body_parsing_rejects_oversized_body_401_X_2(
    mock_adapter: Path,
) -> None:
    """Verify X-2: authentication runs ahead of body buffering, JSON parsing and
    Pydantic validation -- an oversized, malformed body with no Authorization header
    must be rejected 401 before any of that expensive work runs, not 422/413. At
    `main`, `_verify_auth` runs as the first statement *inside* the route body, which
    is downstream of FastAPI's own body parsing/validation, so a malformed or
    oversized unauthenticated body gets 422 or 413 there instead."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="secret")
    client = TestClient(fastapi_app)

    # Oversized (bigger than the new lower cap) AND malformed (not valid JSON), no
    # credentials at all.
    oversized_malformed = b"{" + b"not-json" * (1024 * 1024)
    res = client.post(
        "/v1/chat/completions",
        content=oversized_malformed,
        headers={"Content-Type": "application/json"},
    )
    assert res.status_code == 401

    # Well-formed JSON, but with an oversized `messages` array (see the field
    # constraint test below) -- still no credentials, still 401, not 422.
    too_many_messages = {"messages": [{"role": "user", "content": "hi"} for _ in range(600)]}
    res2 = client.post("/v1/chat/completions", json=too_many_messages)
    assert res2.status_code == 401


def test_serve_messages_array_bounded_by_field_constraint_X_2(mock_adapter: Path) -> None:
    """Verify X-2's models.py half: `ChatCompletionRequest`/`AnthropicMessageRequest`
    reject an oversized `messages` array with a 422 field-level validation error
    (once authenticated) -- proving the constraint actually exists and fires, since
    the 401-before-parsing test above can't distinguish "no constraint" from "never
    reached the constraint"."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k")
    client = TestClient(fastapi_app)

    too_many_messages = [{"role": "user", "content": "hi"} for _ in range(600)]

    res_oai = client.post(
        "/v1/chat/completions",
        json={"messages": too_many_messages},
        headers={"Authorization": "Bearer k"},
    )
    assert res_oai.status_code == 422

    res_claude = client.post(
        "/v1/messages",
        json={"messages": too_many_messages},
        headers={"Authorization": "Bearer k"},
    )
    assert res_claude.status_code == 422


def test_serve_middleware_registration_order_matches_docstring_X_1(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify X-1: `app.user_middleware`'s actual order matches create_app's own
    ordering comment (CORS -> RateLimit -> Auth -> PayloadSizeLimit -> routes). At
    `main`, `add_middleware`'s `insert(0, ...)` semantics make the *last*-registered
    middleware outermost -- the exact inverse of the call order used there, and the
    exact inverse of what the comment at that location claimed."""
    from paw_kit.serve.server import AuthMiddleware, PayloadSizeLimitMiddleware, RateLimitMiddleware

    monkeypatch.setenv("PAW_CORS_ORIGINS", "https://good.example")
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k", requests_per_minute=5)

    classes = [m.cls for m in fastapi_app.user_middleware]
    assert classes == [CORSMiddleware, RateLimitMiddleware, AuthMiddleware, PayloadSizeLimitMiddleware]


def test_serve_throttled_client_oversized_body_gets_429_not_413_X_1(mock_adapter: Path) -> None:
    """Verify X-1: with RateLimit correctly positioned ahead of PayloadSizeLimit, a
    throttled client sending an oversized body gets 429, not 413 -- at `main`,
    PayloadSizeLimitMiddleware was (due to the insert(0, ...) bug) the *outermost*
    middleware, so it saw -- and rejected -- an oversized body before the rate
    limiter ever got a chance to."""
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, allow_anonymous=True, requests_per_minute=1
    )
    client = TestClient(fastapi_app)

    assert client.post("/invoke", json={"input": "Urgent payment failure"}).status_code == 200

    # Bigger than *both* the new lower body cap (2MB) and the old 10MB one -- must
    # stay oversized regardless of which cap is in effect, so this genuinely
    # exercises the ordering (RateLimit ahead of PayloadSizeLimit), not just the cap
    # value: at `main`, an 11MB body is caught by the (there, outermost)
    # PayloadSizeLimitMiddleware before RateLimitMiddleware (there, innermost) ever
    # runs, giving 413 regardless of the caller's rate-limit state.
    large_payload = b"a" * (11 * 1024 * 1024)
    res = client.post("/invoke", content=large_payload)
    assert res.status_code == 429


def test_serve_cors_preflight_succeeds_without_auth_when_both_configured(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the Implementation overview's ordering decision (not a defect at
    `main`, since `_verify_auth` never ran for a preflight OPTIONS request either
    way -- a regression guard for the fix, added because no existing test exercised
    this combination): Auth sits inside CORSMiddleware, not outside it, so an
    unauthenticated cross-origin preflight is still answered by CORSMiddleware
    before AuthMiddleware ever runs. Placing auth outside CORS would 401 every
    preflight whenever both PAW_CORS_ORIGINS and an API key are configured -- the
    expected production configuration, not an edge case."""
    monkeypatch.setenv("PAW_CORS_ORIGINS", "https://good.example")
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="secret")
    client = TestClient(fastapi_app)

    res = client.options(
        "/invoke",
        headers={
            "Origin": "https://good.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "https://good.example"


def test_serve_body_read_timeout_returns_408_on_stall_X_3() -> None:
    """Verify X-3: a chunked request that sends one byte and then stalls past the
    read deadline gets 408, instead of being held open indefinitely (pre-auth,
    pre-rate-limit -- PayloadSizeLimitMiddleware buffers the body before either of
    those layers run, by design). Driven directly at the ASGI layer with a synthetic
    `receive` that stalls after the first chunk -- httpx's TestClient has no way to
    model a `receive()` call that never resolves within a bounded test run, so this
    is the same "manipulate the ASGI layer directly" style as the rate-limiter's
    `_consume()` tests, applied to this middleware's `__call__` instead."""
    from paw_kit.serve.server import PayloadSizeLimitMiddleware

    calls = 0

    async def receive() -> Dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": b"a", "more_body": True}
        await asyncio.sleep(1000)  # never actually reached within the 0.05s deadline
        raise AssertionError("unreachable")

    sent: list = []

    async def send(message: Dict[str, Any]) -> None:
        sent.append(message)

    async def inner_app(scope: object, receive: object, send: object) -> None:
        raise AssertionError("downstream app must not be reached: the body never completed")

    scope = {"type": "http", "headers": []}
    middleware = PayloadSizeLimitMiddleware(
        inner_app, max_body_bytes=10 * 1024 * 1024, body_read_timeout=0.05
    )

    asyncio.run(middleware(scope, receive, send))

    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 408


def test_serve_body_read_deadline_boundary_at_exactly_zero_X_3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify X-3's deadline pre-check (`if remaining <= 0`, at the top of the
    buffering loop) is inclusive of exactly zero remaining time, not just strictly
    negative -- distinct from the test above, which only exercises the
    `asyncio.wait_for(...)` timeout path (a few lines further down) and never hits
    this specific comparison with `remaining == 0`. Deterministic: only
    `paw_kit.serve.server`'s *own* module-level `time` name is replaced (not the
    global `time` module `sys.modules['time']` holds, which asyncio's event loop
    also relies on for its own scheduling) with a fake whose `monotonic()` returns
    a fixed two-value sequence chosen so `remaining` computes to exactly 0.0 on the
    loop's first pass -- real wall-clock timing could never reliably land on that
    exact boundary."""
    import paw_kit.serve.server as server_module
    from paw_kit.serve.server import PayloadSizeLimitMiddleware

    # Call 1: `deadline = time.monotonic() + body_read_timeout` (body_read_timeout
    # is 0.0, so deadline == 100.0). Call 2: the loop's `remaining = deadline -
    # time.monotonic()` -- also 100.0, so remaining == 0.0 exactly.
    clock_calls = iter([100.0, 100.0])
    fake_time = type("FakeTime", (), {"monotonic": staticmethod(lambda: next(clock_calls))})()
    monkeypatch.setattr(server_module, "time", fake_time)

    receive_calls = 0

    async def receive() -> Dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        raise AssertionError("receive() must not be called once the deadline is reached")

    sent: list = []

    async def send(message: Dict[str, Any]) -> None:
        sent.append(message)

    async def inner_app(scope: object, receive: object, send: object) -> None:
        raise AssertionError("downstream app must not be reached")

    scope = {"type": "http", "headers": []}
    middleware = PayloadSizeLimitMiddleware(
        inner_app, max_body_bytes=10 * 1024 * 1024, body_read_timeout=0.0
    )

    asyncio.run(middleware(scope, receive, send))

    assert receive_calls == 0
    start = next(m for m in sent if m["type"] == "http.response.start")
    assert start["status"] == 408


def test_format_max_body_message_integer_vs_fractional_mb_X_2() -> None:
    """Verify X-2's dynamic 413 message text pins both branches of
    `if mb == int(mb)`: an exact-megabyte cap renders as a bare integer ("2MB"),
    not "2.00MB", while a non-exact cap renders with two decimal places. Without
    this, a mutant flipping the comparison (or a future refactor) could silently
    start showing "2.00MB" for the default 2MB cap and nothing would notice --
    exactly the "message describes a limit other than the one enforced" class of
    bug X-2's own docstring calls out."""
    from paw_kit.serve.server import _format_max_body_message

    assert _format_max_body_message(2 * 1024 * 1024) == "Payload Too Large (maximum 2MB)"
    assert _format_max_body_message(int(1.5 * 1024 * 1024)) == "Payload Too Large (maximum 1.50MB)"


def test_serve_invoke_returns_503_when_inference_busy_PAW_SERVE_04(
    mock_adapter: Path,
) -> None:
    """Verify PAW-SERVE-04 / X-7: a caller that can't get the single inference slot
    gets an immediate 503 instead of waiting for it -- admission (X-7) is a
    non-blocking acquire_nowait check on the event loop, not a bounded wait, so no
    timeout needs configuring here at all -- and /health keeps responding throughout
    since it no longer shares the inference threadpool path."""

    class SlowBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            time.sleep(1.5)
            return super().infer(adapter_path, input_text, grammar_constraint)

    fastapi_app = create_app(mock_adapter, backend=SlowBackend(), allow_anonymous=True)
    client = TestClient(fastapi_app)

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow_future = pool.submit(client.post, "/invoke", json={"input": "Urgent payment failure"})
        time.sleep(0.2)  # let the slow request acquire the single semaphore slot first

        busy_future = pool.submit(client.post, "/invoke", json={"input": "Urgent payment failure"})

        # /health is async and never routes through the semaphore, so it stays live.
        assert client.get("/health").status_code == 200

        assert busy_future.result(timeout=5).status_code == 503
        assert slow_future.result(timeout=5).status_code == 200


def test_serve_inference_admission_bounded_under_load_X_7(mock_adapter: Path) -> None:
    """Verify X-7: admission to the single inference slot is immediate (acquire_nowait
    -- no worker thread is ever consumed queueing for it), so under N >> pool-size
    concurrent load, only one request actually runs inference; every other request's
    round trip stays bounded by a concrete p99 figure well under the old unbounded-
    queue / 30s-per-request behaviour, instead of piling up at 50s+ apiece. At
    `main`, the equivalent of this admission wait was `semaphore.acquire(timeout=30)`
    -- reached only *after* FastAPI's sync-route dispatch already granted a worker
    thread, itself an unbounded wait -- so this same N would have taken up to
    N * 30s serialized through the threadpool rather than completing near-instantly."""

    class SlowBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            time.sleep(0.3)
            return super().infer(adapter_path, input_text, grammar_constraint)

    fastapi_app = create_app(mock_adapter, backend=SlowBackend(), allow_anonymous=True)
    client = TestClient(fastapi_app)

    n = 25

    def _timed_request() -> "tuple[int, float]":
        t0 = time.perf_counter()
        res = client.post("/invoke", json={"input": "Urgent payment failure"})
        return res.status_code, time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda _: _timed_request(), range(n)))

    statuses = [status for status, _ in results]
    latencies = sorted(latency for _, latency in results)

    assert statuses.count(200) == 1
    assert statuses.count(503) == n - 1

    # Concrete p99 bound (same percentile-index style as ServerState.get_metrics):
    # every response, including the one that actually ran 0.3s of inference, must
    # complete in well under the old 30s admission wait. 2.0s leaves generous
    # headroom for test-machine scheduling jitter while still proving admission does
    # not queue.
    p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
    assert p99 < 2.0


def test_extract_content_handles_null_text_value_PAW_SERVE_05() -> None:
    """Verify PAW-SERVE-05: a content block shaped {"type": "text", "text": None} no
    longer raises TypeError from " ".join(parts) — .get(key, "")'s default only covers
    an absent key, not a key present with an explicit None value."""
    from paw_kit.serve.server import _extract_content

    assert _extract_content([{"type": "text", "text": None}]) == ""
    assert (
        _extract_content([{"type": "text", "text": "hi"}, {"type": "text", "text": None}])
        == "hi "
    )


def test_serve_chat_completions_parse_failure_recorded_as_error_PAW_SERVE_05(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-SERVE-05: a content-extraction failure is caught by the same
    telemetry boundary as an inference failure — the request is recorded as an error
    and returns a handled 500, instead of an unhandled crash that bypasses
    state.record_request(is_error=True) entirely."""
    import paw_kit.serve.server as server_module

    def _boom(content: object) -> str:
        raise TypeError("simulated malformed content")

    monkeypatch.setattr(server_module, "_extract_content", _boom)
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k")
    client = TestClient(fastapi_app)

    res = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "test"}]},
        headers={"Authorization": "Bearer k"},
    )
    assert res.status_code == 500

    metrics = client.get("/metrics", headers={"Authorization": "Bearer k"}).json()
    assert metrics["total_requests"] == 1
    assert metrics["error_count"] == 1


def test_serve_metrics_requires_auth_PAW_SERVE_06(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-06: /metrics is no longer exempt from auth — it discloses
    request-volume and latency telemetry, so it follows the same policy as inference."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k")
    client = TestClient(fastapi_app)

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer k"}).status_code == 200


def test_serve_docs_routes_require_auth_PAW_SERVE_06(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-06: /docs, /redoc and /openapi.json are gated behind the same
    auth policy as inference, unlike FastAPI's defaults which are open regardless."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k")
    client = TestClient(fastapi_app)

    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 401
        assert client.get(path, headers={"Authorization": "Bearer k"}).status_code == 200


def test_ready_included_in_openapi_schema_like_health(mock_adapter: Path) -> None:
    """/ready is polled by orchestration the same way /health is (see docker.py's
    HEALTHCHECK), so it belongs in the generated OpenAPI schema the same way -- it must
    not be the one endpoint a client generated from that schema doesn't know about."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="k")
    client = TestClient(fastapi_app)

    schema = client.get("/openapi.json", headers={"Authorization": "Bearer k"}).json()
    assert "/ready" in schema["paths"]
    assert "/health" in schema["paths"]


def test_serve_health_response_minimal_PAW_SERVE_06(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-06: /health discloses only status/uptime/version — never the
    adapter filename or backend implementation type it used to — and stays open."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    res = client.get("/health")
    assert res.status_code == 200
    assert set(res.json().keys()) == {"status", "uptime_seconds", "version"}


class _ForbiddenLock:
    """Stand-in lock whose acquisition always fails the test that installs it."""

    def __enter__(self) -> None:
        raise AssertionError("get_uptime() must not acquire the state lock")

    def __exit__(self, *exc_info: object) -> None:
        pass


def test_server_state_get_uptime_lock_free_PAW_SERVE_07() -> None:
    """Verify PAW-SERVE-07: get_uptime() never touches the state lock, so a liveness
    probe reading it can't be blocked behind get_metrics()'s percentile sort. A real
    `threading.Lock` object's methods are read-only (C-level), so the lock itself is
    swapped for a stand-in that fails the test the moment anything acquires it."""
    state = ServerState()
    state._lock = _ForbiddenLock()  # type: ignore[assignment]
    assert state.get_uptime() >= 0.0


def test_server_state_get_metrics_sorts_outside_lock_PAW_SERVE_07(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PAW-SERVE-07: get_metrics() releases the lock before sorting the latency
    snapshot — the lock protects only the O(n) copy, not the O(n log n) sort."""
    import paw_kit.serve.server as server_module

    state = ServerState()
    for lat in [5.0, 1.0, 3.0]:
        state.record_request(lat)

    lock_states_during_sort = []
    real_sorted = sorted

    def spy_sorted(iterable: object, *args: object, **kwargs: object) -> list:
        lock_states_during_sort.append(state._lock.locked())
        return real_sorted(iterable, *args, **kwargs)

    monkeypatch.setattr(server_module, "sorted", spy_sorted, raising=False)
    metrics = state.get_metrics()

    assert lock_states_during_sort == [False]
    assert metrics["p50_latency_ms"] == 3.0


def test_estimate_tokens_uses_char_heuristic_not_split_PAW_SERVE_08() -> None:
    """Verify PAW-SERVE-08: token estimation is len(text)//4, not text.split()*4//3 —
    a single 4000-char word (no whitespace) must not estimate as one token."""
    from paw_kit.serve.server import _estimate_tokens

    assert _estimate_tokens("a" * 4000) == 1000
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("hi") == 1


def test_serve_401_includes_www_authenticate_header_PAW_SERVE_09(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-09: both 401 paths in _verify_auth carry an RFC 6750
    WWW-Authenticate challenge."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="secret")
    client = TestClient(fastapi_app)

    res_missing = client.post("/invoke", json={"input": "test"})
    assert res_missing.status_code == 401
    assert res_missing.headers.get("www-authenticate") == "Bearer"

    res_wrong = client.post(
        "/invoke", json={"input": "test"}, headers={"Authorization": "Bearer wrong"}
    )
    assert res_wrong.status_code == 401
    assert res_wrong.headers.get("www-authenticate") == "Bearer"


def test_serve_non_ascii_bearer_token_returns_401_not_500_X_6(mock_adapter: Path) -> None:
    """Verify X-6: a non-ASCII bearer token (as it arrives after the ASGI layer's
    latin-1 header decoding) returns 401, not an unhandled 500. At `main`,
    `hmac.compare_digest(token, configured_api_key)` compares two `str` objects, and
    raises `TypeError: comparing strings with non-ASCII characters is not supported`
    for any non-ASCII `str` operand -- escaping `_verify_auth` as a bare 500 that
    `ServerErrorMiddleware` re-raises with a full traceback per attempt, and never
    recorded via `record_request(is_error=True)`.

    httpx encodes `str` header values as strict ASCII and raises before the request
    is even sent for one containing a non-ASCII character, so this passes raw bytes
    (a `(bytes, bytes)` header tuple) directly -- httpx does not re-encode those,
    and Starlette's `Headers(scope=scope)` decodes the same raw bytes back to the
    same non-ASCII `str` server-side, exactly reproducing the reported crash."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="secret")
    client = TestClient(fastapi_app)

    raw_auth = b"Bearer " + bytes([0xFF, 0xFE])
    res = client.post(
        "/invoke",
        json={"input": "test"},
        headers=[(b"authorization", raw_auth)],
    )
    assert res.status_code == 401
    assert res.headers.get("www-authenticate") == "Bearer"


def test_serve_auth_failure_logs_and_increments_metrics_X_9(
    mock_adapter: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify X-9: a failed-auth request increments /metrics' counters and produces a
    log record that does not echo the presented (wrong) token. At `main`, auth
    failures are neither logged nor counted at all -- /metrics reads
    total_requests: 0 regardless of how many failed attempts preceded it."""
    caplog.set_level(logging.WARNING, logger="paw_kit.serve")
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, api_key="supersecret-token")
    client = TestClient(fastapi_app)

    res = client.post(
        "/invoke",
        json={"input": "test"},
        headers={"Authorization": "Bearer wrong-token-value"},
    )
    assert res.status_code == 401

    metrics = client.get(
        "/metrics", headers={"Authorization": "Bearer supersecret-token"}
    ).json()
    assert metrics["total_requests"] >= 1
    assert metrics["error_count"] >= 1

    assert "auth" in caplog.text.lower()
    assert "supersecret-token" not in caplog.text
    assert "wrong-token-value" not in caplog.text


def test_serve_api_key_whitespace_warns_at_startup_X_12(
    mock_adapter: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify X-12: configuring an API key with leading/trailing whitespace logs a
    startup warning naming the problem. At `main`, the incoming bearer token is
    `.strip()`ed but the *configured* key never is, so a `PAW_API_KEY`/`--api-key`
    value with stray whitespace can never authenticate and the operator sees only
    "Invalid API key" with no indication why."""
    caplog.set_level(logging.WARNING, logger="paw_kit.serve")
    backend = MockPAWBackend()
    create_app(mock_adapter, backend=backend, api_key="  secret-with-space  ")

    assert "whitespace" in caplog.text.lower()


def test_serve_rate_limit_returns_429_on_exceedance_PAW_SERVE_10(mock_adapter: Path) -> None:
    """Verify PAW-SERVE-10: a client exceeding its per-minute token-bucket budget gets
    429, while /health stays exempt so container healthchecks never trip the limiter."""
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, allow_anonymous=True, requests_per_minute=3
    )
    client = TestClient(fastapi_app)

    for _ in range(3):
        assert client.post("/invoke", json={"input": "Urgent payment failure"}).status_code == 200

    res_limited = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res_limited.status_code == 429
    assert res_limited.headers.get("retry-after") == "1"

    assert client.get("/health").status_code == 200


def test_serve_rate_limit_disabled_when_zero_PAW_SERVE_10(mock_adapter: Path) -> None:
    """Verify requests_per_minute=0 disables the limiter explicitly rather than
    falling back to the environment/default."""
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, allow_anonymous=True, requests_per_minute=0
    )
    client = TestClient(fastapi_app)

    for _ in range(10):
        assert client.post("/invoke", json={"input": "Urgent payment failure"}).status_code == 200


def test_serve_rate_limit_bucket_storage_bounded_PAW_SERVE_10() -> None:
    """Verify the rate limiter's per-client bucket storage never exceeds
    _MAX_RATE_LIMIT_BUCKETS under many distinct client addresses — otherwise the
    limiter would itself become an unbounded-memory DoS surface, the exact problem
    class it exists to defend against. `global_requests_per_minute` is set high
    enough here to be a non-factor -- this test is about storage bounding, not
    throughput, and X-4's separate global-ceiling test below exercises that."""
    from paw_kit.serve.server import _MAX_RATE_LIMIT_BUCKETS, RateLimitMiddleware

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(
        _noop_app, requests_per_minute=60, global_requests_per_minute=10**9
    )
    for i in range(_MAX_RATE_LIMIT_BUCKETS + 50):
        middleware._consume(f"10.0.0.{i}")

    assert len(middleware._buckets) == _MAX_RATE_LIMIT_BUCKETS


def test_serve_rate_limit_eviction_preserves_throttled_bucket_X_5() -> None:
    """Verify X-5: bucket eviction never resets a *throttled* client's budget. At
    `main`, `popitem(last=False)` evicted whichever bucket was oldest regardless of
    its token level, and a fresh key defaults to a full bucket (the old
    `.pop(key, (capacity, now))`), so evicting a throttled client's entry was
    indistinguishable from silently handing it a brand new full one. Throttle one
    key first, then flood past _MAX_RATE_LIMIT_BUCKETS with fresh keys, and assert
    the throttled key's budget is unchanged (same direct
    RateLimitMiddleware._consume() manipulation style as the existing
    bucket-bound test)."""
    from paw_kit.serve.server import _MAX_RATE_LIMIT_BUCKETS, RateLimitMiddleware

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(
        _noop_app, requests_per_minute=60, global_requests_per_minute=10**9
    )
    for _ in range(60):
        middleware._consume("victim")
    tokens_before, _ = middleware._buckets["victim"]
    assert tokens_before < 1.0  # fully drained -- genuinely throttled

    for i in range(_MAX_RATE_LIMIT_BUCKETS + 50):
        middleware._consume(f"flood-{i}")

    assert "victim" in middleware._buckets
    tokens_after, _ = middleware._buckets["victim"]
    assert tokens_after == pytest.approx(tokens_before, abs=1e-6)


def test_serve_rate_limit_ipv6_slash64_shares_bucket_X_4() -> None:
    """Verify X-4: two different addresses inside the same IPv6 /64 collapse onto
    one bucket -- otherwise an attacker with a routable /64 (trivial to obtain,
    unlike an IPv4 /32) evades the limiter entirely by incrementing the low 64 bits
    on every request. A different /64 must get its own, independent bucket."""
    from paw_kit.serve.server import RateLimitMiddleware, _normalize_address_key

    key_a = _normalize_address_key("2001:db8:1234:5678::1")
    key_b = _normalize_address_key("2001:db8:1234:5678:ffff:ffff:ffff:ffff")
    assert key_a == key_b

    key_other = _normalize_address_key("2001:db8:1234:5679::1")
    assert key_other != key_a

    # End-to-end through the middleware itself: exhausting the budget from one
    # address in a /64 throttles a different address in the *same* /64.
    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(
        _noop_app, requests_per_minute=1, global_requests_per_minute=10**9
    )
    scope_a = {"type": "http", "path": "/invoke", "client": ("2001:db8:1234:5678::1", 0)}
    scope_b = {"type": "http", "path": "/invoke", "client": ("2001:db8:1234:5678:1::1", 0)}

    assert asyncio.run(_rate_limit_allows(middleware, scope_a)) is True
    assert asyncio.run(_rate_limit_allows(middleware, scope_b)) is False


async def _rate_limit_allows(middleware: "RateLimitMiddleware", scope: Dict[str, Any]) -> bool:
    """Drive `RateLimitMiddleware.__call__` directly and report whether the
    downstream app was reached (True) or a 429 was returned (False)."""
    reached = False

    async def receive() -> Dict[str, Any]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Dict[str, Any]) -> None:
        pass

    async def inner_app(scope: object, receive: object, send: object) -> None:
        nonlocal reached
        reached = True

    middleware.app = inner_app
    await middleware(scope, receive, send)
    return reached


def test_serve_rate_limit_client_none_logs_warning_not_silent_X_4(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify X-4: `scope["client"] is None` with no trusted proxy header configured
    (the documented reverse-proxy deployment) is loud, not silent -- at `main`, this
    situation collapses every such client onto one shared "unknown" bucket with no
    indication it happened at all."""
    from paw_kit.serve.server import RateLimitMiddleware

    caplog.set_level(logging.WARNING, logger="paw_kit.serve")

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(_noop_app, requests_per_minute=60)
    scope = {"type": "http", "path": "/invoke", "client": None}
    assert asyncio.run(_rate_limit_allows(middleware, scope)) is True

    assert "client" in caplog.text.lower()
    assert "PAW_TRUST_PROXY_HEADER" in caplog.text


def test_serve_rate_limit_global_ceiling_caps_across_all_keys_X_4() -> None:
    """Verify X-4's global ceiling: an attacker rotating through many distinct
    per-key buckets (each starting full, by design -- a fresh key must default to a
    full bucket, or a legitimate first-time client would be throttled before its
    first request) cannot exceed the aggregate ceiling across all of them combined.
    At `main` there is no such ceiling at all: address rotation bypasses the
    limiter entirely, bounded only by however many distinct addresses the attacker
    can produce."""
    from paw_kit.serve.server import RateLimitMiddleware

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(
        _noop_app, requests_per_minute=1000, global_requests_per_minute=5
    )

    allowed = sum(1 for i in range(20) if middleware._consume(f"rotating-{i}"))
    assert allowed == 5


def test_serve_rate_limit_negative_env_value_warns_and_disables_X_8(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify X-8: PAW_RATE_LIMIT_PER_MINUTE=-5 logs a warning and disables the
    limiter instead of passing a negative capacity through. At `main`,
    `int("-5")` succeeds and `if requests_per_minute > 0` silently skips installing
    the limiter with no log line at all."""
    monkeypatch.setenv("PAW_RATE_LIMIT_PER_MINUTE", "-5")
    caplog.set_level(logging.WARNING, logger="paw_kit.serve")
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    for _ in range(15):
        assert client.post("/invoke", json={"input": "Urgent payment failure"}).status_code == 200

    assert "PAW_RATE_LIMIT_PER_MINUTE" in caplog.text


def test_serve_rate_limit_non_integer_env_value_falls_back_X_8(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify X-8: PAW_RATE_LIMIT_PER_MINUTE="not-a-number" falls back cleanly (no
    raw ValueError at startup) and logs a warning. At `main`,
    `int(os.environ.get("PAW_RATE_LIMIT_PER_MINUTE", ...))` raises an unhandled
    ValueError directly out of create_app."""
    monkeypatch.setenv("PAW_RATE_LIMIT_PER_MINUTE", "not-a-number")
    caplog.set_level(logging.WARNING, logger="paw_kit.serve")
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    assert client.post("/invoke", json={"input": "Urgent payment failure"}).status_code == 200
    assert "PAW_RATE_LIMIT_PER_MINUTE" in caplog.text


def test_int_env_with_fallback_zero_is_not_negative_X_8(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Verify the exact boundary in X-8's `_int_env_with_fallback`: `if value < 0`
    must not treat 0 itself as negative. 0 is a legitimate, intentional value here
    (PAW_RATE_LIMIT_PER_MINUTE=0 explicitly disables the limiter, see
    test_serve_rate_limit_disabled_when_zero_PAW_SERVE_10) -- a mutant widening the
    comparison to `<=` would additionally clamp 0 to 0 (a no-op on the return value)
    but log the same "is negative" warning for a value that was never negative,
    which is exactly the false-positive this test would catch even though the
    *return value* is identical either way."""
    from paw_kit.serve.server import _int_env_with_fallback

    monkeypatch.setenv("PAW_RATE_LIMIT_PER_MINUTE", "0")
    caplog.set_level(logging.WARNING, logger="paw_kit.serve")

    result = _int_env_with_fallback("PAW_RATE_LIMIT_PER_MINUTE", 120)

    assert result == 0
    assert "is negative" not in caplog.text


def test_serve_rate_limit_global_bucket_allows_exactly_one_token_X_4() -> None:
    """Verify the exact boundary in X-4's global ceiling: `if self._global_tokens <
    1.0` must allow a consume when exactly 1.0 token is available (not just when
    strictly more than 1.0 is), and correctly deny once it drops to 0.0. A mutant
    widening this to `<=` would reject a caller with a perfectly full one-token
    budget, which no request-level test above catches (they only ever observe the
    limiter through many requests, never pin the single-token boundary case
    directly)."""
    from paw_kit.serve.server import RateLimitMiddleware

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(_noop_app, requests_per_minute=60, global_requests_per_minute=1)
    now = middleware._global_last_refill  # same instant: refill contributes exactly 0

    assert middleware._consume_global_locked(now) is True  # exactly 1.0 tokens -> allowed
    assert middleware._consume_global_locked(now) is False  # now 0.0 tokens -> denied


def test_serve_rate_limit_eviction_boundary_at_epsilon_X_5() -> None:
    """Verify the exact boundary in X-5's eviction guard: `if projected >=
    self.capacity - 1e-9` must still evict a bucket sitting *exactly* at that
    epsilon-tolerance threshold (the "approximately full" case the epsilon exists
    for), not only one strictly above it. A mutant narrowing this to `>` would
    refuse to evict a bucket parked exactly on the boundary, which
    `test_serve_rate_limit_eviction_preserves_throttled_bucket_X_5` cannot detect:
    that test only ever produces buckets far from this exact floating-point
    boundary."""
    from paw_kit.serve.server import RateLimitMiddleware

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(
        _noop_app, requests_per_minute=60, global_requests_per_minute=10**9
    )
    now = time.monotonic()
    # Token count placed exactly on the tolerance boundary, with last_refill == now
    # so the projection adds zero refill -- `projected` is exactly
    # `capacity - 1e-9`, the precise value the comparison tests against.
    middleware._buckets["idle-at-boundary"] = (60.0 - 1e-9, now)

    assert middleware._evict_one_full_bucket_locked(now) is True
    assert "idle-at-boundary" not in middleware._buckets


def test_serve_rate_limit_real_client_does_not_log_none_warning_X_4(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify X-4's log-once guard is a genuine `and`, not `or`:
    `if client_was_none and not self._warned_no_client` must stay silent for a
    request with a real (non-None) client address, even on a fresh middleware
    instance where `_warned_no_client` is still False. A mutant widening this to
    `or` would fire the "scope['client'] is None" warning on literally the first
    request handled by any middleware instance, real client address or not --
    undetectable by the existing client=None test, which never checks the
    negative case."""
    from paw_kit.serve.server import RateLimitMiddleware

    caplog.set_level(logging.WARNING, logger="paw_kit.serve")

    async def _noop_app(scope: object, receive: object, send: object) -> None:
        pass

    middleware = RateLimitMiddleware(_noop_app, requests_per_minute=60)
    scope = {"type": "http", "path": "/invoke", "client": ("10.0.0.1", 12345)}

    assert asyncio.run(_rate_limit_allows(middleware, scope)) is True
    assert caplog.records == []


def test_serve_execution_timeout_returns_exactly_503_X_7(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify the execution-timeout path in `_execute_with_telemetry` (X-7) returns
    exactly 503, pinned as a literal rather than merely "not 200" -- distinct from
    the admission-overflow 503 the busy-inference tests above exercise (a
    different call site in the same function): this specifically forces an
    *admitted* inference to overrun `_INFERENCE_SLOT_TIMEOUT_SECONDS`, which no
    existing test reaches (they all use sub-second sleeps well under the 20s
    default)."""
    import paw_kit.serve.server as server_module

    monkeypatch.setattr(server_module, "_INFERENCE_SLOT_TIMEOUT_SECONDS", 0.05)

    class SlowBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            time.sleep(0.3)
            return super().infer(adapter_path, input_text, grammar_constraint)

    fastapi_app = create_app(mock_adapter, backend=SlowBackend(), allow_anonymous=True)
    client = TestClient(fastapi_app)

    res = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res.status_code == 503
    assert "execution bound" in res.json()["detail"]


def test_serve_trust_proxy_header_env_var_gates_xff_trust_X_4(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify X-4's `PAW_TRUST_PROXY_HEADER` env-var parsing in create_app
    (`... in ("1", "true", "yes", "on")`) actually gates whether
    X-Forwarded-For is trusted, end to end through the real middleware stack --
    the existing X-4 tests construct RateLimitMiddleware directly with an explicit
    `trust_proxy_header=` kwarg and never exercise this env-var-parsing line at
    all. A mutant flipping `in` to `not in` would invert the opt-in entirely."""
    monkeypatch.setenv("PAW_TRUST_PROXY_HEADER", "1")
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, allow_anonymous=True, requests_per_minute=1
    )
    client = TestClient(fastapi_app)

    res1 = client.post(
        "/invoke", json={"input": "x"}, headers={"X-Forwarded-For": "203.0.113.5"}
    )
    assert res1.status_code == 200
    # Different spoofed address, same real TestClient peer -- trusted, so this is a
    # fresh bucket rather than sharing the first request's exhausted one.
    res2 = client.post(
        "/invoke", json={"input": "x"}, headers={"X-Forwarded-For": "203.0.113.9"}
    )
    assert res2.status_code == 200


def test_serve_trust_proxy_header_disabled_by_default_ignores_xff_X_4(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Companion to the test above: with PAW_TRUST_PROXY_HEADER unset (the
    default), X-Forwarded-For must be ignored -- both requests key on the real
    (identical) TestClient peer address and share one budget. Together the two
    tests bracket both sides of the `in` boundary that a `not in` mutant would
    invert."""
    monkeypatch.delenv("PAW_TRUST_PROXY_HEADER", raising=False)
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, allow_anonymous=True, requests_per_minute=1
    )
    client = TestClient(fastapi_app)

    res1 = client.post(
        "/invoke", json={"input": "x"}, headers={"X-Forwarded-For": "203.0.113.5"}
    )
    assert res1.status_code == 200
    res2 = client.post(
        "/invoke", json={"input": "x"}, headers={"X-Forwarded-For": "203.0.113.9"}
    )
    assert res2.status_code == 429


def test_server_state_metrics_calculation() -> None:
    """Verify ServerState percentile calculation across request latencies."""
    state = ServerState()
    for lat in [10.0, 20.0, 30.0, 40.0, 50.0]:
        state.record_request(lat)

    metrics = state.get_metrics()
    assert metrics["total_requests"] == 5
    assert metrics["error_count"] == 0
    assert metrics["p50_latency_ms"] == 30.0
    assert metrics["p95_latency_ms"] == 50.0


def test_docker_exporter_scaffold(mock_adapter: Path, tmp_path: Path) -> None:
    """Verify export_docker_scaffold generates all production deployment files with non-root user (M-4)."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    assert dest.exists()
    assert (dest / "Dockerfile").exists()
    assert (dest / ".dockerignore").exists()
    assert (dest / "docker-compose.yml").exists()
    assert (dest / "README.md").exists()
    assert (dest / "triage.paw").exists()

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    # X-10 pinned the base image by digest (see bug-hunt-F-server.md's named-hazard
    # note): a bare "FROM python:3.12-slim" substring check would keep passing by
    # accident once a digest is appended, so this asserts the actual pinned
    # reference explicitly instead.
    from paw_kit.serve.docker import _PYTHON_BASE_IMAGE

    assert f"FROM {_PYTHON_BASE_IMAGE}" in dockerfile
    assert "USER app" in dockerfile
    assert "paw-serve" in dockerfile
    assert "triage.paw" in dockerfile


def test_docker_exporter_carries_paw_api_key_PAW_SERVE_01(mock_adapter: Path, tmp_path: Path) -> None:
    """Verify the generated container inherits auth-by-default (Phase 1's PAW-SERVE-01):
    docker-compose.yml forwards PAW_API_KEY, and the README documents supplying it and
    includes it in its example curl commands rather than the old auth-less examples."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    compose = (dest / "docker-compose.yml").read_text(encoding="utf-8")
    assert "PAW_API_KEY" in compose

    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "PAW_API_KEY" in readme
    assert "Authorization: Bearer $PAW_API_KEY" in readme


def test_docker_exporter_sanitization(mock_adapter: Path, tmp_path: Path) -> None:
    """Verify L-7: export_docker_scaffold rejects unsafe adapter filenames."""
    bad_adapter = tmp_path / "bad;rm -rf.paw"
    bad_adapter.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="unsafe characters"):
        export_docker_scaffold(bad_adapter, output_dir=tmp_path / "docker_bad")


def test_docker_exporter_reserved_filename_collision_PAW_DOCKER_01(tmp_path: Path) -> None:
    """Verify PAW-DOCKER-01: an adapter path colliding with a file
    export_docker_scaffold generates itself (e.g. the audit's
    `paw export docker ./Dockerfile --out-dir ./deploy` scenario) is rejected rather
    than silently overwriting the just-generated file during the copy-adapter step."""
    out_dir = tmp_path / "deploy"
    out_dir.mkdir()

    # The audit's literal scenario: an "adapter" named exactly like a generated file.
    fake_dockerfile_adapter = out_dir / "Dockerfile"
    fake_dockerfile_adapter.write_text("not a real adapter", encoding="utf-8")

    with pytest.raises(ValueError, match="collides with a file"):
        export_docker_scaffold(fake_dockerfile_adapter, output_dir=out_dir)

    # Every reserved generated filename is rejected the same way.
    for reserved in [".dockerignore", "docker-compose.yml", "README.md", "requirements.txt"]:
        reserved_adapter = tmp_path / reserved
        reserved_adapter.write_text("not a real adapter", encoding="utf-8")
        with pytest.raises(ValueError, match="collides with a file"):
            export_docker_scaffold(reserved_adapter, output_dir=out_dir / "unused")

    # A non-reserved name still requires the .paw extension.
    no_extension_adapter = tmp_path / "harmless-name"
    no_extension_adapter.write_text("not a real adapter", encoding="utf-8")
    with pytest.raises(ValueError, match="must have a '.paw' extension"):
        export_docker_scaffold(no_extension_adapter, output_dir=out_dir / "unused2")


def test_docker_exporter_loopback_only_port_binding_PAW_DOCKER_02(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """Verify PAW-DOCKER-02: the generated docker-compose.yml and README no longer
    default to exposing the service on every host interface — both bind loopback
    only, requiring a deliberate edit to reach the container from the network."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    compose = (dest / "docker-compose.yml").read_text(encoding="utf-8")
    assert '      - "127.0.0.1:8000:8000"' in compose
    assert '      - "8000:8000"' not in compose

    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "docker run -p 127.0.0.1:8000:8000" in readme


def test_docker_exporter_pinned_requirements_txt_PAW_DOCKER_03(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """Verify PAW-DOCKER-03: the exporter generates a requirements.txt with exact,
    pinned versions, and the Dockerfile installs from it instead of an unpinned
    inline package list that would resolve a different set on every build."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    assert (dest / "requirements.txt").exists()
    requirements = (dest / "requirements.txt").read_text(encoding="utf-8")
    # Default backend is "real" (see PAW-DOCKER-04 below): paw-kit is pinned with the
    # real-backend extra, not bare.
    for package in ("paw-kit[real]==", "fastapi==", "uvicorn==", "httpx=="):
        assert package in requirements

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY requirements.txt /app/requirements.txt" in dockerfile
    assert "uv pip install --system -r requirements.txt" in dockerfile
    assert "uv pip install --system paw-kit fastapi uvicorn httpx" not in dockerfile


def test_resolved_dependency_names_reads_real_environment_metadata_X_10() -> None:
    """Verify `_resolved_dependency_names` itself returns the actual names resolved
    from this environment's installed `paw-kit` metadata (non-empty, and matching
    known [project.dependencies] entries) -- called directly, not through
    `_requirements_txt_content`'s own `or list(_FALLBACK_DEPENDENCY_NAMES)`
    fallback, which happens to contain the same names in this environment and so
    cannot by itself distinguish "resolved from the environment" from "silently
    fell back to the hardcoded tuple" (see the mutant `_pkg_requires(package) or
    []` -> `and []`, which makes this function always return `[]` whenever the
    package genuinely has requirements -- masked at the `_requirements_txt_content`
    level by that same fallback, but not here)."""
    from paw_kit.serve.docker import _resolved_dependency_names

    names = _resolved_dependency_names("paw-kit")

    assert names, "expected paw-kit's installed metadata to yield a non-empty list"
    assert "fastapi" in names
    assert "pydantic" in names


def test_requirements_txt_uses_resolved_names_over_fallback_when_present_X_10(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `_requirements_txt_content`'s `_resolved_dependency_names("paw-kit")
    or list(_FALLBACK_DEPENDENCY_NAMES)` actually prefers a non-empty resolved
    list over the fallback -- a mutant flipping `or` to `and` would silently
    discard any real resolved list and substitute the hardcoded fallback tuple
    instead whenever resolution succeeds (i.e. always, in practice), which the
    test above cannot catch on its own since it only inspects
    `_resolved_dependency_names` in isolation, not how its result is actually
    used. Distinguished here by monkeypatching resolution to a name that is not
    in the fallback tuple at all."""
    import paw_kit.serve.docker as docker_module

    monkeypatch.setattr(docker_module, "_resolved_dependency_names", lambda package: ["pytest"])

    content = docker_module._requirements_txt_content("mock")

    assert "pytest==" in content
    assert "fastapi==" not in content
    assert "uvicorn==" not in content


def test_docker_exporter_pins_full_dependency_set_and_no_latest_tags_X_10(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """Verify X-10: the generated requirements.txt covers every name in
    pyproject.toml's [project.dependencies] -- at `main` only fastapi/uvicorn/httpx
    were pinned, leaving interegular/pydantic/pyyaml/typer to float unpinned -- and
    neither the base Python image nor the uv installer image floats on a mutable
    tag (`FROM python:3.12-slim` / `COPY --from=ghcr.io/astral-sh/uv:latest` at
    `main`); both are pinned by immutable digest instead."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    requirements = (dest / "requirements.txt").read_text(encoding="utf-8")
    for package in ("interegular", "pydantic", "pyyaml", "typer", "fastapi", "uvicorn", "httpx"):
        assert f"{package}==" in requirements, f"{package} is not pinned in requirements.txt"

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert ":latest" not in dockerfile
    assert re.search(r"FROM python:[\w.\-]+@sha256:[0-9a-f]{64}", dockerfile)
    assert re.search(r"COPY --from=ghcr\.io/astral-sh/uv:[\w.\-]+@sha256:[0-9a-f]{64}", dockerfile)


def test_docker_exporter_dockerignore_covers_secrets_X_11(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """Verify X-11: the generated .dockerignore excludes .env/.env.*/*.pem/*.key/
    secrets/ -- at `main` it omits all of these even though the generated README
    tells the operator to put PAW_API_KEY in a .env file next to the compose file."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    dockerignore = (dest / ".dockerignore").read_text(encoding="utf-8")
    for pattern in (".env", ".env.*", "*.pem", "*.key", "secrets/"):
        assert pattern in dockerignore.splitlines(), f"{pattern!r} missing from .dockerignore"


def test_docker_exporter_healthcheck_hits_ready_with_cold_start_period(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """The generated HEALTHCHECK polls /ready (readiness), not /health (liveness --
    true before the model has loaded), with a --start-period long enough to cover a
    cold base-model download (measured up to ~110s, see measurements/README.md), and
    --warm is passed in the CMD so the container pays that cost before reporting ready."""
    out_dir = tmp_path / "docker_dist"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert "urlopen('http://localhost:8000/ready')" in dockerfile
    assert "urlopen('http://localhost:8000/health')" not in dockerfile
    assert "--start-period=180s" in dockerfile
    assert (
        '"paw-serve", "/app/triage.paw", "--host", "0.0.0.0", "--port", "8000", "--backend", "real", "--warm"'
        in dockerfile
    )

    compose = (dest / "docker-compose.yml").read_text(encoding="utf-8")
    assert "urlopen('http://localhost:8000/ready')" in compose


# ---------------------------------------------------------------- PAW-DOCKER-04: --backend scaffold choice


def test_docker_exporter_backend_real_default_ships_real_extra_and_warm(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """Default (no `backend=` passed) is "real": a container that can only ever serve
    the mock backend is a demo, not a deployment. The generated CMD passes `--backend
    real`, ships `--warm`, requirements.txt pulls in the project's real-backend extra,
    and the comments describe an actual model being loaded."""
    out_dir = tmp_path / "docker_real"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir)

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert '"--backend", "real"' in dockerfile
    assert '"--warm"' in dockerfile
    assert "--start-period=180s" in dockerfile
    assert "cold" in dockerfile.lower()
    assert "ProgramAsWeights" in dockerfile

    requirements = (dest / "requirements.txt").read_text(encoding="utf-8")
    assert "paw-kit[real]==" in requirements

    compose = (dest / "docker-compose.yml").read_text(encoding="utf-8")
    assert "start_period: 180s" in compose


def test_docker_exporter_backend_mock_omits_warm_and_real_extra(
    mock_adapter: Path, tmp_path: Path
) -> None:
    """`backend="mock"` produces a container that only ever serves the deterministic
    mock: no `--warm` (there is nothing to warm), no real-backend extra in
    requirements.txt, a short --start-period (no cold model download to wait out), and
    comments that say so rather than describing a real model load."""
    out_dir = tmp_path / "docker_mock"
    dest = export_docker_scaffold(mock_adapter, output_dir=out_dir, backend="mock")

    dockerfile = (dest / "Dockerfile").read_text(encoding="utf-8")
    assert (
        '"paw-serve", "/app/triage.paw", "--host", "0.0.0.0", "--port", "8000", "--backend", "mock"]'
        in dockerfile
    )
    # The literal CMD flag must be absent -- the surrounding prose is allowed to
    # mention "--warm" descriptively (explaining why it's omitted).
    assert '"--warm"' not in dockerfile
    assert "--start-period=10s" in dockerfile
    assert "mock" in dockerfile.lower()

    requirements = (dest / "requirements.txt").read_text(encoding="utf-8")
    assert "paw-kit==" in requirements
    assert "paw-kit[real]==" not in requirements

    compose = (dest / "docker-compose.yml").read_text(encoding="utf-8")
    assert "start_period: 10s" in compose
    assert "start_period: 180s" not in compose


def test_docker_exporter_rejects_unknown_backend(mock_adapter: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="backend"):
        export_docker_scaffold(mock_adapter, output_dir=tmp_path / "docker_bad", backend="torch")


def test_cli_export_commands(mock_adapter: Path, tmp_path: Path) -> None:
    """Verify paw-kit export docker and paw-kit export dataset CLI commands."""
    runner = CliRunner()

    # 1. paw-kit export docker
    docker_out = tmp_path / "cli_docker"
    res_docker = runner.invoke(app, ["export", "docker", str(mock_adapter), "--out-dir", str(docker_out)])
    assert res_docker.exit_code == 0
    assert (docker_out / "Dockerfile").exists()

    # 2. paw-kit export docker non-existent file
    res_bad_docker = runner.invoke(app, ["export", "docker", str(tmp_path / "none.paw")])
    assert res_bad_docker.exit_code == 1

    # 2a. --backend mock|real threads through to the generated CMD (PAW-DOCKER-04)
    docker_mock_out = tmp_path / "cli_docker_mock"
    res_docker_mock = runner.invoke(
        app, ["export", "docker", str(mock_adapter), "--out-dir", str(docker_mock_out), "--backend", "mock"]
    )
    assert res_docker_mock.exit_code == 0
    mock_dockerfile = (docker_mock_out / "Dockerfile").read_text(encoding="utf-8")
    assert '"--backend", "mock"' in mock_dockerfile
    assert '"--warm"' not in mock_dockerfile

    # 2b. an unknown --backend choice is rejected rather than silently accepted
    res_docker_bad_backend = runner.invoke(
        app,
        ["export", "docker", str(mock_adapter), "--out-dir", str(tmp_path / "cli_docker_bad"), "--backend", "torch"],
    )
    assert res_docker_bad_backend.exit_code == 1

    # 3. paw-kit export dataset -- PAW-CLI-04: built via a real TraceDB/record_trace,
    # not a hand-built `traces` table shaped like the old (never-real) input/output
    # schema; see test_cli_export_dataset_real_schema_PAW_CLI_04 for the dedicated
    # regression coverage this bug needed.
    db_file = tmp_path / "traces.db"
    trace_db = TraceDB(str(db_file))
    trace_db.record_trace(
        task_id="demo-task",
        input_payload="test user input",
        teacher_output="test output",
        latency_ms=12.5,
    )

    jsonl_out = tmp_path / "dataset.jsonl"
    res_dataset = runner.invoke(app, ["export", "dataset", "--db", str(db_file), "--out", str(jsonl_out)])
    assert res_dataset.exit_code == 0
    assert jsonl_out.exists()
    records = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").strip().split("\n")]
    assert len(records) == 1
    assert records[0]["messages"][0]["content"] == "test user input"
    assert records[0]["messages"][1]["content"] == "test output"
    assert (jsonl_out.stat().st_mode & 0o777) == 0o600

    # 4. paw-kit export dataset non-existent db
    res_bad_db = runner.invoke(app, ["export", "dataset", "--db", str(tmp_path / "no_db.db")])
    assert res_bad_db.exit_code == 1


def test_serve_adapter_runner(mock_adapter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify serve_adapter invokes uvicorn.run with expected arguments and default host (H-1)."""
    import uvicorn
    from paw_kit.serve.server import serve_adapter

    called_args = {}

    def mock_run(app, host, port):
        called_args["app"] = app
        called_args["host"] = host
        called_args["port"] = port

    monkeypatch.setattr(uvicorn, "run", mock_run)
    serve_adapter(mock_adapter, port=9000, allow_anonymous=True)

    # Verify default host is 127.0.0.1 for security
    assert called_args["host"] == "127.0.0.1"
    assert called_args["port"] == 9000


def test_cli_serve_command(mock_adapter: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify paw-kit serve CLI launches server and handles non-existent adapters."""
    from paw_kit.serve import server

    called_kwargs = {}

    def mock_serve_adapter(*args, **kwargs):
        called_kwargs.update(kwargs)

    monkeypatch.setattr(server, "serve_adapter", mock_serve_adapter)

    runner = CliRunner()
    # Good adapter with API key
    res = runner.invoke(app, ["serve", str(mock_adapter), "--port", "8888", "--api-key", "secret123"])
    assert res.exit_code == 0
    assert called_kwargs.get("host") == "127.0.0.1"
    assert called_kwargs.get("api_key") == "secret123"
    assert called_kwargs.get("allow_anonymous") is False

    # Bad adapter
    res_bad = runner.invoke(app, ["serve", "missing_adapter.paw"])
    assert res_bad.exit_code == 1


def test_cli_serve_allow_anonymous_flag_PAW_SERVE_01(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify --allow-anonymous threads from the CLI through to serve_adapter, and that
    its console output reflects the effective auth mode (PAW-SERVE-01 plumbing)."""
    from paw_kit.serve import server

    called_kwargs = {}

    def mock_serve_adapter(*args, **kwargs):
        called_kwargs.update(kwargs)

    monkeypatch.setattr(server, "serve_adapter", mock_serve_adapter)

    runner = CliRunner()
    res = runner.invoke(app, ["serve", str(mock_adapter), "--allow-anonymous"])
    assert res.exit_code == 0
    assert called_kwargs.get("allow_anonymous") is True
    assert called_kwargs.get("api_key") is None
    assert "DISABLED" in res.stdout


def test_ready_503_before_warmup_and_200_after_success(mock_adapter: Path) -> None:
    """Verify /ready: 503 {"status": "loading"} until the adapter has completed one
    successful inference, then 200 {"status": "ready"} -- distinct from /health, which
    is 200 immediately regardless."""
    backend = MockPAWBackend()
    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True)
    client = TestClient(fastapi_app)

    res_before = client.get("/ready")
    assert res_before.status_code == 503
    assert res_before.json() == {"status": "loading"}
    # /health stays live throughout, unlike /ready.
    assert client.get("/health").status_code == 200

    res_invoke = client.post("/invoke", json={"input": "Urgent payment failure"})
    assert res_invoke.status_code == 200

    res_after = client.get("/ready")
    assert res_after.status_code == 200
    assert res_after.json() == {"status": "ready"}


def test_ready_exempt_from_auth_and_rate_limit(mock_adapter: Path) -> None:
    """/ready needs no bearer token and doesn't count against the rate limiter, same as
    /health."""
    backend = MockPAWBackend()
    fastapi_app = create_app(
        mock_adapter, backend=backend, api_key="secret", requests_per_minute=1
    )
    client = TestClient(fastapi_app)

    # No Authorization header at all -- still answers (503, since not warmed/invoked).
    assert client.get("/ready").status_code == 503

    # Exhaust the rate limit budget on an authenticated endpoint...
    assert (
        client.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code
        == 200
    )
    assert (
        client.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code
        == 429
    )
    # ...and /ready is still answered rather than 429ed.
    assert client.get("/ready").status_code == 503


def test_warm_true_runs_one_inference_before_binding_and_marks_ready(
    mock_adapter: Path,
) -> None:
    """--warm (via create_app(warm=True)) calls the backend once with a fixed input
    before the app is even handed back, so /ready is already 200 on the first request."""
    calls = []

    class RecordingBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            calls.append(input_text)
            return super().infer(adapter_path, input_text, grammar_constraint)

    backend = RecordingBackend()
    backend.compile(spec="Classify ticket priority", examples=[], output_path=str(mock_adapter))

    fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True, warm=True)
    assert len(calls) == 1

    client = TestClient(fastapi_app)
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}

    # /metrics' request/latency counters are untouched by the warm-up: it is not a
    # real served request.
    m = client.get("/metrics").json()
    assert m["total_requests"] == 0


def test_warm_with_response_model_calls_backend_infer_directly_not_schema_wrapper(
    mock_adapter: Path,
) -> None:
    """--warm must warm through `backend.infer(...)` directly, not through the
    schema-validating `load()` wrapper `exec_fn` is built from when response_model is
    set: the fixed warm-up input ("ping") almost never parses as an arbitrary caller
    schema, so warming through that wrapper raised PAWSchemaError on essentially every
    adapter with a response_model configured, leaving /ready stuck at 503 forever even
    though the model itself loaded fine. MockPAWBackend.infer("ping") here returns the
    non-JSON fallback string "[mock:ping]", which does not parse as TicketSchema --
    warm-up must still succeed because it never routes through that validation."""
    calls = []

    class RecordingBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            calls.append((adapter_path, input_text))
            return super().infer(adapter_path, input_text, grammar_constraint)

    backend = RecordingBackend()
    backend.compile(spec="Classify ticket priority", examples=[], output_path=str(mock_adapter))

    fastapi_app = create_app(
        mock_adapter,
        backend=backend,
        response_model=TicketSchema,
        allow_anonymous=True,
        warm=True,
    )
    # Warm-up ran exactly once, directly against the backend, with the raw fixed input
    # -- not wrapped, retried, or schema-checked.
    assert len(calls) == 1
    assert calls[0] == (str(mock_adapter), "ping")

    client = TestClient(fastapi_app)
    res = client.get("/ready")
    assert res.status_code == 200
    assert res.json() == {"status": "ready"}


def test_warm_failure_leaves_ready_503_and_does_not_raise(
    mock_adapter: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A warm-up inference failure must not crash app creation; it's logged, and
    /ready stays 503 until a real request succeeds."""

    class FailingBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            raise RuntimeError("model failed to load")

    backend = FailingBackend()
    backend.compile(spec="Classify ticket priority", examples=[], output_path=str(mock_adapter))

    with caplog.at_level("WARNING", logger="paw_kit.serve"):
        fastapi_app = create_app(mock_adapter, backend=backend, allow_anonymous=True, warm=True)

    assert any("Warm-up inference failed" in rec.message for rec in caplog.records)

    client = TestClient(fastapi_app)
    assert client.get("/ready").status_code == 503


def test_cli_serve_warm_flag_threads_through_PAW_SERVE_READY(
    mock_adapter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--warm on `paw-kit serve` reaches serve_adapter (default False when omitted)."""
    from paw_kit.serve import server

    called_kwargs = {}

    def mock_serve_adapter(*args, **kwargs):
        called_kwargs.update(kwargs)

    monkeypatch.setattr(server, "serve_adapter", mock_serve_adapter)

    runner = CliRunner()
    res = runner.invoke(app, ["serve", str(mock_adapter), "--allow-anonymous", "--warm"])
    assert res.exit_code == 0
    assert called_kwargs.get("warm") is True

    res_default = runner.invoke(app, ["serve", str(mock_adapter), "--allow-anonymous"])
    assert res_default.exit_code == 0
    assert called_kwargs.get("warm") is False


def test_server_backend_label_uses_stable_vocabulary():
    """backend_label() is "mock"/"real", never a concrete class name.

    This label used to be `"real" if isinstance(b, RealPAWBackend) else "mock"` -- that
    class was deleted in Track 13 -- which
    mislabelled ProgramAsWeightsBackend -- the only backend proven against a real model --
    as "mock". The fix must not swing to `type(b).__name__`: PAW-SERVE-06 deliberately
    stripped implementation detail out of the telemetry surface, and a concrete class name
    is exactly what it removed. It must also generalise to backends beyond the two the
    original isinstance chain knew about, which is the case nothing previously exercised.
    """
    from paw_kit.backend.base import AbstractPAWBackend
    from paw_kit.serve.server import backend_label

    class ThirdPartyBackend(AbstractPAWBackend):
        """An arbitrary AbstractPAWBackend neither branch was ever tested against."""

        def compile(self, spec, examples, output_path):  # pragma: no cover - unused
            raise NotImplementedError

        def infer(self, adapter_path, input_text, grammar_constraint=None):
            return "ok"

        def is_available(self) -> bool:
            return True

    assert backend_label(MockPAWBackend()) == "mock"

    label = backend_label(ThirdPartyBackend())
    assert label == "real"
    assert "ThirdPartyBackend" not in label

