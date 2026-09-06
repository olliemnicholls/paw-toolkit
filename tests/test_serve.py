"""Unit and integration tests for paw_kit.serve: OpenAI and Anthropic HTTP serving layer and Docker exporter."""

import json
from pathlib import Path
import sqlite3
from typing import Optional
from fastapi.testclient import TestClient
from pydantic import BaseModel
import pytest
from typer.testing import CliRunner

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.cli import app
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
    assert data["backend"] == "mock"
    # Verify basename is returned to prevent directory disclosure (H-3)
    assert data["adapter_path"] == "triage.paw"
    assert data["uptime_seconds"] >= 0.0

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

    # 1. Health and metrics remain open for monitoring
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200

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


def test_server_state_metrics_calculation() -> None:
    """Verify ServerState percentile calculation across request latencies."""
    state = ServerState("test.paw", "mock")
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
    assert "FROM python:3.12-slim" in dockerfile
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
    for reserved in [".dockerignore", "docker-compose.yml", "README.md"]:
        reserved_adapter = tmp_path / reserved
        reserved_adapter.write_text("not a real adapter", encoding="utf-8")
        with pytest.raises(ValueError, match="collides with a file"):
            export_docker_scaffold(reserved_adapter, output_dir=out_dir / "unused")

    # A non-reserved name still requires the .paw extension.
    no_extension_adapter = tmp_path / "harmless-name"
    no_extension_adapter.write_text("not a real adapter", encoding="utf-8")
    with pytest.raises(ValueError, match="must have a '.paw' extension"):
        export_docker_scaffold(no_extension_adapter, output_dir=out_dir / "unused2")


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

    # 3. paw-kit export dataset
    db_file = tmp_path / "traces.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE traces (
            id INTEGER PRIMARY KEY,
            input TEXT,
            output TEXT,
            timestamp TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO traces (input, output, timestamp) VALUES (?, ?, ?);",
        ("test user input", "test output", "2026-09-06T00:00:00Z"),
    )
    conn.commit()
    conn.close()

    jsonl_out = tmp_path / "dataset.jsonl"
    res_dataset = runner.invoke(app, ["export", "dataset", "--db", str(db_file), "--out", str(jsonl_out)])
    assert res_dataset.exit_code == 0
    assert jsonl_out.exists()
    records = [json.loads(line) for line in jsonl_out.read_text(encoding="utf-8").strip().split("\n")]
    assert len(records) == 1
    assert records[0]["messages"][0]["content"] == "test user input"

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
