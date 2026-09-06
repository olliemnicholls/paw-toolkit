"""FastAPI HTTP microservice serving compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints with grammar-constrained decoding and telemetry.
"""

from collections import deque
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
import uuid

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.backend.real import RealPAWBackend
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


class ServerState:
    """Thread-safe runtime server telemetry state."""

    def __init__(self, adapter_path: str, backend_name: str) -> None:
        self.adapter_path = adapter_path
        self.backend_name = backend_name
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        self.latencies: deque[float] = deque(maxlen=10000)
        self._lock = threading.Lock()

    def record_request(self, latency_ms: float, is_error: bool = False) -> None:
        with self._lock:
            self.total_requests += 1
            if is_error:
                self.error_count += 1
            self.latencies.append(latency_ms)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            uptime = max(0.0, now - self.start_time)
            total = self.total_requests
            errors = self.error_count
            if not self.latencies:
                return {
                    "total_requests": total,
                    "error_count": errors,
                    "uptime_seconds": round(uptime, 2),
                    "p50_latency_ms": 0.0,
                    "p95_latency_ms": 0.0,
                    "p99_latency_ms": 0.0,
                }
            sorted_lat = sorted(self.latencies)
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
    """Extract flat string text from string or message content block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)


def _estimate_tokens(text: str) -> int:
    """Rough heuristic for token count estimation."""
    return max(1, len(text.split()) * 4 // 3)


def create_app(
    adapter_path: Union[str, Path],
    backend: Optional[AbstractPAWBackend] = None,
    response_model: Optional[Type[BaseModel]] = None,
    task_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> FastAPI:
    """Factory creating configured FastAPI microservice for the given .paw adapter."""
    path_obj = Path(adapter_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    selected_backend = backend or MockPAWBackend()
    backend_type = "real" if isinstance(selected_backend, RealPAWBackend) else "mock"
    # Return basename to avoid exposing host filesystem directory layout
    state = ServerState(path_obj.name, backend_type)
    configured_api_key = api_key or os.environ.get("PAW_API_KEY")
    inference_lock = threading.Lock()

    # Initialize execution function (with schema validation if model provided)
    if response_model is not None:
        exec_fn = load(
            adapter_path=str(path_obj),
            response_model=response_model,
            backend=selected_backend,
        )
    else:
        exec_fn = lambda inp: selected_backend.infer(str(path_obj), inp)

    app = FastAPI(
        title="PAW-Kit Microservice",
        description=f"Zero-marginal-cost neural microservice serving {path_obj.name}",
        version="0.1.0",
    )

    # Enforce safe CORS defaults (no credentials with wildcard origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Protect against unbounded request body sizes (10MB limit)
    @app.middleware("http")
    async def limit_payload_size(request: Request, call_next: Any) -> Response:
        MAX_BODY = 10 * 1024 * 1024  # 10 MB
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_BODY:
                    return Response(status_code=413, content="Payload Too Large (maximum 10MB)")
            except ValueError:
                return Response(status_code=400, content="Invalid Content-Length header")
        # For chunked encoding or missing Content-Length header, read and cap body
        body = await request.body()
        if len(body) > MAX_BODY:
            return Response(status_code=413, content="Payload Too Large (maximum 10MB)")
        return await call_next(request)

    def _verify_auth(request: Request) -> None:
        """Verify bearer token if PAW_API_KEY is configured using constant-time comparison."""
        if not configured_api_key:
            return
        auth = request.headers.get("authorization")
        if not auth or not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized: Missing or malformed Bearer token")
        token = auth[7:].strip()
        if not hmac.compare_digest(token, configured_api_key):
            raise HTTPException(status_code=401, detail="Unauthorized: Invalid API key")

    @app.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        metrics = state.get_metrics()
        return HealthResponse(
            status="ok",
            adapter_path=state.adapter_path,
            backend=state.backend_name,
            uptime_seconds=metrics["uptime_seconds"],
            version="0.1.0",
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def telemetry_metrics() -> MetricsResponse:
        return MetricsResponse(**state.get_metrics())

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(req: InvokeRequest, request: Request) -> InvokeResponse:
        _verify_auth(request)
        t0 = time.perf_counter()
        try:
            with inference_lock:
                res = exec_fn(req.input)
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=False)

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
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=True)
            logger.exception("Inference failed in /invoke: %s", exc)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(req: ChatCompletionRequest, request: Request) -> ChatCompletionResponse:
        _verify_auth(request)
        if req.stream:
            raise HTTPException(status_code=400, detail="Streaming is not yet supported")

        t0 = time.perf_counter()
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages array cannot be empty")

        # Extract system and last user prompt
        system_content = ""
        user_prompts = []
        for msg in req.messages:
            text = _extract_content(msg.content)
            if msg.role == "system":
                system_content = text
            elif msg.role in ("user", "tool"):
                user_prompts.append(text)

        last_user = user_prompts[-1] if user_prompts else ""
        combined_input = f"{system_content}\n\n{last_user}".strip() if system_content and last_user else (last_user or system_content)
        input_payload = combined_input

        try:
            with inference_lock:
                res = exec_fn(input_payload)
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=False)

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
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=True)
            logger.exception("Inference failed in /v1/chat/completions: %s", exc)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/v1/messages", response_model=AnthropicMessageResponse)
    def anthropic_messages(req: AnthropicMessageRequest, request: Request) -> AnthropicMessageResponse:
        _verify_auth(request)
        t0 = time.perf_counter()
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages array cannot be empty")

        system_text = _extract_content(req.system) if req.system else ""
        user_texts = []
        for msg in req.messages:
            if msg.role == "user":
                user_texts.append(_extract_content(msg.content))

        last_user = user_texts[-1] if user_texts else ""
        combined_input = f"{system_text}\n\n{last_user}".strip() if system_text and last_user else (last_user or system_text)
        input_payload = combined_input

        try:
            with inference_lock:
                res = exec_fn(input_payload)
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=False)

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
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000.0
            state.record_request(latency, is_error=True)
            logger.exception("Inference failed in /v1/messages: %s", exc)
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


def serve_adapter(
    adapter_path: Union[str, Path],
    host: str = "127.0.0.1",
    port: int = 8000,
    backend: Optional[AbstractPAWBackend] = None,
    response_model: Optional[Type[BaseModel]] = None,
    api_key: Optional[str] = None,
) -> None:
    """Start Uvicorn web server hosting the compiled adapter."""
    import uvicorn

    app = create_app(
        adapter_path=adapter_path,
        backend=backend,
        response_model=response_model,
        api_key=api_key,
    )
    uvicorn.run(app, host=host, port=port)
