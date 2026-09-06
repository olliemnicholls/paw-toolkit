"""Data models for OpenAI Chat Completions, Anthropic Messages, RPC Invoke, and Metrics."""

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. OpenAI Chat Completions API Models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """OpenAI message format."""

    role: Literal["system", "user", "assistant", "tool", "function"] = "user"
    content: Union[str, List[Dict[str, Any]]]


class ResponseFormat(BaseModel):
    """Response format specification (e.g. json_object)."""

    type: Optional[str] = "text"


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completion request payload."""

    model: Optional[str] = "default"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = None
    stream: Optional[bool] = False


class ChatCompletionChoiceMessage(BaseModel):
    """Output message in an OpenAI choice."""

    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    """Single completion choice in OpenAI response."""

    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    """Token usage counters."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI Chat Completion response payload."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# ---------------------------------------------------------------------------
# 2. Anthropic Claude Messages API Models
# ---------------------------------------------------------------------------


class AnthropicContentBlock(BaseModel):
    """Single content block in an Anthropic message."""

    type: str = "text"
    text: str


class AnthropicMessage(BaseModel):
    """Anthropic user or assistant message."""

    role: Literal["user", "assistant"] = "user"
    content: Union[str, List[Dict[str, Any]]]


class AnthropicMessageRequest(BaseModel):
    """Anthropic Claude Messages request payload."""

    model: Optional[str] = "default"
    messages: List[AnthropicMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.0


class AnthropicUsage(BaseModel):
    """Anthropic token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicMessageResponse(BaseModel):
    """Anthropic Claude Messages response payload."""

    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[AnthropicContentBlock]
    model: str
    stop_reason: Optional[str] = "end_turn"
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage


# ---------------------------------------------------------------------------
# 3. Direct RPC Invoke Models
# ---------------------------------------------------------------------------


class InvokeRequest(BaseModel):
    """Direct RPC input payload."""

    input: str = Field(description="Raw string input to pass to the compiled neural adapter")
    parameters: Optional[Dict[str, Any]] = None


class InvokeResponse(BaseModel):
    """Direct RPC output payload."""

    output: Any = Field(description="Generated string or validated JSON object")
    latency_ms: float
    adapter: str
    model: str


# ---------------------------------------------------------------------------
# 4. Health and Metrics Models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Microservice health check status.

    PAW-SERVE-06: deliberately minimal. `/health` is unauthenticated by design (Track
    09's generated Dockerfile HEALTHCHECK and docker-compose healthcheck both call it
    with no credentials), so it must not disclose anything beyond liveness — no
    filesystem paths, no backend implementation details.
    """

    status: str = "ok"
    uptime_seconds: float
    version: str = "0.1.0"


class MetricsResponse(BaseModel):
    """Server performance and request metrics."""

    total_requests: int
    error_count: int
    uptime_seconds: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
