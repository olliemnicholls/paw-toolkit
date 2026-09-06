"""High-performance HTTP serving layer for compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints with grammar-constrained decoding.
"""

from paw_kit.serve.models import (
    AnthropicMessageRequest,
    AnthropicMessageResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    InvokeRequest,
    InvokeResponse,
)
from paw_kit.serve.server import create_app, serve_adapter

__all__ = [
    "create_app",
    "serve_adapter",
    "ChatCompletionRequest",
    "ChatCompletionResponse",
    "AnthropicMessageRequest",
    "AnthropicMessageResponse",
    "InvokeRequest",
    "InvokeResponse",
]
