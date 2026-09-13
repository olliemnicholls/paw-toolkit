"""High-performance HTTP serving layer for compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints.

A-10: this used to say "with grammar-constrained decoding" -- the served path does not
have that. `paw_kit.schema.logits_processor.RegexLogitsProcessor` exists and is exported,
but no backend shipped in `paw_kit` calls it, and the SDK backend's `__call__` takes no
grammar hook at all (see `docs/real-backend.md`'s "Two upstream limitations"). What the
served path actually gets is post-hoc Pydantic validation of the model's output, which
`paw.load`'s callers already rely on and which is honestly described in
`paw_kit.schema.loader`. Grammar-constrained decoding is a real, separately measured
capability (see `measurements/`) -- just not one this HTTP layer applies.
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
