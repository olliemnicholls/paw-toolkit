"""High-performance HTTP serving layer for compiled .paw adapters.

Exposes OpenAI-compatible (/v1/chat/completions), Anthropic-compatible (/v1/messages),
and direct RPC (/invoke) endpoints.

A-10: this used to say the served path never applies grammar-constrained decoding.
That was true when the upstream SDK exposed no logits hook; it no longer is. When a
`response_model` is configured, `server.py` calls the schema-validating `paw.load`
(`:server.py`'s `exec_fn = load(...)` branch), and `ProgramAsWeightsBackend` applies
the constraint through that same public hook `infer()` always uses -- on by default
(`constrained_decoding=True`), so a served call gets it whenever the real backend is
selected and `llguidance` is importable, with no extra configuration on the server's
own part. Post-hoc Pydantic validation of the output (`paw_kit.schema.loader`) is
still the mechanism that catches everything the mask does not guarantee -- truncation
at `max_tokens`/the context window, and a masking failure, both of which still fail
open to the configured fallback -- so it remains load-bearing even with the mask on.
The `else` branch (`backend.infer()` called directly, no `response_model`) bypasses
schema handling entirely and gets neither mechanism, by the caller's own choice.
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
