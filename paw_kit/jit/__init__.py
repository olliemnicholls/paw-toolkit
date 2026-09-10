"""paw.jit: Just-In-Time decorator, call tracing, and background compilation."""

from paw_kit.jit.agreement import default_agreement_fn, field_tolerance_agreement
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.jit.decorator import compile_on_hit

__all__ = [
    "BackgroundCompiler",
    "TraceDB",
    "compile_on_hit",
    "default_agreement_fn",
    "field_tolerance_agreement",
]
