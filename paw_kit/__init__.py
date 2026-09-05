"""paw-kit: Production Runtime & Reliability Toolkit for Program-as-Weights (PAW)."""

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.jit.decorator import compile_on_hit
from paw_kit.schema.exceptions import PAWSchemaError, PAWSyntaxError
from paw_kit.schema.grammar import pydantic_to_regex
from paw_kit.schema.loader import load
from paw_kit.schema.logits_processor import RegexLogitsProcessor

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AbstractPAWBackend",
    "BackgroundCompiler",
    "MockPAWBackend",
    "PAWSchemaError",
    "PAWSyntaxError",
    "RegexLogitsProcessor",
    "TraceDB",
    "compile_on_hit",
    "load",
    "pydantic_to_regex",
]
