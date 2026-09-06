"""@compile_on_hit decorator implementing transparent tracing and JIT hot-swapping."""

from functools import wraps
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Optional, Type, TypeVar, Union
from pydantic import BaseModel

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.schema.loader import get_default_backend, load

T = TypeVar("T")

_GLOBAL_COMPILER = BackgroundCompiler()


def _serialize_input(args: tuple, kwargs: dict) -> str:
    """Serialize function arguments into a canonical input string."""
    if len(args) == 1 and not kwargs and isinstance(args[0], str):
        return args[0]
    if len(args) == 1 and not kwargs and isinstance(args[0], BaseModel):
        return args[0].model_dump_json()
    try:
        return json.dumps({"args": args, "kwargs": kwargs}, default=str)
    except Exception:
        return str(args) + str(kwargs)


# PAW-JIT-02: regex-based best-effort scrubbing of obviously-sensitive substrings
# (bearer tokens, password/secret/api_key/token key-value pairs) before a trace is
# persisted to traces.db. This is *not* the primary mitigation for "raw prompt text
# sits in a world-readable SQLite file" -- that's PAW-JIT-01's restrictive file
# permissions. It exists as an explicit, opt-in trade: `traces.db` rows are the
# training corpus for compilation (see compiler.py -- input_payload/teacher_output are
# read straight out of it and passed to backend.compile() as `examples`), not an audit
# log, so redacting them necessarily degrades what the compiled adapter can learn.
# Defaulting this off keeps that quality intact for deployments that don't need it;
# `redact_trace=True` opts in for deployments that would rather trade quality for it.
_REDACTION_RULES = [
    (re.compile(r"(?i)(bearer\s+)[a-zA-Z0-9_\-.]{10,}"), r"\1[REDACTED]"),
    (re.compile(r'(?i)(password|secret|api_key|token)(["\']?\s*[:=]\s*["\']?)[^"\',;\s]+'), r"\1\2[REDACTED]"),
]


def redact_sensitive_text(text: str) -> str:
    """Best-effort scrub of bearer tokens and password/secret/api_key/token values."""
    redacted = text
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def compile_on_hit(
    spec: str,
    threshold: int = 50,
    response_model: Optional[Type[BaseModel]] = None,
    cache_dir: str = "./.paw",
    backend: Optional[AbstractPAWBackend] = None,
    sync_compile: bool = False,
    redact_trace: bool = False,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator converting production LLM API calls into local neural functions.

    During initial invocations (hits < threshold), transparently calls the decorated
    function and logs input/output traces to SQLite. Once threshold is reached,
    background compilation triggers. Subsequent calls are automatically routed to
    the local .paw adapter, falling back to the wrapped function upon error.

    Args:
        spec: Natural language task specification.
        threshold: Hit count required to trigger compilation.
        response_model: Optional Pydantic BaseModel enforcing structured JSON decoding.
        cache_dir: Storage directory for SQLite traces and compiled .paw weights.
        backend: PAW backend implementation. Uses default mock backend if None.
        sync_compile: If True, executes compilation synchronously (useful for testing).
        redact_trace: PAW-JIT-02. If True, best-effort scrub bearer tokens and
            password/secret/api_key/token values out of a call's input/output before
            persisting it to traces.db. Defaults to False: input_payload/teacher_output
            are compiler.py's training corpus for the compiled adapter (read straight
            out of traces.db and passed to backend.compile() as `examples`), not an
            audit log, so redacting them by default would silently degrade every
            task's compiled output the moment it crosses the compile threshold.
            PAW-JIT-01's restrictive file permissions on traces.db are the primary
            mitigation for unredacted trace data at rest; this is an explicit,
            quality-for-confidentiality trade-off for deployments that want it.

    Returns:
        Decorated callable function with JIT execution and fail-open routing.
    """
    db_path = str(Path(cache_dir) / "traces.db")
    db = TraceDB(db_path=db_path)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Compute deterministic task ID from function signature and spec
        qualname = f"{func.__module__}.{func.__qualname__}"
        task_id = hashlib.sha256(f"{qualname}:{spec}".encode("utf-8")).hexdigest()[:16]

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            active_backend = backend or get_default_backend()
            input_payload = _serialize_input(args, kwargs)

            # 1. Check if adapter is compiled and ready
            adapter_path = db.get_adapter_path(task_id)
            if adapter_path and Path(adapter_path).exists():
                try:
                    if response_model is not None:
                        # Load and validate with grammar constraint
                        adapter_fn = load(
                            adapter_path=adapter_path,
                            response_model=response_model,
                            backend=active_backend,
                        )
                        return adapter_fn(input_payload)  # type: ignore[return-value]
                    else:
                        output_str = active_backend.infer(adapter_path, input_payload)
                        return output_str  # type: ignore[return-value]
                except Exception:
                    # Fail-Open Safety: transparently route to wrapped function on local failure
                    return func(*args, **kwargs)

            # 2. Adapter not ready: invoke wrapped function (teacher)
            start_time = time.perf_counter()
            teacher_result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Serialize output for tracing
            if isinstance(teacher_result, BaseModel):
                teacher_output_str = teacher_result.model_dump_json()
            elif isinstance(teacher_result, (dict, list)):
                teacher_output_str = json.dumps(teacher_result)
            else:
                teacher_output_str = str(teacher_result)

            # 3. Record trace and increment counter
            # PAW-JIT-02: redaction (opt-in, see redact_trace docstring above) is
            # applied only to what gets persisted -- input_payload/teacher_result
            # above are untouched, so the function's actual return value to the
            # caller is never redacted.
            traced_input = redact_sensitive_text(input_payload) if redact_trace else input_payload
            traced_output = redact_sensitive_text(teacher_output_str) if redact_trace else teacher_output_str
            call_count = db.record_trace(
                task_id=task_id,
                input_payload=traced_input,
                teacher_output=traced_output,
                latency_ms=latency_ms,
            )

            # 4. Trigger background compilation once threshold reached
            if call_count >= threshold and db.get_status(task_id) == "tracing":
                target_adapter_path = str(Path(cache_dir) / f"{task_id}.paw")
                _GLOBAL_COMPILER.trigger_compilation(
                    task_id=task_id,
                    spec=spec,
                    db=db,
                    backend=active_backend,
                    output_path=target_adapter_path,
                    sync=sync_compile,
                )

            return teacher_result

        # Expose testing and inspection metadata
        wrapper.task_id = task_id  # type: ignore[attr-defined]
        wrapper.db = db  # type: ignore[attr-defined]
        wrapper.get_call_count = lambda: db.get_call_count(task_id)  # type: ignore[attr-defined]
        wrapper.is_compiled = lambda: db.get_adapter_path(task_id) is not None  # type: ignore[attr-defined]
        return wrapper

    return decorator
