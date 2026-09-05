"""@compile_on_hit decorator implementing transparent tracing and JIT hot-swapping."""

from functools import wraps
import hashlib
import json
from pathlib import Path
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


def compile_on_hit(
    spec: str,
    threshold: int = 50,
    response_model: Optional[Type[BaseModel]] = None,
    cache_dir: str = "./.paw",
    backend: Optional[AbstractPAWBackend] = None,
    sync_compile: bool = False,
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
            call_count = db.record_trace(
                task_id=task_id,
                input_payload=input_payload,
                teacher_output=teacher_output_str,
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
