"""@compile_on_hit decorator implementing transparent tracing and JIT hot-swapping."""

from functools import wraps
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.schema.loader import get_default_backend, load

T = TypeVar("T")

logger = logging.getLogger("paw_kit.jit")

_GLOBAL_COMPILER = BackgroundCompiler()

# Deferred-topic fix (conductor/deferred/index.md, "Silent fail-open, no signal"):
# the fail-open except block below used to be silent -- no log, no counter -- on the
# path the README recommends for real production use. `_FAIL_OPEN_COUNTS` is a plain
# in-process counter (reset on restart, not persisted -- this is a signal for "is this
# happening at all," not an audit log) keyed on task_id, read via
# `wrapper.get_fail_open_count()`.
_FAIL_OPEN_COUNTS: Dict[str, int] = {}
_FAIL_OPEN_COUNTS_LOCK = threading.Lock()


def _record_fail_open(task_id: str, exc: Exception) -> None:
    with _FAIL_OPEN_COUNTS_LOCK:
        _FAIL_OPEN_COUNTS[task_id] = _FAIL_OPEN_COUNTS.get(task_id, 0) + 1
        count = _FAIL_OPEN_COUNTS[task_id]
    logger.warning(
        "paw_kit.jit fail-open: task_id=%s fell back to the wrapped function after a "
        "local-inference error (%s: %s). This is call #%d for this task since process "
        "start -- if that number keeps climbing, the compiled adapter is not being used "
        "and every call is silently paying the wrapped function's own cost instead.",
        task_id, type(exc).__name__, exc, count,
    )

# PAW-JIT-05: cache of loaded adapter callables, keyed on
# task_id -> {(adapter_path, stat-identity, backend): callable}. The outer task_id
# level exists so the same-process invalidation hook (below) can drop every cached
# callable for a task in one dict operation whenever TraceDB.set_status writes a new
# status/adapter_path for it -- the only handle that write site has is a task_id, not
# a specific (adapter_path, stat-identity, backend) combination.
#
# The primary invalidation mechanism is the key itself, not this hook: `stat-identity`
# is `(st_mtime_ns, st_size, st_ino)` from a single `os.stat(adapter_path)` call, which
# also serves as the existence check (a deleted adapter raises FileNotFoundError,
# caught by the fail-open try/except in the wrapper below). `backend` is included
# because `paw_kit.schema.loader.set_default_backend` can swap the process-wide
# default backend mid-run (decorator.py re-resolves `backend or get_default_backend()`
# on every call), which would otherwise be a staleness vector the key doesn't cover.
_ADAPTER_CALLABLE_CACHE: Dict[str, Dict[Tuple[str, Tuple[int, int, int], Any], Callable[[str], Any]]] = {}
_ADAPTER_CALLABLE_CACHE_LOCK = threading.Lock()


def _invalidate_adapter_cache(task_id: str) -> None:
    """Same-process backstop invalidation hook, registered against each TraceDB via
    `register_status_listener` (PAW-JIT-05). Not the primary mechanism -- see the
    module-level comment on `_ADAPTER_CALLABLE_CACHE` -- but a backstop for a
    filesystem reporting `st_ino == 0` (some SMB/FUSE mounts) or a third-party
    `AbstractPAWBackend` that rewrites an adapter file in place rather than through
    `paw_kit.atomicio.atomic_write_text`'s `os.replace`. `TraceDB.set_status` is an
    instance method, so this hook only fires in the process that calls it -- exactly
    why the stat-identity component of the key is the mechanism that has to work
    across processes, not this one.
    """
    with _ADAPTER_CALLABLE_CACHE_LOCK:
        _ADAPTER_CALLABLE_CACHE.pop(task_id, None)


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
    # PAW-JIT-05: register the same-process invalidation backstop for every TraceDB
    # this decorator creates -- see _invalidate_adapter_cache's docstring.
    db.register_status_listener(_invalidate_adapter_cache)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Compute deterministic task ID from function signature and spec.
        # PAW-JIT-06: the full 64 hex characters, not a 16-char truncation -- a
        # truncated SHA-256 trades a cryptographically negligible collision
        # probability for one that is merely small, for no benefit (task_id is never
        # human-typed or displayed anywhere space-constrained). Note this changes the
        # generated `.paw` filename derived from task_id below, so upgrading orphans
        # any pre-existing cache entry keyed on the old 16-char id -- not a pure
        # one-liner.
        qualname = f"{func.__module__}.{func.__qualname__}"
        task_id = hashlib.sha256(f"{qualname}:{spec}".encode("utf-8")).hexdigest()

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            active_backend = backend or get_default_backend()
            input_payload = _serialize_input(args, kwargs)

            # 1. Check if adapter is compiled and ready
            adapter_path = db.get_adapter_path(task_id)
            if adapter_path:
                try:
                    # PAW-JIT-05(b): a single os.stat call does double duty -- its
                    # FileNotFoundError *is* the existence check (subsuming the old
                    # Path.exists() pre-check, and falling open via the except below
                    # exactly like a stale-adapter inference failure would), and its
                    # (mtime, size, inode) triple is the cache key's staleness
                    # component when a response_model is in play.
                    stat_result = os.stat(adapter_path)
                    if response_model is not None:
                        cache_key = (
                            adapter_path,
                            (stat_result.st_mtime_ns, stat_result.st_size, stat_result.st_ino),
                            active_backend,
                        )
                        with _ADAPTER_CALLABLE_CACHE_LOCK:
                            cached_fn = _ADAPTER_CALLABLE_CACHE.get(task_id, {}).get(cache_key)
                        if cached_fn is None:
                            # PAW-JIT-05: cache the loaded adapter callable instead of
                            # re-load()-ing (recompiling the grammar regex, rebuilding
                            # the closure) on every single call.
                            cached_fn = load(
                                adapter_path=adapter_path,
                                response_model=response_model,
                                backend=active_backend,
                            )
                            with _ADAPTER_CALLABLE_CACHE_LOCK:
                                _ADAPTER_CALLABLE_CACHE.setdefault(task_id, {})[cache_key] = cached_fn
                        return cached_fn(input_payload)  # type: ignore[return-value]
                    else:
                        output_str = active_backend.infer(adapter_path, input_payload)
                        return output_str  # type: ignore[return-value]
                except Exception as exc:
                    # Fail-Open Safety: transparently route to wrapped function on local failure.
                    # Transparent to the *caller* deliberately stays true -- this only adds a
                    # log line and a counter a developer has to go looking for, not any change
                    # to the return value or exception behavior on this path.
                    _record_fail_open(task_id, exc)
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
        # PAW-JIT: see _record_fail_open -- in-process count of "no signal" fail-opens
        # for this task since process start, not persisted across restarts.
        wrapper.get_fail_open_count = lambda: _FAIL_OPEN_COUNTS.get(task_id, 0)  # type: ignore[attr-defined]
        return wrapper

    return decorator
