"""@compile_on_hit decorator implementing transparent tracing and JIT hot-swapping."""

from functools import partial, wraps
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import re
import threading
import time
import types
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.jit.agreement import default_agreement_fn, stringify_answer
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER, ShadowJob
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


def _record_fail_open(
    task_id: str,
    exc: Exception,
    db: Optional[TraceDB] = None,
    shadow_window: int = 0,
    db_path: Optional[str] = None,
    queue_size: int = 8,
) -> None:
    # Track 14: also bump the *persisted* counter, so `paw-kit report` -- a different
    # process entirely -- can show the figure. The in-process counter below keeps its
    # exact existing meaning and reset-on-restart semantics.
    #
    # Finding 1: the persisted increment is a synchronous SQLite write
    # (`db.increment_fail_open`), and it must never run on *this* thread -- fail-open
    # exists to protect availability, so making the caller wait on a write lock right
    # here would defeat the point. At `shadow_window=0` there is no shadow worker for
    # this task at all (see shadow.py) and this is byte-for-byte the pre-Track-14
    # behaviour: no persisted counter, only the in-process one below. Otherwise the
    # increment is handed to the existing shadow worker's queue (drop-if-full, same as
    # every other shadow job) and applied there.
    if db is not None and shadow_window and db_path is not None:
        try:
            _GLOBAL_SHADOW_RUNNER.submit_fail_open(task_id, db_path, db, queue_size)
        except Exception:  # pragma: no cover - defense in depth
            pass
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


# Hard cap on `audit_rate`: past this the "sampled fraction" stops being a sample and
# starts being a second production workload paid for out of the caller's API budget.
_MAX_AUDIT_RATE = 0.5


def redact_sensitive_text(text: str) -> str:
    """Best-effort scrub of bearer tokens and password/secret/api_key/token values."""
    redacted = text
    for pattern, replacement in _REDACTION_RULES:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _should_audit(rng: random.Random, rate: float) -> bool:
    """Sample the post-promotion audit fraction.

    Module-level so tests can monkeypatch it, and driven by a `random.Random` instance
    owned by the wrapper closure rather than the global `random` module -- a caller who
    seeds `random` for reproducibility must not have their stream perturbed by a
    library's sampling decisions.
    """
    if rate <= 0:
        return False
    return rng.random() < rate


def _make_adapter_runner(
    task_id: str,
    adapter_path: str,
    response_model: Optional[Type[BaseModel]],
    backend: AbstractPAWBackend,
) -> Callable[[str], Any]:
    """Build the zero-I/O closure that runs the compiled adapter on one input.

    Constructing it does nothing; *calling* it performs the `os.stat`, the cache
    lookup and the `load()`/`infer()`. The same closure serves the request path in
    `ready` and the shadow worker in `shadow`, which is what lets `_ADAPTER_CALLABLE_CACHE`
    and `_invalidate_adapter_cache` stay in this module (where `tests/test_jit.py`
    reaches for them by name) while `shadow.py` imports nothing from here.
    """

    def _run(input_str: str) -> Any:
        # PAW-JIT-05(b): a single os.stat call does double duty -- its
        # FileNotFoundError *is* the existence check (subsuming the old
        # Path.exists() pre-check, and falling open via the caller's except
        # exactly like a stale-adapter inference failure would), and its
        # (mtime, size, inode) triple is the cache key's staleness component
        # when a response_model is in play.
        stat_result = os.stat(adapter_path)
        if response_model is not None:
            cache_key = (
                adapter_path,
                (stat_result.st_mtime_ns, stat_result.st_size, stat_result.st_ino),
                backend,
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
                    backend=backend,
                )
                with _ADAPTER_CALLABLE_CACHE_LOCK:
                    _ADAPTER_CALLABLE_CACHE.setdefault(task_id, {})[cache_key] = cached_fn
            return cached_fn(input_str)
        return backend.infer(adapter_path, input_str)

    return _run


def _validate_shadow_params(
    shadow_window: int,
    shadow_threshold: float,
    audit_window: int,
    audit_rate: float,
    demote_threshold: float,
    shadow_queue_size: int,
    shadow_max_pairs: int,
) -> None:
    """Reject an unusable shadow configuration at decoration time, never at call time."""
    if shadow_window < 0:
        raise ValueError(f"shadow_window must be >= 0 (0 disables shadow mode), got {shadow_window}")
    if not 0 < shadow_threshold <= 1:
        raise ValueError(f"shadow_threshold must be in (0, 1], got {shadow_threshold}")
    if audit_window < 1:
        raise ValueError(f"audit_window must be >= 1, got {audit_window}")
    if not 0 <= audit_rate <= _MAX_AUDIT_RATE:
        raise ValueError(
            f"audit_rate must be between 0 and {_MAX_AUDIT_RATE} (it spends real teacher "
            f"calls after promotion), got {audit_rate}"
        )
    if not 0 <= demote_threshold < shadow_threshold:
        raise ValueError(
            "demote_threshold must satisfy 0 <= demote_threshold < shadow_threshold "
            f"(strict hysteresis, so a task cannot flap on window noise), got "
            f"demote_threshold={demote_threshold} shadow_threshold={shadow_threshold}"
        )
    if shadow_queue_size < 1:
        raise ValueError(f"shadow_queue_size must be >= 1, got {shadow_queue_size}")
    if shadow_max_pairs < max(shadow_window, audit_window):
        raise ValueError(
            "shadow_max_pairs must be >= max(shadow_window, audit_window): a retention cap "
            "below the window prunes the table below a full window before it can ever fill, "
            "so the task could never promote or demote. Got "
            f"shadow_max_pairs={shadow_max_pairs}, shadow_window={shadow_window}, "
            f"audit_window={audit_window}"
        )


def compile_on_hit(
    spec: str,
    threshold: int = 50,
    response_model: Optional[Type[BaseModel]] = None,
    cache_dir: str = "./.paw",
    backend: Optional[AbstractPAWBackend] = None,
    sync_compile: bool = False,
    redact_trace: bool = False,
    *,
    shadow_window: int = 20,
    shadow_threshold: float = 0.8,
    audit_window: int = 20,
    audit_rate: float = 0.0,
    demote_threshold: float = 0.6,
    agreement_fn: Optional[Callable[[Any, Any], bool]] = None,
    shadow_queue_size: int = 8,
    shadow_max_pairs: int = 500,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator converting production LLM API calls into local neural functions.

    During initial invocations (hits < threshold), transparently calls the decorated
    function and logs input/output traces to SQLite. Once threshold is reached,
    background compilation triggers -- and the task enters **shadow mode**: the wrapped
    function keeps serving every call while the compiled adapter runs on the same
    inputs off the request path and its answers are compared against the teacher's.
    Only once agreement over a full window of real inputs clears `shadow_threshold`
    does the adapter start serving; from then on it falls back to the wrapped function
    upon error, exactly as before.

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
            Track 14: it now also governs all three text columns of `shadow_pairs`.
            The *comparison* still runs on the raw values -- the adapter is run on the
            input production actually sent, and redaction is applied only on the way
            to disk.
        shadow_window: Comparisons that must agree, in one tumbling window, before the
            adapter is promoted to serving. `0` disables shadow mode entirely and
            restores the pre-Track-14 behaviour of hot-swapping the instant compilation
            returns; it is the exact, tested escape hatch, and it also rescues a task
            already sitting in `shadow` from an earlier run.
        shadow_threshold: Agreement rate required to promote (default 0.8 = 16/20).
        audit_window: Window size for post-promotion drift detection.
        audit_rate: Fraction of served calls that additionally re-run the wrapped
            function on a background thread for a fresh comparison. **Defaults to 0.0
            (off).** It spends real teacher calls after the swap and re-invokes a
            function that may not be thread-safe, so it is opt-in; `0.05` is the
            recommended value when you want a drift signal. With the default, demotion
            is unreachable and there is no post-promotion drift signal.
        demote_threshold: Audit agreement below which a promoted task is demoted back
            to `shadow`. Must be strictly below `shadow_threshold` (hysteresis).
        agreement_fn: `(teacher, adapter) -> bool`. Defaults to `default_agreement_fn`,
            which is deliberately conservative: strings must match exactly after NFC
            normalisation and stripping, and are *not* casefolded. Without a
            `response_model` there is nothing to normalise through, so agreement is
            exact string equality of free-form model output -- which for anything but a
            short closed-vocabulary label will essentially never hold, and such a task
            will sit in `shadow` indefinitely with the teacher serving. Pass an
            `agreement_fn` (see `field_tolerance_agreement`) for those.
        shadow_queue_size: Bounded work-in-flight per task. On a full queue the newest
            comparison is dropped; a dropped comparison is neither an agreement nor a
            disagreement and never enters the window denominator.
        shadow_max_pairs: Per-task retention cap on `shadow_pairs`, oldest-first.

    Returns:
        Decorated callable function with JIT execution and fail-open routing.

    Raises:
        ValueError: at decoration time, for an unusable shadow configuration.
    """
    _validate_shadow_params(
        shadow_window, shadow_threshold, audit_window, audit_rate,
        demote_threshold, shadow_queue_size, shadow_max_pairs,
    )
    resolved_agreement_fn = agreement_fn or default_agreement_fn
    db_path = str(Path(cache_dir) / "traces.db")
    db = TraceDB(db_path=db_path)
    # PAW-JIT-05: register the same-place invalidation backstop for every TraceDB
    # this decorator creates -- see _invalidate_adapter_cache's docstring.
    db.register_status_listener(_invalidate_adapter_cache)

    shadow_config = types.MappingProxyType({
        "shadow_window": shadow_window,
        "shadow_threshold": shadow_threshold,
        "audit_window": audit_window,
        "audit_rate": audit_rate,
        "demote_threshold": demote_threshold,
        "shadow_queue_size": shadow_queue_size,
        "shadow_max_pairs": shadow_max_pairs,
        "agreement_fn": getattr(resolved_agreement_fn, "__name__", repr(resolved_agreement_fn)),
    })
    # Only the four parameters the window arithmetic depends on are reconciled across
    # runs -- see TraceDB.sync_shadow_config.
    persisted_config = {
        "shadow_window": shadow_window,
        "shadow_threshold": shadow_threshold,
        "audit_window": audit_window,
        "demote_threshold": demote_threshold,
    }

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

        # Owned by this closure, never the global `random` module: a caller who seeds
        # `random` for reproducibility must not be perturbed by audit sampling.
        rng = random.Random()

        if shadow_window:
            try:
                db.sync_shadow_config(task_id, persisted_config)
            except Exception:  # pragma: no cover - never break decoration
                logger.debug("paw_kit.jit: could not persist shadow config for %s", task_id)

        def _submit_shadow_job(**kwargs: Any) -> None:
            """Enqueue one comparison. Swallows everything: shadow work never raises."""
            try:
                _GLOBAL_SHADOW_RUNNER.submit(
                    ShadowJob(
                        task_id=task_id,
                        db_path=db_path,
                        db=db,
                        response_model=response_model,
                        agreement_fn=resolved_agreement_fn,
                        redact_fn=redact_sensitive_text if redact_trace else None,
                        shadow_window=shadow_window,
                        audit_window=audit_window,
                        shadow_threshold=shadow_threshold,
                        demote_threshold=demote_threshold,
                        max_pairs=shadow_max_pairs,
                        queue_size=shadow_queue_size,
                        **kwargs,
                    )
                )
            except Exception as exc:  # pragma: no cover - defense in depth
                logger.debug(
                    "paw_kit.jit: task_id=%s could not enqueue a shadow comparison (%s: %s).",
                    task_id, type(exc).__name__, exc,
                )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            active_backend = backend or get_default_backend()
            input_payload = _serialize_input(args, kwargs)

            # 1. One SELECT decides how this call is routed.
            status, adapter_path, state_epoch = db.get_task_routing(task_id)

            # Escape hatch. `shadow_window=0` must also rescue a task that a previous
            # run left sitting in `shadow`: otherwise the opt-out strands exactly the
            # users who tried the default first -- teacher forever, adapter compiled
            # and never used.
            if shadow_window == 0 and status == "shadow" and adapter_path:
                try:
                    db.try_promote(task_id, state_epoch, None, 0, reason="shadow_disabled")
                except Exception:  # pragma: no cover - never break the request path
                    pass
                status, adapter_path, state_epoch = db.get_task_routing(task_id)

            # 2. Promoted: the adapter serves, with the fail-open path unchanged.
            if status == "ready" and adapter_path:
                run_adapter = _make_adapter_runner(
                    task_id, adapter_path, response_model, active_backend
                )
                try:
                    served_start = time.perf_counter()
                    result = run_adapter(input_payload)
                    served_latency_ms = (time.perf_counter() - served_start) * 1000
                except Exception as exc:
                    # Fail-Open Safety: transparently route to wrapped function on local failure.
                    # Transparent to the *caller* deliberately stays true -- this only adds a
                    # log line and a counter a developer has to go looking for, not any change
                    # to the return value or exception behavior on this path. A fail-open is an
                    # infrastructure fault, not semantic drift: it never enters the audit
                    # window and never counts toward demotion.
                    _record_fail_open(task_id, exc, db, shadow_window, db_path, shadow_queue_size)
                    return func(*args, **kwargs)

                if shadow_window and _should_audit(rng, audit_rate):
                    try:
                        _submit_shadow_job(
                            state_epoch=state_epoch,
                            phase="audit",
                            input_payload=input_payload,
                            adapter_output=stringify_answer(result),
                            adapter_latency_ms=served_latency_ms,
                            # Deliberately holds the caller's own args/kwargs by
                            # reference and calls func on a worker thread: the wrapped
                            # function must be thread-safe, and arguments mutated after
                            # this call returns will be seen in their mutated form.
                            run_teacher=partial(func, *args, **kwargs),
                        )
                    except Exception:  # pragma: no cover - defense in depth
                        pass
                return result  # type: ignore[return-value]

            # 3. Adapter not serving: invoke wrapped function (teacher)
            start_time = time.perf_counter()
            teacher_result = func(*args, **kwargs)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Serialize output for tracing. Finding 9: one shared serializer
            # (paw_kit.jit.agreement.stringify_answer) for this trace-persistence
            # path, the shadow-comparison persistence path in shadow.py, and the
            # str-vs-structured comparison rule in agreement.py itself -- a teacher
            # returning e.g. a tuple now serializes identically on every path.
            teacher_output_str = stringify_answer(teacher_result)

            # 4. Record trace and increment counter
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

            # 5. In shadow, hand the same input to the adapter off the request path.
            #    One put_nowait of a frozen dataclass: no I/O on this thread.
            if status == "shadow" and shadow_window and adapter_path:
                _submit_shadow_job(
                    state_epoch=state_epoch,
                    phase="shadow",
                    input_payload=input_payload,
                    teacher_output=teacher_output_str,
                    teacher_latency_ms=latency_ms,
                    run_adapter=_make_adapter_runner(
                        task_id, adapter_path, response_model, active_backend
                    ),
                )

            # 6. Trigger background compilation once threshold reached.
            #    The status here must be a *fresh* read, not the routing snapshot
            #    above: that snapshot predates the teacher call, which can take
            #    seconds, and a background compile finishing during it would leave the
            #    snapshot reading "tracing" and re-trigger compilation underneath a
            #    running shadow worker (compiler.py's duplicate guard is skipped
            #    entirely when sync=True). One extra SELECT, on a path already gated
            #    behind call_count >= threshold, is the correct price.
            if call_count >= threshold and db.get_status(task_id) == "tracing":
                target_adapter_path = str(Path(cache_dir) / f"{task_id}.paw")
                _GLOBAL_COMPILER.trigger_compilation(
                    task_id=task_id,
                    spec=spec,
                    db=db,
                    backend=active_backend,
                    output_path=target_adapter_path,
                    sync=sync_compile,
                    promote_to="shadow" if shadow_window else "ready",
                )

            return teacher_result

        def _get_agreement(n: int = 5) -> Dict[str, Any]:
            report = db.get_task_report(task_id)
            block = report["agreement"]
            runner_stats = _GLOBAL_SHADOW_RUNNER.stats(task_id, db_path)
            return {
                "state": report["status"],
                "phase": block["phase"],
                "rate": block["rate"],
                "window": block["window"],
                "samples": block["samples"],
                "agree": block["agree"],
                "disagree": block["disagree"],
                "error": block["error"],
                "teacher_error": block["teacher_error"],
                # Finding 2: persisted, epoch-scoped -- True once the runner has
                # stopped evaluating this epoch's completed windows for promotion (see
                # shadow.py's stall-guard WARNING for the two ways out). Not to be
                # confused with `stall_subsampled` below.
                "stalled": block["stalled"],
                # In-process, since process start, and per-process only: two processes
                # running the same decorated function share the database (and therefore
                # the promotion decision) but not these.
                "dropped": runner_stats["dropped"],
                # How many comparisons *this process* personally skipped under the
                # stall guard's subsampling -- distinct from `stalled` above, which is
                # the persisted, DB-derived "has this epoch passed the stall point"
                # flag; this can be 0 on a freshly started process even for a task that
                # is, in fact, stalled.
                "stall_subsampled": runner_stats["stalled"],
                "pending": runner_stats["pending"],
                # The *persisted* (therefore possibly redacted) text.
                "last_disagreements": db.get_recent_disagreements(task_id, n),
            }

        # Expose testing and inspection metadata
        wrapper.task_id = task_id  # type: ignore[attr-defined]
        wrapper.db = db  # type: ignore[attr-defined]
        wrapper.get_call_count = lambda: db.get_call_count(task_id)  # type: ignore[attr-defined]
        # Deliberately unchanged: "the compiled adapter is serving this call". In
        # `shadow` the adapter exists but is not serving, so this is False -- which is
        # exactly why the demo and the README example pass shadow_window=0.
        wrapper.is_compiled = lambda: db.get_adapter_path(task_id) is not None  # type: ignore[attr-defined]
        wrapper.has_adapter = lambda: db.get_task_routing(task_id)[0] in ("shadow", "ready")  # type: ignore[attr-defined]
        wrapper.get_agreement = _get_agreement  # type: ignore[attr-defined]
        wrapper.shadow_config = shadow_config  # type: ignore[attr-defined]
        # PAW-JIT: see _record_fail_open -- in-process count of "no signal" fail-opens
        # for this task since process start, not persisted across restarts.
        wrapper.get_fail_open_count = lambda: _FAIL_OPEN_COUNTS.get(task_id, 0)  # type: ignore[attr-defined]
        return wrapper

    return decorator
