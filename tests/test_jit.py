"""Unit and integration tests for paw.jit: SQLite tracing, @compile_on_hit, and hot-swapping."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sqlite3
import stat
import time
from typing import List
from pydantic import BaseModel
import pytest

from paw_kit import (
    MockPAWBackend,
    TraceDB,
    compile_on_hit,
)
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.decorator import redact_sensitive_text


class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float


def _mp_record_trace_worker(db_path: str, task_id: str, n: int) -> None:
    """Top-level (picklable) worker for test_trace_db_survives_genuine_multiprocess_write_contention_PAW_JIT_04."""
    from paw_kit.jit.db import TraceDB

    worker_db = TraceDB(db_path=db_path)
    for i in range(n):
        worker_db.record_trace(task_id, f"input-{i}", "output", 1.0)
    worker_db.close()


def test_trace_db_operations(tmp_path: Path) -> None:
    """Verify TraceDB initialization, trace logging, and status transitions."""
    db_file = str(tmp_path / "test_traces.db")
    db = TraceDB(db_path=db_file)

    task_id = "task_abc123"
    assert db.get_call_count(task_id) == 0
    assert db.get_status(task_id) == "tracing"
    assert db.get_adapter_path(task_id) is None

    # Record traces
    count1 = db.record_trace(task_id, "I love this!", '{"sentiment": "positive"}', 15.2)
    assert count1 == 1
    assert db.get_call_count(task_id) == 1

    count2 = db.record_trace(task_id, "I hate this!", '{"sentiment": "negative"}', 18.7)
    assert count2 == 2
    assert db.get_call_count(task_id) == 2

    # Fetch traces
    traces = db.get_traces(task_id)
    assert len(traces) == 2
    assert traces[0]["input_payload"] == "I love this!"
    assert traces[1]["latency_ms"] == 18.7

    # Status transitions
    db.set_status(task_id, "ready", adapter_path="/path/to/adapter.paw")
    assert db.get_status(task_id) == "ready"
    assert db.get_adapter_path(task_id) == "/path/to/adapter.paw"

    db.close()


def test_trace_db_concurrent_writes(tmp_path: Path) -> None:
    """Verify thread-safe concurrent trace recording without lock contention errors."""
    db_file = str(tmp_path / "concurrent_traces.db")
    db = TraceDB(db_path=db_file)
    task_id = "concurrent_task"

    num_threads = 10
    calls_per_thread = 20

    def worker(thread_idx: int) -> None:
        for i in range(calls_per_thread):
            db.record_trace(
                task_id=task_id,
                input_payload=f"thread_{thread_idx}_call_{i}",
                teacher_output="ok",
                latency_ms=1.0,
            )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in futures:
            f.result()

    assert db.get_call_count(task_id) == num_threads * calls_per_thread
    traces = db.get_traces(task_id)
    assert len(traces) == num_threads * calls_per_thread
    db.close()


def test_compile_on_hit_tracing_and_hotswap(tmp_path: Path) -> None:
    """Verify @compile_on_hit passes through until threshold, then hot-swaps to adapter."""
    cache_dir = str(tmp_path / "paw_cache")
    backend = MockPAWBackend()

    teacher_calls = 0

    @compile_on_hit(
        spec="Classify sentiment into positive, neutral, negative",
        threshold=3,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,  # Synchronous compilation for deterministic test assertions
    )
    def classify_text(text: str) -> str:
        nonlocal teacher_calls
        teacher_calls += 1
        return f"teacher:{text}"

    assert classify_text.get_call_count() == 0  # type: ignore[attr-defined]
    assert not classify_text.is_compiled()  # type: ignore[attr-defined]

    # Call 1: hits teacher
    res1 = classify_text("hello")
    assert res1 == "teacher:hello"
    assert teacher_calls == 1
    assert classify_text.get_call_count() == 1  # type: ignore[attr-defined]

    # Call 2: hits teacher
    res2 = classify_text("world")
    assert res2 == "teacher:world"
    assert teacher_calls == 2
    assert classify_text.get_call_count() == 2  # type: ignore[attr-defined]

    # Call 3: reaches threshold 3 -> triggers compilation!
    res3 = classify_text("great")
    assert res3 == "teacher:great"
    assert teacher_calls == 3
    assert classify_text.get_call_count() == 3  # type: ignore[attr-defined]
    assert classify_text.is_compiled()  # type: ignore[attr-defined]

    # Call 4: should now execute locally on the mock adapter!
    # MockPAWBackend returns the training example output if matching, or deterministic string
    res4 = classify_text("hello")
    assert teacher_calls == 3  # Teacher was NOT called!
    assert res4 == "teacher:hello"  # Example matched from recorded training set!


def test_compile_on_hit_with_pydantic_model(tmp_path: Path) -> None:
    """Verify @compile_on_hit works with structured Pydantic response models."""
    cache_dir = str(tmp_path / "paw_cache_pydantic")
    backend = MockPAWBackend()

    teacher_invocations = 0

    @compile_on_hit(
        spec="Classify sentiment",
        threshold=2,
        response_model=SentimentOutput,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def analyze_sentiment(input_text: str) -> SentimentOutput:
        nonlocal teacher_invocations
        teacher_invocations += 1
        return SentimentOutput(sentiment="positive", confidence=0.95)

    # Call 1: Teacher
    res1 = analyze_sentiment("Great product!")
    assert teacher_invocations == 1
    assert isinstance(res1, SentimentOutput)
    assert res1.sentiment == "positive"

    # Call 2: Hits threshold -> compiles
    res2 = analyze_sentiment("Great product!")
    assert teacher_invocations == 2
    assert analyze_sentiment.is_compiled()  # type: ignore[attr-defined]

    # Call 3: Hot-swapped local execution
    res3 = analyze_sentiment("Great product!")
    assert teacher_invocations == 2  # Teacher not invoked!
    assert isinstance(res3, SentimentOutput)
    assert res3.sentiment == "positive"
    assert res3.confidence == 0.95


def test_compile_on_hit_fail_open_safety(tmp_path: Path, caplog) -> None:
    """Verify Fail-Open safety: local adapter errors fall back transparently to teacher,
    and (conductor/deferred/index.md, "Silent fail-open, no signal") that the fallback
    is no longer silent: it logs and increments a per-task counter a developer can poll."""
    cache_dir = str(tmp_path / "paw_cache_failopen")
    backend = MockPAWBackend()

    teacher_invocations = 0

    @compile_on_hit(
        spec="Failing local adapter test",
        threshold=1,
        response_model=SentimentOutput,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def robust_service(input_text: str) -> SentimentOutput:
        nonlocal teacher_invocations
        teacher_invocations += 1
        return SentimentOutput(sentiment="fallback_positive", confidence=1.0)

    # Call 1: hits threshold and compiles
    robust_service("input")
    assert robust_service.is_compiled()  # type: ignore[attr-defined]
    assert teacher_invocations == 1
    assert robust_service.get_fail_open_count() == 0  # type: ignore[attr-defined]

    # Now intentionally break the compiled adapter to simulate runtime exception/corruption
    adapter_path = robust_service.db.get_adapter_path(robust_service.task_id)  # type: ignore[attr-defined]
    backend.set_default_response(adapter_path, "MALFORMED_OUTPUT_CAUSING_PARSE_ERROR")

    # Call 2: Local adapter throws/fails schema parsing -> must transparently fall back!
    with caplog.at_level("WARNING", logger="paw_kit.jit"):
        res = robust_service("unknown input")
    assert teacher_invocations == 2  # Teacher was engaged as fail-open fallback!
    assert res.sentiment == "fallback_positive"
    assert robust_service.get_fail_open_count() == 1  # type: ignore[attr-defined]
    assert any("fail-open" in rec.message for rec in caplog.records)

    # Call 3: same broken adapter again -> counter accumulates, not just flips a flag.
    robust_service("another unknown input")
    assert robust_service.get_fail_open_count() == 2  # type: ignore[attr-defined]


def test_background_compiler_duplicate_prevention(tmp_path: Path) -> None:
    """Verify BackgroundCompiler prevents duplicate concurrent compilation threads."""
    db_file = str(tmp_path / "compiler_db.db")
    db = TraceDB(db_path=db_file)
    task_id = "test_dup"
    db.record_trace(task_id, "inp", "out", 1.0)

    compiler = BackgroundCompiler()
    backend = MockPAWBackend()
    out_path = str(tmp_path / "out.paw")

    # First trigger
    t1 = compiler.trigger_compilation(task_id, "spec", db, backend, out_path, sync=False)
    assert t1 is not None

    # Immediate duplicate trigger while active
    t2 = compiler.trigger_compilation(task_id, "spec", db, backend, out_path, sync=False)
    assert t2 is None  # Should be ignored / return None

    t1.join(timeout=2.0)
    assert db.get_status(task_id) == "ready"
    db.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits don't apply on Windows")
def test_trace_db_restricts_directory_and_file_permissions_PAW_JIT_01(tmp_path: Path) -> None:
    """Verify the .paw cache directory and traces.db are created owner-only (0700/0600)."""
    cache_dir = tmp_path / "paw_perms_cache"
    db_file = cache_dir / "traces.db"
    db = TraceDB(db_path=str(db_file))

    dir_mode = stat.S_IMODE(os.stat(cache_dir).st_mode)
    file_mode = stat.S_IMODE(os.stat(db_file).st_mode)
    assert dir_mode == 0o700, f"expected cache dir mode 0700, got {oct(dir_mode)}"
    assert file_mode == 0o600, f"expected traces.db mode 0600, got {oct(file_mode)}"
    db.close()


def test_redact_sensitive_text_scrubs_bearer_tokens_and_secrets_PAW_JIT_02() -> None:
    """Verify the redaction helper scrubs bearer tokens and password/secret/api_key/token values."""
    text = 'Authorization: Bearer sk-abcdefghijklmnop, password="hunter2", api_key=sk-live-99999'
    redacted = redact_sensitive_text(text)
    assert "sk-abcdefghijklmnop" not in redacted
    assert "hunter2" not in redacted
    assert "sk-live-99999" not in redacted
    assert "[REDACTED]" in redacted
    # Non-sensitive text is left alone.
    assert redact_sensitive_text("just a normal sentence") == "just a normal sentence"


def test_compile_on_hit_redact_trace_defaults_false_preserves_raw_trace_PAW_JIT_02(tmp_path: Path) -> None:
    """Verify redact_trace defaults to False: traces.db keeps the raw, unredacted text."""
    cache_dir = str(tmp_path / "paw_cache_no_redact")
    backend = MockPAWBackend()

    @compile_on_hit(
        spec="Echo the secret",
        threshold=1,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def handle(text: str) -> str:
        return f"teacher:{text}"

    handle("Authorization: Bearer sk-abcdefghijklmnop")
    traces = handle.db.get_traces(handle.task_id)  # type: ignore[attr-defined]
    assert any("sk-abcdefghijklmnop" in t["input_payload"] for t in traces)


def test_compile_on_hit_redact_trace_true_scrubs_persisted_trace_only_PAW_JIT_02(tmp_path: Path) -> None:
    """Verify redact_trace=True scrubs what's persisted, without altering the actual return value."""
    cache_dir = str(tmp_path / "paw_cache_redact")
    backend = MockPAWBackend()

    @compile_on_hit(
        spec="Echo the secret",
        threshold=1,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
        redact_trace=True,
    )
    def handle(text: str) -> str:
        return f"teacher:{text}"

    result = handle("Authorization: Bearer sk-abcdefghijklmnop")
    # The actual function return value is untouched by redaction.
    assert "sk-abcdefghijklmnop" in result

    traces = handle.db.get_traces(handle.task_id)  # type: ignore[attr-defined]
    assert not any("sk-abcdefghijklmnop" in t["input_payload"] for t in traces)
    assert any("[REDACTED]" in t["input_payload"] for t in traces)


def test_background_compiler_bounded_retry_then_terminal_failed_PAW_JIT_03(tmp_path: Path) -> None:
    """Verify a persistently-failing compile retries a bounded number of times, then goes terminal.

    Regression for the audit's own suggested fix (unconditionally reset status to
    "tracing" on failure), which -- since call_count never decreases -- turns into an
    unbounded retry loop instead of a bounded one.
    """
    db_file = str(tmp_path / "retry_db.db")
    db = TraceDB(db_path=db_file)
    task_id = "always_fails"
    db.record_trace(task_id, "inp", "out", 1.0)

    class AlwaysFailingBackend(MockPAWBackend):
        def compile(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("simulated compilation failure")

    compiler = BackgroundCompiler()
    backend = AlwaysFailingBackend()
    out_path = str(tmp_path / "out.paw")

    for attempt in range(1, BackgroundCompiler._MAX_COMPILE_ATTEMPTS + 1):
        compiler.trigger_compilation(task_id, "spec", db, backend, out_path, sync=True)
        assert db.get_compile_attempts(task_id) == attempt
        if attempt < BackgroundCompiler._MAX_COMPILE_ATTEMPTS:
            assert db.get_status(task_id) == "tracing", "must remain retryable before the cap"
        else:
            assert db.get_status(task_id) == "failed", "must go terminal once the cap is reached"

    # Past exhaustion, even a fresh async trigger must be refused -- not retried forever.
    attempts_before = db.get_compile_attempts(task_id)
    t = compiler.trigger_compilation(task_id, "spec", db, backend, out_path, sync=False)
    assert t is None
    assert db.get_compile_attempts(task_id) == attempts_before
    assert db.get_status(task_id) == "failed"
    db.close()


# --- PAW-JIT-04: BEGIN IMMEDIATE + bounded retry/backoff on OperationalError -------


def test_with_write_retry_retries_transient_operational_error_then_succeeds_PAW_JIT_04(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _with_write_retry retries a transient sqlite3.OperationalError with
    backoff, then returns successfully once the underlying operation stops failing."""
    import paw_kit.jit.db as db_module

    db = TraceDB(db_path=str(tmp_path / "retry.db"))
    sleeps: List[float] = []
    monkeypatch.setattr(db_module.time, "sleep", lambda s: sleeps.append(s))

    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "ok"

    result = db._with_write_retry(flaky)
    assert result == "ok"
    assert attempts["n"] == 3
    assert len(sleeps) == 2
    assert sleeps == sorted(sleeps)  # backoff increases, not constant/decreasing
    db.close()


def test_with_write_retry_gives_up_after_max_attempts_PAW_JIT_04(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify _with_write_retry re-raises once the retry budget is exhausted, rather
    than retrying forever."""
    import paw_kit.jit.db as db_module

    db = TraceDB(db_path=str(tmp_path / "retry_exhaust.db"))
    monkeypatch.setattr(db_module.time, "sleep", lambda s: None)

    call_count = {"n": 0}

    def always_fails() -> None:
        call_count["n"] += 1
        raise sqlite3.OperationalError("database is locked")

    with pytest.raises(sqlite3.OperationalError):
        db._with_write_retry(always_fails)
    assert call_count["n"] == db_module._DB_RETRY_ATTEMPTS
    db.close()


def test_trace_db_uses_begin_immediate_isolation_PAW_JIT_04(tmp_path: Path) -> None:
    """Verify the connection is opened with isolation_level="IMMEDIATE"."""
    db = TraceDB(db_path=str(tmp_path / "immediate.db"))
    assert db._conn.isolation_level == "IMMEDIATE"
    db.close()


def test_trace_db_survives_genuine_multiprocess_write_contention_PAW_JIT_04(tmp_path: Path) -> None:
    """Verify multiple separate OS processes (not threads sharing one connection)
    writing to the same TraceDB concurrently all succeed with no unhandled
    sqlite3.OperationalError -- BEGIN IMMEDIATE + retry/backoff is specifically about
    multi-process contention, which threads sharing a single connection object cannot
    exercise (see test_trace_db_concurrent_writes for that, separate, case)."""
    db_file = str(tmp_path / "mp_traces.db")
    task_id = "mp_task"
    num_procs = 4
    calls_per_proc = 15

    procs = [
        multiprocessing.Process(
            target=_mp_record_trace_worker, args=(db_file, task_id, calls_per_proc)
        )
        for _ in range(num_procs)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, "a worker process crashed (likely an unhandled OperationalError)"

    db = TraceDB(db_path=db_file)
    assert db.get_call_count(task_id) == num_procs * calls_per_proc
    db.close()


# --- PAW-JIT-05: cache the loaded adapter callable, with fail-open-safe invalidation


def test_compile_on_hit_cached_adapter_callable_still_fails_open_PAW_JIT_05(tmp_path: Path) -> None:
    """Verify a cached adapter callable that starts failing still falls open to the
    teacher, not a stale success or an uncaught exception -- the cached call stays
    inside the existing fail-open try/except, unlike the audit's own suggested patch
    which placed it outside."""
    cache_dir = str(tmp_path / "paw_cache_jit05_failopen")
    backend = MockPAWBackend()
    teacher_calls = 0

    @compile_on_hit(
        spec="JIT-05 fail-open test",
        threshold=1,
        response_model=SentimentOutput,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def svc(text: str) -> SentimentOutput:
        nonlocal teacher_calls
        teacher_calls += 1
        return SentimentOutput(sentiment="fallback", confidence=1.0)

    svc("x")  # compiles
    assert svc.is_compiled()  # type: ignore[attr-defined]
    svc("x")  # populates the adapter-callable cache
    assert teacher_calls == 1

    adapter_path = svc.db.get_adapter_path(svc.task_id)  # type: ignore[attr-defined]
    backend.set_default_response(adapter_path, "MALFORMED_OUTPUT_CAUSING_PARSE_ERROR")

    result = svc("y")  # cache hit on the load()'d callable, but its output now fails
    assert teacher_calls == 2  # fell open to the teacher, not an uncaught exception
    assert result.sentiment == "fallback"


def test_compile_on_hit_deleted_adapter_falls_back_to_teacher_PAW_JIT_05(tmp_path: Path) -> None:
    """Verify a paw-clean-style mid-run deletion of the adapter file falls back to the
    teacher on the next call rather than serving the cached callable."""
    cache_dir = str(tmp_path / "paw_cache_jit05_deleted")
    backend = MockPAWBackend()
    teacher_calls = 0

    @compile_on_hit(
        spec="JIT-05 deleted-adapter test",
        threshold=1,
        response_model=SentimentOutput,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def svc(text: str) -> SentimentOutput:
        nonlocal teacher_calls
        teacher_calls += 1
        return SentimentOutput(sentiment="fallback", confidence=1.0)

    svc("x")  # compiles
    svc("x")  # populates the adapter-callable cache
    assert teacher_calls == 1

    adapter_path = svc.db.get_adapter_path(svc.task_id)  # type: ignore[attr-defined]
    os.remove(adapter_path)

    result = svc("y")
    assert teacher_calls == 2  # fell back, did not serve a cached callable for a gone file
    assert result.sentiment == "fallback"


def test_compile_on_hit_sync_recompile_of_ready_task_bypasses_stale_cache_PAW_JIT_05(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """White-box regression (Phase 0 Round 3): a sync=True recompile of an
    already-"ready" task must cause `load` to be re-invoked on the next call, not
    reuse the callable cached before the recompile. A black-box comparison of return
    values would pass vacuously here: load()'s returned closure holds no adapter
    content and calls backend.infer(...) fresh on every invocation regardless of
    whether it was just rebuilt, so it must instead spy on `load` itself."""
    import paw_kit.jit.decorator as decorator_module

    cache_dir = str(tmp_path / "paw_cache_jit05_stale")
    backend = MockPAWBackend()

    @compile_on_hit(
        spec="JIT-05 stale-recompile test",
        threshold=1,
        response_model=SentimentOutput,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
    )
    def svc(text: str) -> SentimentOutput:
        return SentimentOutput(sentiment="teacher", confidence=1.0)

    svc("x")  # compiles (status -> ready)
    svc("x")  # populates the adapter-callable cache

    load_call_count = {"n": 0}
    real_load = decorator_module.load

    def spy_load(*args: object, **kwargs: object):
        load_call_count["n"] += 1
        return real_load(*args, **kwargs)

    monkeypatch.setattr(decorator_module, "load", spy_load)

    svc("x")  # cache hit expected: load must NOT be called again
    assert load_call_count["n"] == 0

    # Recompile the already-"ready" task via the public API surface, exactly as
    # BackgroundCompiler.trigger_compilation(sync=True) allows (compiler.py's
    # status-guard is skipped entirely when sync=True).
    out_path = svc.db.get_adapter_path(svc.task_id)  # type: ignore[attr-defined]
    decorator_module._GLOBAL_COMPILER.trigger_compilation(
        task_id=svc.task_id,  # type: ignore[attr-defined]
        spec="JIT-05 stale-recompile test",
        db=svc.db,  # type: ignore[attr-defined]
        backend=backend,
        output_path=out_path,
        sync=True,
    )

    svc("x")  # the recompile changed the file's stat identity -- must reload
    assert load_call_count["n"] == 1


def test_adapter_cache_invalidation_hook_fires_on_any_status_write_PAW_JIT_05(tmp_path: Path) -> None:
    """Unit-level check that the same-process invalidation hook (the backstop named
    in the Success criteria for a filesystem where inode isn't a reliable staleness
    signal) is actually wired up and fires on any set_status call for a task_id, not
    only ones where the stat-identity component would also have caught it."""
    import paw_kit.jit.decorator as decorator_module

    cache_dir = str(tmp_path / "paw_cache_jit05_hook")
    backend = MockPAWBackend()

    @compile_on_hit(
        spec="JIT-05 hook test", threshold=1, cache_dir=cache_dir, backend=backend, sync_compile=True
    )
    def svc(text: str) -> str:
        return f"teacher:{text}"

    task_id = svc.task_id  # type: ignore[attr-defined]
    # Seed a cache entry as if a prior call had populated it.
    decorator_module._ADAPTER_CALLABLE_CACHE[task_id] = {"sentinel": (lambda x: x)}
    assert task_id in decorator_module._ADAPTER_CALLABLE_CACHE

    svc.db.set_status(task_id, "tracing")  # type: ignore[attr-defined]
    assert task_id not in decorator_module._ADAPTER_CALLABLE_CACHE


def test_compile_on_hit_cached_callable_keyed_on_resolved_backend_PAW_JIT_05(tmp_path: Path) -> None:
    """Verify a mid-run set_default_backend swap does not keep routing to a callable
    cached against the previous backend -- the cache key includes the resolved
    backend, closing the one staleness vector the stat-identity key doesn't cover
    (decorator.py re-resolves `backend or get_default_backend()` on every call)."""
    from paw_kit.schema.loader import get_default_backend, set_default_backend

    cache_dir = str(tmp_path / "paw_cache_jit05_backend_key")
    original_backend = MockPAWBackend()
    old_default = get_default_backend()
    set_default_backend(original_backend)
    try:

        @compile_on_hit(
            spec="JIT-05 backend-key test",
            threshold=1,
            response_model=SentimentOutput,
            cache_dir=cache_dir,
            backend=None,  # resolved per-call via get_default_backend()
            sync_compile=True,
        )
        def svc(text: str) -> SentimentOutput:
            return SentimentOutput(sentiment="orig", confidence=1.0)

        svc("x")  # compiles against original_backend
        svc("x")  # populates the cache keyed on original_backend

        adapter_path = svc.db.get_adapter_path(svc.task_id)  # type: ignore[attr-defined]
        second_backend = MockPAWBackend()
        # A rule only the second backend knows about -- original_backend's in-memory
        # adapter state (from compiling) has no such rule, so if the cache wrongly
        # reused a callable bound to original_backend, this input would fail
        # validation and fall open to the teacher ("orig"), not return "second".
        second_backend.register_rule(adapter_path, "distinguish", '{"sentiment": "second", "confidence": 1.0}')

        set_default_backend(second_backend)
        result = svc("distinguish")
        assert result.sentiment == "second"
    finally:
        set_default_backend(old_default)


def test_compile_on_hit_task_id_is_full_sha256_hash_PAW_JIT_06(tmp_path: Path) -> None:
    """Verify task_id is the full 64 hex characters, not a 16-char truncation."""
    cache_dir = str(tmp_path / "paw_cache_jit06")

    @compile_on_hit(spec="JIT-06 full hash test", cache_dir=cache_dir, backend=MockPAWBackend())
    def svc(text: str) -> str:
        return text

    task_id = svc.task_id  # type: ignore[attr-defined]
    assert len(task_id) == 64
    int(task_id, 16)  # raises ValueError if not valid hex


def test_atomic_write_text_uses_replace_and_leaves_no_temp_file_PAW_JIT_05(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify atomic_write_text writes via a temp file + os.replace, and leaves no
    leftover temp file behind on success."""
    from paw_kit.atomicio import atomic_write_text
    import paw_kit.atomicio as atomicio_module

    real_replace = os.replace
    replace_calls = []

    def spy_replace(src: object, dst: object) -> None:
        replace_calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(atomicio_module.os, "replace", spy_replace)

    target = tmp_path / "out.txt"
    atomic_write_text(target, "hello world")

    assert replace_calls
    assert target.read_text(encoding="utf-8") == "hello world"
    assert [p for p in tmp_path.iterdir() if p.name != "out.txt"] == []


def test_atomic_write_text_cleans_up_temp_file_on_failure_PAW_JIT_05(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a failure during the replace step leaves neither a partially-written
    target nor an orphaned temp file behind."""
    from paw_kit.atomicio import atomic_write_text
    import paw_kit.atomicio as atomicio_module

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomicio_module.os, "replace", boom)

    target = tmp_path / "out.txt"
    with pytest.raises(OSError):
        atomic_write_text(target, "hello world")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_gives_recompiled_adapter_a_fresh_inode_PAW_JIT_05(tmp_path: Path) -> None:
    """Verify the property PAW-JIT-05's cache key depends on: a recompile over the
    same deterministic output path allocates a fresh inode every time, so an
    os.stat-identity-keyed cache can never miss detecting it, including across
    processes."""
    from paw_kit.atomicio import atomic_write_text

    target = tmp_path / "adapter.paw"
    atomic_write_text(target, "v1")
    first_inode = target.stat().st_ino

    atomic_write_text(target, "v2")
    second_inode = target.stat().st_ino

    assert first_inode != second_inode
    assert target.read_text(encoding="utf-8") == "v2"


def test_mock_and_real_backend_compile_route_through_atomic_write_PAW_JIT_05() -> None:
    """Structural check that both compile() implementations that write an adapter
    file actually call the shared atomic_write_text helper, not a bare
    open()/Path.write_text()."""
    import inspect

    from paw_kit.backend.mock import MockPAWBackend
    from paw_kit.backend.real import RealPAWBackend

    assert "atomic_write_text" in inspect.getsource(MockPAWBackend.compile)
    assert "atomic_write_text" in inspect.getsource(RealPAWBackend.compile)
