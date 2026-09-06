"""Unit and integration tests for paw.jit: SQLite tracing, @compile_on_hit, and hot-swapping."""

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
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


def test_compile_on_hit_fail_open_safety(tmp_path: Path) -> None:
    """Verify Fail-Open safety: local adapter errors fall back transparently to teacher."""
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

    # Now intentionally break the compiled adapter to simulate runtime exception/corruption
    adapter_path = robust_service.db.get_adapter_path(robust_service.task_id)  # type: ignore[attr-defined]
    backend.set_default_response(adapter_path, "MALFORMED_OUTPUT_CAUSING_PARSE_ERROR")

    # Call 2: Local adapter throws/fails schema parsing -> must transparently fall back!
    res = robust_service("unknown input")
    assert teacher_invocations == 2  # Teacher was engaged as fail-open fallback!
    assert res.sentiment == "fallback_positive"


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
