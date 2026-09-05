"""Unit and integration tests for paw.jit: SQLite tracing, @compile_on_hit, and hot-swapping."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
