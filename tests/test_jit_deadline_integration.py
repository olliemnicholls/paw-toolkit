"""End-to-end J-2 integration tests: the deadline pools wired into `wrapper` and
`ShadowRunner._run_job`. `tests/test_jit_deadline.py` covers `DeadlinePool` itself
directly; this file covers the two call sites and the C2/C4 verdict coupling.
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from paw_kit import MockPAWBackend, compile_on_hit
import paw_kit.jit.shadow as shadow_module


class BlockingBackend(MockPAWBackend):
    """A backend whose `infer` blocks on an `Event` -- simulates a wedged adapter."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: Optional[threading.Event] = None
        self.infer_calls = 0

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
        self.infer_calls += 1
        if self.gate is not None:
            self.gate.wait(30.0)
        return f"adapter:{input_text}"


def _make(tmp_path: Path, name: str, backend: MockPAWBackend, **kwargs: Any) -> Tuple[Any, Dict[str, int]]:
    calls = {"n": 0}
    spec = f"J-2 integration ({name})"
    cache_dir = str(tmp_path / f"cache_{name}")

    def teacher(text: str) -> str:
        calls["n"] += 1
        return f"teacher:{text}"

    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
        shadow_queue_size=64,
    )
    params.update(kwargs)
    return compile_on_hit(**params)(teacher), calls


def _pairs(wrapper: Any) -> List[Dict[str, Any]]:
    with wrapper.db._lock:
        rows = wrapper.db._conn.execute(
            "SELECT * FROM shadow_pairs WHERE task_id = ? ORDER BY id ASC;", (wrapper.task_id,)
        ).fetchall()
    return [dict(r) for r in rows]


# --- Served path (J-2's original success criterion) --------------------------------


def test_served_path_wedged_adapter_fails_open_within_deadline(tmp_path: Path) -> None:
    """A `ready`-state adapter call that hangs must fail open, not block forever."""
    backend = BlockingBackend()
    svc, calls = _make(tmp_path, "served-wedge", backend, shadow_window=0, adapter_timeout_s=0.3)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "ready"

    backend.gate = threading.Event()  # never set: every further infer() call wedges
    assert svc.get_fail_open_count() == 0

    started = time.perf_counter()
    result = svc("hello")
    elapsed = time.perf_counter() - started

    assert result == "teacher:hello", "a wedged adapter must fail open to the teacher's answer"
    assert calls["n"] == 3  # 2 seed calls + this fail-open's teacher call
    assert svc.get_fail_open_count() == 1
    assert elapsed < 5.0, f"caller waited {elapsed:.2f}s on a 0.3s deadline"

    backend.gate.set()  # release the background worker so it doesn't linger


def test_served_path_promptly_answering_adapter_is_unaffected(tmp_path: Path) -> None:
    """The deadline mechanism must not change behaviour on the non-faulted path."""
    backend = BlockingBackend()
    svc, calls = _make(tmp_path, "served-fast", backend, shadow_window=0, adapter_timeout_s=5.0)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "ready"

    result = svc("world")
    assert result == "adapter:world"
    assert svc.get_fail_open_count() == 0


# --- Shadow path: J-2's deadline, C2/C4's verdict-category coupling ----------------


def test_shadow_path_own_wedge_past_deadline_scores_error(tmp_path: Path) -> None:
    """This task's own adapter exceeding its deadline (slot was free) is `error` --
    it reflects on this adapter, exactly like any other adapter exception."""
    backend = BlockingBackend()
    svc, calls = _make(tmp_path, "shadow-ownwedge", backend, shadow_window=2, adapter_timeout_s=0.1)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    backend.gate = threading.Event()
    svc("compare-me")

    assert shadow_module._GLOBAL_SHADOW_RUNNER.drain(
        svc.task_id, 10.0, db_path=str(svc.db.db_path)
    )
    pairs = _pairs(svc)
    assert pairs, "expected a shadow comparison to have been recorded"
    assert pairs[-1]["verdict"] == "error"
    assert pairs[-1]["error_type"] == "DeadlineExceeded"
    # Not a fail-open: the caller was always served by the teacher in `shadow`.
    assert svc.get_fail_open_count() == 0

    backend.gate.set()


def test_shadow_path_pool_exhaustion_from_other_tasks_scores_excluded_verdict(
    tmp_path: Path,
) -> None:
    """A shadow-pool timeout caused by a DIFFERENT task's wedged comparisons is
    infrastructure, not this adapter's drift -- excluded, not `error` (C2/C4
    coupling: this track introduces the excluded category here, in C2)."""
    pool = shadow_module._SHADOW_DEADLINE_POOL
    held = 0
    while pool._free.acquire(blocking=False):
        held += 1
    assert held == pool._max_workers, "test assumes it can observe the pool's full capacity"

    try:
        backend = MockPAWBackend()
        svc, calls = _make(tmp_path, "shadow-poolexhaust", backend, shadow_window=2)
        svc("seed0")
        svc("seed1")
        assert svc.db.get_status(svc.task_id) == "shadow"

        svc("compare-me")
        assert shadow_module._GLOBAL_SHADOW_RUNNER.drain(
            svc.task_id, 10.0, db_path=str(svc.db.db_path)
        )
        pairs = _pairs(svc)
        assert pairs, "expected a shadow comparison to have been recorded"
        assert pairs[-1]["verdict"] == "pool_exhausted"
        assert pairs[-1]["error_type"] == "PoolExhausted"

        # Excluded from the window: samples must not count it, in either direction.
        stats = svc.db.get_agreement_stats(svc.task_id, svc.db.get_task_routing(svc.task_id)[2], 2, "shadow")
        assert stats["error"] == 0
        assert stats["samples"] == 0

        # Not a fail-open either: the caller was served by the teacher.
        assert svc.get_fail_open_count() == 0
    finally:
        for _ in range(held):
            pool._free.release()
