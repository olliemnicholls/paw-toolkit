"""Fail-open fault-injection tests for @compile_on_hit (bug-hunt Track C).

J-1: a trace-DB fault in any state other than `ready` must degrade to "call the
teacher, return its answer, log once" -- never raise into the caller, never silently
skip the counter/log signal. Three independent fault boundaries in
`decorator.py`'s `wrapper` are exercised here, each with a real (not merely mocked)
fault of the kind the bug-hunt report observed:

- **PRAGMA query_only** (a real read-only-database OperationalError) at step 4,
  `record_trace`.
- **A closed connection** (a real `sqlite3.ProgrammingError`) at step 1, the routing
  read.
- **An injected `sqlite3.DatabaseError`** (mirroring the report's "database disk
  image is malformed" observation, which is impractical to reproduce by actually
  corrupting a file) at step 6, the compile-trigger's fresh status re-read.

Each is exercised once with the task in `tracing` state (no adapter yet) and once in
`shadow` state (compiled, not yet promoted) -- six tests -- plus two tests pinning
the two Phase-0-mandated properties of the routing-read boundary specifically: it
must itself increment the fail-open counter (a bare default-and-continue would be a
*new* silent fail-open), and a faulted read's default must not be able to reach
step 5's shadow-job submission.
"""

from pathlib import Path
import sqlite3
from typing import Any, Callable, Dict, List, Tuple

import pytest

from paw_kit import MockPAWBackend, TraceDB, compile_on_hit
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER


def _make(
    tmp_path: Path, name: str, **kwargs: Any
) -> Tuple[Any, Dict[str, int]]:
    """Decorate a teacher with a per-test spec, so task_ids never collide across tests."""
    calls = {"n": 0}
    spec = f"J-1 fault-injection ({name})"
    cache_dir = str(tmp_path / f"cache_{name}")

    def teacher(text: str) -> str:
        calls["n"] += 1
        return f"teacher:{text}"

    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=MockPAWBackend(),
        sync_compile=True, shadow_window=20, shadow_queue_size=64,
    )
    params.update(kwargs)
    return compile_on_hit(**params)(teacher), calls


def _to_shadow_state(tmp_path: Path, name: str, **kwargs: Any) -> Tuple[Any, Dict[str, int]]:
    """Drive a freshly decorated task from `tracing` to `shadow` (threshold=2)."""
    svc, calls = _make(tmp_path, name, **kwargs)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow", "setup failed: task did not compile"
    return svc, calls


def _pairs(wrapper: Any) -> List[Dict[str, Any]]:
    with wrapper.db._lock:
        rows = wrapper.db._conn.execute(
            "SELECT * FROM shadow_pairs WHERE task_id = ? ORDER BY id ASC;", (wrapper.task_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def _raise_once(original: Callable[..., Any], exc: BaseException) -> Callable[..., Any]:
    """Wrap a bound method so its FIRST call raises `exc`; later calls run `original`."""
    state = {"n": 0}

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            raise exc
        return original(*args, **kwargs)

    return _wrapped


# --- Fault kind 1: closed connection, at the routing read (step 1) -----------------


def test_j1_closed_connection_at_routing_read_falls_open_in_tracing_state(
    tmp_path: Path,
) -> None:
    svc, calls = _make(tmp_path, "closed-tracing")
    svc.db._conn.close()  # real sqlite3.ProgrammingError on every further operation

    result = svc("hello")

    assert result == "teacher:hello"
    assert calls["n"] == 1
    assert svc.get_fail_open_count() >= 1


def test_j1_closed_connection_at_routing_read_falls_open_in_shadow_state(
    tmp_path: Path,
) -> None:
    svc, calls = _to_shadow_state(tmp_path, "closed-shadow")
    calls_before = calls["n"]
    svc.db._conn.close()  # real sqlite3.ProgrammingError on every further operation

    result = svc("world")

    assert result == "teacher:world"
    assert calls["n"] == calls_before + 1
    assert svc.get_fail_open_count() >= 1


# --- Fault kind 2: PRAGMA query_only, at record_trace (step 4) ---------------------


def test_j1_readonly_db_at_record_trace_falls_open_in_tracing_state(tmp_path: Path) -> None:
    svc, calls = _make(tmp_path, "readonly-tracing", threshold=100)
    svc.db._conn.execute("PRAGMA query_only = ON;")  # real write-fault: OperationalError

    result = svc("hello")

    assert result == "teacher:hello"
    assert calls["n"] == 1
    assert svc.get_fail_open_count() == 1
    # The teacher's own answer must still come back on a second faulted call.
    result2 = svc("again")
    assert result2 == "teacher:again"
    assert svc.get_fail_open_count() == 2


def test_j1_readonly_db_at_record_trace_falls_open_in_shadow_state(tmp_path: Path) -> None:
    svc, calls = _to_shadow_state(tmp_path, "readonly-shadow", threshold=2)
    calls_before = calls["n"]
    pairs_before = len(_pairs(svc))
    svc.db._conn.execute("PRAGMA query_only = ON;")

    result = svc("more")

    assert result == "teacher:more"
    assert calls["n"] == calls_before + 1
    assert svc.get_fail_open_count() == 1
    # Step 5 (shadow-job submission) depends only on step 1's status, not on
    # record_trace succeeding -- a fault here must not suppress it.
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 5.0, db_path=str(svc.db.db_path))
    assert len(_pairs(svc)) == pairs_before + 1, (
        "a record_trace fault must not also suppress step 5's shadow-job submission "
        "as a side effect of sharing a try block with it"
    )


# --- Fault kind 3: injected DatabaseError, at the compile trigger's status re-read (step 6) --


def test_j1_injected_database_error_at_compile_trigger_falls_open_in_tracing_state(
    tmp_path: Path,
) -> None:
    svc, calls = _make(tmp_path, "injected-tracing", threshold=1)
    original_get_status = svc.db.get_status
    svc.db.get_status = _raise_once(
        original_get_status, sqlite3.DatabaseError("database disk image is malformed")
    )

    # Call 1 reaches threshold=1: record_trace succeeds, but the fresh get_status()
    # re-read that would gate compilation faults.
    result = svc("first")
    assert result == "teacher:first"
    assert svc.get_fail_open_count() == 1
    # The fault must not have let compilation spuriously trigger nor corrupt state.
    assert original_get_status(svc.task_id) == "tracing"

    # Call 2: get_status is unpatched again (raise-once already fired) and the task
    # catches up -- compilation triggers exactly as it would have on call 1.
    result2 = svc("second")
    assert result2 == "teacher:second"
    assert svc.db.get_status(svc.task_id) == "shadow"


def test_j1_injected_database_error_at_compile_trigger_falls_open_in_shadow_state(
    tmp_path: Path,
) -> None:
    svc, calls = _to_shadow_state(tmp_path, "injected-shadow", threshold=2)
    calls_before = calls["n"]
    original_get_status = svc.db.get_status
    svc.db.get_status = _raise_once(
        original_get_status, sqlite3.DatabaseError("database disk image is malformed")
    )

    # Once compiled, every further teacher-branch call re-reads get_status() (call_count
    # keeps exceeding threshold), so this is reachable in `shadow` state too.
    result = svc("third")

    assert result == "teacher:third"
    assert calls["n"] == calls_before + 1
    assert svc.get_fail_open_count() == 1
    assert original_get_status(svc.task_id) == "shadow", "still correctly in shadow"


# --- Phase-0-mandated properties of the routing-read boundary specifically --------


def test_j1_routing_read_fault_increments_fail_open_counter(tmp_path: Path) -> None:
    """A bare default-and-continue on a step-1 fault would itself be a new, silent
    fail-open (S-14's exact shape); the routing-read boundary must route through the
    same counter/log-once signal every other fail-open path does."""
    svc, _ = _make(tmp_path, "routing-counter")
    original = svc.db.get_task_routing
    svc.db.get_task_routing = _raise_once(original, RuntimeError("routing read exploded"))

    assert svc.get_fail_open_count() == 0
    svc("input")
    assert svc.get_fail_open_count() == 1


def test_j1_routing_read_fault_cannot_reach_shadow_job_submission(tmp_path: Path) -> None:
    """A faulted routing read must default to something that cannot equal `"shadow"`,
    so it can never enqueue a comparison keyed to a wrong or default state_epoch."""
    svc, calls = _to_shadow_state(tmp_path, "routing-isolation", threshold=2)
    pairs_before = len(_pairs(svc))
    original = svc.db.get_task_routing
    svc.db.get_task_routing = _raise_once(original, RuntimeError("routing read exploded"))

    result = svc("isolated")

    assert result == "teacher:isolated"
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 5.0, db_path=str(svc.db.db_path))
    assert len(_pairs(svc)) == pairs_before, (
        "a routing-read fault defaulted to a status equal to 'shadow' (or otherwise "
        "let a faulted read reach step 5), enqueuing a comparison keyed to a wrong "
        "or default state_epoch"
    )
    # And the real, unfaulted state is untouched -- the very next call is shadowed again.
    result2 = svc("not-isolated")
    assert result2 == "teacher:not-isolated"
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 5.0, db_path=str(svc.db.db_path))
    assert len(_pairs(svc)) == pairs_before + 1
