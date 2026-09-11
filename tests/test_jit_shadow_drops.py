"""M-4b (C4b): per-phase drop counting. Before this fix, shadow, audit and
fail_open drops were all conflated in one `_dropped` bucket (`submit_fail_open`
routes through the same `submit()`), so only audit-phase drops are the ones that
delay demotion of a drifting adapter -- and that number had no name of its own.
No change to drop behavior or queue sizing (Track G's Phase F already measured
that tradeoff); this is purely an accounting/observability split.

Every test below synchronizes on a `started` event before submitting further
jobs, so "the first job is dequeued and in flight, the queue is empty again" is
a fact, not a race -- the exact drop counts asserted follow deterministically
from single-threaded, in-order submission after that point.
"""

import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pytest

from paw_kit import MockPAWBackend, compile_on_hit
import paw_kit.jit.decorator as decorator_module
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER


class GatedBackend(MockPAWBackend):
    """Blocks the FIRST infer() call on an Event (signalling `started` first), so
    the shadow-phase worker stays occupied while further jobs queue up."""

    def __init__(self) -> None:
        super().__init__()
        self.gate: Optional[threading.Event] = None
        self.started = threading.Event()

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
        if self.gate is not None:
            self.started.set()
            self.gate.wait(30.0)
        return f"adapter:{input_text}"


class GatedTeacher:
    """A callable teacher that blocks (after signalling `started`) once gated --
    used for the audit phase, where the worker's own call is `run_teacher()`, not
    the backend's `infer()`."""

    def __init__(self) -> None:
        self.gate: Optional[threading.Event] = None
        self.started = threading.Event()
        self.calls = 0
        # compile_on_hit's task-id derivation reads these off `func` directly.
        self.__name__ = "gated_teacher"
        self.__qualname__ = "gated_teacher"

    def __call__(self, text: str) -> str:
        self.calls += 1
        if self.gate is not None:
            self.started.set()
            self.gate.wait(30.0)
        return f"teacher:{text}"


def _make(tmp_path: Path, name: str, backend: MockPAWBackend, teacher: Callable[[str], str], **kwargs: Any) -> Any:
    spec = f"M-4b drop accounting ({name})"
    cache_dir = str(tmp_path / f"cache_{name}")
    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
    )
    params.update(kwargs)
    return compile_on_hit(**params)(teacher)


def _promote(svc: Any, backend: MockPAWBackend) -> None:
    """Drive `svc` from `shadow` to `ready` with a one-comparison window: register
    a rule so the (plain, non-Scripted) MockPAWBackend's answer for "promote-me"
    matches the teacher's, then submit it and wait for the promotion to land."""
    adapter_path = svc.db.get_task_routing(svc.task_id)[1]
    backend.register_rule(adapter_path or "", "promote-me", "teacher:promote-me")
    svc("promote-me")
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))
    assert svc.db.get_status(svc.task_id) == "ready", "setup failed: task did not promote"


def test_stats_reports_dropped_shadow_audit_fail_open_separately(tmp_path: Path) -> None:
    """The three drop categories must be independently countable, and the
    pre-existing total ('dropped') must still equal their sum -- backward
    compatible with every test that already asserts on it."""
    backend = GatedBackend()
    teacher = GatedTeacher()
    svc = _make(tmp_path, "drop-split", backend, teacher, shadow_window=20, shadow_queue_size=1)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    backend.gate = threading.Event()
    svc("shadow-a")  # dequeued by the worker; occupies it (wedged on backend.gate)
    assert backend.started.wait(5.0), "worker never reached infer() for the occupying job"

    svc("shadow-b")  # queue capacity 1: fits, pending
    for i in range(4):
        svc(f"shadow-drop-{i}")  # queue already full: dropped
    _GLOBAL_SHADOW_RUNNER.submit_fail_open(svc.task_id, str(svc.db.db_path), svc.db, queue_size=1)
    _GLOBAL_SHADOW_RUNNER.submit_fail_open(svc.task_id, str(svc.db.db_path), svc.db, queue_size=1)

    stats = _GLOBAL_SHADOW_RUNNER.stats(svc.task_id, str(svc.db.db_path))
    assert stats["dropped_shadow"] == 4
    assert stats["dropped_fail_open"] == 2
    assert stats["dropped_audit"] == 0
    assert stats["dropped"] == stats["dropped_shadow"] + stats["dropped_audit"] + stats["dropped_fail_open"]

    backend.gate.set()
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))


def test_audit_phase_drops_counted_separately_from_shadow_drops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The number M-4b actually cares about: audit-phase drops specifically,
    since only they delay demotion of a drifting adapter. Gates the TEACHER
    (what an audit job's worker call actually invokes), not the backend -- the
    served path itself must stay fast."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = MockPAWBackend()
    teacher = GatedTeacher()
    svc = _make(
        tmp_path, "audit-drop-split", backend, teacher, shadow_window=1, shadow_threshold=1.0,
        shadow_queue_size=32, audit_window=4, audit_rate=0.5, demote_threshold=0.6,
    )
    svc("seed0")
    svc("seed1")
    _promote(svc, backend)

    teacher.gate = threading.Event()
    svc("audit-occupy")  # served instantly (adapter path); its audit job wedges the worker
    assert teacher.started.wait(5.0), "worker never reached run_teacher() for the occupying job"

    svc("audit-pending")  # fits in the (size-32, but now irrelevant) queue
    for i in range(4):
        svc(f"audit-drop-{i}")

    stats = _GLOBAL_SHADOW_RUNNER.stats(svc.task_id, str(svc.db.db_path))
    assert stats["dropped_audit"] == 0, (
        "queue_size=32 should not have overflowed from only 6 submissions -- "
        "this assertion documents that assumption; see the next test for an "
        "actually-saturated queue"
    )

    teacher.gate.set()
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))


def test_audit_phase_drops_under_a_saturated_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same setup, but with a queue small enough to actually saturate."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = MockPAWBackend()
    teacher = GatedTeacher()
    svc = _make(
        tmp_path, "audit-drop-saturate", backend, teacher, shadow_window=1, shadow_threshold=1.0,
        shadow_queue_size=1, audit_window=4, audit_rate=0.5, demote_threshold=0.6,
    )
    svc("seed0")
    svc("seed1")
    _promote(svc, backend)

    teacher.gate = threading.Event()
    svc("audit-occupy")
    assert teacher.started.wait(5.0)

    svc("audit-pending")  # fits: queue capacity 1
    for i in range(4):
        svc(f"audit-drop-{i}")  # queue full: dropped, deterministically, 4 of them

    stats = _GLOBAL_SHADOW_RUNNER.stats(svc.task_id, str(svc.db.db_path))
    assert stats["dropped_audit"] == 4
    assert stats["dropped_shadow"] == 0
    assert stats["dropped_fail_open"] == 0

    teacher.gate.set()
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))


def test_get_agreement_surfaces_per_phase_drops(tmp_path: Path) -> None:
    """wrapper.get_agreement() (in-process only, per M-4b's decided scope) must
    surface the per-phase breakdown, not just the pre-existing total."""
    backend = GatedBackend()
    teacher = GatedTeacher()
    svc = _make(tmp_path, "get-agreement-drops", backend, teacher, shadow_window=20, shadow_queue_size=1)
    svc("seed0")
    svc("seed1")

    backend.gate = threading.Event()
    svc("occupy")
    assert backend.started.wait(5.0)
    svc("pending")
    svc("drop-me")

    agreement = svc.get_agreement()
    assert "dropped_shadow" in agreement
    assert "dropped_audit" in agreement
    assert "dropped_fail_open" in agreement
    assert agreement["dropped"] == (
        agreement["dropped_shadow"] + agreement["dropped_audit"] + agreement["dropped_fail_open"]
    )
    assert agreement["dropped_shadow"] == 1

    backend.gate.set()
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))


def test_stall_subsampled_untouched_by_the_drop_split() -> None:
    """The pre-existing `stall_subsampled` counter (a different concept entirely
    -- comparisons this process personally skipped under the stall guard, not
    drops) must not regress, and must not be conflated with any of the three
    drop keys."""
    stats = _GLOBAL_SHADOW_RUNNER.stats("no-such-task-xyz")
    assert stats["stalled"] == 0
    assert set(stats.keys()) >= {
        "pending", "dropped", "dropped_shadow", "dropped_audit", "dropped_fail_open", "stalled",
    }


def test_audit_drop_warning_names_audit_phase_and_shadow_queue_size(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A WARNING on the first audit-phase drop must name it specifically and point
    at raising shadow_queue_size -- distinct from the generic wording a
    shadow-phase drop gets."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = MockPAWBackend()
    teacher = GatedTeacher()
    svc = _make(
        tmp_path, "audit-drop-warn", backend, teacher, shadow_window=1, shadow_threshold=1.0,
        shadow_queue_size=1, audit_window=4, audit_rate=0.5, demote_threshold=0.6,
    )
    svc("seed0")
    svc("seed1")
    _promote(svc, backend)

    teacher.gate = threading.Event()
    svc("audit-occupy")
    assert teacher.started.wait(5.0)
    svc("audit-pending")

    with caplog.at_level(logging.WARNING, logger="paw_kit.jit.shadow"):
        svc("audit-drop-trigger")

    audit_warnings = [
        r.message for r in caplog.records
        if "audit" in r.message.lower() and "shadow_queue_size" in r.message
    ]
    assert audit_warnings, f"no audit-specific drop warning found in: {[r.message for r in caplog.records]}"

    teacher.gate.set()
    assert _GLOBAL_SHADOW_RUNNER.drain(svc.task_id, 10.0, db_path=str(svc.db.db_path))
