"""Shadow-mode tests for paw.jit (Track 14, Phase T).

Every test drives a `MockPAWBackend` subclass: no GPU, no network, no key.

`ScriptedBackend` below decides agreement per input rather than relying on
`MockPAWBackend`'s example-matching, because the default mock returns the traced
teacher output verbatim for an input it has seen -- i.e. it agrees perfectly, which
makes it useless for testing the case shadow mode exists for.
"""

import hashlib
import os
from pathlib import Path
import sqlite3
import stat
import subprocess
import sys
import textwrap
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel
import pytest

from paw_kit import MockPAWBackend, TraceDB, compile_on_hit
from paw_kit.jit.db import _SCHEMA_VERSION
import paw_kit.jit.decorator as decorator_module
import paw_kit.jit.shadow as shadow_module
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER

DISAGREE = "ADAPTER-DISAGREES"

# A module-level teacher, so two decorations in different tests (or before and after a
# simulated restart) produce the SAME task_id -- task_id is sha256(qualname:spec).
_MODULE_TEACHER_CALLS = {"n": 0}


def _module_teacher(text: str) -> str:
    _MODULE_TEACHER_CALLS["n"] += 1
    return f"teacher:{text}"


class Triage(BaseModel):
    priority: str
    urgency_score: int


class ScriptedBackend(MockPAWBackend):
    """A compiled adapter whose answer is scripted per input.

    `agree_when(text)` True -> returns exactly what the teacher would have returned.
    `fail` -> `infer` raises. `gate` -> `infer` blocks on the event first.
    """

    def __init__(self, agree_when: Optional[Callable[[str], bool]] = None) -> None:
        super().__init__()
        self.agree_when = agree_when or (lambda text: True)
        self.fail = False
        self.gate: Optional[threading.Event] = None
        self.infer_calls = 0

    def infer(
        self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None
    ) -> str:
        self.infer_calls += 1
        if self.gate is not None:
            self.gate.wait(2.0)
        if self.fail:
            raise RuntimeError("adapter exploded")
        return f"teacher:{input_text}" if self.agree_when(input_text) else DISAGREE


def _make(
    tmp_path: Path, name: str, backend: MockPAWBackend, **kwargs: Any
) -> Tuple[Any, Dict[str, int]]:
    """Decorate a teacher with a per-test spec (so task_ids never collide across tests)."""
    calls = {"n": 0}
    spec = f"Echo the input back ({name})"
    cache_dir = str(tmp_path / f"cache_{name}")

    def teacher(text: str) -> str:
        calls["n"] += 1
        return f"teacher:{text}"

    # `shadow_queue_size` deliberately overrides the production default of 8 for every
    # test that does not ask for something else. At the default, a test that submits more
    # than 8 comparisons faster than the worker drains them has the surplus *silently
    # dropped* -- and a dropped comparison never enters the window denominator, so the
    # agreement rate a test asserts on becomes a function of machine speed. That is what
    # made `test_demotion_boundary_from_both_sides` fail ~1 in 12 with rate 0.75 (6/8)
    # instead of 0.6 (6/10). Several tests here already passed an explicit 16 or 32 for
    # exactly this reason; this makes the defence uniform instead of per-site.
    #
    # Queue capacity itself is covered deliberately, and still is:
    # `test_shadow_queue_drops_when_full_and_never_blocks` sets `shadow_queue_size=1`.
    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
        shadow_queue_size=256,
    )
    params.update(kwargs)
    return compile_on_hit(**params)(teacher), calls


def _drain(wrapper: Any, timeout: float = 10.0, allow_drops: bool = False) -> None:
    """Wait for the task's shadow worker to finish, then assert nothing was dropped.

    The drop assertion is the point: a dropped comparison is not an error and is not
    logged above DEBUG after the first one, but it silently removes a sample from the
    window every later assertion is computed over. Without this check that shows up as
    an intermittent wrong *number*, not as a failure pointing at its cause. Tests that
    exercise queue saturation on purpose pass `allow_drops=True`.
    """
    assert _GLOBAL_SHADOW_RUNNER.drain(
        wrapper.task_id, timeout, db_path=str(wrapper.db.db_path)
    ), "shadow worker did not drain in time"
    if not allow_drops:
        dropped = _GLOBAL_SHADOW_RUNNER.stats(
            wrapper.task_id, str(wrapper.db.db_path)
        )["dropped"]
        assert dropped == 0, (
            f"{dropped} shadow comparison(s) were dropped on a full queue; every "
            "assertion about an agreement rate below this point is computed over a "
            "window that is missing samples. Raise shadow_queue_size for this test."
        )


def _query(db: TraceDB, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    with db._lock:
        return [dict(row) for row in db._conn.execute(sql, params).fetchall()]


def _pairs(wrapper: Any) -> List[Dict[str, Any]]:
    return _query(
        wrapper.db,
        "SELECT * FROM shadow_pairs WHERE task_id = ? ORDER BY id ASC;",
        (wrapper.task_id,),
    )


def _transitions(wrapper: Any) -> List[Dict[str, Any]]:
    return _query(
        wrapper.db,
        "SELECT * FROM state_transitions WHERE task_id = ? ORDER BY id ASC;",
        (wrapper.task_id,),
    )


_V1_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    call_count INTEGER DEFAULT 0,
    adapter_path TEXT,
    status TEXT DEFAULT 'tracing',
    compile_attempts INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    input_payload TEXT NOT NULL,
    teacher_output TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
);
CREATE INDEX IF NOT EXISTS idx_traces_task_id ON traces(task_id);
"""


def _build_v1_db(
    db_file: Path,
    task_id: str,
    status: str = "tracing",
    adapter_path: Optional[str] = None,
    call_count: int = 3,
) -> None:
    """Create a database with the pre-Track-14 schema, verbatim."""
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.executescript(_V1_SCHEMA)
    conn.execute(
        "INSERT INTO tasks (task_id, call_count, adapter_path, status, compile_attempts, "
        "created_at, updated_at) VALUES (?, ?, ?, ?, 0, '2026-01-01T00:00:00', "
        "'2026-01-01T00:00:00');",
        (task_id, call_count, adapter_path, status),
    )
    for i in range(call_count):
        conn.execute(
            "INSERT INTO traces (task_id, input_payload, teacher_output, latency_ms, timestamp) "
            "VALUES (?, ?, ?, 1.0, '2026-01-01T00:00:00');",
            (task_id, f"legacy-{i}", f"teacher:legacy-{i}"),
        )
    conn.commit()
    conn.close()


def _task_id_for(func: Callable[..., Any], spec: str) -> str:
    """Mirror of `compile_on_hit`'s derived task_id.

    J-8 widened the hash to include `co_filename` and `co_firstlineno` so that two
    distinct functions sharing a qualname no longer share a task_id (and therefore a
    call_count, a trace corpus, an adapter and a shadow window). This helper tracks
    that formula; it is not an independent check of it.
    """
    qualname = f"{func.__module__}.{func.__qualname__}"
    code = func.__code__
    return hashlib.sha256(
        "\x00".join([qualname, spec, code.co_filename, str(code.co_firstlineno)])
        .encode("utf-8")
    ).hexdigest()


# --- Lifecycle and equivalence ------------------------------------------------


def test_shadow_window_zero_reproduces_legacy_hotswap_exactly(tmp_path: Path) -> None:
    """shadow_window=0 is a true no-op: no shadow state, no rows, no worker thread."""
    backend = ScriptedBackend()
    svc, calls = _make(tmp_path, "legacy", backend, threshold=3, shadow_window=0)

    for i in range(3):
        svc(f"in{i}")
    assert svc.db.get_status(svc.task_id) == "ready"
    assert svc.is_compiled()
    assert calls["n"] == 3

    # Call 4 is served by the adapter, not the teacher.
    assert svc(f"in0") == "teacher:in0"
    assert calls["n"] == 3

    assert _pairs(svc) == []
    # Only the two compile transitions: tracing -> compiling -> ready.
    assert [(t["from_status"], t["to_status"]) for t in _transitions(svc)] == [
        ("tracing", "compiling"),
        ("compiling", "ready"),
    ]
    # And no worker thread was ever started for this task.
    prefix = f"paw-shadow-{svc.task_id[:12]}"
    assert not [t for t in threading.enumerate() if t.name.startswith(prefix)]


def test_default_config_enters_shadow_and_teacher_still_serves(tmp_path: Path) -> None:
    """On the shipped defaults the adapter does not serve on trust."""
    backend = ScriptedBackend()
    svc, calls = _make(tmp_path, "defaults", backend, threshold=3)

    for i in range(3):
        svc(f"in{i}")
    assert svc.db.get_status(svc.task_id) == "shadow"
    assert svc.is_compiled() is False, "is_compiled() means 'the adapter is serving'"
    assert svc.has_adapter() is True
    assert svc.shadow_config["shadow_window"] == 20

    for i in range(3, 6):
        assert svc(f"in{i}") == f"teacher:in{i}"
    assert calls["n"] == 6, "the teacher must still serve every call while in shadow"
    _drain(svc)
    assert len(_pairs(svc)) == 3


def test_promotion_happens_at_exactly_shadow_threshold(tmp_path: Path) -> None:
    """A window below the threshold does not promote; the next window at exactly it does."""
    below = {f"i{n}" for n in range(7)}
    at = {f"j{n}" for n in range(8)}
    backend = ScriptedBackend(agree_when=lambda text: text in below or text in at)
    svc, _ = _make(tmp_path, "boundary", backend, shadow_window=10, shadow_threshold=0.8)

    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    # Window 1: 7/10 agree -> 0.7 < 0.8.
    for n in range(10):
        svc(f"i{n}")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "shadow"

    # Window 2 (tumbling, a fresh draw): 8/10 -> exactly 0.8.
    for n in range(10):
        svc(f"j{n}")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"
    report = svc.db.get_task_report(svc.task_id)
    assert report["promoted_agreement"] == pytest.approx(0.8)


def test_promotion_requires_full_window_not_first_lucky_sample(tmp_path: Path) -> None:
    """One agreeing sample is rate 1.0 and must not promote. This is the bug being fixed."""
    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "lucky", backend, shadow_window=20)

    svc("a")
    svc("b")
    assert svc.db.get_status(svc.task_id) == "shadow"
    svc("c")
    _drain(svc)

    agreement = svc.get_agreement()
    assert agreement["rate"] == 1.0
    assert agreement["samples"] == 1
    assert svc.db.get_status(svc.task_id) == "shadow"


def _promote(tmp_path: Path, name: str, backend: ScriptedBackend, **kwargs: Any) -> Tuple[Any, Dict[str, int]]:
    """Drive a task from tracing to `ready` through a one-comparison shadow window."""
    params: Dict[str, Any] = dict(shadow_window=1, shadow_threshold=1.0)
    params.update(kwargs)
    svc, calls = _make(tmp_path, name, backend, **params)
    svc("seed0")
    svc("seed1")
    svc("promote-me")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready", "setup failed: task did not promote"
    return svc, calls


def test_demotion_on_audit_agreement_below_demote_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sustained audit disagreement after promotion returns the task to shadow."""
    backend = ScriptedBackend()
    svc, calls = _promote(
        tmp_path, "demote", backend, audit_window=10, audit_rate=0.5, demote_threshold=0.6
    )
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)

    backend.agree_when = lambda text: False  # the adapter drifts
    teacher_calls_before = calls["n"]
    for n in range(10):
        assert svc(f"audit{n}") == DISAGREE  # the adapter is still serving
    _drain(svc)

    assert svc.db.get_status(svc.task_id) == "shadow"
    assert calls["n"] > teacher_calls_before, "the audit must have re-run the teacher"
    audit_pairs = [p for p in _pairs(svc) if p["phase"] == "audit"]
    assert len(audit_pairs) == 10
    # And the teacher serves again.
    teacher_calls_before = calls["n"]
    assert svc("after-demotion") == "teacher:after-demotion"
    assert calls["n"] == teacher_calls_before + 1


def test_demotion_boundary_from_both_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The trigger is `rate < demote_threshold`: exactly at the threshold must NOT demote."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)

    # Window of 10 at exactly 0.6 -> stays ready.
    agreeing = {f"ok{n}" for n in range(6)}
    backend = ScriptedBackend(agree_when=lambda text: text in agreeing or text == "promote-me")
    svc, _ = _promote(
        tmp_path, "demote-at", backend, audit_window=10, audit_rate=0.5, demote_threshold=0.6
    )
    for n in range(6):
        svc(f"ok{n}")
    for n in range(4):
        svc(f"bad{n}")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"
    assert svc.get_agreement()["rate"] == pytest.approx(0.6)

    # A window one sample below the threshold -> demotes.
    agreeing2 = {f"ok{n}" for n in range(5)}
    backend2 = ScriptedBackend(agree_when=lambda text: text in agreeing2 or text == "promote-me")
    svc2, _ = _promote(
        tmp_path, "demote-below", backend2, audit_window=10, audit_rate=0.5, demote_threshold=0.6
    )
    for n in range(5):
        svc2(f"ok{n}")
    for n in range(5):
        svc2(f"bad{n}")
    _drain(svc2)
    assert svc2.db.get_status(svc2.task_id) == "shadow"


def test_promotion_and_demotion_record_timestamp_and_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both transitions persist the timestamp, agreement and sample count."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = ScriptedBackend()
    svc, _ = _promote(
        tmp_path, "records", backend, audit_window=4, audit_rate=0.5, demote_threshold=0.6
    )

    report = svc.db.get_task_report(svc.task_id)
    assert report["promoted_at"] is not None
    assert report["promoted_agreement"] == pytest.approx(1.0)
    promotion = [t for t in _transitions(svc) if (t["from_status"], t["to_status"]) == ("shadow", "ready")]
    assert len(promotion) == 1
    assert promotion[0]["agreement"] == pytest.approx(1.0)
    assert promotion[0]["sample_count"] == 1
    assert promotion[0]["timestamp"]

    backend.agree_when = lambda text: False
    for n in range(4):
        svc(f"drift{n}")
    _drain(svc)

    report = svc.db.get_task_report(svc.task_id)
    assert report["demoted_at"] is not None
    assert report["demoted_agreement"] == pytest.approx(0.0)
    demotion = [t for t in _transitions(svc) if (t["from_status"], t["to_status"]) == ("ready", "shadow")]
    assert len(demotion) == 1
    assert demotion[0]["sample_count"] == 4
    assert demotion[0]["timestamp"]


def test_state_epoch_increments_on_every_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Epoch strictly increases across compiling -> shadow -> ready -> shadow."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = ScriptedBackend()
    svc, _ = _promote(
        tmp_path, "epochs", backend, audit_window=2, audit_rate=0.5, demote_threshold=0.6
    )
    backend.agree_when = lambda text: False
    svc("d0")
    svc("d1")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "shadow"

    epochs = [t["state_epoch"] for t in _transitions(svc)]
    assert epochs == sorted(epochs)
    assert len(set(epochs)) == len(epochs), f"epochs must strictly increase, got {epochs}"


def test_mid_flight_promotion_does_not_let_stale_pairs_demote(tmp_path: Path) -> None:
    """Pairs written at an older epoch persist as history but never move the new window."""
    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "midflight", backend, shadow_window=20)
    svc("a")
    svc("b")
    for n in range(5):
        svc(f"c{n}")
    _drain(svc)

    old_epoch = svc.db.get_task_routing(svc.task_id)[2]
    assert len(_pairs(svc)) == 5
    assert svc.db.try_promote(svc.task_id, old_epoch, 1.0, 5) is True

    new_epoch = svc.db.get_task_routing(svc.task_id)[2]
    assert new_epoch == old_epoch + 1
    # The rows are still there (they are audit history) ...
    assert len(_pairs(svc)) == 5
    # ... but the window that decides the next transition is empty.
    assert svc.db.get_agreement_stats(svc.task_id, new_epoch, 20, "shadow")["samples"] == 0
    assert svc.get_agreement()["samples"] == 0


def test_failed_state_is_terminal_and_never_enters_shadow(tmp_path: Path) -> None:
    """Three compile failures go terminal; no adapter, no shadow, no pairs."""

    class AlwaysFailingBackend(MockPAWBackend):
        def compile(self, *args: object, **kwargs: object) -> str:
            raise RuntimeError("simulated compilation failure")

    svc, _ = _make(tmp_path, "failed", AlwaysFailingBackend(), threshold=1)
    for i in range(4):
        svc(f"in{i}")

    assert svc.db.get_status(svc.task_id) == "failed"
    assert svc.db.get_adapter_path(svc.task_id) is None
    assert svc.has_adapter() is False
    assert _pairs(svc) == []


# --- Safety and concurrency ---------------------------------------------------


def test_shadow_worker_thread_is_a_daemon(tmp_path: Path) -> None:
    """The per-task worker thread must be `daemon=True`, checked directly.

    Not testing this any other way (letting the test process exit and checking
    it doesn't hang) is not just slower -- it does not terminate. If this thread
    were ever created non-daemon, nothing pulls it off `queue.Queue.get()`'s
    blocking wait once the test suite is otherwise done, and the interpreter
    waits forever to join it at exit: a mutation of this one flag is not merely
    slow to detect from the outside, it is undetectable within any bounded
    timeout that way. Asserting the attribute directly kills it in milliseconds.
    """
    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "daemoncheck", backend, shadow_window=1)
    svc("a")
    svc("b")
    svc("c")  # past `threshold=2`: an adapter exists now, so this call is shadowed
    _drain(svc)

    key = (str(svc.db.db_path), svc.task_id)
    thread = _GLOBAL_SHADOW_RUNNER._threads[key]
    assert thread.daemon is True, (
        "shadow worker thread must be a daemon thread, or a decorated caller's "
        "process can never exit while shadow mode is active"
    )


def test_shadow_worker_exception_does_not_propagate_and_counts_as_disagreement(
    tmp_path: Path,
) -> None:
    """An adapter that throws is recorded as an error, never raised, never a fail-open."""
    backend = ScriptedBackend()
    svc, calls = _make(tmp_path, "workererr", backend, shadow_window=2)
    svc("a")
    svc("b")
    backend.fail = True

    assert svc("c") == "teacher:c"
    assert svc("d") == "teacher:d"
    _drain(svc)

    pairs = _pairs(svc)
    assert len(pairs) == 2
    assert {p["verdict"] for p in pairs} == {"error"}
    assert {p["error_type"] for p in pairs} == {"RuntimeError"}
    assert all(p["adapter_output"] is None for p in pairs)
    assert svc.db.get_status(svc.task_id) == "shadow", "an erroring adapter must not promote"
    # Shadow errors are NOT fail-opens: the caller was always getting the teacher.
    assert svc.get_fail_open_count() == 0
    assert svc.db.get_task_report(svc.task_id)["fail_open_count"] == 0


def test_shadow_does_not_add_caller_latency(tmp_path: Path) -> None:
    """A 2s adapter must not be on the caller's request path. Upper-bound assertion only."""
    backend = ScriptedBackend()
    gate = threading.Event()
    svc, _ = _make(tmp_path, "latency", backend, shadow_window=20, shadow_queue_size=16)
    svc("a")
    svc("b")
    backend.gate = gate

    started = time.perf_counter()
    for n in range(5):
        time.sleep(0.01)  # a fast "teacher"
        svc(f"slow{n}")
    elapsed = time.perf_counter() - started
    assert elapsed < 0.5, f"caller waited {elapsed:.2f}s on a 2s adapter"

    gate.set()  # release the worker so the suite does not pay for it at exit
    _drain(svc)


def test_shadow_queue_drops_when_full_and_never_blocks(tmp_path: Path) -> None:
    """A saturated queue drops the newest job; drops never enter the window denominator."""
    backend = ScriptedBackend()
    gate = threading.Event()
    svc, _ = _make(tmp_path, "queuefull", backend, shadow_window=20, shadow_queue_size=1)
    svc("a")
    svc("b")
    backend.gate = gate

    started = time.perf_counter()
    for n in range(50):
        svc(f"burst{n}")
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"the caller blocked on a full shadow queue ({elapsed:.2f}s)"

    stats = _GLOBAL_SHADOW_RUNNER.stats(svc.task_id, str(svc.db.db_path))
    assert stats["dropped"] > 0

    gate.set()
    _drain(svc, allow_drops=True)  # dropping is this test's subject
    agreement = svc.get_agreement()
    assert agreement["samples"] + stats["dropped"] <= 50
    assert agreement["samples"] < 50, "dropped jobs must not appear in the denominator"


def test_agreement_fn_exception_counts_as_disagreement(tmp_path: Path) -> None:
    """A caller agreement_fn that raises is an error verdict, and nothing propagates."""

    def exploding(teacher: Any, adapter: Any) -> bool:
        raise ValueError("bad agreement_fn")

    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "badfn", backend, shadow_window=2, agreement_fn=exploding)
    svc("a")
    svc("b")
    assert svc("c") == "teacher:c"
    assert svc("d") == "teacher:d"
    _drain(svc)

    pairs = _pairs(svc)
    assert len(pairs) == 2
    assert {p["verdict"] for p in pairs} == {"error"}
    assert {p["error_type"] for p in pairs} == {"ValueError"}
    assert svc.db.get_status(svc.task_id) == "shadow"


def test_custom_agreement_fn_is_used(tmp_path: Path) -> None:
    """An always-True agreement_fn promotes a wholly-wrong adapter: the hook is wired."""
    backend = ScriptedBackend(agree_when=lambda text: False)
    svc, _ = _make(
        tmp_path, "customfn", backend,
        shadow_window=2, shadow_threshold=1.0, agreement_fn=lambda t, a: True,
    )
    svc("a")
    svc("b")
    svc("c")
    svc("d")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"


def test_audit_sampling_uses_local_rng_not_global_random(tmp_path: Path) -> None:
    """A caller who seeds `random` for reproducibility must not be perturbed."""
    import random

    random.seed(1234)
    expected = [random.random() for _ in range(4)]

    backend = ScriptedBackend()
    svc, _ = _promote(tmp_path, "rng", backend, audit_rate=0.5, audit_window=20)

    random.seed(1234)
    for n in range(20):
        svc(f"served{n}")
    observed = [random.random() for _ in range(4)]
    assert observed == expected
    _drain(svc)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shadow_window": -1},
        {"shadow_threshold": 0.0},
        {"shadow_threshold": 1.5},
        {"audit_window": 0},
        {"audit_rate": -0.1},
        {"audit_rate": 0.9},
        {"demote_threshold": 0.9, "shadow_threshold": 0.8},
        {"demote_threshold": 0.8, "shadow_threshold": 0.8},
        {"demote_threshold": -0.1},
        {"shadow_queue_size": 0},
        {"shadow_max_pairs": 5, "shadow_window": 20},
        {"shadow_max_pairs": 5, "audit_window": 20, "shadow_window": 5},
    ],
)
def test_decoration_time_validation_rejects_every_invalid_parameter(
    tmp_path: Path, kwargs: Dict[str, Any]
) -> None:
    """Every rule raises ValueError at decoration time, never at call time."""
    with pytest.raises(ValueError):
        _make(tmp_path, "validate", ScriptedBackend(), **kwargs)


# --- The headline criterion ---------------------------------------------------


def test_sixty_percent_agreement_adapter_never_promotes_over_many_windows(
    tmp_path: Path,
) -> None:
    """The repo's own measured adapter (60% agreement) must never serve production traffic.

    Driven for well over twenty full windows. A *sliding* window would give it a fresh
    draw on every comparison and promote it eventually; a tumbling one gives it one
    draw per window, and a repeating input set it agrees on exactly 60% of produces a
    rate of exactly 0.60 every time.
    """
    inputs = [f"ticket{n}" for n in range(5)]
    agreeing = set(inputs[:3])  # 3 of 5 -> exactly 60%
    backend = ScriptedBackend(agree_when=lambda text: text in agreeing)
    svc, _ = _make(
        tmp_path, "sixtypercent", backend,
        shadow_window=5, shadow_threshold=0.8, shadow_queue_size=32,
    )
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    epoch = svc.db.get_task_routing(svc.task_id)[2]
    n = 0
    stall_point = shadow_module._SHADOW_STALL_FACTOR * 5
    while svc.db.get_epoch_seq(svc.task_id, epoch) < stall_point:
        svc(inputs[n % 5])
        n += 1
        _drain(svc)
    # Every comparison so far was exhaustive, so the measured rate is exactly 60%.
    assert svc.get_agreement()["rate"] == pytest.approx(0.6)
    assert svc.db.get_status(svc.task_id) == "shadow"
    # Exactly at the stall point the last full window is still evaluated (see
    # `_maybe_transition`'s `seq > factor * window`, strict) -- not stalled yet.
    assert svc.get_agreement()["stalled"] is False

    while svc.db.get_epoch_seq(svc.task_id, epoch) < 100 and n < 3000:
        for _ in range(20):
            svc(inputs[n % 5])
            n += 1
        _drain(svc)

    seq = svc.db.get_epoch_seq(svc.task_id, epoch)
    assert seq >= 100, f"expected at least 20 full windows of 5, got seq={seq}"
    assert svc.db.get_status(svc.task_id) == "shadow"
    assert svc.is_compiled() is False
    # Past the stall point the runner is subsampling, so a *sampled* window can land
    # above the threshold by chance -- and must still not promote. That is why a
    # stalled task's windows are recorded for visibility but never evaluated.
    assert svc.db.get_adapter_path(svc.task_id) is None
    # Finding 2: both `get_agreement()` and `get_task_report()` now say so.
    assert svc.get_agreement()["stalled"] is True
    assert svc.db.get_task_report(svc.task_id)["agreement"]["stalled"] is True


def test_tumbling_window_does_not_promote_on_a_pattern_a_sliding_window_would(
    tmp_path: Path,
) -> None:
    """Finding 3: a deterministic discriminator for the tumbling-vs-sliding rule,
    independent of the stall guard.

    `shadow_window=5`, ten comparisons scripted D,D,A,A,A | A,A,D,D,D (D=disagree,
    A=agree). Tumbling windows measure 3/5=0.6 then 2/5=0.4 -- both below the 0.8
    threshold, so a tumbling window never promotes. A *sliding* 5-window ending at
    comparison 7 (comparisons 3-7) would see A,A,A,A,A = 5/5 = 1.0 and promote. Only
    10 comparisons run, well under the `_SHADOW_STALL_FACTOR * shadow_window == 25`
    stall point, so the stall guard cannot be what is discriminating here.
    """
    sequence = [False, False, True, True, True, True, True, False, False, False]
    inputs = [f"c{i}" for i in range(len(sequence))]
    outcomes = dict(zip(inputs, sequence))
    backend = ScriptedBackend(agree_when=lambda text: outcomes[text])
    svc, _ = _make(tmp_path, "c1regression", backend, shadow_window=5, shadow_threshold=0.8)
    svc("seed0")
    svc("seed1")
    assert svc.db.get_status(svc.task_id) == "shadow"

    for text in inputs:
        svc(text)
    _drain(svc)

    epoch = svc.db.get_task_routing(svc.task_id)[2]
    seq = svc.db.get_epoch_seq(svc.task_id, epoch)
    assert seq == 10
    assert seq < shadow_module._SHADOW_STALL_FACTOR * 5, (
        "must discriminate via the tumbling-window rule, not the stall guard"
    )
    assert svc.get_agreement()["rate"] == pytest.approx(0.4), "trailing (second) window"
    assert svc.db.get_status(svc.task_id) == "shadow"
    assert svc.is_compiled() is False


def test_stall_guard_throttles_a_non_converging_task(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Past `_SHADOW_STALL_FACTOR * shadow_window` comparisons the rate drops, warning once."""
    backend = ScriptedBackend(agree_when=lambda text: False)
    svc, _ = _make(tmp_path, "stall", backend, shadow_window=2, shadow_queue_size=32)
    svc("seed0")
    svc("seed1")
    epoch = svc.db.get_task_routing(svc.task_id)[2]

    stall_point = shadow_module._SHADOW_STALL_FACTOR * 2
    with caplog.at_level("WARNING", logger="paw_kit.jit.shadow"):
        for n in range(stall_point):
            svc(f"pre{n}")
        _drain(svc)
        assert svc.db.get_epoch_seq(svc.task_id, epoch) == stall_point
        assert not [r for r in caplog.records if "not converging" in r.message]

        for n in range(20):
            svc(f"post{n}")
        _drain(svc)

    seq_after = svc.db.get_epoch_seq(svc.task_id, epoch)
    assert seq_after < stall_point + 20, "the stall guard did not throttle"
    assert seq_after > stall_point, "throttling must sample, not stop"
    warnings = [r for r in caplog.records if "not converging" in r.message]
    assert len(warnings) == 1, "the stall warning must fire exactly once per epoch"

    stats = _GLOBAL_SHADOW_RUNNER.stats(svc.task_id, str(svc.db.db_path))
    assert stats["stalled"] > 0
    assert stats["dropped"] == 0, "stall-guard skips must not be reported as queue drops"


# --- Persistence, migration and retention -------------------------------------


def test_migration_from_v1_schema_adds_shadow_tables_and_columns(tmp_path: Path) -> None:
    """A pre-Track-14 traces.db opens, keeps its rows, and gains the v2 surface."""
    db_file = tmp_path / "v1" / "traces.db"
    _build_v1_db(db_file, "legacy-task", status="tracing", call_count=3)

    db = TraceDB(db_path=str(db_file))
    assert db.get_call_count("legacy-task") == 3
    assert len(db.get_traces("legacy-task")) == 3
    assert db.get_status("legacy-task") == "tracing"
    assert db.get_task_routing("legacy-task") == ("tracing", None, 0)

    tables = {
        row["name"]
        for row in _query(db, "SELECT name FROM sqlite_master WHERE type = 'table';")
    }
    assert {"tasks", "traces", "shadow_pairs", "state_transitions"} <= tables
    # J-5 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    # this asserted the literal `2`. J-5 adds a `tasks.compiling_started_at` column, so
    # `_SCHEMA_VERSION` moves 2 -> 3 and the literal had to change with it. Pinned
    # against the module constant rather than a new literal: the subject here is "the
    # forward marker was stamped by this migration", not its numeric value, and the
    # value itself is pinned separately (with an explanation of what a bump means) by
    # `test_migration_adds_the_lease_column_and_bumps_the_marker_J_5` in
    # tests/test_jit_persistence.py.
    assert _query(db, "PRAGMA user_version;")[0]["user_version"] == _SCHEMA_VERSION

    report = db.get_task_report("legacy-task")
    assert report["state_epoch"] == 0
    assert report["call_count"] == 3
    assert report["agreement"]["phase"] is None
    db.close()


def test_migration_is_idempotent_on_second_open(tmp_path: Path) -> None:
    """An unguarded ALTER TABLE ADD COLUMN on an existing column raises; this must not."""
    db_file = tmp_path / "v1b" / "traces.db"
    _build_v1_db(db_file, "legacy-task")

    first = TraceDB(db_path=str(db_file))
    first.close()
    second = TraceDB(db_path=str(db_file))
    second.record_trace("legacy-task", "again", "teacher:again", 1.0)
    assert second.get_call_count("legacy-task") == 4
    # J-5 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    # this asserted the literal `2`. J-5 adds a `tasks.compiling_started_at` column, so
    # `_SCHEMA_VERSION` moves 2 -> 3 and the literal had to change with it. Pinned
    # against the module constant rather than a new literal: the subject here is "the
    # forward marker was stamped by this migration", not its numeric value, and the
    # value itself is pinned separately (with an explanation of what a bump means) by
    # `test_migration_adds_the_lease_column_and_bumps_the_marker_J_5` in
    # tests/test_jit_persistence.py.
    assert _query(second, "PRAGMA user_version;")[0]["user_version"] == _SCHEMA_VERSION
    second.close()


def test_existing_ready_task_keeps_serving_after_upgrade(tmp_path: Path) -> None:
    """Shadow mode gates *new* compiles, not adapters that were already promoted."""
    cache_dir = tmp_path / "upgraded"
    spec = "Echo the input back (upgrade)"
    backend = ScriptedBackend()
    calls = {"n": 0}

    def teacher(text: str) -> str:
        calls["n"] += 1
        return f"teacher:{text}"

    task_id = _task_id_for(teacher, spec)
    adapter_path = backend.compile(
        spec=spec,
        examples=[{"input": "legacy-0", "output": "teacher:legacy-0"}],
        output_path=str(cache_dir / f"{task_id}.paw"),
    )
    _build_v1_db(cache_dir / "traces.db", task_id, status="ready", adapter_path=adapter_path)

    svc = compile_on_hit(spec=spec, threshold=2, cache_dir=str(cache_dir), backend=backend)(teacher)
    assert svc.task_id == task_id
    assert svc("still-served") == "teacher:still-served"
    assert calls["n"] == 0, "an already-promoted adapter must keep serving after the upgrade"
    assert svc.db.get_status(task_id) == "ready"


def test_shadow_window_zero_promotes_a_task_already_in_shadow(tmp_path: Path) -> None:
    """The escape hatch must rescue the users who tried the default first."""
    spec = "Echo the input back (rescue)"
    cache_dir = str(tmp_path / "cache_rescue")
    backend = ScriptedBackend(agree_when=lambda text: False)
    _MODULE_TEACHER_CALLS["n"] = 0

    stuck = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend,
        sync_compile=True, shadow_window=20,
    )(_module_teacher)
    stuck("a")
    stuck("b")
    assert stuck.db.get_status(stuck.task_id) == "shadow"
    stuck.db.close()

    # A restart, with the escape hatch set.
    rescued = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend,
        sync_compile=True, shadow_window=0,
    )(_module_teacher)
    assert rescued.task_id == stuck.task_id
    assert rescued.db.get_status(rescued.task_id) == "shadow"

    before = _MODULE_TEACHER_CALLS["n"]
    assert rescued("anything") == DISAGREE, "the adapter must be serving after the rescue"
    assert _MODULE_TEACHER_CALLS["n"] == before
    assert rescued.db.get_status(rescued.task_id) == "ready"
    assert rescued.is_compiled() is True
    assert "shadow_disabled" in [t["reason"] for t in _transitions(rescued)]


def test_shadow_pairs_pruned_to_cap_oldest_first(tmp_path: Path) -> None:
    """Retention is bounded and oldest-first."""
    db = TraceDB(db_path=str(tmp_path / "prune" / "traces.db"))
    for n in range(40):
        db.record_shadow_pair(
            task_id="t", state_epoch=0, phase="shadow", input_payload=f"in{n}",
            teacher_output="t", adapter_output="a", verdict="agree", max_pairs=10,
        )
    rows = _query(db, "SELECT id, input_payload FROM shadow_pairs ORDER BY id ASC;")
    assert len(rows) <= 10
    assert [r["input_payload"] for r in rows] == [f"in{n}" for n in range(30, 40)]
    db.close()


def test_served_calls_and_state_transitions_are_capped(tmp_path: Path) -> None:
    """`state_transitions` is bounded per task too, not just `shadow_pairs`.

    (`served_calls` itself was cut from this track by the Phase 0 ruling on Md-7; the
    persisted fail-open counter replaces the one figure `paw-kit report` needed.)
    """
    import paw_kit.jit.db as db_module

    db = TraceDB(db_path=str(tmp_path / "cap" / "traces.db"))
    total = db_module._STATE_TRANSITIONS_MAX_ROWS + db_module._PRUNE_EVERY * 3
    for n in range(total):
        db.record_state_transition("t", "shadow", "shadow", None, None, n, reason="churn")
    count = _query(db, "SELECT COUNT(*) AS n FROM state_transitions;")[0]["n"]
    assert count <= db_module._STATE_TRANSITIONS_MAX_ROWS + db_module._PRUNE_EVERY
    db.close()


def test_ready_state_persists_no_input_text_on_the_served_path(tmp_path: Path) -> None:
    """After promotion, a served call writes no raw production input anywhere.

    The full input/teacher/adapter triple is persisted for every shadow comparison and
    for the sampled audit fraction -- but not for an ordinary served call, which has no
    teacher answer to pair with and would only re-open the PAW-JIT-01/02 trade for rows
    of no compile value.
    """
    secret = "SERVED-ONLY-PAYLOAD-9f3a"
    backend = ScriptedBackend()
    svc, _ = _promote(tmp_path, "servedpath", backend, audit_rate=0.0)
    for _ in range(5):
        svc(secret)

    _drain(svc)
    for table in ("traces", "shadow_pairs", "state_transitions"):
        rows = _query(svc.db, f"SELECT * FROM {table};")
        blob = " ".join(str(value) for row in rows for value in row.values())
        assert secret not in blob, f"served-path input text leaked into {table}"


def test_fail_open_in_ready_records_the_fallback_and_does_not_demote(tmp_path: Path) -> None:
    """A broken adapter after promotion falls open, is counted, and does not demote."""
    backend = ScriptedBackend()
    svc, calls = _promote(
        tmp_path, "failopen", backend, audit_window=2, audit_rate=0.5, demote_threshold=0.6
    )
    backend.fail = True

    before = calls["n"]
    for n in range(3):
        assert svc(f"broken{n}") == f"teacher:broken{n}"
    assert calls["n"] == before + 3
    # In-process counter: a plain dict increment on the caller thread, unaffected by
    # finding 1's fix.
    assert svc.get_fail_open_count() == 3
    # The *persisted* counter (finding 1) is now applied on the shadow worker thread,
    # not the caller's -- drain before reading it back.
    _drain(svc)
    assert svc.db.get_task_report(svc.task_id)["fail_open_count"] == 3
    # Infrastructure faults are not semantic drift: no audit pair, no demotion.
    assert svc.db.get_status(svc.task_id) == "ready"
    assert [p for p in _pairs(svc) if p["phase"] == "audit"] == []


def test_fail_open_persisted_increment_never_runs_on_the_caller_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: `db.increment_fail_open` is a synchronous SQLite write and must run
    on the shadow worker thread, never on the caller's request path."""
    backend = ScriptedBackend()
    svc, _ = _promote(tmp_path, "failopenthread", backend, audit_window=5)
    backend.fail = True

    caller_thread = threading.current_thread()
    seen_threads: List[threading.Thread] = []
    original = TraceDB.increment_fail_open

    def spy(self: TraceDB, task_id: str) -> None:
        seen_threads.append(threading.current_thread())
        original(self, task_id)

    monkeypatch.setattr(TraceDB, "increment_fail_open", spy)

    assert svc("broken0") == "teacher:broken0"
    assert svc("broken1") == "teacher:broken1"
    _drain(svc)

    assert len(seen_threads) == 2, "increment_fail_open was not called the expected number of times"
    assert all(t is not caller_thread for t in seen_threads), (
        "the persisted fail-open increment ran on the caller thread"
    )
    assert svc.db.get_task_report(svc.task_id)["fail_open_count"] == 2


def test_fail_open_at_shadow_window_zero_never_persists_a_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding 1: shadow_window=0 is byte-for-byte the pre-Track-14 fail-open path --
    no shadow worker exists for the task at all, so no persisted counter is written,
    only the in-process one."""
    backend = ScriptedBackend()
    svc, calls = _make(tmp_path, "failopenzero", backend, threshold=3, shadow_window=0)
    for i in range(3):
        svc(f"in{i}")
    assert svc.db.get_status(svc.task_id) == "ready"
    backend.fail = True

    called: List[str] = []
    monkeypatch.setattr(
        TraceDB, "increment_fail_open", lambda self, task_id: called.append(task_id)
    )

    assert svc("broken0") == "teacher:broken0"
    assert svc.get_fail_open_count() == 1
    assert called == [], "increment_fail_open must never be called at shadow_window=0"
    assert svc.db.get_task_report(svc.task_id)["fail_open_count"] == 0


def test_redact_trace_applies_to_shadow_pair_columns(tmp_path: Path) -> None:
    """redact_trace=True scrubs all three text columns -- and the comparison ran raw."""
    token = "sk-abcdefghijklmnop"
    secret_input = f"Authorization: Bearer {token}"
    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "redact", backend, shadow_window=20, redact_trace=True)
    svc("seed0")
    svc("seed1")
    svc(secret_input)
    _drain(svc)

    pairs = _pairs(svc)
    assert len(pairs) == 1
    row = pairs[0]
    for column in ("input_payload", "teacher_output", "adapter_output"):
        assert token not in (row[column] or ""), f"{column} was persisted unredacted"
        assert "[REDACTED]" in (row[column] or "")
    # The comparison itself ran on the RAW values: redacting one side before the
    # comparison and the other after it would make every such call a guaranteed
    # disagreement.
    assert row["verdict"] == "agree"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits don't apply on Windows")
def test_shadow_writes_create_no_new_file_and_keep_0600(tmp_path: Path) -> None:
    """Everything lands in the existing 0600 traces.db inside the 0700 cache directory."""
    backend = ScriptedBackend()
    svc, _ = _make(tmp_path, "perms", backend, shadow_window=20)
    svc("a")
    svc("b")
    svc("c")
    _drain(svc)

    cache_dir = svc.db.db_path.parent
    assert stat.S_IMODE(os.stat(cache_dir).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(svc.db.db_path).st_mode) == 0o600
    names = {p.name for p in cache_dir.iterdir()}
    unexpected = {
        n for n in names
        if not (n.startswith("traces.db") or n.endswith(".paw") or n.endswith(".history.jsonl"))
    }
    assert not unexpected, f"shadow mode created new files: {unexpected}"


def test_traces_table_untouched_by_shadow_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compile corpus counts exactly the teacher-served calls, and nothing else."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = ScriptedBackend()
    svc, _ = _make(
        tmp_path, "corpus", backend,
        threshold=2, shadow_window=2, shadow_threshold=1.0, audit_window=2, audit_rate=0.5,
    )
    svc("t0")  # traced
    svc("t1")  # traced, compiles -> shadow
    svc("t2")  # traced + shadow comparison
    svc("t3")  # traced + shadow comparison -> promotes
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"

    svc("s0")  # served by the adapter, audited
    svc("s1")  # served by the adapter, audited
    _drain(svc)

    traces = svc.db.get_traces(svc.task_id)
    assert len(traces) == 4, "only teacher-served calls may enter the compile corpus"
    assert {t["input_payload"] for t in traces} == {"t0", "t1", "t2", "t3"}
    assert len(_pairs(svc)) == 4  # 2 shadow + 2 audit


def test_teacher_error_on_audit_is_excluded_from_numerator_and_denominator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising audit teacher is not the adapter's fault, and must not stall demotion."""
    monkeypatch.setattr(decorator_module, "_should_audit", lambda rng, rate: True)
    backend = ScriptedBackend()
    state = {"raise": False}
    calls = {"n": 0}
    spec = "Echo the input back (teachererr)"
    cache_dir = str(tmp_path / "cache_teachererr")

    def teacher(text: str) -> str:
        calls["n"] += 1
        if state["raise"]:
            raise RuntimeError("teacher outage")
        return f"teacher:{text}"

    svc = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
        shadow_window=1, shadow_threshold=1.0, audit_window=4, audit_rate=0.5,
        demote_threshold=0.6,
    )(teacher)
    svc("seed0")
    svc("seed1")
    svc("promote-me")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"

    # Two audits whose teacher raises: recorded, but they move nothing.
    state["raise"] = True
    svc("outage0")
    svc("outage1")
    _drain(svc)
    teacher_errors = [p for p in _pairs(svc) if p["verdict"] == "teacher_error"]
    assert len(teacher_errors) == 2
    assert all(p["teacher_output"] is None for p in teacher_errors)
    assert all(p["seq"] == 0 for p in teacher_errors)
    assert svc.db.get_status(svc.task_id) == "ready"
    # Finding 5: the *reported* `teacher_error` is scoped to the current trailing
    # window, not the whole epoch -- with zero countable audit comparisons so far this
    # epoch, no window has started yet, so it reads 0 even though two teacher-error
    # rows already exist on disk (checked above via the raw table). Before the fix
    # this grew unbounded next to a `window`-sized rate; it no longer does.
    assert svc.get_agreement()["teacher_error"] == 0

    # The window still reaches audit_window and can still demote.
    state["raise"] = False
    backend.agree_when = lambda text: False
    for n in range(4):
        svc(f"drift{n}")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "shadow"


def test_tuple_answer_serializes_identically_on_trace_path_and_in_comparison(
    tmp_path: Path,
) -> None:
    """Finding 9: one shared serializer. A teacher/adapter returning a tuple -- the
    case only `agreement._stringify` (now `stringify_answer`) used to handle -- must
    serialize to the same string on decorator.py's trace-path persistence and inside
    shadow.py's comparison, since both now call the same function."""
    from paw_kit.jit.agreement import stringify_answer

    class TupleBackend(ScriptedBackend):
        def infer(
            self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None
        ) -> Any:
            self.infer_calls += 1
            return (input_text, 1)

    backend = TupleBackend()
    calls = {"n": 0}
    spec = "Echo the input back (tupleserialize)"
    cache_dir = str(tmp_path / "cache_tupleserialize")

    def teacher(text: str) -> Any:
        calls["n"] += 1
        return (text, 1)

    svc = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend, sync_compile=True,
        shadow_window=20,
    )(teacher)
    svc("seed0")
    svc("seed1")
    svc("tuple-in")
    _drain(svc)

    expected = stringify_answer(("tuple-in", 1))

    trace_row = _query(
        svc.db,
        "SELECT teacher_output FROM traces WHERE task_id = ? ORDER BY id DESC LIMIT 1;",
        (svc.task_id,),
    )[0]
    pair_row = _pairs(svc)[-1]
    assert trace_row["teacher_output"] == expected, "decorator.py's trace-path serialization"
    assert pair_row["teacher_output"] == expected, "the same trace-path string, carried through"
    assert pair_row["adapter_output"] == expected, "shadow.py's comparison-path serialization"
    assert pair_row["verdict"] == "agree"


def test_free_text_teacher_without_response_model_does_not_promote_by_default(
    tmp_path: Path,
) -> None:
    """With no response_model, agreement is exact string equality of free-form text."""

    class DriftingBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            # A realistic paraphrase: same meaning, different bytes.
            return f"Sure! Here is the answer for {input_text}."

    svc, _ = _make(
        tmp_path, "freetext", DriftingBackend(), shadow_window=2, shadow_threshold=0.8
    )
    svc("a")
    svc("b")
    for n in range(6):
        svc(f"q{n}")
    _drain(svc)

    assert svc.db.get_status(svc.task_id) == "shadow"
    agreement = svc.get_agreement()
    assert agreement["rate"] == 0.0
    assert agreement["disagree"] > 0


# --- Concurrency, keying and process exit -------------------------------------


def test_try_promote_is_compare_and_set_single_winner(tmp_path: Path) -> None:
    """Exactly one of two racing writers wins; the epoch advances by exactly one."""
    db = TraceDB(db_path=str(tmp_path / "cas" / "traces.db"))
    db.record_trace("t", "in", "out", 1.0)
    db.set_shadow_started("t", "/tmp/adapter.paw")
    epoch = db.get_task_routing("t")[2]

    results: List[bool] = []
    barrier = threading.Barrier(2)

    def attempt() -> None:
        barrier.wait()
        results.append(db.try_promote("t", epoch, 0.9, 20))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert sorted(results) == [False, True]
    assert db.get_task_routing("t") == ("ready", "/tmp/adapter.paw", epoch + 1)
    promotions = _query(
        db, "SELECT * FROM state_transitions WHERE to_status = 'ready' AND from_status = 'shadow';"
    )
    assert len(promotions) == 1

    # Symmetric for demotion.
    ready_epoch = db.get_task_routing("t")[2]
    results.clear()
    barrier2 = threading.Barrier(2)

    def attempt_demote() -> None:
        barrier2.wait()
        results.append(db.try_demote("t", ready_epoch, 0.1, 20))

    threads = [threading.Thread(target=attempt_demote) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert sorted(results) == [False, True]
    assert db.get_task_routing("t")[2] == ready_epoch + 1
    db.close()


def test_two_cache_dirs_same_spec_do_not_share_a_shadow_worker(tmp_path: Path) -> None:
    """task_id does not include cache_dir, so the runner must be keyed on (db_path, task_id)."""
    spec = "Echo the input back (shared-task-id)"
    backend_a = ScriptedBackend()
    backend_b = ScriptedBackend()

    def teacher(text: str) -> str:
        return f"teacher:{text}"

    def build(cache_dir: Path, backend: ScriptedBackend) -> Any:
        return compile_on_hit(
            spec=spec, threshold=2, cache_dir=str(cache_dir), backend=backend,
            sync_compile=True, shadow_window=20,
        )(teacher)

    svc_a = build(tmp_path / "dir_a", backend_a)
    svc_b = build(tmp_path / "dir_b", backend_b)
    assert svc_a.task_id == svc_b.task_id, "the collision this test exists for did not occur"

    for svc, marker in ((svc_a, "A"), (svc_b, "B")):
        svc(f"seed0{marker}")
        svc(f"seed1{marker}")
        svc(f"pair{marker}")
    _drain(svc_a, timeout=10.0)
    _drain(svc_b, timeout=10.0)

    inputs_a = {p["input_payload"] for p in _pairs(svc_a)}
    inputs_b = {p["input_payload"] for p in _pairs(svc_b)}
    assert inputs_a == {"pairA"}
    assert inputs_b == {"pairB"}

    # Closing one database must not break the other.
    svc_a.db.close()
    svc_b("pairB2")
    _drain(svc_b, timeout=10.0)
    assert {p["input_payload"] for p in _pairs(svc_b)} == {"pairB", "pairB2"}


def test_caller_latency_unaffected_by_worker_prune(tmp_path: Path) -> None:
    """The worker's prune must not show up in the caller's wall clock.

    The docstring here used to say "TraceDB is one connection behind one lock, so the
    worker's prune runs against the caller's critical section". That premise is
    falsified by J-7: the worker now has its own connection and the lock no longer
    serialises database access, so the prune contends only for SQLite's single writer.
    The assertion is unchanged and still holds; only the reason it holds has changed.

    `shadow_queue_size` is raised from 32 because removing the shared lock also
    removed the accidental backpressure it applied to the caller -- the caller now
    enqueues faster than the worker drains and a size-32 queue drops comparisons (27
    of 60 observed). That is J-7 working, not a regression, but a dropped comparison
    never enters a window, so this test's retention assertion needs the queue to hold.
    `shadow_max_pairs` is 40 rather than 20 because J-12 requires
    `>= 2 * max(shadow_window, audit_window)`.
    """
    backend = ScriptedBackend()
    svc, _ = _make(
        tmp_path, "prunelatency", backend,
        shadow_window=20, shadow_max_pairs=40, shadow_queue_size=128,
    )
    svc("a")
    svc("b")

    started = time.perf_counter()
    for n in range(60):
        svc(f"churn{n}")
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"60 calls took {elapsed:.2f}s while the worker was pruning"
    _drain(svc)
    count = _query(svc.db, "SELECT COUNT(*) AS n FROM shadow_pairs;")[0]["n"]
    assert count <= 40


def test_process_exit_with_queued_shadow_jobs_neither_raises_nor_hangs(
    tmp_path: Path,
) -> None:
    """A process that exits with comparisons still queued exits 0, quietly, and promptly."""
    script = tmp_path / "exiting.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import time
            from paw_kit import MockPAWBackend, compile_on_hit

            class SlowBackend(MockPAWBackend):
                def infer(self, adapter_path, input_text, grammar_constraint=None):
                    time.sleep(5.0)
                    return "slow"

            @compile_on_hit(
                spec="exit test", threshold=2, cache_dir={str(tmp_path / "exitcache")!r},
                backend=SlowBackend(), sync_compile=True, shadow_window=20,
                shadow_queue_size=8,
            )
            def teacher(text):
                return "teacher:" + text

            teacher("a")
            teacher("b")
            for i in range(8):
                teacher("q%d" % i)
            """
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
    started = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=60
    )
    elapsed = time.perf_counter() - started

    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stderr, result.stderr
    # The atexit drain is a single global budget, not one per task.
    assert elapsed < 20.0, f"process exit took {elapsed:.1f}s with jobs queued"


def test_get_agreement_shape_and_last_disagreements(tmp_path: Path) -> None:
    """Every documented key is present, newest-first and capped at n."""
    backend = ScriptedBackend(agree_when=lambda text: False)
    svc, _ = _make(tmp_path, "shape", backend, shadow_window=20)

    empty = svc.get_agreement()
    # Finding 2: `stalled` is now the persisted, epoch-scoped boolean flag; the
    # pre-existing in-process subsampling counter (also historically named `stalled`)
    # is exposed as `stall_subsampled` to make room for it.
    expected_keys = {
        "state", "phase", "rate", "window", "samples", "agree", "disagree", "error",
        "teacher_error", "stalled", "dropped",
        # M-4b (C4b, bug-hunt Track C): the same total broken down by which of
        # shadow.py's three submission routes a drop came from -- previously
        # conflated in `dropped` alone.
        "dropped_shadow", "dropped_audit", "dropped_fail_open",
        "stall_subsampled", "pending", "last_disagreements",
    }
    assert set(empty) == expected_keys
    assert empty["rate"] is None
    assert empty["stalled"] is False
    assert empty["last_disagreements"] == []

    svc("a")
    svc("b")
    for n in range(5):
        svc(f"d{n}")
    _drain(svc)

    agreement = svc.get_agreement(n=3)
    assert agreement["state"] == "shadow"
    assert agreement["phase"] == "shadow"
    assert agreement["window"] == 20
    assert agreement["samples"] == 5
    assert agreement["rate"] == 0.0
    assert agreement["stalled"] is False, "5 samples is nowhere near the stall point"
    assert len(agreement["last_disagreements"]) == 3
    payloads = [entry["input_payload"] for entry in agreement["last_disagreements"]]
    assert payloads == ["d4", "d3", "d2"], "newest first"
    assert set(agreement["last_disagreements"][0]) == {
        "input_payload", "teacher_output", "adapter_output", "verdict", "error_type",
        "phase", "timestamp",
    }


def test_config_change_between_runs_starts_a_fresh_window(tmp_path: Path) -> None:
    """Lowering shadow_window must not re-slice an existing epoch's history."""
    spec = "Echo the input back (configdrift)"
    cache_dir = str(tmp_path / "cache_configdrift")
    backend = ScriptedBackend()
    _MODULE_TEACHER_CALLS["n"] = 0

    first = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend,
        sync_compile=True, shadow_window=20,
    )(_module_teacher)
    first("a")
    first("b")
    for n in range(3):
        first(f"c{n}")
    _drain(first)
    epoch_before = first.db.get_task_routing(first.task_id)[2]
    assert len(_pairs(first)) == 3
    first.db.close()

    reconfigured = compile_on_hit(
        spec=spec, threshold=2, cache_dir=cache_dir, backend=backend,
        sync_compile=True, shadow_window=3, shadow_threshold=1.0,
    )(_module_teacher)
    assert reconfigured.task_id == first.task_id
    epoch_after = reconfigured.db.get_task_routing(reconfigured.task_id)[2]
    assert epoch_after == epoch_before + 1
    # The three comparisons from the old configuration are still on disk, but they no
    # longer count toward the new window.
    assert len(_pairs(reconfigured)) == 3
    assert reconfigured.get_agreement()["samples"] == 0
    assert "config_change" in [t["reason"] for t in _transitions(reconfigured)]


def test_config_change_mid_ready_also_starts_a_fresh_epoch(tmp_path: Path) -> None:
    """Finding 6: reconciliation must bump the epoch in `ready`, not only `shadow` --
    otherwise lowering `audit_window`/`demote_threshold` on a promoted task re-slices
    its existing audit history under new arithmetic."""
    backend = ScriptedBackend()
    svc, _ = _promote(tmp_path, "readyconfig", backend, shadow_window=1, shadow_threshold=1.0)
    assert svc.db.get_status(svc.task_id) == "ready"
    epoch_before = svc.db.get_task_routing(svc.task_id)[2]

    bumped = svc.db.sync_shadow_config(
        svc.task_id,
        {
            "shadow_window": 1,
            "shadow_threshold": 1.0,
            "audit_window": 5,  # changed from the default 20
            "demote_threshold": 0.6,
        },
    )
    assert bumped is True

    epoch_after = svc.db.get_task_routing(svc.task_id)[2]
    assert epoch_after == epoch_before + 1
    assert svc.db.get_status(svc.task_id) == "ready", "a config change must not itself change status"
    assert "config_change" in [t["reason"] for t in _transitions(svc)]


# --- The response_model path --------------------------------------------------


class ModelBackend(MockPAWBackend):
    """A compiled adapter returning a fixed JSON payload for a structured task."""

    def __init__(self, payload: str) -> None:
        super().__init__()
        self.payload = payload

    def infer(
        self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None
    ) -> str:
        return self.payload


def _make_structured(tmp_path: Path, name: str, backend: MockPAWBackend, **kwargs: Any) -> Any:
    spec = f"Triage the ticket ({name})"
    params: Dict[str, Any] = dict(
        spec=spec, threshold=2, response_model=Triage, cache_dir=str(tmp_path / f"c_{name}"),
        backend=backend, sync_compile=True,
    )
    params.update(kwargs)

    def teacher(text: str) -> Triage:
        return Triage(priority="high", urgency_score=4)

    return compile_on_hit(**params)(teacher)


def test_shadow_compares_response_model_instances_field_wise(tmp_path: Path) -> None:
    """With a response_model both sides are re-validated and compared field-wise."""
    matching = ModelBackend('{"priority": "high", "urgency_score": 4}')
    svc = _make_structured(tmp_path, "modelagree", matching, shadow_window=2, shadow_threshold=1.0)
    svc("a")
    svc("b")
    svc("c")
    svc("d")
    _drain(svc)
    assert svc.db.get_status(svc.task_id) == "ready"
    assert all(p["verdict"] == "agree" for p in _pairs(svc))

    differing = ModelBackend('{"priority": "low", "urgency_score": 1}')
    svc2 = _make_structured(tmp_path, "modeldisagree", differing, shadow_window=2)
    svc2("a")
    svc2("b")
    svc2("c")
    svc2("d")
    _drain(svc2)
    assert svc2.db.get_status(svc2.task_id) == "shadow"
    assert all(p["verdict"] == "disagree" for p in _pairs(svc2))


def test_adapter_validation_failure_in_the_worker_is_an_error_not_a_crash(
    tmp_path: Path,
) -> None:
    """A response_model parse failure on the adapter side is a recorded error verdict."""
    broken = ModelBackend("MALFORMED_OUTPUT_CAUSING_PARSE_ERROR")
    svc = _make_structured(tmp_path, "modelbroken", broken, shadow_window=2)
    svc("a")
    svc("b")
    assert svc("c").priority == "high"  # the caller is untouched
    svc("d")
    _drain(svc)

    pairs = _pairs(svc)
    assert len(pairs) == 2
    assert {p["verdict"] for p in pairs} == {"error"}
    assert all(p["adapter_output"] is None for p in pairs)
    assert svc.db.get_status(svc.task_id) == "shadow"
    assert svc.get_fail_open_count() == 0


def test_field_tolerance_agreement_promotes_where_the_default_would_not(
    tmp_path: Path,
) -> None:
    """The documented recipe for the case the repo's own measurement hit."""
    from paw_kit import field_tolerance_agreement

    near = ModelBackend('{"priority": "high", "urgency_score": 5}')
    strict = _make_structured(tmp_path, "tolstrict", near, shadow_window=2, shadow_threshold=1.0)
    strict("a")
    strict("b")
    strict("c")
    strict("d")
    _drain(strict)
    assert strict.db.get_status(strict.task_id) == "shadow"

    near2 = ModelBackend('{"priority": "high", "urgency_score": 5}')
    tolerant = _make_structured(
        tmp_path, "toltolerant", near2, shadow_window=2, shadow_threshold=1.0,
        agreement_fn=field_tolerance_agreement({"urgency_score": 1}),
    )
    tolerant("a")
    tolerant("b")
    tolerant("c")
    tolerant("d")
    _drain(tolerant)
    assert tolerant.db.get_status(tolerant.task_id) == "ready"
