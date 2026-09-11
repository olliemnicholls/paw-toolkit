"""Persistence and concurrency regressions for paw.jit (bug hunt track G).

Every test here is a row-level assertion. `PRAGMA integrity_check` staying `ok` is
explicitly *not* evidence for anything in this file: D-1 is logical corruption the
SQLite file format cannot see, and the hunt measured `integrity_check == ok`
throughout every failing trial.

The multiprocess matrix below uses separate OS processes deliberately. Threads
sharing one `TraceDB` serialise on its own lock and cannot exercise the
read-outside-the-write-transaction race at all, which is why the pre-existing
single-process tests passed against the broken code for the whole of its life.
"""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sqlite3
import stat
import threading
from typing import Any, Callable, Dict, List, Tuple

import pytest

from paw_kit.jit.db import TraceDB


# --- multiprocess workers (top level so they are picklable) --------------------


def _seed(db_path: str, task_id: str) -> None:
    db = TraceDB(db_path=db_path)
    db.record_trace(task_id, "seed", "seed", 1.0)
    db.close()


def _w_record_shadow_pair(db_path: str, task_id: str, n: int) -> None:
    db = TraceDB(db_path=db_path)
    for i in range(n):
        db.record_shadow_pair(
            task_id=task_id, state_epoch=0, phase="shadow",
            input_payload=f"in-{os.getpid()}-{i}", teacher_output="t",
            adapter_output="a", verdict="agree", max_pairs=100000,
        )
    db.close()


def _w_set_status(db_path: str, task_id: str, n: int) -> None:
    db = TraceDB(db_path=db_path)
    for i in range(n):
        db.set_status(task_id, "shadow" if i % 2 == 0 else "ready")
    db.close()


def _w_set_shadow_started(db_path: str, task_id: str, n: int) -> None:
    db = TraceDB(db_path=db_path)
    for i in range(n):
        db.set_shadow_started(task_id, f"/tmp/{task_id}.paw")
    db.close()


def _w_set_ready_from_compile(db_path: str, task_id: str, n: int) -> None:
    db = TraceDB(db_path=db_path)
    for i in range(n):
        db.set_ready_from_compile(task_id, f"/tmp/{task_id}.paw")
    db.close()


def _w_sync_shadow_config(db_path: str, task_id: str, n: int) -> None:
    db = TraceDB(db_path=db_path)
    for i in range(n):
        # A value unique to (process, iteration) so every call is a real config
        # change: sync_shadow_config short-circuits on an unchanged config.
        db.sync_shadow_config(task_id, {"shadow_window": os.getpid() * 10000 + i})
    db.close()


def _run_procs(target: Callable[..., None], db_path: str, task_id: str,
               procs: int, per: int) -> None:
    workers = [
        multiprocessing.Process(target=target, args=(db_path, task_id, per))
        for _ in range(procs)
    ]
    for p in workers:
        p.start()
    for p in workers:
        p.join(timeout=120)
        assert p.exitcode == 0, "a worker process crashed"


def _rows(db_path: str, sql: str, params: tuple = ()) -> List[Tuple[Any, ...]]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# --- D-1: the read of a read-modify-write must be inside the write transaction --


def _assert_seq_is_a_clock(db_path: str, task_id: str, expected: int) -> None:
    """`seq` is the tumbling-window clock: one value per countable comparison.

    A duplicated `seq` fires the promotion check twice for one window (a sliding
    draw); a skipped one pushes a completed window past `seq % window == 0` so it is
    never evaluated at all.
    """
    rows = _rows(db_path, "SELECT seq FROM shadow_pairs WHERE task_id = ?;", (task_id,))
    seqs = [r[0] for r in rows]
    assert len(seqs) == expected, f"expected {expected} rows, got {len(seqs)}"
    assert len(set(seqs)) == len(seqs), (
        f"{len(seqs) - len(set(seqs))} duplicate seq value(s): the window clock "
        "assigned one number to several comparisons"
    )
    assert max(seqs) == expected, (
        f"max(seq)={max(seqs)} over {expected} rows: seq values were skipped, so a "
        "completed window can never satisfy seq % window == 0"
    )


def _assert_epochs_are_a_contiguous_run(epochs: List[int], label: str) -> None:
    """`state_epoch` advances by exactly one per recorded transition, with no reuse.

    `state_epoch` is what the shadow-window arithmetic keys on, so two transitions
    sharing an epoch means comparisons from before a transition get counted after it.
    The hunt measured 160 transitions sharing 40 epochs; its own control run (one
    explicit `BEGIN IMMEDIATE`) gave `distinct_epochs == transitions` exactly.
    """
    assert epochs, f"{label}: no transitions were recorded at all"
    assert len(set(epochs)) == len(epochs), (
        f"{label}: {len(epochs)} transitions share only {len(set(epochs))} "
        "distinct epochs -- bumps were lost to a read outside the write transaction"
    )
    lo = min(epochs)
    assert sorted(epochs) == list(range(lo, lo + len(epochs))), (
        f"{label}: epochs {sorted(epochs)[:8]}... are not a contiguous run from {lo}"
    )


def _assert_one_epoch_per_transition(db_path: str, task_id: str, expected: int) -> None:
    """Matrix arm check: one distinct, contiguous epoch per `state_transitions` row.

    The row count is deliberately *not* asserted to equal `expected`. `set_status`
    only records a transition when the status actually changes, so two processes
    happening to write the same status back-to-back correctly produce no row -- which
    is why the hunt's own control figure was 158, not 160, over 160 calls.
    """
    rows = _rows(
        db_path,
        "SELECT state_epoch FROM state_transitions WHERE task_id = ?;",
        (task_id,),
    )
    epochs = [r[0] for r in rows]
    assert len(epochs) >= expected // 2, (
        f"only {len(epochs)} of up to {expected} transitions were recorded"
    )
    _assert_epochs_are_a_contiguous_run(epochs, task_id)
    final = _rows(db_path, "SELECT state_epoch FROM tasks WHERE task_id = ?;", (task_id,))
    assert final[0][0] == max(epochs), (
        f"{task_id}: tasks.state_epoch is {final[0][0]} but the newest transition "
        f"row carries {max(epochs)}"
    )


_MATRIX = {
    "record_shadow_pair": (_w_record_shadow_pair, _assert_seq_is_a_clock),
    "set_status": (_w_set_status, _assert_one_epoch_per_transition),
    "set_shadow_started": (_w_set_shadow_started, _assert_one_epoch_per_transition),
    "set_ready_from_compile": (_w_set_ready_from_compile, _assert_one_epoch_per_transition),
}


@pytest.mark.parametrize("method", sorted(_MATRIX))
def test_read_modify_write_is_serialised_across_processes_D_1(
    tmp_path: Path, method: str
) -> None:
    """Every read-modify-write method holds the write lock across its leading read.

    `isolation_level="IMMEDIATE"` does not do this: Python emits `BEGIN IMMEDIATE`
    only before a DML statement, so the SELECT runs in autocommit and the
    transaction opens on the following write. Measured at main: 0 of 5 trials clean.
    """
    worker, check = _MATRIX[method]
    db_path = str(tmp_path / "matrix.db")
    task_id = f"mp_{method}"
    procs, per = 4, 30
    _seed(db_path, task_id)
    _run_procs(worker, db_path, task_id, procs, per)
    check(db_path, task_id, procs * per)


def test_sync_shadow_config_epoch_bumps_are_distinct_across_processes_D_1(
    tmp_path: Path,
) -> None:
    """`sync_shadow_config`'s bespoke arm of the D-1 matrix.

    It does not fit the parametrisation above: it short-circuits on an unchanged
    config and bumps the epoch only when the task is already `shadow`/`ready`, so it
    needs a per-call-unique config value and a task parked in `shadow` first.
    """
    db_path = str(tmp_path / "syncconfig.db")
    task_id = "mp_sync_shadow_config"
    procs, per = 4, 30
    _seed(db_path, task_id)
    db = TraceDB(db_path=db_path)
    db.sync_shadow_config(task_id, {"shadow_window": -1})
    db.set_status(task_id, "shadow")
    db.close()

    _run_procs(_w_sync_shadow_config, db_path, task_id, procs, per)

    rows = _rows(
        db_path,
        "SELECT state_epoch FROM state_transitions WHERE task_id = ? AND reason = ?;",
        (task_id, "config_change"),
    )
    epochs = [r[0] for r in rows]
    assert len(epochs) == procs * per, (
        f"expected {procs * per} config-change transitions, got {len(epochs)}"
    )
    _assert_epochs_are_a_contiguous_run(epochs, task_id)


def test_init_db_leaves_journal_mode_wal_D_1(tmp_path: Path) -> None:
    """`_init_db` must never be given a transaction boundary.

    Regression guard, not a finding reproduction: this passes at `main` and its whole
    job is to keep passing. `PRAGMA journal_mode=WAL` issued inside an open
    transaction on a fresh database returns `"delete"` and leaves the file in
    rollback-journal mode **with no exception raised** -- so a blanket "wrap every
    read-modify-write method in `_write_txn`" rule, which `_init_db`'s
    probe-then-ALTER shape would otherwise attract, silently destroys the WAL
    guarantee that D-3's durability argument and J-7's per-thread connections both
    rest on.
    """
    db = TraceDB(db_path=str(tmp_path / "wal.db"))
    try:
        mode = db._conn.execute("PRAGMA journal_mode;").fetchone()[0]
    finally:
        db.close()
    assert mode.lower() == "wal", (
        f"journal_mode is {mode!r}, not 'wal' -- _init_db has been given a "
        "transaction boundary"
    )


def test_write_txn_refactor_leaves_the_already_correct_methods_correct(
    tmp_path: Path,
) -> None:
    """`record_trace`/`increment_compile_attempts` keep their exact semantics.

    They were already correct (write-first, so the implicit `BEGIN IMMEDIATE`
    already covered the following read); `_write_txn` is applied to them only for
    uniformity and must be a no-op there.
    """
    db = TraceDB(db_path=str(tmp_path / "uniform.db"))
    try:
        assert db.record_trace("t", "i", "o", 1.0) == 1
        assert db.record_trace("t", "i2", "o2", 1.0) == 2
        assert db.get_call_count("t") == 2
        assert db.increment_compile_attempts("t") == 1
        assert db.increment_compile_attempts("t") == 2
        assert len(db.get_traces("t")) == 2
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: db.record_trace("t", f"x{i}", "o", 1.0), range(40)))
        assert db.get_call_count("t") == 42
    finally:
        db.close()


def test_write_txn_holds_the_sqlite_write_lock_across_the_whole_block_D_1(
    tmp_path: Path,
) -> None:
    """The mechanism itself, asserted directly rather than through its effects.

    `isolation_level="IMMEDIATE"` leaves the write lock unheld until the first DML
    statement runs; `_write_txn` must take it on entry, before any read.
    """
    db_file = tmp_path / "lockheld.db"
    db = TraceDB(db_path=str(db_file))
    other = sqlite3.connect(str(db_file), timeout=0.2, isolation_level=None)
    try:
        with db._write_txn() as conn:
            assert conn.in_transaction, "_write_txn did not open a transaction"
            with pytest.raises(sqlite3.OperationalError):
                other.execute("BEGIN IMMEDIATE;")
        # Released on exit.
        other.execute("BEGIN IMMEDIATE;")
        other.execute("ROLLBACK;")
    finally:
        other.close()
        db.close()


def test_write_txn_does_not_begin_inside_an_open_transaction_D_1(
    tmp_path: Path,
) -> None:
    """The `if not self._conn.in_transaction` guard.

    An unguarded nested `BEGIN IMMEDIATE` raises `OperationalError`, which
    `_with_write_retry` catches and retries five times with backoff before
    re-raising -- laundering a programming error into ~0.75s of latency and a
    lock-shaped error message that says nothing about the real cause.
    """
    db = TraceDB(db_path=str(tmp_path / "nested.db"))
    try:
        db._conn.execute("BEGIN IMMEDIATE;")
        assert db._conn.in_transaction
        with db._write_txn() as conn:
            conn.execute(
                "INSERT INTO tasks (task_id, call_count, status, created_at, "
                "updated_at) VALUES ('n', 0, 'tracing', 'x', 'x');"
            )
        assert db.get_status("n") == "tracing"
    finally:
        db.close()


# --- D-5: retention caps are enforced per task, not per database ----------------


def test_retention_cap_is_enforced_for_every_task_D_5(tmp_path: Path) -> None:
    """`2 * cap` rows across several concurrently-active tasks, every task in bound.

    The counters gating the periodic prune used to count writes across *all* tasks
    while the prune they gate deletes rows for only the task that wrote on the Nth
    call. Under round-robin traffic the Nth call lands on the same task every time
    (N and the task count share a factor), so with T tasks any given task is pruned
    at most 1/T of the time -- measured in the hunt as 50 tasks holding 600 rows each
    against a documented worst case of 249.
    """
    from paw_kit.jit.db import _PRUNE_EVERY, _prune_interval

    cap = _PRUNE_EVERY + 10          # > _PRUNE_EVERY, so the prune really is periodic
    bound = cap + _prune_interval(cap) - 1   # the documented worst case
    tasks = [f"task{i}" for i in range(5)]
    db = TraceDB(db_path=str(tmp_path / "retention.db"))
    try:
        for i in range(2 * cap * len(tasks)):
            t = tasks[i % len(tasks)]
            db.record_shadow_pair(
                task_id=t, state_epoch=0, phase="shadow", input_payload=f"in{i}",
                teacher_output="t", adapter_output="a", verdict="agree",
                max_pairs=cap,
            )
        held = {
            t: db._conn.execute(
                "SELECT COUNT(*) FROM shadow_pairs WHERE task_id = ?;", (t,)
            ).fetchone()[0]
            for t in tasks
        }
    finally:
        db.close()
    over = {t: n for t, n in held.items() if n > bound}
    assert not over, (
        f"{len(over)} of {len(tasks)} tasks are over the documented bound of {bound} "
        f"rows: {over} (all tasks: {held})"
    )


def test_transition_retention_cap_is_enforced_for_every_task_D_5(tmp_path: Path) -> None:
    """The same, for `state_transitions` -- the table the hunt actually measured."""
    from paw_kit.jit.db import _STATE_TRANSITIONS_MAX_ROWS, _prune_interval

    cap = _STATE_TRANSITIONS_MAX_ROWS
    bound = cap + _prune_interval(cap) - 1
    # Two tasks, not three: the shared counter's Nth call only ever lands on one task
    # when the task count and `_prune_interval` share a factor, which is exactly the
    # traffic shape the hunt measured (50 tasks, interval 50, only `t49` ever pruned).
    # With three the residues cycle and the defect happens to hide.
    tasks = [f"tr{i}" for i in range(2)]
    db = TraceDB(db_path=str(tmp_path / "transitions.db"))
    try:
        for i in range(2 * cap * len(tasks)):
            t = tasks[i % len(tasks)]
            db.record_state_transition(t, "shadow", "ready", state_epoch=i)
        held = {
            t: db._conn.execute(
                "SELECT COUNT(*) FROM state_transitions WHERE task_id = ?;", (t,)
            ).fetchone()[0]
            for t in tasks
        }
    finally:
        db.close()
    over = {t: n for t, n in held.items() if n > bound}
    assert not over, (
        f"{len(over)} of {len(tasks)} tasks are over the documented bound of {bound} "
        f"rows: {over} (all tasks: {held})"
    )


def test_prune_counter_is_not_advanced_twice_by_a_retry_D_5(tmp_path: Path) -> None:
    """A retried write transaction counts as one write, not two.

    Both counters used to be incremented *inside* the closure `_with_write_retry`
    re-runs, so a transaction that rolled back at commit and succeeded on the retry
    advanced the prune clock twice for one persisted row.
    """
    import contextlib

    db = TraceDB(db_path=str(tmp_path / "retrycount.db"))
    try:
        real = db._write_txn
        state = {"failed": False}

        @contextlib.contextmanager
        def flaky():
            with real() as conn:
                yield conn
                if not state["failed"]:
                    state["failed"] = True
                    # After the INSERT, before the commit -- the real shape of a
                    # write that loses the race and is retried.
                    raise sqlite3.OperationalError("database is locked")

        db._write_txn = flaky  # type: ignore[method-assign]
        db.record_shadow_pair(
            task_id="r", state_epoch=0, phase="shadow", input_payload="i",
            teacher_output="t", adapter_output="a", verdict="agree",
        )
        db._write_txn = real  # type: ignore[method-assign]
        assert state["failed"], "the injected failure never fired"
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM shadow_pairs WHERE task_id = 'r';"
        ).fetchone()[0]
        assert rows == 1
        assert db._shadow_pair_writes.get("r") == 1, (
            f"one persisted row, but the prune clock advanced to "
            f"{db._shadow_pair_writes.get('r')}"
        )
    finally:
        db.close()


# --- D-7: a write against an unknown task_id must not vanish -------------------


@pytest.mark.parametrize(
    "call,column,expected",
    [
        (lambda db, t: db.set_status(t, "failed"), "status", "failed"),
        (lambda db, t: db.set_shadow_started(t, "/tmp/a.paw"), "status", "shadow"),
        (lambda db, t: db.set_ready_from_compile(t, "/tmp/a.paw"), "status", "ready"),
        (lambda db, t: db.increment_compile_attempts(t), "compile_attempts", 1),
        (lambda db, t: db.increment_fail_open(t), "fail_open_count", 1),
    ],
    ids=["set_status", "set_shadow_started", "set_ready_from_compile",
         "increment_compile_attempts", "increment_fail_open"],
)
def test_write_against_a_missing_task_row_does_not_vanish_D_7(
    tmp_path: Path, call: Callable[[Any, str], Any], column: str, expected: Any
) -> None:
    """All five bare `UPDATE ... WHERE task_id = ?` sites upsert.

    Against a missing `tasks` row a bare UPDATE affects zero rows and reports
    success. `increment_compile_attempts` then returns 0 forever, so
    `BackgroundCompiler`'s `attempts < _MAX_COMPILE_ATTEMPTS` bound is never reached
    and every failed compile is retried -- and paid for -- on every subsequent call.
    `set_status(..., 'failed')` silently does nothing, so the terminal state is
    unreachable. Reachable by deleting `.paw/traces.db` (documented as a cache) while
    a process holds a live wrapper.
    """
    db = TraceDB(db_path=str(tmp_path / f"missing_{column}.db"))
    try:
        call(db, "ghost")
        row = db._conn.execute(
            f"SELECT {column} FROM tasks WHERE task_id = 'ghost';"  # noqa: S608
        ).fetchone()
        assert row is not None, "the write vanished: no tasks row was created"
        assert row[0] == expected
    finally:
        db.close()


def test_no_orphaned_transition_against_a_missing_task_row_D_7(tmp_path: Path) -> None:
    """`set_shadow_started`/`set_ready_from_compile` are worse than the three named.

    Each unconditionally writes a `state_transitions` row straight after its UPDATE,
    so against a missing `tasks` row they left an orphaned transition record
    referencing a task that does not exist.
    """
    db = TraceDB(db_path=str(tmp_path / "orphan.db"))
    try:
        db.set_shadow_started("ghost1", "/tmp/a.paw")
        db.set_ready_from_compile("ghost2", "/tmp/a.paw")
        orphans = db._conn.execute(
            "SELECT COUNT(*) FROM state_transitions st "
            "WHERE NOT EXISTS (SELECT 1 FROM tasks t WHERE t.task_id = st.task_id);"
        ).fetchone()[0]
        assert orphans == 0, f"{orphans} transition row(s) reference no task"
    finally:
        db.close()


def test_increment_compile_attempts_returns_one_then_two_from_nothing_D_7(
    tmp_path: Path,
) -> None:
    """Two consecutive increments return 1 then 2 in every case, missing row included."""
    db = TraceDB(db_path=str(tmp_path / "attempts.db"))
    try:
        assert db.increment_compile_attempts("fresh") == 1
        assert db.increment_compile_attempts("fresh") == 2
        # And with a pre-existing row, unchanged behaviour.
        db.record_trace("known", "i", "o", 1.0)
        assert db.increment_compile_attempts("known") == 1
        assert db.increment_compile_attempts("known") == 2
    finally:
        db.close()


def test_compile_attempts_distinguishes_no_row_from_zero_D_7(tmp_path: Path) -> None:
    """The counter read can tell "no task" from "a task that has never failed"."""
    db = TraceDB(db_path=str(tmp_path / "distinguish.db"))
    try:
        db.record_trace("known", "i", "o", 1.0)
        assert db._get_compile_attempts_locked("absent") is None
        assert db._get_compile_attempts_locked("known") == 0
        # The public accessor keeps its bare-int contract.
        assert db.get_compile_attempts("absent") == 0
        assert db.get_compile_attempts("known") == 0
    finally:
        db.close()


def test_background_compiler_attempts_stay_bounded_after_a_db_deletion_D_7(
    tmp_path: Path,
) -> None:
    """A failing compile reaches `failed` in a bounded number of attempts.

    The reachable case: the user deletes `.paw/traces.db` (documented as a cache)
    while a process holds a live wrapper. Every one of those compiles is paid for, so
    an unbounded retry loop is a money bug, not just a correctness one.
    """
    from paw_kit.jit.compiler import BackgroundCompiler

    class _Failing:
        def compile(self, **kwargs: Any) -> str:
            raise RuntimeError("compile failed")

    db_file = tmp_path / "bounded" / "traces.db"
    db = TraceDB(db_path=str(db_file))
    db.record_trace("t", "i", "o", 1.0)
    compiler = BackgroundCompiler()
    try:
        # The deletion the finding describes: the file goes, the live handle stays.
        # SQLite keeps writing to the unlinked inode, so every tasks row is gone.
        db._conn.execute("DELETE FROM tasks;")
        db._conn.commit()
        for _ in range(BackgroundCompiler._MAX_COMPILE_ATTEMPTS + 3):
            compiler.trigger_compilation(
                task_id="t", spec="s", db=db, backend=_Failing(),  # type: ignore[arg-type]
                output_path=str(tmp_path / "t.paw"), sync=True,
            )
            if db.get_status("t") == "failed":
                break
        assert db.get_status("t") == "failed", (
            f"status is {db.get_status('t')!r} after "
            f"{BackgroundCompiler._MAX_COMPILE_ATTEMPTS + 3} failing compiles; the "
            "retry bound was never reached"
        )
        assert db.get_compile_attempts("t") <= BackgroundCompiler._MAX_COMPILE_ATTEMPTS
    finally:
        db.close()


def test_background_compiler_fails_closed_when_the_counter_stalls_D_7(
    tmp_path: Path,
) -> None:
    """A counter that never advances must terminate the retry, not licence it.

    `attempts < _MAX_COMPILE_ATTEMPTS` is only a bound if the count moves. TraceDB's
    upsert makes a stalled counter unreachable; this pins the compiler side so the
    bound holds whatever the database underneath does.
    """
    from paw_kit.jit.compiler import BackgroundCompiler

    class _Failing:
        def compile(self, **kwargs: Any) -> str:
            raise RuntimeError("compile failed")

    db = TraceDB(db_path=str(tmp_path / "stalled" / "traces.db"))
    db.record_trace("t", "i", "o", 1.0)
    # The historical symptom, injected directly: the write reports success and the
    # count stays put.
    db.increment_compile_attempts = lambda task_id: 0  # type: ignore[method-assign]
    compiler = BackgroundCompiler()
    try:
        compiler.trigger_compilation(
            task_id="t", spec="s", db=db, backend=_Failing(),  # type: ignore[arg-type]
            output_path=str(tmp_path / "t.paw"), sync=True,
        )
        assert db.get_status("t") == "failed", (
            "a stalled attempt counter reset the task to 'tracing', which is the "
            "unbounded paid-retry loop the counter exists to prevent"
        )
    finally:
        db.close()


# --- D-6: the cache directory's permissions are tightened, never set -----------


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def test_group_shared_cache_dir_is_tightened_not_reset_D_6(tmp_path: Path) -> None:
    """A deliberately group-shared `0o2775` directory keeps its setgid bit.

    The old code *set* `0o700` rather than tightening, so importing a module that
    constructs a `TraceDB` rewrote permissions on a directory the user shares with a
    group -- and dropped the setgid bit that makes the sharing work at all.
    """
    parent = tmp_path / "shared"
    parent.mkdir()
    parent.chmod(0o2775)
    assert _mode(parent) == 0o2775, "the filesystem under tmp_path dropped setgid"
    TraceDB(db_path=str(parent / "traces.db")).close()
    assert _mode(parent) == 0o2700, (
        f"expected 0o2700 (group/other cleared, setgid preserved), got {oct(_mode(parent))}"
    )


def test_sticky_cache_dir_keeps_its_sticky_bit_D_6(tmp_path: Path) -> None:
    """`0o1777` -> `0o1700`: the mask clears group/other and touches nothing else."""
    parent = tmp_path / "sticky"
    parent.mkdir()
    parent.chmod(0o1777)
    TraceDB(db_path=str(parent / "traces.db")).close()
    assert _mode(parent) == 0o1700, oct(_mode(parent))


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the write bit")
def test_read_only_cache_dir_is_not_made_writable_D_6(tmp_path: Path) -> None:
    """`0o500` stays `0o500`. Masking never adds a bit, so no special case is needed.

    The old code made a deliberately read-only directory writable, which is how the
    database below gets created at all -- so the assertion that it *cannot* be created
    is the assertion that the directory was not widened.
    """
    parent = tmp_path / "ro"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        with pytest.raises(sqlite3.OperationalError):
            TraceDB(db_path=str(parent / "traces.db"))
        assert _mode(parent) == 0o500, (
            f"a read-only directory was widened to {oct(_mode(parent))}"
        )
    finally:
        parent.chmod(0o700)


def test_traces_db_does_not_chmod_the_process_cwd_D_6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`TraceDB("traces.db")` has `parent == Path(".")` -- it must not chmod the CWD.

    The narrow carve-out: always tighten, except when the resolved parent is the
    process CWD or the user's home directory. Anything broader would stop `TraceDB`
    tightening a pre-existing `.paw`, which is behaviour it correctly provides.
    """
    work = tmp_path / "work"
    work.mkdir()
    work.chmod(0o755)
    monkeypatch.chdir(work)
    TraceDB(db_path="traces.db").close()
    assert _mode(work) == 0o755, (
        f"the process working directory was chmodded to {oct(_mode(work))}"
    )


def test_preexisting_cache_dir_is_still_tightened_D_6(tmp_path: Path) -> None:
    """The carve-out must not stop `.paw` itself being tightened (guard, green at main)."""
    parent = tmp_path / ".paw"
    parent.mkdir()
    parent.chmod(0o755)
    TraceDB(db_path=str(parent / "traces.db")).close()
    assert _mode(parent) == 0o700, oct(_mode(parent))


# --- D-3 / D-8: atomic_write_text ---------------------------------------------


def test_atomic_write_text_fsyncs_the_file_and_the_directory_D_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spy on `os.fsync`, not a simulated power loss.

    The docstring's crash-atomicity guarantee held for a process crash and not for a
    power loss: `strace` counted `fsync calls: 0`, so on ext4 `data=ordered` the
    rename can reach disk before the data blocks. The torn-file outcome itself is
    still unreproduced (it needs failure injection) and is not claimed here.
    """
    from paw_kit import atomicio

    seen: List[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (seen.append(fd), real_fsync(fd))[1])

    # Two missing directory levels, so `parents=True` is load-bearing rather than
    # incidentally satisfied by a single mkdir.
    target = tmp_path / "out" / "nested" / "manifest.paw"
    atomicio.atomic_write_text(target, "payload")

    assert target.read_text() == "payload"
    assert len(seen) >= 2, (
        f"{len(seen)} fsync call(s): the file and its parent directory must both be "
        "synced, or the rename can reach disk before the data blocks"
    )


def test_atomic_write_text_creates_a_dangling_symlinks_target_directory_D_8(
    tmp_path: Path,
) -> None:
    """A symlink can point somewhere that does not exist yet.

    `target.parent` existing says nothing about the *resolved* target's parent, and
    `mkstemp(dir=...)` fails outright if that directory is missing -- so the resolved
    parent has to be created too, all of it.
    """
    from paw_kit.atomicio import atomic_write_text

    real = tmp_path / "deep" / "deeper" / "program.paw"
    link = tmp_path / "dangling.paw"
    link.symlink_to(real)
    atomic_write_text(link, "new")
    assert real.read_text() == "new"
    assert link.is_symlink()


def test_atomic_write_text_follows_a_symlink_target_D_8(tmp_path: Path) -> None:
    """`os.replace` does not follow symlinks: the link was replaced by a regular file.

    The real target then kept its stale content and every other reader of it saw the
    old program.
    """
    real = tmp_path / "real" / "program.paw"
    real.parent.mkdir()
    real.write_text("old")
    link = tmp_path / "current.paw"
    link.symlink_to(real)

    from paw_kit.atomicio import atomic_write_text

    atomic_write_text(link, "new")

    assert link.is_symlink(), "the symlink was replaced by a regular file"
    assert real.read_text() == "new", "the symlink's real target kept stale content"
    assert not list(tmp_path.glob(".*tmp")), "a temp file was left behind"
    assert not list(real.parent.glob(".*tmp")), "a temp file was left behind"


def test_atomic_write_text_creates_its_temp_file_beside_the_resolved_target_D_8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`mkstemp` must use the *resolved* target's directory, or `os.replace` gets EXDEV.

    Same-directory placement is also what guarantees the destination gets a fresh
    inode on every write, which `decorator.py`'s adapter-callable cache key depends on.
    """
    import tempfile as _tempfile

    real = tmp_path / "real" / "program.paw"
    real.parent.mkdir()
    real.write_text("old")
    link = tmp_path / "current.paw"
    link.symlink_to(real)

    seen: List[str] = []
    real_mkstemp = _tempfile.mkstemp

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(str(kwargs.get("dir")))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(_tempfile, "mkstemp", spy)
    from paw_kit.atomicio import atomic_write_text

    atomic_write_text(link, "new")
    assert seen == [str(real.parent)], (
        f"temp file created in {seen}, not beside the resolved target {real.parent}"
    )


def test_atomic_write_text_preserves_an_existing_destination_mode_D_8(
    tmp_path: Path
) -> None:
    """`mkstemp` creates at 0600, so a destination the user had at 0644 silently lost it.

    The mode is applied to the temp file *before* `os.replace`, not after: applying it
    afterwards leaves a window in which the destination is readable only by its owner.
    """
    from paw_kit.atomicio import atomic_write_text

    target = tmp_path / "manifest.paw"
    target.write_text("old")
    target.chmod(0o644)
    atomic_write_text(target, "new")
    assert _mode(target) == 0o644, (
        f"destination mode became {oct(_mode(target))} on rewrite"
    )
    # A destination that does not exist yet keeps mkstemp's restrictive default.
    fresh = tmp_path / "fresh.paw"
    atomic_write_text(fresh, "x")
    assert _mode(fresh) == 0o600, oct(_mode(fresh))


# --- D-9: a retention cap below the window makes the window unscoreable --------


def test_get_agreement_stats_refuses_an_unscoreable_window_D_9(tmp_path: Path) -> None:
    """`max_pairs < window` means no window can ever complete -- silently.

    A window is scored only at `samples == window` and pruning bounds `samples` by the
    cap, so neither promotion nor demotion can ever fire while `paw-kit report` shows
    a healthy rate. The shipped decorator guards the combination at decoration time;
    any other caller of these public methods did not, which is why the guard also has
    to live in the method that does the arithmetic.
    """
    db = TraceDB(db_path=str(tmp_path / "d9.db"))
    try:
        for i in range(30):
            db.record_shadow_pair(
                task_id="t", state_epoch=0, phase="shadow", input_payload=f"i{i}",
                teacher_output="t", adapter_output="a", verdict="agree", max_pairs=5,
            )
        with pytest.raises(ValueError, match="retention"):
            db.get_agreement_stats("t", 0, 20, "shadow")
    finally:
        db.close()


def test_get_agreement_stats_accepts_a_window_the_cap_can_still_fill_D_9(
    tmp_path: Path,
) -> None:
    """The boundary: `max_pairs == window` is still scoreable, and must not raise."""
    db = TraceDB(db_path=str(tmp_path / "d9ok.db"))
    try:
        for i in range(30):
            db.record_shadow_pair(
                task_id="t", state_epoch=0, phase="shadow", input_payload=f"i{i}",
                teacher_output="t", adapter_output="a", verdict="agree", max_pairs=20,
            )
        stats = db.get_agreement_stats("t", 0, 20, "shadow")
        assert stats["samples"] == 20 and stats["rate"] == 1.0
        # And an epoch that has simply not run a full window yet reads normally.
        assert db.get_agreement_stats("t", 9, 20, "shadow")["samples"] == 0
    finally:
        db.close()


# --- J-7: one SQLite connection per thread, inside one TraceDB -----------------


def test_each_thread_gets_its_own_connection_J_7(tmp_path: Path) -> None:
    """The shadow worker must not share the caller's connection.

    Shadow mode roughly doubled the decorator's caller latency because the worker's
    `SELECT MAX(seq)` + INSERT + prune contended with the caller for one `TraceDB`
    connection behind one lock. WAL supports a connection per thread.
    """
    db = TraceDB(db_path=str(tmp_path / "threadlocal.db"))
    try:
        main_conn = db._conn
        seen: List[Any] = []
        # A barrier, because a ThreadPoolExecutor happily runs three instant tasks on
        # one worker thread -- which would make this assert nothing at all.
        barrier = threading.Barrier(3)

        def grab() -> Any:
            barrier.wait(timeout=10)
            return db._conn

        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in [pool.submit(grab) for _ in range(3)]:
                seen.append(f.result())
        assert all(c is not main_conn for c in seen), "a worker reused the caller's connection"
        assert len({id(c) for c in seen}) == 3, "two worker threads shared a connection"
        # Same thread, same connection -- not a new one per call.
        assert db._conn is main_conn
    finally:
        db.close()


def test_close_closes_every_threads_connection_J_7(tmp_path: Path) -> None:
    """`close()` closes one connection per thread that touched the database, not one."""
    db = TraceDB(db_path=str(tmp_path / "closeall.db"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda i: db.record_trace("t", f"i{i}", "o", 1.0), range(3)))
    conns = list(db._conns)
    assert len(conns) == 4, f"expected 4 connections (main + 3 workers), got {len(conns)}"
    db.close()
    for conn in conns:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1;")
    # A thread that never connected must not silently reopen the file either.
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(sqlite3.ProgrammingError):
            pool.submit(db.get_call_count, "t").result()


def test_status_listeners_survive_the_per_thread_connection_J_7(tmp_path: Path) -> None:
    """A promotion driven from a worker thread still fires the registered listener.

    This is the reason J-7's fix is a thread-local connection inside `TraceDB` rather
    than a second `TraceDB` per worker: a second instance would carry its own empty
    `_status_listeners` list, silently stopping `_invalidate_adapter_cache` from
    firing on the worker's `try_promote`/`try_demote`.

    Green at `main` by design: with one shared connection there is nothing for a
    listener to get detached from. This pins that J-7 does not detach it.
    """
    db = TraceDB(db_path=str(tmp_path / "listener.db"))
    fired: List[str] = []
    db.register_status_listener(fired.append)
    try:
        db.record_trace("t", "i", "o", 1.0)
        db.set_shadow_started("t", "/tmp/t.paw")
        fired.clear()
        epoch = db.get_task_routing("t")[2]
        with ThreadPoolExecutor(max_workers=1) as pool:
            won = pool.submit(db.try_promote, "t", epoch, 1.0, 20).result()
        assert won, "the worker-thread promotion did not win"
        assert fired == ["t"], (
            f"listener fired {fired!r} -- the adapter-callable cache is no longer "
            "invalidated by a worker-thread promotion"
        )
    finally:
        db.close()


def test_read_modify_write_is_serialised_across_threads_J_7(tmp_path: Path) -> None:
    """Per-thread connections must not reintroduce D-1 inside a single process.

    The shared lock was what made `record_shadow_pair`'s read-then-write accidentally
    safe within one process. The moment each thread has its own connection that is
    gone, and only D-1's explicit `BEGIN IMMEDIATE` is holding it up -- which is why
    D-1 had to land before J-7 rather than merely before it in a list.

    Green at `main` by design (main's shared lock serialises these threads), so
    `red-at-main` cannot pin it. Proven red instead against this branch with D-1's
    `BEGIN IMMEDIATE` removed from `_write_txn` and J-7 left in place: 180 rows with
    duplicate seq values, i.e. D-1 reproduced inside a single process.
    """
    db = TraceDB(db_path=str(tmp_path / "threadrace.db"))
    threads, per = 6, 30
    try:
        db.record_trace("t", "seed", "seed", 1.0)

        def burst(_: int) -> None:
            for i in range(per):
                db.record_shadow_pair(
                    task_id="t", state_epoch=0, phase="shadow", input_payload=f"i{i}",
                    teacher_output="t", adapter_output="a", verdict="agree",
                    max_pairs=100000,
                )

        with ThreadPoolExecutor(max_workers=threads) as pool:
            list(pool.map(burst, range(threads)))
        seqs = [
            r[0] for r in db._conn.execute(
                "SELECT seq FROM shadow_pairs WHERE task_id = 't';"
            ).fetchall()
        ]
    finally:
        db.close()
    assert len(seqs) == threads * per
    assert len(set(seqs)) == len(seqs), (
        f"{len(seqs) - len(set(seqs))} duplicate seq value(s) across threads"
    )
    assert max(seqs) == threads * per


# --- J-8: task identity must not collide across distinct functions -------------


def _t1(text: str) -> str:
    return "one:" + text


def _t2(text: str) -> str:
    return "two:" + text


def test_same_qualname_in_two_places_gets_two_task_ids_J_8(tmp_path: Path) -> None:
    """`sha256(module.qualname:spec)` collided across distinct functions.

    Two decorations sharing a task_id share one call_count, one trace corpus, one
    adapter and one shadow window -- so after promotion, tenant A's adapter serves
    tenant B's traffic.
    """
    from paw_kit import compile_on_hit

    _t2.__qualname__ = _t1.__qualname__
    _t2.__module__ = _t1.__module__
    dec = compile_on_hit(spec="j8", threshold=10 ** 9, cache_dir=str(tmp_path / "c"))
    a, b = dec(_t1), dec(_t2)
    try:
        assert a.task_id != b.task_id, (
            "two distinct functions share one task identity, so they share one "
            "call_count, one corpus and one adapter"
        )
        a("x")
        a("x")
        b("y")
        assert a.get_call_count() == 2 and b.get_call_count() == 1, (
            "call counts are pooled across the two functions"
        )
    finally:
        a.db.close()


def test_decorator_factory_collision_is_warned_J_8(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Source location cannot separate one `def` decorated once per tenant, so warn.

    This is the exact shape the finding describes: `make.<locals>.classify` for every
    tenant, at the same file and line, so nothing automatic can tell them apart. The
    warning is what points the user at `task_id=`.
    """
    from paw_kit import compile_on_hit

    def make() -> Any:
        @compile_on_hit(spec="j8-factory", threshold=10 ** 9, cache_dir=str(tmp_path / "f"))
        def classify(text: str) -> str:
            return "x"
        return classify

    first = make()
    with caplog.at_level("WARNING", logger="paw_kit.jit"):
        second = make()
    try:
        assert first.task_id == second.task_id  # the collision this warns about
        assert any("already derived for a different function" in r.message
                   for r in caplog.records), (
            "a second decoration resolving to an existing task_id was not warned about"
        )
    finally:
        first.db.close()


def test_explicit_task_id_override_J_8(tmp_path: Path) -> None:
    """`task_id=` is the way out of the factory collision."""
    from paw_kit import compile_on_hit

    def make(tenant: str) -> Any:
        @compile_on_hit(
            spec="j8-override", threshold=10 ** 9,
            cache_dir=str(tmp_path / "o"), task_id=f"tenant-{tenant}",
        )
        def classify(text: str) -> str:
            return f"{tenant}:{text}"
        return classify

    a, b = make("a"), make("b")
    try:
        assert (a.task_id, b.task_id) == ("tenant-a", "tenant-b")
        a("x")
        b("y")
        b("z")
        assert a.get_call_count() == 1 and b.get_call_count() == 2
    finally:
        a.db.close()


# --- J-11: one logical call, one payload --------------------------------------


def _payloads(wrapper: Any) -> List[str]:
    return [t["input_payload"] for t in wrapper.db.get_traces(wrapper.task_id)]


def test_positional_and_keyword_calls_share_one_payload_J_11(tmp_path: Path) -> None:
    """`f("hello")` and `f(text="hello")` are the same call and must record the same row.

    They used to record `hello` and `{"args": [], "kwargs": {"text": "hello"}}`, so a
    caller mixing styles trained the adapter on one encoding and served it another.
    """
    from paw_kit import compile_on_hit

    @compile_on_hit(spec="j11-bind", threshold=10 ** 9, cache_dir=str(tmp_path / "b"))
    def f(text: str) -> str:
        return "ok"

    try:
        f("hello")
        f(text="hello")
        assert _payloads(f) == ["hello", "hello"]
    finally:
        f.db.close()


def test_keyword_order_does_not_change_the_payload_J_11(tmp_path: Path) -> None:
    """`sort_keys=True`: keyword order is a calling detail, not part of the input."""
    from paw_kit import compile_on_hit

    @compile_on_hit(spec="j11-sort", threshold=10 ** 9, cache_dir=str(tmp_path / "s"))
    def f(**kw: Any) -> str:
        return "ok"

    try:
        f(b=1, a=2)
        f(a=2, b=1)
        seen = _payloads(f)
        assert seen[0] == seen[1], f"keyword order leaked into the payload: {seen}"
    finally:
        f.db.close()


def test_defaults_are_applied_before_serialising_J_11(tmp_path: Path) -> None:
    """An omitted default and an explicitly-passed one are the same call."""
    from paw_kit import compile_on_hit

    @compile_on_hit(spec="j11-def", threshold=10 ** 9, cache_dir=str(tmp_path / "d"))
    def f(text: str, mode: str = "fast") -> str:
        return "ok"

    try:
        f("x")
        f("x", mode="fast")
        seen = _payloads(f)
        assert seen[0] == seen[1], f"an omitted default changed the payload: {seen}"
    finally:
        f.db.close()


def test_unserializable_argument_is_refused_not_repr_encoded_J_11(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`default=str` embedded `id()`-bearing reprs into the JSON payload.

    Three equal-but-distinct objects therefore produced three distinct payloads that
    *looked* like clean JSON. The refusal does not make such a payload canonical --
    nothing can -- but it stops it masquerading as one, and says so in the log.
    """
    from paw_kit import compile_on_hit
    from paw_kit.jit import decorator as dec_mod

    class Opaque:
        pass

    @compile_on_hit(spec="j11-refuse", threshold=10 ** 9, cache_dir=str(tmp_path / "r"))
    def f(obj: Any) -> str:
        return "ok"

    dec_mod._UNSERIALIZABLE_WARNED.discard(f.__qualname__)
    try:
        with caplog.at_level("WARNING", logger="paw_kit.jit"):
            f(Opaque())
        payload = _payloads(f)[0]
        assert not payload.startswith('{"args"'), (
            f"an unserializable argument was still JSON-encoded via repr: {payload}"
        )
        assert any("JSON cannot represent" in r.message for r in caplog.records), (
            "the refusal was silent"
        )
    finally:
        f.db.close()


# --- J-12: a window is scored at or past its boundary, exactly once ------------


def test_decoration_rejects_a_cap_that_holds_only_one_window_J_12(
    tmp_path: Path,
) -> None:
    """`shadow_max_pairs >= 2 * max(window, audit_window)`. API-breaking, deliberately.

    At exactly one window's worth of retention, one `teacher_error` row makes the
    boundary unscoreable -- and the window was discarded without ever being retried,
    while `seq` kept advancing and the stall budget kept counting it.
    """
    from paw_kit import compile_on_hit

    with pytest.raises(ValueError, match="2 \\* max"):
        compile_on_hit(
            spec="j12", cache_dir=str(tmp_path / "x"),
            shadow_window=20, audit_window=20, shadow_max_pairs=20,
        )
    # Twice the window is accepted, and so are the shipped defaults.
    compile_on_hit(
        spec="j12", cache_dir=str(tmp_path / "y"),
        shadow_window=20, audit_window=20, shadow_max_pairs=40,
    )
    compile_on_hit(spec="j12", cache_dir=str(tmp_path / "z"))


class _ScriptedStats:
    """Minimal `TraceDB` stand-in for driving `_maybe_transition` directly."""

    def __init__(self, scripted: List[Dict[str, Any]]) -> None:
        self.scripted = scripted
        self.stats_calls = 0
        self.promotions: List[int] = []

    def get_agreement_stats(self, task_id: str, epoch: int, window: int, phase: str) -> Dict[str, Any]:
        out = self.scripted[min(self.stats_calls, len(self.scripted) - 1)]
        self.stats_calls += 1
        return out

    def try_promote(self, task_id: str, epoch: int, rate: Any, samples: Any, reason: Any = None) -> bool:
        self.promotions.append(samples)
        return True


def _job(db: Any) -> Any:
    from paw_kit.jit.shadow import ShadowJob

    return ShadowJob(
        task_id="t", db_path="/tmp/j12.db", db=db, state_epoch=0, phase="shadow",
        input_payload="i", shadow_window=20, shadow_threshold=0.8, max_pairs=40,
    )


def test_a_window_short_at_its_boundary_is_retried_J_12() -> None:
    """The boundary is evaluated at or past its exact multiple, not only on it.

    A window that could not be scored at `seq == window` used to be discarded outright
    -- the next check was at `2 * window`, over a different set of comparisons -- so a
    single unscoreable boundary silently threw a whole window away.
    """
    from paw_kit.jit.shadow import ShadowRunner

    db = _ScriptedStats([
        {"rate": 1.0, "samples": 19, "seq": 20},   # short at the boundary
        {"rate": 1.0, "samples": 20, "seq": 21},   # complete one comparison later
    ])
    runner = ShadowRunner()
    job = _job(db)
    runner._maybe_transition(job, 20)
    assert db.promotions == [], "promoted on a window that was one sample short"
    runner._maybe_transition(job, 21)
    assert db.promotions == [20], (
        "the retried boundary was never evaluated -- the whole window was discarded"
    )


def test_a_scored_window_is_not_rescored_on_every_comparison_J_12() -> None:
    """Evaluating at-or-past a boundary must stay tumbling, not become sliding.

    A sliding window gives a below-threshold adapter a fresh draw on every single
    comparison and promotes it eventually, which is the exact failure shadow mode
    exists to prevent.

    Green at `main` by design: `seq % window == 0` does not re-score either. This is
    the anti-regression half of J-12's at-or-past-the-boundary change, whose positive
    half (`..._is_retried_J_12`) is red at main.
    """
    from paw_kit.jit.shadow import ShadowRunner

    db = _ScriptedStats([{"rate": 1.0, "samples": 20, "seq": 20}])
    runner = ShadowRunner()
    job = _job(db)
    runner._maybe_transition(job, 20)
    assert db.stats_calls == 1 and db.promotions == [20]
    for seq in range(21, 40):
        runner._maybe_transition(job, seq)
    assert db.stats_calls == 1, (
        f"the window was re-scored {db.stats_calls - 1} more time(s) before the next "
        "boundary -- that is a sliding window"
    )
    runner._maybe_transition(job, 40)
    assert db.stats_calls == 2, "the next completed window was not evaluated"
