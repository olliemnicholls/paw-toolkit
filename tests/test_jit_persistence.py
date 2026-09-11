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
