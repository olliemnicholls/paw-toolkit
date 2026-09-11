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
