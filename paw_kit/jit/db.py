"""Thread-safe SQLite tracing database for paw.jit."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

# INSERT ... ON CONFLICT ... DO UPDATE (used by record_trace) requires SQLite >= 3.24.
_MIN_SQLITE_VERSION = (3, 24, 0)

T = TypeVar("T")

# PAW-JIT-04: the 30s connection busy_timeout and WAL journal mode already in place
# (both from Track 03) reduce but don't eliminate multi-process write contention.
#
# D-1 (bug hunt 2026-09-11): `isolation_level="IMMEDIATE"` does NOT make a method's
# leading read part of its write transaction, which is what the comment that used to
# stand here asserted. Python's sqlite3 emits `BEGIN IMMEDIATE` only immediately
# before a *DML* statement, so a read-then-write method runs its SELECT in autocommit
# and opens the transaction on the following write -- leaving the classic lost-update
# window wide open across processes. Measured on this source: `record_shadow_pair`
# over 6 processes x 30 writes gave rows=180 with max(seq) 173-175 and up to 4
# duplicate seq values in 5 of 5 trials, and `set_status`'s epoch bumps over
# 4 processes x 40 changes gave 160 transitions sharing 40 distinct epochs in 5 of 5
# trials -- with `PRAGMA integrity_check` reading `ok` throughout, because this is
# logical corruption the file format cannot see.
#
# The fix is `_write_txn()` below: an explicit `BEGIN IMMEDIATE` issued before the
# leading read, factored into the same context manager that takes the instance lock so
# a new method cannot acquire one without the other. `isolation_level="IMMEDIATE"` is
# kept because it still covers every single-statement write correctly.
#
# The retry/backoff loop below is defense-in-depth on top of that, not a replacement.
_DB_RETRY_ATTEMPTS = 5
_DB_RETRY_BASE_DELAY_SECONDS = 0.05

# Track 14 (shadow mode), schema v2.
#
# `PRAGMA user_version` is a *forward marker only*. It reads 0 on every pre-v2
# traces.db AND on a brand-new one, so it is ambiguous and nothing branches v1->v2 on
# it -- the `PRAGMA table_info(tasks)` column probe in `_init_db` stays authoritative
# for deciding what to add. The write is a monotone bump (read, then set only if
# lower), never an unconditional set: unconditionally stamping 2 would let an older
# paw-kit roll a future v3 database's marker backwards, and the newer version would
# then re-run a migration it has already applied -- precisely the failure a forward
# marker exists to prevent. `PRAGMA user_version` takes no bind parameter, which is
# why this is an int module constant interpolated into the SQL and never
# caller-supplied.
_SCHEMA_VERSION = 2

# Per-task retention cap on `state_transitions`. `shadow_pairs`' cap is the caller's
# `shadow_max_pairs` (decorator.py), passed in per write.
_STATE_TRANSITIONS_MAX_ROWS = 200

# Prune every N inserts rather than on every one: the oldest-first delete below is
# index-covered but its subquery still walks `cap` index entries, and it runs while
# this class's single `self._lock` is held -- the same lock the caller's routing read
# and `record_trace` take. A cap at or below this interval is pruned on every insert
# instead (the subquery is then trivially small), which is what makes a deliberately
# tiny cap an exact bound rather than an approximate one. The worst-case row count for
# a table is therefore `cap + prune_interval - 1`.
_PRUNE_EVERY = 50

# Track 14 shadow-mode stall guard. Once a task has run this many multiples of a
# window's worth of comparisons at one epoch without promoting, `ShadowRunner`
# (paw_kit.jit.shadow) stops evaluating completed windows for promotion, and this
# module marks the task `stalled` in `get_task_report`/`get_agreement()` (finding 2).
# Canonical home: the window arithmetic (`get_agreement_stats`, `get_epoch_seq`)
# already lives here, and `shadow.py` imports the constant from this module instead of
# duplicating it.
_SHADOW_STALL_FACTOR = 5


def _prune_interval(cap: int) -> int:
    """How often to prune a table with this retention cap. See _PRUNE_EVERY."""
    return 1 if cap <= _PRUNE_EVERY else _PRUNE_EVERY


class TraceDB:
    """Embedded SQLite database tracking production API calls and compilation triggers."""

    def __init__(self, db_path: str = "./.paw/traces.db") -> None:
        if sqlite3.sqlite_version_info < _MIN_SQLITE_VERSION:
            raise RuntimeError(
                f"paw.jit requires SQLite >= {'.'.join(map(str, _MIN_SQLITE_VERSION))} "
                f"for upsert support (found {sqlite3.sqlite_version})."
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # PAW-JIT-01: traces.db holds raw, unredacted prompt/response text by default
        # (see decorator.py's `redact_trace`, PAW-JIT-02, for why redaction is not the
        # primary mitigation here) -- restrict the cache directory to owner-only access
        # rather than leaving it at the process umask's default (often world-readable).
        # mkdir(..., exist_ok=True) doesn't retroactively tighten an already-existing
        # directory's mode, and mkdir's own `mode` argument is subject to umask, so
        # chmod explicitly.
        self._chmod_best_effort(self.db_path.parent, 0o700)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
            # PAW-JIT-04: BEGIN IMMEDIATE instead of the default deferred BEGIN --
            # see the module-level comment on _DB_RETRY_ATTEMPTS.
            isolation_level="IMMEDIATE",
        )
        self._conn.row_factory = sqlite3.Row
        # PAW-JIT-05: callbacks invoked with `task_id` whenever set_status() writes a
        # new status/adapter_path for that task -- the same-process fast-path
        # invalidation hook the adapter-callable cache in jit/decorator.py uses.
        # decorator.py already imports this module, so the dependency has to run this
        # direction: whoever holds both a TraceDB and a cache to invalidate registers
        # itself via register_status_listener() rather than this module importing
        # decorator.py back.
        self._status_listeners: List[Callable[[str], None]] = []
        # Track 14: insert counters driving the periodic oldest-first prunes below.
        # Guarded by self._lock, like every other write on this connection.
        self._shadow_pair_writes = 0
        self._transition_writes = 0
        # Narrow the window the file spends at the (looser) default permissions
        # sqlite3.connect() just created it with, before any trace data is written.
        self._chmod_best_effort(self.db_path, 0o600)
        self._init_db()
        # WAL mode (enabled in _init_db) creates -wal/-shm sidecar files that also
        # contain trace data; they don't necessarily inherit the main file's mode.
        self._chmod_best_effort(self.db_path.with_name(self.db_path.name + "-wal"), 0o600)
        self._chmod_best_effort(self.db_path.with_name(self.db_path.name + "-shm"), 0o600)

    @staticmethod
    def _chmod_best_effort(path: Path, mode: int) -> None:
        """Restrict `path`'s permissions, tolerating platforms/paths that don't support it.

        Best-effort: a missing sidecar file (e.g. -wal/-shm not yet created) or a
        filesystem without POSIX permission bits (e.g. Windows, some network mounts)
        must not prevent the database from opening.
        """
        try:
            path.chmod(mode)
        except OSError:
            pass

    def register_status_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback invoked with `task_id` after every `set_status` write.

        PAW-JIT-05's same-process invalidation hook. Not the primary invalidation
        mechanism for the adapter-callable cache it exists for -- that's the cache
        key's own `os.stat` identity component -- but a backstop for a filesystem
        that reports `st_ino == 0` (some SMB/FUSE mounts) or a third-party backend
        that rewrites an adapter file in place rather than via `os.replace`.
        """
        with self._lock:
            self._status_listeners.append(listener)

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """One write transaction whose **leading read** is already inside it (D-1).

        Every read-modify-write method on this class must use this and nothing else.
        Acquiring `self._lock` and issuing `BEGIN IMMEDIATE` are deliberately the same
        gesture: a method that forgets the transaction cannot get the lock either, so
        the two can never drift apart the way they did before D-1.

        The `in_transaction` guard is load-bearing. A nested `BEGIN IMMEDIATE` raises
        `OperationalError("cannot start a transaction within a transaction")`, which
        `_with_write_retry` would catch and retry five times with exponential backoff
        before re-raising -- laundering a programming error into ~0.75s of latency and
        a lock-shaped message that says nothing about the real cause.

        **`_init_db` deliberately does not use this.** `PRAGMA journal_mode=WAL` issued
        inside an open transaction on a fresh database returns `"delete"` and leaves
        the file in rollback-journal mode *with no exception raised* (verified; on an
        already-populated database it raises instead, which is louder but no better).
        Wrapping `_init_db` -- whose column-probe-then-`ALTER TABLE` shape reads then
        writes, and so attracts any blanket rule -- would therefore silently destroy
        the WAL guarantee that this module's concurrency story rests on.
        """
        with self._lock, self._conn:
            if not self._conn.in_transaction:
                self._conn.execute("BEGIN IMMEDIATE;")
            yield self._conn

    def _with_write_retry(self, fn: Callable[[], T]) -> T:
        """Run `fn` (one write transaction) with bounded exponential-backoff retry on
        `sqlite3.OperationalError` (PAW-JIT-04) -- defense-in-depth on top of
        `isolation_level="IMMEDIATE"` and the connection's 30s busy_timeout, both of
        which reduce but don't eliminate multi-process write contention.
        """
        for attempt in range(_DB_RETRY_ATTEMPTS):
            try:
                return fn()
            except sqlite3.OperationalError:
                if attempt == _DB_RETRY_ATTEMPTS - 1:
                    raise
                time.sleep(_DB_RETRY_BASE_DELAY_SECONDS * (2**attempt))
        raise AssertionError("unreachable")  # pragma: no cover

    def _init_db(self) -> None:
        """Create tables and enable WAL mode for high concurrency."""
        # Track 14: schema v2 adds several ALTER TABLE statements and a
        # `PRAGMA user_version` write to what used to be two CREATE TABLE IF NOT
        # EXISTS calls. That is a longer write transaction, and several processes
        # opening the same traces.db at once (PAW-JIT-04's case) can now collide on
        # it -- so the migration goes through the same bounded retry/backoff wrapper
        # as every other write rather than raising OperationalError out of the
        # constructor.
        def _do() -> None:
            with self._lock, self._conn:
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        task_id TEXT PRIMARY KEY,
                        call_count INTEGER DEFAULT 0,
                        adapter_path TEXT,
                        status TEXT DEFAULT 'tracing',
                        compile_attempts INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                # PAW-JIT-03: migrate a pre-existing tasks table created before
                # compile_attempts existed (CREATE TABLE IF NOT EXISTS above is a no-op
                # against an already-existing table, it doesn't add new columns).
                existing_cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(tasks);")}
                if "compile_attempts" not in existing_cols:
                    self._conn.execute("ALTER TABLE tasks ADD COLUMN compile_attempts INTEGER DEFAULT 0;")
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS traces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        input_payload TEXT NOT NULL,
                        teacher_output TEXT NOT NULL,
                        latency_ms REAL NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                    );
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_traces_task_id ON traces(task_id);"
                )

                # --- Track 14 (shadow mode), schema v2 ----------------------------
                # Same `existing_cols` probe idiom as compile_attempts above: an
                # unconditional ALTER TABLE ADD COLUMN raises OperationalError on a
                # column that already exists, so a second open must be a no-op.
                for column, ddl in (
                    ("state_epoch", "state_epoch INTEGER DEFAULT 0"),
                    ("shadow_started_at", "shadow_started_at TEXT"),
                    ("promoted_at", "promoted_at TEXT"),
                    ("promoted_agreement", "promoted_agreement REAL"),
                    ("demoted_at", "demoted_at TEXT"),
                    ("demoted_agreement", "demoted_agreement REAL"),
                    # Persisted fail-open count. `decorator.py`'s `_FAIL_OPEN_COUNTS` is
                    # in-process and resets on restart, so `paw-kit report` (a different
                    # process entirely) could not otherwise show the figure the track's
                    # success criteria require of it.
                    ("fail_open_count", "fail_open_count INTEGER DEFAULT 0"),
                    # Resolved shadow configuration (JSON), written at decoration time.
                    # Two jobs: it is what lets `get_task_report` state the window a rate
                    # is measured over, and comparing it against the resolved config on
                    # open is what starts a fresh epoch when a caller changes
                    # `shadow_window`/`shadow_threshold` between runs instead of
                    # re-slicing an existing epoch's history under new arithmetic.
                    ("shadow_config", "shadow_config TEXT"),
                ):
                    if column not in existing_cols:
                        self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {ddl};")

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS shadow_pairs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        state_epoch INTEGER NOT NULL,
                        seq INTEGER NOT NULL,
                        phase TEXT NOT NULL,
                        input_payload TEXT NOT NULL,
                        teacher_output TEXT,
                        adapter_output TEXT,
                        verdict TEXT NOT NULL,
                        error_type TEXT,
                        teacher_latency_ms REAL,
                        adapter_latency_ms REAL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                    );
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_shadow_pairs_task_epoch "
                    "ON shadow_pairs(task_id, state_epoch, id);"
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS state_transitions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        task_id TEXT NOT NULL,
                        from_status TEXT NOT NULL,
                        to_status TEXT NOT NULL,
                        agreement REAL,
                        sample_count INTEGER,
                        state_epoch INTEGER NOT NULL,
                        reason TEXT,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                    );
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_state_transitions_task_id "
                    "ON state_transitions(task_id, id);"
                )

                # Forward marker only -- see _SCHEMA_VERSION. Do NOT branch v1->v2 on
                # this value; 0 is ambiguous between "old database" and "brand new one".
                current_version = self._conn.execute("PRAGMA user_version;").fetchone()[0]
                if current_version < _SCHEMA_VERSION:
                    self._conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION};")


        self._with_write_retry(_do)

    def record_trace(
        self,
        task_id: str,
        input_payload: str,
        teacher_output: str,
        latency_ms: float,
    ) -> int:
        """Record an API execution trace and atomically increment call count.

        Returns:
            The updated call count for the task.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> int:
            with self._write_txn():
                # 1. Upsert task record and increment call_count
                # Avoid RETURNING (requires SQLite >= 3.35): follow up with a plain SELECT instead.
                self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, status, created_at, updated_at)
                    VALUES (?, 1, 'tracing', ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        call_count = call_count + 1,
                        updated_at = excluded.updated_at;
                    """,
                    (task_id, now, now),
                )
                new_count = self._get_call_count_locked(task_id)

                # 2. Insert trace record
                self._conn.execute(
                    """
                    INSERT INTO traces (task_id, input_payload, teacher_output, latency_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (task_id, input_payload, teacher_output, latency_ms, now),
                )
                return new_count

        # PAW-JIT-04: bounded retry/backoff on top of BEGIN IMMEDIATE + busy_timeout.
        return self._with_write_retry(_do)

    def _get_call_count_locked(self, task_id: str) -> int:
        """Read call_count for task_id. Caller must already hold self._lock."""
        cur = self._conn.execute(
            "SELECT call_count FROM tasks WHERE task_id = ?;", (task_id,)
        )
        row = cur.fetchone()
        return row["call_count"] if row else 0

    def get_call_count(self, task_id: str) -> int:
        """Retrieve total calls recorded for task."""
        with self._lock:
            return self._get_call_count_locked(task_id)

    def get_status(self, task_id: str) -> str:
        """Retrieve task lifecycle status (tracing | compiling | ready | failed)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT status FROM tasks WHERE task_id = ?;", (task_id,)
            )
            row = cur.fetchone()
            return row["status"] if row else "tracing"

    def set_status(
        self,
        task_id: str,
        status: str,
        adapter_path: Optional[str] = None,
    ) -> None:
        """Update task lifecycle status and optional adapter path.

        PAW-JIT-05: fires every registered status listener with `task_id` after the
        write commits (not while `self._lock`/the SQLite transaction is held, so a
        listener can never deadlock against this method or another TraceDB call).

        Track 14: a status *change* also bumps `state_epoch` and writes a
        `state_transitions` row, so the lifecycle is auditable from the database
        alone. The epoch bump is what makes shadow-window arithmetic immune to
        comparisons that were in flight across a transition.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                changed = previous is not None and previous != status
                new_epoch = epoch + 1 if changed else epoch
                if adapter_path is not None:
                    self._conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, adapter_path = ?, state_epoch = ?, updated_at = ?
                        WHERE task_id = ?;
                        """,
                        (status, adapter_path, new_epoch, now, task_id),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE tasks
                        SET status = ?, state_epoch = ?, updated_at = ?
                        WHERE task_id = ?;
                        """,
                        (status, new_epoch, now, task_id),
                    )
                if changed:
                    self._record_transition_locked(
                        task_id, previous or "tracing", status, None, None, new_epoch, now
                    )

        # PAW-JIT-04: bounded retry/backoff on top of BEGIN IMMEDIATE + busy_timeout.
        self._with_write_retry(_do)

        for listener in list(self._status_listeners):
            listener(task_id)

    def get_compile_attempts(self, task_id: str) -> int:
        """Retrieve the number of compilation attempts made so far for task_id (PAW-JIT-03)."""
        with self._lock:
            return self._get_compile_attempts_locked(task_id)

    def _get_compile_attempts_locked(self, task_id: str) -> int:
        """Read compile_attempts for task_id. Caller must already hold self._lock."""
        cur = self._conn.execute(
            "SELECT compile_attempts FROM tasks WHERE task_id = ?;", (task_id,)
        )
        row = cur.fetchone()
        return row["compile_attempts"] if row else 0

    def increment_compile_attempts(self, task_id: str) -> int:
        """Atomically increment and return the compilation attempt counter for task_id.

        PAW-JIT-03: this is what lets BackgroundCompiler cap retries at a fixed number
        instead of either deadlocking in "failed" forever (the pre-fix bug) or retrying
        unconditionally on every subsequent call once the task's (never-decreasing)
        call_count has crossed the compilation threshold (the audit's own suggested
        fix, which turns into an unbounded retry loop).
        """
        def _do() -> int:
            with self._write_txn():
                self._conn.execute(
                    "UPDATE tasks SET compile_attempts = compile_attempts + 1 WHERE task_id = ?;",
                    (task_id,),
                )
                return self._get_compile_attempts_locked(task_id)

        # PAW-JIT-04: bounded retry/backoff on top of BEGIN IMMEDIATE + busy_timeout.
        return self._with_write_retry(_do)

    def get_adapter_path(self, task_id: str) -> Optional[str]:
        """Retrieve path to compiled adapter if task is ready."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT adapter_path, status FROM tasks WHERE task_id = ?;", (task_id,)
            )
            row = cur.fetchone()
            if row and row["status"] == "ready":
                return row["adapter_path"]
            return None

    def get_traces(
        self,
        task_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve recorded traces for training example generation."""
        with self._lock:
            query = "SELECT input_payload, teacher_output, latency_ms, timestamp FROM traces WHERE task_id = ? ORDER BY id ASC"
            params: List[Any] = [task_id]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            cur = self._conn.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    # --- Track 14: shadow mode ---------------------------------------------------

    def _get_status_and_epoch_locked(self, task_id: str) -> Tuple[Optional[str], int]:
        """Read (status, state_epoch) for task_id, or (None, 0) if there is no row.

        Caller must already hold self._lock.
        """
        cur = self._conn.execute(
            "SELECT status, state_epoch FROM tasks WHERE task_id = ?;", (task_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None, 0
        return row["status"], (row["state_epoch"] or 0)

    def get_task_routing(self, task_id: str) -> Tuple[str, Optional[str], int]:
        """One SELECT returning everything the decorator needs to route a call.

        Returns `(status, adapter_path, state_epoch)`. `adapter_path` is returned
        **ungated by status** -- that is what lets the `shadow` branch reach the
        compiled adapter while the teacher keeps serving; `get_adapter_path` keeps its
        `status == 'ready'` gate and its existing meaning. A task with no row reads
        `("tracing", None, 0)`, matching `get_status`'s default.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT status, adapter_path, state_epoch FROM tasks WHERE task_id = ?;",
                (task_id,),
            )
            row = cur.fetchone()
            if row is None:
                return ("tracing", None, 0)
            return (row["status"], row["adapter_path"], row["state_epoch"] or 0)

    def _record_transition_locked(
        self,
        task_id: str,
        from_status: str,
        to_status: str,
        agreement: Optional[float],
        sample_count: Optional[int],
        state_epoch: int,
        now: str,
        reason: Optional[str] = None,
    ) -> None:
        """Insert a lifecycle transition row. Caller must already hold self._lock."""
        self._conn.execute(
            """
            INSERT INTO state_transitions
                (task_id, from_status, to_status, agreement, sample_count,
                 state_epoch, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (task_id, from_status, to_status, agreement, sample_count, state_epoch, reason, now),
        )
        self._transition_writes += 1
        interval = _prune_interval(_STATE_TRANSITIONS_MAX_ROWS)
        if self._transition_writes % interval == 0:
            self._prune_transitions_locked(task_id)

    def record_state_transition(
        self,
        task_id: str,
        from_status: str,
        to_status: str,
        agreement: Optional[float] = None,
        sample_count: Optional[int] = None,
        state_epoch: int = 0,
        reason: Optional[str] = None,
    ) -> None:
        """Public wrapper around `_record_transition_locked` (used by tests and tooling)."""
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                self._record_transition_locked(
                    task_id, from_status, to_status, agreement, sample_count,
                    state_epoch, now, reason,
                )

        self._with_write_retry(_do)

    def set_shadow_started(self, task_id: str, adapter_path: str) -> None:
        """Move a freshly-compiled task into `shadow`: the adapter exists but does not serve."""
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                new_epoch = epoch + 1
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'shadow', adapter_path = ?, shadow_started_at = ?,
                        state_epoch = ?, updated_at = ?
                    WHERE task_id = ?;
                    """,
                    (adapter_path, now, new_epoch, now, task_id),
                )
                self._record_transition_locked(
                    task_id, previous or "compiling", "shadow", None, None, new_epoch, now
                )

        self._with_write_retry(_do)
        for listener in list(self._status_listeners):
            listener(task_id)

    def set_ready_from_compile(self, task_id: str, adapter_path: str) -> None:
        """Promote straight to `ready` on compile -- the `shadow_window=0` path."""
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                new_epoch = epoch + 1
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'ready', adapter_path = ?, promoted_at = ?,
                        promoted_agreement = NULL, state_epoch = ?, updated_at = ?
                    WHERE task_id = ?;
                    """,
                    (adapter_path, now, new_epoch, now, task_id),
                )
                self._record_transition_locked(
                    task_id, previous or "compiling", "ready", None, None, new_epoch, now,
                    reason="shadow_disabled",
                )

        self._with_write_retry(_do)
        for listener in list(self._status_listeners):
            listener(task_id)

    def try_promote(
        self,
        task_id: str,
        expected_epoch: int,
        agreement: Optional[float],
        samples: Optional[int],
        reason: Optional[str] = None,
    ) -> bool:
        """Compare-and-set `shadow` -> `ready`. Returns False if another writer won.

        The `WHERE status='shadow' AND state_epoch=?` clause is what makes promotion
        idempotent across threads *and* processes sharing one traces.db: exactly one
        writer sees `rowcount == 1`, every loser no-ops.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> bool:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'ready', state_epoch = state_epoch + 1,
                        promoted_at = ?, promoted_agreement = ?, updated_at = ?
                    WHERE task_id = ? AND status = 'shadow' AND state_epoch = ?;
                    """,
                    (now, agreement, now, task_id, expected_epoch),
                )
                if cur.rowcount == 0:
                    return False
                self._record_transition_locked(
                    task_id, "shadow", "ready", agreement, samples,
                    expected_epoch + 1, now, reason,
                )
                return True

        won = self._with_write_retry(_do)
        if won:
            for listener in list(self._status_listeners):
                listener(task_id)
        return won

    def try_demote(
        self,
        task_id: str,
        expected_epoch: int,
        agreement: Optional[float],
        samples: Optional[int],
        reason: Optional[str] = None,
    ) -> bool:
        """Compare-and-set `ready` -> `shadow`. Symmetric with `try_promote`."""
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> bool:
            with self._lock, self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = 'shadow', state_epoch = state_epoch + 1,
                        demoted_at = ?, demoted_agreement = ?, updated_at = ?
                    WHERE task_id = ? AND status = 'ready' AND state_epoch = ?;
                    """,
                    (now, agreement, now, task_id, expected_epoch),
                )
                if cur.rowcount == 0:
                    return False
                self._record_transition_locked(
                    task_id, "ready", "shadow", agreement, samples,
                    expected_epoch + 1, now, reason,
                )
                return True

        won = self._with_write_retry(_do)
        if won:
            for listener in list(self._status_listeners):
                listener(task_id)
        return won

    def sync_shadow_config(self, task_id: str, config: Dict[str, Any]) -> bool:
        """Persist the resolved shadow configuration, starting a fresh epoch if it changed.

        Config lives in the decorator and state lives here, and nothing else
        reconciles them across runs: lowering `shadow_window` from 20 to 5 between
        runs would otherwise re-slice an existing epoch's history under new
        arithmetic and could promote a task the previous configuration was correctly
        refusing. Bumping the epoch on a config change makes the next window a fresh,
        honest one. Returns True if the epoch was bumped.

        Finding 6: this bumps for `status in ('shadow', 'ready')`, not `'shadow'`
        alone -- `audit_window`/`demote_threshold` re-slice a `ready` task's audit
        history exactly the same way `shadow_window`/`shadow_threshold` re-slice a
        `shadow` task's, and a `ready` task changing config out from under a running
        audit was unreachable only because the shipped default is `audit_rate=0.0`.
        """
        now = datetime.now(timezone.utc).isoformat()
        encoded = json.dumps(config, sort_keys=True)

        def _do() -> bool:
            with self._write_txn():
                cur = self._conn.execute(
                    "SELECT status, state_epoch, shadow_config FROM tasks WHERE task_id = ?;",
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    self._conn.execute(
                        """
                        INSERT INTO tasks (task_id, call_count, status, shadow_config,
                                           created_at, updated_at)
                        VALUES (?, 0, 'tracing', ?, ?, ?)
                        ON CONFLICT(task_id) DO UPDATE SET
                            shadow_config = excluded.shadow_config,
                            updated_at = excluded.updated_at;
                        """,
                        (task_id, encoded, now, now),
                    )
                    return False
                if row["shadow_config"] == encoded:
                    return False
                changed_mid_run = (
                    row["shadow_config"] is not None
                    and row["status"] in ("shadow", "ready")
                )
                epoch = (row["state_epoch"] or 0) + (1 if changed_mid_run else 0)
                self._conn.execute(
                    "UPDATE tasks SET shadow_config = ?, state_epoch = ?, updated_at = ? "
                    "WHERE task_id = ?;",
                    (encoded, epoch, now, task_id),
                )
                if changed_mid_run:
                    self._record_transition_locked(
                        task_id, row["status"], row["status"], None, None, epoch, now,
                        reason="config_change",
                    )
                return changed_mid_run

        return self._with_write_retry(_do)

    def get_shadow_config(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Read back the persisted resolved shadow configuration, if any."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT shadow_config FROM tasks WHERE task_id = ?;", (task_id,)
            )
            row = cur.fetchone()
        if row is None or not row["shadow_config"]:
            return None
        try:
            return json.loads(row["shadow_config"])
        except ValueError:
            return None

    def increment_fail_open(self, task_id: str) -> None:
        """Bump the persisted fail-open counter (`paw-kit report`'s Fail-open column).

        The in-process counter in `decorator.py` resets on restart and is invisible to
        any other process; this one is neither.
        """
        def _do() -> None:
            with self._write_txn():
                self._conn.execute(
                    "UPDATE tasks SET fail_open_count = COALESCE(fail_open_count, 0) + 1 "
                    "WHERE task_id = ?;",
                    (task_id,),
                )

        self._with_write_retry(_do)

    def record_shadow_pair(
        self,
        task_id: str,
        state_epoch: int,
        phase: str,
        input_payload: str,
        teacher_output: Optional[str],
        adapter_output: Optional[str],
        verdict: str,
        error_type: Optional[str] = None,
        teacher_latency_ms: Optional[float] = None,
        adapter_latency_ms: Optional[float] = None,
        max_pairs: int = 500,
    ) -> Dict[str, int]:
        """Persist one teacher/adapter comparison and return its per-epoch sequence number.

        `seq` is a monotone counter over *countable* comparisons at `(task_id,
        state_epoch)`, assigned in the same transaction as the insert. It is what
        makes the promotion window **tumbling** rather than sliding: a transition is
        evaluated once per completed window (`seq % window == 0`), not on every
        comparison. A `COUNT(*)`-based trigger cannot be used, because retention
        pruning deletes rows and would make it misfire and repeat.

        A `teacher_error` row is not a countable comparison (the adapter is not at
        fault when the audit's teacher call raises), so it is stored with `seq = 0`
        and never advances the window.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> Dict[str, int]:
            with self._write_txn():
                if verdict == "teacher_error":
                    seq = 0
                else:
                    cur = self._conn.execute(
                        "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM shadow_pairs "
                        "WHERE task_id = ? AND state_epoch = ?;",
                        (task_id, state_epoch),
                    )
                    seq = cur.fetchone()["max_seq"] + 1
                self._conn.execute(
                    """
                    INSERT INTO shadow_pairs
                        (task_id, state_epoch, seq, phase, input_payload, teacher_output,
                         adapter_output, verdict, error_type, teacher_latency_ms,
                         adapter_latency_ms, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        task_id, state_epoch, seq, phase, input_payload, teacher_output,
                        adapter_output, verdict, error_type, teacher_latency_ms,
                        adapter_latency_ms, now,
                    ),
                )
                self._shadow_pair_writes += 1
                if self._shadow_pair_writes % _prune_interval(max_pairs) == 0:
                    self._prune_shadow_pairs_locked(task_id, max_pairs)
                return {"seq": seq}

        return self._with_write_retry(_do)

    def get_epoch_seq(self, task_id: str, state_epoch: int) -> int:
        """Highest countable-comparison sequence number reached at this epoch.

        Survives pruning (unlike `COUNT(*)`), which is what the stall guard needs.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM shadow_pairs "
                "WHERE task_id = ? AND state_epoch = ?;",
                (task_id, state_epoch),
            )
            return cur.fetchone()["max_seq"]

    def get_agreement_stats(
        self, task_id: str, state_epoch: int, window: int, phase: str = "shadow"
    ) -> Dict[str, Any]:
        """Counts over the newest `window` *countable* comparisons at this epoch.

        The `verdict != 'teacher_error'` filter sits **inside** the `LIMIT` subquery,
        not outside it. Outside, teacher-error rows would still spend the window's
        LIMIT budget, so `samples` could never reach `audit_window` once one landed in
        the trailing window -- and `samples == audit_window` is the demotion trigger.
        One flaky teacher call would disable drift detection for that task forever.

        Finding 5: `teacher_error` is scoped to that *same* trailing window, not the
        whole epoch -- otherwise it grows unbounded next to a `window`-sized rate.
        Simplest implementation: teacher-error rows are not countable and carry no
        `seq` of their own (they are stored with `seq = 0`), so the boundary is the
        `id` of the oldest of the `window` countable rows above, and a teacher-error
        row counts if it is at least that recent -- i.e. it is interleaved with the
        current window. With fewer than `window` countable rows so far this epoch, the
        boundary is simply the first one recorded, so nothing is double-counted or
        missed; with none at all yet, nothing has started, and it reads 0.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT verdict, COUNT(*) AS n FROM (
                    SELECT verdict FROM shadow_pairs
                    WHERE task_id = ? AND state_epoch = ? AND phase = ?
                      AND verdict != 'teacher_error'
                    ORDER BY id DESC LIMIT ?
                ) GROUP BY verdict;
                """,
                (task_id, state_epoch, phase, window),
            ).fetchall()
            boundary_id = self._conn.execute(
                """
                SELECT MIN(id) AS boundary_id FROM (
                    SELECT id FROM shadow_pairs
                    WHERE task_id = ? AND state_epoch = ? AND phase = ?
                      AND verdict != 'teacher_error'
                    ORDER BY id DESC LIMIT ?
                );
                """,
                (task_id, state_epoch, phase, window),
            ).fetchone()["boundary_id"]
            if boundary_id is None:
                teacher_errors = 0
            else:
                teacher_errors = self._conn.execute(
                    "SELECT COUNT(*) AS n FROM shadow_pairs "
                    "WHERE task_id = ? AND state_epoch = ? AND phase = ? "
                    "AND verdict = 'teacher_error' AND id >= ?;",
                    (task_id, state_epoch, phase, boundary_id),
                ).fetchone()["n"]
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM shadow_pairs "
                "WHERE task_id = ? AND state_epoch = ?;",
                (task_id, state_epoch),
            ).fetchone()["max_seq"]

        counts = {row["verdict"]: row["n"] for row in rows}
        agree = counts.get("agree", 0)
        disagree = counts.get("disagree", 0)
        error = counts.get("error", 0)
        samples = agree + disagree + error
        return {
            "phase": phase,
            "window": window,
            "samples": samples,
            "agree": agree,
            "disagree": disagree,
            # An adapter that throws is at least as unfit to serve as one that answers
            # wrongly, so an error counts against the promotion denominator.
            "error": error,
            "teacher_error": teacher_errors,
            "rate": (agree / samples) if samples else None,
            "seq": seq,
        }

    def get_recent_disagreements(self, task_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Newest-first tail of `verdict in ('disagree', 'error')` rows for a task.

        Returns the *persisted* (therefore possibly redacted, see `redact_trace`) text.
        """
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT input_payload, teacher_output, adapter_output, verdict,
                       error_type, phase, timestamp
                FROM shadow_pairs
                WHERE task_id = ? AND verdict IN ('disagree', 'error')
                ORDER BY id DESC LIMIT ?;
                """,
                (task_id, limit),
            )
            return [dict(row) for row in cur.fetchall()]

    def list_task_ids(self) -> List[str]:
        """Every task_id known to this database, oldest first."""
        with self._lock:
            cur = self._conn.execute("SELECT task_id FROM tasks ORDER BY created_at ASC;")
            return [row["task_id"] for row in cur.fetchall()]

    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """Rich per-task view for `wrapper.get_agreement()` and `paw-kit report`.

        `get_status` deliberately keeps its bare-`str` signature and return; this is
        the additive rich API rather than a change to it.
        """
        with self._lock:
            cur = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?;", (task_id,))
            row = cur.fetchone()
        if row is None:
            base: Dict[str, Any] = {
                "task_id": task_id, "status": "tracing", "adapter_path": None,
                "call_count": 0, "compile_attempts": 0, "state_epoch": 0,
                "created_at": None, "updated_at": None, "shadow_started_at": None,
                "promoted_at": None, "promoted_agreement": None,
                "demoted_at": None, "demoted_agreement": None, "fail_open_count": 0,
            }
        else:
            record = dict(row)
            base = {
                "task_id": task_id,
                "status": record.get("status") or "tracing",
                "adapter_path": record.get("adapter_path"),
                "call_count": record.get("call_count") or 0,
                "compile_attempts": record.get("compile_attempts") or 0,
                "state_epoch": record.get("state_epoch") or 0,
                "created_at": record.get("created_at"),
                "updated_at": record.get("updated_at"),
                "shadow_started_at": record.get("shadow_started_at"),
                "promoted_at": record.get("promoted_at"),
                "promoted_agreement": record.get("promoted_agreement"),
                "demoted_at": record.get("demoted_at"),
                "demoted_agreement": record.get("demoted_agreement"),
                "fail_open_count": record.get("fail_open_count") or 0,
            }

        config = self.get_shadow_config(task_id) or {}
        status = base["status"]
        if status == "shadow":
            phase: Optional[str] = "shadow"
            window = int(config.get("shadow_window") or 0)
        elif status == "ready":
            phase = "audit"
            window = int(config.get("audit_window") or 0)
        else:
            phase = None
            window = 0

        if phase is None:
            agreement = {
                "phase": None, "rate": None, "window": 0, "samples": 0,
                "agree": 0, "disagree": 0, "error": 0, "teacher_error": 0,
                "stalled": False,
            }
        else:
            stats = self.get_agreement_stats(task_id, base["state_epoch"], window, phase)
            # Finding 2: `stalled` mirrors ShadowRunner._maybe_transition's own "past
            # the stall point" check exactly (`seq > _SHADOW_STALL_FACTOR * window`,
            # `shadow` phase only -- `audit` never stalls, there is no audit subsample
            # guard) so that a task the runner has stopped evaluating for promotion at
            # this epoch does not read, from `get_agreement()`/`paw-kit report` alone,
            # like one still converging.
            stalled = (
                phase == "shadow" and window > 0
                and stats["seq"] > _SHADOW_STALL_FACTOR * window
            )
            agreement = {
                "phase": phase, "rate": stats["rate"], "window": window,
                "samples": stats["samples"], "agree": stats["agree"],
                "disagree": stats["disagree"], "error": stats["error"],
                "teacher_error": stats["teacher_error"],
                "stalled": stalled,
            }
        base["agreement"] = agreement
        return base

    def _prune_shadow_pairs_locked(self, task_id: str, max_pairs: int) -> None:
        """Oldest-first retention on `shadow_pairs`. Caller must hold self._lock.

        An id-threshold delete rather than `id NOT IN (SELECT ... LIMIT ?)`: the
        subquery form materialises up to `cap` ids and re-scans them per row, and this
        runs while the single connection lock the caller's routing read also takes is
        held.
        """
        self._conn.execute(
            """
            DELETE FROM shadow_pairs
            WHERE task_id = ?
              AND id < (SELECT MIN(id) FROM (
                    SELECT id FROM shadow_pairs WHERE task_id = ? ORDER BY id DESC LIMIT ?
                  ));
            """,
            (task_id, task_id, max_pairs),
        )

    def _prune_transitions_locked(self, task_id: str) -> None:
        """Oldest-first retention on `state_transitions`. Caller must hold self._lock."""
        self._conn.execute(
            """
            DELETE FROM state_transitions
            WHERE task_id = ?
              AND id < (SELECT MIN(id) FROM (
                    SELECT id FROM state_transitions WHERE task_id = ? ORDER BY id DESC LIMIT ?
                  ));
            """,
            (task_id, task_id, _STATE_TRANSITIONS_MAX_ROWS),
        )

    def close(self) -> None:
        """Close SQLite connection."""
        with self._lock:
            self._conn.close()
