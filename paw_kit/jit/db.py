"""Thread-safe SQLite tracing database for paw.jit."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3
import stat
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

logger = logging.getLogger("paw_kit.jit.db")

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

# Schema v3. v2 was Track 14 (shadow mode); v3 adds `tasks.compiling_started_at`, the
# compile lease stamp J-5 needs to tell a running compile apart from a wedged one.
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
_SCHEMA_VERSION = 3

# Per-task retention cap on `state_transitions`. `shadow_pairs`' cap is the caller's
# `shadow_max_pairs` (decorator.py), passed in per write.
_STATE_TRANSITIONS_MAX_ROWS = 200

# Prune every N inserts rather than on every one: the oldest-first delete below is
# index-covered but its subquery still walks `cap` index entries, and (since D-5) it
# runs in its own short write transaction, which still contends for SQLite's single
# writer with the caller's `record_trace`. A cap at or below this interval is pruned
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
        # D-6: probed *before* the mkdir. `Path.mkdir(exist_ok=True)` returns None
        # either way, so "did this call create the directory" cannot be inferred from
        # it -- and the CWD/home carve-out below must not apply to a directory this
        # constructor made itself.
        parent_existed = self.db_path.parent.exists()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # PAW-JIT-01: traces.db holds raw, unredacted prompt/response text by default
        # (see decorator.py's `redact_trace`, PAW-JIT-02, for why redaction is not the
        # primary mitigation here) -- restrict the cache directory to owner-only access
        # rather than leaving it at the process umask's default (often world-readable).
        # mkdir(..., exist_ok=True) doesn't retroactively tighten an already-existing
        # directory's mode, and mkdir's own `mode` argument is subject to umask, so
        # chmod explicitly.
        self._tighten_dir_best_effort(self.db_path.parent, parent_existed)
        # J-7: `self._lock` no longer serialises database access -- it guards
        # `_status_listeners` and nothing else. See `_connection` below.
        self._lock = threading.Lock()
        # One SQLite connection per thread (J-7), all recorded here so `close()` can
        # close every one of them.
        self._local = threading.local()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False
        # Connect eagerly for the constructing thread: the `chmod(0o600)` below must
        # find the database file already created (PAW-JIT-01), and a read-only parent
        # directory must still raise out of the constructor rather than at first use.
        self._connect()
        # PAW-JIT-05: callbacks invoked with `task_id` whenever set_status() writes a
        # new status/adapter_path for that task -- the same-process fast-path
        # invalidation hook the adapter-callable cache in jit/decorator.py uses.
        # decorator.py already imports this module, so the dependency has to run this
        # direction: whoever holds both a TraceDB and a cache to invalidate registers
        # itself via register_status_listener() rather than this module importing
        # decorator.py back.
        self._status_listeners: List[Callable[[str], None]] = []
        # Track 14: insert counters driving the periodic oldest-first prunes below.
        #
        # D-5: keyed **per task**, and bumped **outside** the closure
        # `_with_write_retry` re-runs. One counter across all tasks gated a prune that
        # deletes rows for a single task, so with T tasks any given task was pruned at
        # most 1/T of the time -- and under round-robin traffic the Nth call lands on
        # the same task every time, so in practice exactly one task was ever pruned
        # (50 tasks held 600 rows each against a documented worst case of 249).
        # Bumping inside the retried closure additionally advanced the clock twice for
        # a single persisted row whenever a transaction lost the race and was retried.
        self._shadow_pair_writes: Dict[str, int] = {}
        self._transition_writes: Dict[str, int] = {}
        # Deliberately not `self._lock`: these are bumped outside the write
        # transaction, and reusing the connection lock for them would pull unrelated
        # bookkeeping back inside the critical section D-1 just widened.
        self._counter_lock = threading.Lock()
        # Narrow the window the file spends at the (looser) default permissions
        # sqlite3.connect() just created it with, before any trace data is written.
        self._chmod_best_effort(self.db_path, 0o600)
        self._init_db()
        # WAL mode (enabled in _init_db) creates -wal/-shm sidecar files that also
        # contain trace data; they don't necessarily inherit the main file's mode.
        self._chmod_best_effort(self.db_path.with_name(self.db_path.name + "-wal"), 0o600)
        self._chmod_best_effort(self.db_path.with_name(self.db_path.name + "-shm"), 0o600)

    def _connect(self) -> sqlite3.Connection:
        """Open this thread's connection and register it for `close()` (J-7)."""
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
            # PAW-JIT-04: BEGIN IMMEDIATE instead of the default deferred BEGIN --
            # see the module-level comment on _DB_RETRY_ATTEMPTS. Single-statement
            # writes rely on it; read-modify-write goes through `_write_txn`.
            isolation_level="IMMEDIATE",
        )
        conn.row_factory = sqlite3.Row
        with self._conns_lock:
            if self._closed:
                conn.close()
                raise sqlite3.ProgrammingError(
                    "Cannot operate on a closed database."
                )
            self._conns.append(conn)
        self._local.conn = conn
        return conn

    @property
    def _conn(self) -> sqlite3.Connection:
        """This thread's SQLite connection, opened on first use (J-7).

        Shadow mode roughly doubled the decorator's caller latency, contradicting
        "Nothing in shadow mode adds latency to it": the comparison work is off the
        request path but its *persistence* was not, because the shadow worker and the
        caller shared one connection behind one lock and therefore contended for both.
        WAL supports one connection per thread, so they no longer do.

        This is a **thread-local connection inside one `TraceDB`**, not a second
        `TraceDB` per worker. A second instance would carry its own empty
        `_status_listeners` list, silently stopping `_invalidate_adapter_cache` from
        firing on the worker's `try_promote`/`try_demote` -- a behaviour change no
        finding asked for.

        What now makes concurrent access safe is D-1's `_write_txn` (an explicit
        `BEGIN IMMEDIATE`) plus the 30s busy_timeout, exactly as it already was across
        processes. Before D-1, giving the worker its own connection would have
        reintroduced D-1 *inside* a single process: the shared lock was what made the
        racy read-then-write accidentally safe in-process.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
        return conn

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

    @staticmethod
    def _tighten_dir_best_effort(parent: Path, existed: bool) -> None:
        """Clear group and other bits on the cache directory. Never set an absolute mode.

        D-6: this used to be `chmod(0o700)`, which *sets* rather than tightens. A
        deliberately group-shared `0o2775` directory was reduced to `0o700` and lost
        its setgid bit, and a read-only `0o500` directory was silently made writable --
        so merely importing a module that constructs a `TraceDB` rewrote permissions on
        a directory the user shares with a group. Worse, `TraceDB("traces.db")` has
        `Path(...).parent == Path(".")`, so it chmodded the process working directory.

        `stat.S_IMODE(...) & ~0o077` can only ever clear bits, which is what makes
        every case fall out without a special case: `0o2775` -> `0o2700` (setgid kept),
        `0o1777` -> `0o1700` (sticky kept), `0o755` -> `0o700`, `0o644` -> `0o600`, and
        `0o500` -> `0o500`, still not writable.

        The carve-out is deliberately narrow -- warn and skip *only* when the resolved
        parent is the process CWD or the user's home directory. Anything broader (e.g.
        "skip a directory the caller didn't create") would stop `TraceDB` tightening a
        pre-existing `.paw`, which is behaviour this class correctly provides.
        """
        try:
            current = stat.S_IMODE(parent.stat().st_mode)
        except OSError:
            return
        target = current & ~0o077
        if target == current:
            # Already closed. An optimisation (fewer syscalls), not a correctness
            # requirement -- the chmod below would be a no-op anyway.
            return
        if existed:
            try:
                resolved = parent.resolve()
                landmarks = {Path.cwd().resolve(), Path.home().resolve()}
            except (OSError, RuntimeError):  # pragma: no cover - no cwd, no HOME
                resolved, landmarks = None, set()
            if resolved is not None and resolved in landmarks:
                logger.warning(
                    "paw_kit.jit: refusing to tighten permissions on %s -- it is the "
                    "process working directory or your home directory, not a paw cache "
                    "directory. traces.db holds raw prompt and response text; pass a "
                    "cache_dir (e.g. './.paw') so it lives somewhere this library may "
                    "restrict to owner-only access. Current mode %s.",
                    resolved, oct(current),
                )
                return
        try:
            parent.chmod(target)
        except OSError:
            pass

    def register_status_listener(self, listener: Callable[[str], None]) -> None:
        """Register a callback invoked with `task_id` after every `set_status` write.

        PAW-JIT-05's same-process invalidation hook. Not the primary invalidation
        mechanism for the adapter-callable cache it exists for -- that's the cache
        key's own `os.stat` identity component -- but a backstop for a filesystem
        that reports `st_ino == 0` (some SMB/FUSE mounts) or a third-party backend
        that rewrites an adapter file in place rather than via `os.replace`.

        J-7: `self._lock` guards this list and nothing else now -- database access is
        serialised by SQLite across per-thread connections, not by this lock.
        """
        with self._lock:
            self._status_listeners.append(listener)

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """One write transaction whose **leading read** is already inside it (D-1).

        Every read-modify-write method on this class must use this and nothing else.
        It is the only place `BEGIN IMMEDIATE` is issued, so a method either goes
        through it and is correct or does not and is visibly not a transaction at all
        -- there is no third shape to get subtly wrong, which is how D-1 survived.

        Since J-7 the connection is per thread, so this no longer serialises writers
        in-process: SQLite does, exactly as it already did across processes. That is
        why D-1 had to land first.

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
        conn = self._conn
        with conn:
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE;")
            yield conn

    def _note_write(self, counter: Dict[str, int], task_id: str, interval: int) -> bool:
        """Bump a per-task prune clock and report whether a prune is now due (D-5).

        Called *after* `_with_write_retry` returns, never from inside the closure it
        re-runs: one persisted row must advance the clock exactly once however many
        attempts it took.
        """
        with self._counter_lock:
            count = counter.get(task_id, 0) + 1
            counter[task_id] = count
        return count % interval == 0

    def _after_transition_write(self, task_id: str) -> None:
        """Advance the `state_transitions` prune clock for `task_id` and prune if due."""
        if self._note_write(
            self._transition_writes, task_id, _prune_interval(_STATE_TRANSITIONS_MAX_ROWS)
        ):
            def _do() -> None:
                with self._write_txn():
                    self._prune_transitions_locked(task_id)

            self._with_write_retry(_do)

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
            # NB: `with self._conn` and deliberately NOT `_write_txn` -- see its
            # docstring. `PRAGMA journal_mode=WAL` inside an open transaction on a
            # fresh database silently leaves the file in rollback-journal mode.
            with self._conn:
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
                    # --- schema v3 (J-5) --------------------------------------
                    # When the current `compiling` lease began. Its whole job is to
                    # let `reclaim_stale_compile` tell a compile that is *running*
                    # from one whose process was killed: without it, a task wedged in
                    # `compiling` is indistinguishable from a healthy one and the
                    # teacher is paid on every call forever.
                    ("compiling_started_at", "compiling_started_at TEXT"),
                ):
                    if column not in existing_cols:
                        self._conn.execute(f"ALTER TABLE tasks ADD COLUMN {ddl};")
                        if column == "compiling_started_at":
                            # Backfill, because the databases that actually have J-5's
                            # bug are the ones that predate this column. The reclaim
                            # deliberately requires a stamp -- a `compiling` row of
                            # unknown age must never be yanked out from under a live
                            # compile -- so without this an already-wedged task would
                            # stay wedged forever after the upgrade. `updated_at` is
                            # when the status was last written, i.e. when the lease
                            # began, which is exactly the value wanted here.
                            self._conn.execute(
                                "UPDATE tasks SET compiling_started_at = updated_at "
                                "WHERE status = 'compiling' AND compiling_started_at IS NULL;"
                            )

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
        """Read call_count for task_id. Runs on this thread's connection."""
        cur = self._conn.execute(
            "SELECT call_count FROM tasks WHERE task_id = ?;", (task_id,)
        )
        row = cur.fetchone()
        return row["call_count"] if row else 0

    def get_call_count(self, task_id: str) -> int:
        """Retrieve total calls recorded for task."""
        return self._get_call_count_locked(task_id)

    def get_status(self, task_id: str) -> str:
        """Retrieve task lifecycle status (tracing | compiling | shadow | ready | failed)."""
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
        write commits (never while the SQLite transaction is held, so a listener can
        never deadlock against this method or another TraceDB call).

        Track 14: a status *change* also bumps `state_epoch` and writes a
        `state_transitions` row, so the lifecycle is auditable from the database
        alone. The epoch bump is what makes shadow-window arithmetic immune to
        comparisons that were in flight across a transition.

        D-7: upsert, not a bare `UPDATE ... WHERE task_id = ?`. Against a missing
        `tasks` row the bare form affected zero rows and reported success, so
        `set_status(..., 'failed')` silently did nothing and the terminal state was
        unreachable -- reachable in practice by deleting `.paw/traces.db` (documented
        as a cache) while a process holds a live wrapper.
        """
        now = datetime.now(timezone.utc).isoformat()
        # J-5: every status writer maintains the compile lease, so the invariant
        # "status == 'compiling' implies compiling_started_at is set" has no holes --
        # and leaving `compiling` clears the stamp, so a later wedge cannot inherit an
        # old one. Without this, a direct `set_status(task_id, 'compiling')` would
        # produce exactly the NULL-stamp row the reclaim cannot act on.
        lease = now if status == "compiling" else None

        def _do() -> bool:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                changed = previous is not None and previous != status
                new_epoch = epoch + 1 if changed else epoch
                if adapter_path is not None:
                    self._conn.execute(
                        """
                        INSERT INTO tasks (task_id, call_count, status, adapter_path,
                                           state_epoch, compiling_started_at,
                                           created_at, updated_at)
                        VALUES (?, 0, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(task_id) DO UPDATE SET
                            status = excluded.status,
                            adapter_path = excluded.adapter_path,
                            state_epoch = excluded.state_epoch,
                            compiling_started_at = excluded.compiling_started_at,
                            updated_at = excluded.updated_at;
                        """,
                        (task_id, status, adapter_path, new_epoch, lease, now, now),
                    )
                else:
                    self._conn.execute(
                        """
                        INSERT INTO tasks (task_id, call_count, status, state_epoch,
                                           compiling_started_at, created_at, updated_at)
                        VALUES (?, 0, ?, ?, ?, ?, ?)
                        ON CONFLICT(task_id) DO UPDATE SET
                            status = excluded.status,
                            state_epoch = excluded.state_epoch,
                            compiling_started_at = excluded.compiling_started_at,
                            updated_at = excluded.updated_at;
                        """,
                        (task_id, status, new_epoch, lease, now, now),
                    )
                if changed:
                    self._record_transition_locked(
                        task_id, previous or "tracing", status, None, None, new_epoch, now
                    )
                return changed

        # PAW-JIT-04: bounded retry/backoff on top of BEGIN IMMEDIATE + busy_timeout.
        if self._with_write_retry(_do):
            self._after_transition_write(task_id)

        for listener in list(self._status_listeners):
            listener(task_id)

    def try_begin_compile(self, task_id: str) -> bool:
        """Compare-and-set `tracing` -> `compiling`. True only for the writer that won.

        D-2. `BackgroundCompiler.trigger_compilation` read `get_status`, decided, and
        then called `set_status(..., "compiling")`. Its own `threading.RLock` cannot
        close that window: the lock is per process and `traces.db` is shared, so N
        processes crossing the compile threshold together each read `tracing`, each
        decide to compile, and each buy a paid compile. Measured at 6 of 6 processes on
        both the sync and async paths. This method is the claim those callers branch on.

        **Why this is not the one-line `UPDATE` the finding's first draft described.**
        Atomicity was never the missing property; *commit* was. Under
        `isolation_level="IMMEDIATE"` sqlite3 emits `BEGIN IMMEDIATE` before a DML
        statement and then leaves the transaction **open**. A bare
        `self._conn.execute("UPDATE ... WHERE status='tracing'")` would therefore report
        `rowcount == 1` to a caller that is about to spend money, while every other
        process still reads `tracing` and can win the same CAS, while holding the write
        lock so every other writer burns its whole `_with_write_retry` budget, and while
        leaving the row to revert to `tracing` if this process dies -- after the money
        was spent. That is strictly worse than the race it would be fixing. Hence
        `_with_write_retry` + `_write_txn()`, exactly like every other read-modify-write
        on this class.

        **Why it is an upsert and not an UPDATE.** A bare UPDATE affects **zero** rows
        when no `tasks` row exists yet -- and today's guard *wins* in that case, because
        `get_status` returns `"tracing"` for a missing row and `set_status` upserts. An
        UPDATE-only CAS would silently refuse the first-ever compile for every task:
        verbatim the D-7 defect this repo has already paid for once.

        **Why an allow-list of one is equivalent to the deny-list it replaces.** The old
        guard refused `compiling|shadow|ready|failed`. The statuses this module ever
        *persists* are exactly `tracing|compiling|shadow|ready|failed` -- `stalled` is
        derived in `get_task_report` and never stored -- so the complement of that
        deny-list is `{tracing}` plus the no-row case, which is what this method accepts.
        The equivalence holds *because* of the upsert above; without it the no-row case
        would change behaviour.

        Parity with `set_status` is deliberate and load-bearing: a win bumps
        `state_epoch`, writes the `state_transitions` audit row, advances that task's
        prune clock and fires `_status_listeners`. Skipping any of them would drop the
        audit trail, the epoch bump shadow-window arithmetic depends on, or the
        adapter-cache invalidation hook -- three regressions behind a correct-looking CAS.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> Optional[int]:
            with self._write_txn():
                previous, _ = self._get_status_and_epoch_locked(task_id)
                cur = self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, status, state_epoch,
                                       compiling_started_at, created_at, updated_at)
                    VALUES (?, 0, 'compiling', 1, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        status = 'compiling',
                        state_epoch = tasks.state_epoch + 1,
                        -- J-5: the lease starts here, on the transition that actually
                        -- claims the compile.
                        compiling_started_at = excluded.compiling_started_at,
                        updated_at = excluded.updated_at
                    WHERE tasks.status = 'tracing';
                    """,
                    (task_id, now, now, now),
                )
                if cur.rowcount == 0:
                    return None
                # Re-read rather than recompute: on the insert branch the epoch is 1, on
                # the conflict branch it is whatever SQLite incremented it to, and the
                # `state_transitions` row has to name the epoch that was actually stored.
                _, new_epoch = self._get_status_and_epoch_locked(task_id)
                self._record_transition_locked(
                    task_id, previous or "tracing", "compiling", None, None, new_epoch, now
                )
                # The epoch, not a tuple. This used to also hand back
                # `previous or "tracing"`, which no caller read -- the mutation gate found
                # that second copy of the fallback unkillable, which is what dead data in
                # a return value looks like from the outside. The one copy that matters is
                # in the audit row above, pinned by
                # `test_try_begin_compile_names_the_previous_status_in_a_first_ever_compile_D_2`.
                return new_epoch

        won = self._with_write_retry(_do)
        if won is None:
            return False
        self._after_transition_write(task_id)
        for listener in list(self._status_listeners):
            listener(task_id)
        return True

    def reclaim_stale_compile(
        self,
        task_id: str,
        lease_seconds: float,
        max_attempts: Optional[int] = None,
    ) -> bool:
        """Release a `compiling` lease older than `lease_seconds`. True only for the winner.

        J-5. `status` is written `compiling` *before* the compile worker runs and nothing
        clears it if that process dies, while the decorator's trigger gate requires
        `status == 'tracing'`. So a task whose compiling process was killed is stuck:
        the teacher answers every subsequent call -- and is paid for every one -- with no
        compile in flight and none ever going to complete. This is the way out.

        A compare-and-set, for exactly D-2's reason: two processes both noticing the same
        wedged task must not both reclaim it and both pay. Same
        `_with_write_retry` + `_write_txn()` shape, with the `compile_attempts` bump in
        the **same transaction** as the status write.

        `compiling_started_at IS NOT NULL` is required, deliberately. A `compiling` row
        of unknown age must never be reclaimed: doing so would yank a *running*,
        already-billed compile out from under itself and buy a second one. Every writer
        of `compiling` stamps the lease (`set_status`, `try_begin_compile`), and
        `_init_db` backfills pre-v3 rows from `updated_at`, so in practice a NULL here
        means nothing this module wrote.

        `max_attempts`, when given, makes the reclaim terminal (`failed`) rather than
        retryable (`tracing`) once the bumped count reaches it. Without that, a task that
        wedges on every attempt would be handed another paid compile forever -- the same
        unbounded paid retry loop PAW-JIT-03 bounds on the failure path, reached by a
        different route. It is a parameter rather than an import because
        `BackgroundCompiler` owns the cap and `compiler.py` imports *this* module.

        One assumption worth stating, since this is the first ordered timestamp comparison
        in this module (everything else orders by `id`): `compiling_started_at` is compared
        **lexicographically**, which is correct only because every writer stores
        `datetime.now(timezone.utc).isoformat()` -- a fixed-width, fixed-offset
        (`+00:00`) format. A writer that stored a local-offset or non-padded timestamp
        would make this comparison silently wrong rather than raise, so keep the format.
        """
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        cutoff = (now_dt - timedelta(seconds=lease_seconds)).isoformat()

        def _do() -> Optional[Tuple[str, int]]:
            with self._write_txn():
                cur = self._conn.execute(
                    "SELECT state_epoch, compile_attempts FROM tasks WHERE task_id = ?;",
                    (task_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                attempts = (row["compile_attempts"] or 0) + 1
                target = (
                    "failed"
                    if max_attempts is not None and attempts >= max_attempts
                    else "tracing"
                )
                new_epoch = (row["state_epoch"] or 0) + 1
                updated = self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?,
                        state_epoch = state_epoch + 1,
                        compile_attempts = COALESCE(compile_attempts, 0) + 1,
                        compiling_started_at = NULL,
                        updated_at = ?
                    WHERE task_id = ?
                      AND status = 'compiling'
                      AND compiling_started_at IS NOT NULL
                      AND compiling_started_at < ?;
                    """,
                    (target, now, task_id, cutoff),
                )
                if updated.rowcount == 0:
                    return None
                self._record_transition_locked(
                    task_id, "compiling", target, None, None, new_epoch, now,
                    reason="stale_compile_lease",
                )
                return (target, new_epoch)

        won = self._with_write_retry(_do)
        if won is None:
            return False
        self._after_transition_write(task_id)
        for listener in list(self._status_listeners):
            listener(task_id)
        return True

    def get_compile_attempts(self, task_id: str) -> int:
        """Retrieve the number of compilation attempts made so far for task_id (PAW-JIT-03).

        Keeps its bare-`int` contract: a task with no row reads 0, exactly like a task
        that has never failed a compile. Callers that need to tell those two apart use
        `_get_compile_attempts_locked`, which returns `None` for the former (D-7).
        """
        attempts = self._get_compile_attempts_locked(task_id)
        return 0 if attempts is None else attempts

    def _get_compile_attempts_locked(self, task_id: str) -> Optional[int]:
        """Read compile_attempts for task_id, or None if there is no row at all.

        D-7: "no task" and "a task that has never failed a compile" both used to read
        0, which is what let `increment_compile_attempts` return 0 forever against a
        missing row and turned `BackgroundCompiler`'s bounded retry into an unbounded,
        paid one. Runs on this thread's connection; call it inside `_write_txn` when
        the value must agree with a write in the same transaction.
        """
        cur = self._conn.execute(
            "SELECT compile_attempts FROM tasks WHERE task_id = ?;", (task_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return row["compile_attempts"] or 0

    def increment_compile_attempts(self, task_id: str) -> int:
        """Atomically increment and return the compilation attempt counter for task_id.

        PAW-JIT-03: this is what lets BackgroundCompiler cap retries at a fixed number
        instead of either deadlocking in "failed" forever (the pre-fix bug) or retrying
        unconditionally on every subsequent call once the task's (never-decreasing)
        call_count has crossed the compilation threshold (the audit's own suggested
        fix, which turns into an unbounded retry loop).

        D-7: upserts, so two consecutive calls return 1 then 2 in every case -- a
        missing `tasks` row included, which is exactly the case where the bare UPDATE
        returned 0 forever and every failed compile was retried and paid for again.
        """
        def _do() -> int:
            with self._write_txn():
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, compile_attempts, status,
                                       created_at, updated_at)
                    VALUES (?, 0, 1, 'tracing', ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        compile_attempts = COALESCE(tasks.compile_attempts, 0) + 1,
                        updated_at = excluded.updated_at;
                    """,
                    (task_id, now, now),
                )
                attempts = self._get_compile_attempts_locked(task_id)
                # Unreachable after the upsert above. Reporting 0 rather than raising
                # keeps this off the raising paths the fail-open invariant protects;
                # BackgroundCompiler fails *closed* when the count does not advance.
                return 0 if attempts is None else attempts

        # PAW-JIT-04: bounded retry/backoff on top of BEGIN IMMEDIATE + busy_timeout.
        return self._with_write_retry(_do)

    def get_adapter_path(self, task_id: str) -> Optional[str]:
        """Retrieve path to compiled adapter if task is ready."""
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

        Call it inside `_write_txn`: this is the leading read of a read-modify-write,
        and D-1 is precisely what happens when it is not.
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
        """Insert a lifecycle transition row. Caller must be inside `_write_txn`."""
        self._conn.execute(
            """
            INSERT INTO state_transitions
                (task_id, from_status, to_status, agreement, sample_count,
                 state_epoch, reason, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (task_id, from_status, to_status, agreement, sample_count, state_epoch, reason, now),
        )
        # D-5: the prune clock is *not* advanced here. Every caller calls
        # `_after_transition_write(task_id)` once the transaction has committed, so a
        # retried transaction counts once and the clock is per task, not per database.

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
        self._after_transition_write(task_id)

    def set_shadow_started(self, task_id: str, adapter_path: str) -> None:
        """Move a freshly-compiled task into `shadow`: the adapter exists but does not serve.

        D-7: upsert. This is one of the two sites *worse* than the three the hunt
        named, because it unconditionally writes a `state_transitions` row straight
        after -- so against a missing `tasks` row the status write vanished and left
        an orphaned transition record pointing at a task that does not exist.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                new_epoch = epoch + 1
                self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, status, adapter_path,
                                       shadow_started_at, state_epoch,
                                       created_at, updated_at)
                    VALUES (?, 0, 'shadow', ?, ?, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        status = 'shadow',
                        adapter_path = excluded.adapter_path,
                        shadow_started_at = excluded.shadow_started_at,
                        state_epoch = excluded.state_epoch,
                        updated_at = excluded.updated_at;
                    """,
                    (task_id, adapter_path, now, new_epoch, now, now),
                )
                self._record_transition_locked(
                    task_id, previous or "compiling", "shadow", None, None, new_epoch, now
                )

        self._with_write_retry(_do)
        self._after_transition_write(task_id)
        for listener in list(self._status_listeners):
            listener(task_id)

    def set_ready_from_compile(self, task_id: str, adapter_path: str) -> None:
        """Promote straight to `ready` on compile -- the `shadow_window=0` path.

        D-7: upsert, for the same reason as `set_shadow_started` above.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> None:
            with self._write_txn():
                previous, epoch = self._get_status_and_epoch_locked(task_id)
                new_epoch = epoch + 1
                self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, status, adapter_path,
                                       promoted_at, promoted_agreement, state_epoch,
                                       created_at, updated_at)
                    VALUES (?, 0, 'ready', ?, ?, NULL, ?, ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        status = 'ready',
                        adapter_path = excluded.adapter_path,
                        promoted_at = excluded.promoted_at,
                        promoted_agreement = NULL,
                        state_epoch = excluded.state_epoch,
                        updated_at = excluded.updated_at;
                    """,
                    (task_id, adapter_path, now, new_epoch, now, now),
                )
                self._record_transition_locked(
                    task_id, previous or "compiling", "ready", None, None, new_epoch, now,
                    reason="shadow_disabled",
                )

        self._with_write_retry(_do)
        self._after_transition_write(task_id)
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
            with self._conn:
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
            # D-5: a won compare-and-set wrote a `state_transitions` row, so it
            # advances that task's own prune clock -- outside the retried closure.
            self._after_transition_write(task_id)
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
            with self._conn:
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
            # D-5: a won compare-and-set wrote a `state_transitions` row, so it
            # advances that task's own prune clock -- outside the retried closure.
            self._after_transition_write(task_id)
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

        bumped = self._with_write_retry(_do)
        if bumped:
            # D-5: the config-change transition row advances this task's own clock.
            self._after_transition_write(task_id)
        return bumped

    def get_shadow_config(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Read back the persisted resolved shadow configuration, if any."""
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
                now = datetime.now(timezone.utc).isoformat()
                self._conn.execute(
                    """
                    INSERT INTO tasks (task_id, call_count, fail_open_count, status,
                                       created_at, updated_at)
                    VALUES (?, 0, 1, 'tracing', ?, ?)
                    ON CONFLICT(task_id) DO UPDATE SET
                        fail_open_count = COALESCE(tasks.fail_open_count, 0) + 1,
                        updated_at = excluded.updated_at;
                    """,
                    (task_id, now, now),
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
        and never advances the window. J-2/C2: a `pool_exhausted` row (a shadow-path
        deadline-pool timeout caused by a *different* task's wedged comparisons,
        not this adapter's own drift -- see shadow.py) is excluded the same way,
        for the same reason: it is infrastructure, not a countable comparison of
        this adapter against the teacher.
        """
        now = datetime.now(timezone.utc).isoformat()

        def _do() -> Dict[str, int]:
            with self._write_txn():
                if verdict in ("teacher_error", "pool_exhausted"):
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
                return {"seq": seq}

        result = self._with_write_retry(_do)
        # D-5: per task, and outside the retried closure. The prune is its own short
        # transaction rather than a tail on the insert's: it is retention maintenance,
        # not part of the row's atomicity, and keeping it out means a retried insert
        # cannot re-run a prune that already happened.
        if self._note_write(self._shadow_pair_writes, task_id, _prune_interval(max_pairs)):
            def _prune() -> None:
                with self._write_txn():
                    self._prune_shadow_pairs_locked(task_id, max_pairs)

            self._with_write_retry(_prune)
        return result

    def get_epoch_seq(self, task_id: str, state_epoch: int) -> int:
        """Highest countable-comparison sequence number reached at this epoch.

        Survives pruning (unlike `COUNT(*)`), which is what the stall guard needs.
        """
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

        The `verdict NOT IN ('teacher_error', 'pool_exhausted')` filter sits
        **inside** the `LIMIT` subquery, not outside it. Outside, an excluded row
        would still spend the window's LIMIT budget, so `samples` could never
        reach `audit_window` once one landed in the trailing window -- and
        `samples == audit_window` is the demotion trigger. One flaky teacher call
        (or, J-2/C2, one shadow-path deadline-pool timeout caused by a different
        task's wedged comparisons) would disable drift detection for that task
        forever.

        Finding 5: `teacher_error` is scoped to that *same* trailing window, not the
        whole epoch -- otherwise it grows unbounded next to a `window`-sized rate.
        Simplest implementation: excluded-verdict rows are not countable and carry
        no `seq` of their own (they are stored with `seq = 0`, see
        `record_shadow_pair`), so the boundary is the `id` of the oldest of the
        `window` countable rows above, and a teacher-error row counts if it is at
        least that recent -- i.e. it is interleaved with the current window. With
        fewer than `window` countable rows so far this epoch, the boundary is
        simply the first one recorded, so nothing is double-counted or missed;
        with none at all yet, nothing has started, and it reads 0.

        `pool_exhausted` (J-2/C2) shares the exclusion mechanism but not this
        reporting: only `teacher_error` gets its own named, windowed count below
        (`stats["teacher_error"]`) -- a pool-exhaustion timeout is rare enough,
        and purely an artifact of process-wide pool sizing rather than of this
        task, that a second named counter was not worth adding for it here.
        """
        rows = self._conn.execute(
            """
            SELECT verdict, COUNT(*) AS n FROM (
                SELECT verdict FROM shadow_pairs
                WHERE task_id = ? AND state_epoch = ? AND phase = ?
                  AND verdict NOT IN ('teacher_error', 'pool_exhausted')
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
                  AND verdict NOT IN ('teacher_error', 'pool_exhausted')
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
        # D-9: refuse to do arithmetic over a window the retention cap cannot fill.
        #
        # A window is scored only at `samples == window`, and pruning bounds `samples`
        # by `shadow_max_pairs`. With the cap below the window nothing ever completes,
        # so neither promotion nor demotion can *ever* fire -- silently, with
        # `paw-kit report` showing a healthy rate over a short window. The shipped
        # decorator rejects the combination at decoration time; any other caller of
        # these public methods did not, which is why the guard also belongs here, in
        # the method that actually does the arithmetic.
        #
        # `seq` is the right witness: it is monotone over countable comparisons at this
        # epoch and survives pruning, so `seq >= window` means at least a window's
        # worth has been recorded. If fewer than `window` of them are still retained,
        # retention is the reason and no amount of further traffic will help.
        #
        # This raises. `_with_write_retry` catches only `OperationalError`, so it
        # propagates -- reaching `ShadowRunner._maybe_transition`, whose own
        # `except Exception: return` swallows it (the comparison is simply not scored,
        # the caller was served by the teacher either way) and `get_task_report`, where
        # it surfaces to `paw-kit report`. Neither weakens the fail-open invariant:
        # nothing on the request path calls this.
        if window > 0 and seq >= window and samples < window:
            raise ValueError(
                f"task {task_id!r} at epoch {state_epoch} has recorded {seq} "
                f"comparisons but retains only {samples} of the {window} a window "
                "needs: the retention cap (shadow_max_pairs) is below the window, so "
                "no window can ever complete and neither promotion nor demotion can "
                "ever fire. Raise shadow_max_pairs to at least twice "
                "max(shadow_window, audit_window), or lower the window."
            )
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
        cur = self._conn.execute("SELECT task_id FROM tasks ORDER BY created_at ASC;")
        return [row["task_id"] for row in cur.fetchall()]

    def get_task_report(self, task_id: str) -> Dict[str, Any]:
        """Rich per-task view for `wrapper.get_agreement()` and `paw-kit report`.

        `get_status` deliberately keeps its bare-`str` signature and return; this is
        the additive rich API rather than a change to it.
        """
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
        """Oldest-first retention on `shadow_pairs`. Caller must be inside `_write_txn`.

        An id-threshold delete rather than `id NOT IN (SELECT ... LIMIT ?)`: the
        subquery form materialises up to `cap` ids and re-scans them per row, and this
        runs inside a write transaction holding SQLite's single writer.
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
        """Oldest-first retention on `state_transitions`. Caller must be inside `_write_txn`."""
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
        """Close every connection this TraceDB has handed out (J-7).

        One per thread that has touched it, not one overall. After this, a thread
        that has never connected gets `ProgrammingError` rather than silently
        reopening the file -- the same failure a thread holding an already-closed
        connection sees, which `shadow.py`'s worker already handles.
        """
        with self._conns_lock:
            self._closed = True
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            try:
                conn.close()
            except Exception:  # pragma: no cover - closing twice, or mid-statement
                pass
