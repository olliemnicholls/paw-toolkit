"""Thread-safe SQLite tracing database for paw.jit."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional

# INSERT ... ON CONFLICT ... DO UPDATE (used by record_trace) requires SQLite >= 3.24.
_MIN_SQLITE_VERSION = (3, 24, 0)


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
        )
        self._conn.row_factory = sqlite3.Row
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

    def _init_db(self) -> None:
        """Create tables and enable WAL mode for high concurrency."""
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
        with self._lock, self._conn:
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
        """Update task lifecycle status and optional adapter path."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn:
            if adapter_path is not None:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, adapter_path = ?, updated_at = ?
                    WHERE task_id = ?;
                    """,
                    (status, adapter_path, now, task_id),
                )
            else:
                self._conn.execute(
                    """
                    UPDATE tasks
                    SET status = ?, updated_at = ?
                    WHERE task_id = ?;
                    """,
                    (status, now, task_id),
                )

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
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET compile_attempts = compile_attempts + 1 WHERE task_id = ?;",
                (task_id,),
            )
            return self._get_compile_attempts_locked(task_id)

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

    def close(self) -> None:
        """Close SQLite connection."""
        with self._lock:
            self._conn.close()
