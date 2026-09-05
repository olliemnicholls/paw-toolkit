"""Thread-safe SQLite tracing database for paw.jit."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, List, Optional


class TraceDB:
    """Embedded SQLite database tracking production API calls and compilation triggers."""

    def __init__(self, db_path: str = "./.paw/traces.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

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
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
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
            cur = self._conn.execute(
                """
                INSERT INTO tasks (task_id, call_count, status, created_at, updated_at)
                VALUES (?, 1, 'tracing', ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    call_count = call_count + 1,
                    updated_at = excluded.updated_at
                RETURNING call_count;
                """,
                (task_id, now, now),
            )
            row = cur.fetchone()
            new_count = row[0] if row else 1

            # 2. Insert trace record
            self._conn.execute(
                """
                INSERT INTO traces (task_id, input_payload, teacher_output, latency_ms, timestamp)
                VALUES (?, ?, ?, ?, ?);
                """,
                (task_id, input_payload, teacher_output, latency_ms, now),
            )
            return new_count

    def get_call_count(self, task_id: str) -> int:
        """Retrieve total calls recorded for task."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT call_count FROM tasks WHERE task_id = ?;", (task_id,)
            )
            row = cur.fetchone()
            return row["call_count"] if row else 0

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
