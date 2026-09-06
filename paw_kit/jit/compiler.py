"""Background compilation worker and adapter cache manager for paw.jit."""

from pathlib import Path
import threading
from typing import Dict, List, Optional

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.jit.db import TraceDB


class BackgroundCompiler:
    """Dispatches and tracks asynchronous compilation jobs for @compile_on_hit."""

    # PAW-JIT-03: cap on retryable compilation attempts. Without this, a failure
    # either deadlocks the task in "failed" forever (never retried), or -- if failure
    # naively reset status back to "tracing" -- retries unconditionally on every
    # subsequent call once call_count (which never decreases) has crossed the
    # compilation threshold, i.e. an unbounded retry loop.
    _MAX_COMPILE_ATTEMPTS = 3

    def __init__(self) -> None:
        self._active_threads: Dict[str, threading.Thread] = {}
        self._lock = threading.RLock()

    def is_compiling(self, task_id: str) -> bool:
        """Check if compilation is currently active for task_id."""
        with self._lock:
            thread = self._active_threads.get(task_id)
            return thread is not None and thread.is_alive()

    def trigger_compilation(
        self,
        task_id: str,
        spec: str,
        db: TraceDB,
        backend: AbstractPAWBackend,
        output_path: str,
        sync: bool = False,
    ) -> Optional[threading.Thread]:
        """Initiate background compilation of traces into a .paw adapter.

        Args:
            task_id: Unique task identifier.
            spec: Natural language task specification.
            db: Active TraceDB instance to pull training examples from.
            backend: AbstractPAWBackend implementation to perform compilation.
            output_path: Destination path for the compiled adapter.
            sync: If True, executes synchronously instead of in a background thread.

        Returns:
            The spawned Thread if asynchronous, or None if synchronous or already compiling.
        """
        with self._lock:
            status = db.get_status(task_id)
            if status in ("compiling", "ready", "failed") and not sync:
                return None
            db.set_status(task_id, "compiling")

        def _worker() -> None:
            try:
                traces = db.get_traces(task_id)
                examples: List[Dict[str, str]] = [
                    {"input": t["input_payload"], "output": t["teacher_output"]}
                    for t in traces
                ]
                compiled_path = backend.compile(
                    spec=spec,
                    examples=examples,
                    output_path=output_path,
                )
                db.set_status(task_id, "ready", adapter_path=compiled_path)
            except Exception:
                # PAW-JIT-03: bounded retry, not an unconditional reset to "tracing"
                # (see _MAX_COMPILE_ATTEMPTS docstring) and not a permanent deadlock.
                attempts = db.increment_compile_attempts(task_id)
                if attempts < self._MAX_COMPILE_ATTEMPTS:
                    db.set_status(task_id, "tracing")
                else:
                    db.set_status(task_id, "failed")
            finally:
                with self._lock:
                    self._active_threads.pop(task_id, None)

        if sync:
            _worker()
            return None

        thread = threading.Thread(
            target=_worker,
            name=f"paw-compile-{task_id}",
            daemon=True,
        )
        with self._lock:
            self._active_threads[task_id] = thread
        thread.start()
        return thread
