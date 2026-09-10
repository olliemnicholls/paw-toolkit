"""Off-request-path shadow comparison worker for @compile_on_hit (Track 14).

The caller thread does no I/O for shadow mode: it performs one `put_nowait` of a
frozen dataclass and returns. Everything else -- loading the adapter, running it,
re-validating both answers, comparing them, writing the `shadow_pairs` row and
deciding a promotion or demotion -- happens on a daemon worker thread, one per
`(db_path, task_id)`.

**Why the key includes `db_path`.** `task_id` is `sha256(qualname:spec)` and does not
include `cache_dir`, so two decorations of the same function and spec against
different cache directories -- exactly what the test suite does with `tmp_path` --
collide on `task_id`. A runner keyed on `task_id` alone would bind one worker to the
first `TraceDB` it saw and write every later task's pairs into that database.

Nothing in this module imports `decorator.py`: the adapter runner, the teacher runner
and the redaction function all arrive as callables on the job. That is what keeps
`_ADAPTER_CALLABLE_CACHE` (and its status-listener invalidation hook) in
`decorator.py` where the existing tests reach for it, with no import cycle.
"""

import atexit
from dataclasses import dataclass
import json
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

from pydantic import BaseModel

from paw_kit.jit.agreement import safe_agreement

logger = logging.getLogger("paw_kit.jit.shadow")

# Once a task has accumulated this many times `shadow_window` comparisons at one epoch
# without promoting, the runner drops to sampling one comparison in every
# `shadow_window`. This bounds the "an adapter that never promotes doubles compute
# forever" cost without adding a seventh tuning knob.
_SHADOW_STALL_FACTOR = 5

# Global (not per-task) best-effort drain budget at interpreter exit. Per-task would
# cost N times this for N decorated functions; and with nothing pending the drain
# returns immediately, so a short CLI run that happened to import a decorated module
# pays nothing.
_ATEXIT_DRAIN_SECONDS = 2.0

_RunnerKey = Tuple[str, str]


def serialize_answer(value: Any) -> str:
    """Serialize an answer for persistence, matching `decorator.py`'s trace serialization."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value)
        except Exception:
            return str(value)
    return str(value)


def _coerce(value: Any, response_model: Optional[Type[BaseModel]]) -> Any:
    """Re-validate a persisted answer back into the shape the comparison runs on.

    With a `response_model` both sides are validated into model instances, so the
    comparison is field-wise over real objects and the stored row is a faithful record
    of exactly what was compared. A validation failure here is an *adapter error*, not
    a crash: the caller has already been served by the teacher.
    """
    if response_model is None:
        return value
    if isinstance(value, BaseModel):
        return value
    return response_model.model_validate_json(value if isinstance(value, str) else str(value))


@dataclass(frozen=True)
class ShadowJob:
    """One teacher/adapter comparison to run off the request path.

    The two output pairs are inverted between the phases, which is why all four are
    optional: a `shadow` job knows the teacher's answer at enqueue and computes the
    adapter's on the worker; an `audit` job knows the adapter's answer at enqueue (the
    value just served) and computes the teacher's on the worker.

    `input_payload` is the **unredacted** string. Redaction is applied on the worker
    immediately before the row is written, symmetrically to all three text columns:
    the adapter must be run on the input production actually sent, and comparing a
    redacted teacher answer against an unredacted adapter answer would be
    systematically wrong in both directions. The worker is in-process; the database is
    the trust boundary.
    """

    task_id: str
    db_path: str
    db: Any
    state_epoch: int
    phase: str
    input_payload: str
    teacher_output: Optional[str] = None
    teacher_latency_ms: Optional[float] = None
    adapter_output: Optional[str] = None
    adapter_latency_ms: Optional[float] = None
    run_adapter: Optional[Callable[[str], Any]] = None
    run_teacher: Optional[Callable[[], Any]] = None
    response_model: Optional[Type[BaseModel]] = None
    agreement_fn: Optional[Callable[[Any, Any], bool]] = None
    redact_fn: Optional[Callable[[str], str]] = None
    shadow_window: int = 0
    audit_window: int = 0
    shadow_threshold: float = 1.0
    demote_threshold: float = 0.0
    max_pairs: int = 500
    queue_size: int = 8

    @property
    def key(self) -> _RunnerKey:
        return (self.db_path, self.task_id)

    @property
    def window(self) -> int:
        return self.shadow_window if self.phase == "shadow" else self.audit_window


class ShadowRunner:
    """Per-(db_path, task_id) daemon worker draining a bounded queue of comparisons."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queues: Dict[_RunnerKey, "queue.Queue[ShadowJob]"] = {}
        self._threads: Dict[_RunnerKey, threading.Thread] = {}
        self._pending: Dict[_RunnerKey, int] = {}
        self._dropped: Dict[_RunnerKey, int] = {}
        self._stalled: Dict[_RunnerKey, int] = {}
        self._drop_warned: Set[_RunnerKey] = set()
        self._stall_warned: Set[Tuple[_RunnerKey, int]] = set()
        self._stall_counter: Dict[Tuple[_RunnerKey, int], int] = {}
        self._seq_cache: Dict[Tuple[_RunnerKey, int], int] = {}
        self._error_logged: Set[Tuple[str, str]] = set()

    # --- caller-thread surface ------------------------------------------------

    def submit(self, job: ShadowJob) -> bool:
        """Enqueue a comparison. Never blocks, never raises into the caller.

        On a full queue the **newest** job is dropped: O(1), and it never evicts work
        already accepted. A dropped job is neither an agreement nor a disagreement --
        it is simply not sampled, and never enters the window denominator.
        """
        key = job.key
        try:
            q = self._ensure_worker(job)
            with self._lock:
                self._pending[key] = self._pending.get(key, 0) + 1
            try:
                q.put_nowait(job)
            except queue.Full:
                with self._lock:
                    self._pending[key] = max(0, self._pending.get(key, 1) - 1)
                    self._dropped[key] = self._dropped.get(key, 0) + 1
                    first = key not in self._drop_warned
                    self._drop_warned.add(key)
                if first:
                    logger.warning(
                        "paw_kit.jit.shadow: task_id=%s shadow queue is full (size=%d); "
                        "dropping this comparison. Dropped comparisons are not counted as "
                        "disagreements -- they are simply not sampled. Further drops for "
                        "this task log at DEBUG.",
                        job.task_id, job.queue_size,
                    )
                else:
                    logger.debug(
                        "paw_kit.jit.shadow: task_id=%s shadow queue full, comparison dropped.",
                        job.task_id,
                    )
                return False
            return True
        except Exception as exc:  # pragma: no cover - defense in depth
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s could not enqueue a comparison (%s: %s).",
                job.task_id, type(exc).__name__, exc,
            )
            return False

    # --- worker plumbing ------------------------------------------------------

    def _ensure_worker(self, job: ShadowJob) -> "queue.Queue[ShadowJob]":
        key = job.key
        with self._lock:
            q = self._queues.get(key)
            if q is None:
                q = queue.Queue(maxsize=max(1, job.queue_size))
                self._queues[key] = q
            thread = self._threads.get(key)
            if thread is None or not thread.is_alive():
                thread = threading.Thread(
                    target=self._worker_loop,
                    args=(key,),
                    name=f"paw-shadow-{job.task_id[:12]}",
                    daemon=True,
                )
                self._threads[key] = thread
                thread.start()
            return q

    def _worker_loop(self, key: _RunnerKey) -> None:
        q = self._queues[key]
        while True:
            try:
                job = q.get(timeout=1.0)
            except queue.Empty:
                continue
            except Exception:  # pragma: no cover - the queue itself failing
                return
            try:
                self._run_job(job)
            except BaseException as exc:  # noqa: BLE001 - a worker must never die
                logger.debug(
                    "paw_kit.jit.shadow: task_id=%s comparison failed outright (%s: %s).",
                    job.task_id, type(exc).__name__, exc,
                )
            finally:
                with self._lock:
                    self._pending[key] = max(0, self._pending.get(key, 1) - 1)

    def _current_seq(self, job: ShadowJob) -> int:
        cache_key = (job.key, job.state_epoch)
        with self._lock:
            cached = self._seq_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            seq = job.db.get_epoch_seq(job.task_id, job.state_epoch)
        except Exception:
            seq = 0
        with self._lock:
            self._seq_cache.setdefault(cache_key, seq)
        return seq

    def _stall_throttled(self, job: ShadowJob) -> bool:
        """True when this comparison should be skipped by the stall guard."""
        if job.phase != "shadow" or job.shadow_window <= 0:
            return False
        seq = self._current_seq(job)
        if seq < _SHADOW_STALL_FACTOR * job.shadow_window:
            return False
        cache_key = (job.key, job.state_epoch)
        with self._lock:
            first = cache_key not in self._stall_warned
            self._stall_warned.add(cache_key)
            count = self._stall_counter.get(cache_key, 0) + 1
            self._stall_counter[cache_key] = count
        if first:
            try:
                stats = job.db.get_agreement_stats(
                    job.task_id, job.state_epoch, job.shadow_window, "shadow"
                )
                rate = stats["rate"]
            except Exception:
                rate = None
            logger.warning(
                "paw_kit.jit.shadow: task_id=%s is not converging: agreement %s over %d "
                "samples at this epoch; the teacher is still serving. Sampling one "
                "comparison in every %d from here to bound the cost.",
                job.task_id, "unknown" if rate is None else f"{rate:.2f}", seq,
                job.shadow_window,
            )
        if count % job.shadow_window != 0:
            with self._lock:
                self._stalled[job.key] = self._stalled.get(job.key, 0) + 1
            return True
        return False

    def _log_shadow_error(self, task_id: str, error_type: str, exc: BaseException) -> None:
        """WARNING on the first occurrence per (task_id, error_type), DEBUG thereafter.

        A 6 s/call adapter that fails every time must not flood the log. Note these are
        *not* fail-opens: in `shadow` the caller was always getting the teacher, so
        touching `_record_fail_open` here would corrupt an existing signal.
        """
        marker = (task_id, error_type)
        with self._lock:
            first = marker not in self._error_logged
            self._error_logged.add(marker)
        if first:
            logger.warning(
                "paw_kit.jit.shadow: task_id=%s shadow comparison failed (%s: %s). This is "
                "recorded as a disagreement, not a fail-open -- the caller was served by "
                "the teacher either way. Further %s failures for this task log at DEBUG.",
                task_id, error_type, exc, error_type,
            )
        else:
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s shadow comparison failed (%s: %s).",
                task_id, error_type, exc,
            )

    def _run_job(self, job: ShadowJob) -> None:
        if self._stall_throttled(job):
            return

        teacher_output = job.teacher_output
        teacher_latency = job.teacher_latency_ms
        adapter_output = job.adapter_output
        adapter_latency = job.adapter_latency_ms
        adapter_value: Any = None
        verdict: Optional[str] = None
        error_type: Optional[str] = None

        if job.phase == "shadow":
            started = time.perf_counter()
            try:
                adapter_value = job.run_adapter(job.input_payload)  # type: ignore[misc]
                adapter_latency = (time.perf_counter() - started) * 1000
                adapter_output = serialize_answer(adapter_value)
            except BaseException as exc:  # noqa: BLE001
                verdict = "error"
                error_type = type(exc).__name__
                adapter_output = None
                self._log_shadow_error(job.task_id, error_type, exc)
        else:
            started = time.perf_counter()
            try:
                teacher_value_raw = job.run_teacher()  # type: ignore[misc]
                teacher_latency = (time.perf_counter() - started) * 1000
                teacher_output = serialize_answer(teacher_value_raw)
            except BaseException as exc:  # noqa: BLE001
                # Not the adapter's fault: excluded from numerator *and* denominator.
                verdict = "teacher_error"
                error_type = type(exc).__name__
                teacher_output = None
                self._log_shadow_error(job.task_id, error_type, exc)

        if verdict is None:
            try:
                teacher_compare = _coerce(teacher_output, job.response_model)
                if job.phase == "shadow":
                    adapter_compare = (
                        adapter_value
                        if isinstance(adapter_value, BaseModel) or job.response_model is None
                        else _coerce(adapter_output, job.response_model)
                    )
                else:
                    adapter_compare = _coerce(adapter_output, job.response_model)
            except Exception as exc:
                verdict = "error"
                error_type = type(exc).__name__
                self._log_shadow_error(job.task_id, error_type, exc)
            else:
                agreement_fn = job.agreement_fn
                agreed, agreement_error = safe_agreement(
                    agreement_fn, teacher_compare, adapter_compare  # type: ignore[arg-type]
                )
                if agreement_error is not None:
                    verdict = "error"
                    error_type = agreement_error
                else:
                    verdict = "agree" if agreed else "disagree"

        redact = job.redact_fn
        stored_input = redact(job.input_payload) if redact else job.input_payload
        stored_teacher = redact(teacher_output) if (redact and teacher_output) else teacher_output
        stored_adapter = redact(adapter_output) if (redact and adapter_output) else adapter_output

        try:
            result = job.db.record_shadow_pair(
                task_id=job.task_id,
                state_epoch=job.state_epoch,
                phase=job.phase,
                input_payload=stored_input,
                teacher_output=stored_teacher,
                adapter_output=stored_adapter,
                verdict=verdict,
                error_type=error_type,
                teacher_latency_ms=teacher_latency,
                adapter_latency_ms=adapter_latency,
                max_pairs=job.max_pairs,
            )
        except Exception as exc:
            # A TraceDB closed out from under the worker raises sqlite3.ProgrammingError,
            # which is *not* an OperationalError and so is not covered by the write-retry
            # wrapper. Nothing here may propagate.
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s could not persist a comparison (%s: %s).",
                job.task_id, type(exc).__name__, exc,
            )
            return

        seq = result.get("seq", 0)
        if seq:
            with self._lock:
                self._seq_cache[(job.key, job.state_epoch)] = seq
        self._maybe_transition(job, seq)

    def _maybe_transition(self, job: ShadowJob, seq: int) -> None:
        """Evaluate a promotion/demotion once per *completed* window.

        The window is tumbling, not sliding: the trigger is `seq % window == 0` on the
        just-inserted row, so a 60%-agreement adapter gets one draw per window rather
        than a fresh draw on every single comparison. A sliding window promotes such an
        adapter eventually, which is the exact failure this feature exists to prevent.
        """
        window = job.window
        if window <= 0 or seq <= 0 or seq % window != 0:
            return
        try:
            stats = job.db.get_agreement_stats(
                job.task_id, job.state_epoch, window, job.phase
            )
        except Exception:
            return
        rate = stats["rate"]
        samples = stats["samples"]
        # A *full* window, not merely a rate: one agreeing sample gives rate == 1.0.
        if rate is None or samples != window:
            return
        try:
            if job.phase == "shadow":
                if rate >= job.shadow_threshold:
                    if job.db.try_promote(job.task_id, job.state_epoch, rate, samples):
                        logger.info(
                            "paw_kit.jit.shadow: task_id=%s promoted to 'ready' -- agreement "
                            "%.2f over %d comparisons.", job.task_id, rate, samples,
                        )
            elif rate < job.demote_threshold:
                if job.db.try_demote(job.task_id, job.state_epoch, rate, samples):
                    logger.warning(
                        "paw_kit.jit.shadow: task_id=%s demoted to 'shadow' -- audit "
                        "agreement %.2f over %d comparisons is below the demote threshold "
                        "%.2f. The wrapped function is serving again.",
                        job.task_id, rate, samples, job.demote_threshold,
                    )
        except Exception as exc:
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s transition check failed (%s: %s).",
                job.task_id, type(exc).__name__, exc,
            )

    # --- introspection and test hooks ----------------------------------------

    def _keys_for(self, task_id: str, db_path: Optional[str] = None) -> list:
        with self._lock:
            return [
                k for k in self._queues
                if k[1] == task_id and (db_path is None or k[0] == db_path)
            ]

    def stats(self, task_id: str, db_path: Optional[str] = None) -> Dict[str, int]:
        """In-process counters for this task since process start (never persisted)."""
        keys = self._keys_for(task_id, db_path)
        with self._lock:
            return {
                "pending": sum(self._pending.get(k, 0) for k in keys),
                "dropped": sum(self._dropped.get(k, 0) for k in keys),
                "stalled": sum(self._stalled.get(k, 0) for k in keys),
            }

    def drain(self, task_id: str, timeout: float = 5.0, db_path: Optional[str] = None) -> bool:
        """Block until this task's queue is empty and the in-flight job has finished.

        Test hook only -- library code never calls it.
        """
        deadline = time.monotonic() + timeout
        while True:
            keys = self._keys_for(task_id, db_path)
            with self._lock:
                pending = sum(self._pending.get(k, 0) for k in keys)
            if pending == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)

    def shutdown(self, timeout: float = _ATEXIT_DRAIN_SECONDS) -> bool:
        """Best-effort drain of everything still queued, under one global budget.

        Returns immediately when nothing is pending, so a process that merely imported
        a decorated module pays nothing at exit. Worker threads are daemons, so a
        budget overrun costs nothing but the unfinished comparisons.
        """
        try:
            deadline = time.monotonic() + timeout
            while True:
                with self._lock:
                    pending = sum(self._pending.values())
                if pending == 0:
                    return True
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.01)
        except Exception:  # pragma: no cover - exit-path defense in depth
            return False


_GLOBAL_SHADOW_RUNNER = ShadowRunner()


def _atexit_drain() -> None:
    try:
        _GLOBAL_SHADOW_RUNNER.shutdown(_ATEXIT_DRAIN_SECONDS)
    except BaseException:  # pragma: no cover - nothing may escape at exit
        pass


atexit.register(_atexit_drain)
