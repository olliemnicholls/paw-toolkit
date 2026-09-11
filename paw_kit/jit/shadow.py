"""Off-request-path shadow comparison worker for @compile_on_hit (Track 14).

The caller thread does no I/O for shadow mode: it performs one `put_nowait` of a
frozen dataclass and returns. Everything else -- loading the adapter, running it,
re-validating both answers, comparing them, writing the `shadow_pairs` row and
deciding a promotion or demotion -- happens on a daemon worker thread, one per
`(db_path, task_id)`.

**Why the key includes `db_path`.** `task_id` is derived from the function's identity
and spec (J-8: qualname, spec, `co_filename`, `co_firstlineno`) and does not include
`cache_dir`, so two decorations of the same function and spec against
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
import logging
import queue
import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type

from pydantic import BaseModel

from paw_kit.jit.agreement import safe_agreement, stringify_answer
from paw_kit.jit.db import _SHADOW_STALL_FACTOR
from paw_kit.jit.deadline import DeadlinePool, PoolExhausted

logger = logging.getLogger("paw_kit.jit.shadow")

# J-2: a single, process-wide bounded pool of daemon worker threads for every
# shadow-phase adapter call, across every task's own per-(db_path, task_id)
# worker thread -- separate from decorator.py's `_SERVED_DEADLINE_POOL` (see
# this track's Dependency check for why two pools, not one shared one).
#
# Residual limitation, stated plainly rather than glossed over: this pool IS
# shared across every task's shadow comparisons. N wedged shadow comparisons
# (from N different tasks, or repeated ones from a single wedged task) can
# still starve *other* tasks' shadow comparisons of a slot -- a real, accepted,
# and much narrower limitation than today's "one wedged adapter blocks its own
# task's queue forever," not a claim of full cross-task isolation. A timeout
# caused by this kind of cross-task pool exhaustion is infrastructure, not this
# adapter's own drift, and is scored `verdict = "pool_exhausted"` below --
# excluded from both the promotion numerator and denominator the same way
# `get_agreement_stats` (paw_kit.jit.db) already excludes `teacher_error`,
# rather than `error`, which counts against the adapter being compared.
_SHADOW_POOL_MAX_WORKERS = 4
_SHADOW_DEADLINE_POOL = DeadlinePool(max_workers=_SHADOW_POOL_MAX_WORKERS, name="paw-shadow-deadline")

# Once a task has accumulated this many times `shadow_window` comparisons at one epoch
# without promoting, the runner drops to sampling one comparison in every
# `shadow_window`. This bounds the "an adapter that never promotes doubles compute
# forever" cost without adding a seventh tuning knob. Canonically defined in db.py --
# `get_task_report`/`get_agreement()` need the same threshold to report `stalled`
# (finding 2) without every caller of `TraceDB` alone pulling in this module's
# global runner and atexit hook -- and imported here rather than duplicated.

# Global (not per-task) best-effort drain budget at interpreter exit. Per-task would
# cost N times this for N decorated functions; and with nothing pending the drain
# returns immediately, so a short CLI run that happened to import a decorated module
# pays nothing.
_ATEXIT_DRAIN_SECONDS = 2.0

_RunnerKey = Tuple[str, str]


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
    # J-2: deadline for this job's own run_adapter call (phase="shadow" only --
    # unused, harmless, for phase in ("audit", "fail_open")). Carried per job
    # rather than read from a module constant so a caller's `adapter_timeout_s=`
    # decoration parameter (decorator.py) applies here too, not just the served
    # path.
    adapter_timeout_s: float = 10.0
    # Finding 1: a "fail_open" job carries nothing but enough to key the worker and
    # call `db.increment_fail_open(task_id)` there -- the persisted fail-open counter
    # must never be written on the caller thread. Every field above this one is
    # meaningless for that kind and left at its default.
    kind: str = "compare"

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
        self._seq_cache: Dict[Tuple[_RunnerKey, int], int] = {}
        # J-12: windows actually *scored* at (key, epoch, phase). See
        # `_maybe_transition` -- this is what keeps the window tumbling now that a
        # boundary is evaluated at or past its exact multiple rather than only on it.
        self._windows_scored: Dict[Tuple[_RunnerKey, int, str], int] = {}
        self._error_logged: Set[Tuple[str, str]] = set()
        # Stall-guard subsampling. Random rather than a deterministic "every Nth job"
        # counter: systematic sampling aliases against periodic traffic (a task called
        # in a repeating cycle would have the *same* residue class sampled forever), and
        # the agreement figure `paw-kit report` shows for a stalled task would then be
        # an artefact of that alignment rather than an estimate of its real rate.
        self._stall_rng = random.Random()

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
                what = "comparison" if job.kind == "compare" else "fail-open"
                if first:
                    logger.warning(
                        "paw_kit.jit.shadow: task_id=%s shadow queue is full (size=%d); "
                        "dropping this %s. A dropped comparison is not counted as a "
                        "disagreement -- it is simply not sampled; a dropped fail-open is "
                        "simply not counted. Further drops for this task log at DEBUG.",
                        job.task_id, job.queue_size, what,
                    )
                else:
                    logger.debug(
                        "paw_kit.jit.shadow: task_id=%s shadow queue full, %s dropped.",
                        job.task_id, what,
                    )
                return False
            return True
        except Exception as exc:  # pragma: no cover - defense in depth
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s could not enqueue a comparison (%s: %s).",
                job.task_id, type(exc).__name__, exc,
            )
            return False

    def submit_fail_open(self, task_id: str, db_path: str, db: Any, queue_size: int = 8) -> bool:
        """Enqueue a persisted fail-open counter increment onto the task's worker thread.

        Finding 1: `TraceDB.increment_fail_open` is a synchronous SQLite write, so it
        must never run on the caller thread -- the fail-open path exists specifically
        to protect availability, and making a request wait on a write lock right
        there defeats the point. Shares `submit`'s contract exactly (never blocks,
        never raises, drops the newest job on a full queue): a dropped fail-open is
        simply not counted in the persisted figure, the same way a dropped comparison
        is simply not sampled. The in-process counter in decorator.py is unaffected
        either way -- it is a dict increment, not I/O, and stays on the caller thread.
        """
        return self.submit(
            ShadowJob(
                task_id=task_id,
                db_path=db_path,
                db=db,
                state_epoch=0,
                phase="fail_open",
                input_payload="",
                queue_size=queue_size,
                kind="fail_open",
            )
        )

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
            keep = self._stall_rng.random() < (1.0 / job.shadow_window)
        if first:
            try:
                stats = job.db.get_agreement_stats(
                    job.task_id, job.state_epoch, job.shadow_window, "shadow"
                )
                rate = stats["rate"]
            except Exception:
                rate = None
            # Finding 2: say plainly that this is a one-way door for the rest of this
            # epoch (`_maybe_transition` stops evaluating completed windows for
            # promotion past this point, see its own comment) and name both ways out --
            # a task stuck here otherwise looks, from `get_agreement()`/`paw-kit
            # report` alone, just like one still converging.
            logger.warning(
                "paw_kit.jit.shadow: task_id=%s is not converging: agreement %s over %d "
                "samples at this epoch; the teacher is still serving. From here, "
                "completed windows are no longer evaluated for promotion at this epoch "
                "-- comparisons keep being recorded (sampled one in every %d, so "
                "`get_agreement()`/`paw-kit report` keep moving) but nothing can "
                "promote until the epoch advances. Recover by changing one of this "
                "task's persisted shadow config values (shadow_window, "
                "shadow_threshold, audit_window, demote_threshold) or by passing "
                "shadow_window=0.",
                job.task_id, "unknown" if rate is None else f"{rate:.2f}", seq,
                job.shadow_window,
            )
        if not keep:
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

    def _prune_stale_epoch_state(self, key: _RunnerKey, current_epoch: int) -> None:
        """Drop `_seq_cache`/`_stall_warned` entries for `key` at an older epoch.

        Finding 4: all three structures are keyed on `(key, state_epoch, ...)`, and
        without this
        every epoch a long-lived task ever passed through (a compile retry, a
        promotion, a demotion, a config change) left one entry behind for the life of
        the process. `state_epoch` only ever increases for a given task, so anything
        strictly older than the epoch just observed is safe to drop.
        """
        with self._lock:
            for k in [k for k in self._seq_cache if k[0] == key and k[1] < current_epoch]:
                del self._seq_cache[k]
            for k in [k for k in self._stall_warned if k[0] == key and k[1] < current_epoch]:
                self._stall_warned.discard(k)
            for k in [k for k in self._windows_scored if k[0] == key and k[1] < current_epoch]:
                del self._windows_scored[k]

    def _run_fail_open_job(self, job: ShadowJob) -> None:
        """Worker-thread side of `submit_fail_open` -- the only thing a fail-open job does."""
        try:
            job.db.increment_fail_open(job.task_id)
        except Exception as exc:
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s could not persist a fail-open (%s: %s).",
                job.task_id, type(exc).__name__, exc,
            )

    def _run_job(self, job: ShadowJob) -> None:
        if job.kind == "fail_open":
            self._run_fail_open_job(job)
            return
        self._prune_stale_epoch_state(job.key, job.state_epoch)
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
                # J-2: the same deadline mechanism as the served path, on the
                # separate shadow-path pool -- a wedged adapter must not stall
                # this task's queue forever.
                adapter_value = _SHADOW_DEADLINE_POOL.call(
                    job.run_adapter, job.input_payload, job.adapter_timeout_s
                )
                adapter_latency = (time.perf_counter() - started) * 1000
                adapter_output = stringify_answer(adapter_value)
            except PoolExhausted as exc:
                # Infrastructure, not this adapter's drift (see the pool's own
                # module-level comment above): a different task's wedged
                # comparisons consumed every slot. Excluded from both the
                # promotion numerator and denominator, same as `teacher_error`
                # -- never `error`, which would count against this adapter.
                verdict = "pool_exhausted"
                error_type = type(exc).__name__
                adapter_output = None
                self._log_shadow_error(job.task_id, error_type, exc)
            except BaseException as exc:  # noqa: BLE001
                # Includes DeadlineExceeded: unlike PoolExhausted, this call got
                # its own slot and still didn't finish in time -- that reflects
                # on *this* adapter (too slow, or itself wedged) exactly the way
                # any other adapter exception does, so it stays `error`.
                verdict = "error"
                error_type = type(exc).__name__
                adapter_output = None
                self._log_shadow_error(job.task_id, error_type, exc)
        else:
            started = time.perf_counter()
            try:
                teacher_value_raw = job.run_teacher()  # type: ignore[misc]
                teacher_latency = (time.perf_counter() - started) * 1000
                teacher_output = stringify_answer(teacher_value_raw)
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

        The window is tumbling, not sliding: a 60%-agreement adapter gets one draw per
        window rather than a fresh draw on every single comparison. A sliding window
        promotes such an adapter eventually, which is the exact failure this feature
        exists to prevent.

        J-12: the trigger used to be `seq % window == 0`, tested only at an exact
        multiple and never retried. A boundary that could not be scored -- one
        `teacher_error` row at `shadow_max_pairs == window` was enough -- therefore
        discarded its whole window silently while `seq` kept advancing. The trigger is
        now "at or past the next unscored boundary", with `_windows_scored` recording
        how many windows have actually been scored at this `(task, epoch, phase)`.
        That record is what keeps the window tumbling: without it, evaluating at
        `seq >= boundary` would re-evaluate on every subsequent comparison, which *is*
        a sliding window.

        It also makes the stall budget count scored windows rather than merely
        completed ones, as J-12 asks: with a boundary now retried until it is scored,
        `seq // window` and the scored count agree, so `get_task_report`'s own
        seq-based `stalled` flag (db.py, the other of the two places this count lives)
        stays consistent with the runner without needing a new persisted column.
        """
        window = job.window
        if window <= 0 or seq <= 0:
            return
        cache_key = (job.key, job.state_epoch, job.phase)
        with self._lock:
            scored = self._windows_scored.get(cache_key, 0)
        if seq < (scored + 1) * window:
            return
        if job.phase == "shadow" and seq > _SHADOW_STALL_FACTOR * window:
            # A task that has run `_SHADOW_STALL_FACTOR` full windows at one epoch
            # without ever clearing the threshold is not converging, and from here the
            # runner is only sampling one comparison in every `shadow_window` -- a
            # *systematic* subsample, which aliases badly against periodic traffic and
            # can hand a below-threshold adapter a window drawn entirely from the
            # inputs it happens to get right. Comparisons keep being recorded (so
            # `paw-kit report` and `get_agreement()` still show the drift) but they no
            # longer promote: the fail-safe direction, and the only reading under which
            # "an adapter that disagrees more often than shadow_threshold never serves
            # production traffic" is actually true rather than merely probable. Any
            # epoch bump -- a config change, a demotion -- gives the task a fresh start.
            logger.debug(
                "paw_kit.jit.shadow: task_id=%s window complete at seq=%d but the task is "
                "past the stall point; not evaluating promotion.", job.task_id, seq,
            )
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
        # J-12: returning here no longer burns the window. The next comparison is past
        # the boundary and `_windows_scored` has not advanced, so it is retried.
        if rate is None or samples != window:
            return
        with self._lock:
            # Scored. Catch `_windows_scored` up to where `seq` actually is: a process
            # that started mid-epoch, or a boundary retried a few comparisons late,
            # must not then score several windows back to back.
            self._windows_scored[cache_key] = max(scored + 1, seq // window)
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
