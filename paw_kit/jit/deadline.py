"""Bounded daemon-thread deadline pool for adapter-inference calls (J-2).

The served path (`decorator.py`'s `ready` branch) and the shadow worker
(`shadow.py`'s `_run_job`, shadow phase) both call into arbitrary third-party
backend code with no deadline and no watchdog today: a backend whose `infer`
blocks (a deadlocked llama.cpp, a stalled mmap on a network mount, a socket with
no timeout) never reaches the caller's `except Exception` fail-open at all -- it
just hangs the caller, or the shadow worker's queue, forever.

**Why not `concurrent.futures.ThreadPoolExecutor`.** It was this track's own
leading candidate and the original bug-hunt report's suggested fix, and it does
not work, confirmed by execution during this track's Phase 0 review:
`ThreadPoolExecutor.__init__` has no `daemon` parameter, and its workers are
`daemon=False` on this Python version unconditionally. A module-level pool that
receives repeated calls to a function that never returns, each awaited with
`future.result(timeout=...)`, correctly bounds the *thread count* -- and still
hangs the process forever at interpreter exit (confirmed: `EXIT: 124` after a
20s wall-clock kill). The known
`concurrent.futures.thread._threads_queues.clear()` workaround does not help
either: it defeats `_python_exit`'s own atexit hook, not
`threading._shutdown`'s unconditional join of every live non-daemon thread.
`ThreadPoolExecutor` must not be used for this, in any form.

**The mechanism.** A small, fixed number of `daemon=True` `threading.Thread`
workers, started once at construction (never per call -- an unbounded per-call
`daemon=True` thread was considered and rejected too: it leaks one thread per
wedged call, unbounded, since the served path is per-request). A
`threading.Semaphore` sized to `max_workers` gates *submission*, not just
execution: acquiring a permit is what admits a call to the internal queue at
all, and a permit is released only when that call's worker has actually
finished (successfully, by exception, or is simply still running past the
caller's own deadline -- the *worker* keeps running until the call returns;
only the caller stops waiting). This is what makes the queue capacity-bounded
without a separate `maxsize`: the number of not-yet-finished jobs can never
exceed `max_workers`, because admission requires a permit and permits are
capped at `max_workers`. Once every permit is held by a call that has not
returned, a further call fails open *immediately* (`PoolExhausted`, no
deadline paid) rather than queueing behind a wedged one and paying the
deadline anyway -- verified at Phase 0 by execution: 20 wedged calls against 4
slots cost 2.0s total (4 concurrent slots x 0.5s, then immediate fail-open for
the remaining 16), not 10.0s.

Happy-path overhead measured at Phase 0: ~17 microseconds/call.
"""

import queue
import threading
from typing import Any, Callable, Tuple


class PoolExhausted(TimeoutError):
    """Every worker slot is already held by a call that has not returned.

    Raised immediately (no deadline paid) rather than queueing this call behind
    one that may never finish -- see the module docstring's "mechanism" section.
    """


class DeadlineExceeded(TimeoutError):
    """The call was accepted (a slot was free) but did not finish within the deadline.

    The worker thread itself keeps running the call to completion in the
    background; only the caller stops waiting. The slot is not released until
    the worker actually finishes, which is what makes `PoolExhausted` above the
    correct signal for "this pool is now permanently wedged" rather than a
    fresh `DeadlineExceeded` on every subsequent call.
    """


class DeadlinePool:
    """A fixed pool of daemon worker threads enforcing a per-call deadline.

    One instance per *path* (served, shadow), not per task and not per call --
    see decorator.py/shadow.py for why two separate pools exist rather than one
    shared one. Threads are started once, in the constructor, and never again.
    """

    def __init__(self, max_workers: int = 4, name: str = "paw-deadline") -> None:
        if max_workers < 1:
            raise ValueError(f"max_workers must be >= 1, got {max_workers}")
        self._max_workers = max_workers
        self._queue: "queue.Queue[Tuple[Callable[[Any], Any], Any, dict, threading.Event]]" = queue.Queue()
        # Gates submission, not just execution -- see module docstring. Sized to
        # max_workers so the queue behind it can never hold more than
        # max_workers not-yet-finished jobs: an implicit, construction-time
        # bound rather than a second `maxsize` to keep in sync with it.
        self._free = threading.Semaphore(max_workers)
        self._threads = tuple(
            threading.Thread(
                target=self._worker_loop, name=f"{name}-{i}", daemon=True
            )
            for i in range(max_workers)
        )
        for t in self._threads:
            t.start()

    @property
    def worker_threads(self) -> Tuple[threading.Thread, ...]:
        """Test hook: this pool's own worker threads, for a direct `.daemon` assertion."""
        return self._threads

    def _worker_loop(self) -> None:
        while True:
            fn, arg, box, done = self._queue.get()
            try:
                box["value"] = fn(arg)
            except BaseException as exc:  # noqa: BLE001 - a worker must never die
                box["error"] = exc
            finally:
                done.set()
                self._free.release()

    def call(self, fn: Callable[[Any], Any], arg: Any, deadline_s: float) -> Any:
        """Run `fn(arg)` on a pooled worker, bounded by `deadline_s`.

        Raises `PoolExhausted` immediately if every worker is already occupied
        by a call that has not returned. Raises `DeadlineExceeded` if a slot was
        free but the call did not finish within `deadline_s`. Otherwise returns
        `fn(arg)`'s value, or re-raises whatever `fn` itself raised.
        """
        if not self._free.acquire(blocking=False):
            raise PoolExhausted(
                f"no free worker in this pool (max_workers={self._max_workers}); "
                "failing open immediately rather than queueing behind a call that "
                "may be permanently wedged"
            )
        box: dict = {}
        done = threading.Event()
        self._queue.put((fn, arg, box, done))
        if not done.wait(deadline_s):
            raise DeadlineExceeded(
                f"adapter inference exceeded its {deadline_s}s deadline"
            )
        if "error" in box:
            raise box["error"]
        return box["value"]
