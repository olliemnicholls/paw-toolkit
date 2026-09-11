"""Direct, non-end-to-end tests of `paw_kit.jit.deadline.DeadlinePool` (J-2).

Every property the track's Dependency check requires is checked directly here:
daemon-ness by attribute, bounded thread count by construction, immediate
fail-open once exhausted by wall-clock, and a capacity-bounded backlog by
semaphore accounting -- never by an end-to-end "the process didn't hang" test,
which cannot fail fast for this exact defect shape (see
`tests/test_jit_shadow.py::test_shadow_worker_thread_is_a_daemon`, the existing
test this file's daemon test is modelled on).
"""

import threading
import time

import pytest

from paw_kit.jit.deadline import DeadlineExceeded, DeadlinePool, PoolExhausted


def test_pool_worker_threads_are_daemons() -> None:
    """Every worker thread this pool starts must be `daemon=True`, checked directly.

    Not end-to-end: if this were ever non-daemon, nothing pulls the worker off
    `queue.Queue.get()`'s blocking wait once the test session is otherwise done,
    and the interpreter would wait forever to join it at exit -- undetectable by
    any bounded-timeout test, only by this direct attribute check.
    """
    pool = DeadlinePool(max_workers=3, name="daemon-check")
    assert len(pool.worker_threads) == 3
    for t in pool.worker_threads:
        assert t.daemon is True, "every DeadlinePool worker thread must be daemon=True"
        assert t.is_alive()


def test_thread_count_does_not_grow_under_repeated_wedging() -> None:
    """Many calls to a function that never returns must not leak one thread each.

    Threads are started once, at construction, and never again -- this is what a
    per-call `daemon=True` thread (considered and rejected at Phase 0) would get
    wrong: it bounds nothing, since the served path is per-request.
    """
    pool = DeadlinePool(max_workers=3, name="bounded-count")
    block_forever = threading.Event()

    def wedge(x: int) -> int:
        block_forever.wait()
        return x

    def make_call() -> None:
        try:
            pool.call(wedge, 1, 0.05)
        except TimeoutError:
            pass

    threads = [threading.Thread(target=make_call) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    pool_threads = [t for t in threading.enumerate() if t.name.startswith("bounded-count")]
    assert len(pool_threads) == 3, "thread count must stay fixed at max_workers, not grow with load"

    block_forever.set()  # release the (daemon) workers so they don't linger


def test_wedged_calls_past_capacity_fail_open_immediately_not_after_deadline() -> None:
    """Phase 0's own probe methodology, re-executed directly against the shipped class.

    20 calls against a 4-worker pool, each wedged forever, with a 0.2s deadline:
    the 4 that get a slot each pay the deadline once; the other 16 must fail open
    in ~0ms, not queue behind a wedged call and pay the deadline anyway. Total
    wall-clock time must therefore read as roughly one deadline period, not five.
    """
    pool = DeadlinePool(max_workers=4, name="fast-fail")
    block_forever = threading.Event()
    deadline = 0.2
    n = 20
    latencies = []
    lock = threading.Lock()

    def wedge(x: int) -> int:
        block_forever.wait()
        return x

    def worker(i: int) -> None:
        started = time.perf_counter()
        try:
            pool.call(wedge, i, deadline)
        except TimeoutError:
            pass
        elapsed = time.perf_counter() - started
        with lock:
            latencies.append(elapsed)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    total_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    total = time.perf_counter() - total_start
    block_forever.set()

    assert total < deadline * 2 + 1.0, (
        f"20 wedged calls against 4 slots took {total:.2f}s -- expected roughly one "
        f"deadline period ({deadline}s), not five (serial) worth"
    )
    fast = sorted(latencies)[: n - 4]
    assert max(fast) < 0.05, (
        f"a call past pool capacity took {max(fast):.3f}s instead of failing open "
        "immediately (~0ms)"
    )


def test_pool_exhausted_raised_the_instant_every_slot_is_held() -> None:
    """Directly: with every permit held, `.call()` raises PoolExhausted in ~0ms."""
    pool = DeadlinePool(max_workers=2, name="exhaustion-direct")
    assert pool._free.acquire(blocking=False)
    assert pool._free.acquire(blocking=False)
    try:
        started = time.perf_counter()
        with pytest.raises(PoolExhausted):
            pool.call(lambda x: x, 1, 5.0)
        elapsed = time.perf_counter() - started
        assert elapsed < 0.05
    finally:
        pool._free.release()
        pool._free.release()


def test_queue_backlog_cannot_exceed_pool_capacity() -> None:
    """The queue behind the pool must not accept an unbounded backlog.

    With every permit held, every further call must be refused at admission
    (PoolExhausted) rather than queued -- so the internal queue can never carry
    more than `max_workers` not-yet-finished jobs, a *staleness* hazard distinct
    from `shadow.py`'s own `shadow_queue_size` (which governs comparison
    dropping under load for an unrelated reason).
    """
    pool = DeadlinePool(max_workers=2, name="backlog-bound")
    block_forever = threading.Event()

    def wedge(x: int) -> int:
        block_forever.wait()
        return x

    occupiers = [threading.Thread(target=lambda: _safe_call(pool, wedge, 5.0)) for _ in range(2)]
    for t in occupiers:
        t.start()
    # Wait for both permits to actually be consumed (deterministic: poll the
    # semaphore's own counter rather than a fixed sleep).
    deadline = time.monotonic() + 2.0
    while pool._free.acquire(blocking=False):
        pool._free.release()
        if time.monotonic() > deadline:
            raise AssertionError("occupier threads never acquired their permits")
        time.sleep(0.005)

    rejected = 0
    for _ in range(50):
        try:
            pool.call(wedge, 0, 5.0)
        except PoolExhausted:
            rejected += 1
    assert rejected == 50, "every call past capacity must be rejected, not queued"
    assert pool._queue.qsize() <= 2, (
        f"queue backlog grew to {pool._queue.qsize()}, past max_workers=2"
    )

    block_forever.set()
    for t in occupiers:
        t.join(timeout=5)


def _safe_call(pool: DeadlinePool, fn, deadline_s: float) -> None:
    try:
        pool.call(fn, None, deadline_s)
    except TimeoutError:
        pass


def test_happy_path_overhead_is_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """Microbenchmark: overhead over a direct call, budgeted against Phase 0's
    measured ~17us/call. A generous bound (500us) is used here to avoid CI/sandbox
    timing flakiness while still catching a gross regression (e.g. accidental lock
    contention); see the track doc for the full-precision number as measured."""
    pool = DeadlinePool(max_workers=4, name="overhead-check")

    def fast(x: int) -> int:
        return x

    n = 5000
    started = time.perf_counter()
    for i in range(n):
        pool.call(fast, i, 5.0)
    via_pool_s = time.perf_counter() - started

    started = time.perf_counter()
    for i in range(n):
        fast(i)
    direct_s = time.perf_counter() - started

    overhead_per_call_us = (via_pool_s - direct_s) / n * 1e6
    assert overhead_per_call_us < 500, (
        f"DeadlinePool overhead is {overhead_per_call_us:.1f}us/call, budget is 500us "
        "(Phase 0 measured ~17us/call for the same mechanism)"
    )


def test_exceptions_from_fn_propagate_through_call() -> None:
    pool = DeadlinePool(max_workers=2, name="propagate-check")

    def boom(x: int) -> int:
        raise ValueError(f"boom {x}")

    with pytest.raises(ValueError, match="boom 1"):
        pool.call(boom, 1, 5.0)


def test_pool_exhausted_and_deadline_exceeded_are_timeout_errors() -> None:
    assert issubclass(PoolExhausted, TimeoutError)
    assert issubclass(DeadlineExceeded, TimeoutError)


def test_deadline_exceeded_raised_when_slot_free_but_call_too_slow() -> None:
    pool = DeadlinePool(max_workers=1, name="deadline-exceeded-check")
    release = threading.Event()

    def slow(x: int) -> int:
        release.wait(2.0)
        return x

    with pytest.raises(DeadlineExceeded):
        pool.call(slow, 1, 0.05)
    release.set()


def test_max_workers_must_be_positive() -> None:
    with pytest.raises(ValueError):
        DeadlinePool(max_workers=0)
