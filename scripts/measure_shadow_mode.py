"""Shadow-mode measurement: does the gate in front of the 60%-agreement adapter actually hold,
what does the audit path cost, and does shadow mode really add no caller latency?

`docs/shadow-mode.md` makes four claims that had never been measured on real hardware:

1. a 60%-agreement adapter never promotes under the shipped defaults (`shadow_window=20`,
   `shadow_threshold=0.8`), and stalls after five windows;
2. `audit_rate=0.05` costs one teacher call per twenty served calls, and `audit_rate=0.0`
   costs none at all;
3. "Nothing in shadow mode ... adds latency to [the caller]";
4. shadow mode creates no new file and `traces.db` stays mode 0600.

This script measures all four against the real adapter from the run that produced the 60%
figure (`measurements/triage-semantic-agreement-3080-20260909-002033.json`) running on the
real `ProgramAsWeightsBackend`, with a **replay teacher**: a pure function that returns the
recorded live-Claude answer for each recorded ticket. No teacher money is spent and the
comparison is still against real teacher labels.

**No compile is performed.** The adapter already exists as
`measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw` (an upstream program id;
weights come from the SDK cache). Every task is put into `shadow` by calling
`TraceDB.set_shadow_started` directly with that manifest, and `threshold` is set to 10**9 so
the compile path is unreachable. `backend.compile` is wrapped in a counter and the count is
written into the JSON; it is expected to be 0.

Limitations, stated up front and repeated in `measurements/README.md`:

* the 20 recorded tickets are cycled (or resampled), so inputs repeat. The adapter is
  deterministic at temperature 0, so per-input agreement is fixed; what is being exercised is
  the **window arithmetic**, not fresh traffic;
* one machine (RTX 3080, CUDA), one run;
* the replay teacher is ~free and instantaneous, which inverts the real latency ratio
  (a live teacher is ~11x *slower* than this adapter, not ~1000x faster).

Usage:
    PAW_API_KEY=paw_sk_...  # only needed if the program is not already in the SDK cache
    uv run python scripts/measure_shadow_mode.py --label 3080 --work-dir /tmp/paw-shadow
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from paw_kit import ProgramAsWeightsBackend
from paw_kit.jit.agreement import field_tolerance_agreement
from paw_kit.jit.decorator import compile_on_hit
from paw_kit.jit.shadow import _GLOBAL_SHADOW_RUNNER
from paw_kit.schema.loader import load as load_adapter

SPEC = (
    "Classify a customer support ticket into priority (low, medium, high, or critical), "
    "department (billing, technical, sales, or general), and urgency_score (integer 1-5)."
)
RECORDED_RUN = "measurements/triage-semantic-agreement-3080-20260909-002033.json"
ADAPTER = "measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw"

# The documented recipe for the exact scoring rule the 60% figure used: priority and
# department exact, urgency_score within 1.
AGREEMENT_FN = field_tolerance_agreement({"urgency_score": 1})


class Triage(BaseModel):
    priority: str
    department: str
    urgency_score: int


# --- log capture ----------------------------------------------------------------

CURRENT_CALL = {"phase": "", "index": 0}
CAPTURED_LOGS: List[Dict[str, Any]] = []


class _Capture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        CAPTURED_LOGS.append({
            "logger": record.name,
            "level": record.levelname,
            "phase": CURRENT_CALL["phase"],
            "call_index": CURRENT_CALL["index"],
            "message": record.getMessage(),
        })


def install_log_capture() -> None:
    """One handler, on the parent logger only: `paw_kit.jit.shadow` propagates to
    `paw_kit.jit`, so attaching to both records every message twice."""
    handler = _Capture(level=logging.INFO)
    logging.getLogger("paw_kit.jit.shadow").setLevel(logging.INFO)
    parent = logging.getLogger("paw_kit.jit")
    parent.setLevel(logging.INFO)
    parent.addHandler(handler)


# --- replay teacher -------------------------------------------------------------

class ReplayTeacher:
    """Returns the recorded live-Claude answer for a recorded ticket. Thread-safe.

    `calls` counts every invocation, including the ones the audit path makes on the
    shadow worker thread -- which is exactly the number experiment 2 exists to measure.
    """

    def __init__(self, answers: Dict[str, Dict[str, Any]]) -> None:
        self._answers = answers
        self._lock = threading.Lock()
        self.calls = 0

    def __call__(self, ticket: str) -> Triage:
        with self._lock:
            self.calls += 1
        return Triage(**self._answers[ticket])


# --- task construction ----------------------------------------------------------

def make_task(
    name: str,
    cache_dir: Path,
    teacher: ReplayTeacher,
    backend: Any,
    **shadow_kwargs: Any,
) -> Callable[..., Any]:
    """Decorate a uniquely-named replay-teacher function. `threshold=10**9`: no compile."""

    def _fn(ticket: str) -> Triage:
        return teacher(ticket)

    _fn.__name__ = name
    _fn.__qualname__ = f"measure_shadow_mode.{name}"
    return compile_on_hit(
        SPEC,
        threshold=10 ** 9,
        response_model=Triage,
        cache_dir=str(cache_dir),
        backend=backend,
        agreement_fn=AGREEMENT_FN,
        **shadow_kwargs,
    )(_fn)


def seed_shadow(wrapped: Any, adapter_path: str) -> int:
    """Put the task into `shadow` with the already-compiled adapter. Returns the epoch."""
    wrapped.db.set_shadow_started(wrapped.task_id, adapter_path)
    return wrapped.db.get_task_routing(wrapped.task_id)[2]


def drain(wrapped: Any, timeout: float = 300.0) -> bool:
    return _GLOBAL_SHADOW_RUNNER.drain(
        wrapped.task_id, timeout=timeout, db_path=str(Path(wrapped.db.db_path))
    )


def runner_stats(wrapped: Any) -> Dict[str, int]:
    return _GLOBAL_SHADOW_RUNNER.stats(wrapped.task_id, str(Path(wrapped.db.db_path)))


def pairs(wrapped: Any) -> List[Dict[str, Any]]:
    with wrapped.db._lock:
        rows = wrapped.db._conn.execute(
            "SELECT id, state_epoch, seq, phase, verdict, error_type, input_payload, "
            "teacher_output, adapter_output, adapter_latency_ms, teacher_latency_ms "
            "FROM shadow_pairs ORDER BY id;"
        ).fetchall()
    return [dict(r) for r in rows]


def pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def latency_block(values: List[float]) -> Dict[str, Any]:
    return {
        "n": len(values),
        "median_ms": statistics.median(values) if values else None,
        "mean_ms": statistics.fmean(values) if values else None,
        "p95_ms": pct(values, 0.95),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
    }


def window_rates(rows: List[Dict[str, Any]], epoch: int, phase: str, window: int) -> List[Dict[str, Any]]:
    """Per-completed-window agreement, computed from the raw rows the runner wrote."""
    countable = [r for r in rows if r["state_epoch"] == epoch and r["phase"] == phase and r["seq"] > 0]
    countable.sort(key=lambda r: r["seq"])
    out = []
    for start in range(0, len(countable) - window + 1, window):
        chunk = countable[start:start + window]
        agree = sum(1 for r in chunk if r["verdict"] == "agree")
        out.append({
            "window": start // window + 1,
            "seq_range": [chunk[0]["seq"], chunk[-1]["seq"]],
            "agree": agree,
            "disagree": sum(1 for r in chunk if r["verdict"] == "disagree"),
            "error": sum(1 for r in chunk if r["verdict"] == "error"),
            "rate": agree / len(chunk),
        })
    return out


# --- experiments ----------------------------------------------------------------

def experiment_defaults(
    tickets: List[str],
    teacher_answers: Dict[str, Dict[str, Any]],
    backend: Any,
    work: Path,
    n_calls: int,
    name: str,
    draw: str,
    seed: int = 0,
) -> Dict[str, Any]:
    """1 / 1b: shipped defaults. Does the 60% adapter promote? Paced so nothing is dropped."""
    cache_dir = work / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = ReplayTeacher(teacher_answers)
    wrapped = make_task(name, cache_dir, teacher, backend)  # all shadow defaults
    epoch = seed_shadow(wrapped, ADAPTER)
    window = wrapped.shadow_config["shadow_window"]
    rng = random.Random(seed)

    snapshots: List[Dict[str, Any]] = []
    latencies: List[float] = []
    last_seq = 0
    promoted_at: Optional[int] = None
    CURRENT_CALL["phase"] = name
    for i in range(n_calls):
        CURRENT_CALL["index"] = i + 1
        ticket = rng.choice(tickets) if draw == "random" else tickets[i % len(tickets)]
        t0 = time.perf_counter()
        wrapped(ticket)
        latencies.append((time.perf_counter() - t0) * 1000)
        drain(wrapped)
        seq = wrapped.db.get_epoch_seq(wrapped.task_id, epoch)
        if seq // window > last_seq // window:
            block = wrapped.get_agreement(n=0)
            snapshots.append({
                "window_completed": seq // window,
                "calls_made": i + 1,
                "seq": seq,
                "rate_over_window": block["rate"],
                "samples": block["samples"],
                "state_after_window": block["state"],
                "stalled": block["stalled"],
            })
        last_seq = seq
        if wrapped.db.get_task_routing(wrapped.task_id)[0] == "ready":
            promoted_at = i + 1
            break

    drain(wrapped)
    rows = pairs(wrapped)
    final = wrapped.get_agreement(n=5)
    stall_logs = [
        entry for entry in CAPTURED_LOGS
        if entry["phase"] == name and "not converging" in entry["message"]
    ]
    per_ticket = {}
    for r in rows:
        if r["phase"] != "shadow":
            continue
        per_ticket.setdefault(r["input_payload"], {"n": 0, "agree": 0, "adapter_output": r["adapter_output"]})
        per_ticket[r["input_payload"]]["n"] += 1
        per_ticket[r["input_payload"]]["agree"] += int(r["verdict"] == "agree")
    return {
        "cache_dir": str(cache_dir),
        "task_id": wrapped.task_id,
        "shadow_config": dict(wrapped.shadow_config),
        "input_draw": draw,
        "seed": seed if draw == "random" else None,
        "calls_made": len(latencies),
        "teacher_calls": teacher.calls,
        "comparisons_recorded": len([r for r in rows if r["seq"] > 0]),
        "promoted": promoted_at is not None,
        "promoted_at_call": promoted_at,
        "final_state": wrapped.db.get_task_routing(wrapped.task_id)[0],
        "per_window": window_rates(rows, epoch, "shadow", window),
        "state_after_each_window": snapshots,
        "stall_warning_fired": bool(stall_logs),
        "stall_warning": stall_logs[0] if stall_logs else None,
        "comparisons_skipped_by_stall_subsampling": runner_stats(wrapped)["stalled"],
        "dropped": runner_stats(wrapped)["dropped"],
        "fail_open_count": wrapped.get_fail_open_count(),
        "final_get_agreement": final,
        "caller_latency_paced": latency_block(latencies),
        "per_ticket_agreement": [
            {"ticket": k, "seen": v["n"], "agreed": v["agree"], "adapter_output": v["adapter_output"]}
            for k, v in per_ticket.items()
        ],
    }


def audit_cost_block(
    teacher_calls: int,
    served: int,
    audit_rate: float,
    serve_calls_requested: int,
    stopped_at_first_audit_window: bool,
    calls_to_first_completed_window: Optional[int],
) -> Dict[str, Any]:
    """The audit-cost fields of an audit arm. Pure, so B-7 is unit testable.

    B-7: a serving loop that stops on the call completing the 20th audit sample yields
    `teacher_calls / served == 20 / N` for a negative-binomial `N`. That is a stopping
    time. Published as `teacher_calls_per_served_call` it reads as a measured rate: the
    2026-09-10 run's 20/621 = 0.032 against a configured 0.05 became "20 teacher calls over
    621 served calls", understating audit cost by ~36%, while the `audit_rate=0.5` arm gave
    20/29 = 0.69 by the same arithmetic.

    `teacher_calls_per_served_call` is therefore reported only when the served-call count
    was fixed before the run. Otherwise it is `None` and the stopping time is reported
    under its own name.
    """
    ratio = (teacher_calls / served) if served else None
    return {
        "audit_rate_configured": audit_rate,
        "stopped_at_first_audit_window": stopped_at_first_audit_window,
        "serve_calls_requested": serve_calls_requested,
        # Only an estimate of the configured audit rate when the number of served calls was
        # fixed in advance. Otherwise None -- see `stopping_time_ratio`.
        "teacher_calls_per_served_call": (
            None if stopped_at_first_audit_window else ratio
        ),
        "teacher_calls_per_served_call_note": (
            "null on purpose: this arm stopped on the call that completed the audit window, "
            "so teacher_calls/served is the stopping time 20/N for a negative-binomial N, "
            "not a rate. See stopping_time_ratio and report B-7."
            if stopped_at_first_audit_window else
            "teacher calls per served call, over a number of served calls fixed before the "
            "run. Comparable to audit_rate_configured."
        ),
        "stopping_time_ratio": ratio if stopped_at_first_audit_window else None,
        # The stopping time itself, named as one.
        "calls_to_first_completed_window": calls_to_first_completed_window,
    }


def experiment_audit(
    tickets: List[str],
    teacher_answers: Dict[str, Dict[str, Any]],
    backend: Any,
    work: Path,
    name: str,
    audit_rate: float,
    serve_calls: int,
    shadow_threshold: float = 0.5,
    demote_threshold: float = 0.4,
    force_promote: bool = False,
    stop_at_first_audit_window: bool = True,
) -> Dict[str, Any]:
    """2 / 2b: promotion with a lowered `shadow_threshold`, then the cost of the audit path.

    B-7 (bug hunt 2026-09-11): with `stop_at_first_audit_window=True` this serving loop
    breaks on the call that completes the 20th audit sample, so `teacher_calls / served`
    is `20 / N` for a negative-binomial `N` -- a stopping time, not an estimate of
    `audit_rate`. This draw gave 20/621 = 0.032 against a configured 0.05 and was published
    as "20 teacher calls over 621 served calls", understating audit cost by ~36%; the
    `audit_rate=0.5` arm gave 20/29 = 0.69 by the same arithmetic. The field was literally
    named `teacher_calls_per_served_call`, which is what made it readable as a rate.

    So: `calls_to_first_completed_window` is the stopping time, named as one, and
    `teacher_calls_per_served_call` is reported **only** when the arm served a fixed number
    of calls (`stop_at_first_audit_window=False`) and the ratio is therefore an estimate of
    the configured rate. When the loop stopped early the field is `None` and
    `stopping_time_ratio` carries `20/N` with a note saying what it is.

    `shadow_threshold=0.5` is a knob turned to make promotion reachable for *this* adapter --
    it is not the shipped default (0.8) and this adapter does not clear the shipped default
    (that is experiment 1). `demote_threshold` must be strictly below it, so 0.4.

    With `force_promote` the task is promoted by calling `TraceDB.try_promote` directly,
    leaving `shadow_threshold` at its shipped value: used by experiment 2c, which is about
    the demotion half of the audit path rather than about promotion.
    """
    cache_dir = work / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = ReplayTeacher(teacher_answers)
    wrapped = make_task(
        name, cache_dir, teacher, backend,
        shadow_threshold=shadow_threshold, demote_threshold=demote_threshold,
        audit_rate=audit_rate,
    )
    epoch = seed_shadow(wrapped, ADAPTER)
    window = wrapped.shadow_config["shadow_window"]
    audit_window = wrapped.shadow_config["audit_window"]

    CURRENT_CALL["phase"] = f"{name}/promote"
    promote_calls = 0
    if force_promote:
        wrapped.db.try_promote(wrapped.task_id, epoch, None, 0, reason="measurement_forced")
    else:
        for i in range(window * 5):
            CURRENT_CALL["index"] = i + 1
            wrapped(tickets[i % len(tickets)])
            promote_calls += 1
            drain(wrapped)
            if wrapped.db.get_task_routing(wrapped.task_id)[0] == "ready":
                break
    promoted = wrapped.db.get_task_routing(wrapped.task_id)[0] == "ready"
    teacher_calls_at_promotion = teacher.calls
    traces_at_promotion = len(wrapped.db.get_traces(wrapped.task_id))
    rows_at_promotion = len(pairs(wrapped))
    ready_epoch = wrapped.db.get_task_routing(wrapped.task_id)[2]

    CURRENT_CALL["phase"] = f"{name}/serve"
    latencies: List[float] = []
    audit_window_completed_at: Optional[int] = None
    served = 0
    for i in range(serve_calls):
        CURRENT_CALL["index"] = i + 1
        t0 = time.perf_counter()
        wrapped(tickets[i % len(tickets)])
        latencies.append((time.perf_counter() - t0) * 1000)
        served += 1
        if audit_window_completed_at is None:
            seq = wrapped.db.get_epoch_seq(wrapped.task_id, ready_epoch)
            if seq >= audit_window:
                drain(wrapped)
                audit_window_completed_at = served
                if stop_at_first_audit_window:
                    break
    drain(wrapped)

    rows = pairs(wrapped)
    audit_rows = [r for r in rows if r["phase"] == "audit"]
    teacher_calls_serving = teacher.calls - teacher_calls_at_promotion
    return {
        "cache_dir": str(cache_dir),
        "task_id": wrapped.task_id,
        "shadow_config": dict(wrapped.shadow_config),
        "shadow_threshold_used": shadow_threshold,
        "demote_threshold_used": demote_threshold,
        "promotion_forced": force_promote,
        "promoted": promoted,
        "calls_to_promotion": promote_calls,
        "promotion_agreement": wrapped.db.get_task_report(wrapped.task_id)["promoted_agreement"],
        "shadow_windows_before_promotion": window_rates(rows, epoch, "shadow", window),
        "served_calls_after_promotion": served,
        "teacher_calls_after_promotion": teacher_calls_serving,
        **audit_cost_block(
            teacher_calls=teacher_calls_serving,
            served=served,
            audit_rate=audit_rate,
            serve_calls_requested=serve_calls,
            stopped_at_first_audit_window=stop_at_first_audit_window,
            calls_to_first_completed_window=audit_window_completed_at,
        ),
        "audit_comparisons_recorded": len(audit_rows),
        "audit_window_completed_after_served_calls": audit_window_completed_at,
        "audit_window_rate": (
            sum(1 for r in audit_rows if r["verdict"] == "agree") / len(audit_rows)
            if audit_rows else None
        ),
        "state_after_audit_window": wrapped.db.get_task_routing(wrapped.task_id)[0],
        "demoted": wrapped.db.get_task_report(wrapped.task_id)["demoted_at"] is not None,
        "trace_rows_written_while_ready": len(wrapped.db.get_traces(wrapped.task_id)) - traces_at_promotion,
        "traces_at_promotion": traces_at_promotion,
        "rows_at_promotion": rows_at_promotion,
        "dropped": runner_stats(wrapped)["dropped"],
        "fail_open_count": wrapped.get_fail_open_count(),
        "caller_latency_ready": latency_block(latencies),
        "final_get_agreement": wrapped.get_agreement(n=5),
    }


def experiment_latency(
    tickets: List[str],
    teacher_answers: Dict[str, Dict[str, Any]],
    backend: Any,
    work: Path,
    n_calls: int,
) -> Dict[str, Any]:
    """3: per-call wall time of the wrapped function in tracing vs shadow vs ready.

    Unpaced: the caller loop runs flat out, exactly the condition under which the claim
    "adds no caller latency" has to hold. In `shadow` that means the bounded queue fills
    and comparisons are dropped -- the dropped count is part of the result.
    """
    name = "exp3_latency"
    cache_dir = work / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = ReplayTeacher(teacher_answers)
    wrapped = make_task(name, cache_dir, teacher, backend)  # shipped defaults
    out: Dict[str, Any] = {"cache_dir": str(cache_dir), "task_id": wrapped.task_id,
                           "shadow_config": dict(wrapped.shadow_config), "n_calls": n_calls}

    def run(phase: str) -> List[float]:
        CURRENT_CALL["phase"] = f"{name}/{phase}"
        lat = []
        for i in range(n_calls):
            CURRENT_CALL["index"] = i + 1
            t0 = time.perf_counter()
            wrapped(tickets[i % len(tickets)])
            lat.append((time.perf_counter() - t0) * 1000)
        return lat

    before = runner_stats(wrapped)
    out["tracing"] = latency_block(run("tracing"))
    out["tracing_state"] = wrapped.db.get_task_routing(wrapped.task_id)[0]

    epoch = seed_shadow(wrapped, ADAPTER)
    shadow_lat = run("shadow")
    mid = runner_stats(wrapped)
    out["shadow"] = latency_block(shadow_lat)
    out["shadow_state"] = wrapped.db.get_task_routing(wrapped.task_id)[0]
    out["shadow_dropped"] = mid["dropped"] - before["dropped"]
    out["shadow_pending_at_end_of_phase"] = mid["pending"]
    drained = drain(wrapped)
    out["shadow_worker_drained_within_timeout"] = drained
    out["shadow_comparisons_recorded"] = len(
        [r for r in pairs(wrapped) if r["phase"] == "shadow" and r["seq"] > 0]
    )
    out["shadow_agreement_after_phase"] = wrapped.get_agreement(n=0)

    # Forced promotion: this adapter cannot earn `ready` at the shipped defaults (that is
    # the point of experiment 1), so the state is set directly to measure the served path.
    epoch = wrapped.db.get_task_routing(wrapped.task_id)[2]
    wrapped.db.try_promote(wrapped.task_id, epoch, None, 0, reason="measurement_forced")
    after_mid = runner_stats(wrapped)
    out["ready"] = latency_block(run("ready"))
    out["ready_state"] = wrapped.db.get_task_routing(wrapped.task_id)[0]
    out["ready_dropped"] = runner_stats(wrapped)["dropped"] - after_mid["dropped"]
    out["ready_note"] = (
        "audit_rate=0.0 here, so `ready` is the bare served path: its wall time is the "
        "adapter's own GPU inference, not shadow-mode overhead."
    )
    out["fail_open_count"] = wrapped.get_fail_open_count()
    out["teacher_calls_total"] = teacher.calls
    return out



def experiment_latency_control(
    tickets: List[str],
    teacher_answers: Dict[str, Dict[str, Any]],
    backend: Any,
    work: Path,
    n_calls: int,
) -> Dict[str, Any]:
    """3b: control for experiment 3, which came out with `shadow` *faster* than `tracing`.

    A wrapper in `tracing` does strictly less work than one in `shadow` (same teacher call,
    same trace write, minus one `put_nowait`), so a faster `shadow` cannot be caused by the
    wrapper. The one thing that differs between those two phases of experiment 3 is that in
    `shadow` a background thread is hammering the GPU. This repeats the `tracing` phase --
    same wrapper, same state, same code path -- with an artificial background load thread
    running adapter inferences, and nothing else changed.
    """
    name = "exp3b_control"
    cache_dir = work / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    teacher = ReplayTeacher(teacher_answers)
    wrapped = make_task(name, cache_dir, teacher, backend)

    def measure(phase: str) -> List[float]:
        CURRENT_CALL["phase"] = f"{name}/{phase}"
        lat = []
        for i in range(n_calls):
            CURRENT_CALL["index"] = i + 1
            t0 = time.perf_counter()
            wrapped(tickets[i % len(tickets)])
            lat.append((time.perf_counter() - t0) * 1000)
        return lat

    idle = measure("tracing_idle")

    stop = threading.Event()
    infers = {"n": 0}

    def loader() -> None:
        fn = load_adapter(adapter_path=ADAPTER, response_model=Triage, backend=backend)
        while not stop.is_set():
            fn(tickets[0])
            infers["n"] += 1

    thread = threading.Thread(target=loader, daemon=True)
    thread.start()
    time.sleep(1.0)  # let the load thread get going
    loaded = measure("tracing_under_load")
    stop.set()
    thread.join(timeout=30)

    governor = None
    try:
        governor = Path(
            "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ).read_text().strip()
    except Exception:
        pass

    return {
        "cache_dir": str(cache_dir),
        "task_id": wrapped.task_id,
        "state_during_both": wrapped.db.get_task_routing(wrapped.task_id)[0],
        "tracing_idle": latency_block(idle),
        "tracing_under_background_gpu_load": latency_block(loaded),
        "background_inferences_run": infers["n"],
        "cpu_governor": governor,
        "note": (
            "Both blocks are the same wrapper in the same `tracing` state, so any difference "
            "between them is the machine responding to background load, not shadow-mode code."
        ),
    }


def sanity(cache_dirs: Dict[str, Path]) -> Dict[str, Any]:
    """4: `paw-kit report` output, file modes, and what files exist in the cache dir.

    `traces.db` runs in WAL mode, so `traces.db-wal` / `traces.db-shm` are expected
    alongside it; the claim under test is that *shadow mode* adds no file of its own.
    """
    exe = Path(sys.executable).with_name("paw-kit")
    env = dict(os.environ, COLUMNS="120")
    sqlite_files = {"traces.db", "traces.db-wal", "traces.db-shm"}
    out: Dict[str, Any] = {}
    for name, cache_dir in cache_dirs.items():
        db_path = cache_dir / "traces.db"
        before = sorted(p.name for p in cache_dir.iterdir())
        if exe.exists():
            cmd = [str(exe), "report", "--db", str(db_path)]
        else:  # pragma: no cover - console script always present in this venv
            cmd = [sys.executable, "-m", "paw_kit.cli", "report", "--db", str(db_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        after = sorted(p.name for p in cache_dir.iterdir())
        out[name] = {
            "files_before_report": before,
            "files_after_report": after,
            "file_modes": {p.name: oct(p.stat().st_mode & 0o777) for p in sorted(cache_dir.iterdir())},
            "no_file_beyond_sqlite": set(before) <= sqlite_files and set(after) <= sqlite_files,
            "traces_db_mode": oct(db_path.stat().st_mode & 0o777),
            "traces_db_mode_is_0600": (db_path.stat().st_mode & 0o777) == 0o600,
            "cache_dir_mode": oct(cache_dir.stat().st_mode & 0o777),
            "paw_kit_report_returncode": proc.returncode,
            "paw_kit_report_stdout": proc.stdout,
            "paw_kit_report_stderr": proc.stderr,
        }
    return out


def binomial_residual(p: float, window: int, threshold: float, windows: int) -> Dict[str, Any]:
    """Exact chance a genuinely random p-agreement adapter clears `threshold` by luck.

    `docs/shadow-mode.md` states this residual as "roughly a 2.5% chance" over the first
    five windows at the shipped defaults; this is the arithmetic for that claim.
    """
    need = math.ceil(threshold * window)
    per_window = sum(
        math.comb(window, k) * p ** k * (1 - p) ** (window - k)
        for k in range(need, window + 1)
    )
    return {
        "p_agreement": p,
        "window": window,
        "threshold": threshold,
        "successes_needed": need,
        "p_one_window_clears": per_window,
        "windows_before_stall": windows,
        "p_promotes_within_stall": 1 - (1 - per_window) ** windows,
        "doc_claim": "roughly a 2.5% chance of producing one 16-of-20 window",
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--label", default="3080")
    ap.add_argument("--out-dir", default="measurements")
    ap.add_argument("--work-dir", default=None,
                    help="Where the per-experiment .paw cache dirs live. Default: a temp dir.")
    ap.add_argument("--calls", type=int, default=200,
                    help="Calls for experiments 1/1b and each phase of experiment 3.")
    ap.add_argument("--serve-calls", type=int, default=1200,
                    help="Cap on served calls while hunting one completed audit window.")
    ap.add_argument("--fixed-serve-calls", type=int, default=1200,
                    help="Served calls for experiment 2d, the fixed-N audit-cost arm. "
                         "Fixed before the run, so its teacher-calls-per-served-call IS an "
                         "estimate of audit_rate (B-7).")
    ap.add_argument("--binomial-p", type=float, default=None,
                    help="Per-call agreement probability for the binomial residual. "
                         "Default: derived from the recorded run, preferring its held-out "
                         "rate over its all-rows rate. The 2026-09-09 run's 0.6 is itself "
                         "downstream of B-1's leak; the held-out figure is 0.467.")
    ap.add_argument("--keep-work-dir", action="store_true")
    args = ap.parse_args()

    install_log_capture()

    recorded = json.loads(Path(RECORDED_RUN).read_text())
    tickets = [c["ticket"] for c in recorded["cases"]]
    teacher_answers = {c["ticket"]: c["teacher_fresh"] for c in recorded["cases"]}
    recorded_adapter = {c["ticket"]: c["adapter_parsed"] for c in recorded["cases"]}

    # B-7 / B-1: `p` for the binomial residual used to be the literal 0.6 at the call site,
    # which is the leaked all-rows rate from the 2026-09-09 triage run. Prefer the recorded
    # run's held-out rate when it has one (runs from the fixed
    # measure_triage_semantic_agreement.py do), fall back to the all-rows rate, and let
    # --binomial-p override either. Whichever it is, say so in the artifact.
    if args.binomial_p is not None:
        binomial_p = args.binomial_p
        binomial_p_source = "--binomial-p on the command line"
    elif recorded.get("full_agreement_rate_heldout") is not None:
        binomial_p = recorded["full_agreement_rate_heldout"] / 100.0
        binomial_p_source = f"{RECORDED_RUN}:full_agreement_rate_heldout"
    else:
        binomial_p = recorded["full_agreement_rate"] / 100.0
        binomial_p_source = (
            f"{RECORDED_RUN}:full_agreement_rate -- all 20 scored rows, five of which were "
            "folded into the adapter's own spec (report B-1). The held-out rate is 46.7%; "
            "pass --binomial-p 0.467 for the corrected residual."
        )
    print(f"[binomial] p={binomial_p} from {binomial_p_source}")

    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)
    compile_calls = {"n": 0, "args": []}
    _orig_compile = backend.compile

    def _counted_compile(*a: Any, **k: Any) -> Any:  # pragma: no cover - must never fire
        compile_calls["n"] += 1
        compile_calls["args"].append(str(a)[:200])
        return _orig_compile(*a, **k)

    backend.compile = _counted_compile  # type: ignore[method-assign]

    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="paw-shadow-"))
    work.mkdir(parents=True, exist_ok=True)

    started = time.time()
    results: Dict[str, Any] = {}

    print(f"[work] {work}")
    print("[1] shipped defaults, cyclic inputs")
    results["experiment_1_defaults_cyclic"] = experiment_defaults(
        tickets, teacher_answers, backend, work, args.calls, "exp1_defaults_cyclic", "cyclic"
    )
    r1 = results["experiment_1_defaults_cyclic"]
    print(f"    promoted={r1['promoted']} windows={[w['rate'] for w in r1['per_window']]} "
          f"stall_warning={r1['stall_warning_fired']}")

    print("[1b] shipped defaults, random draw with replacement (seed 0)")
    results["experiment_1b_defaults_random"] = experiment_defaults(
        tickets, teacher_answers, backend, work, args.calls, "exp1b_defaults_random",
        "random", seed=0,
    )
    r1b = results["experiment_1b_defaults_random"]
    print(f"    promoted={r1b['promoted']} windows={[w['rate'] for w in r1b['per_window']]}")

    print("[2] audit path cost at audit_rate=0.05")
    results["experiment_2_audit_rate_005"] = experiment_audit(
        tickets, teacher_answers, backend, work, "exp2_audit005", 0.05, args.serve_calls
    )
    r2 = results["experiment_2_audit_rate_005"]
    print(f"    promoted_at={r2['calls_to_promotion']} served={r2['served_calls_after_promotion']} "
          f"teacher_calls={r2['teacher_calls_after_promotion']} "
          f"audit_window_after={r2['audit_window_completed_after_served_calls']}")

    print(f"[2d] audit path cost at audit_rate=0.05 over a FIXED {args.fixed_serve_calls} "
          "served calls (B-7)")
    results["experiment_2d_audit_rate_005_fixed_n"] = experiment_audit(
        tickets, teacher_answers, backend, work, "exp2d_audit005_fixedn", 0.05,
        args.fixed_serve_calls, stop_at_first_audit_window=False,
    )
    r2d = results["experiment_2d_audit_rate_005_fixed_n"]
    print(f"    served={r2d['served_calls_after_promotion']} "
          f"teacher_calls={r2d['teacher_calls_after_promotion']} "
          f"teacher_calls_per_served_call={r2d['teacher_calls_per_served_call']} "
          f"(configured {r2d['audit_rate_configured']}) "
          f"first_window_after={r2d['calls_to_first_completed_window']}")

    print("[2b] audit_rate=0.0 makes zero teacher calls after promotion")
    results["experiment_2b_audit_rate_zero"] = experiment_audit(
        tickets, teacher_answers, backend, work, "exp2b_audit000", 0.0, 200
    )
    r2b = results["experiment_2b_audit_rate_zero"]
    print(f"    served={r2b['served_calls_after_promotion']} "
          f"teacher_calls={r2b['teacher_calls_after_promotion']}")

    print("[2c] does the audit path demote? demote_threshold nudged to 0.61")
    results["experiment_2c_demotion"] = experiment_audit(
        tickets, teacher_answers, backend, work, "exp2c_demote", 0.5, 200,
        shadow_threshold=0.8, demote_threshold=0.61, force_promote=True,
    )
    r2c = results["experiment_2c_demotion"]
    print(f"    served={r2c['served_calls_after_promotion']} "
          f"audit_rate={r2c['audit_window_rate']} demoted={r2c['demoted']} "
          f"state={r2c['state_after_audit_window']}")

    print("[3] caller-path latency: tracing vs shadow vs ready")
    results["experiment_3_caller_latency"] = experiment_latency(
        tickets, teacher_answers, backend, work, args.calls
    )
    r3 = results["experiment_3_caller_latency"]
    print(f"    tracing p50={r3['tracing']['median_ms']:.3f}ms "
          f"shadow p50={r3['shadow']['median_ms']:.3f}ms "
          f"ready p50={r3['ready']['median_ms']:.3f}ms dropped={r3['shadow_dropped']}")

    print("[3b] control: same tracing wrapper, with and without background GPU load")
    results["experiment_3b_latency_control"] = experiment_latency_control(
        tickets, teacher_answers, backend, work, args.calls
    )
    r3b = results["experiment_3b_latency_control"]
    print(f"    tracing idle p50={r3b['tracing_idle']['median_ms']:.3f}ms "
          f"tracing under load p50={r3b['tracing_under_background_gpu_load']['median_ms']:.3f}ms "
          f"governor={r3b['cpu_governor']}")

    print("[4] sanity: paw-kit report, file modes, file listing")
    results["experiment_4_sanity"] = sanity({
        "exp1_defaults_cyclic": work / "exp1_defaults_cyclic",
        "exp1b_defaults_random": work / "exp1b_defaults_random",
        "exp2_audit005": work / "exp2_audit005",
        "exp2b_audit000": work / "exp2b_audit000",
        "exp2c_demote": work / "exp2c_demote",
        "exp3_latency": work / "exp3_latency",
        "exp3b_control": work / "exp3b_control",
    })

    live_vs_recorded = [
        {
            "ticket": row["ticket"],
            "recorded_adapter": recorded_adapter[row["ticket"]],
            "live_adapter": json.loads(row["adapter_output"]) if row["adapter_output"] else None,
            "same": (json.loads(row["adapter_output"]) == recorded_adapter[row["ticket"]])
            if row["adapter_output"] else False,
        }
        for row in r1["per_ticket_agreement"]
    ]

    summary = {
        "label": args.label,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "duration_s": time.time() - started,
        "machine": "RTX 3080 (desktop), CUDA, llama-cpp-python CUDA build",
        "spec": SPEC,
        "adapter_manifest": json.loads(Path(ADAPTER).read_text()),
        "recorded_run_reused": RECORDED_RUN,
        "recorded_full_agreement_rate": recorded["full_agreement_rate"],
        "teacher": (
            "replay: a pure function returning the recorded live-Claude answer for each of the "
            "20 recorded tickets. No teacher API call is made by this script."
        ),
        "agreement_fn": "field_tolerance_agreement({'urgency_score': 1})",
        "compile_calls": compile_calls["n"],
        "compile_calls_note": (
            "backend.compile is wrapped in a counter for the whole run; every task is put into "
            "`shadow` by calling TraceDB.set_shadow_started directly with the existing manifest, "
            "and threshold=10**9 makes the compile path unreachable."
        ),
        "work_dir": str(work),
        "binomial_residual": binomial_residual(binomial_p, 20, 0.8, 5),
        "binomial_p_source": binomial_p_source,
        "live_adapter_vs_recorded_adapter": live_vs_recorded,
        "captured_logs": CAPTURED_LOGS,
        "limitations": [
            "Replay teacher: the teacher's answers are the recorded ones from "
            f"{RECORDED_RUN}, not fresh calls. Real teacher drift is therefore invisible here.",
            "Repeated inputs: the same 20 tickets are cycled (experiments 1, 2, 3) or resampled "
            "with replacement (1b). The adapter is deterministic at temperature 0, so agreement "
            "per input is fixed and what is exercised is the window arithmetic, not fresh traffic.",
            "The replay teacher is instantaneous and free, which inverts the real latency ratio: "
            "a live teacher is roughly 11x slower than this adapter (see the JIT section of "
            "measurements/README.md), not ~1000x faster.",
            "One machine (RTX 3080, CUDA), one run, one adapter, one spec.",
            "Experiment 2 lowers shadow_threshold from the shipped 0.8 to 0.5 so this adapter can "
            "reach `ready` at all; that is a knob turned to make the audit path measurable, not a "
            "recommendation and not the shipped behaviour.",
            "Experiment 3's `ready` phase is entered by calling TraceDB.try_promote directly, for "
            "the same reason.",
        ],
        **results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"shadow-mode-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\ncompile calls: {compile_calls['n']}")
    print(f"wrote {out_path}")

    if not args.keep_work_dir and args.work_dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
