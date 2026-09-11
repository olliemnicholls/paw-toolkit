"""Real @compile_on_hit hot-swap measurement: a live Claude teacher vs the compiled
local adapter, on the same task, same process.

Every prior demo of paw.jit (`paw-kit demo`, `examples/triage_ticket/run.py`) fakes
BOTH sides of this comparison: `time.sleep(0.2)` standing in for "the remote API" and a
hardcoded if/elif dict lookup standing in for "the compiled model". This script is the
first time either side is real: the "teacher" is an actual Claude API call, and the
compiled adapter is ProgramAsWeightsBackend running the real upstream compile + local
llama.cpp inference (see scripts/measure_real_backend.py, which proved that path works).

Uses sync_compile=True so the hot-swap point is exact (the call that crosses `threshold`
blocks until compilation finishes, rather than racing a background thread) -- that's a
deliberate simplification for a clean before/after measurement, not how you'd use this
decorator in production (there you want threshold calls to stay fast and let compilation
happen in the background).

Prerequisites:
    pip install anthropic
    pip install programasweights --extra-index-url https://pypi.programasweights.com/simple/
    export ANTHROPIC_API_KEY=sk-ant-...
    export PAW_API_KEY=paw_sk_...

Usage:
    uv run python scripts/measure_jit_speedup.py --threshold 5 --total-calls 20 --label 3080
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel

from paw_kit import ProgramAsWeightsBackend, compile_on_hit

try:
    import anthropic
except ImportError:
    # Deferred, not exit(2) here: this runs at import time, before split_latencies/
    # SPLIT_PHASES below are even defined. tests/test_measurement_scripts.py imports this
    # module by path specifically to exercise those pure functions offline, with no
    # anthropic installed and no network -- exiting here made that impossible, so every
    # test in that file failed importing the module rather than testing anything (found
    # 2026-09-11 reproducing a red CI run outside any dev .venv that happens to have
    # anthropic already installed, which is why this was never seen locally).
    anthropic = None  # type: ignore[assignment]

TEACHER_MODEL = "claude-haiku-4-5-20251001"

SPEC = (
    "Classify a customer support ticket into priority (low, medium, high, or critical), "
    "department (billing, technical, sales, or general), and urgency_score (integer 1-5)."
)

# Same ticket set as examples/triage_ticket/run.py, plus extra so a 20-call run doesn't
# repeat an input verbatim (repeats are a legitimate trace, but varied inputs make a more
# honest test of what the compiled adapter actually learned).
TICKETS = [
    "I was charged twice on my credit card for invoice #INV-9821. Please refund immediately!",
    "Our production API gateway is throwing 502 Bad Gateway errors across all regions!",
    "We are interested in an enterprise annual contract for 250 seats. Who can we talk to?",
    "How do I update my profile picture in the settings page? I can't find the upload button.",
    "Database replication lag is exceeding 45 minutes on our primary PostgreSQL cluster.",
    "Can you provide a copy of your SOC-2 Type II audit report for our compliance review?",
    "My password reset email is never arriving in my inbox or spam folder.",
    "We need to add 5 more team members to our billing subscription plan.",
    "The iOS app crashes immediately upon opening on iOS 18 beta.",
    "Thank you for the quick support yesterday, everything is working smoothly now!",
    "Getting a 403 Forbidden error when calling the /v2/export endpoint since this morning.",
    "Please cancel my subscription and confirm the final invoice amount.",
    "Is there a discount for non-profit organizations on the enterprise tier?",
    "The dashboard chart colors are unreadable in dark mode, can this be fixed?",
    "Urgent: our webhook deliveries have been failing silently for 3 hours, losing orders.",
    "Just wanted to say the new onboarding flow is much clearer, nice work.",
    "Requesting a data export of all our account's usage logs for the last 12 months.",
    "Two-factor authentication codes via SMS aren't arriving for EU phone numbers.",
    "What's the SLA for critical incidents on the enterprise support plan?",
    "The billing page shows a negative balance that doesn't match our invoice history.",
]


class Triage(BaseModel):
    priority: str
    department: str
    urgency_score: int


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# --------------------------------------------------------------------------- the split
#
# B-2 (bug hunt 2026-09-11): the summary used to split the per-call latencies at
# `threshold` and publish the two means as "before" and "after". With `sync_compile=True`
# that is wrong in both directions, because two of the twenty calls are neither a teacher
# call nor steady-state local inference:
#
#   * call `threshold` itself pays the full synchronous compile on top of its teacher call
#     (5449.66 ms in the committed 2026-09-08 run), which inflated the "before" mean from
#     987.33 ms to 1879.80 ms;
#   * call `threshold + 1` is the first *local* call, and pays the adapter download plus
#     the llama.cpp model load (7599.32 ms), which inflated the "after" mean from
#     88.37 ms to 589.10 ms -- and made the old `post_threshold.note` ("no network call")
#     untrue of that one call.
#
# Net effect: the artifact's own `speedup_x` read 3.19 while every published page said
# ~11x. The split is now a pure function of `(latencies, threshold)` so it can be unit
# tested against the recorded run, and `speedup_x` has one stated definition.

#: The four phases of a `sync_compile=True` hot-swap run, in the order they occur.
SPLIT_PHASES = ("teacher_only", "compile_call", "cold_first_local", "steady_state")

#: How `speedup_x` is defined, recorded in the artifact so a reader need not guess.
SPEEDUP_DEFINITION = "mean(teacher_only) / mean(steady_state)"


def split_latencies(latencies: "Sequence[float]", threshold: int) -> dict[str, list[float]]:
    """Partition per-call latencies into the four phases above. Pure; no I/O, no globals.

    `latencies[i]` is call `i + 1`. With a threshold of 5 and 20 calls:

        teacher_only      calls 1-4    teacher API call only
        compile_call      call  5      teacher API call + the whole synchronous compile
        cold_first_local  call  6      first local call: adapter download + model load
        steady_state      calls 7-20   warm local llama.cpp inference

    Phases that the run was too short to reach come back empty rather than raising, so a
    `--total-calls 6` run still produces a well-formed (if uninteresting) summary.
    """
    if threshold < 1:
        raise ValueError(f"threshold must be >= 1, got {threshold}")
    lat = list(latencies)
    return {
        "teacher_only": lat[: threshold - 1],
        "compile_call": lat[threshold - 1 : threshold],
        "cold_first_local": lat[threshold : threshold + 1],
        "steady_state": lat[threshold + 1 :],
    }


def phase_stats(latencies: "Sequence[float]", threshold: int) -> dict[str, Any]:
    """`split_latencies` plus the per-phase arithmetic and `speedup_x`. Also pure.

    Verified against `measurements/jit-speedup-3080-20260908-165914.json`'s `calls[]`:
    teacher_only 987.33 ms, compile_call 5449.66 ms, cold_first_local 7599.32 ms,
    steady_state 88.37 ms, speedup_x 11.17.
    """
    split = split_latencies(latencies, threshold)
    first_call_of = {
        "teacher_only": 1,
        "compile_call": threshold,
        "cold_first_local": threshold + 1,
        "steady_state": threshold + 2,
    }
    phases = {
        name: {
            "n": len(split[name]),
            "first_call": first_call_of[name] if split[name] else None,
            "mean_ms": statistics.fmean(split[name]) if split[name] else None,
            "ms": list(split[name]),
        }
        for name in SPLIT_PHASES
    }
    teacher_mean = phases["teacher_only"]["mean_ms"]
    steady_mean = phases["steady_state"]["mean_ms"]
    return {
        "phases": phases,
        "speedup_definition": SPEEDUP_DEFINITION,
        "speedup_x": (teacher_mean / steady_mean) if teacher_mean and steady_mean else None,
    }


def call_teacher(client: "anthropic.Anthropic", ticket_body: str) -> Triage:
    """A real frontier API call. This is the function the user's own code would already
    have -- @compile_on_hit wraps it unmodified."""
    resp = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{SPEC}\n\nRespond with ONLY a JSON object with exactly the keys "
                    f'"priority", "department", "urgency_score". No other text.\n\n'
                    f"Ticket: {ticket_body}"
                ),
            }
        ],
    )
    text = "".join(block.text for block in resp.content if hasattr(block, "text"))
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"Teacher returned no JSON: {text!r}")
    data = json.loads(match.group(0))
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return Triage(**data), usage  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=int, default=5)
    ap.add_argument("--total-calls", type=int, default=20)
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--cache-dir", default="./.paw_jit_speedup_demo")
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    if anthropic is None:
        print("pip install anthropic", file=sys.stderr)
        return 2

    if args.total_calls <= args.threshold:
        print("--total-calls must exceed --threshold to observe any post-swap calls", file=sys.stderr)
        return 2

    shutil.rmtree(args.cache_dir, ignore_errors=True)  # start from a clean trace DB each run

    client = anthropic.Anthropic()
    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)

    teacher_usage: list[dict] = []

    def teacher_fn(ticket_body: str) -> Triage:
        result, usage = call_teacher(client, ticket_body)
        teacher_usage.append(usage)
        return result

    decorated = compile_on_hit(
        spec=SPEC,
        threshold=args.threshold,
        response_model=Triage,
        backend=backend,
        cache_dir=args.cache_dir,
        sync_compile=True,  # exact hot-swap point for a clean before/after split
        # Track 14: pinned so this script keeps the hot-swap-immediately behaviour the
        # numbers in measurements/README.md were produced under. With the default
        # shadow_window this adapter (60% agreement, measured) would never promote and
        # the recorded run would not be reproducible. Measuring shadow mode itself is a
        # separate measurement.
        shadow_window=0,
    )(teacher_fn)

    calls: list[dict] = []
    for i in range(args.total_calls):
        ticket = TICKETS[i % len(TICKETS)]
        billed_before = len(teacher_usage)
        t0 = time.perf_counter()
        out = decorated(ticket)
        dt_ms = (time.perf_counter() - t0) * 1000
        # B-8h: `phase` is what the call *index* says should have happened; `served_by` is
        # what actually did. `teacher_fn` appends to `teacher_usage` on every real teacher
        # call, so a post-threshold fail-open fallback -- which would otherwise be averaged
        # into the "local inference" mean while the script printed "they never left the
        # machine" -- is visible per call instead of being assumed away.
        phase = "teacher_or_compile" if i < args.threshold else "adapter_or_fallback"
        served_by = "teacher" if len(teacher_usage) > billed_before else "adapter"
        calls.append({"call": i + 1, "ms": dt_ms, "phase": phase,
                      "served_by": served_by, "output": out.model_dump()})
        print(f"call {i + 1:2d} [{phase:18s}] served_by={served_by:7s} {dt_ms:8.1f}ms -> {out.model_dump()}")

    pre = [c["ms"] for c in calls[: args.threshold]]
    post = [c["ms"] for c in calls[args.threshold :]]
    teacher_served_after_threshold = sum(
        1 for c in calls[args.threshold :] if c["served_by"] == "teacher"
    )

    summary = {
        "label": args.label,
        "teacher_model": TEACHER_MODEL,
        "threshold": args.threshold,
        "total_calls": args.total_calls,
        # Kept for continuity with the 2026-09-08 artifacts, and superseded: a bare split
        # at `threshold` puts the synchronous compile in "pre" and the adapter download in
        # "post". Read `phases` instead. (B-2.)
        "pre_threshold": {
            "n": len(pre),
            "mean_ms": statistics.fmean(pre) if pre else None,
            "note": "index split only: the real Claude API call latency each time, and the "
            "threshold-th call additionally includes the full synchronous compile step, so "
            "this mean is not the cost of a teacher call. Superseded by phases.teacher_only.",
        },
        "post_threshold": {
            "n": len(post),
            "mean_ms": statistics.fmean(post) if post else None,
            "note": "index split only: local llama.cpp inference via "
            "ProgramAsWeightsBackend, except that the first post-threshold call downloads "
            "the adapter and loads the model (so it DOES make a network call), and a "
            "fail-open fallback would appear here too -- see served_by per call. "
            "Superseded by phases.steady_state.",
        },
        "teacher_served_after_threshold": teacher_served_after_threshold,
        **phase_stats([c["ms"] for c in calls], args.threshold),
        "teacher_tokens_total": {
            "input": sum(u["input_tokens"] for u in teacher_usage),
            "output": sum(u["output_tokens"] for u in teacher_usage),
            "calls_billed": len(teacher_usage),
        },
        "calls": calls,
    }

    for name in SPLIT_PHASES:
        ph = summary["phases"][name]
        mean = "n/a" if ph["mean_ms"] is None else f"{ph['mean_ms']:8.1f}ms"
        print(f"{name:17s} n={ph['n']:2d} mean: {mean}")
    if summary["speedup_x"]:
        print(f"speedup: {summary['speedup_x']:.2f}x  ({SPEEDUP_DEFINITION})")
    tail = (
        "they never left the machine"
        if teacher_served_after_threshold == 0
        else f"WARNING: {teacher_served_after_threshold} post-threshold call(s) fell back "
             "to the teacher and DID leave the machine"
    )
    print(
        f"teacher tokens actually billed: {summary['teacher_tokens_total']['input']} in / "
        f"{summary['teacher_tokens_total']['output']} out, over {summary['teacher_tokens_total']['calls_billed']} calls "
        f"({tail})"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"jit-speedup-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
