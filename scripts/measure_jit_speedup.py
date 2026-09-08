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

from pydantic import BaseModel

from paw_kit import ProgramAsWeightsBackend, compile_on_hit

try:
    import anthropic
except ImportError:
    print("pip install anthropic", file=sys.stderr)
    sys.exit(2)

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
    )(teacher_fn)

    calls: list[dict] = []
    for i in range(args.total_calls):
        ticket = TICKETS[i % len(TICKETS)]
        t0 = time.perf_counter()
        out = decorated(ticket)
        dt_ms = (time.perf_counter() - t0) * 1000
        served_by = "teacher_or_compile" if i < args.threshold else "adapter_or_fallback"
        calls.append({"call": i + 1, "ms": dt_ms, "phase": served_by, "output": out.model_dump()})
        print(f"call {i + 1:2d} [{served_by:18s}] {dt_ms:8.1f}ms -> {out.model_dump()}")

    pre = [c["ms"] for c in calls[: args.threshold]]
    post = [c["ms"] for c in calls[args.threshold :]]

    summary = {
        "label": args.label,
        "teacher_model": TEACHER_MODEL,
        "threshold": args.threshold,
        "total_calls": args.total_calls,
        "pre_threshold": {
            "n": len(pre),
            "mean_ms": statistics.fmean(pre) if pre else None,
            "note": "includes the real Claude API call latency each time; the threshold-th "
            "call additionally includes the full synchronous compile step",
        },
        "post_threshold": {
            "n": len(post),
            "mean_ms": statistics.fmean(post) if post else None,
            "note": "local llama.cpp inference via ProgramAsWeightsBackend, no network call",
        },
        "speedup_x": (statistics.fmean(pre) / statistics.fmean(post)) if pre and post else None,
        "teacher_tokens_total": {
            "input": sum(u["input_tokens"] for u in teacher_usage),
            "output": sum(u["output_tokens"] for u in teacher_usage),
            "calls_billed": len(teacher_usage),
        },
        "calls": calls,
    }

    print(f"\npre-threshold  ({len(pre)} calls) mean: {summary['pre_threshold']['mean_ms']:.1f}ms")
    print(f"post-threshold ({len(post)} calls) mean: {summary['post_threshold']['mean_ms']:.1f}ms")
    if summary["speedup_x"]:
        print(f"speedup: {summary['speedup_x']:.1f}x")
    print(
        f"teacher tokens actually billed: {summary['teacher_tokens_total']['input']} in / "
        f"{summary['teacher_tokens_total']['output']} out, over {summary['teacher_tokens_total']['calls_billed']} calls "
        f"(post-threshold calls used 0 tokens -- they never left the machine)"
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
