"""Semantic-agreement measurement for the ticket-triage JIT test (the review's top-priority
follow-up): are the compiled adapter's classifications actually right, or merely fast and
schema-shaped?

# The A-vs-B diff/judge pattern here is now shipped: see paw_kit.test.compare/judge (`paw-test compare`/`judge`).

`measurements/jit-speedup-3080-*.json` already showed calls 7-20 returning
medium/technical/3 nine times out of fourteen -- worth checking isn't near-degenerate output
before quoting the JIT speedup as a like-for-like win. This script checks it directly: compile
a fresh adapter for the exact same spec (folding in the same 5 traced examples
`measure_jit_speedup.py` would have used), then for all 20 tickets, compare the adapter's
classification against a FRESH, independent teacher call on the same ticket (not the original
traced response -- a live second opinion). Exact agreement on priority/department and
urgency_score within 1 is the pass criterion; every disagreement is printed in full so a human
(or a follow-up judge call) can decide which side, if either, is actually right.

Prerequisites:
    export PAW_API_KEY=paw_sk_...
    export ANTHROPIC_API_KEY=sk-ant-...

Usage:
    uv run python scripts/measure_triage_semantic_agreement.py --label 3080
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import BaseModel

from paw_kit import ProgramAsWeightsBackend

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


import re
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def call_teacher(client: "anthropic.Anthropic", ticket_body: str) -> Triage:
    resp = client.messages.create(
        model=TEACHER_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                f"{SPEC}\n\nRespond with ONLY a JSON object with exactly the keys "
                f'"priority", "department", "urgency_score". No other text.\n\n'
                f"Ticket: {ticket_body}"
            ),
        }],
    )
    text = "".join(b.text for b in resp.content if hasattr(b, "text"))
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"Teacher returned no JSON: {text!r}")
    return Triage(**json.loads(match.group(0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    client = anthropic.Anthropic()
    backend = ProgramAsWeightsBackend(compiler="paw-4b-qwen3-0.6b", max_spec_examples=16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(out_dir / "triage_semantic_agreement-paw-4b-qwen3-0.6b.paw")

    print("[fold-in] getting 5 traced examples from the teacher, same as measure_jit_speedup.py's threshold")
    examples = []
    for t in TICKETS[:5]:
        out = call_teacher(client, t)
        examples.append({"input": t, "output": json.dumps(out.model_dump())})
        print(f"  traced: {t[:50]!r} -> {out.model_dump()}")

    print(f"\n[compile] spec={SPEC!r}")
    t0 = time.perf_counter()
    backend.compile(SPEC, examples, adapter_path)
    print(f"[compile] done in {time.perf_counter() - t0:.1f}s -> {adapter_path}")

    rows = []
    exact_agree = 0
    urgency_close = 0
    for i, ticket in enumerate(TICKETS):
        adapter_raw = backend.infer(adapter_path, ticket)
        try:
            adapter_out = Triage(**json.loads(_JSON_RE.search(adapter_raw).group(0))) if _JSON_RE.search(adapter_raw) else None
        except Exception:
            adapter_out = None
        teacher_out = call_teacher(client, ticket)  # fresh second opinion, not the traced one

        if adapter_out is None:
            agree_priority = agree_department = agree_urgency = False
        else:
            agree_priority = adapter_out.priority == teacher_out.priority
            agree_department = adapter_out.department == teacher_out.department
            agree_urgency = abs(adapter_out.urgency_score - teacher_out.urgency_score) <= 1

        exact = bool(adapter_out) and agree_priority and agree_department and agree_urgency
        exact_agree += int(exact)
        urgency_close += int(agree_urgency)

        row = {
            "ticket": ticket,
            "adapter_raw": adapter_raw,
            "adapter_parsed": adapter_out.model_dump() if adapter_out else None,
            "teacher_fresh": teacher_out.model_dump(),
            "agree_priority": agree_priority,
            "agree_department": agree_department,
            "agree_urgency_within_1": agree_urgency,
            "full_agreement": exact,
        }
        rows.append(row)
        marker = "AGREE" if exact else "DIFFER"
        print(f"  [{marker:6s}] {ticket[:45]!r:47s} adapter={row['adapter_parsed']} teacher={row['teacher_fresh']}")

    summary = {
        "label": args.label,
        "teacher_model": TEACHER_MODEL,
        "n": len(TICKETS),
        "full_agreement_rate": exact_agree / len(TICKETS) * 100.0,
        "urgency_within_1_rate": urgency_close / len(TICKETS) * 100.0,
        "note": (
            "Compares the compiled adapter's output against a FRESH independent teacher call on the "
            "same ticket, not the originally-traced response -- this is a true second opinion, not a "
            "training-label check. Disagreement does not necessarily mean the adapter is wrong (the "
            "teacher itself is not perfectly consistent call to call); it means the two disagree and "
            "a human should look at the specific case."
        ),
        "cases": rows,
    }
    print(f"\nfull agreement (priority + department + urgency within 1): {summary['full_agreement_rate']:.0f}%")
    print(f"urgency_score within 1 alone: {summary['urgency_within_1_rate']:.0f}%")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"triage-semantic-agreement-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
