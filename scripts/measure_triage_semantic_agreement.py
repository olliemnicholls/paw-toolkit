"""Semantic-agreement measurement for the ticket-triage JIT test (the review's top-priority
follow-up): are the compiled adapter's classifications actually right, or merely fast and
schema-shaped?

# The A-vs-B diff/judge pattern here is now shipped: see paw_kit.test.compare/judge (`paw-test compare`/`judge`).

`measurements/jit-speedup-3080-*.json` already showed calls 7-20 returning
medium/technical/3 nine times out of fourteen -- worth checking isn't near-degenerate output
before quoting the JIT speedup as a like-for-like win. This script checks it directly: compile
a fresh adapter for the exact same spec, folding in 5 traced examples drawn from a folding
pool that is **disjoint from the 20 evaluation tickets**, then for all 20 tickets compare the
adapter's classification against a FRESH, independent teacher call on the same ticket (not the
original traced response -- a live second opinion). Exact agreement on priority/department and
urgency_score within 1 is the pass criterion; every disagreement is printed in full so a human
(or a follow-up judge call) can decide which side, if either, is actually right.

B-1 (bug hunt 2026-09-11): the 2026-09-09 run folded `TICKETS[:5]` and then scored all 20,
so five of the twenty tickets carried their own answer inside the adapter's prompt. Those
five agreed 5/5 = 100.0%; the fifteen held out agreed 7/15 = 46.7%; the published figure was
the mixture, 12/20 = 60.0%. Recomputed from the committed
`measurements/triage-semantic-agreement-3080-20260909-002033.json`, whose `cases[]` is
untouched. A fresh teacher call answers teacher label drift -- which is what the old note
claimed -- but not the adapter having both the question and the answer in its context.

Two things follow, and both are now enforced here rather than remembered:

  * the folding pool is a separate constant (`FOLDING_TICKETS`), and `main()` refuses to
    run if it intersects `TICKETS`;
  * the summary reports the folded slice, the held-out slice and a held-out denominator
    separately, so a mixture can never again be quoted as a single rate.

The adapter also writes to a **new** path. The old
`measurements/triage_semantic_agreement-paw-4b-qwen3-0.6b.paw` is pinned by
`scripts/measure_shadow_mode.py` as `ADAPTER` and is embedded in
`measurements/shadow-mode-3080-20260910-124735.json` as `adapter_manifest`, so overwriting it
would destroy that measurement's reproducibility.

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
    # Deferred, not exit(2) here: this runs at import time, before this module's pure
    # functions (assert_folding_pool_disjoint, score_rows, build_summary, ...) are even
    # defined. tests/test_measurement_scripts.py imports this module by path specifically
    # to exercise those offline, with no anthropic installed and no network -- exiting
    # here made that impossible, so every test in that file failed importing the module
    # rather than testing anything (found 2026-09-11 reproducing a red CI run outside any
    # dev .venv that happens to have anthropic already installed).
    anthropic = None  # type: ignore[assignment]

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


# Five folding-only tickets. They are deliberately *not* members of `TICKETS`: the whole
# point of B-1's fix is that nothing the adapter was shown is also scored. They cover the
# same four departments and the same priority span as the evaluation set, so the folded
# examples still demonstrate the output format and the judgement range.
FOLDING_TICKETS = [
    "Our invoice shows a duplicate line item for last month's overage charges.",
    "All background jobs have been stuck in the queue for the past 90 minutes.",
    "We would like a quote for adding a second workspace to our current plan.",
    "Where in the admin console do I change the default timezone for reports?",
    "Production checkout is down and customers cannot complete any purchase.",
]

#: The adapter path. Deliberately different from the 2026-09-09 run's
#: `triage_semantic_agreement-paw-4b-qwen3-0.6b.paw`, which `measure_shadow_mode.py` pins
#: as `ADAPTER` and which is embedded in the committed shadow-mode artifact as
#: `adapter_manifest`. Overwriting that path would make the shadow-mode measurement
#: irreproducible. (B-1, Phase 0 edit 11.)
ADAPTER_FILENAME = "triage_semantic_agreement_heldout-paw-4b-qwen3-0.6b.paw"


def assert_folding_pool_disjoint(folding: list, evaluation: list) -> None:
    """Refuse to run if anything folded into the spec is also scored.

    This is B-1 in one line. Called from `main()` before any money is spent, and unit
    tested against the module's own constants.
    """
    overlap = sorted(set(folding) & set(evaluation))
    if overlap:
        raise RuntimeError(
            "folding pool overlaps the evaluation set, which is exactly the leak B-1 "
            f"recorded: {overlap}. Fold from FOLDING_TICKETS only."
        )


def score_rows(rows: list, folded_inputs: set) -> dict:
    """Split scored rows into folded and held-out slices and report both. Pure.

    `rows` are this script's own `cases[]` entries (or a recorded run's -- the shape has
    not changed), `folded_inputs` the ticket bodies that were folded into the spec text.

    Verified against `measurements/triage-semantic-agreement-3080-20260909-002033.json`
    with `folded_inputs = set(TICKETS[:5])`: folded 5/5 = 100.0%, held out 7/15 = 46.7%,
    all 20 = 12/20 = 60.0%.
    """
    def block(subset: list) -> dict:
        n = len(subset)
        full = sum(1 for r in subset if r["full_agreement"])
        urg = sum(1 for r in subset if r["agree_urgency_within_1"])
        return {
            "n": n,
            "full_agreement": full,
            "full_agreement_rate": (full / n * 100.0) if n else None,
            "urgency_within_1": urg,
            "urgency_within_1_rate": (urg / n * 100.0) if n else None,
        }

    folded = [r for r in rows if r["ticket"] in folded_inputs]
    heldout = [r for r in rows if r["ticket"] not in folded_inputs]
    return {
        "all_scored": block(rows),
        "folded_into_spec": block(folded),
        "heldout": block(heldout),
    }


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


def build_summary(label: str, adapter_path: str, manifest: dict, rows: list,
                  folded_inputs: set) -> dict:
    """Assemble the artifact. Every field is derived from its arguments, except
    `n_folding_pool` (reports the constant pool size, not this run's fold) -- see below.

    Extracted from `main()` so the summary-completeness properties B-1 asks for -- a
    held-out denominator, a held-out rate, and a leak flag counting scored rows that were
    folded into the spec -- are unit testable without an API key.
    """
    slices = score_rows(rows, folded_inputs)
    leaked_rows = [r["ticket"] for r in rows if r["folded_into_spec"]]
    all_scored = slices["all_scored"]
    return {
        "label": label,
        "teacher_model": TEACHER_MODEL,
        "adapter_path": adapter_path,
        "program_id": manifest.get("program_id"),
        # A-2: the manifest key that records the *requested* visibility was renamed
        # `public` -> `public_requested`, and `public_confirmed` (three-state: True /
        # False / None-with-a-reason) now carries what the server actually reported.
        # Read the new name with a legacy fallback so a summary built from a manifest
        # written before the rename still reports the value instead of silently None.
        "public_requested": manifest.get("public_requested", manifest.get("public")),
        "public_confirmed": manifest.get("public_confirmed"),
        "public_confirmed_reason": manifest.get("public_confirmed_reason"),
        # A-6: `examples_folded_into_spec` is now the count of examples that actually
        # reached the spec text, not the number offered to compile() (a malformed example
        # used to inflate it, and this artifact is where that inflated number was
        # published). `folded_example_ids` is mirrored alongside it so the artifact names
        # *which* examples those were, not just how many -- the ids are SHA-256 digests of
        # input+output, so this publishes no traced text.
        "examples_folded_into_spec": manifest.get("examples_folded_into_spec"),
        "folded_example_ids": manifest.get("folded_example_ids"),
        "n": all_scored["n"],
        "n_folding_pool": len(FOLDING_TICKETS),
        "full_agreement_rate": all_scored["full_agreement_rate"],
        "urgency_within_1_rate": all_scored["urgency_within_1_rate"],
        # The held-out denominator, reported separately and unconditionally. This is the
        # number to publish; `full_agreement_rate` above is over every scored row and is
        # only equal to it while no scored row was folded.
        "n_heldout": slices["heldout"]["n"],
        "full_agreement_rate_heldout": slices["heldout"]["full_agreement_rate"],
        "urgency_within_1_rate_heldout": slices["heldout"]["urgency_within_1_rate"],
        "slices": slices,
        "leak_flags": {
            # Checked against `folded_inputs`/`rows` (this call's arguments), not the module
            # constants `FOLDING_TICKETS`/`TICKETS` -- a flag built from the constants would
            # report on what the script *would* fold by default, not on what this run
            # actually folded, and would stay True even if a future caller passed a
            # leaking `folded_inputs` by mistake. `scored_rows_folded_into_spec` below is
            # the direct evidence either way; this flag is the same check from the other
            # direction.
            "folding_pool_disjoint_from_eval": not (
                folded_inputs & {r["ticket"] for r in rows}
            ),
            "scored_rows_folded_into_spec": len(leaked_rows),
            "leaked_tickets": leaked_rows,
            "note": (
                "scored_rows_folded_into_spec must be 0. The 2026-09-09 run folded "
                "TICKETS[:5] and scored all 20, which inflated the headline from 46.7% "
                "(7/15 held out) to 60.0% (12/20). See report B-1."
            ),
        },
        "note": (
            "Compares the compiled adapter's output against a FRESH independent teacher call on the "
            "same ticket, not the originally-traced response -- this is a true second opinion, not a "
            "training-label check. Disagreement does not necessarily mean the adapter is wrong (the "
            "teacher itself is not perfectly consistent call to call); it means the two disagree and "
            "a human should look at the specific case. A fresh teacher call does NOT address "
            "train/eval overlap: that is what the disjoint folding pool and the held-out "
            "denominator are for."
        ),
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--out-dir", default="measurements")
    args = ap.parse_args()

    if anthropic is None:
        print("pip install anthropic", file=sys.stderr)
        return 2

    # B-1: refuse to spend anything if the folding pool and the evaluation set intersect.
    assert_folding_pool_disjoint(FOLDING_TICKETS, TICKETS)

    client = anthropic.Anthropic()
    backend = ProgramAsWeightsBackend(
        compiler="paw-4b-qwen3-0.6b",
        max_spec_examples=16,
        # Stated, not inherited. `ProgramAsWeightsBackend` defaults `public` to False, but
        # this script relied on that default silently while the artifact recorded nothing
        # about the visibility it got -- and a public compile publishes the folded example
        # bodies verbatim. `public` is mirrored into the summary below from the manifest
        # the compile actually wrote.
        public=False,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = str(out_dir / ADAPTER_FILENAME)

    print(f"[fold-in] tracing {len(FOLDING_TICKETS)} folding-only tickets "
          f"(disjoint from the {len(TICKETS)} scored tickets)")
    examples = []
    for t in FOLDING_TICKETS:
        out = call_teacher(client, t)
        examples.append({"input": t, "output": json.dumps(out.model_dump())})
        print(f"  traced: {t[:50]!r} -> {out.model_dump()}")

    print(f"\n[compile] spec={SPEC!r}")
    t0 = time.perf_counter()
    backend.compile(SPEC, examples, adapter_path)
    print(f"[compile] done in {time.perf_counter() - t0:.1f}s -> {adapter_path}")
    manifest = json.loads(Path(adapter_path).read_text())
    print(f"[compile] program_id={manifest.get('program_id')} "
          f"public_requested={manifest.get('public_requested', manifest.get('public'))} "
          f"public_confirmed={manifest.get('public_confirmed')} "
          f"folded={manifest.get('examples_folded_into_spec')}")

    folded_inputs = {ex["input"] for ex in examples}
    rows = []
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

        row = {
            "ticket": ticket,
            # Always False now that the folding pool is disjoint, and recorded anyway: it
            # is the field that makes the leak visible in the artifact rather than only in
            # the source. (B-1.)
            "folded_into_spec": ticket in folded_inputs,
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

    summary = build_summary(args.label, adapter_path, manifest, rows, folded_inputs)
    print(f"\nfull agreement over all {summary['n']} scored tickets: {summary['full_agreement_rate']:.1f}%")
    print(f"full agreement held out ({summary['n_heldout']} tickets): "
          f"{summary['full_agreement_rate_heldout']:.1f}%")
    print(f"urgency_score within 1 alone: {summary['urgency_within_1_rate']:.1f}%")
    print(f"scored rows that were folded into the spec: "
          f"{summary['leak_flags']['scored_rows_folded_into_spec']} (must be 0)")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"triage-semantic-agreement-{args.label}-{stamp}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
