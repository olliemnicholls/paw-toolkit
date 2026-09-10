"""Customer Support Ticket Triage Example using paw-kit @compile_on_hit.

Demonstrates:
1. Zero-friction migration: wrapping an existing LLM function with @compile_on_hit.
2. Background tracing: storing real production calls in local SQLite (.paw/traces.db).
3. Hot-swapping: transparent transition from remote API to local compiled neural function.
4. Fail-open safety: resilient fallback if an error occurs.
"""

import json
import os
import shutil
import time
from typing import Dict
from pydantic import BaseModel, Field

from paw_kit import compile_on_hit
from paw_kit.backend.mock import MockPAWBackend


class TriageResult(BaseModel):
    priority: str = Field(description="Priority level: low, medium, high, critical")
    department: str = Field(description="Assigned department: billing, technical, sales, general")
    urgency_score: int = Field(description="Urgency score from 1 (lowest) to 5 (highest)")


# Sample support ticket dataset
SAMPLE_TICKETS = [
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
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".paw_demo_triage")


def simulated_remote_llm(ticket_body: str) -> TriageResult:
    """Simulates a remote frontier LLM API call (e.g., Claude 3.5 Sonnet or GPT-4o)."""
    # Simulate network latency (200ms)
    time.sleep(0.2)

    lower = ticket_body.lower()
    if "502" in lower or "crashes" in lower or "lag" in lower:
        return TriageResult(priority="critical", department="technical", urgency_score=5)
    elif "charged" in lower or "refund" in lower:
        return TriageResult(priority="high", department="billing", urgency_score=4)
    elif "contract" in lower or "enterprise" in lower or "subscription" in lower:
        return TriageResult(priority="medium", department="sales", urgency_score=3)
    elif "soc-2" in lower or "compliance" in lower:
        return TriageResult(priority="medium", department="general", urgency_score=2)
    elif "password" in lower or "upload" in lower:
        return TriageResult(priority="low", department="technical", urgency_score=2)
    else:
        return TriageResult(priority="low", department="general", urgency_score=1)


class DemoBackend(MockPAWBackend):
    """Simulated local adapter (MockPAWBackend, no model) — dictionary lookup, <1ms, no inference cost."""

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: str | None = None) -> str:
        lower = input_text.lower()
        if "502" in lower or "crashes" in lower or "lag" in lower:
            res = {"priority": "critical", "department": "technical", "urgency_score": 5}
        elif "charged" in lower or "refund" in lower:
            res = {"priority": "high", "department": "billing", "urgency_score": 4}
        elif "contract" in lower or "enterprise" in lower or "subscription" in lower:
            res = {"priority": "medium", "department": "sales", "urgency_score": 3}
        elif "soc-2" in lower or "compliance" in lower:
            res = {"priority": "medium", "department": "general", "urgency_score": 2}
        elif "password" in lower or "upload" in lower:
            res = {"priority": "low", "department": "technical", "urgency_score": 2}
        else:
            res = {"priority": "low", "department": "general", "urgency_score": 1}
        return json.dumps(res)


# Clean previous run state if present
if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

# Deterministic backend for zero-GPU, reproducible demonstration
backend = DemoBackend()

SPECIFICATION = (
    "Classify customer support ticket text into a structured triage report with "
    "priority (low, medium, high, critical), department (billing, technical, sales, general), "
    "and an integer urgency_score between 1 and 5."
)


@compile_on_hit(
    spec=SPECIFICATION,
    threshold=5,  # Trigger background compilation after 5 real production hits
    response_model=TriageResult,
    backend=backend,
    cache_dir=CACHE_DIR,
    # Shadow mode is on by default; this 10-ticket example has nowhere near enough
    # calls to fill an agreement window, so turn it off to keep the printed
    # "routed to the local adapter" claim (and is_compiled()) true at call 6.
    shadow_window=0,
)
def triage_ticket(ticket_body: str) -> TriageResult:
    return simulated_remote_llm(ticket_body)


def main():
    print("=" * 70)
    print("PAW-Kit Example: Support Ticket Triage with @compile_on_hit")
    print("=" * 70)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Compilation threshold: 5 hits\n")

    for i, ticket in enumerate(SAMPLE_TICKETS, start=1):
        t0 = time.perf_counter()
        result = triage_ticket(ticket)
        duration_ms = (time.perf_counter() - t0) * 1000

        is_local = triage_ticket.is_compiled()
        mode_str = "[LOCAL (mock)]" if is_local else "[REMOTE TEACHER]"

        print(f"Call {i:02d} | {mode_str:16} | {duration_ms:6.1f}ms")
        print(f"  Input:    {ticket[:60]}...")
        print(f"  Output:   dept={result.department} | priority={result.priority} | urgency={result.urgency_score}")
        print("-" * 70)

        # Brief pause to allow background compilation thread to finish at hit 5
        if i == 5:
            print("\n>>> Hit threshold (5 calls) reached! Triggering JIT background compilation...")
            time.sleep(0.3)
            print(">>> Local adapter compiled and hot-swapped!\n")

    print("\n[SUCCESS] All 10 tickets processed.")
    print("Calls 1-5 executed via Remote Teacher and logged to SQLite trace DB.")
    print("Calls 6-10 routed to the local adapter -- here a MockPAWBackend keyword lookup, not a model.")
    print("The timings above show the harness's own overhead, not inference; see measurements/ for real numbers.\n")


if __name__ == "__main__":
    main()
