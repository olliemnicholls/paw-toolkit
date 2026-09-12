"""PII scrubber example: paw.load with a Pydantic schema.

Demonstrates:
1. Schema binding: paw.load validates every output against a nested Pydantic model.
2. Fail-open fallback: an output that fails validation is replaced by the fallback's.

The adapter here is a MockPAWBackend subclass (a rule-based stub, no model). No shipped
backend applies the schema's regex grammar during decoding; validation happens after
generation.
"""

import json
import os
import re
import shutil
import time
from typing import List
from pydantic import BaseModel, Field

import paw_kit as paw
from paw_kit.backend.mock import MockPAWBackend


class PIIEntity(BaseModel):
    entity_type: str = Field(description="PII type: EMAIL, PHONE, SSN, CREDIT_CARD")
    value: str = Field(description="Detected sensitive value")


class PIIScrubResult(BaseModel):
    sanitized_text: str = Field(description="Cleaned text with sensitive PII replaced by [REDACTED]")
    entities: List[PIIEntity] = Field(description="List of detected PII entities")
    total_redacted: int = Field(description="Count of redacted items")


# Sample production data stream containing sensitive information
SAMPLE_STREAM = [
    "Contact me at alice.smith@enterprise.org or call 555-019-2834 regarding invoice #992.",
    "My social security number is 000-12-3456 and my backup email is secret_user@gmail.com.",
    "Please charge the recurring fee to card 4111-2222-3333-4444 before expiration.",
    "System log: User bob@corp.io failed login from IP 192.168.1.1 on 2026-09-05.",
    "No sensitive data here! Just asking about the weather in Seattle.",
]

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".paw_demo_pii")


def fallback_scrubber(raw_text: str) -> PIIScrubResult:
    """Fallback processor (remote LLM or rule engine) used if local adapter fails."""
    # Simple regex-based fallback for the demo
    entities: List[PIIEntity] = []
    sanitized = raw_text

    # Email pattern
    for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", raw_text):
        val = m.group(0)
        entities.append(PIIEntity(entity_type="EMAIL", value=val))
        sanitized = sanitized.replace(val, "[REDACTED_EMAIL]")

    # Phone pattern
    for m in re.finditer(r"\b\d{3}-\d{3}-\d{4}\b", raw_text):
        val = m.group(0)
        entities.append(PIIEntity(entity_type="PHONE", value=val))
        sanitized = sanitized.replace(val, "[REDACTED_PHONE]")

    # SSN pattern
    for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", raw_text):
        val = m.group(0)
        entities.append(PIIEntity(entity_type="SSN", value=val))
        sanitized = sanitized.replace(val, "[REDACTED_SSN]")

    # Credit Card pattern
    for m in re.finditer(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", raw_text):
        val = m.group(0)
        entities.append(PIIEntity(entity_type="CREDIT_CARD", value=val))
        sanitized = sanitized.replace(val, "[REDACTED_CARD]")

    return PIIScrubResult(
        sanitized_text=sanitized,
        entities=entities,
        total_redacted=len(entities),
    )


class PIIMockBackend(MockPAWBackend):
    """Simulated local adapter (MockPAWBackend, no model, no grammar enforcement — see fallback_scrubber below)."""

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: str | None = None) -> str:
        # Rule-based stub standing in for a model; grammar_constraint is ignored here.
        res = fallback_scrubber(input_text)
        return res.model_dump_json()


def main():
    print("=" * 75)
    print("PAW-Kit Example: PII scrubber with paw.load")
    print("=" * 75)
    print(f"Schema: {PIIScrubResult.__name__} (validated after generation)")

    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR, exist_ok=True)

    backend = PIIMockBackend()
    adapter_path = os.path.join(CACHE_DIR, "pii_scrubber.paw")

    # 1. Compile adapter specification
    spec = (
        "Extract all PII (EMAIL, PHONE, SSN, CREDIT_CARD) from the input string, "
        "replace PII occurrences with [REDACTED], and return a validated PIIScrubResult."
    )
    print(f"\n[1/3] Compiling local neural adapter to {adapter_path}...")
    backend.compile(
        spec=spec,
        examples=[
            {
                "input": "Call John at 555-123-4567 or email john@example.com",
                "output": json.dumps({
                    "sanitized_text": "Call John at [REDACTED_PHONE] or email [REDACTED_EMAIL]",
                    "entities": [
                        {"entity_type": "PHONE", "value": "555-123-4567"},
                        {"entity_type": "EMAIL", "value": "john@example.com"},
                    ],
                    "total_redacted": 2,
                }),
            }
        ],
        output_path=adapter_path,
    )

    # 2. Bind adapter to Pydantic schema with fail-open fallback
    print("[2/3] Binding adapter with paw.load(..., response_model=PIIScrubResult)...")
    scrub_fn = paw.load(
        adapter_path=adapter_path,
        response_model=PIIScrubResult,
        backend=backend,
        fallback_provider=fallback_scrubber,
    )

    # 3. Process stream through the (mock) local adapter
    print("\n[3/3] Processing five strings through the bound function:")
    print("-" * 75)

    total_time = 0.0
    for i, raw_text in enumerate(SAMPLE_STREAM, start=1):
        t0 = time.perf_counter()
        result: PIIScrubResult = scrub_fn(raw_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_time += elapsed_ms

        print(f"Item #{i} [{elapsed_ms:4.2f}ms] Redacted: {result.total_redacted} entity(ies)")
        print(f"  Raw:       {raw_text}")
        print(f"  Sanitized: {result.sanitized_text}")
        if result.entities:
            print(f"  Extracted: {[{e.entity_type: e.value} for e in result.entities]}")
        print("-" * 75)

    avg_ms = total_time / len(SAMPLE_STREAM)
    print(f"\n[SUCCESS] Stream processed. Average local latency: {avg_ms:.2f}ms.")
    print("Every output validated against PIIScrubResult (mock adapter, no model; timing is not a benchmark).\n")


if __name__ == "__main__":
    main()
