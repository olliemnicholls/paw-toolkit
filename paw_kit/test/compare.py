"""Adapter-vs-adapter comparison: run every suite case through two adapters and diff them.

Lifted from an ad hoc workflow used to produce the "Finetune compiler" section of
`measurements/README.md`: the headline finding there (132/134 byte-identical outputs
between `paw-4b-qwen3-0.6b` and `paw-ft-bs48`, two adversarial-probe outliers) was decided
by a per-case diff, not by the aggregate structural/semantic percentages -- both of which
sat inside the LLM judge's own measured +-2-point run-to-run noise (see
`paw_kit.test.judge`'s module docstring and `conductor/deferred/index.md`, "Semantic judge
is non-deterministic"). This module is that per-case diff, made reusable: it runs both
adapters through the same backend, reuses `evaluate_assertion` (does not duplicate it) to
determine pass/fail, and reports which cases actually differ -- deliberately not just an
aggregate pass-rate delta, which is exactly what buried the finding the first time.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.runner import evaluate_assertion
from paw_kit.test.suite import TestSuiteConfig

# Mirrors paw_kit.cli's _MAX_MANIFEST_BYTES / _declared_adapter_backend cap for the same
# reason: a manifest is a few hundred bytes to a few KB of JSON, and this is only ever
# read for *display* metadata in a report, not to drive any write.
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024


def read_adapter_manifest(adapter_path: str) -> Dict[str, Any]:
    """Best-effort manifest read for either backend's `.paw` artifact.

    Tries the strict `ProgramAsWeightsBackend.read_manifest` first (the existing manifest
    reader) so a real adapter's `compiler`/`program_id` land in the report unchanged. If
    that raises (not that backend's manifest format -- true of every `MockPAWBackend`
    adapter and any third-party `.paw` file), falls back to a generic bounded JSON read
    so whatever the file *does* declare (e.g. mock's `spec`/`examples_count`) still shows
    up. Returns `{}` only when the file cannot be read as JSON at all.
    """
    try:
        return dict(ProgramAsWeightsBackend.read_manifest(adapter_path))
    except (FileNotFoundError, ValueError):
        pass

    path = Path(adapter_path)
    try:
        if not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class CompareRow(BaseModel):
    """One case run through both adapters."""

    input: str
    output_a: str
    output_b: str
    identical: bool
    pass_a: bool
    pass_b: bool
    failed_rules_a: List[str] = Field(default_factory=list)
    failed_rules_b: List[str] = Field(default_factory=list)
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    execution_error_a: str | None = None
    execution_error_b: str | None = None


class CompareReport(BaseModel):
    """Structured report produced by `compare_adapters`."""

    __test__ = False

    task_name: str
    adapter_a: str
    adapter_b: str
    manifest_a: Dict[str, Any] = Field(default_factory=dict)
    manifest_b: Dict[str, Any] = Field(default_factory=dict)
    total_cases: int = 0
    identical_count: int = 0
    a_pass_count: int = 0
    b_pass_count: int = 0
    only_a_pass_count: int = 0
    only_b_pass_count: int = 0
    rows: List[CompareRow] = Field(default_factory=list)

    @property
    def differing_rows(self) -> List[CompareRow]:
        """Rows where the two adapters disagree -- different output, or different pass
        status. This is deliberately the headline view: a per-case diff is what actually
        decided the finetune-compiler comparison, not the aggregate counts below it."""
        return [r for r in self.rows if not r.identical or r.pass_a != r.pass_b]


def _infer_safely(backend: AbstractPAWBackend, adapter_path: str, inp: str) -> tuple[str, float, str | None]:
    """Run one inference, mirroring `TestRunner.run`'s exception handling (PAW-TEST-08):
    a backend failure becomes a placeholder output plus a separate error field, not a
    crash that would abort the whole comparison over one bad case."""
    t0 = time.perf_counter()
    try:
        out = backend.infer(adapter_path, inp)
        error = None
    except Exception as exc:  # noqa: BLE001 -- see TestRunner.run's identical handling
        out = "[EXECUTION_ERROR]"
        error = str(exc)
    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms, error


def compare_adapters(
    adapter_a: str,
    adapter_b: str,
    suite: TestSuiteConfig,
    backend: AbstractPAWBackend,
    *,
    include_fuzz: bool = False,
) -> CompareReport:
    """Run every standard case (and, if asked, fuzz case) in `suite` through both
    adapters via `backend`, and diff the results.

    Read-only: this never calls `backend.compile()`. Assertion pass/fail reuses
    `paw_kit.test.runner.evaluate_assertion` -- the same function `TestRunner`/`paw-test
    check` use -- so a "pass" here means the same thing it means there.
    """
    inputs: List[str] = [c.input for c in suite.standard_cases]
    if include_fuzz:
        seed_inputs = [c.input for c in suite.standard_cases]
        inputs.extend(AdversarialFuzzer.generate(suite.fuzzing, base_inputs=seed_inputs))

    rows: List[CompareRow] = []
    identical_count = a_pass_count = b_pass_count = only_a = only_b = 0

    for inp in inputs:
        out_a, lat_a, err_a = _infer_safely(backend, adapter_a, inp)
        out_b, lat_b, err_b = _infer_safely(backend, adapter_b, inp)

        failed_a = []
        for rule in suite.assertions:
            ok, reason = evaluate_assertion(out_a, rule, abstain_value=suite.abstain_value)
            if not ok:
                failed_a.append(f"{rule.rule}: {reason}")
        failed_b = []
        for rule in suite.assertions:
            ok, reason = evaluate_assertion(out_b, rule, abstain_value=suite.abstain_value)
            if not ok:
                failed_b.append(f"{rule.rule}: {reason}")

        pass_a = not failed_a
        pass_b = not failed_b
        identical = out_a == out_b

        identical_count += int(identical)
        a_pass_count += int(pass_a)
        b_pass_count += int(pass_b)
        if pass_a and not pass_b:
            only_a += 1
        if pass_b and not pass_a:
            only_b += 1

        rows.append(
            CompareRow(
                input=inp,
                output_a=out_a,
                output_b=out_b,
                identical=identical,
                pass_a=pass_a,
                pass_b=pass_b,
                failed_rules_a=failed_a,
                failed_rules_b=failed_b,
                latency_a_ms=lat_a,
                latency_b_ms=lat_b,
                execution_error_a=err_a,
                execution_error_b=err_b,
            )
        )

    return CompareReport(
        task_name=suite.task_name,
        adapter_a=str(adapter_a),
        adapter_b=str(adapter_b),
        manifest_a=read_adapter_manifest(str(adapter_a)),
        manifest_b=read_adapter_manifest(str(adapter_b)),
        total_cases=len(rows),
        identical_count=identical_count,
        a_pass_count=a_pass_count,
        b_pass_count=b_pass_count,
        only_a_pass_count=only_a,
        only_b_pass_count=only_b,
        rows=rows,
    )
