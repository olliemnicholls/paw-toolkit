"""Test execution engine and assertion evaluator for PAW test suites."""

from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.schema.loader import get_default_backend
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.suite import AssertionRule, TestSuiteConfig


class TestCaseResult(BaseModel):
    """Result of evaluating assertions on a single input case."""

    input: str
    output: str
    passed: bool
    failed_rules: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class TestRunReport(BaseModel):
    """Structured report produced by running a test suite."""

    __test__ = False

    task_name: str
    adapter_path: str
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    results: List[TestCaseResult] = Field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Return True if 100% of cases passed all assertions."""
        return self.failed_cases == 0 and self.total_cases > 0

    @property
    def pass_rate(self) -> float:
        """Return pass percentage (0.0 to 100.0)."""
        return (self.passed_cases / self.total_cases * 100.0) if self.total_cases > 0 else 0.0

    def get_failing_inputs(self) -> List[Tuple[str, str, List[str]]]:
        """Return list of (input, output, failure_reasons) for all failing cases."""
        return [
            (r.input, r.output, r.failed_rules)
            for r in self.results
            if not r.passed
        ]


def evaluate_assertion(output: str, rule: AssertionRule) -> Tuple[bool, str]:
    """Evaluate an assertion rule against an adapter output.

    Returns:
        (passed, error_message)
    """
    name = rule.rule

    if name == "regex_match":
        if not rule.pattern:
            return False, "regex_match requires a 'pattern' field"
        matched = bool(re.search(rule.pattern, output))
        return matched, f"Output '{output}' does not match pattern '{rule.pattern}'"

    if name == "max_length":
        max_len = int(rule.value)
        passed = len(output) <= max_len
        return passed, f"Output length {len(output)} exceeds max_length {max_len}"

    if name == "min_length":
        min_len = int(rule.value)
        passed = len(output) >= min_len
        return passed, f"Output length {len(output)} is below min_length {min_len}"

    if name == "exact_match":
        expected = str(rule.value)
        passed = output == expected
        return passed, f"Output '{output}' != expected '{expected}'"

    if name == "not_contains":
        forbidden = str(rule.value)
        passed = forbidden not in output
        return passed, f"Output contains forbidden substring '{forbidden}'"

    return False, f"Unknown assertion rule: {name}"


class TestRunner:
    """Executes test suites and reports compliance metrics."""

    __test__ = False

    def __init__(self, backend: Optional[AbstractPAWBackend] = None) -> None:
        self.backend = backend or get_default_backend()

    def run(self, config: TestSuiteConfig) -> TestRunReport:
        """Execute standard cases and fuzzer inputs against the configured adapter."""
        inputs_to_test: List[str] = []

        # 1. Standard cases
        for case in config.standard_cases:
            inputs_to_test.append(case.input)

        # 2. Adversarial fuzzer cases
        seed_inputs = [c.input for c in config.standard_cases]
        fuzzed_cases = AdversarialFuzzer.generate(config.fuzzing, base_inputs=seed_inputs)
        inputs_to_test.extend(fuzzed_cases)

        results: List[TestCaseResult] = []
        passed_count = 0
        failed_count = 0

        for inp in inputs_to_test:
            t0 = time.perf_counter()
            try:
                out = self.backend.infer(config.adapter_path, inp)
            except Exception as exc:
                out = f"[EXCEPTION: {exc}]"
            latency = (time.perf_counter() - t0) * 1000

            # Evaluate assertions
            failed_rules: List[str] = []
            for rule in config.assertions:
                passed, reason = evaluate_assertion(out, rule)
                if not passed:
                    failed_rules.append(f"{rule.rule}: {reason}")

            case_passed = len(failed_rules) == 0
            if case_passed:
                passed_count += 1
            else:
                failed_count += 1

            results.append(
                TestCaseResult(
                    input=inp,
                    output=out,
                    passed=case_passed,
                    failed_rules=failed_rules,
                    latency_ms=latency,
                )
            )

        return TestRunReport(
            task_name=config.task_name,
            adapter_path=config.adapter_path,
            total_cases=len(results),
            passed_cases=passed_count,
            failed_cases=failed_count,
            results=results,
        )
