"""Test execution engine and assertion evaluator for PAW test suites."""

import concurrent.futures
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.schema.loader import get_default_backend
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.suite import AssertionRule, TestSuiteConfig

# PAW-TEST-03: bounds mirroring logits_processor._compile_fsm_safe's two-tier shape
# (Track 09) -- a cheap length pre-check as the *primary* defense, since a bare
# wall-clock timeout cannot actually interrupt a running re.search (Python has no way
# to cancel a thread). Here that matters more than in the FSM case: this runs once
# per (test case x assertion), so relying on the timeout alone would let abandoned
# runaway threads accumulate across an entire suite run instead of just one.
_MAX_REGEX_MATCH_PATTERN_LENGTH = 1000
_MAX_REGEX_MATCH_OUTPUT_LENGTH = 10_000
_REGEX_MATCH_TIMEOUT_SECONDS = 2.0


def _regex_search_safe(pattern: str, output: str) -> Tuple[bool, Optional[str]]:
    """Bounded `regex_match` evaluation (PAW-TEST-03).

    Returns `(matched, timeout_reason)`; `timeout_reason` is `None` on an ordinary
    match/no-match, or a description of why the match was refused/couldn't complete.
    `re.error` (a malformed pattern) is deliberately *not* caught here -- see
    `evaluate_assertion`, which catches it at the one call site that needs to convert
    it into a failed assertion (PAW-TEST-04) rather than an uncaught crash.
    """
    if len(pattern) > _MAX_REGEX_MATCH_PATTERN_LENGTH:
        return False, f"regex_match pattern exceeds the maximum of {_MAX_REGEX_MATCH_PATTERN_LENGTH} characters"
    if len(output) > _MAX_REGEX_MATCH_OUTPUT_LENGTH:
        return False, f"output exceeds the maximum of {_MAX_REGEX_MATCH_OUTPUT_LENGTH} characters for regex_match"

    def _search() -> bool:
        return bool(re.search(pattern, output))

    # Secondary backstop, not joined on timeout -- see _compile_fsm_safe's docstring
    # for why `with ThreadPoolExecutor(...)` (which calls `shutdown(wait=True)`
    # unconditionally on exit) would defeat the timeout entirely.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_search)
        try:
            return future.result(timeout=_REGEX_MATCH_TIMEOUT_SECONDS), None
        except concurrent.futures.TimeoutError:
            return (
                False,
                f"regex_match timed out after {_REGEX_MATCH_TIMEOUT_SECONDS}s -- pattern is likely pathological",
            )
    finally:
        executor.shutdown(wait=False)


class TestCaseResult(BaseModel):
    """Result of evaluating assertions on a single input case."""

    input: str
    output: str
    passed: bool
    failed_rules: List[str] = Field(default_factory=list)
    # Deferred-topic fix (conductor/deferred/index.md, "Off-spec label leakage under
    # forced classification, no signal"): `failed_rules` is human-readable free text
    # ("not_contains: Output contains forbidden substring 'neutral'") -- fine to read,
    # but a caller who wants to branch on *which rule* failed without string-parsing
    # had no structured field to check. This is that field: just the rule names, in the
    # same order as `failed_rules`, e.g. `["not_contains"]`.
    failed_rule_names: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    # PAW-TEST-08: the raw exception text from a failed backend.infer() call used to
    # be embedded directly in `output` (e.g. "[EXCEPTION: <str(exc)>]"), which could
    # surface internal detail (paths, stack fragments) wherever a report's `output`
    # field gets displayed or logged. `output` now holds a generic placeholder for
    # that case; the actual detail, if any diagnostic value is needed, lives here
    # instead, in a field callers can choose to surface or not.
    execution_error: Optional[str] = None


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


def evaluate_assertion(
    output: str, rule: AssertionRule, abstain_value: Optional[str] = None
) -> Tuple[bool, str]:
    """Evaluate an assertion rule against an adapter output.

    Args:
        output: The candidate output being checked.
        rule: The assertion rule to evaluate.
        abstain_value: When set and `output` equals it exactly, the assertion passes
            unconditionally, regardless of `rule`. This is the suite-level "I don't
            know" escape hatch (`TestSuiteConfig.abstain_value`) -- see its docstring
            for why: a model shouldn't be forced to hallucinate a shaped-but-wrong
            answer just to satisfy an assertion when the input has no legal answer.

    Returns:
        (passed, error_message)
    """
    if abstain_value is not None and output == abstain_value:
        return True, ""

    name = rule.rule

    if name == "regex_match":
        if not rule.pattern:
            return False, "regex_match requires a 'pattern' field"
        # PAW-TEST-04: a malformed pattern (re.error) fails just this one assertion
        # instead of crashing the entire suite run. AssertionRule doesn't validate
        # `pattern` is a compilable regex at suite-load time (unlike `value`, see
        # AssertionRule._validate_value), so this is reachable via a normal suite.yaml.
        try:
            matched, timeout_reason = _regex_search_safe(rule.pattern, output)
        except re.error as exc:
            return False, f"regex_match pattern {rule.pattern!r} is invalid: {exc}"
        if timeout_reason is not None:
            return False, timeout_reason
        return matched, f"Output '{output}' does not match pattern '{rule.pattern}'"

    if name == "max_length":
        # PAW-TEST-04: AssertionRule._validate_value already enforces this for any
        # suite that came through load_suite(), but a TestSuiteConfig/AssertionRule
        # constructed directly (or via Pydantic's validation-skipping
        # model_construct()) can still reach here with a non-integer `value`.
        try:
            max_len = int(rule.value)
        except (TypeError, ValueError):
            return False, f"max_length assertion value must be an integer, got {rule.value!r}"
        passed = len(output) <= max_len
        return passed, f"Output length {len(output)} exceeds max_length {max_len}"

    if name == "min_length":
        try:
            min_len = int(rule.value)
        except (TypeError, ValueError):
            return False, f"min_length assertion value must be an integer, got {rule.value!r}"
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
            execution_error: Optional[str] = None
            try:
                out = self.backend.infer(config.adapter_path, inp)
            except Exception as exc:
                # PAW-TEST-08: a generic placeholder in `output`, not the raw
                # exception text -- see TestCaseResult.execution_error's docstring.
                out = "[EXECUTION_ERROR]"
                execution_error = str(exc)
            latency = (time.perf_counter() - t0) * 1000

            # Evaluate assertions
            failed_rules: List[str] = []
            failed_rule_names: List[str] = []
            for rule in config.assertions:
                passed, reason = evaluate_assertion(out, rule, abstain_value=config.abstain_value)
                if not passed:
                    failed_rules.append(f"{rule.rule}: {reason}")
                    failed_rule_names.append(rule.rule)

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
                    failed_rule_names=failed_rule_names,
                    latency_ms=latency,
                    execution_error=execution_error,
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
