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
from paw_kit.test.matching import values_equivalent, values_equivalent_unquoted
from paw_kit.test.reporting import ScoredRate, scored_denominator
from paw_kit.test.suite import EXECUTION_ERROR_PLACEHOLDER, AssertionRule, TestSuiteConfig

# measurements/README.md, "Finetune compiler on a rule the base model does not know",
# "Tool feedback" point 1: a suite carrying the exact answer in each case's `expected`
# field used to be read only by paw_kit.test.active (to seed the active-learning
# dataset) -- the runner itself never compared output to it, so an adapter that was
# 10% correct on ground truth still scored `Pass rate: 100.0% (300/300)`. Truncation
# length for the "expected: got ... want ..." failure reason below, so an
# arbitrarily long adapter output/expected value can't blow up a printed report.
_EXPECTED_MISMATCH_TRUNCATE_LENGTH = 120


def _truncate_for_reason(text: str, length: int = _EXPECTED_MISMATCH_TRUNCATE_LENGTH) -> str:
    """Clip `text` for embedding in a failure reason string."""
    return text if len(text) <= length else text[:length] + "..."


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
    # measurements/README.md, "Tool feedback" point 1: whether this case's output
    # matches its suite-carried `expected` field, using the same JSON-or-whitespace
    # normalised equality `paw-test compare` uses for its equivalence count
    # (`values_equivalent`, `paw_kit.test.matching`) -- `None` when the case has no
    # `expected` (a fuzz case, or a standard_case that simply doesn't set one), never
    # coerced to False, so a caller can tell "no ground truth to check" apart from
    # "checked, and it was wrong".
    expected_match: Optional[bool] = None
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
    # H-1: how many cases the backend raised on. `TestCaseResult.execution_error` was
    # recorded per case but was not an input to `case_passed`, `failed_cases`,
    # `is_success`, `pass_rate` or the exit code, and the report had no errored counter
    # at all (unlike `CompareReport.errored_count`) -- a backend raising on every case
    # printed `Pass rate: 100.0% (10/10)` and exited 0. The only thing protecting the
    # committed runs was accidental: every published suite carries
    # `not_contains: ERROR`, and the placeholder contains "ERROR".
    #
    # An errored case is also counted in `failed_cases`; this field reports the *reason*
    # from a different angle, it does not double the failure.
    errored_cases: int = 0
    # H-2: how many cases returned exactly `abstain_value` (and did *not* error -- see
    # the precedence rule in `TestRunner.run`). An abstained case still passes the
    # suite's assertions, by design (`TestSuiteConfig.abstain_value`), so it stays in
    # `passed_cases`; what changed is that it is no longer counted as *correct against
    # the answer key*, which is what let an adapter abstaining on all ten cases report
    # `Correct against expected: 10/10 (100.0%)` at a true correctness of 0/10.
    abstained_cases: int = 0
    # H-3: how many of `total_cases` came from `standard_cases` (the rest are
    # fuzz-generated). The "Correct against expected" line needs this to say how many
    # cases *could* have carried an answer key: a 10-case suite where 5 were authored
    # as `expected:` with nothing after the colon (valid YAML, key looks present)
    # reported `5/5 (100.0%)` while true correctness was 5/10. `total_cases` alone
    # cannot express that, because it includes fuzz cases which never carry one.
    standard_cases_count: int = 0
    # H-12: fuzz cases generated and then cut by the fuzzer's total cap. Non-zero means
    # this run covered less than the suite asked for, which was previously invisible.
    fuzz_cases_dropped: int = 0
    # measurements/README.md, "Tool feedback" point 1: how many cases carried an
    # `expected` field at all (`expected_total`) and how many of those matched
    # (`expected_matched`) -- a suite with no `expected` anywhere leaves both 0, and
    # `expected_match_rate` 0.0, same as before this field existed. A case whose
    # `expected` does not match is *also* counted in `failed_cases` above (see
    # `TestRunner.run`); these fields don't double the failure, they report it from
    # a different angle -- "how good is the answer key match", not "did the case pass".
    expected_total: int = 0
    expected_matched: int = 0
    # H-2's precedence rule: of the `expected_total` cases carrying an answer key, how
    # many got no verdict because they abstained, and how many because the backend
    # raised. These are two distinct buckets on purpose -- "errored beats abstained" is
    # decided on `execution_error is not None`, never on the output string, because the
    # two collided today: `runner.py` substitutes "[EXECUTION_ERROR]" for a raised
    # case, and a suite is free to declare that same string as its `abstain_value`.
    # Executed against pre-fix source: with that abstain_value and a backend raising on
    # every case, `paw-test check` reported `pass_rate: 100.0%` at exit 0.
    #
    # Together with `expected_matched` these partition `expected_total`, which is the
    # one genuine partition in this package -- see `expected_denominator`.
    expected_abstained: int = 0
    expected_errored: int = 0
    # measurements/README.md, "Tool feedback" point 1 (quoted-scalar follow-up):
    # `expected_matched` under `values_equivalent_unquoted` instead of the strict
    # `values_equivalent` -- always >= `expected_matched`, since the unquoted check is
    # a strict relaxation of the exact one. Reporting only; never changes
    # `expected_match`/`case_passed` above, so what counts as a passing case is
    # unaffected by this field's existence.
    expected_matched_unquoted: int = 0
    results: List[TestCaseResult] = Field(default_factory=list)

    @property
    def is_success(self) -> bool:
        """Return True if 100% of cases passed all assertions **and none errored**.

        H-1: `errored_cases == 0` is a separate clause rather than relying on
        `failed_cases`, so this stays correct even if a future change to the run loop
        stops folding an errored case into `failed_cases`. A run where the adapter
        never actually ran is not a success no matter what the assertions said about
        the placeholder output.
        """
        return self.failed_cases == 0 and self.errored_cases == 0 and self.total_cases > 0

    @property
    def pass_rate(self) -> float:
        """Return pass percentage (0.0 to 100.0).

        Denominator unchanged by H-1: an errored case is a *failure*, counted in
        `failed_cases` and still in `total_cases`, not something excluded from the
        rate. `pass_denominator` below carries the note that says so.
        """
        return (self.passed_cases / self.total_cases * 100.0) if self.total_cases > 0 else 0.0

    @property
    def expected_scored(self) -> int:
        """H-2: how many answer-key cases actually got a verdict -- `expected_total`
        minus the ones that abstained and the ones that errored. This, not
        `expected_total`, is `expected_match_rate`'s denominator."""
        return self.expected_total - self.expected_abstained - self.expected_errored

    @property
    def pass_denominator(self) -> ScoredRate:
        """The `pass_rate` denominator, with a note naming what it excluded: nothing.

        `excluded` is deliberately empty even when cases errored or abstained. Those
        cases *are* scored -- an errored one as a failure, an abstained one as a pass --
        so listing them as exclusions would print a sentence that is simply false. They
        are reported on their own lines instead (`errored_cases`/`abstained_cases`).
        """
        return scored_denominator(
            total=self.total_cases,
            scored=self.total_cases,
            excluded={},
            label="pass",
        )

    @property
    def expected_denominator(self) -> ScoredRate:
        """The `expected_match_rate` denominator and its note. A genuine partition of
        `expected_total` into matched-or-not / abstained / errored, so the helper's
        arithmetic check is switched on here."""
        return scored_denominator(
            total=self.expected_total,
            scored=self.expected_scored,
            excluded={"abstained": self.expected_abstained, "errored": self.expected_errored},
            label="correct against expected",
        )

    @property
    def expected_match_rate(self) -> float:
        """Percentage of *scored* answer-key cases whose output matched it (0.0 to
        100.0). 0.0, not a division error, when nothing was scored.

        H-2 changed this denominator from `expected_total` to `expected_scored`. An
        adapter that abstains on every case now reports `0/0`, not `10/10`: it has
        answered nothing, and the honest reading of "how often was it right" over zero
        answers is not 100%.
        """
        return self.expected_denominator.rate(self.expected_matched)

    @property
    def expected_match_rate_unquoted(self) -> float:
        """`expected_match_rate` under `values_equivalent_unquoted` -- same denominator
        (`expected_scored`), same 0.0-on-nothing-scored behaviour."""
        return self.expected_denominator.rate(self.expected_matched_unquoted)

    @property
    def expected_keyless_standard_cases(self) -> int:
        """H-3: standard cases that carry **no** answer key, and so are silently absent
        from the only correctness number the CLI prints. At 300 cases one mis-indented
        `expected:` quietly removes cases from that number, in the direction that
        flatters the adapter."""
        return max(0, self.standard_cases_count - self.expected_total)

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
        # H-14: one semantics, pinned deliberately. `exact_match` used bare `==` while
        # a case's own `expected` field used `values_equivalent`, so `{"a":1}` vs
        # `{"a": 1}` failed one and passed the other -- two comparisons named as though
        # they mean the same thing, disagreeing. `values_equivalent` is the one that
        # "decides every published number", so `exact_match` routes through it.
        #
        # This lands *after* H-4 and only after it: pre-H-4 `values_equivalent("true",
        # "1")` was True, so routing here first would have let an `exact_match: "1"`
        # rule accept an output of `true` -- and `active.py`'s teacher-label gate runs
        # every candidate label through `evaluate_assertion`, so that would have been a
        # poisoning hole, not just a reporting one.
        #
        # `values_equivalent` (not `values_equivalent_unquoted`) on purpose: unquoting
        # is a *reporting* relaxation, and an assertion is not reporting.
        passed = values_equivalent(output, expected)
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
        # Parallel to inputs_to_test: the case's `expected` field, or None for a
        # standard_case that doesn't set one and for every fuzz-generated case (fuzz
        # cases have no ground truth to check against).
        expected_values: List[Optional[str]] = []

        # 1. Standard cases
        for case in config.standard_cases:
            inputs_to_test.append(case.input)
            expected_values.append(case.expected)

        # 2. Adversarial fuzzer cases
        seed_inputs = [c.input for c in config.standard_cases]
        # H-12: `generate_detailed`, so the cases the 500-case cap cut are reported
        # rather than silently absent -- a suite with more than 500 custom probes used
        # to have whole enabled mutation categories run zero cases while still
        # declaring them on.
        fuzz_result = AdversarialFuzzer.generate_detailed(config.fuzzing, base_inputs=seed_inputs)
        fuzzed_cases = fuzz_result.cases
        inputs_to_test.extend(fuzzed_cases)
        expected_values.extend([None] * len(fuzzed_cases))

        results: List[TestCaseResult] = []
        passed_count = 0
        failed_count = 0
        errored_count = 0
        abstained_count = 0
        expected_total = 0
        expected_matched = 0
        expected_abstained = 0
        expected_errored = 0
        expected_matched_unquoted = 0

        for inp, expected in zip(inputs_to_test, expected_values):
            t0 = time.perf_counter()
            execution_error: Optional[str] = None
            try:
                out = self.backend.infer(config.adapter_path, inp)
            except Exception as exc:
                # PAW-TEST-08: a generic placeholder in `output`, not the raw
                # exception text -- see TestCaseResult.execution_error's docstring.
                out = EXECUTION_ERROR_PLACEHOLDER
                execution_error = str(exc)
            latency = (time.perf_counter() - t0) * 1000

            # ---- H-1/H-2 precedence rule -------------------------------------------
            # ERRORED BEATS ABSTAINED, and it is decided on `execution_error is not
            # None` -- never on the output string. The two findings collide on exactly
            # one input: `out` for a raised case is the literal "[EXECUTION_ERROR]"
            # placeholder, and nothing stopped a suite declaring that same string as
            # its `abstain_value`. Executed against pre-fix source with that
            # abstain_value and an all-raising backend: `Pass rate: 100.0% (10/10)`,
            # exit 0. Deciding on the string rather than the flag would swap that
            # defect for an under-count (an all-erroring run reporting "no verdict" and
            # dropping out of the denominator entirely) rather than fixing it.
            # `load_suite` also refuses that abstain_value outright (defence in depth).
            errored = execution_error is not None
            abstained = (
                not errored
                and config.abstain_value is not None
                and out == config.abstain_value
            )

            failed_rules: List[str] = []
            failed_rule_names: List[str] = []
            if errored:
                # First in the list, so the reason a case failed leads with "the
                # backend never ran" rather than with whatever the placeholder happened
                # to do to the user's assertions. The raw exception text is deliberately
                # NOT inlined here -- PAW-TEST-08 keeps it in `execution_error`, which
                # the CLI prints separately, so a failure reason string can't leak a
                # path or a stack fragment wherever it gets displayed.
                failed_rules.append(
                    "execution_error: the backend raised on this case, so it was never "
                    "scored (see execution_error for the message)"
                )
                failed_rule_names.append("execution_error")

            for rule in config.assertions:
                passed, reason = evaluate_assertion(out, rule, abstain_value=config.abstain_value)
                if not passed:
                    failed_rules.append(f"{rule.rule}: {reason}")
                    failed_rule_names.append(rule.rule)

            # measurements/README.md, "Tool feedback" point 1: a case carrying an
            # `expected` field is graded against it too, not just the suite's
            # assertions -- a suite-wide `exact_match` rule can't express "every case
            # has its own answer", which is exactly what let a 10%-correct adapter
            # score 100% before this fix.
            #
            # H-2: an abstention is no longer a *match*. `expected_match = True` on an
            # abstain gave an always-abstaining adapter `Correct against expected:
            # 10/10 (100.0%)` at a true correctness of 0/10, and the forward risk was
            # concrete -- docs/results.md recommends uncommenting `abstain_value:
            # "UNPARSEABLE"` in the shipped date-normalizer suite, after which that
            # adapter reports 82/82. An abstention is "no verdict" (`None`), counted in
            # its own bucket and removed from the rate's denominator. It still *passes*
            # the suite's assertions -- that escape hatch is deliberate and unchanged --
            # it just no longer counts as being right about the answer key.
            expected_match: Optional[bool] = None
            if expected is not None:
                expected_total += 1
                if errored:
                    expected_errored += 1
                elif abstained:
                    expected_abstained += 1
                else:
                    expected_match = values_equivalent(out, expected)
                    if expected_match:
                        expected_matched += 1
                    else:
                        reason = (
                            f"expected: got {_truncate_for_reason(out)} "
                            f"want {_truncate_for_reason(expected)}"
                        )
                        failed_rules.append(reason)
                        failed_rule_names.append("expected")
                    # Reporting only (see TestRunReport.expected_matched_unquoted's
                    # docstring) -- does not affect expected_match/case_passed above.
                    # `expected_match` implies this (values_equivalent_unquoted is a
                    # strict relaxation of values_equivalent), so `or` short-circuits.
                    if expected_match or values_equivalent_unquoted(out, expected):
                        expected_matched_unquoted += 1

            if errored:
                errored_count += 1
            if abstained:
                abstained_count += 1

            # H-1: `not errored` is stated explicitly rather than left to the synthetic
            # failed_rule above, so an errored case cannot pass even if some future
            # change stops adding that rule. This is the clause whose absence let a
            # backend that raised ten times out of ten print ten `[PASS]` lines.
            case_passed = (not errored) and len(failed_rules) == 0
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
                    expected_match=expected_match,
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
            errored_cases=errored_count,
            abstained_cases=abstained_count,
            standard_cases_count=len(config.standard_cases),
            fuzz_cases_dropped=fuzz_result.dropped_count,
            expected_total=expected_total,
            expected_matched=expected_matched,
            expected_abstained=expected_abstained,
            expected_errored=expected_errored,
            expected_matched_unquoted=expected_matched_unquoted,
            results=results,
        )
