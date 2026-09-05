"""Unit and integration tests for paw.test: suite parser, adversarial fuzzer, and active learning loop."""

from pathlib import Path
from typing import Dict, List
import pytest

from paw_kit import (
    ActiveLearningReport,
    AdversarialFuzzer,
    AssertionRule,
    MockPAWBackend,
    TestRunReport,
    TestRunner,
    TestSuiteConfig,
    evaluate_assertion,
    load_suite,
    run_active_learning_loop,
)
from paw_kit.test.suite import ActiveLearningConfig, FuzzingConfig, StandardTestCase


SAMPLE_SUITE_YAML = """
task_name: date_normalizer
spec: "Convert informal English date expressions into strict ISO-8601 (YYYY-MM-DD)."
adapter_path: "./models/date_normalizer.paw"

standard_cases:
  - input: "yesterday"
    expected: "2026-09-04"
  - input: "tomorrow"
    expected: "2026-09-06"

assertions:
  - rule: regex_match
    pattern: '^(\\d{4}-\\d{2}-\\d{2}|INVALID)$'
  - rule: max_length
    value: 10
  - rule: min_length
    value: 7
  - rule: not_contains
    value: "ERROR"

fuzzing:
  inject_unicode: true
  empty_inputs: true
  whitespace_flood: true
  payload_extremes: true
  adversarial_probes:
    - "February 30th"
    - "December 32, 2026"

active_learning:
  auto_recompile: true
  teacher_model: "claude-3-5-sonnet-20241022"
  max_iterations: 3
"""


def test_suite_yaml_parsing(tmp_path: Path) -> None:
    """Verify parsing of suite.yaml from raw string and filesystem path."""
    # From string
    config = load_suite(SAMPLE_SUITE_YAML)
    assert config.task_name == "date_normalizer"
    assert len(config.standard_cases) == 2
    assert len(config.assertions) == 4
    assert config.fuzzing.inject_unicode is True
    assert "February 30th" in config.fuzzing.adversarial_probes
    assert config.active_learning.max_iterations == 3

    # From file
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text(SAMPLE_SUITE_YAML, encoding="utf-8")
    loaded_file_config = load_suite(suite_file)
    assert loaded_file_config.task_name == "date_normalizer"

    # Invalid yaml error handling
    with pytest.raises(ValueError, match="root must be a YAML mapping"):
        load_suite("- just a list")


def test_adversarial_fuzzer_generation() -> None:
    """Verify adversarial fuzzer generates Unicode, whitespace, and payload extremes."""
    config = FuzzingConfig(
        inject_unicode=True,
        empty_inputs=True,
        whitespace_flood=True,
        payload_extremes=True,
        adversarial_probes=["custom_probe_1"],
    )

    fuzzed = AdversarialFuzzer.generate(config, base_inputs=["base"])
    assert "custom_probe_1" in fuzzed
    assert "" in fuzzed  # Empty inputs
    assert any("\u200B" in x for x in fuzzed)  # Unicode zero-width space
    assert any("   " in x for x in fuzzed)  # Whitespace flood
    assert any(len(x) >= 1000 for x in fuzzed)  # Payload extremes


def test_assertion_evaluator() -> None:
    """Verify all built-in assertion rules evaluate correctly."""
    # regex_match
    passed, _ = evaluate_assertion("2026-09-05", AssertionRule(rule="regex_match", pattern=r"^\d{4}-\d{2}-\d{2}$"))
    assert passed is True

    failed, reason = evaluate_assertion("bad-date", AssertionRule(rule="regex_match", pattern=r"^\d{4}-\d{2}-\d{2}$"))
    assert failed is False
    assert "does not match pattern" in reason

    # max_length
    p_max, _ = evaluate_assertion("short", AssertionRule(rule="max_length", value=10))
    assert p_max is True
    f_max, _ = evaluate_assertion("very_long_string", AssertionRule(rule="max_length", value=5))
    assert f_max is False

    # min_length
    p_min, _ = evaluate_assertion("sufficient", AssertionRule(rule="min_length", value=5))
    assert p_min is True
    f_min, _ = evaluate_assertion("tiny", AssertionRule(rule="min_length", value=10))
    assert f_min is False

    # exact_match
    p_exact, _ = evaluate_assertion("exact", AssertionRule(rule="exact_match", value="exact"))
    assert p_exact is True
    f_exact, _ = evaluate_assertion("diff", AssertionRule(rule="exact_match", value="exact"))
    assert f_exact is False

    # not_contains
    p_nc, _ = evaluate_assertion("clean output", AssertionRule(rule="not_contains", value="FAIL"))
    assert p_nc is True
    f_nc, _ = evaluate_assertion("contains FAIL here", AssertionRule(rule="not_contains", value="FAIL"))
    assert f_nc is False

    # Unknown rule
    unk_pass, unk_msg = evaluate_assertion("val", AssertionRule(rule="unknown_rule"))
    assert unk_pass is False
    assert "Unknown assertion rule" in unk_msg


def test_test_runner_execution(tmp_path: Path) -> None:
    """Verify TestRunner evaluates all inputs and produces structured reports."""
    adapter_path = str(tmp_path / "test_run.paw")
    backend = MockPAWBackend()
    backend.compile(
        spec="Normalize date",
        examples=[{"input": "yesterday", "output": "2026-09-04"}],
        output_path=adapter_path,
    )

    config = TestSuiteConfig(
        task_name="runner_test",
        spec="Normalize date",
        adapter_path=adapter_path,
        standard_cases=[StandardTestCase(input="yesterday", expected="2026-09-04")],
        assertions=[AssertionRule(rule="regex_match", pattern=r"^\d{4}-\d{2}-\d{2}$")],
        fuzzing=FuzzingConfig(empty_inputs=False),
    )

    runner = TestRunner(backend=backend)
    report = runner.run(config)

    assert report.task_name == "runner_test"
    assert report.total_cases == 1
    assert report.passed_cases == 1
    assert report.failed_cases == 0
    assert report.is_success is True
    assert report.pass_rate == 100.0


def test_active_learning_self_healing_loop(tmp_path: Path) -> None:
    """Verify Active Learning Loop catches failing edge cases, queries teacher, and auto-repairs."""
    adapter_path = str(tmp_path / "date_repair.paw")
    backend = MockPAWBackend()

    # Initial adapter: only knows "yesterday"
    # For "February 30th", it has no rule, so it returns bad output "2026-02-30" (or fallback mock)
    backend.compile(
        spec="Normalize date",
        examples=[{"input": "yesterday", "output": "2026-09-04"}],
        output_path=adapter_path,
    )
    # Simulate bad inference on invalid date
    backend.register_rule(adapter_path, "February 30th", "INVALID_DATE_FORMAT_LONG")

    config = TestSuiteConfig(
        task_name="auto_repair_test",
        spec="Normalize date",
        adapter_path=adapter_path,
        standard_cases=[StandardTestCase(input="yesterday", expected="2026-09-04")],
        assertions=[
            AssertionRule(rule="regex_match", pattern=r"^(\d{4}-\d{2}-\d{2}|INVALID)$"),
            AssertionRule(rule="max_length", value=10),
        ],
        fuzzing=FuzzingConfig(
            adversarial_probes=["February 30th"],
            empty_inputs=False,
            whitespace_flood=False,
            inject_unicode=False,
        ),
        active_learning=ActiveLearningConfig(),
    )

    teacher_queries: List[str] = []

    def mock_frontier_teacher(query_input: str) -> str:
        teacher_queries.append(query_input)
        if query_input == "February 30th":
            return "INVALID"
        return "2026-01-01"

    # Execute active learning self-healing loop
    al_report = run_active_learning_loop(
        config=config,
        backend=backend,
        teacher_provider=mock_frontier_teacher,
    )

    assert al_report.is_success is True
    assert al_report.recompiled is True
    assert al_report.repaired_edge_cases == 1
    assert "February 30th" in teacher_queries
    assert al_report.iterations_run == 2  # Failed iteration 1, repaired & passed iteration 2


def test_active_learning_iteration_limit(tmp_path: Path) -> None:
    """Verify loop halts at max_iterations if assertions continuously fail."""
    adapter_path = str(tmp_path / "unrepairable.paw")
    backend = MockPAWBackend()
    backend.compile(spec="Spec", examples=[], output_path=adapter_path)

    config = TestSuiteConfig(
        task_name="limit_test",
        spec="Spec",
        adapter_path=adapter_path,
        standard_cases=[StandardTestCase(input="impossible", expected="val")],
        assertions=[AssertionRule(rule="exact_match", value="TARGET")],
        fuzzing=FuzzingConfig(),
    )
    config.active_learning.max_iterations = 2

    # Teacher returns output that still doesn't match TARGET
    def stubborn_teacher(inp: str) -> str:
        return "NOT_TARGET"

    report = run_active_learning_loop(
        config=config,
        backend=backend,
        teacher_provider=stubborn_teacher,
    )

    assert report.is_success is False
    assert report.iterations_run == 2
    assert report.recompiled is True


def test_active_learning_missing_teacher_error(tmp_path: Path) -> None:
    """Verify that auto-repair without teacher callable raises descriptive ValueError."""
    adapter_path = str(tmp_path / "no_teacher.paw")
    backend = MockPAWBackend()
    backend.compile(spec="Spec", examples=[], output_path=adapter_path)

    config = TestSuiteConfig(
        task_name="no_teacher_test",
        spec="Spec",
        adapter_path=adapter_path,
        standard_cases=[StandardTestCase(input="failing_case", expected="expected")],
        assertions=[AssertionRule(rule="exact_match", value="expected")],
    )

    with pytest.raises(ValueError, match="requires a valid teacher_provider"):
        run_active_learning_loop(config=config, backend=backend, teacher_provider=None)
