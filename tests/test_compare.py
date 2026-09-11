"""Tests for paw_kit.test.compare and `paw-test compare`."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.cli import test_app
from paw_kit.test.compare import CompareReport, compare_adapters, read_adapter_manifest
from paw_kit.test.suite import load_suite

runner = CliRunner()

SUITE_YAML = """
task_name: compare_smoke
spec: "Echo the input in upper case"
adapter_path: "{adapter_path}"

standard_cases:
  - input: "hello"
    expected: "HELLO"
  - input: "world"
    expected: "WORLD"

assertions:
  - rule: max_length
    value: 100
  - rule: not_contains
    value: "wat"

fuzzing:
  adversarial_probes:
    - "fuzzy input"
"""


def _write_mock_manifest(path: Path, rules: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "backend": "mock",
                "spec": "x",
                "examples": [],
                "rules": rules,
            }
        ),
        encoding="utf-8",
    )


def _write_suite(tmp_path: Path, adapter_path: Path) -> Path:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(SUITE_YAML.replace("{adapter_path}", str(adapter_path)), encoding="utf-8")
    return suite_path


def test_compare_adapters_identical_outputs_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two adapters with byte-identical outputs everywhere -> identical_count == total,
    no only-A/only-B passes."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    rules = {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"}
    _write_mock_manifest(adapter_a, rules)
    _write_mock_manifest(adapter_b, dict(rules))

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.total_cases == 2
    assert report.identical_count == 2
    assert report.only_a_pass_count == 0
    assert report.only_b_pass_count == 0
    assert all(row.identical for row in report.rows)
    assert report.differing_rows == []


def test_compare_adapters_detects_differing_output_and_pass_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """B returns a forbidden substring on one case: outputs differ, and B fails that
    case's assertion while A passes it."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "wat"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.total_cases == 2
    assert report.identical_count == 1
    assert report.a_pass_count == 2
    assert report.b_pass_count == 1
    assert report.only_a_pass_count == 1
    assert report.only_b_pass_count == 0

    diffs = report.differing_rows
    assert len(diffs) == 1
    assert diffs[0].input == "world"
    assert diffs[0].output_a == "WORLD"
    assert diffs[0].output_b == "wat"
    assert diffs[0].pass_a is True
    assert diffs[0].pass_b is False


def test_compare_adapters_include_fuzz_runs_fuzz_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    without_fuzz = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)
    with_fuzz = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=True)

    assert without_fuzz.total_cases == 2
    assert with_fuzz.total_cases > without_fuzz.total_cases
    assert any(row.input == "fuzzy input" for row in with_fuzz.rows)


def test_compare_report_carries_manifests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)
    assert report.manifest_a.get("backend") == "mock"
    assert report.manifest_b.get("backend") == "mock"


def test_read_adapter_manifest_missing_file_returns_empty(tmp_path: Path) -> None:
    assert read_adapter_manifest(str(tmp_path / "nope.paw")) == {}


def test_read_adapter_manifest_unreadable_returns_empty(tmp_path: Path) -> None:
    binary_path = tmp_path / "weights.bin"
    binary_path.write_bytes(b"\x00\x01\x02\x03not json")
    assert read_adapter_manifest(str(binary_path)) == {}


def test_compare_cli_lists_differences_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI output must show the differing rows before the one-line summary."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "wat"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path)])
    assert result.exit_code == 0
    out = result.output

    diff_idx = out.index("Differences")
    summary_idx = out.index("Summary:")
    world_idx = out.index("world")
    assert diff_idx < world_idx < summary_idx, "differing rows must be printed before the summary line"
    assert "wat" in out


def test_compare_cli_no_differences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path)])
    assert result.exit_code == 0
    assert "No differences" in result.output


def test_compare_cli_json_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "wat"})
    suite_path = _write_suite(tmp_path, adapter_a)
    out_path = Path("report.json")

    result = runner.invoke(
        test_app,
        ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz", "--json", str(out_path)],
    )
    assert result.exit_code == 0
    assert out_path.exists()

    data = json.loads(out_path.read_text(encoding="utf-8"))
    parsed = CompareReport.model_validate(data)
    assert parsed.total_cases == 2
    assert parsed.only_a_pass_count == 1


def test_compare_cli_missing_adapter_is_hard_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), "missing.paw", str(suite_path)])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_compare_cli_unreadable_manifest_is_hard_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    adapter_b.write_bytes(b"\x00\x01not a manifest")
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path)])
    assert result.exit_code == 1
    assert "Could not read a manifest" in result.output


def test_compare_cli_missing_suite_is_hard_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO"})

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), "nope.yaml"])
    assert result.exit_code == 1
    assert "does not exist" in result.output


def test_compare_never_calls_compile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`compare` must be read-only: it must never call backend.compile()."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO"})
    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))

    backend = MockPAWBackend()

    def _boom(*args, **kwargs):
        raise AssertionError("compare_adapters must never call backend.compile()")

    monkeypatch.setattr(backend, "compile", _boom)
    compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=True)


# ------------------------------------------------------------ finding 1: execution errors


def test_compare_adapters_records_execution_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces finding 1: a backend that raises on every case must not be invisible
    -- `errored_count` and each row's `execution_error_a/_b` must carry the failure."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    def _boom(*args, **kwargs):
        raise RuntimeError("backend offline")

    monkeypatch.setattr(backend, "infer", _boom)

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.errored_count == 2
    assert all(row.execution_error_a == "backend offline" for row in report.rows)
    assert all(row.execution_error_b == "backend offline" for row in report.rows)
    # Both adapters raise identically -> "[EXECUTION_ERROR]" for both, same pass status
    # -- differing_rows alone would hide this entirely (the bug in finding 1).
    assert report.differing_rows == []


def test_compare_cli_reports_execution_errors_suppresses_green_line_and_exits_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)

    broken_backend = MockPAWBackend()

    def _boom(*args, **kwargs):
        raise RuntimeError("backend offline")

    monkeypatch.setattr(broken_backend, "infer", _boom)

    import paw_kit.cli as cli_module

    monkeypatch.setattr(cli_module, "_resolve_cli_backend", lambda backend_type: broken_backend)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"])

    assert result.exit_code != 0
    assert "Errors (2/2)" in result.output
    assert "backend offline" in result.output
    assert "No differences" not in result.output
    assert "errored 2/2" in result.output


# --------------------------------------------------------- finding 1 (cont.): manifest gate


def test_read_adapter_manifest_rejects_json_without_backend_key(tmp_path: Path) -> None:
    path = tmp_path / "not_a_manifest.paw"
    path.write_text(json.dumps({"foo": "bar", "examples": []}), encoding="utf-8")
    assert read_adapter_manifest(str(path)) == {}


def test_compare_cli_manifest_lacking_expected_shape_is_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON dict that isn't shaped like a manifest (no `backend` field) must be
    rejected by the same gate that rejects unparseable-as-JSON files, with the error
    text the CLI already claims ('not the expected shape')."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    adapter_b.write_text(json.dumps({"foo": "bar", "examples": []}), encoding="utf-8")
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path)])
    assert result.exit_code == 1
    assert "Could not read a manifest" in result.output
    assert "expected shape" in result.output


# ----------------------------------------------------------------- finding 2: abstain_value

ABSTAIN_SUITE_YAML = """
task_name: abstain_parity
spec: "Parse the date or abstain"
adapter_path: "{adapter_path}"
abstain_value: "IDK"

standard_cases:
  - input: "impossible date"
    expected: "2026-01-01"

assertions:
  - rule: exact_match
    value: "2026-01-01"
"""


def test_compare_and_check_agree_on_abstain_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduces finding 2: compare must thread `suite.abstain_value` through
    `evaluate_assertion` the same way `TestRunner.run` does, or the same abstaining
    output passes `check` and fails `compare`."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    rules = {"impossible date": "IDK"}
    _write_mock_manifest(adapter_a, rules)
    _write_mock_manifest(adapter_b, dict(rules))

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(ABSTAIN_SUITE_YAML.replace("{adapter_path}", str(adapter_a)), encoding="utf-8")
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    compare_report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)
    assert compare_report.a_pass_count == 1
    assert compare_report.b_pass_count == 1

    from paw_kit.test.runner import TestRunner

    check_report = TestRunner(backend=backend).run(suite)
    assert check_report.is_success


# --------------------------------------------------------------------- finding 3: fuzz default


def test_compare_adapters_defaults_to_include_fuzz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    default_report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend)
    explicit_report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=True)

    assert default_report.total_cases == explicit_report.total_cases
    assert any(row.input == "fuzzy input" for row in default_report.rows)


def test_compare_cli_runs_fuzz_cases_by_default_and_no_fuzz_opts_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD", "fuzzy input": "FUZZY INPUT"})
    suite_path = _write_suite(tmp_path, adapter_a)

    with_fuzz_out = Path("with_fuzz.json")
    result = runner.invoke(
        test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--json", str(with_fuzz_out)]
    )
    assert result.exit_code == 0
    with_fuzz = json.loads(with_fuzz_out.read_text(encoding="utf-8"))
    assert with_fuzz["total_cases"] == 3  # 2 standard_cases + 1 adversarial_probes entry

    no_fuzz_out = Path("no_fuzz.json")
    result = runner.invoke(
        test_app,
        ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz", "--json", str(no_fuzz_out)],
    )
    assert result.exit_code == 0
    no_fuzz = json.loads(no_fuzz_out.read_text(encoding="utf-8"))
    assert no_fuzz["total_cases"] == 2


# --------------------------------------------------------------------- finding 10: docs/help


def test_compare_cli_help_mentions_memory_and_cold_load() -> None:
    result = runner.invoke(test_app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "memory" in result.output
    assert "cold load" in result.output


# ------------------------------------------------------------------- finding 11: manifest fields


def test_compare_report_manifest_projected_to_display_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.manifest_a.get("backend") == "mock"
    for forbidden in ("examples", "spec", "rules"):
        assert forbidden not in report.manifest_a
        assert forbidden not in report.manifest_b


# ------------------------------------------------------- finding 2: normalized equivalence


def test_compare_adapters_json_whitespace_only_difference_is_equivalent_not_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two JSON outputs that differ only in `json.dumps` spacing must not be
    byte-identical, but must count as `equivalent` -- the exact shape of finding 2's
    37/60 ticket-triage cases."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": '{"priority":"high","urgency":5}', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": '{"priority": "high", "urgency": 5}', "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.identical is False
    assert hello_row.match_kind == "equivalent"

    world_row = next(r for r in report.rows if r.input == "world")
    assert world_row.identical is True
    assert world_row.match_kind == "byte_identical"

    assert report.identical_count == 1  # only "world"
    assert report.equivalent_count == 2  # "world" (identical) + "hello" (equivalent)
    # differing_rows stays byte-level -- unchanged by finding 2.
    assert [r.input for r in report.differing_rows] == ["hello"]
    # ...but the whitespace-only split moves it out of the "real" differences.
    assert [r.input for r in report.genuinely_differing_rows] == []
    assert [r.input for r in report.equivalent_only_rows] == ["hello"]


def test_compare_adapters_json_semantic_difference_is_not_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two JSON outputs that parse to different values (not just different whitespace)
    must be classified `different`, not `equivalent`, and must stay in
    `genuinely_differing_rows`."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": '{"priority": "high"}', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": '{"priority": "low"}', "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.match_kind == "different"
    assert report.equivalent_count == 1  # only "world"
    assert [r.input for r in report.genuinely_differing_rows] == ["hello"]
    assert [r.input for r in report.equivalent_only_rows] == []


def test_compare_adapters_non_json_whitespace_collapse_is_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-JSON text that differs only in whitespace (not both-parse-as-JSON) still
    counts as equivalent, via Unicode NFC + whitespace-collapse comparison."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": "the   quick brown fox", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "the quick brown fox", "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.identical is False
    assert hello_row.match_kind == "equivalent"


def test_compare_adapters_only_one_side_json_falls_back_to_text_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If only one side parses as JSON, that is not 'both parse' -- comparison must
    fall through to the whitespace-normalized *text*, not compare a parsed value
    against unparsed text."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": '{"a": 1}', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "not json at all", "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.match_kind == "different"


def test_compare_cli_shows_equivalent_count_and_collapses_whitespace_only_heading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI must show a second, clearly-named equivalent-output count in the summary
    line, and list whitespace-only pairs under their own collapsed heading rather than
    the main "Differences" listing."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": '{"a": 1}', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": '{"a":  1}', "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(
        test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"]
    )
    assert result.exit_code == 0
    out = result.output

    assert "Whitespace- or quoting-only differences (1/2)" in out
    assert "1 identical output" in out
    assert "2 equivalent output" in out
    # "hello" is whitespace-only, not a real disagreement -- must not appear under the
    # main "Differences" heading (which should be suppressed entirely here, since there
    # are zero genuine differences).
    assert "Differences (" not in out


def test_compare_cli_genuine_differences_and_whitespace_only_both_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mix of one genuine difference and one whitespace-only difference must show
    both sections, with the genuine one under "Differences" and the whitespace-only one
    only under the collapsed heading."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": '{"a": 1}', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": '{"a":  1}', "world": "wat"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(
        test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"]
    )
    assert result.exit_code == 0
    out = result.output

    assert "Differences (1/2)" in out
    assert "Whitespace- or quoting-only differences (1/2)" in out
    diff_idx = out.index("Differences (1/2)")
    whitespace_idx = out.index("Whitespace- or quoting-only differences")
    world_idx = out.index("world", diff_idx)
    assert diff_idx < world_idx < whitespace_idx


# ------------------------------------------------ quoted-scalar follow-up: equivalent_unquoted


def test_compare_adapters_quoted_scalar_is_equivalent_unquoted_not_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One adapter's output is the other's, JSON-string-quoted -- `values_equivalent`
    calls it different (a quoting defect is real), but it must land in the new
    `"equivalent_unquoted"` match_kind rather than plain `"different"`, and count in
    `equivalent_unquoted_count` but NOT in the strict `equivalent_count`."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": '"RG-M2"', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "RG-M2", "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.match_kind == "equivalent_unquoted"
    assert report.equivalent_count == 1  # only "world" (byte-identical)
    assert report.equivalent_unquoted_count == 2  # "world" + the unquoted "hello"
    assert [r.input for r in report.equivalent_only_rows] == ["hello"]
    assert [r.input for r in report.genuinely_differing_rows] == []


def test_compare_adapters_json_string_versus_json_number_stays_different(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`'"5"'` vs. `5`: both sides parse as JSON, a genuine type mismatch -- must stay
    `"different"`, not fall into `"equivalent_unquoted"`, mirroring
    `values_equivalent_unquoted`'s own carve-out."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "a.paw"
    adapter_b = tmp_path / "b.paw"
    _write_mock_manifest(adapter_a, {"hello": '"5"', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "5", "world": "WORLD"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    hello_row = next(r for r in report.rows if r.input == "hello")
    assert hello_row.match_kind == "different"
    assert report.equivalent_unquoted_count == report.equivalent_count == 1


def test_compare_cli_shows_equivalent_unquoted_segment_only_when_it_exceeds_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The summary line's extra "equivalent output once unwrapped" segment must show
    only when unquoting actually widens the count, and the quoted-only row must land
    under the (renamed) collapsed heading, not the main "Differences" listing."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": '"RG-M2"', "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "RG-M2", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(
        test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"]
    )
    assert result.exit_code == 0
    out = result.output

    assert "Whitespace- or quoting-only differences (1/2)" in out
    assert "Differences (" not in out
    assert "1 equivalent output" in out
    assert "2 equivalent output once unwrapped" in out


def test_compare_cli_omits_equivalent_unquoted_segment_when_nothing_is_quoted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: when unquoting adds nothing, the summary line must not carry the
    extra segment at all."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("a.paw")
    adapter_b = Path("b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "WORLD"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(
        test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"]
    )
    assert result.exit_code == 0
    out = result.output

    assert "equivalent output once unwrapped" not in out


# --------------------------------------------------------------------- finding: adapter labels


def test_compare_adapters_labels_by_file_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """CompareReport.label_a/label_b default to each adapter's file stem, and
    adapter_a/adapter_b (the full paths) stay unchanged for backward compatibility
    (measurements/README.md, Tool feedback point 2)."""
    monkeypatch.chdir(tmp_path)
    adapter_a = tmp_path / "paw-4b-qwen3-0.6b.paw"
    adapter_b = tmp_path / "paw-ft-bs48.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.label_a == "paw-4b-qwen3-0.6b"
    assert report.label_b == "paw-ft-bs48"
    assert report.adapter_a == str(adapter_a)
    assert report.adapter_b == str(adapter_b)


def test_compare_adapters_labels_fall_back_to_full_path_on_stem_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two adapters with the same file stem in different directories must not collapse
    to the same label -- fall back to the full path for both."""
    monkeypatch.chdir(tmp_path)
    dir_a = tmp_path / "arm_a"
    dir_b = tmp_path / "arm_b"
    dir_a.mkdir()
    dir_b.mkdir()
    adapter_a = dir_a / "model.paw"
    adapter_b = dir_b / "model.paw"
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO"})

    suite_path = _write_suite(tmp_path, adapter_a)
    suite = load_suite(str(suite_path))
    backend = MockPAWBackend()

    report = compare_adapters(str(adapter_a), str(adapter_b), suite, backend, include_fuzz=False)

    assert report.label_a == str(adapter_a)
    assert report.label_b == str(adapter_b)
    assert report.label_a != report.label_b


def test_compare_cli_prints_adapter_labels_by_stem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI must show each adapter's file stem in place of the fixed A/B labels,
    in both the per-row diff and the summary line."""
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("model_a.paw")
    adapter_b = Path("model_b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO", "world": "WORLD"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO", "world": "wat"})
    suite_path = _write_suite(tmp_path, adapter_a)

    result = runner.invoke(test_app, ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz"])
    assert result.exit_code == 0
    out = result.output

    assert "model_a ->" in out
    assert "model_b ->" in out
    assert "model_a pass" in out
    assert "model_b pass" in out
    assert "only-model_a-pass" in out
    assert "only-model_b-pass" in out


def test_compare_cli_json_output_carries_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    adapter_a = Path("model_a.paw")
    adapter_b = Path("model_b.paw")
    _write_mock_manifest(adapter_a, {"hello": "HELLO"})
    _write_mock_manifest(adapter_b, {"hello": "HELLO"})
    suite_path = _write_suite(tmp_path, adapter_a)
    out_path = Path("report.json")

    result = runner.invoke(
        test_app,
        ["compare", str(adapter_a), str(adapter_b), str(suite_path), "--no-fuzz", "--json", str(out_path)],
    )
    assert result.exit_code == 0
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["label_a"] == "model_a"
    assert data["label_b"] == "model_b"
    assert data["adapter_a"] == str(adapter_a)
    assert data["adapter_b"] == str(adapter_b)


def test_no_unescaped_console_interpolations_in_compare_and_judge_commands():
    """The repo-wide `_e()` AST rule (tests/test_cli.py) scans the whole of cli.py, so
    the `compare`/`judge` commands added there are already covered by it -- this test
    just asserts that shared enforcement actually reaches the new code, rather than
    only reasoning about it.
    """
    import ast

    source = Path(__file__).parent.parent.joinpath("paw_kit", "cli.py").read_text()
    tree = ast.parse(source)
    command_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("compare_cmd", "judge_cmd"):
            command_names.add(node.name)
    assert command_names == {"compare_cmd", "judge_cmd"}, "expected both new commands to be present in cli.py"


# =====================================================================================
# Report section 6, H-5 and G-4 (bug-hunt-remediation, Track B, Phase B3)
# =====================================================================================

_H5_SUITE = """
task_name: h5_lookup
spec: "Look up the region code."
adapter_path: "a.paw"

standard_cases:
  - input: "case-0"
    expected: "RG-0"
  - input: "case-1"
    expected: "RG-1"
  - input: "case-2"
    expected: "RG-2"

assertions:
  - rule: max_length
    value: 100

active_learning:
  auto_recompile: false
"""


def _write_lookup_adapter(path: Path, rules: dict) -> None:
    MockPAWBackend().compile(
        spec="Look up the region code.",
        examples=[{"input": k, "output": v} for k, v in rules.items()],
        output_path=str(path),
    )


def test_h5_two_adapters_of_known_different_correctness_are_distinguishable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-5: `compare` read `suite.standard_cases` only for `.input`, and `CompareReport`
    had no expected field at all. Adapter A 3/3 correct and adapter B 0/3 reported
    `identical=0 equivalent=0 A pass=3/3 B pass=3/3 only_a_pass=0 only_b_pass=0`,
    exit 0 -- the exact defect fixed in `runner.py` and left unfixed in the sibling
    command. `measurements/README.md:1633` prints this line for the lookup arms, where
    one adapter is ~33% correct and another 97.7%.
    """
    monkeypatch.chdir(tmp_path)
    _write_lookup_adapter(tmp_path / "a.paw", {f"case-{i}": f"RG-{i}" for i in range(3)})
    _write_lookup_adapter(tmp_path / "b.paw", {f"case-{i}": f"WRONG-{i}" for i in range(3)})
    (tmp_path / "suite.yaml").write_text(_H5_SUITE, encoding="utf-8")
    suite = load_suite(str(tmp_path / "suite.yaml"))

    report = compare_adapters("a.paw", "b.paw", suite, MockPAWBackend(), include_fuzz=False)

    # Both adapters pass every assertion -- that is the point: pass rate cannot tell
    # them apart, and before H-5 nothing else could either.
    assert report.a_pass_count == report.b_pass_count == 3
    assert report.only_a_pass_count == report.only_b_pass_count == 0

    assert report.expected_total == 3
    assert report.a_expected_matched == 3
    assert report.b_expected_matched == 0
    assert len(report.expected_disagreeing_rows) == 3
    # And every such row reaches the headline diff.
    for row in report.expected_disagreeing_rows:
        assert row in report.differing_rows
        assert row not in report.equivalent_only_rows


def test_h5_expected_precedence_errored_beats_abstained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_infer_safely` writes the same "[EXECUTION_ERROR]" placeholder `runner.py`
    does, so `compare` has the identical H-1/H-2 collision and needs the identical
    precedence rule: errored is decided on the error field, never on the output."""
    monkeypatch.chdir(tmp_path)
    _write_lookup_adapter(tmp_path / "a.paw", {f"case-{i}": f"RG-{i}" for i in range(3)})
    _write_lookup_adapter(tmp_path / "b.paw", {f"case-{i}": "IDK" for i in range(3)})
    (tmp_path / "suite.yaml").write_text(_H5_SUITE + 'abstain_value: "IDK"\n', encoding="utf-8")
    suite = load_suite(str(tmp_path / "suite.yaml"))

    class _BRaises(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str) -> str:  # type: ignore[override]
            if adapter_path.endswith("b.paw"):
                raise RuntimeError("b is broken")
            return super().infer(adapter_path, input_text)

    report = compare_adapters("a.paw", "b.paw", suite, _BRaises(), include_fuzz=False)

    assert report.a_expected_matched == 3
    assert report.a_expected_errored == 0
    assert report.a_expected_abstained == 0
    # B raised: errored, never abstained, even though its configured output was the
    # abstain value.
    assert report.b_expected_errored == 3
    assert report.b_expected_abstained == 0
    assert report.b_expected_matched == 0
    assert "3 errored" in report.b_expected_denominator.note
    assert report.errored_count == 3


def test_h5_abstaining_adapter_is_not_scored_as_correct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_lookup_adapter(tmp_path / "a.paw", {f"case-{i}": f"RG-{i}" for i in range(3)})
    _write_lookup_adapter(tmp_path / "b.paw", {f"case-{i}": "IDK" for i in range(3)})
    (tmp_path / "suite.yaml").write_text(_H5_SUITE + 'abstain_value: "IDK"\n', encoding="utf-8")
    suite = load_suite(str(tmp_path / "suite.yaml"))

    report = compare_adapters("a.paw", "b.paw", suite, MockPAWBackend(), include_fuzz=False)

    assert report.b_expected_abstained == 3
    assert report.b_expected_matched == 0
    assert report.b_expected_denominator.rate(report.b_expected_matched) == 0.0
    assert "3 abstained" in report.b_expected_denominator.note


def test_g4_quoting_difference_that_flips_pass_reaches_the_collapsed_heading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G-4: `equivalent_only_rows` required `pass_a == pass_b`, so quoted output that
    flipped pass status never reached the "equivalent once unwrapped" summary it was
    added to provide -- and `compare` contradicted itself five lines apart, listing
    three rows as full `Differences` while also summarising them as "3 equivalent
    output once unwrapped"."""
    monkeypatch.chdir(tmp_path)
    # Same values, one side JSON-string-quoted. A max_length rule the quoted side
    # fails makes the pass status flip.
    _write_lookup_adapter(tmp_path / "a.paw", {"case-0": "RG-0"})
    _write_lookup_adapter(tmp_path / "b.paw", {"case-0": '"RG-0"'})
    (tmp_path / "suite.yaml").write_text(
        "task_name: g4\nspec: s\nadapter_path: \"a.paw\"\n"
        'standard_cases:\n  - input: "case-0"\n'
        "assertions:\n  - rule: max_length\n    value: 4\n"
        "active_learning:\n  auto_recompile: false\n",
        encoding="utf-8",
    )
    suite = load_suite(str(tmp_path / "suite.yaml"))

    report = compare_adapters("a.paw", "b.paw", suite, MockPAWBackend(), include_fuzz=False)
    row = report.rows[0]

    assert row.match_kind == "equivalent_unquoted"
    assert row.pass_a != row.pass_b, "the fixture must actually flip pass status"
    assert row in report.differing_rows
    # The finding: this row used to be absent from here while being counted in
    # `equivalent_unquoted_count` -- listed as a real difference and summarised as not
    # one, at the same time.
    assert row in report.equivalent_only_rows
    assert row not in report.genuinely_differing_rows
    assert report.equivalent_unquoted_count == 1


def test_g4_expected_disagreement_is_not_collapsed_as_a_quoting_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The condition that replaced `pass_a == pass_b` is a different claim, and it
    must hold: a row where the two adapters disagree about the *answer key* is not a
    formatting difference at any level."""
    monkeypatch.chdir(tmp_path)
    _write_lookup_adapter(tmp_path / "a.paw", {"case-0": "RG-0"})
    _write_lookup_adapter(tmp_path / "b.paw", {"case-0": '"RG-0"'})
    (tmp_path / "suite.yaml").write_text(
        "task_name: g4b\nspec: s\nadapter_path: \"a.paw\"\n"
        'standard_cases:\n  - input: "case-0"\n    expected: "RG-0"\n'
        "assertions:\n  - rule: max_length\n    value: 100\n"
        "active_learning:\n  auto_recompile: false\n",
        encoding="utf-8",
    )
    suite = load_suite(str(tmp_path / "suite.yaml"))

    report = compare_adapters("a.paw", "b.paw", suite, MockPAWBackend(), include_fuzz=False)
    row = report.rows[0]

    assert row.a_expected_match is True
    assert row.b_expected_match is False  # quoted output is a real defect, strictly
    assert row.pass_a == row.pass_b       # assertions cannot tell them apart
    assert row in report.differing_rows
    assert row not in report.equivalent_only_rows
    assert row in report.genuinely_differing_rows
