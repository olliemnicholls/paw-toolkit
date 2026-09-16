"""Unit and integration tests for Typer developer CLI commands."""

import json
import os
from pathlib import Path
import re
from typer.testing import CliRunner
import pytest

from paw_kit.cli import app, inspect, clean, test_app as paw_test_app

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI/Rich color escape codes from captured CLI output.

    Rich's `Console` forces color output (regardless of the CliRunner's non-tty
    stream) when `FORCE_COLOR` is set in the environment -- true of some dev/CI
    setups, not just interactive terminals. Assertions on captured CLI text must not
    assume plain output; strip color codes first so the test is deterministic either
    way, rather than asserting on colorized text that only happens to be plain in
    whatever environment the test was last run in.
    """
    return _ANSI_RE.sub("", text)


def plain(text: str) -> str:
    """Strip ANSI codes *and* collapse every run of whitespace to a single space.

    Rich hard-wraps to the console width, and under CliRunner that width is the
    ambient one (80 on a CI runner). Where a wrapped line contains an interpolated
    path, the break lands at a column that depends on how long that path happens
    to be -- so a `pytest-of-runner` temp path splits `"... is not valid UTF-8"`
    across a newline and fails an assertion that passes locally under a shorter
    username, for no reason connected to what the test is checking.

    Every assertion using this one is about what the CLI *says*, never about how
    it is laid out, so the fix is to compare against de-wrapped text rather than
    to pin a width the user's terminal does not have to agree with.
    """
    return " ".join(_ANSI_RE.sub("", text).split())

runner = CliRunner()

VALID_SUITE_YAML = """
task_name: cli_date_normalizer
spec: "Convert dates to ISO-8601"
adapter_path: "{adapter_path}"

standard_cases:
  - input: "today"
    expected: "2026-09-05"

assertions:
  - rule: regex_match
    pattern: '^(\\d{4}-\\d{2}-\\d{2}|INVALID)$'

fuzzing:
  empty_inputs: false
  adversarial_probes:
    - "February 30th"

active_learning:
  auto_recompile: true
  max_iterations: 2
"""


def test_cli_check_missing_file() -> None:
    """Verify check exits with code 1 when suite file is missing."""
    result = runner.invoke(app, ["check", "non_existent_suite.yaml"])
    assert result.exit_code == 1
    assert "does not exist" in plain(result.output)


def test_cli_check_invalid_yaml(tmp_path: Path) -> None:
    """Verify check exits with code 1 when suite YAML is invalid."""
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- not a mapping", encoding="utf-8")
    result = runner.invoke(app, ["check", str(bad_yaml)])
    assert result.exit_code == 1
    assert "Error parsing suite" in result.output


def test_cli_check_passing_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check exits with code 0 on passing test suite."""
    # PAW-CLI-02: auto_recompile is enabled (VALID_SUITE_YAML's default), so
    # adapter_path must resolve under cwd -- chdir into tmp_path so it does.
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "model.paw"
    # Seed mock adapter with standard output
    adapter_path.write_text(
        json.dumps({
            "backend": "mock",
            "spec": "Convert dates",
            "examples": [
                {"input": "today", "output": "2026-09-05"},
                {"input": "February 30th", "output": "INVALID"},
            ],
            "rules": {
                "today": "2026-09-05",
                "February 30th": "INVALID",
            },
        }),
        encoding="utf-8",
    )

    suite_path = tmp_path / "suite.yaml"
    suite_content = VALID_SUITE_YAML.replace("{adapter_path}", str(adapter_path))
    suite_path.write_text(suite_content, encoding="utf-8")

    result = runner.invoke(app, ["check", str(suite_path)])
    assert result.exit_code == 0
    # C-2: the pre-seeded adapter above already exists, so auto-recompile is now
    # disabled (any existing adapter is protected, not only non-mock ones) and this
    # run takes the read-only path -- which is still exit 0, since the seeded mock
    # adapter already passes every case. "All assertions passed! (Iterations: ...)"
    # is printed only by the active-learning branch, which does not run here.
    assert "Pass rate: 100.0% (2/2)" in result.output


def test_cli_check_json_report_carries_backend_C_5(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A `--json` consumer must be able to tell which backend actually ran.

    Before this fix, `TestRunReport` had no `backend` field at all -- the only place
    that ever recorded which backend ran was Rich console text.
    """
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "model.paw"
    adapter_path.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [{"input": "today", "output": "x"}]}),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(VALID_SUITE_YAML.replace("{adapter_path}", str(adapter_path)), encoding="utf-8")
    json_out = tmp_path / "report.json"

    result = runner.invoke(app, ["check", str(suite_path), "--json", str(json_out)])
    # Pass/fail is irrelevant here; only that a JSON report was written and carries
    # the backend that actually ran.
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert data["backend"] == "MockPAWBackend"


def test_cli_check_json_at_a_directory_is_refused_cleanly_C_12(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--json <existing directory>` must be refused up front, not crash after the
    whole run completes.

    Before this fix, `--json` had no `dir_okay=False`, so a directory argument was
    accepted, the entire suite ran, and only then did `Path.write_text` raise an
    uncaught `IsADirectoryError` -- discarding the report and, for an
    otherwise-passing run, turning exit 0 into a traceback.
    """
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "model.paw"
    adapter_path.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [{"input": "today", "output": "x"}]}),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(VALID_SUITE_YAML.replace("{adapter_path}", str(adapter_path)), encoding="utf-8")
    json_out_dir = tmp_path / "a_directory"
    json_out_dir.mkdir()

    result = runner.invoke(app, ["check", str(suite_path), "--json", str(json_out_dir)])

    assert result.exit_code != 0
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_cli_check_no_auto_recompile_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify check exits with code 1 when assertions fail and auto_recompile is disabled."""
    # PAW-TEST-02: load_suite now requires adapter_path to resolve under cwd
    # unconditionally (not only when auto_recompile could trigger a write).
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "model_fail.paw"
    adapter_path.write_text(
        json.dumps({
            "backend": "mock",
            "spec": "Convert dates",
            "examples": [],
            "rules": {"today": "INVALID_OUTPUT_DATE"},
        }),
        encoding="utf-8",
    )

    suite_path = tmp_path / "suite_fail.yaml"
    suite_content = VALID_SUITE_YAML.replace("{adapter_path}", str(adapter_path))
    suite_path.write_text(suite_content, encoding="utf-8")

    result = runner.invoke(app, ["check", str(suite_path), "--no-auto-recompile"])
    assert result.exit_code == 1
    assert "Pass rate:" in result.output


# --- Tool feedback (measurements/README.md, "Finetune compiler on a rule the base
# model does not know (fiscal weeks)"): `paw-test check` used to never read
# `expected`, so a suite carrying the exact answer in every case still scored
# `Pass rate: 100.0%`, exit 0, for an adapter that was wrong on every single case, as
# long as the suite's own assertion was loose enough not to notice. --------------


def test_cli_check_reports_wrong_adapter_as_failing_despite_a_loose_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression, directly reproducing the measured defect: a mock adapter that
    returns the same structurally-valid-but-wrong answer for every case, checked
    against a suite carrying the correct answer in each case's `expected`. `check`
    must report this as failing -- not `Pass rate: 100.0% (3/3)`, exit 0."""
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "fiscal_wrong.paw"
    adapter_path.write_text(
        json.dumps(
            {
                "backend": "mock",
                "spec": "Fiscal week",
                "examples": [],
                "rules": {},
                # Matches the assertion's shape unconditionally, but is never the
                # case's actual `expected` answer below.
                "default_response": "FY0000-W00",
            }
        ),
        encoding="utf-8",
    )

    suite_path = tmp_path / "fiscal_suite.yaml"
    suite_path.write_text(
        "task_name: fiscal_week\n"
        'spec: "Label the fiscal week for a date."\n'
        f'adapter_path: "{adapter_path}"\n'
        "standard_cases:\n"
        '  - input: "2026-03-03"\n'
        '    expected: "FY2026-W05"\n'
        '  - input: "2026-01-20"\n'
        '    expected: "FY2025-W51"\n'
        '  - input: "2027-02-01"\n'
        '    expected: "FY2027-W01"\n'
        "assertions:\n"
        r"  - rule: regex_match" "\n"
        r"    pattern: '^FY\d{4}-W\d{2}$'" "\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check", str(suite_path)])
    out = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "Pass rate: 0.0% (0/3)" in out
    assert "Correct against expected: 0/3 (0.0%)" in out


def test_cli_check_omits_expected_line_when_no_case_has_expected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: a suite with no `expected` anywhere must behave exactly as before
    this fix -- no "Correct against expected" line at all."""
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "model.paw"
    adapter_path.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [], "rules": {"today": "2026-09-05"}}),
        encoding="utf-8",
    )
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "task_name: no_expected\n"
        'spec: "Convert dates"\n'
        f'adapter_path: "{adapter_path}"\n'
        "standard_cases:\n"
        '  - input: "today"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 20\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check", str(suite_path)])
    out = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Correct against expected" not in out


# --- Quoted-scalar follow-up (measurements/README.md, "Tool feedback"): the fast
# compiler's lookup adapter quoted every output, scoring 0/300 against `expected`
# despite the unquoted answers being right a third of the time. `check` must surface
# that gap, but only when it exists. -----------------------------------------------


def test_cli_check_shows_unquoted_line_when_adapter_quotes_correct_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter that answers every case correctly, but JSON-string-quoted, must
    still score 0 against the strict "Correct against expected" line (a quoted output
    is a real defect) but also report that the underlying lookup was actually right.

    REWRITTEN by G-5 (bug-hunt-remediation Track B; see "Justified assertion changes").
    The behaviour asserted here is unchanged -- 0/2 strict, 2 recoverable by unquoting.
    What changed is how those two numbers are *presented*. This test previously pinned
    the exact old wording, `"Correct after unquoting a JSON string: 2/2 (100.0%)"`, a
    second headline printed with equal weight directly above `Pass rate: 0.0%` with
    nothing saying which of the two a reader should believe -- G-5's finding verbatim.
    The scored number is now the headline and the unquoted count is subordinate to it,
    so the assertion had to move with the text.
    """
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "quoted_sku.paw"
    adapter_path.write_text(
        json.dumps(
            {
                "backend": "mock",
                "spec": "SKU lookup",
                "examples": [],
                "rules": {
                    "widget-a": '"RG-M2"',
                    "widget-b": '"RG-M3"',
                },
            }
        ),
        encoding="utf-8",
    )

    suite_path = tmp_path / "sku_suite.yaml"
    suite_path.write_text(
        "task_name: sku_lookup\n"
        'spec: "Look up the SKU for a product."\n'
        f'adapter_path: "{adapter_path}"\n'
        "standard_cases:\n"
        '  - input: "widget-a"\n'
        '    expected: "RG-M2"\n'
        '  - input: "widget-b"\n'
        '    expected: "RG-M3"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 100\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check", str(suite_path)])
    out = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "Correct against expected: 0/2 (0.0%)" in out
    # Rich word-wraps long lines at the console width, so compare against
    # whitespace-collapsed output rather than the exact line.
    collapsed = " ".join(out.split())
    assert (
        "Of the ones not counted correct, 2 would match if a JSON string quote were "
        "stripped -- the adapter wraps its answers in quotes" in collapsed
    )
    # G-5: the scored number is stated again inside the subordinate clause, so the two
    # figures cannot be read as two competing headlines.
    assert "the scored number above stays 0/2" in collapsed


def test_cli_check_omits_unquoted_line_when_it_would_add_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: when the unquoted count doesn't exceed the strict count (nothing
    is actually quoted), the second line must not appear -- covers both the
    already-fully-correct case and the still-fully-wrong-even-unquoted case."""
    monkeypatch.chdir(tmp_path)
    adapter_path = tmp_path / "plain_sku.paw"
    adapter_path.write_text(
        json.dumps(
            {
                "backend": "mock",
                "spec": "SKU lookup",
                "examples": [],
                "rules": {"widget-a": "RG-M2"},
            }
        ),
        encoding="utf-8",
    )

    suite_path = tmp_path / "sku_suite.yaml"
    suite_path.write_text(
        "task_name: sku_lookup\n"
        'spec: "Look up the SKU for a product."\n'
        f'adapter_path: "{adapter_path}"\n'
        "standard_cases:\n"
        '  - input: "widget-a"\n'
        '    expected: "RG-M2"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 100\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["check", str(suite_path)])
    out = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "Correct against expected: 1/1 (100.0%)" in out
    assert "Correct after unquoting a JSON string" not in out


def test_cli_check_adapter_flag_overrides_suite_adapter_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--adapter` runs the suite against a different compiled adapter, so one suite
    can be checked against several adapters without a near-identical suite.yaml per
    adapter (measurements/README.md, Tool feedback point 2)."""
    monkeypatch.chdir(tmp_path)
    suite_adapter = tmp_path / "suite_adapter.paw"
    suite_adapter.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [], "rules": {"today": "WRONG"}}),
        encoding="utf-8",
    )
    override_adapter = tmp_path / "override.paw"
    override_adapter.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [], "rules": {"today": "2026-09-05"}}),
        encoding="utf-8",
    )

    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "task_name: override_test\n"
        'spec: "s"\n'
        f'adapter_path: "{suite_adapter}"\n'
        "standard_cases:\n"
        '  - input: "today"\n'
        '    expected: "2026-09-05"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 20\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )

    # Against the suite's own adapter_path: fails (it answers "WRONG").
    without_override = runner.invoke(app, ["check", str(suite_path)])
    assert without_override.exit_code == 1

    # Against the override: passes, and the header reports the overridden path.
    with_override = runner.invoke(app, ["check", str(suite_path), "--adapter", str(override_adapter)])
    out = strip_ansi(with_override.output)
    assert with_override.exit_code == 0
    assert "Pass rate: 100.0% (1/1)" in out
    assert str(override_adapter) in out


def test_cli_check_adapter_flag_rejects_path_outside_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--adapter` keeps the same containment guarantee suite.yaml's own adapter_path
    gets (PAW-TEST-02): an override outside cwd is rejected, not silently used."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    adapter_path = workdir / "in.paw"
    adapter_path.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [], "rules": {"today": "2026-09-05"}}),
        encoding="utf-8",
    )
    suite_path = workdir / "suite.yaml"
    suite_path.write_text(
        "task_name: override_outside_test\n"
        'spec: "s"\n'
        'adapter_path: "in.paw"\n'
        "standard_cases:\n"
        '  - input: "today"\n'
        '    expected: "2026-09-05"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 20\n"
        "active_learning:\n"
        "  auto_recompile: false\n",
        encoding="utf-8",
    )
    outside_adapter = tmp_path / "outside.paw"
    outside_adapter.write_text(
        json.dumps({"backend": "mock", "spec": "s", "examples": [], "rules": {"today": "2026-09-05"}}),
        encoding="utf-8",
    )

    monkeypatch.chdir(workdir)
    result = runner.invoke(app, ["check", str(suite_path), "--adapter", str(outside_adapter)])
    out = strip_ansi(result.output)

    assert result.exit_code == 1
    assert "not contained within" in " ".join(out.split())


def test_cli_check_help_mentions_adapter_override() -> None:
    result = runner.invoke(app, ["check", "--help"])
    assert result.exit_code == 0
    # `plain`, not the raw output: Typer renders its help through Rich, which
    # colorizes on a CI runner even where the app's own output there is plain.
    assert "--adapter" in plain(result.output)
    assert "adapter_path" in plain(result.output)


def test_cli_check_rejects_adapter_path_outside_cwd_PAW_CLI_02(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a suite.yaml adapter_path outside cwd is rejected before any recompile write.

    Mirrors the audit's own attack scenario: a contributed suite.yaml sets
    adapter_path to something like "../../.github/workflows/deploy.yml" so that a
    later `active_backend.compile(..., output_path=adapter_path)` overwrites an
    arbitrary file when assertions fail and auto_recompile (the suite default) fires.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside_target = tmp_path / "outside_target.paw"
    outside_target.write_text("pretend this is someone else's file", encoding="utf-8")

    monkeypatch.chdir(workdir)
    suite_path = Path("suite.yaml")
    suite_content = VALID_SUITE_YAML.replace("{adapter_path}", str(outside_target))
    suite_path.write_text(suite_content, encoding="utf-8")

    result = runner.invoke(app, ["check", str(suite_path)])
    assert result.exit_code == 1
    assert "not contained within" in " ".join(result.output.split())
    # The out-of-bounds file must be untouched, not overwritten by a recompile.
    assert outside_target.read_text(encoding="utf-8") == "pretend this is someone else's file"


def test_cli_check_rejects_adapter_path_outside_cwd_even_without_recompile_PAW_TEST_02(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify an out-of-cwd adapter_path is rejected even when auto_recompile is off.

    Superseded by Phase 6 (PAW-TEST-02): Phase 3's PAW-CLI-02 fix only validated
    containment in cli.py's own recompile-triggering path, so a suite with
    auto_recompile disabled (no write could occur) was intentionally left unchecked.
    Phase 6 tightens this at the suite loader itself (suite.py's load_suite), applying
    unconditionally regardless of auto_recompile -- so this scenario, which used to be
    allowed, is now rejected too.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside_adapter = tmp_path / "outside_model.paw"
    outside_adapter.write_text(
        json.dumps({
            "backend": "mock",
            "spec": "Convert dates",
            "examples": [
                {"input": "today", "output": "2026-09-05"},
                {"input": "February 30th", "output": "INVALID"},
            ],
            "rules": {
                "today": "2026-09-05",
                "February 30th": "INVALID",
            },
        }),
        encoding="utf-8",
    )

    monkeypatch.chdir(workdir)
    suite_path = Path("suite.yaml")
    suite_content = VALID_SUITE_YAML.replace("{adapter_path}", str(outside_adapter))
    suite_path.write_text(suite_content, encoding="utf-8")

    result = runner.invoke(app, ["check", str(suite_path), "--no-auto-recompile"])
    assert result.exit_code == 1
    assert "not contained within" in " ".join(result.output.split())


def test_cli_inspect_adapter(tmp_path: Path) -> None:
    """Verify inspect displays properties for both JSON and binary adapter files."""
    # Missing file
    res_err = runner.invoke(app, ["inspect", str(tmp_path / "missing.paw")])
    assert res_err.exit_code == 1

    # JSON adapter
    json_adapter = tmp_path / "triage.paw"
    json_adapter.write_text(
        json.dumps({"spec": "Ticket triage", "backend": "mock", "examples_count": 42}),
        encoding="utf-8",
    )
    res_json = runner.invoke(app, ["inspect", str(json_adapter)])
    assert res_json.exit_code == 0
    assert "Ticket triage" in res_json.output
    assert "JSON Simulation" in res_json.output

    # Binary adapter
    bin_adapter = tmp_path / "weights.bin"
    bin_adapter.write_bytes(b"\x00\x01\x02\x03" * 100)
    res_bin = runner.invoke(app, ["inspect", str(bin_adapter)])
    assert res_bin.exit_code == 0
    assert "Binary / Raw Weights" in res_bin.output


def test_cli_inspect_shows_program_id_and_compiler(tmp_path: Path) -> None:
    """A ProgramAsWeights-style manifest's program_id and compiler must be visible --
    the whole point of `paw-inspect` is answering "what did this adapter compile to,"
    and those two fields were previously dropped entirely."""
    adapter = tmp_path / "real.paw"
    adapter.write_text(
        json.dumps(
            {
                "backend": "programasweights",
                "manifest_version": 2,
                "program_id": "prog_abc123",
                "compiler": "paw-4b-qwen3-0.6b",
                "compiled_at": "2026-01-01T00:00:00Z",
                "spec": "Normalize a date.",
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["inspect", str(adapter)])
    out = strip_ansi(result.output)
    assert result.exit_code == 0
    assert "prog_abc123" in out
    assert "paw-4b-qwen3-0.6b" in out
    # Spec is printed last among the manifest-derived rows.
    spec_pos = out.index("Normalize a date.")
    program_id_pos = out.index("prog_abc123")
    assert program_id_pos < spec_pos


def test_cli_inspect_json_flag_prints_raw_manifest(tmp_path: Path) -> None:
    adapter = tmp_path / "a.paw"
    manifest = {"backend": "mock", "spec": "s", "manifest_version": 2, "examples_count": 0}
    adapter.write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(app, ["inspect", str(adapter), "--json"])
    assert result.exit_code == 0
    assert json.loads(strip_ansi(result.output)) == manifest


def test_cli_inspect_json_flag_errors_on_non_json_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "weights.bin"
    adapter.write_bytes(b"\x00\x01\x02\x03")
    result = runner.invoke(app, ["inspect", str(adapter), "--json"])
    assert result.exit_code == 1


def test_cli_history_appended_twice_and_printed(tmp_path: Path) -> None:
    """Two compiles append two lines to the sidecar log, and `paw-kit history`
    prints both."""
    from paw_kit.backend.mock import MockPAWBackend

    backend = MockPAWBackend()
    adapter = tmp_path / "a.paw"
    backend.compile("v1", [{"input": "a", "output": "1"}], str(adapter))
    backend.compile("v2", [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}], str(adapter))

    result = runner.invoke(app, ["history", str(adapter)])
    out = strip_ansi(result.output)
    assert result.exit_code == 0
    assert out.count("mock") >= 2
    # Two data rows, not counting the header.
    assert "1" in out and "2" in out


def test_cli_history_compiled_at_is_short_timestamp_G_6(tmp_path: Path) -> None:
    """`history` must render `compiled_at` through `_short_timestamp`, the same way
    `report` already renders `promoted_at`/`demoted_at` -- otherwise the same kind of
    value prints two different ways depending on which command shows it.

    `compiled_at` is only ever written by `ProgramAsWeightsBackend` (the real
    backend writes it; `MockPAWBackend` has none, by design -- see `mock.py`'s own
    comment), so the sidecar line is built directly here rather than via a real
    compile, matching the field a real backend's history line actually carries.
    """
    adapter = tmp_path / "a.paw"
    adapter.write_text(json.dumps({"backend": "programasweights"}), encoding="utf-8")
    raw_compiled_at = "2026-09-13T02:30:45Z"
    assert len(raw_compiled_at) > 19  # a full ISO-8601 timestamp
    history_path = tmp_path / "a.paw.history.jsonl"
    history_path.write_text(json.dumps({"compiled_at": raw_compiled_at}) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["history", str(adapter)])
    out = strip_ansi(result.output)
    assert result.exit_code == 0
    assert raw_compiled_at not in out
    # Rich's narrow "Compiled At" column wraps the date and time onto two table
    # lines with a border between them, so check each half rather than one
    # contiguous "date time" string.
    short = raw_compiled_at[:19].replace("T", " ")
    date_part, time_part = short.split(" ")
    assert date_part in out
    assert time_part in out


def test_cli_history_reads_the_rotated_generation_D_ADD_2(tmp_path: Path) -> None:
    """`history` must show the rotated-out `.1` generation too, oldest first.

    `manifest_lineage.py`'s rotation (`_HISTORY_ROTATIONS = 1`) moves a full sidecar
    to `<adapter>.history.jsonl.1` so a long-lived adapter's lineage stays bounded --
    but before this fix, `history` read only the live file, so the oldest lineage was
    retained on disk but invisible to the one command that exists to show it.
    """
    adapter = tmp_path / "a.paw"
    adapter.write_text(json.dumps({"backend": "programasweights"}), encoding="utf-8")
    rotated = tmp_path / "a.paw.history.jsonl.1"
    rotated.write_text(json.dumps({"program_id": "prog_old", "compiler": "fast"}) + "\n", encoding="utf-8")
    live = tmp_path / "a.paw.history.jsonl"
    live.write_text(json.dumps({"program_id": "prog_new", "compiler": "finetune"}) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["history", str(adapter)])
    out = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "prog_old" in out
    assert "prog_new" in out
    # Oldest first: the rotated entry's row number must precede the live entry's.
    assert out.index("prog_old") < out.index("prog_new")


def test_cli_history_reads_only_the_rotated_generation_when_live_file_is_absent_D_ADD_2(
    tmp_path: Path,
) -> None:
    """A `.1` generation with no live file yet (freshly rotated) is not an error."""
    adapter = tmp_path / "a.paw"
    adapter.write_text(json.dumps({"backend": "programasweights"}), encoding="utf-8")
    rotated = tmp_path / "a.paw.history.jsonl.1"
    rotated.write_text(json.dumps({"program_id": "prog_old"}) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["history", str(adapter)])
    out = strip_ansi(result.output)

    assert result.exit_code == 0
    assert "prog_old" in out


def test_cli_history_missing_log_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", str(tmp_path / "nope.paw")])
    assert result.exit_code == 1
    assert "no history log" in strip_ansi(result.output)


def test_cli_history_survives_a_non_utf8_byte_in_the_sidecar_C_8(tmp_path: Path) -> None:
    """One non-UTF-8 byte anywhere in the sidecar must not crash the command.

    Before this fix, `log_path.read_text(encoding="utf-8")` raised an uncaught
    UnicodeDecodeError -- defeating the size/JSON-corruption defences this command
    already has, which all assume a decode failure cannot happen.
    """
    from paw_kit.backend.mock import MockPAWBackend

    adapter = tmp_path / "a.paw"
    MockPAWBackend().compile("v1", [{"input": "a", "output": "1"}], str(adapter))
    history_path = tmp_path / "a.paw.history.jsonl"
    with open(history_path, "ab") as f:
        f.write(b"\xff\xfe not valid utf-8\n")

    result = runner.invoke(app, ["history", str(adapter)])

    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert result.exit_code == 0
    # The one legitimate (valid-JSON) line still prints; the corrupted line is
    # skipped by the existing per-line JSON guard, same as any other malformed line.
    assert "mock" in strip_ansi(result.output)


def test_cli_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify clean command handles missing dirs, dry-run, confirmation, and actual purging."""
    # PAW-CLI-01: cache_dir must resolve under cwd, so exercise this from a cwd chdir'd
    # into tmp_path, using a cache_dir that is a genuine subdirectory of it.
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("test_cache")

    # Non-existent
    res_empty = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir)])
    assert res_empty.exit_code == 0
    assert "does not exist" in plain(res_empty.output)

    cache_dir.mkdir()
    f1 = cache_dir / "trace.db"
    f1.write_text("trace", encoding="utf-8")

    # Dry-run
    res_dry = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir), "--dry-run"])
    assert res_dry.exit_code == 0
    assert "Dry run" in res_dry.output
    # G-3: "1 file", not "1 files".
    assert "1 file in" in res_dry.output
    assert "1 files in" not in res_dry.output
    assert f1.exists()

    # Real clean, declining the confirmation prompt: file survives
    res_decline = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir)], input="n\n")
    assert res_decline.exit_code == 0
    assert "Aborted" in res_decline.output
    assert f1.exists()

    # Real clean via --yes (skips the prompt): file is removed
    res_real = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir), "--yes"])
    assert res_real.exit_code == 0
    assert "Cache cleaned successfully" in res_real.output
    assert not f1.exists()


def test_cli_clean_confirmation_prompt_accept_PAW_CLI_01(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify answering 'y' to the confirmation prompt (no --yes flag) deletes files."""
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("cache")
    cache_dir.mkdir()
    f1 = cache_dir / "trace.db"
    f1.write_text("trace", encoding="utf-8")

    result = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir)], input="y\n")
    assert result.exit_code == 0
    assert "Cache cleaned successfully" in result.output
    assert not f1.exists()


def test_cli_clean_rejects_cwd_itself_PAW_CLI_01(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `paw-clean -c .` (the exact audit attack scenario) is rejected, not run."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "important_source_file.py").write_text("do not delete me", encoding="utf-8")

    result = runner.invoke(app, ["clean", "--cache-dir", "."])
    assert result.exit_code == 1
    assert "not contained within" in " ".join(result.output.split())
    assert (tmp_path / "important_source_file.py").exists()


def test_cli_clean_rejects_path_outside_cwd_PAW_CLI_01(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a --cache-dir pointing outside cwd (the /etc/my_app audit scenario) is rejected."""
    workdir = tmp_path / "project"
    workdir.mkdir()
    outside = tmp_path / "unrelated_directory"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unrelated data", encoding="utf-8")

    monkeypatch.chdir(workdir)
    result = runner.invoke(app, ["clean", "--cache-dir", str(outside), "--yes"])
    assert result.exit_code == 1
    assert "not contained within" in " ".join(result.output.split())
    assert sentinel.exists()


def test_cli_clean_reports_failure_when_every_delete_fails_C_3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run where every delete fails must not print success or exit 0.

    Before this fix the per-file `except` printed a red failure line and continued;
    nothing accumulated those failures, so "Cache cleaned successfully." and exit 0
    were unconditional -- reproduced here exactly as the report did, with the cache
    dir made undeletable.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores permission bits, so this reproduction cannot fire")

    monkeypatch.chdir(tmp_path)
    cache_dir = Path("cache")
    cache_dir.mkdir()
    f1 = cache_dir / "trace.db"
    f1.write_text("trace", encoding="utf-8")
    cache_dir.chmod(0o500)  # r-x: list allowed, unlink inside denied
    try:
        result = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir), "--yes"])
    finally:
        cache_dir.chmod(0o700)  # restore so tmp_path cleanup can remove it

    assert result.exit_code == 1
    assert "Cache cleaned successfully" not in result.output
    assert "could not be deleted" in result.output
    assert f1.exists()


def test_cli_clean_reports_directories_separately_and_does_not_delete_them_C_7(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory entry must be reported apart from files, never counted as purged.

    Before this fix, `files_to_remove = list(resolved_cache.glob("*"))` counted and
    listed a subdirectory under "Purging N files", then `if file.is_file(): unlink()`
    silently skipped it -- "Cache cleaned successfully." printed anyway with no mention
    that the directory was left behind.
    """
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("cache")
    cache_dir.mkdir()
    f1 = cache_dir / "trace.db"
    f1.write_text("trace", encoding="utf-8")
    subdir = cache_dir / "examples_cache"
    subdir.mkdir()
    (subdir / "nested.txt").write_text("x", encoding="utf-8")

    result = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir), "--yes"])

    assert result.exit_code == 0
    assert "Cache cleaned successfully" in result.output
    assert not f1.exists()
    assert subdir.exists()  # left in place, not silently "cleaned"
    assert "subdirectory" in result.output
    assert "examples_cache" in result.output


def test_cli_demo_triage() -> None:
    """Verify paw-kit demo runs the triage demo cleanly with code 0."""
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    assert "Ticket Triage" in result.output
    assert "LOCAL (mock)" in result.output


def test_cli_demo_pii() -> None:
    """Verify paw-kit demo --scenario pii runs the PII scrubber demo with code 0."""
    result = runner.invoke(app, ["demo", "--scenario", "pii"])
    assert result.exit_code == 0
    assert "PII Scrubber" in result.output
    assert "REDACTED" in result.output


def test_cli_demo_invalid_scenario() -> None:
    """Verify paw-kit demo with invalid scenario exits with code 1."""
    result = runner.invoke(app, ["demo", "--scenario", "invalid_scenario"])
    assert result.exit_code == 1
    assert "Unknown scenario" in result.output


def test_cli_demo_triage_cleans_up_temp_dir_on_exception_PAW_CLI_08(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PAW-CLI-08: the demo's temp directory is removed even when an exception
    propagates mid-run -- tempfile.TemporaryDirectory's context manager cleans up on
    every exit path, unlike the old manual mkdtemp()+rmtree() pair, which only reached
    rmtree() after every ticket in the loop finished without raising."""
    import tempfile as tempfile_module

    real_mkdtemp = tempfile_module.mkdtemp
    captured: dict = {}

    def spy_mkdtemp(*args: object, **kwargs: object) -> str:
        path = real_mkdtemp(*args, **kwargs)
        prefix = kwargs.get("prefix") or (args[1] if len(args) > 1 else None)
        if prefix == "paw_demo_":
            captured["path"] = path
        return path

    monkeypatch.setattr(tempfile_module, "mkdtemp", spy_mkdtemp)

    import paw_kit.cli as cli_module

    real_sleep = cli_module.time.sleep
    call_count = {"n": 0}

    def flaky_sleep(seconds: float) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated mid-demo failure")
        real_sleep(seconds)

    monkeypatch.setattr(cli_module.time, "sleep", flaky_sleep)

    result = runner.invoke(app, ["demo"])
    assert result.exit_code != 0
    assert "path" in captured
    assert not Path(captured["path"]).exists()


def test_cli_inspect_rejects_oversized_file_PAW_CLI_06(tmp_path: Path) -> None:
    """Verify PAW-CLI-06: a file larger than the JSON-parse cap is reported as binary
    rather than fed to json.load, which would read the whole thing into memory first."""
    import paw_kit.cli as cli_module

    big_adapter = tmp_path / "huge.paw"
    # Write real (but oversized) JSON content so a successful parse -- were the cap
    # not enforced -- would otherwise happen, isolating the size guard as the cause.
    with open(big_adapter, "w", encoding="utf-8") as f:
        f.write('{"spec": "x", "padding": "')
        f.write("a" * (cli_module._MAX_INSPECT_FILE_BYTES + 1))
        f.write('"}')

    result = runner.invoke(app, ["inspect", str(big_adapter)])
    assert result.exit_code == 0
    assert "Binary / Raw Weights" in result.output


def test_cli_inspect_rejects_non_regular_file_PAW_CLI_06(tmp_path: Path) -> None:
    """Verify PAW-CLI-06: a non-regular file (e.g. a named pipe, standing in for the
    audit's /dev/zero scenario) is never opened for parsing -- json.load on a stream
    with no natural end-of-file would hang the process indefinitely."""
    import os

    fifo_path = tmp_path / "not_a_regular_file.paw"
    os.mkfifo(fifo_path)

    result = runner.invoke(app, ["inspect", str(fifo_path)])
    assert result.exit_code == 0
    assert "Binary / Raw Weights" in result.output


def test_cli_export_dataset_real_schema_PAW_CLI_04(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PAW-CLI-04: `paw export dataset` succeeds against a real TraceDB. The
    query used to name columns ("input", "output") that no TraceDB schema has ever
    had -- the real columns are input_payload/teacher_output -- so this exact
    invocation raised sqlite3.OperationalError on every real database before the fix,
    caught by a bare `except Exception` and reported as a generic export failure."""
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("real_traces.db")
    trace_db = TraceDB(str(db_file))
    trace_db.record_trace(
        task_id="t1", input_payload="hello", teacher_output="world", latency_ms=1.0
    )

    out_file = Path("out.jsonl")
    result = runner.invoke(app, ["export", "dataset", "--db", str(db_file), "--out", str(out_file)])
    assert result.exit_code == 0
    assert out_file.exists()
    record = json.loads(out_file.read_text(encoding="utf-8").strip())
    assert record["messages"][0]["content"] == "hello"
    assert record["messages"][1]["content"] == "world"


def test_cli_export_dataset_empty_db_exits_zero_PAW_CLI_04(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-CLI-04's second half: an empty real TraceDB now exits 0, not 1.
    `typer.Exit(code=0)` for "no traces found" used to sit inside the write's
    try/except -- typer.Exit subclasses RuntimeError, so the bare `except Exception`
    there swallowed it and re-raised as exit 1, printing both the warning and
    "Error exporting dataset:" for what should have been a clean no-op."""
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("empty_traces.db")
    TraceDB(str(db_file))  # creates the schema; no rows recorded

    result = runner.invoke(app, ["export", "dataset", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "No traces found" in result.output
    assert "Error exporting dataset" not in result.output


def test_cli_export_dataset_file_created_0600_PAW_CLI_05(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-CLI-05: the exported JSONL file is created 0600 by `os.open`'s mode
    argument, not by a bare `open(..., "w")` subject to the process umask (commonly
    0644, world-readable). Exported content can include unredacted prompts/PII
    (PAW-JIT-02), so it must never be world-readable, not even briefly.

    Added at Phase F: the 0600 assertion existed only inside
    `tests/test_serve.py::test_cli_export_commands`, which is not named for this
    finding, so PAW-CLI-05 was the one finding of the 32 without a dedicated
    finding-ID-named regression test as Phase T requires.
    """
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("perm_traces.db")
    trace_db = TraceDB(str(db_file))
    trace_db.record_trace(task_id="t1", input_payload="in", teacher_output="out", latency_ms=1.0)

    # A permissive umask would leave a bare open(..., "w") at 0666; os.open's mode
    # argument is what actually holds the file at 0600 regardless.
    old_umask = os.umask(0o000)
    try:
        out_file = Path("perm.jsonl")
        result = runner.invoke(app, ["export", "dataset", "--db", str(db_file), "--out", str(out_file)])
        assert result.exit_code == 0
        assert (out_file.stat().st_mode & 0o777) == 0o600
    finally:
        os.umask(old_umask)


def test_cli_export_dataset_requires_jsonl_extension_PAW_CLI_03(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-CLI-03: --out must end in .jsonl."""
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("traces.db")
    TraceDB(str(db_file)).record_trace(task_id="t", input_payload="i", teacher_output="o", latency_ms=1.0)

    result = runner.invoke(app, ["export", "dataset", "--db", str(db_file), "--out", "dataset.txt"])
    assert result.exit_code == 1
    assert "'.jsonl' extension" in strip_ansi(result.output)


def test_cli_export_dataset_confirms_before_overwrite_PAW_CLI_03(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-CLI-03: an existing --out is not silently clobbered -- declining the
    prompt leaves it untouched, and --force skips the prompt entirely."""
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("traces.db")
    TraceDB(str(db_file)).record_trace(task_id="t", input_payload="i", teacher_output="o", latency_ms=1.0)

    out_file = Path("dataset.jsonl")
    out_file.write_text("pre-existing content\n", encoding="utf-8")

    # Decline: file is untouched.
    res_decline = runner.invoke(
        app, ["export", "dataset", "--db", str(db_file), "--out", str(out_file)], input="n\n"
    )
    assert res_decline.exit_code == 0
    assert "Aborted" in res_decline.output
    assert out_file.read_text(encoding="utf-8") == "pre-existing content\n"

    # --force: overwrites without prompting.
    res_force = runner.invoke(
        app, ["export", "dataset", "--db", str(db_file), "--out", str(out_file), "--force"]
    )
    assert res_force.exit_code == 0
    assert out_file.read_text(encoding="utf-8") != "pre-existing content\n"


def test_cli_export_dataset_overwrite_prompt_is_not_backslash_mangled_C_9(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overwrite prompt must show the literal filename, not a Rich-escaped one.

    `typer.confirm` is click, not Rich -- it prints raw. Before this fix the
    filename was run through `_e()` (Rich's markup escaper) first, which *added*
    visible backslashes: `[v2]out.jsonl` prompted `\\[v2]out.jsonl already exists.
    Overwrite?`, on the one prompt whose whole job is to name the file about to be
    destroyed.
    """
    from paw_kit.jit.db import TraceDB

    monkeypatch.chdir(tmp_path)
    db_file = Path("traces.db")
    TraceDB(str(db_file)).record_trace(task_id="t", input_payload="i", teacher_output="o", latency_ms=1.0)

    out_file = Path("[v2]dataset.jsonl")
    out_file.write_text("pre-existing content\n", encoding="utf-8")

    result = runner.invoke(
        app, ["export", "dataset", "--db", str(db_file), "--out", str(out_file)], input="n\n"
    )
    assert result.exit_code == 0
    assert "[v2]dataset.jsonl already exists" in result.output
    assert "\\[v2]dataset.jsonl" not in result.output


def test_cli_serve_api_key_warns_on_commandline_PAW_CLI_07(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify PAW-CLI-07: passing --api-key on the command line prints a warning (the
    process-table/shell-history exposure the finding is about), while setting
    PAW_API_KEY in the environment instead does not."""
    from paw_kit.serve import server

    monkeypatch.setattr(server, "serve_adapter", lambda *a, **k: None)
    adapter = tmp_path / "a.paw"
    adapter.write_text("{}", encoding="utf-8")

    res_cli_flag = runner.invoke(app, ["serve", str(adapter), "--api-key", "secret-on-argv"])
    assert res_cli_flag.exit_code == 0
    assert "Warning" in res_cli_flag.output
    assert "ps aux" in res_cli_flag.output

    monkeypatch.setenv("PAW_API_KEY", "secret-from-env")
    res_env = runner.invoke(app, ["serve", str(adapter)])
    assert res_env.exit_code == 0
    assert "Warning" not in res_env.output



# ---------------------------------------------------------------- backend resolution


def test_resolve_cli_backend_mock_is_default():
    """`--backend mock` (the default) resolves to MockPAWBackend, unchanged."""
    from paw_kit.backend.mock import MockPAWBackend
    from paw_kit.cli import _resolve_cli_backend

    assert isinstance(_resolve_cli_backend("mock"), MockPAWBackend)


def test_resolve_cli_backend_real_returns_upstream_not_mock(monkeypatch):
    """`--backend real` resolves to ProgramAsWeightsBackend when the SDK is available.

    Regression test for the central bug this replaced: `_resolve_cli_backend` used to
    return `MockPAWBackend()` on *every* path, so `paw-test check --backend real`
    silently exercised a dictionary lookup instead of a compiled adapter.
    """
    from paw_kit.backend.mock import MockPAWBackend
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend
    from paw_kit.cli import _resolve_cli_backend

    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "has_api_key", lambda self: True)
    # _resolve_cli_backend forces the SDK import (is_available only calls find_spec).
    # Stub it: the real import succeeds on a dev box with the [real] extra and fails in
    # CI without it, which would make this test's result depend on the environment.
    monkeypatch.setattr(ProgramAsWeightsBackend, "_paw", lambda self: object())

    backend = _resolve_cli_backend("real")
    assert isinstance(backend, ProgramAsWeightsBackend)
    assert not isinstance(backend, MockPAWBackend)


def test_resolve_cli_backend_real_falls_back_loudly_without_sdk(monkeypatch, capsys):
    """Without the upstream SDK, `--backend real` degrades to mock but says so."""
    from paw_kit.backend.mock import MockPAWBackend
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend
    from paw_kit.cli import _resolve_cli_backend

    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: False)

    backend = _resolve_cli_backend("real")
    assert isinstance(backend, MockPAWBackend)
    # C-5: this fallback warning now goes to stderr, not stdout.
    captured = capsys.readouterr()
    assert captured.out == ""
    err = strip_ansi(captured.err)
    assert "not a model" in err
    assert "programasweights" in err


def test_resolve_cli_backend_real_without_api_key_still_returns_upstream(monkeypatch, capsys):
    """A missing PAW_API_KEY is a warning, not a downgrade: cached-program inference works."""
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend
    from paw_kit.cli import _resolve_cli_backend

    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "has_api_key", lambda self: False)
    monkeypatch.setattr(ProgramAsWeightsBackend, "_paw", lambda self: object())

    backend = _resolve_cli_backend("real")
    assert isinstance(backend, ProgramAsWeightsBackend)
    assert "PAW_API_KEY is not set" in strip_ansi(capsys.readouterr().out)


def test_resolve_cli_backend_rejects_unknown_value():
    """A typo'd backend name is an error, not a silent fall back to mock."""
    import typer
    from paw_kit.cli import _resolve_cli_backend

    with pytest.raises(typer.BadParameter):
        _resolve_cli_backend("rael")


def test_resolve_cli_backend_real_falls_back_when_sdk_present_but_unimportable(monkeypatch, capsys):
    """A present-but-broken SDK must fail at resolution, not silently as a 0% pass rate.

    `is_available()` only calls `find_spec` -- it never imports. A package that is
    installed but unimportable (mismatched llama_cpp, broken CUDA build, half-finished
    install) therefore used to pass the availability check, get announced as
    `ProgramAsWeightsBackend`, and then raise inside every `infer()`, where TestRunner's
    blanket `except Exception` turned it into "[EXECUTION_ERROR]" and a 0.0% pass rate.
    The user, having been told they were on a real backend, reads that as the *model*
    failing their assertions. Infrastructure failure must not masquerade as model failure.
    """
    from paw_kit.backend.mock import MockPAWBackend
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend
    from paw_kit.cli import _resolve_cli_backend

    def _boom(self):
        raise ImportError("libllama.so: cannot open shared object file")

    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "_paw", _boom)

    backend = _resolve_cli_backend("real")

    assert isinstance(backend, MockPAWBackend)
    # C-5: this fallback warning now goes to stderr, not stdout -- a caller piping
    # stdout to a report/log file must still see it.
    captured = capsys.readouterr()
    assert captured.out == ""
    err = strip_ansi(captured.err)
    assert "could not be loaded" in err
    assert "libllama.so" in err
    assert "not a model" in err


def _real_backend_suite(tmp_path):
    """A suite with auto_recompile on, plus a stub adapter, for the guard tests below."""
    adapter = tmp_path / "guard.paw"
    adapter.write_text(
        json.dumps({"backend": "mock", "task_name": "guard", "spec": "s", "examples": []})
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: guard\n"
        'spec: "Normalize a date."\n'
        f'adapter_path: "{adapter.name}"\n'
        "standard_cases:\n"
        '  - input: "January 15, 2026"\n'
        '    expected: "2026-01-15"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 3\n"
    )
    return suite


def test_check_real_backend_fallback_disables_auto_recompile_too_C_13(tmp_path, monkeypatch):
    """A fallback from --backend real to MockPAWBackend must not silently compile a
    mock adapter at the path the user asked for a real compile at.

    Before this fix, `is_real = not isinstance(backend, MockPAWBackend)` was False
    after a fallback, so this guard never fired for exactly the scenario it exists to
    prevent: a fresh `--backend real` invocation (no existing adapter, so C-2's
    existence guard does not apply either) with an unavailable SDK compiled a *mock*
    adapter from cli_teacher's fabricated labels at the real adapter's path, and could
    report `[SUCCESS]` with nothing distinguishing it from an actual real compile.
    """
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend

    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: False)
    adapter = tmp_path / "prod.paw"
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: c13\n"
        'spec: "Normalize a date."\n'
        f'adapter_path: "{adapter.name}"\n'
        "standard_cases:\n"
        '  - input: "February 30, 2026"\n'
        '    expected: "INVALID"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 2\n"
    )
    monkeypatch.chdir(tmp_path)
    assert not adapter.exists()

    result = runner.invoke(paw_test_app, ["check", "suite.yaml", "--backend", "real"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" in out
    assert "fell back to" in out
    assert "MockPAWBackend" in out
    # The point of the test: no mock adapter was silently compiled at this path.
    assert not adapter.exists()


def test_check_real_backend_disables_auto_recompile_by_default(tmp_path, monkeypatch, capsys):
    """`--backend real` must not fire a paid, destructive recompile implicitly.

    auto_recompile defaults to True and the shipped example suite sets it true, so before
    this guard `paw-test check <suite> --backend real` -- one flag added to the command
    the README prints -- submitted up to max_iterations-1 real upstream compiles built on
    labels invented by the CLI's demo stub teacher, and overwrote the adapter in place.
    """
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend

    suite = _real_backend_suite(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "has_api_key", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "_paw", lambda self: object())

    def _must_not_compile(self, *a, **kw):
        raise AssertionError("compile() reached with a real backend and no explicit opt-in")

    monkeypatch.setattr(ProgramAsWeightsBackend, "compile", _must_not_compile)
    monkeypatch.setattr(
        ProgramAsWeightsBackend, "infer", lambda self, *a, **kw: "2026-01-15"
    )

    result = runner.invoke(paw_test_app, ["check", str(suite), "--backend", "real"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" in out
    assert "overwrite" in out
    # The run still happens -- read-only, not skipped.
    assert "Pass rate:" in out
    assert "backend: ProgramAsWeightsBackend" in out


def test_check_real_backend_refuses_explicit_auto_recompile(tmp_path, monkeypatch):
    """Explicit --auto-recompile is refused, not honoured: the CLI teacher is a stub.

    The built-in `cli_teacher` answers "2026-01-01" to almost any input. Those labels
    must never become training signal for a paid compile that overwrites the adapter, so
    the explicit opt-in path errors and points the user at the code-level API instead.
    """
    from paw_kit.backend.programasweights import ProgramAsWeightsBackend

    suite = _real_backend_suite(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ProgramAsWeightsBackend, "is_available", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "has_api_key", lambda self: True)
    monkeypatch.setattr(ProgramAsWeightsBackend, "_paw", lambda self: object())

    def _must_not_compile(self, *a, **kw):
        raise AssertionError("compile() reached despite the stub-teacher refusal")

    monkeypatch.setattr(ProgramAsWeightsBackend, "compile", _must_not_compile)

    result = runner.invoke(
        paw_test_app, ["check", str(suite), "--backend", "real", "--auto-recompile"]
    )

    assert result.exit_code == 2
    assert "demo stub" in strip_ansi(result.output)


def test_check_recompile_announcement_is_per_iteration_not_whole_run_B_CLI_1(tmp_path, monkeypatch):
    """"Recompiling..." must announce only the iterations that actually recompiled.

    Before this fix the announcement was gated on `recompiles_performed > 0`, a
    whole-run aggregate: once any iteration recompiled, every later non-final
    iteration was announced as recompiling too, even one that skipped its own
    recompile because the teacher's label was unchanged (idempotent) from the one
    already compiled. Reproduced here: the CLI's own demo teacher answers
    "2026-01-01" for "today", which matches this suite's `expected`, so iteration 1
    genuinely recompiles -- and every later iteration re-offers the identical label,
    so none of them do (H-9). The backend keeps failing regardless (infer always
    returns "WRONG"), so the loop runs to `max_iterations` without ever succeeding.
    """
    from paw_kit.backend.mock import MockPAWBackend

    def _always_wrong(self, *a, **kw):
        return "WRONG"

    monkeypatch.setattr(MockPAWBackend, "infer", _always_wrong)

    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: bcli1\n"
        'spec: "Normalize a date."\n'
        'adapter_path: "fresh.paw"\n'
        "standard_cases:\n"
        '  - input: "today"\n'
        '    expected: "2026-01-01"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 3\n"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(paw_test_app, ["check", "suite.yaml", "--backend", "mock"])
    out = strip_ansi(result.output)

    iter1 = out.index("Iteration 1:")
    iter2 = out.index("Iteration 2:")
    iter3 = out.index("Iteration 3:")
    # Announced exactly once, between iterations 1 and 2 -- not before iteration 3,
    # and not a second time between 2 and 3.
    assert out.count("Recompiling adapter") == 1
    action_at = out.index("Recompiling adapter")
    assert iter1 < action_at < iter2 < iter3


def test_check_mock_backend_recompiles_freely_when_no_adapter_exists_yet(tmp_path, monkeypatch):
    """The first-compile case (no existing file at adapter_path) stays unaffected.

    Renamed from `test_check_mock_backend_still_recompiles_freely` (C-2): that test
    used `_real_backend_suite`, whose fixture adapter *already exists on disk*
    declaring `backend: "mock"` -- which after C-2's fix is exactly the case that
    must now be protected, not the case this test's docstring claimed to cover ("the
    mock costs nothing"). This version points at an adapter path that does not exist
    yet, which is the actual claim: recompiling into nothing is always allowed.
    """
    # "February 30" is the one input the CLI's demo teacher (`cli_teacher`) answers
    # "INVALID" for -- matching `expected` below, so the recompile is a legitimate
    # one and not rejected by H-8's poisoned-label guard (an input/expected pair the
    # stub teacher would actually disagree with never compiles at all).
    suite_dir = tmp_path
    adapter = suite_dir / "fresh.paw"
    suite = suite_dir / "suite.yaml"
    suite.write_text(
        "task_name: fresh\n"
        'spec: "Normalize a date."\n'
        f'adapter_path: "{adapter.name}"\n'
        "standard_cases:\n"
        '  - input: "February 30, 2026"\n'
        '    expected: "INVALID"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 3\n"
    )
    monkeypatch.chdir(tmp_path)
    assert not adapter.exists()

    result = runner.invoke(paw_test_app, ["check", str(suite), "--backend", "mock"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" not in out
    assert "Iteration 1:" in out
    assert adapter.exists()  # the first compile did happen


def test_check_surfaces_backend_execution_error(tmp_path, monkeypatch):
    """A backend that cannot run must say so, not read as failed assertions.

    TestRunner catches backend exceptions into "[EXECUTION_ERROR]" and stashes the message
    on TestCaseResult.execution_error. The CLI never printed that field, so a backend
    failing on every call presented as a 0.0% pass rate against the user's assertions.
    """
    from paw_kit.backend.mock import MockPAWBackend

    suite = _real_backend_suite(tmp_path)
    monkeypatch.chdir(tmp_path)

    def _boom(self, *a, **kw):
        raise RuntimeError("llama runtime unavailable")

    monkeypatch.setattr(MockPAWBackend, "infer", _boom)

    result = runner.invoke(
        paw_test_app, ["check", str(suite), "--backend", "mock", "--no-auto-recompile"]
    )
    out = strip_ansi(result.output)

    assert "backend error: llama runtime unavailable" in out
    assert "Pass rate: 0.0%" in out


def test_check_refuses_to_recompile_a_non_mock_adapter(tmp_path, monkeypatch):
    """The mock backend must not overwrite an adapter another backend produced.

    `MockPAWBackend.compile()` writes a real file (atomic_write_text) -- it is not the
    in-memory no-op the first pass at the --backend real guard assumed. So plain
    `paw-test check suite.yaml`, default backend, default auto_recompile=True, used to
    replace whatever `adapter_path` pointed at with a mock stub whose examples are
    cli_teacher's fabricated labels. Reproduced against a real programasweights manifest
    before this guard: it was destroyed wholesale, on the *default* invocation.
    """
    adapter = tmp_path / "real.paw"
    original = json.dumps(
        {"backend": "programasweights", "program_id": "prog_abc123", "compiler": "paw-4b"}
    )
    adapter.write_text(original)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: n1\n"
        'spec: "Normalize a date."\n'
        'adapter_path: "real.paw"\n'
        "standard_cases:\n"
        '  - input: "January 15, 2026"\n'
        '    expected: "2026-01-15"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 2\n"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(paw_test_app, ["check", "suite.yaml"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" in out
    # Rich's terminal-width wrapping can insert a line break between these two
    # words at test width, so compare with whitespace collapsed rather than the
    # exact substring (C-2's longer message pushed the wrap point earlier).
    assert "programasweights adapter" in " ".join(out.split())
    # The point of the test: the file on disk is untouched.
    assert adapter.read_text() == original


def test_check_refuses_to_recompile_an_existing_mock_adapter_C_2(tmp_path, monkeypatch):
    """A mock-declared adapter that already exists is a real user artifact too.

    Renamed from `test_check_still_recompiles_a_mock_adapter`, which asserted C-2's
    exact defect as correct behaviour: the pre-fix guard exempted anything whose
    manifest declared `backend == "mock"` from protection, but mock adapters are real
    user artifacts (`paw-kit demo`, `MockPAWBackend.compile()`, `schema.loader`, and
    `@compile_on_hit`'s cache all write them) -- not a safe default overwrite target.
    `"examples": []` in the fixture below is why the old assertion's loss was
    invisible: there was nothing in the file for a silent overwrite to be seen
    destroying. Mirrors `test_check_refuses_to_recompile_a_non_mock_adapter`'s shape;
    existence is now the only gate, not the declared backend.
    """
    adapter = tmp_path / "mock.paw"
    original = json.dumps({"backend": "mock", "spec": "s", "examples": ["not empty"]})
    adapter.write_text(original)
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: n1mock\n"
        'spec: "Normalize a date."\n'
        'adapter_path: "mock.paw"\n'
        "standard_cases:\n"
        '  - input: "January 15, 2026"\n'
        '    expected: "2026-01-15"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
        "active_learning:\n"
        "  auto_recompile: true\n"
        "  max_iterations: 2\n"
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(paw_test_app, ["check", "suite.yaml"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" in out
    assert "mock" in out
    # The point of the test: the file on disk is untouched.
    assert adapter.read_text() == original


def test_check_output_survives_rich_markup_in_paths_and_errors(tmp_path, monkeypatch):
    """Bracketed text in a path or backend error must not be eaten or crash the CLI.

    Rich parses square brackets as markup tags. An unescaped interpolation either
    silently deletes the bracketed span (`[date]` vanished from install advice, printing
    `pip install 'paw-kit'`) or, for a path-shaped tag, raises an uncaught MarkupError
    and takes the whole command down.
    """
    from paw_kit.backend.mock import MockPAWBackend

    adapter = tmp_path / "[v2]adapter.paw"
    adapter.write_text(json.dumps({"backend": "mock", "spec": "s", "examples": []}))
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "task_name: markup\n"
        'spec: "Normalize a date."\n'
        'adapter_path: "[v2]adapter.paw"\n'
        "standard_cases:\n"
        '  - input: "January 15, 2026"\n'
        '    expected: "2026-01-15"\n'
        "assertions:\n"
        "  - rule: max_length\n"
        "    value: 10\n"
    )
    monkeypatch.chdir(tmp_path)

    def _boom(self, *a, **kw):
        raise RuntimeError("cannot open [/usr/lib/libllama.so]")

    monkeypatch.setattr(MockPAWBackend, "infer", _boom)

    result = runner.invoke(
        paw_test_app, ["check", "suite.yaml", "--no-auto-recompile"]
    )
    out = strip_ansi(result.output)

    # No MarkupError escaped as a crash...
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # ...and neither bracketed span was silently swallowed. Newlines stripped
    # (not just collapsed to a space) before matching: M-2 now prints an absolute
    # path (suite_dir-resolved), and Rich's terminal-width wrapping can hard-wrap
    # -- no space -- inside the longer filename at test width.
    joined = out.replace("\n", "")
    assert "[v2]adapter.paw" in joined
    assert "[/usr/lib/libllama.so]" in joined


def test_inspect_and_history_table_titles_survive_rich_markup_in_filename_C_6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bracketed adapter filename must appear in the table title, not be eaten.

    Rich parses square brackets as markup. `Table(title=...)` was never visited by
    `test_no_unescaped_console_interpolations` (it only ever walked
    `console.print(...)` calls), so an adapter named `[v2]model.paw` printed a table
    titled "PAW Adapter: model.paw" -- silently dropping the bracketed span and
    naming a *different* file than the one on disk.
    """
    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "[v2]model.paw"
    adapter.write_text(json.dumps({"backend": "mock", "spec": "s", "examples": []}), encoding="utf-8")

    inspect_result = runner.invoke(app, ["inspect", str(adapter)])
    assert "[v2]model.paw" in strip_ansi(inspect_result.output)

    history_path = tmp_path / "[v2]model.paw.history.jsonl"
    history_path.write_text(json.dumps({"compiled_at": "now"}) + "\n", encoding="utf-8")
    history_result = runner.invoke(app, ["history", str(adapter)])
    assert "[v2]model.paw" in strip_ansi(history_result.output)


def test_no_unescaped_console_interpolations():
    """Every string interpolated into a Rich console.print must go through `_e()`.

    Rich parses square brackets as markup. An unescaped interpolation either silently
    deletes the bracketed span or raises MarkupError and kills the command -- both were
    live bugs here (an install hint printed `pip install 'paw-kit'` because Rich ate
    "[real]"; an adapter named `[v2]model.paw` crashed `paw-inspect` and made another
    command report a different filename than the one it wrote). Three review rounds each
    found more instances of this same class, so it is enforced mechanically rather than
    by eye.

    To add an entry to the allowlist below, the value must be provably never a string:
    an int, a float, or a literal. A `Path`, a name, an exception, or anything derived
    from user input, the filesystem, or model output does not qualify -- wrap it in `_e()`.
    """
    import ast
    from pathlib import Path as _Path

    numeric_or_literal_allowlist = {
        "i", "ms", "port", "result.total_redacted",
        "report.pass_rate", "report.passed_cases", "report.total_cases",
        "rep.pass_rate", "rep.passed_cases", "rep.total_cases",
        "al_report.iterations_run",
        "len(files_to_remove)", "len(rows)", "len(dirs_skipped)", "len(failed)", "file_word",
        "'Dry run: would remove' if dry_run else 'Purging'",
    }

    source = _Path(__file__).parent.parent.joinpath("paw_kit", "cli.py").read_text()
    offenders = []
    for node in ast.walk(ast.parse(source)):
        is_console_print = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "print"
            and isinstance(node.func.value, ast.Name)
            # C-5 added a second Console instance (`stderr_console`) for the
            # real-to-mock fallback warnings; same markup-parsing risk, same guard.
            and node.func.value.id in ("console", "stderr_console")
        )
        # C-6: `Table(title=...)`/`Panel(title=...)` render Rich markup exactly like
        # `console.print`, but were never visited by this guard -- an adapter named
        # `[v2]model.paw` printed a table titled "PAW Adapter: model.paw", naming a
        # *different* file than the one on disk. `add_row`/`add_column` were audited
        # by hand instead of mechanically: every real (non-demo) call site already
        # escapes every dynamic argument, and the one exception
        # (`_run_triage_demo`'s `mode` variable) is a literal `"[green]...[/green]"`/
        # `"[yellow]...[/yellow]"` string constant meant to *be* markup, from a
        # hardcoded demo ticket list -- never attacker- or environment-controlled --
        # so mechanizing that check would need an allowlist entry for the one case
        # that must stay unescaped, for no live-defect coverage gained.
        is_table_or_panel_title = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("Table", "Panel")
            and any(kw.arg == "title" for kw in node.keywords)
        )
        if not (is_console_print or is_table_or_panel_title):
            continue
        walk_targets = (
            [node]
            if is_console_print
            else [kw.value for kw in node.keywords if kw.arg == "title"]
        )
        for target in walk_targets:
            for inner in ast.walk(target):
                if not isinstance(inner, ast.FormattedValue):
                    continue
                expr = inner.value
                escaped = (
                    isinstance(expr, ast.Call)
                    and isinstance(expr.func, ast.Name)
                    and expr.func.id in ("_e", "escape")
                )
                src = ast.unparse(expr)
                if not escaped and src not in numeric_or_literal_allowlist:
                    offenders.append(f"cli.py:{node.lineno}: {src}")

    assert not offenders, (
        "Unescaped interpolation into Rich markup (wrap in _e(), or add to the "
        "allowlist only if provably never a string):\n  " + "\n  ".join(offenders)
    )


def test_no_rich_escaping_before_plain_click_output():
    """The inverse mistake (C-9): `_e()`/`escape()` before a plain-click sink.

    `typer.confirm`/`typer.prompt`/`typer.echo`/`typer.BadParameter` are click, not
    Rich -- they print raw. Running a value through Rich's markup escaper first
    *adds* visible backslashes instead of protecting anything: a file named
    `[v2]out.jsonl` prompted `\\[v2]out.jsonl already exists. Overwrite?` on the one
    prompt whose whole job is to name the file about to be destroyed.
    """
    import ast
    from pathlib import Path as _Path

    source = _Path(__file__).parent.parent.joinpath("paw_kit", "cli.py").read_text()
    offenders = []
    for node in ast.walk(ast.parse(source)):
        is_plain_click_sink = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "typer"
            and node.func.attr in ("confirm", "prompt", "echo", "BadParameter")
        )
        if not is_plain_click_sink:
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id in ("_e", "escape")
            ):
                offenders.append(f"cli.py:{node.lineno}: {ast.unparse(inner)}")

    assert not offenders, (
        "Rich-escaped value passed to a plain-click sink (typer.confirm/prompt/echo/"
        "BadParameter never parses markup, so escaping it only adds visible "
        "backslashes):\n  " + "\n  ".join(offenders)
    )


def test_cli_lint_spec_text_argument_warns_and_exits_zero():
    result = runner.invoke(
        app, ["lint-spec", "Pull out the phone number and format it consistently."]
    )
    out = strip_ansi(result.output)
    assert result.exit_code == 0
    assert "output-format-unpinned" in out


def test_cli_lint_spec_error_severity_exits_one():
    result = runner.invoke(app, ["lint-spec", "short"])
    assert result.exit_code == 1
    assert "spec-too-short" in strip_ansi(result.output)


def test_cli_lint_spec_reads_from_file(tmp_path: Path):
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("Translate this sentence into French.", encoding="utf-8")
    result = runner.invoke(app, ["lint-spec", "--file", str(spec_file)])
    assert result.exit_code == 0


def test_cli_lint_spec_file_rejects_non_utf8_cleanly_C_8(tmp_path: Path):
    """A non-UTF-8 spec file must exit 1 with a message, not an uncaught traceback."""
    spec_file = tmp_path / "spec.txt"
    spec_file.write_bytes(b"\xff\xfe not valid utf-8")
    result = runner.invoke(app, ["lint-spec", "--file", str(spec_file)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "not valid UTF-8" in plain(result.output)


def test_cli_lint_spec_examples_file_rejects_non_utf8_cleanly_C_8(tmp_path: Path):
    """A non-UTF-8 examples file must exit 1 with a message, not an uncaught traceback."""
    examples_file = tmp_path / "examples.jsonl"
    examples_file.write_bytes(b"\xff\xfe not valid utf-8\n")
    result = runner.invoke(app, ["lint-spec", "do it", "--examples", str(examples_file)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "not valid UTF-8" in plain(result.output)


def test_cli_lint_spec_json_output(tmp_path: Path):
    result = runner.invoke(app, ["lint-spec", "short", "--json"])
    assert result.exit_code == 1
    findings = json.loads(strip_ansi(result.output))
    assert findings[0]["rule_id"] == "spec-too-short"
    assert findings[0]["severity"] == "error"


def test_cli_lint_spec_examples_file_flags_single_form(tmp_path: Path):
    examples_file = tmp_path / "examples.jsonl"
    examples_file.write_text(
        '{"input": "a", "output": "(555) 123-4567"}\n'
        '{"input": "b", "output": "(555) 999-0000"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "lint-spec",
            "Extract the phone number from this text.",
            "--examples",
            str(examples_file),
        ],
    )
    assert result.exit_code == 0
    assert "examples-single-form" in strip_ansi(result.output)


def test_cli_lint_spec_no_spec_or_file_errors():
    result = runner.invoke(app, ["lint-spec"])
    assert result.exit_code == 1


def test_public_api_is_importable_and_excludes_deleted_backends():
    """Every name `paw_kit` advertises must resolve, and RealPAWBackend must not be one.

    Track 13 deleted `RealPAWBackend`, the in-process PyTorch/PEFT placeholder whose
    compile() and infer() raised NotImplementedError. paw-kit wraps the upstream SDK; it
    does not reimplement upstream's runtime, so a real backend is either
    `ProgramAsWeightsBackend` or a caller's own `AbstractPAWBackend` subclass (see
    conductor decisions.md section 3). Reintroducing an in-process runtime needs a track,
    not a patch -- this test makes that a deliberate act rather than an accidental one.

    Asserting `__all__` resolves in full also catches the more common failure: deleting a
    module and leaving its name advertised, so `from paw_kit import *` breaks at runtime
    while every ordinary import still passes.
    """
    import paw_kit
    import paw_kit.backend

    for name in paw_kit.__all__:
        assert hasattr(paw_kit, name), f"paw_kit.__all__ advertises {name!r} but it does not resolve"
    for name in paw_kit.backend.__all__:
        assert hasattr(paw_kit.backend, name), f"paw_kit.backend.__all__ advertises {name!r}"

    assert "RealPAWBackend" not in paw_kit.__all__
    assert "RealPAWBackend" not in paw_kit.backend.__all__
    assert not hasattr(paw_kit, "RealPAWBackend")


# --- Track 14: `paw-kit report` ------------------------------------------------


def _seed_shadow_db(db_file: Path) -> str:
    """Build a trace database holding one promoted task with a recorded disagreement."""
    from paw_kit.jit.db import TraceDB

    db = TraceDB(str(db_file))
    task_id = "a" * 64
    db.sync_shadow_config(
        task_id,
        {"shadow_window": 2, "shadow_threshold": 0.8, "audit_window": 2, "demote_threshold": 0.6},
    )
    db.record_trace(task_id, "hello", "teacher:hello", 1.0)
    db.record_trace(task_id, "world", "teacher:world", 1.0)
    db.set_shadow_started(task_id, str(db_file.parent / "adapter.paw"))
    epoch = db.get_task_routing(task_id)[2]
    db.record_shadow_pair(
        task_id, epoch, "shadow", "hello", "teacher:hello", "teacher:hello", "agree"
    )
    db.record_shadow_pair(
        task_id, epoch, "shadow", "world", "teacher:world", "[v2] wrong", "disagree"
    )
    db.try_promote(task_id, epoch, 0.5, 2)
    db.increment_fail_open(task_id)
    db.close()
    return task_id


def test_cli_report_missing_db_exits_one(tmp_path: Path) -> None:
    """A missing trace database is a clear error, exit 1 -- mirroring `export dataset`."""
    result = runner.invoke(app, ["report", "--db", str(tmp_path / "nope.db")])
    assert result.exit_code == 1
    # Rich wraps the path, so normalise the line breaks before matching.
    assert "not exist" in " ".join(result.stdout.split())


def test_cli_report_empty_db_exits_zero(tmp_path: Path) -> None:
    """An empty database is not an error."""
    from paw_kit.jit.db import TraceDB

    db_file = tmp_path / "empty" / "traces.db"
    TraceDB(str(db_file)).close()
    result = runner.invoke(app, ["report", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "No tasks recorded" in result.stdout


def test_cli_report_renders_state_agreement_and_fail_open(tmp_path: Path) -> None:
    """The table shows per-task state, calls, agreement and the persisted fail-open count."""
    db_file = tmp_path / "report" / "traces.db"
    task_id = _seed_shadow_db(db_file)

    result = runner.invoke(app, ["report", "--db", str(db_file)])
    assert result.exit_code == 0, result.stdout
    assert task_id[:12] in result.stdout
    assert "ready" in result.stdout
    # The disagreement panel renders the adapter's bracketed output without Rich
    # eating it or raising MarkupError.
    assert "disagree" in result.stdout
    assert "[v2] wrong" in result.stdout


def test_cli_report_shows_stalled_marker_for_a_task_past_the_stall_point(tmp_path: Path) -> None:
    """Finding 2: a `shadow` task the runner has stopped evaluating for promotion at
    this epoch shows a `stalled` marker next to its state -- otherwise indistinguishable
    in the report from one still converging."""
    from paw_kit.jit.db import TraceDB
    from paw_kit.jit.shadow import _SHADOW_STALL_FACTOR

    db_file = tmp_path / "stalled" / "traces.db"
    db = TraceDB(str(db_file))
    task_id = "b" * 64
    window = 2
    db.sync_shadow_config(
        task_id,
        {"shadow_window": window, "shadow_threshold": 0.8, "audit_window": 2, "demote_threshold": 0.6},
    )
    db.record_trace(task_id, "hello", "teacher:hello", 1.0)
    db.set_shadow_started(task_id, str(db_file.parent / "adapter.paw"))
    epoch = db.get_task_routing(task_id)[2]
    for n in range(_SHADOW_STALL_FACTOR * window + 1):
        db.record_shadow_pair(
            task_id, epoch, "shadow", f"in{n}", f"teacher:{n}", "wrong", "disagree"
        )
    db.close()

    result = runner.invoke(app, ["report", "--db", str(db_file)])
    assert result.exit_code == 0, result.stdout
    output = strip_ansi(result.stdout)
    assert task_id[:12] in output
    # Substring, not the full word: Rich can truncate a narrow "State" column to
    # "(stalle…" in the CliRunner's default terminal width.
    assert "stall" in output

    json_result = runner.invoke(app, ["report", "--db", str(db_file), "--json"])
    assert json_result.exit_code == 0, json_result.stdout
    payload = json.loads(json_result.stdout)
    assert payload["tasks"][0]["agreement"]["stalled"] is True


def test_cli_report_json_output_matches_get_task_report(tmp_path: Path) -> None:
    """--json emits exactly what TraceDB.get_task_report reports, plus disagreements."""
    from paw_kit.jit.db import TraceDB

    db_file = tmp_path / "reportjson" / "traces.db"
    task_id = _seed_shadow_db(db_file)

    result = runner.invoke(app, ["report", "--db", str(db_file), "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [entry["task_id"] for entry in payload["tasks"]] == [task_id]

    db = TraceDB(str(db_file))
    expected = db.get_task_report(task_id)
    db.close()
    entry = payload["tasks"][0]
    for key, value in expected.items():
        assert entry[key] == value
    assert len(entry["last_disagreements"]) == 1


def test_cli_report_migrates_v1_db_in_place(tmp_path: Path) -> None:
    """`report` opens a real TraceDB, so it migrates a pre-v2 database. Documented, not a bug."""
    import sqlite3

    db_file = tmp_path / "v1" / "traces.db"
    db_file.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_file))
    conn.executescript(
        """
        CREATE TABLE tasks (
            task_id TEXT PRIMARY KEY, call_count INTEGER DEFAULT 0, adapter_path TEXT,
            status TEXT DEFAULT 'tracing', compile_attempts INTEGER DEFAULT 0,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE traces (
            id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
            input_payload TEXT NOT NULL, teacher_output TEXT NOT NULL,
            latency_ms REAL NOT NULL, timestamp TEXT NOT NULL);
        INSERT INTO tasks VALUES ('b0', 7, NULL, 'tracing', 0, '2026-01-01', '2026-01-01');
        """
    )
    conn.commit()
    conn.close()

    result = runner.invoke(app, ["report", "--db", str(db_file)])
    assert result.exit_code == 0, result.stdout
    assert "tracing" in result.stdout

    conn = sqlite3.connect(str(db_file))
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table';")}
    assert {"shadow_pairs", "state_transitions"} <= tables
    # J-5 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    # this asserted the literal `2`. J-5 adds a `tasks.compiling_started_at` column, so
    # `_SCHEMA_VERSION` moves 2 -> 3 and the literal had to change with it. Pinned
    # against the module constant rather than a new literal: the subject here is "the
    # forward marker was stamped by this migration", not its numeric value, and the
    # value itself is pinned separately (with an explanation of what a bump means) by
    # `test_migration_adds_the_lease_column_and_bumps_the_marker_J_5` in
    # tests/test_jit_persistence.py.
    from paw_kit.jit.db import _SCHEMA_VERSION

    assert conn.execute("PRAGMA user_version;").fetchone()[0] == _SCHEMA_VERSION
    conn.close()


def test_cli_report_task_filter_selects_one_task(tmp_path: Path) -> None:
    """--task narrows the report to a single task_id."""
    from paw_kit.jit.db import TraceDB

    db_file = tmp_path / "filter" / "traces.db"
    task_id = _seed_shadow_db(db_file)
    db = TraceDB(str(db_file))
    db.record_trace("c" * 64, "other", "teacher:other", 1.0)
    db.close()

    result = runner.invoke(app, ["report", "--db", str(db_file), "--task", task_id, "--json"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert [entry["task_id"] for entry in payload["tasks"]] == [task_id]


# =====================================================================================
# Report section 6, H-1 (bug-hunt-remediation, Track B, Phase B2)
# =====================================================================================


def test_h1_check_exits_one_and_prints_the_error_when_the_backend_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-1 at the CLI: a backend raising on every case printed `[PASS]` for each one,
    `Pass rate: 100.0%`, and exited **0** -- and never printed the error text, because
    `res.execution_error` was only shown in the FAIL branch.

    Not reachable from the golden-snapshot suite: `MockPAWBackend.infer` never raises
    (a missing adapter returns its `[mock:...]` fallback string), and `--backend real`
    is out of scope there by construction. So the raising backend is injected here.
    """
    import paw_kit.cli as cli_module
    from paw_kit.backend.mock import MockPAWBackend

    class _RaisingBackend(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str) -> str:  # type: ignore[override]
            raise RuntimeError("model file is corrupt")

    monkeypatch.chdir(tmp_path)
    adapter = tmp_path / "broken.paw"
    MockPAWBackend().compile(spec="s", examples=[{"input": "today", "output": "x"}],
                             output_path=str(adapter))
    # Deliberately no `not_contains: ERROR` rule: every published suite carries one,
    # and "[EXECUTION_ERROR]" contains "ERROR", which is the *only* thing that was
    # protecting the committed runs from this finding.
    (tmp_path / "suite.yaml").write_text(
        'task_name: h1\nspec: s\nadapter_path: "broken.paw"\n'
        'standard_cases:\n  - input: "today"\n    expected: "2026-09-11"\n'
        "assertions:\n  - rule: min_length\n    value: 1\n"
        "active_learning:\n  auto_recompile: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_module, "_resolve_cli_backend", lambda _bt: _RaisingBackend())

    result = runner.invoke(paw_test_app, ["check", "suite.yaml"])
    out = strip_ansi(result.stdout)

    assert result.exit_code == 1, out
    assert "Pass rate: 0.0%" in out
    assert "Errored: 1/1" in out
    assert "[PASS]" not in out
    # The error text is printed regardless of which branch the case took.
    assert "model file is corrupt" in out
    # And the answer-key line does not claim a verdict it never got.
    assert "Correct against expected: 0/0" in out


def test_h1_check_refuses_a_missing_adapter_on_the_read_only_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """H-1's adapter-existence gate, matching `compare`'s. Scoped to the path where
    recompilation is already off -- whether `check` may *create* an adapter is M-1, an
    open policy decision that belongs to Track H's `cli.py` guard."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "suite.yaml").write_text(
        'task_name: h1g\nspec: s\nadapter_path: "absent.paw"\n'
        'standard_cases:\n  - input: "today"\n'
        "active_learning:\n  auto_recompile: false\n",
        encoding="utf-8",
    )
    result = runner.invoke(paw_test_app, ["check", "suite.yaml"])
    out = plain(result.stdout)
    assert result.exit_code == 1, out
    assert "does not exist" in out
    # The normal first-compile case is untouched: with auto-recompile on, an absent
    # adapter is still allowed through.
    (tmp_path / "suite2.yaml").write_text(
        'task_name: h1g\nspec: s\nadapter_path: "absent.paw"\n'
        'standard_cases:\n  - input: "today"\n'
        "active_learning:\n  auto_recompile: true\n",
        encoding="utf-8",
    )
    result2 = runner.invoke(paw_test_app, ["check", "suite2.yaml"])
    assert "does not exist, and recompilation is off" not in strip_ansi(result2.stdout)


# =====================================================================================
# Report section 6, H-16 (bug-hunt-remediation, Track B, Phase B6)
# =====================================================================================


def test_h16_lint_spec_errors_when_examples_file_yields_nothing(tmp_path: Path) -> None:
    """H-16: unparseable `--examples` lines were skipped individually, so a JSONL file
    written as a JSON array yielded zero examples and `lint-spec` printed
    "No issues found." at exit 0 -- having silently run one fewer rule than asked for."""
    examples = tmp_path / "examples.jsonl"
    # The exact mistake: a JSON array instead of one object per line. Every line is
    # unparseable as a standalone object -- the array brackets and the trailing commas.
    examples.write_text(
        '[\n  {"input": "a", "output": "(555) 123-4567"},\n'
        '  {"input": "b", "output": "(555) 765-4321"},\n]\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["lint-spec", "Extract the phone number.", "--examples", str(examples)],
    )
    out = strip_ansi(result.output)

    assert result.exit_code == 1, out
    assert "No issues found" not in out
    assert "yielded no usable examples" in " ".join(out.split())


def test_h16_partially_usable_examples_file_warns(tmp_path: Path) -> None:
    """The partial case, same shape as H-16: a *pretty-printed* JSON array has exactly
    one line that parses (the last element, which carries no trailing comma), so it
    slips past the zero-usable check with 1 of 4 examples -- and rule 5, which needs
    two, quietly does not run."""
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '[\n  {"input": "a", "output": "(555) 123-4567"},\n'
        '  {"input": "b", "output": "(555) 765-4321"}\n]\n',
        encoding="utf-8",
    )

    result = runner.invoke(
        app, ["lint-spec", "Extract the phone number.", "--examples", str(examples)]
    )
    collapsed = " ".join(strip_ansi(result.output).split())

    assert "3 of 4 non-blank line(s)" in collapsed
    assert "were not usable JSON objects and were skipped" in collapsed


def test_h16_single_line_json_array_also_errors(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    examples.write_text('[{"input": "a", "output": "b"}]\n', encoding="utf-8")
    result = runner.invoke(
        app, ["lint-spec", "Extract the phone number.", "--examples", str(examples)]
    )
    assert result.exit_code == 1
    assert "yielded no usable examples" in " ".join(strip_ansi(result.output).split())


def test_h16_valid_jsonl_is_unaffected(tmp_path: Path) -> None:
    """The guard must not fire on a file that does work."""
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        '{"input": "a", "output": "(555) 123-4567"}\n'
        '{"input": "b", "output": "(555) 765-4321"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["lint-spec", "Extract the phone number.", "--examples", str(examples)]
    )
    assert "yielded no usable examples" not in strip_ansi(result.output)


def test_h16_an_empty_examples_file_is_not_an_error(tmp_path: Path) -> None:
    """An empty file supplies no examples and claims none -- distinct from a file full
    of content that produced nothing, which is the finding."""
    examples = tmp_path / "examples.jsonl"
    examples.write_text("\n\n", encoding="utf-8")
    result = runner.invoke(
        app, ["lint-spec", "Extract the phone number.", "--examples", str(examples)]
    )
    assert "yielded no usable examples" not in strip_ansi(result.output)
