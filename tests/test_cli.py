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
    assert "does not exist" in result.output


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
    assert "All assertions passed" in result.output


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
    assert "--adapter" in result.output
    assert "adapter_path" in result.output


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


def test_cli_history_missing_log_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", str(tmp_path / "nope.paw")])
    assert result.exit_code == 1
    assert "no history log" in strip_ansi(result.output)


def test_cli_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify clean command handles missing dirs, dry-run, confirmation, and actual purging."""
    # PAW-CLI-01: cache_dir must resolve under cwd, so exercise this from a cwd chdir'd
    # into tmp_path, using a cache_dir that is a genuine subdirectory of it.
    monkeypatch.chdir(tmp_path)
    cache_dir = Path("test_cache")

    # Non-existent
    res_empty = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir)])
    assert res_empty.exit_code == 0
    assert "does not exist" in res_empty.output

    cache_dir.mkdir()
    f1 = cache_dir / "trace.db"
    f1.write_text("trace", encoding="utf-8")

    # Dry-run
    res_dry = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir), "--dry-run"])
    assert res_dry.exit_code == 0
    assert "Dry run" in res_dry.output
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
    out = strip_ansi(capsys.readouterr().out)
    assert "not a model" in out
    assert "programasweights" in out


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
    out = strip_ansi(capsys.readouterr().out)
    assert "could not be loaded" in out
    assert "libllama.so" in out
    assert "not a model" in out


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


def test_check_mock_backend_still_recompiles_freely(tmp_path, monkeypatch):
    """The guard is scoped to real backends: the mock costs nothing and is unaffected."""
    suite = _real_backend_suite(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(paw_test_app, ["check", str(suite), "--backend", "mock"])
    out = strip_ansi(result.output)

    assert "auto-recompile is disabled" not in out
    assert "Iteration 1:" in out


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
    assert "programasweights adapter" in out
    # The point of the test: the file on disk is untouched.
    assert adapter.read_text() == original


def test_check_still_recompiles_a_mock_adapter(tmp_path, monkeypatch):
    """The N1 guard is scoped to foreign adapters: a mock adapter recompiles as before."""
    adapter = tmp_path / "mock.paw"
    adapter.write_text(json.dumps({"backend": "mock", "spec": "s", "examples": []}))
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

    assert "auto-recompile is disabled" not in out
    assert "Iteration 1:" in out


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
    # ...and neither bracketed span was silently swallowed.
    assert "[v2]adapter.paw" in out
    assert "[/usr/lib/libllama.so]" in out


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
        "len(files_to_remove)", "len(rows)",
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
            and node.func.value.id == "console"
        )
        if not is_console_print:
            continue
        for inner in ast.walk(node):
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
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == 2
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
