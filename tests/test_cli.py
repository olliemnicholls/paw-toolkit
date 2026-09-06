"""Unit and integration tests for Typer developer CLI commands."""

import json
from pathlib import Path
from typer.testing import CliRunner
import pytest

from paw_kit.cli import app, inspect, clean, test_app as paw_test_app

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


def test_cli_check_no_auto_recompile_failure(tmp_path: Path) -> None:
    """Verify check exits with code 1 when assertions fail and auto_recompile is disabled."""
    adapter_path = tmp_path / "model_fail.paw"
    adapter_path.write_text(
        json.dumps({
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


def test_cli_check_allows_adapter_path_outside_cwd_without_recompile_PAW_CLI_02(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify an out-of-cwd adapter_path is fine when auto_recompile can't trigger a write."""
    workdir = tmp_path / "work"
    workdir.mkdir()
    outside_adapter = tmp_path / "outside_model.paw"
    outside_adapter.write_text(
        json.dumps({
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
    assert result.exit_code == 0
    assert "Pass rate: 100.0%" in result.output


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
    assert "LOCAL 0.6B" in result.output


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

