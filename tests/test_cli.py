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


def test_cli_check_passing_suite(tmp_path: Path) -> None:
    """Verify check exits with code 0 on passing test suite."""
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


def test_cli_clean(tmp_path: Path) -> None:
    """Verify clean command handles missing dirs, dry-run, and actual purging."""
    cache_dir = tmp_path / "test_cache"

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

    # Real clean
    res_real = runner.invoke(app, ["clean", "--cache-dir", str(cache_dir)])
    assert res_real.exit_code == 0
    assert "Cache cleaned successfully" in res_real.output
    assert not f1.exists()


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

