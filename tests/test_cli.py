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

