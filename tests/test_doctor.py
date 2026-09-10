"""Tests for `paw_kit.doctor` and the `paw-kit doctor` CLI command."""

from dataclasses import asdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List

import httpx
import pytest
from typer.testing import CliRunner

from paw_kit import doctor
from paw_kit.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------- check_sdk_importable


def test_sdk_importable_pass() -> None:
    result = doctor.check_sdk_importable()
    assert result.status == "PASS"
    assert "version" in result.detail


def test_sdk_not_importable_is_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "programasweights", None)
    result = doctor.check_sdk_importable()
    assert result.status == "FAIL"
    assert "not importable" in result.detail
    assert result.remedy


# ---------------------------------------------------------------- check_llama_cpp


def test_llama_cpp_not_importable_is_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    result = doctor.check_llama_cpp()
    assert result.status == "FAIL"
    assert "not importable" in result.detail


def test_llama_cpp_gpu_offload_true_is_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import llama_cpp

    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: True)
    result = doctor.check_llama_cpp()
    assert result.status == "PASS"
    assert "GPU offload supported" in result.detail


def test_llama_cpp_gpu_offload_false_is_warn_not_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import llama_cpp

    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", lambda: False)
    result = doctor.check_llama_cpp()
    assert result.status == "WARN"
    assert "CPU-only" in result.detail
    assert "90x" in result.remedy
    assert "measurements/README.md" in result.remedy


def test_llama_cpp_gpu_offload_raises_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    import llama_cpp

    def _boom():
        raise RuntimeError("native crash")

    monkeypatch.setattr(llama_cpp, "llama_supports_gpu_offload", _boom)
    result = doctor.check_llama_cpp()
    assert result.status == "WARN"
    assert "native crash" in result.detail


# ---------------------------------------------------------------- check_gpu_visible


class _FakeCompletedProcess:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout


def test_gpu_visible_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        return _FakeCompletedProcess(0, "NVIDIA GeForce RTX 3080, 10240 MiB\n")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = doctor.check_gpu_visible()
    assert result.status == "PASS"
    assert "RTX 3080" in result.detail


def test_gpu_not_found_is_warn_never_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = doctor.check_gpu_visible()
    assert result.status == "WARN"
    assert result.status != "FAIL"


def test_gpu_timeout_is_warn_never_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess as real_subprocess

    def fake_run(*args: Any, **kwargs: Any):
        raise real_subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5.0)

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = doctor.check_gpu_visible()
    assert result.status == "WARN"


def test_gpu_nonzero_exit_no_output_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        return _FakeCompletedProcess(1, "")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)
    result = doctor.check_gpu_visible()
    assert result.status == "WARN"


# ---------------------------------------------------------------- check_api_key


def test_api_key_set_is_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAW_API_KEY", "paw_sk_super_secret_value")
    result = doctor.check_api_key()
    assert result.status == "PASS"
    assert result.detail == "set"
    assert "paw_sk_super_secret_value" not in result.detail
    assert "paw_sk_super_secret_value" not in result.remedy


def test_api_key_unset_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    result = doctor.check_api_key()
    assert result.status == "WARN"
    assert result.detail == "not set"


# ---------------------------------------------------------------- check_service_health


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_health_pass_when_gpu_services_nonempty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/health"
        return httpx.Response(200, json={"status": "ok", "gpu_services": {"worker-1": "up"}})

    result = doctor.check_service_health(api_url="https://fake.example", transport=_transport(handler))
    assert result.status == "PASS"
    assert "1 gpu service" in result.detail


def test_health_warn_when_gpu_services_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "gpu_services": {}})

    result = doctor.check_service_health(api_url="https://fake.example", transport=_transport(handler))
    assert result.status == "WARN"
    assert "issue #5" in result.remedy


def test_health_fail_on_non_200() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    result = doctor.check_service_health(api_url="https://fake.example", transport=_transport(handler))
    assert result.status == "FAIL"
    assert "503" in result.detail


def test_health_fail_on_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    result = doctor.check_service_health(api_url="https://fake.example", transport=_transport(handler))
    assert result.status == "FAIL"
    assert "unreachable" in result.detail


def test_health_fail_on_missing_gpu_services_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    result = doctor.check_service_health(api_url="https://fake.example", transport=_transport(handler))
    assert result.status == "WARN"


# ---------------------------------------------------------------- check_base_model_cached


def test_base_model_cached_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from programasweights import cache as paw_cache

    fake_path = tmp_path / "qwen3-0.6b-q6_k.gguf"
    fake_path.write_bytes(b"x" * (1024 * 1024))

    monkeypatch.setattr(paw_cache, "get_base_runtime_manifest", lambda interpreter: {"fake": True})
    monkeypatch.setattr(paw_cache, "get_cached_base_model_path", lambda manifest: fake_path)

    result = doctor.check_base_model_cached()
    assert result.status == "PASS"
    assert "1 MB" in result.detail


def test_base_model_not_cached_is_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    from programasweights import cache as paw_cache

    monkeypatch.setattr(paw_cache, "get_base_runtime_manifest", lambda interpreter: {"fake": True})
    monkeypatch.setattr(paw_cache, "get_cached_base_model_path", lambda manifest: None)

    result = doctor.check_base_model_cached()
    assert result.status == "WARN"
    assert "download ~600 MB" in result.remedy


# ---------------------------------------------------------------- check_cached_programs


def test_cached_programs_count_only(monkeypatch: pytest.MonkeyPatch) -> None:
    import programasweights as paw

    monkeypatch.setattr(paw, "list_cached_programs", lambda: [{"program_id": "a"}, {"program_id": "b"}])
    result = doctor.check_cached_programs(adapter_path=None)
    assert result.status == "PASS"
    assert "2 cached program" in result.detail


def test_cached_programs_with_adapter_offline_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import programasweights as paw

    manifest_path = tmp_path / "a.paw"
    manifest_path.write_text(
        json.dumps({"backend": "programasweights", "program_id": "prog-1", "slug": None})
    )

    monkeypatch.setattr(paw, "list_cached_programs", lambda: [])
    monkeypatch.setattr(paw, "is_offline_ready", lambda program_id: True)

    result = doctor.check_cached_programs(adapter_path=str(manifest_path))
    assert result.status == "PASS"
    assert "offline_ready=True" in result.detail


def test_cached_programs_with_adapter_not_offline_ready_is_warn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import programasweights as paw

    manifest_path = tmp_path / "a.paw"
    manifest_path.write_text(
        json.dumps({"backend": "programasweights", "program_id": "prog-1", "slug": None})
    )

    monkeypatch.setattr(paw, "list_cached_programs", lambda: [])
    monkeypatch.setattr(paw, "is_offline_ready", lambda program_id: False)

    result = doctor.check_cached_programs(adapter_path=str(manifest_path))
    assert result.status == "WARN"
    assert "offline_ready=False" in result.detail


def test_cached_programs_with_mock_adapter_is_warn_not_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--adapter` pointed at a mock-backend manifest (what the default backend and
    `paw-kit demo` write) must WARN, not FAIL: `ProgramAsWeightsBackend.read_manifest`
    raises `ValueError` for any manifest whose `backend` isn't "programasweights", and
    offline readiness simply does not apply to a mock adapter -- it is not a broken
    environment."""
    import programasweights as paw
    from paw_kit.backend.mock import MockPAWBackend

    manifest_path = tmp_path / "mock.paw"
    MockPAWBackend().compile(spec="demo", examples=[], output_path=str(manifest_path))
    assert json.loads(manifest_path.read_text())["backend"] == "mock"

    monkeypatch.setattr(paw, "list_cached_programs", lambda: [])

    result = doctor.check_cached_programs(adapter_path=str(manifest_path))
    assert result.status == "WARN"
    assert result.status != "FAIL"
    assert result.detail == "offline readiness does not apply to a mock-backend manifest"


# ---------------------------------------------------------------- check_rate_limit_note


def test_rate_limit_note_always_pass() -> None:
    result = doctor.check_rate_limit_note()
    assert result.status == "PASS"
    assert "20 compiles/hour" in result.detail
    assert "60 compiles/hour" in result.detail


# ---------------------------------------------------------------- guarded exceptions


def test_guarded_check_exception_becomes_fail_not_traceback() -> None:
    @doctor._guarded("Exploding check")
    def _boom() -> doctor.CheckResult:
        raise ValueError("kaboom")

    result = _boom()
    assert result.status == "FAIL"
    assert result.name == "Exploding check"
    assert "kaboom" in result.detail


# ---------------------------------------------------------------- run_checks


@pytest.fixture
def fast_cached_programs(monkeypatch: pytest.MonkeyPatch) -> None:
    """`list_cached_programs()` re-validates every cached program's base-model checksum
    on each call, which is real (if slow, ~1s/program) SDK behaviour, not something
    under test here -- stub it so `run_checks()` tests stay fast regardless of how many
    programs happen to be cached on the machine running the suite."""
    import programasweights as paw

    monkeypatch.setattr(paw, "list_cached_programs", lambda: [])


def test_run_checks_offline_skips_network_check(fast_cached_programs: None) -> None:
    results = doctor.run_checks(offline=True)
    health = next(r for r in results if r.name == "Upstream service health")
    assert health.status == "WARN"
    assert "--offline" in health.detail


def test_run_checks_online_uses_service_health_check(
    fast_cached_programs: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"gpu_services": {"a": "up"}})

    results = doctor.run_checks(offline=False, api_url="https://fake.example", transport=_transport(handler))
    health = next(r for r in results if r.name == "Upstream service health")
    assert health.status == "PASS"


def test_run_checks_skips_sdk_dependent_checks_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stock install without the optional SDK used to emit 3-4 opaque
    ModuleNotFoundError FAILs (llama_cpp, upstream service health's default-URL
    lookup, base-model cache, cached programs) on top of check #1's one real FAIL.
    Once `check_sdk_importable()` itself fails, every one of those must be skipped as
    a WARN instead of actually run -- leaving exactly one FAIL in the whole run."""
    monkeypatch.setitem(sys.modules, "programasweights", None)

    results = doctor.run_checks(offline=False)
    by_name = {r.name: r for r in results}

    assert by_name["programasweights SDK"].status == "FAIL"
    for name in ("llama_cpp", "Upstream service health", "Base model cache", "Cached programs"):
        assert by_name[name].status == "WARN", name
        assert by_name[name].detail == "skipped (SDK not installed)"

    fails = [r for r in results if r.status == "FAIL"]
    assert len(fails) == 1
    assert fails[0].name == "programasweights SDK"

    # Checks that don't need the SDK at all keep running normally.
    assert by_name["GPU"].status in ("PASS", "WARN")
    assert by_name["PAW_API_KEY"].status in ("PASS", "WARN")
    assert by_name["Rate limits"].status == "PASS"


def test_run_checks_offline_message_wins_over_sdk_missing_for_service_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--offline's "skipped (--offline)" message takes priority over the SDK-missing
    skip for the one check both apply to (upstream service health) -- the two reasons
    aren't conflated into one ambiguous message."""
    monkeypatch.setitem(sys.modules, "programasweights", None)

    results = doctor.run_checks(offline=True)
    health = next(r for r in results if r.name == "Upstream service health")
    assert health.status == "WARN"
    assert "--offline" in health.detail
    assert "SDK not installed" not in health.detail


def test_run_checks_sdk_missing_still_runs_service_health_when_api_url_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_service_health only needs the SDK to resolve a *default* API URL; an
    explicit api_url needs no SDK import at all, so it must not be skipped just
    because the SDK happens to be missing."""
    monkeypatch.setitem(sys.modules, "programasweights", None)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"gpu_services": {"a": "up"}})

    results = doctor.run_checks(offline=False, api_url="https://fake.example", transport=_transport(handler))
    health = next(r for r in results if r.name == "Upstream service health")
    assert health.status == "PASS"


def test_cli_doctor_exit_code_1_only_for_the_real_fail_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: `paw-kit doctor --offline` against the real (unmocked) checks with
    the SDK unimportable exits 1 (the SDK really is missing -- that is a genuine FAIL),
    but reports exactly one FAIL rather than a wall of ModuleNotFoundErrors."""
    monkeypatch.setitem(sys.modules, "programasweights", None)

    result = runner.invoke(app, ["doctor", "--offline", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    fails = [entry for entry in payload if entry["status"] == "FAIL"]
    assert len(fails) == 1
    assert fails[0]["name"] == "programasweights SDK"


def test_run_checks_returns_all_expected_check_names(fast_cached_programs: None) -> None:
    results = doctor.run_checks(offline=True)
    names = {r.name for r in results}
    assert names == {
        "programasweights SDK",
        "llama_cpp",
        "GPU",
        "PAW_API_KEY",
        "Upstream service health",
        "Base model cache",
        "Cached programs",
        "Rate limits",
    }


# ---------------------------------------------------------------- CLI: `paw-kit doctor`


def test_cli_doctor_json_shape_and_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` prints a JSON list of {name, status, detail, remedy} dicts, and the
    process exits 0 when nothing FAILed."""
    fake_results = [
        doctor.CheckResult("Check A", "PASS", "all good", ""),
        doctor.CheckResult("Check B", "WARN", "meh", "fix it"),
    ]
    monkeypatch.setattr("paw_kit.doctor.run_checks", lambda **kw: fake_results)

    result = runner.invoke(app, ["doctor", "--json", "--offline"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == [asdict(r) for r in fake_results]


def test_cli_doctor_exits_1_on_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        doctor.CheckResult("Check A", "PASS", "all good", ""),
        doctor.CheckResult("Check B", "FAIL", "broken", "fix it now"),
    ]
    monkeypatch.setattr("paw_kit.doctor.run_checks", lambda **kw: fake_results)

    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload[1]["status"] == "FAIL"


def test_cli_doctor_table_renders_status_and_remedy(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_results = [
        doctor.CheckResult("Check A", "FAIL", "something broke", "run `paw-kit doctor` again"),
    ]
    monkeypatch.setattr("paw_kit.doctor.run_checks", lambda **kw: fake_results)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    output = strip_ansi(result.stdout)
    assert "Check A" in output
    assert "FAIL" in output
    assert "something broke" in output
    assert "run `paw-kit doctor` again" in output


def test_cli_doctor_table_remedy_column_folds_long_urls_instead_of_truncating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long remedy (typically a URL) must wrap onto further lines (Rich's "fold"
    overflow) rather than being truncated with an ellipsis -- a truncated remedy URL
    is not clickable/copyable as a working link."""
    long_remedy = (
        "Get one at https://programasweights.com/settings/really/quite/a/long/path/"
        "that/would/otherwise/be/truncated/by/the/default/rich/table/overflow/policy"
    )
    fake_results = [
        doctor.CheckResult("PAW_API_KEY", "WARN", "not set", long_remedy),
    ]
    monkeypatch.setattr("paw_kit.doctor.run_checks", lambda **kw: fake_results)

    import paw_kit.cli as cli_module

    monkeypatch.setattr(cli_module.console, "width", 60)

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    output = strip_ansi(result.stdout)
    assert "…" not in output
    # The full remedy text must survive somewhere in the rendered output once its
    # fold-wrapped line breaks are removed, never cut short with an ellipsis.
    flat = output.replace("\n", "").replace(" ", "").replace("│", "")
    assert "programasweights.com" in flat
    assert "long/path/that/would/otherwise/be/truncated" in flat
    # And it actually wrapped onto more than one line -- proof "fold" is active
    # rather than the column simply being wide enough to fit it on one.
    assert output.count("\n") > 4


def test_cli_doctor_json_output_not_mangled_by_rich_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    """A detail string containing square brackets must survive `--json` byte-for-byte
    -- Rich would otherwise parse it as markup and silently eat it."""
    fake_results = [
        doctor.CheckResult("Check A", "WARN", "value looked like [a tag] in the output", ""),
    ]
    monkeypatch.setattr("paw_kit.doctor.run_checks", lambda **kw: fake_results)

    result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.stdout)
    assert payload[0]["detail"] == "value looked like [a tag] in the output"


def test_cli_doctor_passes_adapter_and_offline_flags(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: Dict[str, Any] = {}

    def fake_run_checks(**kwargs: Any) -> List[doctor.CheckResult]:
        captured.update(kwargs)
        return [doctor.CheckResult("Check A", "PASS", "ok", "")]

    monkeypatch.setattr("paw_kit.doctor.run_checks", fake_run_checks)

    adapter_path = tmp_path / "a.paw"
    adapter_path.write_text("{}")

    result = runner.invoke(app, ["doctor", "--offline", "--adapter", str(adapter_path), "--json"])
    assert result.exit_code == 0
    assert captured["offline"] is True
    assert captured["adapter_path"] == str(adapter_path)


def test_cli_doctor_real_checks_offline(fast_cached_programs: None, tmp_path: Path) -> None:
    """End-to-end smoke test against the real (unmocked) checks, network skipped."""
    result = runner.invoke(app, ["doctor", "--offline", "--json"])
    payload = json.loads(result.stdout)
    assert isinstance(payload, list)
    assert len(payload) == 8
    for entry in payload:
        assert set(entry.keys()) == {"name", "status", "detail", "remedy"}
        assert entry["status"] in ("PASS", "WARN", "FAIL")
