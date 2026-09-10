"""Tests for ProgramAsWeightsBackend against a fake SDK (no network, no model)."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List

import pytest

from paw_kit import AbstractPAWBackend
from paw_kit.backend.programasweights import (
    FAST_COMPILER,
    FINETUNE_COMPILER,
    ProgramAsWeightsBackend,
    _render_spec_with_examples,
)


class _Program:
    def __init__(self, id: str, slug: str = "slug", status: str = "completed", error: str | None = None):
        self.id, self.slug, self.status, self.error = id, slug, status, error


class FakeSDK:
    """Duck-types the subset of `programasweights` the backend uses."""

    def __init__(
        self,
        statuses: List[str] | None = None,
        precheck_cached: bool = False,
        precheck_raises: bool = False,
    ):
        self.compile_calls: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.precheck_calls: List[Dict[str, Any]] = []
        self._statuses = list(statuses or ["queued", "running", "completed"])
        self._polls = 0
        self._precheck_cached = precheck_cached
        self._precheck_raises = precheck_raises

    def compile(self, spec: str, compiler: str | None = None, **kw):
        self.compile_calls.append({"spec": spec, "compiler": compiler, **kw})
        return _Program(id="prog-fast", slug="fast-slug")

    def compile_async(self, spec: str, compiler: str, **kw):
        self.compile_calls.append({"spec": spec, "compiler": compiler, "async": True, **kw})
        return {"job_id": "job-1", "status": "queued", "program_id": None}

    def precheck_compile(self, spec: str, compiler: str | None = None):
        self.precheck_calls.append({"spec": spec, "compiler": compiler})
        if self._precheck_raises:
            raise RuntimeError("precheck unavailable")
        return {
            "cached": self._precheck_cached,
            "program_id": "prog-cached" if self._precheck_cached else None,
        }

    def get_compile_status(self, job_id: str):
        status = self._statuses[min(self._polls, len(self._statuses) - 1)]
        self._polls += 1
        done = status in ("completed", "failed")
        return {
            "job_id": job_id,
            "status": status,
            "program_id": "prog-ft" if status == "completed" else None,
            "slug": "ft-slug",
            "error": "boom" if status == "failed" else None,
            "completed_at": "now" if done else None,
        }

    def function(self, program_id: str, **kw):
        self.function_calls.append({"program_id": program_id, **kw})

        def fn(text: str, max_tokens=None, temperature=0.0) -> str:
            return f"out({program_id}):{text}"

        return fn


@pytest.fixture
def key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAW_API_KEY", "paw_sk_test")


def test_conforms_to_protocol() -> None:
    assert isinstance(ProgramAsWeightsBackend(sdk=FakeSDK()), AbstractPAWBackend)


def test_compile_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    with pytest.raises(RuntimeError, match="PAW_API_KEY"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))


def test_fast_compile_writes_manifest_and_folds_examples(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=1)
    out = tmp_path / "triage.paw"
    examples = [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}]

    assert backend.compile("Classify tickets.", examples, str(out)) == str(out)

    sent = sdk.compile_calls[0]
    assert sent["compiler"] == FAST_COMPILER
    assert "Input: a" in sent["spec"] and "Input: b" not in sent["spec"]  # capped at 1

    manifest = json.loads(out.read_text())
    assert manifest["backend"] == "programasweights"
    assert manifest["program_id"] == "prog-fast"
    assert manifest["examples_count"] == 2
    assert manifest["examples_folded_into_spec"] == 1


def test_finetune_compile_polls_until_done(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(statuses=["queued", "running", "completed"])
    backend = ProgramAsWeightsBackend(compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0)
    out = tmp_path / "ft.paw"
    backend.compile("spec", [], str(out))
    assert sdk.compile_calls[0]["async"] is True
    assert json.loads(out.read_text())["program_id"] == "prog-ft"
    assert sdk._polls == 3


def test_finetune_compile_failure_raises(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(statuses=["running", "failed"])
    backend = ProgramAsWeightsBackend(compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0)
    with pytest.raises(RuntimeError, match="failed"):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))


def test_finetune_compile_timeout(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(statuses=["running"])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_timeout_s=0
    )
    with pytest.raises(TimeoutError):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))


def test_infer_loads_function_once_and_caches(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, n_gpu_layers=0, max_tokens=32)
    out = str(tmp_path / "t.paw")
    backend.compile("spec", [], out)

    assert backend.infer(out, "hello") == "out(prog-fast):hello"
    assert backend.infer(out, "again") == "out(prog-fast):again"
    assert len(sdk.function_calls) == 1
    assert sdk.function_calls[0]["n_gpu_layers"] == 0
    assert sdk.function_calls[0]["offline"] is False


def test_recompile_invalidates_cached_function(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = str(tmp_path / "t.paw")
    backend.compile("spec", [], out)
    backend.infer(out, "x")
    backend.compile("spec v2", [], out)
    backend.infer(out, "y")
    assert len(sdk.function_calls) == 2


def test_grammar_constraint_warns_once_and_is_ignored(key: None, tmp_path: Path) -> None:
    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    out = str(tmp_path / "t.paw")
    backend.compile("spec", [], out)
    with pytest.warns(UserWarning, match="grammar_constraint"):
        backend.infer(out, "a", grammar_constraint=r"\{.*\}")
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.infer(out, "b", grammar_constraint=r"\{.*\}")  # no second warning


def test_infer_rejects_foreign_manifest(tmp_path: Path) -> None:
    mock_style = tmp_path / "mock.paw"
    mock_style.write_text(json.dumps({"backend": "mock", "examples": []}))
    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    with pytest.raises(ValueError, match="not a ProgramAsWeights manifest"):
        backend.infer(str(mock_style), "x")
    with pytest.raises(FileNotFoundError):
        backend.infer(str(tmp_path / "missing.paw"), "x")


def test_not_available_without_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod, "_sdk_installed", lambda: False)
    backend = ProgramAsWeightsBackend()
    assert backend.is_available() is False
    manifest = tmp_path / "x.paw"
    manifest.write_text(json.dumps({"backend": "programasweights", "program_id": "p"}))
    with pytest.raises(RuntimeError, match="pip install programasweights"):
        backend.infer(str(manifest), "hi")


def test_render_spec_with_examples() -> None:
    assert _render_spec_with_examples("S", [], 5) == "S"
    assert _render_spec_with_examples("S", [{"input": "a", "output": "b"}], 0) == "S"
    rendered = _render_spec_with_examples("S", [{"input": "a", "output": "b"}, {"bad": 1}], 5)
    assert rendered.startswith("S\n\nExamples of correct behaviour:\nInput: a\nOutput: b")


# ---------------------------------------------------------------- public / ephemeral


def test_compile_forwards_public_false_and_ephemeral_false_by_default(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    backend.compile("spec", [], str(tmp_path / "a.paw"))
    sent = sdk.compile_calls[0]
    assert sent["public"] is False
    assert sent["ephemeral"] is False


def test_compile_forwards_public_true(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, ephemeral=True)
    backend.compile("spec", [], str(tmp_path / "a.paw"))
    sent = sdk.compile_calls[0]
    assert sent["public"] is True
    assert sent["ephemeral"] is True
    # public=True skips the cache-hit precheck entirely.
    assert sdk.precheck_calls == []


def test_finetune_compile_forwards_public_and_ephemeral(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(statuses=["completed"])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, public=True, ephemeral=True
    )
    backend.compile("spec", [], str(tmp_path / "ft.paw"))
    sent = sdk.compile_calls[0]
    assert sent["public"] is True
    assert sent["ephemeral"] is True


def test_public_compile_warns_when_examples_folded(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, max_spec_examples=5)
    examples = [{"input": "a", "output": "b"}]
    with pytest.warns(UserWarning, match="publicly"):
        backend.compile("Classify.", examples, str(tmp_path / "a.paw"))


def test_private_compile_does_not_warn(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, max_spec_examples=5)
    examples = [{"input": "a", "output": "b"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.compile("Classify.", examples, str(tmp_path / "a.paw"))


def test_public_compile_with_no_examples_folded_does_not_warn(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, max_spec_examples=0)
    examples = [{"input": "a", "output": "b"}]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.compile("Classify.", examples, str(tmp_path / "a.paw"))


def test_manifest_records_public_and_ephemeral(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, ephemeral=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public"] is True
    assert manifest["ephemeral"] is True


# ---------------------------------------------------------------- cache-hit precheck


def test_cache_hit_warns_when_public_false(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(precheck_cached=True)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    with pytest.warns(UserWarning, match="already has a compiled program"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))


def test_no_cache_hit_warning_when_not_cached(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(precheck_cached=False)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.compile("spec", [], str(tmp_path / "a.paw"))


def test_precheck_failure_is_swallowed_and_manifest_records_null(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(precheck_raises=True)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["cache_hit"] is None


# ---------------------------------------------------------------- manifest v2 / lineage


def test_manifest_v2_records_lineage_fields(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=1)
    out = tmp_path / "a.paw"
    examples = [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}]

    backend.compile("Classify tickets.", examples, str(out))

    manifest = json.loads(out.read_text())
    assert manifest["manifest_version"] == 2
    assert manifest["spec_sha256"] == hashlib.sha256(b"Classify tickets.").hexdigest()
    # full_spec_sha256 is the hash of the spec *with* the folded example appended --
    # distinct from spec_sha256 because max_spec_examples=1 folds one example in.
    assert manifest["full_spec_sha256"] != manifest["spec_sha256"]
    rendered = _render_spec_with_examples("Classify tickets.", examples, 1)
    assert manifest["full_spec_sha256"] == hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    assert manifest["folded_example_ids"] == [hashlib.sha256(b"a\x1f1").hexdigest()]
    assert manifest["parent_program_id"] is None
    assert manifest["parent_manifest_sha256"] is None
    assert isinstance(manifest["compile_wall_s"], float)
    assert manifest["compile_wall_s"] >= 0.0
    assert manifest["compiler_snapshot"] is None


def test_manifest_v2_records_parent_program_id_on_recompile(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = tmp_path / "a.paw"

    backend.compile("v1", [], str(out))
    first = json.loads(out.read_text())
    first_raw = out.read_text()

    backend.compile("v2", [], str(out))
    second = json.loads(out.read_text())

    assert first["program_id"] == "prog-fast"
    assert second["parent_program_id"] == "prog-fast"
    assert second["parent_manifest_sha256"] == hashlib.sha256(first_raw.encode("utf-8")).hexdigest()


def test_read_manifest_accepts_v1_manifest_with_no_manifest_version_key(tmp_path: Path) -> None:
    """A v1 manifest (written before this feature existed) has no manifest_version
    key at all -- read_manifest must keep accepting it unchanged."""
    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    legacy = tmp_path / "legacy.paw"
    legacy.write_text(
        json.dumps(
            {
                "backend": "programasweights",
                "program_id": "prog-legacy",
                "slug": "legacy-slug",
                "compiler": FAST_COMPILER,
                "status": "completed",
                "spec": "legacy spec",
                "examples_folded_into_spec": 0,
                "examples_count": 0,
                "public": False,
                "ephemeral": False,
                "cache_hit": None,
                "compiled_at": "2026-01-01T00:00:00Z",
            }
        )
    )
    manifest = ProgramAsWeightsBackend.read_manifest(str(legacy))
    assert manifest["program_id"] == "prog-legacy"
    assert "manifest_version" not in manifest


def test_history_log_appended_once_per_compile(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = tmp_path / "a.paw"
    history_path = Path(str(out) + ".history.jsonl")

    backend.compile("v1", [], str(out))
    backend.compile("v2", [], str(out))

    lines = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert all("spec" not in entry for entry in lines)
    assert lines[0]["program_id"] == "prog-fast"
    assert lines[1]["parent_program_id"] == "prog-fast"
    assert oct(history_path.stat().st_mode)[-3:] == "600"
