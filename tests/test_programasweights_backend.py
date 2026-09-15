"""Tests for ProgramAsWeightsBackend against a fake SDK (no network, no model)."""

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
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
        precheck_exception: BaseException | None = None,
        program_meta: Dict[str, Any] | None = None,
        meta_exception: BaseException | None = None,
    ):
        self.compile_calls: List[Dict[str, Any]] = []
        self.function_calls: List[Dict[str, Any]] = []
        self.precheck_calls: List[Dict[str, Any]] = []
        self._statuses = list(statuses or ["queued", "running", "completed"])
        self._polls = 0
        self._precheck_cached = precheck_cached
        self._precheck_raises = precheck_raises
        # A-3: the precheck's `except` is narrowed to HTTP-level failures, so which
        # exception the fake raises is now load-bearing. `precheck_raises=True` keeps
        # its historical meaning via a default httpx error; `precheck_exception` names
        # an exact one (e.g. the AttributeError of an upstream rename).
        self._precheck_exception = precheck_exception or httpx.ConnectError("precheck unavailable")
        # A-2: the `verify_visibility` seam. `get_program_meta` is a PAWClient method
        # upstream, not a module-level function, so the fake exposes a PAWClient too.
        self.program_meta: Dict[str, Any] = (
            {"public": True} if program_meta is None else program_meta
        )
        self._meta_exception = meta_exception
        self.meta_calls: List[str] = []
        self.client_kwargs: List[Dict[str, Any]] = []
        outer = self

        class _FakePAWClient:
            def __init__(self, **kwargs: Any) -> None:
                outer.client_kwargs.append(kwargs)

            def get_program_meta(self, program_id: str) -> Dict[str, Any]:
                outer.meta_calls.append(program_id)
                if outer._meta_exception is not None:
                    raise outer._meta_exception
                return outer.program_meta

        self.PAWClient = _FakePAWClient

    def get_api_url(self) -> str:
        return "https://programasweights.invalid"

    def get_api_key(self) -> str:
        return "paw_sk_test"

    def compile(self, spec: str, compiler: str | None = None, **kw):
        self.compile_calls.append({"spec": spec, "compiler": compiler, **kw})
        return _Program(id="prog-fast", slug="fast-slug")

    def compile_async(self, spec: str, compiler: str, **kw):
        self.compile_calls.append({"spec": spec, "compiler": compiler, "async": True, **kw})
        return {"job_id": "job-1", "status": "queued", "program_id": None}

    def precheck_compile(self, spec: str, compiler: str | None = None):
        self.precheck_calls.append({"spec": spec, "compiler": compiler})
        if self._precheck_raises:
            raise self._precheck_exception
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


def test_vocabulary_build_failure_warns_once_and_disables_grammar_constraint(
    key: None, tmp_path: Path
) -> None:
    """Money route (iii-a): `FakeSDK.function()` returns a plain closure with no
    `_llm` attribute, so `_build_vocabulary` raises on the first model load.
    `applies_grammar_constraint` must flip False, exactly one `UserWarning` must be
    emitted (naming the vocabulary, not the old "cannot apply grammar_constraint"
    message this backend used to emit unconditionally), and `infer()` must still
    return the LOCAL (unconstrained) output rather than raising -- never a raise on
    every call, per the track's "No money leak" invariant. Needs llguidance actually
    importable -- this is route (iii-a), which only exists to test once route (i)
    (engine absent) has already been ruled out.
    """
    pytest.importorskip("llguidance")
    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    out = str(tmp_path / "t.paw")
    backend.compile("spec", [], out)
    assert backend.applies_grammar_constraint is True  # llguidance IS importable here

    with pytest.warns(UserWarning, match="grammar-constraint vocabulary"):
        result = backend.infer(out, "a", grammar_constraint=r"\{.*\}")
    assert result == "out(prog-fast):a"
    assert backend.applies_grammar_constraint is False

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result2 = backend.infer(out, "b", grammar_constraint=r"\{.*\}")  # no second warning
    assert result2 == "out(prog-fast):b"


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
    """A-2 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    `manifest["public"]` became `manifest["public_requested"]`. The assertion is not
    weakened -- it is the *same* assertion under the name that says what the value
    actually is. The finding is precisely that this field records the request while
    reading as the fact, so the rename is the fix and re-pointing the test is how it is
    pinned. `public` is asserted absent so a reader of a new manifest cannot reach for
    the ambiguous name at all."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, ephemeral=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public_requested"] is True
    assert "public" not in manifest
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
    """A-3 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`):
    this test's `warnings.catch_warnings()` / `simplefilter("error")` block was removed.

    Both of its real subjects are intact and still asserted -- an HTTP-level precheck
    failure does not break the compile, and `cache_hit` records `None` rather than
    guessing. What the removed block additionally pinned was *silence*, and silence is
    the half A-3 fixes: a swallowed failure that warns nothing is indistinguishable from
    "checked, and there is no cache hit", which is the opposite conclusion. The warning
    is now asserted positively in
    `test_precheck_http_failure_warns_distinctly_A_3`, so the behaviour is pinned
    tighter than before, not looser. `precheck_raises=True` now raises an `httpx` error
    rather than a bare `RuntimeError`, because under A-3 only HTTP-level failures are
    swallowed at all (see `test_precheck_attribute_error_is_not_swallowed_A_3`).
    """
    sdk = FakeSDK(precheck_raises=True)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    with pytest.warns(UserWarning, match="could not check"):
        backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["cache_hit"] is None


# ---------------------------------------------------------------- manifest v2 / lineage


def test_manifest_v3_records_lineage_fields(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=1)
    out = tmp_path / "a.paw"
    examples = [{"input": "a", "output": "1"}, {"input": "b", "output": "2"}]

    backend.compile("Classify tickets.", examples, str(out))

    manifest = json.loads(out.read_text())
    # D-ADD-1: bumped 2 -> 3 -- A-2 removed `public` and added `public_requested`/
    # `public_confirmed`/`public_confirmed_reason`/`cached_program_id`, a strictly
    # larger change than the purely additive one that took this constant 1 -> 2.
    assert manifest["manifest_version"] == 3
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
# ---------------------------------------------------------------- compile retry/error-wrap


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://programasweights.com/api/v1/compile")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


class _FlakyCompileSDK(FakeSDK):
    """`FakeSDK` variant whose `compile`/`compile_async` raise a scripted queue of
    exceptions (one per call) before falling through to the real fake behaviour."""

    def __init__(self, exceptions: List[Exception], **kwargs: Any):
        super().__init__(**kwargs)
        self._exceptions = list(exceptions)
        self.attempts = 0

    def compile(self, spec: str, compiler: str | None = None, **kw: Any):
        self.attempts += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return super().compile(spec, compiler=compiler, **kw)

    def compile_async(self, spec: str, compiler: str, **kw: Any):
        self.attempts += 1
        if self._exceptions:
            raise self._exceptions.pop(0)
        return super().compile_async(spec, compiler, **kw)


def test_compile_5xx_exhausted_raises_runtime_error_naming_doctor(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5xx that outlives every retry becomes a RuntimeError naming the service, the
    status, and `paw-kit doctor` -- not a raw httpx.HTTPStatusError."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(503), _status_error(503)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=1)

    with pytest.raises(RuntimeError, match="ProgramAsWeights compile service returned HTTP 503") as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert "paw-kit doctor" in str(exc_info.value)
    assert sdk.attempts == 2  # initial attempt + 1 retry, then exhausted


def test_compile_5xx_retried_then_succeeds(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 5xx followed by success is retried transparently -- compile() returns normally,
    honouring `compile_retries`."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(500)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=2)

    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    assert sdk.attempts == 2  # one failure, one success
    assert json.loads(out.read_text())["program_id"] == "prog-fast"


def test_compile_4xx_never_retried(key: None, tmp_path: Path) -> None:
    """A 4xx (bad request / invalid key / rate limit) fails immediately, with no retry
    -- retrying would waste a rate-limited attempt on something that fails identically."""
    sdk = _FlakyCompileSDK([_status_error(422)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=3)

    with pytest.raises(RuntimeError, match="HTTP 422"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert sdk.attempts == 1


def test_compile_readtimeout_is_never_retried(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read timeout means the request may already have reached the server -- it must
    raise immediately (never retried), and the message must say the compile may still
    be running server-side and that a re-run will hit the compile cache."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([httpx.ReadTimeout("timed out"), httpx.ReadTimeout("timed out")])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=3)

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    message = str(exc_info.value)
    assert "may still be running" in message
    assert "compile cache" in message
    # Only the one, un-retried attempt was made -- a second ReadTimeout is still
    # queued in the fake SDK and was never consumed.
    assert sdk.attempts == 1


def test_compile_connect_error_is_retried_then_succeeds(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike a read timeout, a connect error/timeout means the request provably never
    reached the server -- it is safe to retry, and is."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([httpx.ConnectError("connection refused")])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=1)

    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    assert sdk.attempts == 2
    assert json.loads(out.read_text())["program_id"] == "prog-fast"


def test_compile_connect_timeout_exhausted_raises_runtime_error(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([httpx.ConnectTimeout("timed out"), httpx.ConnectTimeout("timed out")])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=1)

    with pytest.raises(RuntimeError, match="unreachable") as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert "paw-kit doctor" in str(exc_info.value)
    assert sdk.attempts == 2


def test_compile_504_is_never_retried(key: None, tmp_path: Path) -> None:
    """504 (gateway timeout) carries the same "already landed, still working"
    ambiguity as a read timeout -- excluded from the retryable 5xx set."""
    sdk = _FlakyCompileSDK([_status_error(504), _status_error(504)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=3)

    with pytest.raises(RuntimeError, match="HTTP 504"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert sdk.attempts == 1


def test_compile_error_appends_response_body_capped_at_300_chars(
    key: None, tmp_path: Path
) -> None:
    """Finding 6: the upstream response body (where the actual 4xx/5xx reason lives)
    must be appended to the wrapped error, capped at 300 characters."""
    request = httpx.Request("POST", "https://programasweights.com/api/v1/compile")
    long_body = "invalid spec: " + ("x" * 500)
    response = httpx.Response(422, request=request, text=long_body)
    exc = httpx.HTTPStatusError("HTTP 422", request=request, response=response)
    sdk = _FlakyCompileSDK([exc])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=0)

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    message = str(exc_info.value)
    assert "invalid spec:" in message
    # Capped at 300 chars of body text, not the full 500+ characters.
    assert len(message) < len(long_body)


def test_compile_4xx_does_not_advise_running_doctor(key: None, tmp_path: Path) -> None:
    """Finding 6: a 4xx is a client-side problem (bad spec, bad key, rate limit) that
    `paw-kit doctor` (an environment/service diagnostic) cannot help with -- that
    advice is dropped from the 4xx branch specifically."""
    sdk = _FlakyCompileSDK([_status_error(422)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=3)

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert "paw-kit doctor" not in str(exc_info.value)


def test_compile_5xx_still_advises_running_doctor(key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The doctor advice is only dropped for 4xx -- a 5xx (server-side) still gets it."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(503), _status_error(503)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=1)

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert "paw-kit doctor" in str(exc_info.value)


def test_compile_retries_default_is_one(key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`compile_retries` defaults to 1: a single 5xx is absorbed transparently."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(502)])
    backend = ProgramAsWeightsBackend(sdk=sdk)  # compile_retries not passed

    backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert sdk.attempts == 2


def test_finetune_compile_submission_5xx_wrapped(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The async/finetune path gets the same error wrapping on the initial submission
    call, without changing the polling behaviour in `_wait_for_job`.

    A-1 (named hazard, listed in `conductor/tracks/bug-hunt-D-money-privacy.md`): this
    test's `sdk.attempts == 2` became `== 1`. It was pinning the retry that A-1 removes
    from the async path, not the error wrapping this test is named for -- the wrapping
    assertion (`pytest.raises(... "HTTP 503")`) is unchanged and still the subject. The
    `compile_retries=1` and the second queued 503 are kept deliberately: they are what
    makes `attempts == 1` mean "the retry budget existed and was correctly not spent",
    rather than "there was nothing to retry with".
    """
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(503), _status_error(503)], statuses=["completed"])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_retries=1
    )

    with pytest.raises(RuntimeError, match="ProgramAsWeights compile service returned HTTP 503"):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert sdk.attempts == 1


# ---------------------------------------------------------------- _wait_for_job poll retry


class _FlakyStatusSDK(FakeSDK):
    """`FakeSDK` variant whose `get_compile_status` raises a scripted queue of
    exceptions (one per call) before falling through to the real fake behaviour --
    exercises `_wait_for_job`'s consecutive-transient-failure tolerance."""

    def __init__(self, exceptions: List[Exception], **kwargs: Any):
        super().__init__(**kwargs)
        self._status_exceptions = list(exceptions)
        self.status_attempts = 0

    def get_compile_status(self, job_id: str):
        self.status_attempts += 1
        if self._status_exceptions:
            raise self._status_exceptions.pop(0)
        return super().get_compile_status(job_id)


def test_wait_for_job_tolerates_transient_failures_then_succeeds(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transient poll failure (timeout, connect error, or 5xx) must not throw away
    an in-progress compile -- status polling is idempotent, unlike the submission
    call. The failure counter resets on the next successful poll."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_POLL_FAILURE_BACKOFF_S", (0.0,) * 5)
    sdk = _FlakyStatusSDK(
        [httpx.ReadTimeout("timed out"), httpx.ConnectError("refused")],
        statuses=["completed"],
    )
    backend = ProgramAsWeightsBackend(compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0)

    out = tmp_path / "ft.paw"
    backend.compile("spec", [], str(out))
    assert json.loads(out.read_text())["program_id"] == "prog-ft"
    # 2 failed polls followed by 1 successful poll.
    assert sdk.status_attempts == 3


def test_wait_for_job_gives_up_after_six_consecutive_failures(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tolerates up to 5 consecutive transient poll failures (with backoff); the 6th
    consecutive failure gives up and raises a RuntimeError naming the job_id so the
    caller can poll the (possibly still-running) compile again later by hand."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_POLL_FAILURE_BACKOFF_S", (0.0,) * 5)
    sdk = _FlakyStatusSDK([_status_error(502) for _ in range(6)])
    backend = ProgramAsWeightsBackend(compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0)

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert "job-1" in str(exc_info.value)
    assert sdk.status_attempts == 6


def test_wait_for_job_poll_failures_still_bounded_by_total_wall_clock_cap(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The overall `compile_timeout_s` cap still applies on top of the per-failure
    tolerance: a compile_timeout_s of 0 times out on the very first transient failure
    instead of working through all 5 tolerated failures (and their backoff) first."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_POLL_FAILURE_BACKOFF_S", (0.0,) * 5)
    sdk = _FlakyStatusSDK([httpx.ReadTimeout("timed out")] * 6)
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_timeout_s=0
    )

    with pytest.raises(TimeoutError):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert sdk.status_attempts == 1


# ================================================================== A-1: retry safety
#
# Decided in Phase 0 (option (b)): retry the cache-keyed *sync* `paw.compile` only;
# never retry `paw.compile_async` on a 5xx. The money is entirely on the async path
# (it is reachable only via FINETUNE_COMPILER) and only the async path is
# unreconcilable -- a retried submission discards the first attempt's `job_id`, which
# makes that paid job unpollable and uncancellable. Connect-failure retry is
# unchanged on both paths: the request provably never reached the server.


@pytest.mark.parametrize("status_code", [500, 502, 503])
def test_finetune_compile_async_5xx_is_never_retried_A_1(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    """`compile_async` is invoked exactly once for 500/502/503 -- the report's own
    required assertion. A non-504 5xx proves nothing about whether the compile landed,
    and a retry that buys a second finetune compile costs 96-223s of paid GPU while
    orphaning the first job_id."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(status_code)], statuses=["completed"])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_retries=3
    )

    with pytest.raises(RuntimeError, match=f"HTTP {status_code}"):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert sdk.attempts == 1, (
        "compile_async was resubmitted after a 5xx: that buys a second paid finetune "
        "compile and orphans the first job_id"
    )


def test_sync_compile_still_retries_5xx_A_1(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half of option (b): the sync `paw.compile` path keeps its 5xx retry.
    Upstream's compile cache is keyed on spec text, so a resubmitted identical spec
    normally returns the existing program, and there is no job_id to orphan."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([_status_error(500)])
    backend = ProgramAsWeightsBackend(sdk=sdk, compile_retries=2)

    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    assert sdk.attempts == 2
    assert json.loads(out.read_text())["program_id"] == "prog-fast"


def test_finetune_compile_async_connect_error_is_still_retried_A_1(
    key: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Option (b) narrows the *5xx* retry only. A connect failure means the TCP
    handshake itself never completed, so the request provably never reached the server
    and nothing was billed -- that retry stays in place on the async path too."""
    import paw_kit.backend.programasweights as mod

    monkeypatch.setattr(mod.ProgramAsWeightsBackend, "_COMPILE_RETRY_BACKOFF_S", 0.0)
    sdk = _FlakyCompileSDK([httpx.ConnectError("refused")], statuses=["completed"])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_retries=1
    )

    out = tmp_path / "ft.paw"
    backend.compile("spec", [], str(out))
    assert sdk.attempts == 2
    assert json.loads(out.read_text())["program_id"] == "prog-ft"


# ============================================ A-5: unrecognised terminal job states
#
# `_wait_for_job` only knows two vocabularies (`_SUCCESS_STATES`, `_FAILED_STATES`).
# Anything else -- `infrastructure_error`, `redis_unavailable`, a `completed` that
# named no program -- fell through to the sleep loop and was polled for the full
# `compile_timeout_s` (720 GETs at the defaults), with the server's own `error` string
# never read and `cancel_compile` never called, on the paid finetune path.


class _CancellableSDK(FakeSDK):
    """`FakeSDK` plus a `cancel_compile` spy and a scriptable status payload."""

    def __init__(self, payloads: List[Dict[str, Any]], **kwargs: Any):
        super().__init__(**kwargs)
        self._payloads = list(payloads)
        self.status_attempts = 0
        self.cancel_calls: List[str] = []

    def get_compile_status(self, job_id: str):
        self.status_attempts += 1
        payload = self._payloads[min(self.status_attempts - 1, len(self._payloads) - 1)]
        return {"job_id": job_id, **payload}

    def cancel_compile(self, job_id: str):
        self.cancel_calls.append(job_id)
        return {"job_id": job_id, "status": "cancelled"}


def test_wait_for_job_unknown_status_with_error_fails_fast_A_5(
    key: None, tmp_path: Path
) -> None:
    """An unrecognised status carrying a non-null `error` is terminal failure, raised
    on the first poll with the server's `error` surfaced verbatim -- not polled for the
    full hour."""
    sdk = _CancellableSDK(
        [{"status": "infrastructure_error", "program_id": None,
          "error": "redis_unavailable: queue backend down"}]
    )
    backend = ProgramAsWeightsBackend(
        # A real `compile_timeout_s` is 3600s; 2.0s keeps the *failing* (pre-fix) run
        # bounded for the red-at-main gate, which would otherwise spin for an hour.
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0.05, compile_timeout_s=2.0
    )

    with pytest.raises(RuntimeError) as exc_info:
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    message = str(exc_info.value)
    assert "redis_unavailable: queue backend down" in message, "the server's error string must be surfaced verbatim"
    assert "infrastructure_error" in message
    assert "job-1" in message
    assert sdk.status_attempts == 1, (
        f"polled {sdk.status_attempts} times for a job the server had already given up "
        "on; an unrecognised terminal state must fail within one poll interval"
    )


def test_wait_for_job_success_status_with_no_program_id_but_an_error_fails_fast_A_5(
    key: None, tmp_path: Path
) -> None:
    """`completed` with a null `program_id` and a non-null `error` is the same defect
    wearing a recognised status name: the job is over, there is no program, and the
    old code slept on it until `compile_timeout_s`."""
    sdk = _CancellableSDK(
        [{"status": "completed", "program_id": None, "error": "adapter upload failed"}]
    )
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0.05, compile_timeout_s=2.0
    )

    with pytest.raises(RuntimeError, match="adapter upload failed"):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert sdk.status_attempts == 1


def test_wait_for_job_cancels_the_job_once_on_a_genuine_timeout_A_5(
    key: None, tmp_path: Path
) -> None:
    """When the poll loop itself gives up, the queued job is still the caller's to pay
    for -- attempt `cancel_compile` exactly once so it does not run on anyway."""
    sdk = _CancellableSDK([{"status": "running", "program_id": None, "error": None}])
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0, compile_timeout_s=0
    )

    with pytest.raises(TimeoutError):
        backend.compile("spec", [], str(tmp_path / "ft.paw"))
    assert sdk.cancel_calls == ["job-1"]


def test_wait_for_job_unknown_status_without_an_error_keeps_polling_A_5(
    key: None, tmp_path: Path
) -> None:
    """Negative control (green at `main` by design -- see the track file). The fix must
    not turn every unfamiliar status name into a failure: a server that introduces
    `provisioning` as a *transient* state carries no `error`, and the compile the user
    already paid for must keep being polled."""
    sdk = _CancellableSDK(
        [
            {"status": "provisioning", "program_id": None, "error": None},
            {"status": "completed", "program_id": "prog-ft", "slug": "s", "error": None,
             "completed_at": "now"},
        ]
    )
    backend = ProgramAsWeightsBackend(compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0)

    out = tmp_path / "ft.paw"
    backend.compile("spec", [], str(out))
    assert json.loads(out.read_text())["program_id"] == "prog-ft"
    assert sdk.status_attempts == 2
    assert sdk.cancel_calls == []


# ===================================================== A-9: public/offline validation


def test_public_none_is_coerced_to_false_A_9(key: None, tmp_path: Path) -> None:
    """`public: bool = False` was unvalidated, so `public=None` forwarded
    `{"public": null}` to a service whose own default is `True` -- a leak out of a
    falsy-looking argument. Anything that is not literally `True` is `False`."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=None)  # type: ignore[arg-type]
    assert backend.public is False
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    assert sdk.compile_calls[0]["public"] is False
    # And the manifest records False, not None -- honest about what was asked for.
    assert json.loads(out.read_text())["public_requested"] is False


@pytest.mark.parametrize("value", [None, 0, "", "true", 1, "yes"])
def test_only_literal_true_opts_into_public_A_9(value: object) -> None:
    """Every truthy-or-falsy non-`True` value resolves to private. `"true"` and `1` are
    the dangerous ones: both are truthy, so a looser `bool(public)` check would publish."""
    backend = ProgramAsWeightsBackend(sdk=FakeSDK(), public=value)  # type: ignore[arg-type]
    assert backend.public is False


def test_offline_backend_refuses_to_compile_A_9(tmp_path: Path) -> None:
    """`offline=True` documents "never touch the network"; it skipped the API-key guard
    and then POSTed anyway. The raise is at the top of compile(), before any paid work
    -- not on a request path."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, offline=True)
    with pytest.raises(RuntimeError, match="offline=True"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert sdk.compile_calls == []
    assert sdk.precheck_calls == []


def test_offline_backend_can_still_infer_A_9(key: None, tmp_path: Path) -> None:
    """Negative control: `offline=True` is an *inference* mode (that is what the only
    in-repo construction of it uses it for). The new raise must not reach infer()."""
    out = str(tmp_path / "t.paw")
    ProgramAsWeightsBackend(sdk=FakeSDK()).compile("spec", [], out)
    offline_backend = ProgramAsWeightsBackend(sdk=FakeSDK(), offline=True)
    assert offline_backend.infer(out, "hello") == "out(prog-fast):hello"


# ======================================== A-2: requested vs confirmed visibility
#
# Confirmed live (parent track doc, "Report §16 live verification"): all six recorded
# programs report `public: True` from the server while the local manifests record only
# what was *asked for* -- and the `public` key is absent from all six, because it did
# not exist before b47ddea. This renames the request to `public_requested`, adds an
# opt-in `verify_visibility` that records the server's answer as a THREE-state value,
# and lands the free half (`cached_program_id`, which the precheck already fetched and
# the old code threw away) unconditionally.


def test_manifest_separates_requested_from_confirmed_visibility_A_2(
    key: None, tmp_path: Path
) -> None:
    """The headline assertion: the manifest must distinguish "this is what I asked for"
    from "this is what the server says it is"."""
    sdk = FakeSDK(program_meta={"public": True})
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))

    manifest = json.loads(out.read_text())
    assert manifest["public_requested"] is False
    assert manifest["public_confirmed"] is True, (
        "the server said this program is public; the manifest must say so too instead "
        "of repeating the request back"
    )
    assert manifest["public_confirmed_reason"] == "server"
    assert sdk.meta_calls == ["prog-fast"]


def test_manifest_records_confirmed_private_A_2(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(program_meta={"public": False})
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public_confirmed"] is False
    assert manifest["public_confirmed_reason"] == "server"


def test_unverified_visibility_is_none_not_false_A_2(key: None, tmp_path: Path) -> None:
    """Pattern 5, which is the whole shape of A-2: "not checked" must never render as
    "private". The default path does not call the server at all."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public_confirmed"] is None
    assert manifest["public_confirmed_reason"] == "not_attempted"
    assert sdk.meta_calls == []


def test_visibility_response_without_a_visibility_key_is_none_A_2(
    key: None, tmp_path: Path
) -> None:
    """A server response that carries no visibility field must read as unknown, not as
    private -- the same Pattern 5 from the other direction."""
    sdk = FakeSDK(program_meta={"id": "prog-fast", "slug": "s"})
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public_confirmed"] is None
    assert manifest["public_confirmed_reason"] == "no_visibility_key"


def test_visibility_check_failure_never_loses_the_paid_compile_A_2(
    key: None, tmp_path: Path
) -> None:
    """Phase 0 F8, the money-losing failure this fix must not introduce:
    `get_program_meta` does a 10s httpx.get with raise_for_status(), and it runs *after*
    the billed compile returned. If it could raise out of compile(), an expired key or a
    transient 5xx would leave no manifest written and the program_id of a paid compile
    lost. No outcome of the verification may prevent the manifest write."""
    sdk = FakeSDK(meta_exception=httpx.HTTPStatusError(
        "404",
        request=httpx.Request("GET", "https://programasweights.invalid/api/v1/programs/prog-fast"),
        response=httpx.Response(404, request=httpx.Request(
            "GET", "https://programasweights.invalid/api/v1/programs/prog-fast")),
    ))
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    out = tmp_path / "a.paw"

    backend.compile("spec", [], str(out))  # must NOT raise

    manifest = json.loads(out.read_text())
    assert manifest["program_id"] == "prog-fast", "the paid compile's program_id survived"
    assert manifest["public_confirmed"] is None
    assert manifest["public_confirmed_reason"].startswith("request_failed")
    assert "HTTPStatusError" in manifest["public_confirmed_reason"]


def test_visibility_check_crash_never_loses_the_paid_compile_A_2(
    key: None, tmp_path: Path
) -> None:
    """The same guarantee for a non-HTTP fault (an upstream rename of
    `get_program_meta`, a client constructor that changed signature)."""
    sdk = FakeSDK(meta_exception=AttributeError("no attribute 'get_program_meta'"))
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["program_id"] == "prog-fast"
    assert manifest["public_confirmed"] is None
    assert "AttributeError" in manifest["public_confirmed_reason"]


def test_visibility_is_not_verified_without_an_api_key_A_2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_program_meta` needs a key. Without one, record *why* it is unknown rather
    than spending a request that will 401.

    Tested directly on the helper rather than through `compile()`, because `compile()`
    already refuses to run without a key -- so this branch is unreachable from there
    today. It is kept (and pinned) anyway: it is the correct answer for the state, and
    the enumeration of reasons is the part of A-2 that has to stay exhaustive. A reason
    set with a hole in it is how "unknown" turns back into "private".
    """
    monkeypatch.delenv("PAW_API_KEY", raising=False)
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False, verify_visibility=True)
    assert backend._confirm_visibility(sdk, "prog-fast") == (None, "no_api_key")
    assert sdk.meta_calls == []


def test_finetune_path_also_records_confirmed_visibility_A_2(
    key: None, tmp_path: Path
) -> None:
    """The expensive path is the one where getting this wrong matters most."""
    sdk = FakeSDK(statuses=["completed"], program_meta={"public": True})
    backend = ProgramAsWeightsBackend(
        compiler=FINETUNE_COMPILER, sdk=sdk, poll_interval_s=0,
        public=False, verify_visibility=True,
    )
    out = tmp_path / "ft.paw"
    backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["public_confirmed"] is True
    assert sdk.meta_calls == ["prog-ft"]


def test_cache_hit_program_id_is_recorded_A_2(key: None, tmp_path: Path) -> None:
    """The free half: `precheck["program_id"]` is the id of the already-compiled program
    a cache hit will hand back. The old code fetched it and threw it away, leaving a
    warning that said "an existing program will be returned" without naming which."""
    sdk = FakeSDK(precheck_cached=True)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    with pytest.warns(UserWarning, match="already has a compiled program"):
        backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["cache_hit"] is True
    assert manifest["cached_program_id"] == "prog-cached"


def test_cached_program_id_is_none_without_a_cache_hit_A_2(key: None, tmp_path: Path) -> None:
    sdk = FakeSDK(precheck_cached=False)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    backend.compile("spec", [], str(out))
    assert json.loads(out.read_text())["cached_program_id"] is None


def test_visibility_is_never_verified_when_offline_A_2(tmp_path: Path) -> None:
    """`offline=True` must not make a network call even to verify visibility. A-9's
    raise gets there first, so this is a belt-and-braces assertion on both findings."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, offline=True, verify_visibility=True)
    with pytest.raises(RuntimeError, match="offline=True"):
        backend.compile("spec", [], str(tmp_path / "a.paw"))
    assert sdk.meta_calls == []


# ============================= A-3: the precheck's bare `except` and the contract


def test_precheck_attribute_error_is_not_swallowed_A_3(key: None, tmp_path: Path) -> None:
    """The defect: `except Exception` around `paw.precheck_compile(...)` means an
    upstream *rename* silently disables the cache-hit leak warning forever, and nothing
    in the suite notices. An AttributeError is a broken contract, not a transient
    service fault, and must not be absorbed."""
    sdk = FakeSDK(precheck_raises=True, precheck_exception=AttributeError("precheck_compile"))
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    with pytest.raises(AttributeError):
        backend.compile("spec", [], str(tmp_path / "a.paw"))


def test_precheck_http_failure_warns_distinctly_A_3(key: None, tmp_path: Path) -> None:
    """A genuine HTTP-level precheck failure is still swallowed -- but it now says so.
    Silence was indistinguishable from "checked, and there is no cache hit", which is
    the opposite conclusion."""
    sdk = FakeSDK(precheck_raises=True)
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    with pytest.warns(UserWarning, match="could not check"):
        backend.compile("spec", [], str(out))
    manifest = json.loads(out.read_text())
    assert manifest["cache_hit"] is None
    assert manifest["program_id"] == "prog-fast"


# ============================================== A-6: the folded-example count
#
# `examples_folded_into_spec` was `min(len(examples), max_spec_examples)` while the spec
# renderer filtered to usable dicts first. Any malformed example therefore inflated the
# reported count above what actually reached the spec -- and that count is mirrored into
# every published measurement artifact.


def _rendered_inputs(spec_text: str) -> List[str]:
    """The `Input:` lines `_render_spec_with_examples` actually emitted -- the ground
    truth `examples_folded_into_spec` is supposed to report."""
    return [line[len("Input: "):] for line in spec_text.splitlines() if line.startswith("Input: ")]


def test_folded_count_matches_what_was_actually_rendered_A_6(
    key: None, tmp_path: Path
) -> None:
    """The count is what reached the spec, not what was offered."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=8)
    out = tmp_path / "a.paw"
    examples = [
        {"input": "a", "output": "1"},
        {"bad": 1},                      # no input/output keys: never rendered
        {"input": "b"},                  # half an example: never rendered
        {"input": "c", "output": "3"},
        "not even a dict",               # type: ignore[list-item]
    ]

    backend.compile("Classify.", examples, str(out))  # type: ignore[arg-type]

    manifest = json.loads(out.read_text())
    rendered = _rendered_inputs(sdk.compile_calls[0]["spec"])
    assert rendered == ["a", "c"]
    assert manifest["examples_folded_into_spec"] == len(rendered), (
        f"manifest claims {manifest['examples_folded_into_spec']} examples were folded "
        f"into the spec, but {len(rendered)} actually were -- and that claim is mirrored "
        "into every published artifact"
    )
    assert manifest["examples_count"] == len(examples), "the offered count is still reported"


def test_folded_count_includes_a_non_str_example_that_still_renders_A_6(
    key: None, tmp_path: Path
) -> None:
    """The case that separates the right fix from the plausible wrong one.

    `len(folded_example_ids)` looks like the obvious denominator and is wrong here.
    `select_folded_examples` (and the renderer) require only that the `input`/`output`
    **keys** exist, and the renderer formats the values through an f-string -- so
    `{"input": 3, "output": 4}` *is* folded into the spec. `example_id` additionally
    requires both values be `str`, so it yields no id for it. Counting ids would
    therefore **under**-report what was published, which is the same class of wrong
    answer as over-reporting it.

    So the two numbers are deliberately different, and the manifest is honest about
    both: `examples_folded_into_spec` is what reached the spec, `folded_example_ids` is
    the subset that could be identified.
    """
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=8)
    out = tmp_path / "a.paw"
    examples = [{"input": "a", "output": "1"}, {"input": 3, "output": 4}]

    backend.compile("Classify.", examples, str(out))  # type: ignore[arg-type]

    manifest = json.loads(out.read_text())
    rendered = _rendered_inputs(sdk.compile_calls[0]["spec"])
    assert rendered == ["a", "3"], "the non-str example is rendered into the spec"
    assert manifest["examples_folded_into_spec"] == 2 == len(rendered)
    assert len(manifest["folded_example_ids"]) == 1, (
        "an unidentifiable example must not get a lineage id -- but it was still "
        "published, so it must still be counted"
    )


def test_folded_count_respects_the_cap_after_filtering_A_6(key: None, tmp_path: Path) -> None:
    """The cap applies to the *usable* examples, so malformed ones do not consume slots."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, max_spec_examples=2)
    out = tmp_path / "a.paw"
    examples = [{"bad": 1}, {"input": "a", "output": "1"}, {"input": "b", "output": "2"},
                {"input": "c", "output": "3"}]
    backend.compile("Classify.", examples, str(out))  # type: ignore[arg-type]
    assert _rendered_inputs(sdk.compile_calls[0]["spec"]) == ["a", "b"]
    assert json.loads(out.read_text())["examples_folded_into_spec"] == 2


def test_public_leak_warning_stops_firing_when_nothing_was_folded_A_6(
    key: None, tmp_path: Path
) -> None:
    """A-6's knock-on. The public-leak warning is gated on `folded_count > 0`, so an
    inflated count made it warn about publishing traced examples when none were
    published. A warning that fires when it should not is how a warning stops being
    read."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=True, max_spec_examples=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        backend.compile("Classify.", [{"bad": 1}, {"input": "x"}], str(tmp_path / "a.paw"))  # type: ignore[arg-type]


def test_renderer_and_selector_share_one_predicate_A_6() -> None:
    """Pattern 2, closed: `_render_spec_with_examples` used to re-implement
    `select_folded_examples`'s "usable" filter, so the count and the content could drift
    apart silently. They are now one predicate, and this pins that they agree -- including
    on the inputs where a hand-copied duplicate would most plausibly have diverged.
    """
    from paw_kit.backend.manifest_lineage import select_folded_examples

    cases: List[Any] = [
        [],
        [{"input": "a", "output": "b"}],
        [{"bad": 1}],
        [{"input": "a"}, {"output": "b"}],
        [{"input": None, "output": None}],
        [{"input": 3, "output": 4}],
        ["string", 7, None, {"input": "a", "output": "b"}],
        [{"input": "a", "output": "b", "extra": "kept"}],
    ]
    for examples in cases:
        for limit in (0, 1, 5):
            selected = select_folded_examples(examples, limit)
            rendered = _rendered_inputs(_render_spec_with_examples("S", examples, limit))
            assert len(rendered) == len(selected), (examples, limit)
            assert rendered == [f"{ex['input']}" for ex in selected], (examples, limit)


@pytest.mark.parametrize("raw_id", [None, "", 42, {"id": "x"}])
def test_cached_program_id_is_none_unless_the_precheck_named_a_real_one_A_2(
    key: None, tmp_path: Path, raw_id: object
) -> None:
    """`cached_program_id` is a *provenance* field, so a non-answer must read as None.

    `CompilePrecheck.program_id` is `str | None` upstream, and this value is written into
    the manifest and read back by measurement scripts. An empty string or a non-string
    that slipped through would be recorded as if it identified a program -- the same
    shape of defect as A-2 itself, one field down.

    (Kills `programasweights.py bool and->or  isinstance(raw_id, str) and raw_id`, which
    the gate-3 run found alive: under `or`, a truthy non-string is recorded verbatim and
    an empty string still reads as identified.)
    """
    class _PrecheckSDK(FakeSDK):
        def precheck_compile(self, spec: str, compiler: str | None = None):
            self.precheck_calls.append({"spec": spec, "compiler": compiler})
            return {"cached": True, "program_id": raw_id}

    sdk = _PrecheckSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk, public=False)
    out = tmp_path / "a.paw"
    with pytest.warns(UserWarning, match="already has a compiled program"):
        backend.compile("spec", [], str(out))
    assert json.loads(out.read_text())["cached_program_id"] is None


# ---------------------------------------------------------------- D-1 & D-2 tests


def test_d2_staleness_cross_process_recompile_resolves_new_program_id(key: None, tmp_path: Path) -> None:
    """D-2 staleness: a manifest rewritten at adapter_path by any other process or instance
    causes the next infer() to resolve and load the NEW program_id."""
    from paw_kit.atomicio import atomic_write_text

    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = str(tmp_path / "t.paw")
    backend.compile("spec v1", [], out)

    assert backend.infer(out, "in1") == "out(prog-fast):in1"
    assert len(sdk.function_calls) == 1
    assert sdk.function_calls[0]["program_id"] == "prog-fast"

    # Simulate another process/instance recompiling into the same manifest path:
    # rewrite the manifest via atomic_write_text (which does temp file + os.replace,
    # moving the inode as in production) with a new program_id.
    manifest_data = json.loads(Path(out).read_text(encoding="utf-8"))
    manifest_data["program_id"] = "prog-v2"
    atomic_write_text(out, json.dumps(manifest_data))

    assert backend.infer(out, "in2") == "out(prog-v2):in2"
    assert len(sdk.function_calls) == 2
    assert sdk.function_calls[1]["program_id"] == "prog-v2"


def test_d2_cost_same_program_id_does_not_reload(key: None, tmp_path: Path) -> None:
    """D-2 cost: a manifest rewrite that leaves program_id unchanged must NOT reload the model.

    This catches an over-eager fix that reloads on any stat change without checking program_id.
    """
    from paw_kit.atomicio import atomic_write_text

    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = str(tmp_path / "t.paw")
    backend.compile("spec v1", [], out)

    assert backend.infer(out, "in1") == "out(prog-fast):in1"
    assert len(sdk.function_calls) == 1

    # Rewrite the manifest with atomic_write_text (moving inode/mtime) but SAME program_id
    manifest_data = json.loads(Path(out).read_text(encoding="utf-8"))
    manifest_data["compile_wall_s"] = 99.9  # metadata change
    atomic_write_text(out, json.dumps(manifest_data))

    assert backend.infer(out, "in2") == "out(prog-fast):in2"
    # Cost check: function() was called exactly ONCE across both infers!
    assert len(sdk.function_calls) == 1


def test_d2_steady_state_no_reload(key: None, tmp_path: Path) -> None:
    """D-2 steady state: N infers against an untouched manifest invoke function() exactly once."""
    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out = str(tmp_path / "t.paw")
    backend.compile("spec", [], out)

    for i in range(10):
        assert backend.infer(out, f"x{i}") == f"out(prog-fast):x{i}"
    assert len(sdk.function_calls) == 1


def test_d2_eviction_safety_in_flight_caller_no_teardown(key: None, tmp_path: Path) -> None:
    """D-2 eviction safety (regression test for the no-teardown safety invariant F3.3):

    Eviction drops the dict reference and NOTHING ELSE. Never call close(), _cleanup_resources(),
    reset(), or any teardown method on an evicted callable.
    An in-flight caller's reference keeps the underlying llama.cpp model alive; closing it
    at eviction would cause a use-after-free.
    """
    import concurrent.futures
    from paw_kit.atomicio import atomic_write_text
    from paw_kit.backend.programasweights import _MAX_CACHED_FUNCTIONS

    teardown_invoked: List[Tuple[str, str]] = []

    class ClosableFunction:
        def __init__(self, program_id: str):
            self.program_id = program_id
            self.closed = False

        def __call__(self, text: str, **kwargs: Any) -> str:
            if self.closed:
                raise RuntimeError(f"USE-AFTER-FREE: callable for {self.program_id} was closed!")
            return f"out({self.program_id}):{text}"

        def close(self) -> None:
            self.closed = True
            teardown_invoked.append((self.program_id, "close"))

        def _cleanup_resources(self) -> None:
            self.closed = True
            teardown_invoked.append((self.program_id, "_cleanup_resources"))

    class TeardownTrackingSDK(FakeSDK):
        def function(self, program_id: str, **kw: Any):
            super().function(program_id, **kw)
            return ClosableFunction(program_id)

    sdk = TeardownTrackingSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    main_path = str(tmp_path / "main.paw")
    backend.compile("spec main", [], main_path)

    # Simulate an in-flight caller taking a reference to the callable under lock,
    # then releasing the lock (as infer() does before calling fn(...)).
    held_caller_fn = backend._get_function(main_path)
    assert not held_caller_fn.closed

    # Arm (a): Evict via invalidation (manifest rewritten with different program_id)
    # from another thread.
    manifest_data = json.loads(Path(main_path).read_text(encoding="utf-8"))
    manifest_data["program_id"] = "prog-new"
    atomic_write_text(main_path, json.dumps(manifest_data))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(backend.infer, main_path, "from_thread_a")
        res_other = fut.result(timeout=5)
        assert res_other == "out(prog-new):from_thread_a"

    # Now complete the held caller's call: it must succeed and teardown must NOT have fired.
    assert held_caller_fn("held_input_a") == "out(prog-fast):held_input_a"
    assert held_caller_fn.closed is False
    assert teardown_invoked == []

    # Arm (b): Evict via LRU overflow (drive _MAX_CACHED_FUNCTIONS + 1 distinct adapters)
    # from another thread.
    # First, re-acquire a held reference to current main_path function:
    held_caller_fn_b = backend._get_function(main_path)
    assert not held_caller_fn_b.closed

    def flood_lru() -> None:
        for idx in range(_MAX_CACHED_FUNCTIONS + 2):
            p = str(tmp_path / f"flood_{idx}.paw")
            backend.compile(f"spec flood {idx}", [], p)
            backend._get_function(p)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(flood_lru)
        fut.result(timeout=10)

    # Verify main_path was evicted from backend._functions
    with backend._lock:
        assert main_path not in backend._functions

    # Now complete the held caller's call: must succeed without exception, teardown never called!
    assert held_caller_fn_b("held_input_b") == "out(prog-new):held_input_b"
    assert held_caller_fn_b.closed is False
    assert teardown_invoked == []


def test_d2_fallback_chain_deleted_manifest_engages_teacher(key: None, tmp_path: Path) -> None:
    """D-2 fallback chain: deleting the manifest file between two served calls on a real
    @compile_on_hit-decorated, ready-state task engages the teacher and increments
    get_fail_open_count().

    This proves that the os.stat FileNotFoundError raise site falls open correctly end-to-end.
    """
    import os
    from paw_kit import compile_on_hit

    sdk = FakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    cache_dir = str(tmp_path / "paw_cache_d2_fallback")
    teacher_calls = 0

    @compile_on_hit(
        spec="D2 fallback task",
        threshold=1,
        cache_dir=cache_dir,
        backend=backend,
        sync_compile=True,
        shadow_window=0,
    )
    def svc(text: str) -> str:
        nonlocal teacher_calls
        teacher_calls += 1
        return f"teacher:{text}"

    # Call 1: traces and compiles, transitions to ready state
    out1 = svc("hello")
    assert out1 == "teacher:hello"
    assert teacher_calls == 1

    # Call 2: served by adapter via infer() -> _get_function
    out2 = svc("hello")
    assert out2 == "out(prog-fast):hello"
    assert teacher_calls == 1
    assert svc.get_fail_open_count() == 0

    # Delete the manifest file from disk
    adapter_path = svc.db.get_adapter_path(svc.task_id)  # type: ignore[attr-defined]
    os.remove(adapter_path)

    # Call 3: infer() hits os.stat -> FileNotFoundError -> fails open to teacher!
    out3 = svc("hello")
    assert out3 == "teacher:hello"
    assert teacher_calls == 2  # Teacher was engaged!
    assert svc.get_fail_open_count() == 1  # fail open count climbed!


def test_d1_honest_availability_and_doctor_routing(key: None, tmp_path: Path) -> None:
    """D-1: is_available() docstring clarifies import-only check, and inference failure
    re-raises / chains with an error message naming `paw-kit doctor`.
    """
    import inspect

    backend = ProgramAsWeightsBackend(sdk=FakeSDK())
    # 1. is_available docstring check
    doc = inspect.getdoc(backend.is_available)
    assert doc is not None
    assert "paw-kit doctor" in doc
    assert "import" in doc.lower()

    # 2. Module docstring check
    import paw_kit.backend.programasweights as mod
    mod_doc = inspect.getdoc(mod)
    assert mod_doc is not None
    assert "paw-kit doctor" in mod_doc
    assert "is_available()" in mod_doc

    # 3. Exception in _paw().function(...) chains and mentions doctor
    class BrokenFunctionSDK(FakeSDK):
        def function(self, program_id: str, **kw: Any):
            raise OSError("libllama.so: cannot open shared object file")

    broken_backend = ProgramAsWeightsBackend(sdk=BrokenFunctionSDK())
    out = str(tmp_path / "t.paw")
    broken_backend.compile("spec", [], out)

    with pytest.raises(RuntimeError, match="paw-kit doctor") as exc_info:
        broken_backend.infer(out, "test")
    assert isinstance(exc_info.value.__cause__, OSError)
    assert "libllama.so" in str(exc_info.value.__cause__)
    assert "llama_cpp" in str(exc_info.value)
    assert "GPU offload" in str(exc_info.value)

    # 4. Exception in fn(...) callable itself chains and mentions doctor
    class BrokenCallableSDK(FakeSDK):
        def function(self, program_id: str, **kw: Any):
            def bad_fn(text: str, **call_kw: Any) -> str:
                raise RuntimeError("llama_decode failed with code -1")

            return bad_fn

    broken_callable_backend = ProgramAsWeightsBackend(sdk=BrokenCallableSDK())
    out2 = str(tmp_path / "t2.paw")
    broken_callable_backend.compile("spec", [], out2)

    with pytest.raises(RuntimeError, match="paw-kit doctor") as exc_info2:
        broken_callable_backend.infer(out2, "test")
    assert isinstance(exc_info2.value.__cause__, RuntimeError)
    assert "llama_decode failed" in str(exc_info2.value.__cause__)


# ============================================================================
# Phase T: constrained-decoding-real-backend -- backend-level integration tests
#
# These exercise infer()/paw.load with a FAKE model that carries the minimal
# llama_cpp.Llama surface _build_vocabulary needs (n_vocab, token_eos, detokenize,
# tokenize) and a fake fn that applies logits_processor exactly like the real SDK's
# guarded_processor (runtime_llamacpp.py:504-523): catch, discard the poisoned token,
# re-raise unwrapped after the loop. Engine-present tests need llguidance importable
# (it is, in this dev checkout); the one engine-absent test (money route i) forces
# absence via `monkeypatch.setitem(sys.modules, "llguidance", None)` -- the same
# reproducible mechanism used throughout this suite for programasweights/llama_cpp
# (see tests/test_doctor.py) -- rather than assuming anything about the environment.
#
# `math.isfinite`, not numpy, is used to inspect a returned masked-scores array: this
# keeps every test in this file numpy-independent except where the real constraint
# engine itself is exercised (which already depends on numpy transitively, lazily, and
# only when llguidance is actually importable).
# ============================================================================

import logging
import math
import sys

from paw_kit import load
from paw_kit.schema.constraint import ConstraintUnavailable
from paw_kit.schema.exceptions import PAWSchemaError
from paw_kit.schema.grammar import pydantic_to_regex
from pydantic import BaseModel


class _CDSimpleModel(BaseModel):
    s: str


def _cd_build_tokens() -> "Tuple[List[bytes], int]":
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    tokens.append(b"\xc3\xa9")  # dedicated 2-byte token, unused by these tests directly
    eos = len(tokens)
    tokens.append(b"<eos>")
    return tokens, eos


class _FakeLlamaModel:
    """Minimal llama_cpp.Llama-shaped fake: exactly the surface
    `ProgramAsWeightsBackend._build_vocabulary` and a driven generation loop need."""

    def __init__(self) -> None:
        self.tokens, self.eos = _cd_build_tokens()
        self._by_bytes: Dict[bytes, int] = {}
        for i, b in enumerate(self.tokens):
            if b and b not in self._by_bytes:
                self._by_bytes[b] = i
        self._maxlen = max(len(b) for b in self._by_bytes)

    def n_vocab(self) -> int:
        return len(self.tokens)

    def token_eos(self) -> int:
        return self.eos

    def detokenize(self, ids: "List[int]") -> bytes:
        return b"".join(self.tokens[i] for i in ids)

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> "List[int]":
        out: "List[int]" = []
        i = 0
        while i < len(data):
            for ln in range(min(self._maxlen, len(data) - i), 0, -1):
                tid = self._by_bytes.get(data[i:i + ln])
                if tid is not None:
                    out.append(tid)
                    i += ln
                    break
            else:  # pragma: no cover -- unreachable, every single byte has a token
                i += 1
        return out

    def encode(self, x: "str | bytes") -> "List[int]":
        if isinstance(x, str):
            x = x.encode("utf-8")
        return self.tokenize(x)


class _ConstrainedFakeFn:
    """Simulates `PawFunction.__call__`/`_generate`'s shape closely enough for Phase
    T's masking-effect and prompt-offset assertions: applies `logits_processor` at
    every generation step, picks the scripted token when the mask allows it (else the
    lowest allowed id, i.e. a "model" that mostly cooperates with the grammar), and
    propagates a processor exception unwrapped after the loop -- exactly like
    `runtime_llamacpp.py`'s `guarded_processor` (catch, discard the poisoned token,
    re-raise once control returns to Python).
    """

    def __init__(self, llm: _FakeLlamaModel, scripted_output: str) -> None:
        self._llm = llm
        self._scripted_ids = llm.encode(scripted_output)
        self.last_processor: Any = None

    def __call__(
        self,
        input_text: str,
        max_tokens: "int | None" = None,
        temperature: float = 0.0,
        logits_processor: Any = None,
    ) -> str:
        context = list(self._llm.encode(input_text))
        processor = logits_processor[0] if logits_processor else None
        self.last_processor = processor
        n_vocab = self._llm.n_vocab()
        limit = max_tokens if max_tokens is not None else len(self._scripted_ids) + 1
        output: "List[int]" = []
        processor_error: "BaseException | None" = None
        for i in range(limit):
            scores = [0.0] * n_vocab
            if processor is not None:
                try:
                    masked = processor(context, scores)
                except BaseException as exc:  # noqa: BLE001 -- mirrors guarded_processor
                    processor_error = exc
                    break
                allowed = [idx for idx, v in enumerate(masked) if math.isfinite(v)]
                if i < len(self._scripted_ids) and self._scripted_ids[i] in allowed:
                    token = self._scripted_ids[i]
                else:
                    token = allowed[0] if allowed else self._llm.eos
            else:
                token = self._scripted_ids[i] if i < len(self._scripted_ids) else self._llm.eos
            if token == self._llm.eos:
                break
            output.append(token)
            context.append(token)
        if processor_error is not None:
            raise processor_error
        return self._llm.detokenize(output).decode("utf-8", errors="replace")


class _AlwaysRejectingFakeFn:
    """A `fn` whose "model" ignores the mask entirely and emits a fixed, grammar-
    incompatible token sequence every time -- so the constraint's `consume_token()`
    fails on every call, deterministically. Used for money route (iii-b): S-14's
    fallback-counting and warn-once mitigation for a constraint that fails on every
    single call."""

    def __init__(self, llm: _FakeLlamaModel, bad_output: str = "not json at all") -> None:
        self._llm = llm
        self._bad_ids = llm.encode(bad_output)

    def __call__(
        self,
        input_text: str,
        max_tokens: "int | None" = None,
        temperature: float = 0.0,
        logits_processor: Any = None,
    ) -> str:
        context = list(self._llm.encode(input_text))
        processor = logits_processor[0] if logits_processor else None
        output: "List[int]" = []
        processor_error: "BaseException | None" = None
        for tok in self._bad_ids:
            if processor is not None:
                try:
                    processor(context, [0.0] * self._llm.n_vocab())
                except BaseException as exc:  # noqa: BLE001 -- mirrors guarded_processor
                    processor_error = exc
                    break
            output.append(tok)
            context.append(tok)
        if processor_error is not None:
            raise processor_error
        return self._llm.detokenize(output).decode("utf-8", errors="replace")


class ConstrainedFakeSDK(FakeSDK):
    """A FakeSDK whose `function()` returns a callable carrying a real (fake)
    `_llm`, so `_build_vocabulary` succeeds and grammar-constrained decoding actually
    engages -- unlike the base `FakeSDK`, whose plain closure has no `_llm` at all
    (that absence is what money route iii-a's own test already covers)."""

    def __init__(self, scripted_output: str = '{"s": "hi"}', **kw: Any) -> None:
        super().__init__(**kw)
        self.llm = _FakeLlamaModel()
        self.fn = _ConstrainedFakeFn(self.llm, scripted_output)

    def function(self, program_id: str, **kw: Any):
        self.function_calls.append({"program_id": program_id, **kw})
        return self.fn


class RejectingFakeSDK(FakeSDK):
    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.llm = _FakeLlamaModel()
        self.fn = _AlwaysRejectingFakeFn(self.llm)

    def function(self, program_id: str, **kw: Any):
        self.function_calls.append({"program_id": program_id, **kw})
        return self.fn


def test_cd_masking_effect_end_to_end_via_infer(key: None, tmp_path: Path) -> None:
    """A masking effect, not merely a successful generation (per the track's
    safety invariants: the `Llama.sample()` reuse hazard means every "constraint
    applied" assertion here is on invocation/masked-count, not generation success)."""
    pytest.importorskip("llguidance")
    sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)
    assert backend.applies_grammar_constraint is True

    pat = pydantic_to_regex(_CDSimpleModel)
    result = backend.infer(out_path, "hello", grammar_constraint=pat)

    assert result != ""
    processor = sdk.fn.last_processor
    assert processor is not None
    assert processor.invocations > 0
    assert len(processor.masked_per_step) == processor.invocations
    assert all(m > 0 for m in processor.masked_per_step)


def test_cd_prompt_offset_full_generation_returns_local_output_no_fallback(
    key: None, tmp_path: Path
) -> None:
    """H-2: a full generation through a fake fn that passes the prompt (the SDK's own
    loop, simulated by `_ConstrainedFakeFn`) returns LOCAL output with no fallback
    call -- the prompt-offset bookkeeping does not mistake prompt tokens for illegal
    generated ones."""
    pytest.importorskip("llguidance")
    sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)

    fallback_calls: List[str] = []

    def fallback(text: str) -> _CDSimpleModel:
        fallback_calls.append(text)
        return _CDSimpleModel(s="FALLBACK")

    fn = load(out_path, _CDSimpleModel, backend=backend, fallback_provider=fallback)
    result = fn("a reasonably long unrelated prompt, so the offset is not trivially zero")

    assert result == _CDSimpleModel(s="hi")
    assert fallback_calls == []


def test_cd_money_route_i_engine_absent_flag_false_warns_once_no_teacher_call(
    monkeypatch: pytest.MonkeyPatch, key: None, tmp_path: Path
) -> None:
    """Money route (i): forcing llguidance absent via
    `monkeypatch.setitem(sys.modules, "llguidance", None)` (the same mechanism
    `tests/test_doctor.py` uses for `programasweights`/`llama_cpp`; verified this
    makes both `importlib.util.find_spec` and a plain `import` fail). The instance
    flag goes False, exactly one UserWarning fires at construction naming the `paw`
    extra, and `paw.load` calls `infer()` with `grammar_constraint=None` -- returning
    the LOCAL (unconstrained) output, never the teacher's."""
    monkeypatch.setitem(sys.modules, "llguidance", None)

    with pytest.warns(UserWarning, match="llguidance"):
        sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
        backend = ProgramAsWeightsBackend(sdk=sdk)
    assert backend.applies_grammar_constraint is False

    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)

    teacher_calls: List[str] = []

    def fallback(text: str) -> _CDSimpleModel:
        teacher_calls.append(text)
        return _CDSimpleModel(s="teacher")

    with pytest.warns(UserWarning, match="cannot apply grammar_constraint"):
        fn = load(out_path, _CDSimpleModel, backend=backend, fallback_provider=fallback)
    result = fn("hello")

    assert result == _CDSimpleModel(s="hi")  # local output
    assert teacher_calls == []
    assert sdk.fn.last_processor is None  # no logits_processor was ever built/passed


class _BrokenLLGuidanceModule:
    """A stand-in for `sys.modules["llguidance"]` that is present (so `import
    llguidance` trivially succeeds -- Python finds it already in `sys.modules` and
    never re-imports it) but raises the moment anything on it is touched, simulating
    an ABI-mismatched or partially-installed native extension (P-2): a real
    `llguidance` package whose `.so` cannot actually be used. `find_spec("llguidance")`
    is called only once, at `ProgramAsWeightsBackend.__init__`, before this object is
    installed in the tests below -- so it sees the real (working) package and returns
    a real spec, exactly as it would on a machine where the engine LOOKS installed."""

    def __getattr__(self, name: str) -> Any:
        raise ImportError(f"llguidance.{name} is unavailable (simulated broken install)")


def test_cd_p2_present_but_unimportable_engine_never_pays_the_teacher(
    monkeypatch: pytest.MonkeyPatch, key: None, tmp_path: Path
) -> None:
    """P-2: a present-but-unimportable engine must not leave `applies_grammar_constraint`
    True while every call quietly pays the teacher. The engine is genuinely importable
    at construction (so `find_spec` succeeds and no route-(i) warning fires and the flag
    starts True), then broken -- via `monkeypatch.setitem(sys.modules, "llguidance",
    _BrokenLLGuidanceModule())` -- before the first model load, so
    `_get_function_and_vocabulary`'s forced `vocabulary.llguidance_tokenizer()` call
    (the P-2 fix) is what discovers it, not `build_constraint`. Three `paw.load`-bound
    calls: zero fallback/teacher calls, all three return the LOCAL output, exactly one
    UserWarning total, and the flag ends False -- the same shape as money route (iii),
    because that is the route this fix lands the failure in."""
    pytest.importorskip("llguidance")
    sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
    backend = ProgramAsWeightsBackend(sdk=sdk)
    assert backend.applies_grammar_constraint is True  # real engine at construction

    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)

    # Break the engine only NOW -- after construction/compile, so find_spec already
    # saw the real package. This reproduces "present but unimportable", not "absent".
    monkeypatch.setitem(sys.modules, "llguidance", _BrokenLLGuidanceModule())

    teacher_calls: List[str] = []

    def fallback(text: str) -> _CDSimpleModel:
        teacher_calls.append(text)
        return _CDSimpleModel(s="teacher")

    fn = load(out_path, _CDSimpleModel, backend=backend, fallback_provider=fallback)

    with pytest.warns(UserWarning, match="grammar-constraint vocabulary"):
        result0 = fn("hello 0")
    assert result0 == _CDSimpleModel(s="hi")

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second/third warning would now raise
        result1 = fn("hello 1")
        result2 = fn("hello 2")
    assert result1 == _CDSimpleModel(s="hi")
    assert result2 == _CDSimpleModel(s="hi")

    assert teacher_calls == []  # zero fallback calls across all three
    assert backend.applies_grammar_constraint is False
    assert sdk.fn.last_processor is None  # no logits_processor was ever built/passed


def test_cd_money_route_iii_b_constraint_raising_every_call_counted_and_warned_once(
    key: None, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """S-14 / money route (iii-b): a constraint that raises on every single call is
    counted by `get_local_fallback_count()` and logged (via `logger.warning`, not
    `warnings.warn` -- `caplog`, not `pytest.warns`) exactly once, with later
    occurrences dropping to DEBUG."""
    pytest.importorskip("llguidance")
    sdk = RejectingFakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)
    assert backend.applies_grammar_constraint is True

    teacher_calls: List[str] = []

    def fallback(text: str) -> _CDSimpleModel:
        teacher_calls.append(text)
        return _CDSimpleModel(s="teacher")

    fn = load(out_path, _CDSimpleModel, backend=backend, fallback_provider=fallback)

    with caplog.at_level(logging.WARNING, logger="paw_kit.schema.loader"):
        for text in ("a", "b", "c"):
            result = fn(text)
            assert result == _CDSimpleModel(s="teacher")

    assert teacher_calls == ["a", "b", "c"]
    assert fn.get_local_fallback_count() == 3
    warning_records = [
        r for r in caplog.records
        if r.name == "paw_kit.schema.loader" and r.levelno == logging.WARNING
    ]
    assert len(warning_records) == 1


def test_cd_paw_schema_error_reaches_infer_unwrapped_not_runtimeerror(
    key: None, tmp_path: Path
) -> None:
    """The safety invariant this whole track hinges the remedy-string rewrite on:
    `infer()` lets `PAWSchemaError` from the constraint through its `RuntimeError`
    wrapper unwrapped, so a grammar/masking failure is never mistaken for (and never
    sent to check) a GPU-offload/environment problem."""
    pytest.importorskip("llguidance")
    sdk = RejectingFakeSDK()
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)

    pat = pydantic_to_regex(_CDSimpleModel)
    with pytest.raises(PAWSchemaError) as excinfo:
        backend.infer(out_path, "hello", grammar_constraint=pat)
    assert "GPU offload" not in str(excinfo.value)
    assert "paw-kit doctor" not in str(excinfo.value)
    # This is a mid-generation failure (consume_token() rejects a sampled token), not a
    # construction-time refusal -- so it must be plain PAWSchemaError, not the
    # ConstraintUnavailable subclass infer() specifically catches to degrade per-schema
    # (Change 2).
    assert not isinstance(excinfo.value, ConstraintUnavailable)
    assert type(excinfo.value) is PAWSchemaError


def _make_over_budget_pattern(n_fields: int) -> str:
    """A pattern `build_constraint` refuses AT CONSTRUCTION with `ConstraintUnavailable`
    (fuel exceeded): `n_fields` `Field(ge=0, le=255)` int fields cost a flat ~2,574 fuel
    each past the first (Change 1's re-derivation, `constraint.py`'s module docstring),
    so 39+ fields exceed `INITIAL_LEXER_FUEL=100_000`."""
    from pydantic import Field, create_model

    model = create_model(
        f"OverBudget{n_fields}",
        **{f"f{i}": (int, Field(ge=0, le=255)) for i in range(n_fields)},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pydantic_to_regex(model)


def test_cd_construction_time_refusal_degrades_per_schema_warns_once_across_calls(
    key: None, tmp_path: Path
) -> None:
    """Change 2: a grammar refused AT CONSTRUCTION (`ConstraintUnavailable` -- fuel
    exceeded here) degrades PER SCHEMA, not per call and not per instance: exactly one
    UserWarning across three calls with the SAME over-budget pattern, `infer()` returns
    the LOCAL output every time (never raises, never reaches a teacher/fallback), `fn`
    is called with NO `logits_processor`, and `applies_grammar_constraint` stays `True`
    throughout -- the engine works, only this one grammar is refused."""
    pytest.importorskip("llguidance")
    sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)
    assert backend.applies_grammar_constraint is True

    pat = _make_over_budget_pattern(40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            result = backend.infer(out_path, "hello", grammar_constraint=pat)
            assert result != ""
            assert sdk.fn.last_processor is None

    matching = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "grammar-constrained decoding is unavailable" in str(w.message)
    ]
    assert len(matching) == 1, f"expected exactly one warning, got {len(matching)}: {matching}"
    assert backend.applies_grammar_constraint is True


def test_cd_construction_time_refusal_different_pattern_warns_again_once(
    key: None, tmp_path: Path
) -> None:
    """A DIFFERENT refused pattern gets its own warning: the warn-once set is keyed
    per-grammar (a hash of the pattern text), not a single instance-wide latch."""
    pytest.importorskip("llguidance")
    sdk = ConstrainedFakeSDK(scripted_output='{"s": "hi"}')
    backend = ProgramAsWeightsBackend(sdk=sdk)
    out_path = str(tmp_path / "t.paw")
    backend.compile("spec", [], out_path)

    pat1 = _make_over_budget_pattern(40)
    pat2 = _make_over_budget_pattern(45)
    assert pat1 != pat2

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        backend.infer(out_path, "hello", grammar_constraint=pat1)
        backend.infer(out_path, "hello", grammar_constraint=pat2)

    matching = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and "grammar-constrained decoding is unavailable" in str(w.message)
    ]
    assert len(matching) == 2, f"expected two distinct-grammar warnings, got {matching}"
    assert backend.applies_grammar_constraint is True


def test_cd_construction_time_refusal_engine_level_is_constraint_unavailable() -> None:
    """Engine-level check (no backend involved): `build_constraint` itself raises
    `ConstraintUnavailable` -- an instance of `PAWSchemaError` -- on an over-budget
    grammar."""
    pytest.importorskip("llguidance")
    from paw_kit.schema.constraint import Vocabulary, build_constraint

    llm = _FakeLlamaModel()
    vocab = Vocabulary(
        tokens=llm.tokens,
        eos_token_id=llm.eos,
        special_token_ids=(llm.eos,),
        encode=llm.encode,
    )
    pat = _make_over_budget_pattern(40)
    with pytest.raises(ConstraintUnavailable):
        build_constraint(pat, vocab)
    # And it IS a PAWSchemaError, per Change 2's contract.
    try:
        build_constraint(pat, vocab)
    except ConstraintUnavailable as exc:
        assert isinstance(exc, PAWSchemaError)
