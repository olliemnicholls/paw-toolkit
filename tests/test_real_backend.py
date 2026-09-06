"""Unit tests for RealPAWBackend hardware bridge protocol and error handling."""

from pathlib import Path
import pytest
from paw_kit import AbstractPAWBackend
from paw_kit.backend.real import RealPAWBackend


def test_real_backend_protocol_conformance() -> None:
    """Verify RealPAWBackend conforms to AbstractPAWBackend ABC."""
    backend = RealPAWBackend()
    assert isinstance(backend, AbstractPAWBackend)


def test_real_backend_availability_and_runtime_error(tmp_path: Path) -> None:
    """Verify real backend raises descriptive RuntimeError when PyTorch environment is missing."""
    backend = RealPAWBackend()
    adapter_path = str(tmp_path / "real_model.paw")

    # If PyTorch / transformers are not installed in CI environment
    if not backend.is_available():
        with pytest.raises(RuntimeError, match="paw-kit\\[torch\\]"):
            backend.compile(spec="Test", examples=[], output_path=adapter_path)

        with pytest.raises(RuntimeError, match="paw-kit\\[torch\\]"):
            backend.infer(adapter_path=adapter_path, input_text="Hello")


def test_real_backend_not_implemented_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify real backend raises NotImplementedError (instead of returning dummy fake strings) when available."""
    backend = RealPAWBackend()
    monkeypatch.setattr(backend, "is_available", lambda: True)

    adapter_path = str(tmp_path / "test.paw")
    with pytest.raises(NotImplementedError, match="under active development for paw-kit v0.2"):
        backend.compile(spec="Test spec", examples=[], output_path=adapter_path)

    with pytest.raises(NotImplementedError, match="under active development for paw-kit v0.2"):
        backend.infer(adapter_path=adapter_path, input_text="Hello world")


def test_cli_backend_resolution_fallback() -> None:
    """Verify CLI _resolve_cli_backend gracefully warns and falls back to MockPAWBackend."""
    from paw_kit.cli import _resolve_cli_backend
    from paw_kit.backend.mock import MockPAWBackend

    backend = _resolve_cli_backend("real")
    assert isinstance(backend, MockPAWBackend)

    backend_mock = _resolve_cli_backend("mock")
    assert isinstance(backend_mock, MockPAWBackend)


def test_real_backend_custom_runtime_executor(tmp_path: Path) -> None:
    """Verify RealPAWBackend works with runtime_executor bridge."""
    def mock_runtime_executor(adapter_path: str, input_text: str, constraint: str | None) -> str:
        return f"custom_executor:{input_text}:constraint={bool(constraint)}"

    backend = RealPAWBackend(runtime_executor=mock_runtime_executor)
    assert backend.is_available() is True

    adapter_path = str(tmp_path / "custom.paw")
    backend.compile(spec="Test Spec", examples=[], output_path=adapter_path)
    assert Path(adapter_path).exists()

    result = backend.infer(adapter_path=adapter_path, input_text="ping", grammar_constraint="pattern")
    assert result == "custom_executor:ping:constraint=True"


def test_real_backend_accepts_valid_hf_repo_id_PAW_BACKEND_02() -> None:
    """Verify a well-formed 'owner/repo-name' Hugging Face repo ID is accepted."""
    backend = RealPAWBackend(base_model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    assert backend.base_model_name_or_path == "Qwen/Qwen2.5-0.5B-Instruct"

    backend2 = RealPAWBackend(base_model_name_or_path="gpt2")
    assert backend2.base_model_name_or_path == "gpt2"


def test_real_backend_accepts_existing_local_directory_PAW_BACKEND_02(tmp_path: Path) -> None:
    """Verify an existing local directory is accepted even if it doesn't look like a
    repo ID (e.g. an absolute path to a locally fine-tuned checkpoint)."""
    local_model_dir = tmp_path / "my local checkpoint!!"
    local_model_dir.mkdir()

    backend = RealPAWBackend(base_model_name_or_path=str(local_model_dir))
    assert backend.base_model_name_or_path == str(local_model_dir)


def test_real_backend_rejects_invalid_base_model_name_PAW_BACKEND_02(tmp_path: Path) -> None:
    """Verify a value that is neither an existing local directory nor a well-formed
    HF repo ID is rejected at construction, not silently stored and never validated."""
    for bad_value in ("../../etc/passwd", "; rm -rf /", str(tmp_path / "does_not_exist")):
        with pytest.raises(ValueError, match="base_model_name_or_path"):
            RealPAWBackend(base_model_name_or_path=bad_value)
