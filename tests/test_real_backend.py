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
