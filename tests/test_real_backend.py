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
        with pytest.raises(RuntimeError, match="requires PyTorch and transformers"):
            backend.compile(spec="Test", examples=[], output_path=adapter_path)

        with pytest.raises(RuntimeError, match="requires PyTorch and transformers"):
            backend.infer(adapter_path=adapter_path, input_text="Hello")


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
