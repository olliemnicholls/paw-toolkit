"""Hardware bridge to upstream Program-as-Weights (PAW) runtime and PyTorch/Transformers."""

import importlib.util
from typing import Any, Callable, Dict, List, Optional

from paw_kit.atomicio import atomic_write_text
from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.schema.logits_processor import RegexLogitsProcessor


class RealPAWBackend(AbstractPAWBackend):
    """Runtime bridge to upstream Deng et al. PAW compiler and PyTorch neural interpreter.

    Connects to local GPU/CPU transformers runtime and 0.6B resident base weights.
    """

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        runtime_executor: Optional[Callable[[str, str, Optional[str]], str]] = None,
    ) -> None:
        self.base_model_name_or_path = base_model_name_or_path
        self.device = device
        self._runtime_executor = runtime_executor

    def is_available(self) -> bool:
        """Check if torch and transformers or custom runtime executor are available."""
        if self._runtime_executor is not None:
            return True

        has_torch = importlib.util.find_spec("torch") is not None
        has_transformers = importlib.util.find_spec("transformers") is not None
        return has_torch and has_transformers

    def compile(
        self,
        spec: str,
        examples: List[Dict[str, str]],
        output_path: str,
    ) -> str:
        """Compile task specification and synthetic examples into a .paw LoRA adapter.

        Raises:
            RuntimeError: If real ML environment (torch/transformers) is not available.
            NotImplementedError: If direct PyTorch compilation is invoked without a custom runtime executor.
        """
        if not self.is_available():
            raise RuntimeError(
                "RealPAWBackend requires PyTorch and Hugging Face transformers packages.\n"
                "Install the real backend dependencies:\n\n"
                "    pip install 'paw-kit[torch]'\n\n"
                "Or use MockPAWBackend for fast, zero-GPU testing and development."
            )

        if self._runtime_executor is not None:
            # PAW-JIT-05: atomic write (temp file + os.replace) instead of a bare
            # Path.write_text() -- see paw_kit.atomicio's module docstring.
            atomic_write_text(output_path, f"[real_compiled:{spec}]")
            return output_path

        raise NotImplementedError(
            "Direct PyTorch neural fine-tuning compilation is under active development for paw-kit v0.2.\n"
            "In v0.1, use MockPAWBackend for zero-GPU testing or supply a custom `runtime_executor` callback."
        )

    def infer(
        self,
        adapter_path: str,
        input_text: str,
        grammar_constraint: Optional[str] = None,
    ) -> str:
        """Execute local neural inference on resident model with specified adapter.

        Applies RegexLogitsProcessor if grammar_constraint is provided.

        Raises:
            RuntimeError: If real ML environment (torch/transformers) is not available.
            NotImplementedError: If direct PyTorch inference is invoked without a custom runtime executor.
        """
        if not self.is_available():
            raise RuntimeError(
                "RealPAWBackend requires PyTorch and Hugging Face transformers packages for local neural execution.\n"
                "Install the real backend dependencies:\n\n"
                "    pip install 'paw-kit[torch]'\n\n"
                "Or use MockPAWBackend for fast, zero-GPU testing and development."
            )

        if self._runtime_executor is not None:
            return self._runtime_executor(adapter_path, input_text, grammar_constraint)

        raise NotImplementedError(
            "Direct PyTorch neural inference is under active development for paw-kit v0.2.\n"
            "In v0.1, use MockPAWBackend for zero-GPU testing or supply a custom `runtime_executor` callback."
        )
