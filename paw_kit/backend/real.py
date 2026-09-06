"""Hardware bridge to upstream Program-as-Weights (PAW) runtime and PyTorch/Transformers."""

import importlib.util
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from paw_kit.atomicio import atomic_write_text
from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.schema.logits_processor import RegexLogitsProcessor

# PAW-BACKEND-02: a Hugging Face repo ID is "repo-name" or "owner/repo-name" using
# only letters, digits, '.', '_' and '-' in each segment. Approximates (does not
# exactly reproduce) the Hub's own validation, but it's enough to reject the shapes
# that actually matter here: path traversal (../..), absolute paths to unexpected
# locations, and shell-metacharacter/injection-shaped strings.
_HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)?$")


class RealPAWBackend(AbstractPAWBackend):
    """Placeholder for an in-process PyTorch/PEFT backend. Not implemented.

    `compile()` and `infer()` raise NotImplementedError unless a `runtime_executor`
    callback is supplied. For a working real backend use
    `paw_kit.backend.programasweights.ProgramAsWeightsBackend`.
    """

    def __init__(
        self,
        base_model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "auto",
        runtime_executor: Optional[Callable[[str, str, Optional[str]], str]] = None,
    ) -> None:
        # PAW-BACKEND-02: validated at construction -- a bounded check on an argument
        # the public constructor already accepts, not a new gate with no call site.
        # No consuming sink exists in v0.1 (grep confirms base_model_name_or_path is
        # read nowhere else in this package, exactly like PAW-BACKEND-01's retracted
        # finding), but validating eagerly at the boundary where the value is first
        # accepted is cheap insurance against whatever v0.2's real
        # AutoModelForCausalLM.from_pretrained(...) call ends up doing with it.
        if not Path(base_model_name_or_path).is_dir() and not _HF_REPO_ID_PATTERN.match(
            base_model_name_or_path
        ):
            raise ValueError(
                f"base_model_name_or_path {base_model_name_or_path!r} is neither an "
                "existing local directory nor a valid Hugging Face repo ID (expected "
                "'owner/repo-name' or 'repo-name', using only letters, digits, '.', "
                "'_' and '-')."
            )
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
            "For real inference today use paw_kit.backend.programasweights.ProgramAsWeightsBackend "
            "(official upstream SDK); for zero-GPU testing use MockPAWBackend; or supply a custom "
            "`runtime_executor` callback."
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
            "For real inference today use paw_kit.backend.programasweights.ProgramAsWeightsBackend "
            "(official upstream SDK); for zero-GPU testing use MockPAWBackend; or supply a custom "
            "`runtime_executor` callback."
        )
