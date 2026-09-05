"""Abstract base class for PAW backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class AbstractPAWBackend(ABC):
    """Protocol interface for PAW compilation and inference backends."""

    @abstractmethod
    def compile(
        self,
        spec: str,
        examples: List[Dict[str, str]],
        output_path: str,
    ) -> str:
        """Compile a specification and dataset into a .paw adapter artifact.

        Args:
            spec: Natural language task specification.
            examples: Synthetic or traced demonstration pairs (e.g. [{"input": ..., "output": ...}]).
            output_path: Destination filesystem path for the compiled adapter artifact.

        Returns:
            The path to the compiled adapter artifact.
        """
        pass

    @abstractmethod
    def infer(
        self,
        adapter_path: str,
        input_text: str,
        grammar_constraint: Optional[str] = None,
    ) -> str:
        """Run inference using the shared base interpreter and specified adapter.

        Args:
            adapter_path: Path to the compiled .paw adapter artifact.
            input_text: Input prompt or payload for the neural function.
            grammar_constraint: Optional regex or CFG grammar string constraining decoding.

        Returns:
            Generated output text adhering to the specification/grammar constraint.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend runtime requirements (hardware, weights, dependencies) are met.

        Returns:
            True if the backend is ready for compilation and inference, False otherwise.
        """
        pass
