"""Deterministic pure-Python mock backend for GPU-free testing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from paw_kit.backend.base import AbstractPAWBackend


class MockPAWBackend(AbstractPAWBackend):
    """Pure-Python mock backend providing deterministic compilation and inference in <10ms.

    Enables 100% unit and integration test coverage without GPU hardware or real weights.
    """

    def __init__(self) -> None:
        self._adapters: Dict[str, Dict[str, Any]] = {}

    def compile(
        self,
        spec: str,
        examples: List[Dict[str, str]],
        output_path: str,
    ) -> str:
        """Simulate fast in-memory compilation and write lightweight metadata artifact.

        Args:
            spec: Natural language task specification.
            examples: Training/demonstration example pairs.
            output_path: Destination path for the simulated .paw adapter.

        Returns:
            The path to the created mock adapter artifact.
        """
        adapter_data = {
            "spec": spec,
            "examples": examples,
            "examples_count": len(examples),
            "backend": "mock",
            "rules": {},
            "default_response": None,
        }
        self._adapters[output_path] = adapter_data

        # Ensure directory exists and write simulated artifact to disk
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(adapter_data, f, indent=2)

        return output_path

    def infer(
        self,
        adapter_path: str,
        input_text: str,
        grammar_constraint: Optional[str] = None,
    ) -> str:
        """Run simulated deterministic inference.

        Checks registered rules first, then training examples, then custom default,
        and finally falls back to a deterministic string.

        Args:
            adapter_path: Path to the .paw adapter artifact.
            input_text: Prompt or input payload.
            grammar_constraint: Optional constraint specification.

        Returns:
            Deterministic simulated response string.
        """
        adapter = self._adapters.get(adapter_path)
        if adapter is None and Path(adapter_path).exists():
            # Load metadata from disk if present
            try:
                with open(adapter_path, "r", encoding="utf-8") as f:
                    adapter = json.load(f)
                    self._adapters[adapter_path] = adapter
            except Exception:
                adapter = None

        if adapter:
            # 1. Exact rule match
            rules = adapter.get("rules", {})
            if input_text in rules:
                return rules[input_text]

            # 2. Check examples match
            for ex in adapter.get("examples", []):
                if ex.get("input") == input_text and "output" in ex:
                    return ex["output"]

            # 3. Custom default response
            if adapter.get("default_response") is not None:
                return adapter["default_response"]

        # 4. Fallback deterministic output
        return f"[mock:{input_text}]"

    def is_available(self) -> bool:
        """Check availability. Always True for pure-Python mock backend."""
        return True

    def register_rule(self, adapter_path: str, input_text: str, output: str) -> None:
        """Register a canned deterministic response for a specific input."""
        if adapter_path not in self._adapters:
            self._adapters[adapter_path] = {
                "spec": "",
                "examples": [],
                "rules": {},
                "default_response": None,
            }
        self._adapters[adapter_path]["rules"][input_text] = output

    def set_default_response(self, adapter_path: str, output: str) -> None:
        """Set a default response for an adapter when no rule or example matches."""
        if adapter_path not in self._adapters:
            self._adapters[adapter_path] = {
                "spec": "",
                "examples": [],
                "rules": {},
                "default_response": None,
            }
        self._adapters[adapter_path]["default_response"] = output

    def get_adapter(self, adapter_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve in-memory adapter state."""
        return self._adapters.get(adapter_path)

    def reset(self) -> None:
        """Clear all registered mock adapters and rules."""
        self._adapters.clear()
