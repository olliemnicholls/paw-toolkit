"""Deterministic pure-Python mock backend for GPU-free testing."""

from collections import OrderedDict
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from paw_kit.atomicio import atomic_write_text
from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.manifest_lineage import (
    append_history_entry,
    folded_example_ids,
    read_parent_lineage,
    sha256_text,
)

# Mirrors programasweights.MANIFEST_VERSION -- see that module's constant for what
# bumped it. The mock backend has no upstream `program_id`/`compiler`/`compiled_at`
# concept of its own, so those three fields stay absent here; everything else in the
# v2 lineage schema (spec hash, folded example ids, parent linkage, compile wall
# time, compiler_snapshot) applies equally to a mock manifest.
MANIFEST_VERSION = 2

# PAW-BACKEND-03: caps how many distinct adapter_path entries MockPAWBackend keeps
# resident in memory at once, evicted least-recently-used. Without this, a
# long-running process that compiles/infers against many distinct adapter paths over
# its lifetime (e.g. one .paw file per task_id, as paw_kit.jit.decorator generates)
# retains every one of them forever.
_MAX_CACHED_ADAPTERS = 256

# PAW-BACKEND-04: size cap on a reloaded adapter JSON file, checked before json.load
# reads the whole thing into memory -- mirrors paw_kit.cli's PAW-CLI-06 fix for the
# same underlying risk (an unbounded read/parse attempt against an arbitrarily large
# or crafted file).
_MAX_ADAPTER_FILE_BYTES = 50 * 1024 * 1024


class MockAdapterMiss(LookupError):
    """J-3: raised by `MockPAWBackend.infer` when `strict_misses=True` and no
    rule, example or `default_response` matched the input.

    Opt-in (default `strict_misses=False`, preserving the `[mock:...]` sentinel
    exactly): the sentinel is a live dependency of `serve`'s warm-up liveness
    check and Track B's CLI reporting/goldens, so it cannot change process-wide.
    A caller who wants "an unmemorised input is a hard failure, not a plausible
    wrong value silently served" -- the shape J-3 (bug-hunt-2026-09-11) is about
    -- opts in explicitly instead.
    """


def _is_valid_adapter_shape(data: Any) -> bool:
    """PAW-BACKEND-04: shape-validate a reloaded adapter JSON file before trusting
    `examples`/`rules` have the structure `infer()` assumes -- a bare list of
    `{"input": ..., "output": ...}` dicts, and a bare string-to-string mapping,
    respectively. A crafted or corrupted file could otherwise reach `.get()`/iteration
    calls against data of an unexpected type deep inside `infer()`'s matching logic.
    """
    if not isinstance(data, dict):
        return False
    examples = data.get("examples", [])
    if not isinstance(examples, list) or not all(isinstance(ex, dict) for ex in examples):
        return False
    rules = data.get("rules", {})
    if not isinstance(rules, dict) or not all(isinstance(v, str) for v in rules.values()):
        return False
    default_response = data.get("default_response")
    if default_response is not None and not isinstance(default_response, str):
        return False
    return True


class MockPAWBackend(AbstractPAWBackend):
    """Pure-Python mock backend providing deterministic compilation and inference in <10ms.

    Enables 100% unit and integration test coverage without GPU hardware or real weights.
    """

    def __init__(self, strict_misses: bool = False) -> None:
        self._adapters: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        # PAW-BACKEND-03: guards every access to _adapters below. The existing test
        # suite runs entirely single-threaded against MockPAWBackend, and this lock
        # changes none of that behavior -- see
        # test_mock_backend_concurrent_compile_and_infer_PAW_BACKEND_03 for the
        # genuine multi-threaded regression coverage this finding requires.
        self._lock = threading.Lock()
        # J-3: default False -- see MockAdapterMiss's docstring for why this must
        # never change process-wide.
        self.strict_misses = strict_misses

    def _put_adapter_locked(self, adapter_path: str, adapter_data: Dict[str, Any]) -> None:
        """Insert/replace an entry and evict the least-recently-used one past the cap.

        Caller must already hold `self._lock`.
        """
        self._adapters[adapter_path] = adapter_data
        self._adapters.move_to_end(adapter_path)
        while len(self._adapters) > _MAX_CACHED_ADAPTERS:
            self._adapters.popitem(last=False)

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
        # Read whatever manifest already sits at output_path *before* it is
        # overwritten below -- same ordering requirement as
        # ProgramAsWeightsBackend.compile(), for the same reason (see
        # manifest_lineage.read_parent_lineage's docstring).
        parent_program_id, parent_manifest_sha256 = read_parent_lineage(
            output_path, _MAX_ADAPTER_FILE_BYTES
        )

        compile_started = time.monotonic()
        adapter_data = {
            "spec": spec,
            "examples": examples,
            "examples_count": len(examples),
            "backend": "mock",
            "manifest_version": MANIFEST_VERSION,
            "spec_sha256": sha256_text(spec),
            # Unlike ProgramAsWeightsBackend, the mock never renders spec+examples
            # into a separate "text actually sent" -- there is no compile service to
            # send anything to. Nothing distinct from spec_sha256 to record here.
            "full_spec_sha256": None,
            # The mock has no max_spec_examples cap and folds nothing into spec text;
            # every well-formed example handed to compile() is retained verbatim in
            # "examples" and is live training signal for infer()'s example-match
            # step, so all of them count as "folded" here (limit=None).
            "folded_example_ids": folded_example_ids(examples, limit=None),
            "parent_program_id": parent_program_id,
            "parent_manifest_sha256": parent_manifest_sha256,
            "compiler_snapshot": None,
            "rules": {},
            "default_response": None,
        }
        adapter_data["compile_wall_s"] = time.monotonic() - compile_started
        with self._lock:
            self._put_adapter_locked(output_path, adapter_data)

        # PAW-JIT-05: atomic write (temp file + os.replace) instead of a bare open()
        # -- see paw_kit.atomicio's module docstring for the TOCTOU window this closes
        # and why it's load-bearing for the adapter-callable cache's staleness check.
        # Deliberately outside the lock above: this is disk I/O, and holding the lock
        # across it would serialize every concurrent infer()/compile() call against
        # any other adapter_path behind it too.
        atomic_write_text(output_path, json.dumps(adapter_data, indent=2))
        append_history_entry(output_path, adapter_data)

        return output_path

    def _load_adapter_from_disk(self, adapter_path: str) -> Optional[Dict[str, Any]]:
        """PAW-BACKEND-04: bounded, shape-validated reload of a `.paw` adapter file.

        Returns `None` on anything that isn't a regular file within the size cap and
        valid, expected-shape JSON -- matching the existing bare `except Exception:
        adapter = None` fallback's "just treat it as absent" behavior, not a new
        failure mode.
        """
        path_obj = Path(adapter_path)
        try:
            if not path_obj.is_file() or path_obj.stat().st_size > _MAX_ADAPTER_FILE_BYTES:
                return None
            with open(path_obj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        return data if _is_valid_adapter_shape(data) else None

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
        with self._lock:
            adapter = self._adapters.get(adapter_path)
            if adapter is not None:
                self._adapters.move_to_end(adapter_path)

        if adapter is None:
            # PAW-BACKEND-04: bounded, shape-validated load instead of a bare
            # open()+json.load() with no size/shape check.
            loaded = self._load_adapter_from_disk(adapter_path)
            if loaded is not None:
                adapter = loaded
                with self._lock:
                    self._put_adapter_locked(adapter_path, adapter)

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

        # 4. Fallback deterministic output -- or, opt-in (J-3), a hard failure
        # instead of a plausible-looking sentinel a caller's own validation
        # cannot distinguish from a real answer.
        if self.strict_misses:
            raise MockAdapterMiss(
                f"MockPAWBackend(strict_misses=True): no rule, example or "
                f"default_response matched {input_text!r} for adapter "
                f"{adapter_path!r}. The non-strict default would silently "
                f"return the '[mock:{input_text}]' sentinel instead."
            )
        return f"[mock:{input_text}]"

    def is_available(self) -> bool:
        """Check availability. Always True for pure-Python mock backend."""
        return True

    def register_rule(self, adapter_path: str, input_text: str, output: str) -> None:
        """Register a canned deterministic response for a specific input."""
        with self._lock:
            if adapter_path not in self._adapters:
                self._put_adapter_locked(
                    adapter_path, {"spec": "", "examples": [], "rules": {}, "default_response": None}
                )
            self._adapters[adapter_path]["rules"][input_text] = output
            self._adapters.move_to_end(adapter_path)

    def set_default_response(self, adapter_path: str, output: str) -> None:
        """Set a default response for an adapter when no rule or example matches."""
        with self._lock:
            if adapter_path not in self._adapters:
                self._put_adapter_locked(
                    adapter_path, {"spec": "", "examples": [], "rules": {}, "default_response": None}
                )
            self._adapters[adapter_path]["default_response"] = output
            self._adapters.move_to_end(adapter_path)

    def get_adapter(self, adapter_path: str) -> Optional[Dict[str, Any]]:
        """Retrieve in-memory adapter state."""
        with self._lock:
            adapter = self._adapters.get(adapter_path)
            if adapter is not None:
                self._adapters.move_to_end(adapter_path)
            return adapter

    def reset(self) -> None:
        """Clear all registered mock adapters and rules."""
        with self._lock:
            self._adapters.clear()
