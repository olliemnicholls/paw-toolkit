"""Backend bridging paw-kit to the official ProgramAsWeights SDK.

This is the first backend in paw-kit that runs a real model. It delegates both halves
of `AbstractPAWBackend` to the upstream `programasweights` package published by the
paper's authors (https://github.com/programasweights/programasweights-python):

- `compile()` submits the spec to the ProgramAsWeights compile service (remote; needs a
  `PAW_API_KEY`) and records the resulting program ID in a small JSON manifest at
  `output_path`. The manifest *is* paw-kit's `.paw` artifact for this backend; the
  actual adapter weights live in the SDK's own local cache.
- `infer()` resolves that manifest to an upstream program, loads it through
  `programasweights.function(...)` (local llama.cpp inference on a frozen Qwen3-0.6B
  or GPT-2 interpreter, GPU if available) and calls it.

Two honest limitations, both inherited from the upstream API as of September 2026:

1. **The upstream compiler does not accept training examples.** It generates its own
   from the spec via teacher models. paw-kit's traced examples (from `@compile_on_hit`)
   and active-learning gold labels can therefore only reach the compiler as *few-shot
   demonstrations appended to the spec text*, capped by `max_spec_examples`. That is
   what this backend does. Whether it measurably helps is an open question that the
   `paw-test` harness exists to answer; do not assume it does.
2. **No token-level grammar enforcement.** The upstream callable exposes no grammar,
   JSON-schema or logits-processor hook, so `grammar_constraint` cannot be applied
   during decoding here. It is accepted for interface compatibility and ignored (with a
   one-time warning). Schema safety for this backend comes from `paw_kit.schema.load`'s
   *post-hoc* Pydantic validation plus fail-open fallback, not from constrained decoding.

Nothing in this module is imported at package import time except the standard library
and paw-kit's own helpers; `programasweights` is imported lazily so the rest of paw-kit
keeps working (and its test suite keeps running) without it installed.
"""

from __future__ import annotations

from collections import OrderedDict
import importlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional
import warnings

from paw_kit.atomicio import atomic_write_text
from paw_kit.backend.base import AbstractPAWBackend

# Public compiler names as documented in the upstream README. `paw-4b-qwen3-0.6b` is
# the server default (single-forward-pass "fast" compiler from the original PAW paper);
# `paw-ft-bs48` is the finetune compiler described in "Compile by Training"
# (arXiv:2609.04199) and must go through compile_async.
FAST_COMPILER = "paw-4b-qwen3-0.6b"
FINETUNE_COMPILER = "paw-ft-bs48"

MANIFEST_BACKEND_NAME = "programasweights"
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_CACHED_FUNCTIONS = 8  # each holds a loaded llama.cpp model; keep this small

# Upstream does not publish an enum of job states. These are the terminal states we
# treat as failure; anything else with a `program_id` set is treated as success. Adjust
# here if the service's vocabulary turns out to differ.
_FAILED_STATES = frozenset({"failed", "error", "cancelled", "canceled"})
_SUCCESS_STATES = frozenset({"completed", "complete", "succeeded", "success", "ready", "done"})


def _sdk_installed() -> bool:
    return importlib.util.find_spec("programasweights") is not None


def _render_spec_with_examples(spec: str, examples: List[Dict[str, str]], limit: int) -> str:
    """Append up to `limit` input/output pairs to the spec as few-shot demonstrations."""
    usable = [ex for ex in examples if isinstance(ex, dict) and "input" in ex and "output" in ex]
    if limit <= 0 or not usable:
        return spec
    lines = [spec.rstrip(), "", "Examples of correct behaviour:"]
    for ex in usable[:limit]:
        lines.append(f"Input: {ex['input']}")
        lines.append(f"Output: {ex['output']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class ProgramAsWeightsBackend(AbstractPAWBackend):
    """`AbstractPAWBackend` implemented on top of the official `programasweights` SDK.

    Args:
        compiler: Upstream compiler name. `FAST_COMPILER` (default) returns in seconds
            via the synchronous endpoint; `FINETUNE_COMPILER` is queued through
            `compile_async` and polled.
        n_gpu_layers: Passed to `programasweights.function`. `None` lets the SDK decide
            (GPU if available); `0` forces CPU.
        n_ctx: Context window passed to `programasweights.function`.
        max_tokens: Per-call output cap passed to the loaded function. `None` = SDK default.
        offline: If True, never touch the network at inference time; the program, runtime
            and base model must already be in the SDK cache (see `prepare_program`).
        max_spec_examples: How many traced/gold examples to fold into the spec text at
            compile time (see module docstring, limitation 1). `0` disables it.
        poll_interval_s / compile_timeout_s: Polling cadence and ceiling for async compiles.
        sdk: Test seam. Any object exposing `compile`, `compile_async`,
            `get_compile_status` and `function` with the upstream signatures. Defaults to
            the real `programasweights` module, imported lazily on first use.
    """

    def __init__(
        self,
        compiler: str = FAST_COMPILER,
        *,
        n_gpu_layers: Optional[int] = None,
        n_ctx: int = 2048,
        max_tokens: Optional[int] = None,
        offline: bool = False,
        max_spec_examples: int = 16,
        poll_interval_s: float = 5.0,
        compile_timeout_s: float = 3600.0,
        sdk: Any = None,
    ) -> None:
        self.compiler = compiler
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.offline = offline
        self.max_spec_examples = max_spec_examples
        self.poll_interval_s = poll_interval_s
        self.compile_timeout_s = compile_timeout_s
        self._sdk = sdk
        self._functions: "OrderedDict[str, Callable[..., str]]" = OrderedDict()
        self._lock = threading.Lock()
        self._warned_grammar = False

    # ------------------------------------------------------------------ plumbing

    def _paw(self) -> Any:
        if self._sdk is None:
            if not _sdk_installed():
                raise RuntimeError(
                    "ProgramAsWeightsBackend requires the official SDK:\n\n"
                    "    pip install programasweights "
                    "--extra-index-url https://pypi.programasweights.com/simple/\n\n"
                    "and a PAW_API_KEY (https://programasweights.com/settings) for compilation."
                )
            self._sdk = importlib.import_module("programasweights")
        return self._sdk

    def is_available(self) -> bool:
        """True if the SDK is importable (or injected). Does not check the API key: inference
        on an already-cached program works without one."""
        return self._sdk is not None or _sdk_installed()

    def has_api_key(self) -> bool:
        return bool(os.environ.get("PAW_API_KEY"))

    # ------------------------------------------------------------------ compile

    def compile(self, spec: str, examples: List[Dict[str, str]], output_path: str) -> str:
        paw = self._paw()
        if not self.offline and not self.has_api_key():
            raise RuntimeError(
                "Compilation needs PAW_API_KEY in the environment "
                "(https://programasweights.com/settings). Inference on cached programs does not."
            )

        full_spec = _render_spec_with_examples(spec, examples, self.max_spec_examples)

        if self.compiler == FINETUNE_COMPILER:
            job = paw.compile_async(full_spec, compiler=self.compiler)
            program_id, slug, status = self._wait_for_job(paw, job)
        else:
            program = paw.compile(full_spec, compiler=self.compiler)
            program_id = getattr(program, "id", None) or (program.get("id") if isinstance(program, dict) else None)
            slug = getattr(program, "slug", None) or (program.get("slug") if isinstance(program, dict) else None)
            status = getattr(program, "status", None) or (program.get("status") if isinstance(program, dict) else None)
            if not program_id or (status and str(status).lower() in _FAILED_STATES):
                err = getattr(program, "error", None) or (program.get("error") if isinstance(program, dict) else None)
                raise RuntimeError(f"ProgramAsWeights compile failed (status={status!r}): {err!r}")

        manifest = {
            "backend": MANIFEST_BACKEND_NAME,
            "program_id": program_id,
            "slug": slug,
            "compiler": self.compiler,
            "status": status,
            "spec": spec,
            "examples_folded_into_spec": min(len(examples), self.max_spec_examples),
            "examples_count": len(examples),
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_text(output_path, json.dumps(manifest, indent=2))

        # Drop any loaded function for this path: the program behind it just changed.
        with self._lock:
            self._functions.pop(output_path, None)
        return output_path

    def _wait_for_job(self, paw: Any, job: Any) -> tuple[str, Optional[str], str]:
        """Poll `get_compile_status` until the job reaches a terminal state."""
        job_id = job.get("job_id") if isinstance(job, dict) else getattr(job, "job_id", None)
        if not job_id:
            raise RuntimeError(f"compile_async returned no job_id: {job!r}")

        deadline = time.monotonic() + self.compile_timeout_s
        while True:
            status_obj = paw.get_compile_status(job_id)
            get = status_obj.get if isinstance(status_obj, dict) else lambda k, d=None: getattr(status_obj, k, d)
            status = str(get("status") or "").lower()
            program_id = get("program_id")
            if status in _FAILED_STATES:
                raise RuntimeError(f"ProgramAsWeights finetune compile {job_id} {status}: {get('error')!r}")
            if program_id and (status in _SUCCESS_STATES or get("completed_at")):
                return str(program_id), get("slug"), status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"ProgramAsWeights compile {job_id} still {status!r} after {self.compile_timeout_s}s"
                )
            time.sleep(self.poll_interval_s)

    # ------------------------------------------------------------------ infer

    @staticmethod
    def read_manifest(adapter_path: str) -> Dict[str, Any]:
        """Load and shape-check the JSON manifest written by `compile()`."""
        path = Path(adapter_path)
        if not path.is_file():
            raise FileNotFoundError(f"No adapter manifest at {adapter_path}")
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise ValueError(f"Adapter manifest {adapter_path} exceeds {_MAX_MANIFEST_BYTES} bytes")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("backend") != MANIFEST_BACKEND_NAME:
            raise ValueError(
                f"{adapter_path} is not a ProgramAsWeights manifest "
                f"(backend={data.get('backend') if isinstance(data, dict) else None!r})"
            )
        program_id = data.get("program_id") or data.get("slug")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError(f"{adapter_path} has no program_id/slug")
        return data

    def _get_function(self, adapter_path: str) -> Callable[..., str]:
        with self._lock:
            fn = self._functions.get(adapter_path)
            if fn is not None:
                self._functions.move_to_end(adapter_path)
                return fn

        manifest = self.read_manifest(adapter_path)
        program_id = manifest.get("program_id") or manifest["slug"]
        kwargs: Dict[str, Any] = {"n_ctx": self.n_ctx, "offline": self.offline}
        if self.n_gpu_layers is not None:
            kwargs["n_gpu_layers"] = self.n_gpu_layers
        fn = self._paw().function(program_id, **kwargs)

        with self._lock:
            self._functions[adapter_path] = fn
            self._functions.move_to_end(adapter_path)
            while len(self._functions) > _MAX_CACHED_FUNCTIONS:
                self._functions.popitem(last=False)
        return fn

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
        if grammar_constraint is not None and not self._warned_grammar:
            self._warned_grammar = True
            warnings.warn(
                "ProgramAsWeightsBackend cannot apply grammar_constraint at decoding time: the "
                "upstream SDK exposes no grammar/logits hook. Output is validated after "
                "generation by paw_kit.schema.load instead (fail-open on mismatch).",
                UserWarning,
                stacklevel=2,
            )
        fn = self._get_function(adapter_path)
        call_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        out = fn(input_text, **call_kwargs)
        return out if isinstance(out, str) else str(out)

    def reset(self) -> None:
        """Drop every loaded function (frees the underlying llama.cpp models)."""
        with self._lock:
            self._functions.clear()
