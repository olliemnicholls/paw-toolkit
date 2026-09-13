"""Environment diagnostics for running a real `ProgramAsWeightsBackend`.

`paw-kit doctor` (wired in `cli.py`) exists because the failure modes of the real
backend are almost all *environment* problems -- a CPU-only `llama-cpp-python` wheel
that is ~90x slower, a base model that hasn't been downloaded yet, an upstream compile
service that returns a healthy-looking `200` while every GPU worker behind it is down
(see upstream issue #5) -- and every one of them currently surfaces as a confusing
failure deep inside `compile()`/`infer()` rather than as a clear, actionable diagnostic
up front. See `measurements/README.md` for the measurements this module's remedies cite.

Every check below returns a `CheckResult` and is guarded so that an unexpected exception
inside the check itself becomes a FAIL carrying the exception text, never an uncaught
traceback -- one broken probe (e.g. a corrupted `programasweights` install) must not take
the rest of `paw-kit doctor` down with it.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import os
import subprocess
from typing import Any, Callable, List, Optional, TypeVar

import httpx

_F = TypeVar("_F", bound=Callable[..., "CheckResult"])

# Upstream docs, as of September 2026: https://programasweights.com/docs/rate-limits
_ANONYMOUS_COMPILES_PER_HOUR = 20
_AUTHENTICATED_COMPILES_PER_HOUR = 60

_DEFAULT_HEALTH_TIMEOUT_S = 10.0
_DEFAULT_GPU_TIMEOUT_S = 5.0

_QWEN_INTERPRETER = "Qwen/Qwen3-0.6B"


@dataclass
class CheckResult:
    """One `paw-kit doctor` check outcome.

    `status` is one of `"PASS"`, `"WARN"`, `"FAIL"`. `remedy` is a one-line suggestion
    for fixing a WARN/FAIL; empty for a PASS that needs no action.
    """

    name: str
    status: str
    detail: str
    remedy: str = ""


def _guarded(name: str) -> Callable[[_F], _F]:
    """Decorator: turn any exception a check function raises into a FAIL `CheckResult`
    carrying the exception text, instead of letting it propagate as an uncaught
    traceback that would crash the whole `doctor` run over one bad probe."""

    def decorator(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> CheckResult:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                return CheckResult(
                    name=name,
                    status="FAIL",
                    detail=f"check raised {type(exc).__name__}: {exc}",
                    remedy="Run the check directly for a full traceback if this is unexpected.",
                )

        return wrapper  # type: ignore[return-value]

    return decorator


@_guarded("programasweights SDK")
def check_sdk_importable() -> CheckResult:
    """Is the official `programasweights` SDK importable, and what version."""
    try:
        import programasweights
    except Exception as exc:
        return CheckResult(
            "programasweights SDK",
            "FAIL",
            f"not importable: {exc}",
            "pip install programasweights (or: uv sync --extra real)",
        )
    version = getattr(programasweights, "__version__", "unknown")
    return CheckResult("programasweights SDK", "PASS", f"version {version}", "")


@_guarded("llama_cpp")
def check_llama_cpp() -> CheckResult:
    """Is `llama_cpp` importable, its version, and whether it was built with GPU offload.

    WARNs (never FAILs) on a CPU-only build: inference still works, just ~90x slower
    (measured: 5.9s vs 65ms per call, see `measurements/README.md`).
    """
    try:
        import llama_cpp
    except Exception as exc:
        return CheckResult(
            "llama_cpp",
            "FAIL",
            f"not importable: {exc}",
            "pip install programasweights (bundles llama-cpp-python as a dependency)",
        )
    version = getattr(llama_cpp, "__version__", "unknown")
    try:
        gpu_offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:
        return CheckResult(
            "llama_cpp",
            "WARN",
            f"version {version}; llama_supports_gpu_offload() raised {type(exc).__name__}: {exc}",
            "See docs/install.md 'GPU support'.",
        )
    if gpu_offload:
        return CheckResult("llama_cpp", "PASS", f"version {version}; GPU offload supported", "")
    return CheckResult(
        "llama_cpp",
        "WARN",
        f"version {version}; GPU offload NOT supported (CPU-only wheel)",
        "Measured ~90x slower on CPU (5.9s vs 65ms per call). See docs/install.md "
        "'GPU support' for installing a CUDA wheel of llama-cpp-python.",
    )


def check_gpu_visible(timeout_s: float = _DEFAULT_GPU_TIMEOUT_S) -> CheckResult:
    """Is an NVIDIA GPU visible via `nvidia-smi`. Never FAILs -- CPU-only is a WARN,
    not an error, since inference still works (just much slower)."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except Exception as exc:
        return CheckResult(
            "GPU",
            "WARN",
            f"nvidia-smi unavailable ({type(exc).__name__}: {exc})",
            "No NVIDIA GPU detected; inference will run on CPU (~90x slower, see "
            "measurements/README.md).",
        )
    output = (proc.stdout or "").strip()
    if proc.returncode != 0 or not output:
        return CheckResult(
            "GPU",
            "WARN",
            "nvidia-smi ran but reported no GPU",
            "No NVIDIA GPU detected; inference will run on CPU (~90x slower, see "
            "measurements/README.md).",
        )
    return CheckResult("GPU", "PASS", output.splitlines()[0], "")


# Every `PAW_API_KEY` this project's own scripts and docs use (`scripts/measure_*.py`,
# `paw_kit/serve/docker.py`) is `paw_sk_...` -- not a documented, versioned format
# guarantee from upstream, just the one shape ever observed. Used only to say whether a
# set value *looks like* a key, never to claim it is one -- see `check_api_key`.
_API_KEY_PREFIX = "paw_sk_"


@_guarded("PAW_API_KEY")
def check_api_key() -> CheckResult:
    """Is `PAW_API_KEY` set, and does it look like a `paw_sk_...` key. Never prints the
    key itself -- only "set" / "not set" and whether the prefix matches.

    This check cannot, and does not claim to, validate the key against the service:
    `precheck_compile` (`ProgramAsWeightsBackend`'s readiness probe) returns HTTP 200
    for an absent key, a syntactically invalid key, and a valid key alike -- see
    `measurements/README.md`'s "Finetune compiler" section, "What compile C actually
    did". Nothing short of an actual compile call tells you whether the key is good, so
    the PASS/WARN below is honest about *format* only: a set, correctly-prefixed value
    still WARNs, worded to say a wrong key will only surface on the first compile.
    """
    key = os.environ.get("PAW_API_KEY")
    if not key:
        return CheckResult(
            "PAW_API_KEY",
            "WARN",
            "not set",
            "Inference on an already-cached program needs no key; compiling a new one does. "
            "Get one at https://programasweights.com/settings",
        )
    if not key.startswith(_API_KEY_PREFIX):
        return CheckResult(
            "PAW_API_KEY",
            "WARN",
            f"set, but does not look like a {_API_KEY_PREFIX}... key",
            "This is a format check only, not a validity check (nothing short of a "
            "real compile call validates a key against the service) -- but every "
            f"key this project has seen starts with '{_API_KEY_PREFIX}'. Double-check "
            "the value if compiles fail with an authentication error.",
        )
    return CheckResult(
        "PAW_API_KEY",
        "PASS",
        f"set and looks like a {_API_KEY_PREFIX}... key",
        "Format looks right; this does not confirm the key is valid -- a wrong key "
        "only surfaces on the first real compile call, not here.",
    )


@_guarded("Upstream service health")
def check_service_health(
    api_url: Optional[str] = None,
    timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    transport: Optional[httpx.BaseTransport] = None,
) -> CheckResult:
    """GET `<api_url>/api/v1/health`.

    FAIL only on a non-200 status or a network error. Otherwise WARN on any of:
    - `status` present and not `"ok"`/`"healthy"` (e.g. `"degraded"`) -- reported
      verbatim.
    - a non-empty `warnings` list -- every entry is listed in the message, not just
      counted. When any entry contains `redis_unavailable`, the remedy specifically
      calls out that async compiles (the `paw-ft-bs48` finetune compiler) are likely to
      be refused (HTTP 503 `durable_queue_unavailable`) while fast compiles may still
      work -- this is the one check that would have predicted that failure and
      previously said nothing about it (`measurements/README.md`'s "Finetune compiler"
      section, "paw-test feedback" item 4).
    - an empty `gpu_services` dict -- kept as a WARN (the service can report healthy
      while every compile backend behind it is down, see upstream issue #5), but its
      remedy is softened: the fast compiler has been observed to compile successfully
      even when `gpu_services` is empty, so this alone is not a reliable predictor of
      failure.
    PASS only when none of the above apply.

    `api_url`, `timeout_s` and `transport` are parameters (rather than always reading
    `programasweights.get_api_url()` and hitting the network directly) so tests can
    point this at a fake server via `httpx.MockTransport`.
    """
    if api_url is None:
        from programasweights import get_api_url

        api_url = get_api_url()
    url = api_url.rstrip("/") + "/api/v1/health"

    try:
        if transport is not None:
            with httpx.Client(transport=transport) as client:
                resp = client.get(url, timeout=timeout_s)
        else:
            resp = httpx.get(url, timeout=timeout_s)
    except httpx.HTTPError as exc:
        return CheckResult(
            "Upstream service health",
            "FAIL",
            f"{url} unreachable: {type(exc).__name__}: {exc}",
            "Check network connectivity and PAW_API_URL, then run `paw-kit doctor` again.",
        )

    if resp.status_code != 200:
        return CheckResult(
            "Upstream service health",
            "FAIL",
            f"HTTP {resp.status_code} from {url}",
            "The compile service is unhealthy; try again later.",
        )

    try:
        data = resp.json()
    except Exception:
        data = None
    if not isinstance(data, dict):
        data = {}

    gpu_services = data.get("gpu_services")
    gpu_services_empty = not (isinstance(gpu_services, dict) and gpu_services)

    reported_status = data.get("status")
    status_text = str(reported_status).strip() if isinstance(reported_status, str) else ""
    status_unhealthy = bool(status_text) and status_text.lower() not in ("ok", "healthy")

    raw_warnings = data.get("warnings")
    warnings_list = [str(w) for w in raw_warnings] if isinstance(raw_warnings, list) else []
    redis_unavailable = any("redis_unavailable" in w for w in warnings_list)

    if not status_unhealthy and not warnings_list and not gpu_services_empty:
        return CheckResult(
            "Upstream service health",
            "PASS",
            f"200 OK, {len(gpu_services)} gpu service(s) registered",
            "",
        )

    detail_parts = []
    remedy_parts = []

    if status_unhealthy:
        detail_parts.append(f"status={status_text!r}")
        remedy_parts.append(f"service reports status={status_text!r}, not ok/healthy.")

    if warnings_list:
        detail_parts.append("warnings=[" + "; ".join(warnings_list) + "]")
        if redis_unavailable:
            remedy_parts.append(
                "warnings include redis_unavailable: async compiles (the paw-ft-bs48 "
                "finetune compiler) are likely to be refused (HTTP 503 "
                "durable_queue_unavailable) while fast compiles may still work."
            )
        else:
            remedy_parts.append("see the warnings listed above for detail.")

    if gpu_services_empty:
        detail_parts.append("gpu_services is empty")
        remedy_parts.append(
            "gpu_services is empty; the fast compiler has been observed to compile "
            "successfully even so -- this alone is not a reliable predictor of compile "
            "failure, see upstream issue #5."
        )

    return CheckResult(
        "Upstream service health",
        "WARN",
        "200 OK but " + "; ".join(detail_parts),
        " ".join(remedy_parts),
    )


@_guarded("Base model cache")
def check_base_model_cached() -> CheckResult:
    """Is the Qwen3-0.6B GGUF base model already in the SDK's local cache."""
    from programasweights import cache

    manifest = cache.get_base_runtime_manifest(_QWEN_INTERPRETER)
    path = cache.get_cached_base_model_path(manifest)
    if path is None:
        return CheckResult(
            "Base model cache",
            "WARN",
            f"{_QWEN_INTERPRETER} base model not cached",
            "first call will download ~600 MB",
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    return CheckResult("Base model cache", "PASS", f"{path} ({size_mb:.0f} MB)", "")


@_guarded("Cached programs")
def check_cached_programs(adapter_path: Optional[str] = None) -> CheckResult:
    """Count locally cached compiled programs, and, if `adapter_path` is given, whether
    that specific adapter's program is fully offline-ready.

    `--adapter` is meant for a `ProgramAsWeightsBackend` manifest; `read_manifest`
    raises `ValueError` for any other backend's manifest shape (e.g. the mock backend,
    which is what the default backend and `paw-kit demo` write). That is not a broken
    environment -- "offline readiness" simply does not apply to a mock adapter -- so it
    is reported as a WARN, not treated as a check failure.

    C-11: two distinct misdiagnoses this used to produce, both closed by checking the
    path and the manifest's own declared `backend` up front instead of inferring
    everything from `read_manifest`'s exception type:
    1. A mistyped `--adapter` path raised `FileNotFoundError` (`read_manifest`'s own
       `is_file()` guard), which `_guarded` turned into "FAIL | check raised
       FileNotFoundError" with the remedy "Run the check directly for a full
       traceback" -- blaming the environment for a typo.
    2. `read_manifest` raises the same `ValueError` for a genuinely malformed *real*
       manifest (oversized, or a dict whose `backend` field is missing/wrong) as it
       does for an honest mock manifest, and the old code reported both as "offline
       readiness does not apply to a mock-backend manifest" -- the converse
       mislabel: a corrupted real manifest read as "that's just a mock".
    """
    from programasweights import is_offline_ready, list_cached_programs

    programs = list_cached_programs()
    detail = f"{len(programs)} cached program(s)"

    if not adapter_path:
        return CheckResult("Cached programs", "PASS", detail, "")

    from pathlib import Path as _Path
    import json as _json

    if not _Path(adapter_path).is_file():
        return CheckResult(
            "Cached programs",
            "WARN",
            f"no adapter manifest at {adapter_path}",
            "Check the --adapter path for a typo.",
        )

    from paw_kit.backend.programasweights import ProgramAsWeightsBackend

    try:
        manifest = ProgramAsWeightsBackend.read_manifest(adapter_path)
    except ValueError:
        # Peek at the raw JSON directly rather than trusting read_manifest's
        # exception alone to mean "this is a mock manifest" -- read_manifest raises
        # the identical ValueError for an oversized file, a non-dict, or a dict
        # whose `backend` is simply wrong, none of which are "just a mock".
        declared_backend = None
        try:
            raw = _json.loads(_Path(adapter_path).read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                declared_backend = raw.get("backend")
        except (OSError, ValueError):
            pass
        if declared_backend == "mock":
            return CheckResult(
                "Cached programs",
                "WARN",
                "offline readiness does not apply to a mock-backend manifest",
                "",
            )
        return CheckResult(
            "Cached programs",
            "WARN",
            f"{adapter_path} is not a readable programasweights manifest "
            f"(declared backend={declared_backend!r})",
            "Check the --adapter path points at a real ProgramAsWeightsBackend manifest.",
        )
    program_id = manifest.get("program_id") or manifest.get("slug")
    ready = is_offline_ready(program_id)
    detail += f"; {adapter_path} (program_id={program_id}) offline_ready={ready}"
    if ready:
        return CheckResult("Cached programs", "PASS", detail, "")
    return CheckResult(
        "Cached programs",
        "WARN",
        detail,
        "Run once with network access to populate the cache before using --offline.",
    )


@_guarded("Rate limits")
def check_rate_limit_note() -> CheckResult:
    """Informational reminder of upstream's compile rate limits. Always PASS."""
    return CheckResult(
        "Rate limits",
        "PASS",
        f"anonymous: {_ANONYMOUS_COMPILES_PER_HOUR} compiles/hour; "
        f"authenticated: {_AUTHENTICATED_COMPILES_PER_HOUR} compiles/hour (upstream docs)",
        "",
    )


def run_checks(
    *,
    offline: bool = False,
    adapter_path: Optional[str] = None,
    api_url: Optional[str] = None,
    timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    transport: Optional[httpx.BaseTransport] = None,
) -> List[CheckResult]:
    """Run every `paw-kit doctor` check and return their results in display order.

    `offline` skips every check that touches the network (today, just the upstream
    service health check) instead of running it.

    If `check_sdk_importable()` itself fails, every check that needs the SDK to do
    anything meaningful (`llama_cpp` -- bundled by the SDK install; the upstream
    service health check's default-URL lookup; the base-model cache; cached programs)
    is emitted as a WARN "skipped (SDK not installed)" instead of being run. Each of
    those would otherwise raise its own opaque `ModuleNotFoundError` and FAIL, burying
    check #1's one real cause under 3-4 copies of the same underlying symptom on a
    stock install. A WARN never affects the process exit code -- `paw-kit doctor`
    exits 1 only when a real FAIL is present -- so a stock install without the
    optional SDK now reports exactly one FAIL, not several.
    """
    sdk_result = check_sdk_importable()
    sdk_missing = sdk_result.status == "FAIL"

    def _skipped(name: str) -> CheckResult:
        return CheckResult(name, "WARN", "skipped (SDK not installed)", sdk_result.remedy)

    results: List[CheckResult] = [sdk_result]
    results.append(_skipped("llama_cpp") if sdk_missing else check_llama_cpp())
    results.append(check_gpu_visible())
    results.append(check_api_key())

    if offline:
        results.append(
            CheckResult(
                "Upstream service health",
                "WARN",
                "skipped (--offline)",
                "Run without --offline to check upstream service health.",
            )
        )
    elif sdk_missing and api_url is None:
        # check_service_health only touches `programasweights` itself to resolve the
        # default API URL when none is given (see its `if api_url is None:` branch);
        # an explicit api_url (tests, or a future --api-url flag) needs no SDK at all.
        results.append(_skipped("Upstream service health"))
    else:
        results.append(check_service_health(api_url=api_url, timeout_s=timeout_s, transport=transport))

    results.append(_skipped("Base model cache") if sdk_missing else check_base_model_cached())
    results.append(
        _skipped("Cached programs") if sdk_missing else check_cached_programs(adapter_path)
    )
    results.append(check_rate_limit_note())
    return results
