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
            "See measurements/README.md 'If inference is unexpectedly slow'.",
        )
    if gpu_offload:
        return CheckResult("llama_cpp", "PASS", f"version {version}; GPU offload supported", "")
    return CheckResult(
        "llama_cpp",
        "WARN",
        f"version {version}; GPU offload NOT supported (CPU-only wheel)",
        "Measured ~90x slower on CPU (5.9s vs 65ms per call). See the README and "
        "measurements/README.md 'If inference is unexpectedly slow' for installing a "
        "CUDA wheel of llama-cpp-python.",
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


@_guarded("PAW_API_KEY")
def check_api_key() -> CheckResult:
    """Is `PAW_API_KEY` set. Never prints the key itself -- only "set" / "not set"."""
    if os.environ.get("PAW_API_KEY"):
        return CheckResult("PAW_API_KEY", "PASS", "set", "")
    return CheckResult(
        "PAW_API_KEY",
        "WARN",
        "not set",
        "Inference on an already-cached program needs no key; compiling a new one does. "
        "Get one at https://programasweights.com/settings",
    )


@_guarded("Upstream service health")
def check_service_health(
    api_url: Optional[str] = None,
    timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S,
    transport: Optional[httpx.BaseTransport] = None,
) -> CheckResult:
    """GET `<api_url>/api/v1/health`.

    PASS only on 200 with a non-empty `gpu_services` dict; WARN on 200 with an empty
    one (the service can report healthy while every compile backend behind it is down --
    see upstream issue #5); FAIL on a non-200 status or a network error.

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
    gpu_services = data.get("gpu_services") if isinstance(data, dict) else None

    if isinstance(gpu_services, dict) and gpu_services:
        return CheckResult(
            "Upstream service health",
            "PASS",
            f"200 OK, {len(gpu_services)} gpu service(s) registered",
            "",
        )
    return CheckResult(
        "Upstream service health",
        "WARN",
        "200 OK but gpu_services is empty",
        "service reports healthy but no GPU services are registered; compiles are "
        "likely to fail or hang, see upstream issue #5",
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
    that specific adapter's program is fully offline-ready."""
    from programasweights import is_offline_ready, list_cached_programs

    programs = list_cached_programs()
    detail = f"{len(programs)} cached program(s)"

    if not adapter_path:
        return CheckResult("Cached programs", "PASS", detail, "")

    from paw_kit.backend.programasweights import ProgramAsWeightsBackend

    manifest = ProgramAsWeightsBackend.read_manifest(adapter_path)
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
    """
    results: List[CheckResult] = [
        check_sdk_importable(),
        check_llama_cpp(),
        check_gpu_visible(),
        check_api_key(),
    ]

    if offline:
        results.append(
            CheckResult(
                "Upstream service health",
                "WARN",
                "skipped (--offline)",
                "Run without --offline to check upstream service health.",
            )
        )
    else:
        results.append(check_service_health(api_url=api_url, timeout_s=timeout_s, transport=transport))

    results.append(check_base_model_cached())
    results.append(check_cached_programs(adapter_path))
    results.append(check_rate_limit_note())
    return results
