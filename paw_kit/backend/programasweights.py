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

Three honest limitations, all reflecting upstream SDK behavior as of September 2026:

1. **The upstream compiler does not accept training examples.** It generates its own
   from the spec via teacher models. paw-kit's traced examples (from `@compile_on_hit`)
   and active-learning gold labels can therefore only reach the compiler as *few-shot
   demonstrations appended to the spec text*, capped by `max_spec_examples`. That is
   what this backend does. Whether it measurably helps is an open question that the
   `paw-test` harness exists to answer; do not assume it does.
2. **Token-level grammar enforcement is applied, on a shape guarantee only.** Since
   `programasweights==0.4.6` (PR #6) the upstream callable exposes a public
   `logits_processor` hook, and `infer()` uses it: when constructed with
   `constrained_decoding=True` (the default) and a working `llguidance` install, a
   fresh `paw_kit.schema.constraint` matcher is built per call from
   `pydantic_to_regex`'s output and masks every generation step, so a token that would
   leave the compiled regex's language cannot be sampled. This is a **shape** guarantee
   only -- it proves the output parses as the target schema, not that its field values
   are correct, and it does not change what `paw.load` does after generation: post-hoc
   Pydantic validation and fail-open fallback (mechanism 2, `decisions.md` §2) stay
   load-bearing regardless, because generation truncated at `max_tokens` or the context
   window is still an invalid object, and a masking engine error is itself a fail-open
   exception (`PAWSchemaError`, propagated unwrapped -- see `infer()`). Pass
   `constrained_decoding=False` to opt out (e.g. an adapter compiled for a different
   schema than the one it is now being asked for, where forcing shape produces
   syntactically valid but semantically poor output; see `measurements/README.md`).
   Three distinct failure modes, three distinct degradations, never a raise on every
   call for any of them: (a) `llguidance` is not importable, or the vocabulary object
   built from the loaded model fails its own verification -- this **instance**
   degrades to unconstrained decoding for every schema, with one `UserWarning` at
   construction or model load naming the cause (`applies_grammar_constraint` becomes
   `False`); (b) a grammar is refused AT CONSTRUCTION -- fuel/grammar limits exceeded,
   or an empty/EOS-only initial mask (`paw_kit.schema.constraint.ConstraintUnavailable`)
   -- this degrades **that one schema's grammar**, not the instance: the flag stays
   `True`, one `UserWarning` names the pattern and reason, and every call for that same
   grammar runs unconstrained thereafter (`infer()`); (c) a failure discovered
   **during generation** -- `consume_token()` rejects a sampled token, or the matcher
   enters an error state after a later step -- is not a property of the grammar alone
   (partial output may already exist), so it is never silently degraded: it raises
   plain `PAWSchemaError`, propagated unwrapped through `infer()`, and reaches
   `paw.load`'s ordinary fail-open fallback like any other local-execution failure.
3. **`is_available()` only proves SDK importability.** `is_available()` verifies that
   the `programasweights` package is importable (or an injected double is present), but
   proves nothing about whether the underlying llama.cpp runtime actually works, whether
   GPU offload is functional, or whether required model weights are cached. A half-built
   or CPU-only llama.cpp installation surfaces on `infer()`; run `paw-kit doctor` to
   verify runtime health before serving production traffic.

Privacy: upstream's `paw.compile`/`paw.compile_async` default to `public=True`, which
lists the compiled program on programasweights.com with its full spec text readable by
anyone, no login required. That spec is not just what the caller wrote -- `compile()`
in this module folds up to `max_spec_examples` traced production input/output pairs into
it first (see `_render_spec_with_examples`), so a public compile publishes a sample of
real traffic. `ProgramAsWeightsBackend` therefore defaults to `public=False`, opposite to
upstream; pass `public=True` explicitly to opt into hub listing. One caveat `public=False`
cannot fix: upstream's compile cache is keyed on spec text alone and ignores `public` on a
hit, so recompiling a spec that was previously compiled public returns that same public
program unchanged, regardless of what `public` is passed this time (`compile()` best-effort
checks for this via `precheck_compile` and warns; it cannot change the existing program's
visibility).

**What the manifest can and cannot tell you about visibility (A-2).** `public_requested`
is what this backend *asked* the service for. It is not, and has never been, evidence of
what the compiled program's visibility actually is -- a cache hit ignores `public=`
entirely, and the precheck that detects a cache hit exposes no visibility field at all.
`public_confirmed` is the separate, three-state record of what the server answered when
asked directly: `True`, `False`, or `None` with `public_confirmed_reason` saying why there
is no answer. It is `None` unless the backend was constructed with
`verify_visibility=True`, and an unanswered question is never reported as "private".
Confirmed live on 2026-09-11: six programs this project compiled report `public: True`
from the server, and their local manifests recorded no visibility at all. Note also that
visibility is not retroactive -- nothing in this module can make an already-compiled
public program private.

Nothing in this module is imported at package import time except the standard library
and paw-kit's own helpers; `programasweights` is imported lazily so the rest of paw-kit
keeps working (and its test suite keeps running) without it installed.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import httpx

from paw_kit.atomicio import atomic_write_text
from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.manifest_lineage import (
    append_history_entry,
    extract_snapshot,
    folded_example_ids,
    read_parent_lineage,
    select_folded_examples,
    sha256_text,
)
from paw_kit.schema.constraint import ConstraintUnavailable, Vocabulary, build_constraint
from paw_kit.schema.exceptions import PAWSchemaError

# Public compiler names as documented in the upstream README. `paw-4b-qwen3-0.6b` is
# the server default (single-forward-pass "fast" compiler from the original PAW paper);
# `paw-ft-bs48` is the finetune compiler described in "Compile by Training"
# (arXiv:2609.04199) and must go through compile_async.
FAST_COMPILER = "paw-4b-qwen3-0.6b"
FINETUNE_COMPILER = "paw-ft-bs48"

MANIFEST_BACKEND_NAME = "programasweights"
# v1 -> v2: compile() started recording lineage (spec/full-spec hashes, which
# examples were actually folded, parent-manifest linkage, compile wall time, and
# whatever compiler_snapshot the SDK returns).
# v2 -> v3 (D-ADD-1): A-2 removed `public` and added `public_requested`,
# `public_confirmed`, `public_confirmed_reason`, and `cached_program_id` -- a
# strictly larger, key-set-changing edit than v1->v2's purely additive one, so a
# pre-A-2 and a post-A-2 manifest need their own distinguishing version too (the
# whole reason this field exists). `read_manifest` does not gate on this -- a
# version-1 manifest (no `manifest_version` key at all) must keep loading unchanged
# -- this constant exists for `paw-inspect`/`paw-kit history` to display, not to
# reject anything.
MANIFEST_VERSION = 3
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_CACHED_FUNCTIONS = 8  # each holds a loaded llama.cpp model; keep this small
# How many distinct grammars' "ConstraintUnavailable at construction" UserWarning this
# instance remembers already having emitted (`infer()`'s warn-once, keyed on a hash of
# the pattern text -- see `_warn_constraint_unavailable_once`). Bounded like
# `_MAX_CACHED_FUNCTIONS` above for the same reason: an unbounded set would leak memory
# for a long-lived instance driven by many distinct schemas. Losing an old entry to
# eviction and warning again for a schema not seen in a while is an acceptable
# degradation -- it costs one extra log line, never a raise and never a teacher call.
_MAX_WARNED_CONSTRAINT_UNAVAILABLE_GRAMMARS = 64

# Upstream does not publish an enum of job states. These are the terminal states we
# treat as failure; anything else with a `program_id` set is treated as success. Adjust
# here if the service's vocabulary turns out to differ.
_FAILED_STATES = frozenset({"failed", "error", "cancelled", "canceled"})
_SUCCESS_STATES = frozenset({"completed", "complete", "succeeded", "success", "ready", "done"})


def _sdk_installed() -> bool:
    return importlib.util.find_spec("programasweights") is not None


def _render_spec_with_examples(spec: str, examples: List[Dict[str, str]], limit: int) -> str:
    """Append up to `limit` input/output pairs to the spec as few-shot demonstrations.

    A-6: the "usable" filter used to be re-implemented here, a second copy of
    `select_folded_examples`'s predicate (Pattern 2 waiting to happen -- the count and the
    content could drift apart with nothing noticing). There is one predicate now, and
    `select_folded_examples` is it, so `examples_folded_into_spec` and the spec text can
    no longer disagree about what was folded.
    """
    folded = select_folded_examples(examples, limit)
    if not folded:
        return spec
    lines = [spec.rstrip(), "", "Examples of correct behaviour:"]
    for ex in folded:
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
        constrained_decoding: Whether to apply grammar-constrained decoding (module
            docstring, limitation 2) through upstream's public `logits_processor` hook.
            Default `True`. `applies_grammar_constraint` (an instance attribute, not a
            class-level constant -- see `AbstractPAWBackend`) is computed from this
            **and** whether `llguidance` is actually importable; requesting it without
            the engine installed emits one `UserWarning` naming the `paw` extra at
            construction, not a raise, and the instance degrades to unconstrained
            decoding. A grammar refused at construction for a *particular* schema
            (fuel/grammar limits, an empty/EOS-only initial mask) degrades only that
            schema, per call, with its own one-time warning -- see `infer()` and
            `_warn_constraint_unavailable_once` -- without touching this flag. Pass
            `False` to opt out even when the engine is available (e.g. an adapter being
            asked for a schema it was not compiled for).
        max_spec_examples: How many traced/gold examples to fold into the spec text at
            compile time (see module docstring, limitation 1). `0` disables it.
        poll_interval_s / compile_timeout_s: Polling cadence and ceiling for async compiles.
        public: Whether compiled programs are listed on the public programasweights.com
            hub with their full spec text (including any folded traced examples) readable
            by anyone, unauthenticated. Upstream defaults this to `True`; paw-kit defaults
            it to `False` (see module docstring, Privacy). Pass `True` to opt in.
            **Only the literal `True` opts in** (A-9): anything else is coerced to
            `False` in `__init__`, because `public=None` used to forward
            `{"public": null}` to a service whose own default is `True` -- a leak out of
            a falsy-looking argument.
        verify_visibility: After a successful compile, ask the service what the compiled
            program's visibility actually *is* and record the answer in the manifest
            (A-2). Off by default because it costs one extra authenticated GET per
            compile. The recorded `public_confirmed` is deliberately three-state --
            `True`/`False`/`None` -- and never collapses an unanswered question into
            "private"; see `_confirm_visibility`.
        ephemeral: Forwarded to upstream `compile`/`compile_async` as-is; see the SDK's
            own documentation for its effect.
        compile_retries: Extra attempts for the actual `paw.compile`/`paw.compile_async`
            HTTP call, after a connect failure (`httpx.ConnectTimeout`/`ConnectError` --
            the request provably never reached the server), with a short backoff between
            attempts. A 5xx other than 504 is additionally retried on the **synchronous**
            `paw.compile` path only; `compile_async` (the paid finetune submission) never
            retries a 5xx, because a duplicate there both double-bills and orphans the
            first attempt's `job_id` (A-1). A read timeout, a 504, or a 4xx (bad request,
            invalid API key, rate limit) is never retried on either path -- see
            `_invoke_compile`'s docstring for the full policy and its residual risk.
            `0` disables retrying.
        sdk: Test seam. Any object exposing `compile`, `compile_async`,
            `get_compile_status` and `function` with the upstream signatures. Defaults to
            the real `programasweights` module, imported lazily on first use.
    """

    #: Base delay between compile retries, multiplied by the attempt number (1, 2, ...).
    _COMPILE_RETRY_BACKOFF_S = 0.5

    def __init__(
        self,
        compiler: str = FAST_COMPILER,
        *,
        n_gpu_layers: Optional[int] = None,
        n_ctx: int = 2048,
        max_tokens: Optional[int] = None,
        offline: bool = False,
        constrained_decoding: bool = True,
        max_spec_examples: int = 16,
        poll_interval_s: float = 5.0,
        compile_timeout_s: float = 3600.0,
        public: bool = False,
        verify_visibility: bool = False,
        ephemeral: bool = False,
        compile_retries: int = 1,
        sdk: Any = None,
    ) -> None:
        self.compiler = compiler
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.max_tokens = max_tokens
        self.offline = offline
        self.constrained_decoding = constrained_decoding
        # `applies_grammar_constraint` is an INSTANCE attribute (X-1 / this track's
        # Phase 3), not the class-level `False` this backend used to hard-code (module
        # docstring, limitation 2). Money route (i) from the track's safety invariants:
        # requested but the engine is not importable -> flag False, exactly one
        # UserWarning here at construction, never a raise on every `infer()` call.
        # `_get_function` may additionally flip this to False post-construction (money
        # route iii) if the vocabulary object built from a loaded model fails its own
        # verification; it never flips it back to True.
        engine_importable = importlib.util.find_spec("llguidance") is not None
        self.applies_grammar_constraint = constrained_decoding and engine_importable
        if constrained_decoding and not engine_importable:
            warnings.warn(
                "ProgramAsWeightsBackend was constructed with constrained_decoding=True "
                "(the default) but the 'llguidance' package is not importable, so "
                "grammar-constrained decoding is unavailable for this instance. Install "
                "the 'paw' extra (pip install 'paw-kit[paw]') to enable it. Output is "
                "still validated after generation by paw_kit.schema.load instead "
                "(fail-open on mismatch).",
                UserWarning,
                stacklevel=2,
            )
        self.max_spec_examples = max_spec_examples
        self.poll_interval_s = poll_interval_s
        self.compile_timeout_s = compile_timeout_s
        # A-9: validate rather than trust. `public` is forwarded verbatim to a service
        # whose own default is `True`, so every value that is not the literal `True`
        # must resolve to `False` *here* -- not at the request, and not via `bool()`,
        # which would publish on `1` or `"true"`. Coercing in `__init__` also makes the
        # manifest record `False` instead of echoing back whatever was passed.
        self.public = public is True
        self.verify_visibility = verify_visibility
        self.ephemeral = ephemeral
        self.compile_retries = compile_retries
        self._sdk = sdk
        # Fourth element is the Vocabulary object built for this cached function (None
        # if grammar-constrained decoding is off, unavailable, or failed verification
        # for this program) -- cached BESIDE the function so LRU eviction (A-11) drops
        # both together, since the encoder inside Vocabulary is a live callable bound to
        # this exact loaded model.
        self._functions: (
            "OrderedDict[str, Tuple[Tuple[int, int, int], str, Callable[..., str], "
            "Optional[Vocabulary]]]"
        ) = OrderedDict()
        # `infer()`'s warn-once set for `ConstraintUnavailable` (a grammar refused AT
        # CONSTRUCTION -- fuel/grammar refused, or an empty/EOS-only initial mask):
        # keyed on a hash of the pattern text, bounded to
        # `_MAX_WARNED_CONSTRAINT_UNAVAILABLE_GRAMMARS` distinct grammars, LRU-evicted
        # the same shape as `_functions` above. See `_warn_constraint_unavailable_once`.
        self._constraint_unavailable_warned: "OrderedDict[str, None]" = OrderedDict()
        self._lock = threading.Lock()

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
        """True if the SDK is importable (or injected).

        Proves that the `programasweights` package can be imported (or an injected double
        is present), but does not check whether the underlying llama.cpp runtime works,
        whether GPU offload is functional, or whether model weights are cached. Does not
        check the API key: inference on an already-cached program works without one.
        Use `paw-kit doctor` for deeper environment and runtime diagnostics.
        """
        return self._sdk is not None or _sdk_installed()

    def has_api_key(self) -> bool:
        return bool(os.environ.get("PAW_API_KEY"))

    # ------------------------------------------------------------------ compile

    def compile(self, spec: str, examples: List[Dict[str, str]], output_path: str) -> str:
        paw = self._paw()
        # A-9: `offline=True` documents "never touch the network", and it skipped the
        # API-key guard below and then POSTed to the compile service anyway. Raising
        # here -- at the top, beside the existing API-key raise, before any paid work
        # and on nothing the request path depends on -- is the only placement that does
        # not weaken the campaign's fail-open invariant. Every in-repo construction of
        # `offline=True` (the measurement scripts under `scripts/`) is inference-only
        # and never calls compile().
        if self.offline:
            raise RuntimeError(
                "ProgramAsWeightsBackend was constructed with offline=True, which means "
                "never contact the compile service. Compilation is a network operation: "
                "construct a second backend without offline=True to compile, or call "
                "prepare_program() to populate the cache this instance reads from."
            )
        # The `not self.offline and` this condition used to carry is dropped, not
        # overlooked: the raise above makes it unreachably false, and leaving it in would
        # tell a reader that an offline backend can still get here.
        if not self.has_api_key():
            raise RuntimeError(
                "Compilation needs PAW_API_KEY in the environment "
                "(https://programasweights.com/settings). Inference on cached programs does not."
            )

        full_spec = _render_spec_with_examples(spec, examples, self.max_spec_examples)
        # A-6: what was actually folded, not what was offered. This was
        # `min(len(examples), self.max_spec_examples)`, which counted every example handed
        # in -- including malformed ones the renderer silently skipped -- and that inflated
        # number was mirrored into every published measurement artifact.
        #
        # `len(folded_ids)` is the plausible wrong answer and is rejected deliberately:
        # `select_folded_examples` requires only that the `input`/`output` *keys* exist,
        # which is exactly what the renderer folds, while `example_id` additionally
        # requires both *values* be `str`. So `{"input": 3, "output": 4}` reaches the spec
        # but yields no id, and counting ids would **under**-report what was published.
        # Two honest numbers, not one guess: this is what went in, `folded_example_ids` is
        # the subset that could be identified.
        folded_count = len(select_folded_examples(examples, self.max_spec_examples))
        folded_ids = folded_example_ids(examples, self.max_spec_examples)

        # Read whatever manifest already sits at output_path *before* it is
        # overwritten below -- this is the only point at which its lineage is still
        # recoverable (see manifest_lineage.read_parent_lineage's docstring).
        parent_program_id, parent_manifest_sha256 = read_parent_lineage(output_path, _MAX_MANIFEST_BYTES)

        if self.public and folded_count > 0:
            warnings.warn(
                f"ProgramAsWeightsBackend.compile() is about to publish this spec, "
                f"including {folded_count} traced example(s), publicly on "
                "programasweights.com: public=True lists the full spec text, readable "
                "without login. Pass public=False (the default), or add "
                "redact_trace=True to the @compile_on_hit decorator, if that traffic "
                "should stay private.",
                UserWarning,
                stacklevel=2,
            )

        # Upstream's compile cache is keyed on spec text alone and ignores `public` on a
        # hit: compiling an already-cached spec with public=False can still return an
        # existing *public* program, unchanged. Best-effort check; never let a precheck
        # failure block the compile itself.
        cache_hit: Optional[bool] = None
        cached_program_id: Optional[str] = None
        if not self.public:
            try:
                precheck = paw.precheck_compile(full_spec, compiler=self.compiler)
                cache_hit = bool(precheck.get("cached")) if isinstance(precheck, dict) else bool(
                    getattr(precheck, "cached", False)
                )
                # A-2, the free half: `CompilePrecheck` is a plain dict at runtime and
                # `program_id` is a real key on it -- the id of the existing program a
                # cache hit will hand back. The old code fetched this and discarded it,
                # leaving a warning that said "an existing program will be returned"
                # without ever naming which one. No extra request.
                raw_id = (
                    precheck.get("program_id") if isinstance(precheck, dict)
                    else getattr(precheck, "program_id", None)
                )
                cached_program_id = raw_id if isinstance(raw_id, str) and raw_id else None
            # A-3: narrowed from a bare `except Exception`. `APIError` subclasses
            # `httpx.HTTPStatusError`, so `httpx.HTTPError` covers every way the
            # *service* can decline this request while still letting a broken contract
            # through -- an upstream rename of `precheck_compile` raises AttributeError,
            # which the bare except absorbed, silently disabling the cache-hit leak
            # warning forever with nothing in the suite noticing.
            except httpx.HTTPError as exc:
                cache_hit = None
                # Distinct from the cache-hit warning below, per the finding: silence
                # was indistinguishable from "checked, and there is no cache hit",
                # which is the opposite conclusion.
                warnings.warn(
                    "ProgramAsWeights could not check whether this spec is already "
                    f"compiled ({type(exc).__name__}: {exc}). If it is, that existing "
                    "program is returned unchanged and public=False cannot make it "
                    "private -- proceeding without that warning, not without that risk.",
                    UserWarning,
                    stacklevel=2,
                )
            if cache_hit:
                named = f" ({cached_program_id})" if cached_program_id else ""
                warnings.warn(
                    f"ProgramAsWeights already has a compiled program{named} for this "
                    "exact spec and will return it instead of compiling a new one. "
                    "paw-kit cannot "
                    "change that existing program's public/private visibility -- if it was "
                    "compiled public, it stays public regardless of public=False here. "
                    "Rephrase the spec (or its folded examples) if you need a fresh, "
                    "private compile. Pass verify_visibility=True to record what the "
                    "server actually reports for it.",
                    UserWarning,
                    stacklevel=2,
                )

        compile_started = time.monotonic()
        if self.compiler == FINETUNE_COMPILER:
            # A-1: no `retry_5xx=True` here, deliberately. See `_invoke_compile`.
            job = self._invoke_compile(
                paw.compile_async,
                full_spec, compiler=self.compiler, public=self.public, ephemeral=self.ephemeral,
            )
            program_id, slug, status, compiler_snapshot = self._wait_for_job(paw, job)
        else:
            program = self._invoke_compile(
                paw.compile,
                full_spec, compiler=self.compiler, public=self.public, ephemeral=self.ephemeral,
                retry_5xx=True,
            )
            program_id = getattr(program, "id", None) or (program.get("id") if isinstance(program, dict) else None)
            slug = getattr(program, "slug", None) or (program.get("slug") if isinstance(program, dict) else None)
            status = getattr(program, "status", None) or (program.get("status") if isinstance(program, dict) else None)
            if not program_id or (status and str(status).lower() in _FAILED_STATES):
                err = getattr(program, "error", None) or (program.get("error") if isinstance(program, dict) else None)
                raise RuntimeError(f"ProgramAsWeights compile failed (status={status!r}): {err!r}")
            compiler_snapshot = extract_snapshot(program)
        compile_wall_s = time.monotonic() - compile_started

        # A-2: ask the server what this program's visibility actually is. Everything
        # about this call is arranged so that no outcome of it can prevent the manifest
        # write below -- at this point the compile has been *billed* and `program_id` is
        # the only record of it.
        public_confirmed, public_confirmed_reason = self._confirm_visibility(paw, program_id)

        manifest = {
            "backend": MANIFEST_BACKEND_NAME,
            "manifest_version": MANIFEST_VERSION,
            "program_id": program_id,
            "slug": slug,
            "compiler": self.compiler,
            "status": status,
            "spec": spec,
            "spec_sha256": sha256_text(spec),
            "full_spec_sha256": sha256_text(full_spec),
            "examples_folded_into_spec": folded_count,
            "examples_count": len(examples),
            "folded_example_ids": folded_ids,
            # A-2: `public` used to sit here, recording `self.public` -- i.e. what was
            # *asked for* -- under a name every reader took for what the program's
            # visibility *is*. Confirmed live (parent track doc, §"Report §16 live
            # verification"): all six historical programs report `public: True` from the
            # server. The two are now separate keys, and the confirmed one is
            # three-state so that "never checked" can never be mistaken for "private".
            "public_requested": self.public,
            "public_confirmed": public_confirmed,
            "public_confirmed_reason": public_confirmed_reason,
            "ephemeral": self.ephemeral,
            "cache_hit": cache_hit,
            "cached_program_id": cached_program_id,
            "parent_program_id": parent_program_id,
            "parent_manifest_sha256": parent_manifest_sha256,
            "compile_wall_s": compile_wall_s,
            "compiler_snapshot": compiler_snapshot,
            "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_write_text(output_path, json.dumps(manifest, indent=2))
        append_history_entry(output_path, manifest)

        # Drop any loaded function for this path: the program behind it just changed.
        with self._lock:
            self._functions.pop(output_path, None)
        return output_path

    #: Keys `get_program_meta`'s JSON may carry the program's visibility under, in
    #: preference order. Only a real `bool` is accepted: a string `"true"` or an int `1`
    #: is a shape this code has never seen from the service, and guessing at one is how
    #: a visibility claim becomes wrong rather than unknown.
    _VISIBILITY_KEYS = ("public", "is_public")

    @staticmethod
    def _paw_client(paw: Any) -> Any:
        """A `PAWClient`-shaped object for the calls that have no module-level wrapper.

        `get_program_meta` is a `PAWClient` method upstream and `PAWClient` is not even
        in `programasweights.__all__`, so it is reached through the submodule. The
        `getattr` first is the test seam: an injected `sdk` supplies its own factory.
        """
        factory = getattr(paw, "PAWClient", None)
        if factory is None:
            factory = importlib.import_module("programasweights.client").PAWClient
        kwargs: Dict[str, Any] = {}
        for name, key in (("get_api_url", "api_url"), ("get_api_key", "api_key")):
            getter = getattr(paw, name, None)
            if callable(getter):
                kwargs[key] = getter()
        return factory(**kwargs)

    def _confirm_visibility(
        self, paw: Any, program_id: Optional[str]
    ) -> "tuple[Optional[bool], str]":
        """A-2: `(public_confirmed, reason)` -- the server's answer, or why there isn't one.

        Three states, never two. `None` means *unknown*, and a missing answer must never
        render as "private": that is Pattern 5 (a control degrading silently into a
        reassuring value) and it is the entire shape of A-2, whose own finding is a
        manifest field recording a request while reading as a fact.

        **Nothing this method does can propagate.** It runs after the compile has been
        billed and before the manifest is written, so an expired key, a 404, a transient
        5xx or an upstream rename must not cost the caller the `program_id` of a compile
        they have already paid for (Phase 0 F8). The blanket `except BaseException` is
        deliberate and is the narrow case where one is correct: the alternative to
        swallowing is losing money.
        """
        if not self.verify_visibility:
            return None, "not_attempted"
        # Belt for A-9's raise: an offline backend may never make this request either.
        if self.offline:
            return None, "offline"
        if not program_id:
            return None, "no_program_id"
        if not self.has_api_key():
            return None, "no_api_key"
        try:
            meta = self._paw_client(paw).get_program_meta(str(program_id))
        except BaseException as exc:  # noqa: BLE001 -- see docstring
            return None, f"request_failed: {type(exc).__name__}: {exc}"
        if not isinstance(meta, dict):
            return None, "no_visibility_key"
        for key in self._VISIBILITY_KEYS:
            value = meta.get(key)
            if isinstance(value, bool):
                return value, "server"
        return None, "no_visibility_key"

    def _invoke_compile(
        self,
        fn: Callable[..., Any],
        *args: Any,
        retry_5xx: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Call `fn` (`paw.compile` or `paw.compile_async`), converting an httpx
        connect failure or HTTP error response into a `RuntimeError` naming the
        service and what happened -- rather than letting a raw httpx exception (whose
        message says nothing about which call failed or what to do about it) reach
        the caller.

        Retry policy, deliberately narrow: `paw.compile`/`paw.compile_async` is a POST
        that is *not idempotent*. Resubmitting it after the server has already seen the
        request queues a second compile, spends a second unit of the rate-limited
        quota, and (on the finetune path) discards the first attempt's `job_id`, making
        that job unpollable and uncancellable. So only
        `httpx.ConnectTimeout`/`httpx.ConnectError` (the TCP handshake itself failed or
        timed out -- the request provably never reached the server) are retried
        unconditionally, up to `self.compile_retries` additional times with a short
        backoff. Every other timeout (`httpx.ReadTimeout` and friends: the request was
        sent and the server may already be compiling) is raised immediately, *not*
        retried -- the message explains that the compile may still be running
        server-side and that re-running with the same spec will hit the compile cache
        once it finishes, instead of paying for a second compile. A 4xx (bad request,
        invalid API key, rate limit) is never retried either -- retrying it wastes a
        rate-limited attempt on something that will fail again identically.

        **A-1: a 5xx is retried only when the caller opts in with `retry_5xx=True`,
        and only the synchronous `paw.compile` call site does.** A non-504 5xx (a
        gateway 502, or a 500 raised while serialising the response for a job that was
        already enqueued) proves nothing about whether the compile landed, so retrying
        it can buy a second compile. `compile_async` is reachable only via
        `FINETUNE_COMPILER`, where a duplicate compile is 96-223s of paid GPU *and*
        orphans the first `job_id` -- unpollable and uncancellable -- so it never
        retries a 5xx. The unsafe direction requires an explicit opt-in precisely so a
        future third call site cannot inherit the retry silently. 504 (gateway timeout)
        stays excluded even when `retry_5xx=True`: it carries the same "already landed,
        still working" ambiguity as a read timeout.

        **Residual risk on the sync path, stated rather than hidden:** upstream's
        compile cache is keyed on spec text, so a resubmitted identical spec usually
        returns the program the first attempt created -- but only once that program is
        actually cached. If the first POST landed and the retry arrives before it is,
        the retry buys a second *fast* compile. That is bounded to the fast compiler and
        to `self.compile_retries` (default 1) extra attempts, and there is no `job_id`
        to orphan. The cache does not make the retry free.
        """
        attempt = 0
        while True:
            try:
                return fn(*args, **kwargs)
            except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
                if attempt < self.compile_retries:
                    attempt += 1
                    time.sleep(self._COMPILE_RETRY_BACKOFF_S * attempt)
                    continue
                raise RuntimeError(
                    "ProgramAsWeights compile service was unreachable after "
                    f"{attempt + 1} attempt(s) (connection never established). Run "
                    "`paw-kit doctor` to check service health."
                ) from exc
            except httpx.TimeoutException as exc:
                raise RuntimeError(
                    "ProgramAsWeights compile service timed out waiting for a "
                    f"response ({type(exc).__name__}). This was not retried: the "
                    "request may already have reached the server, and the compile "
                    "may still be running there. Re-running this compile with the "
                    "same spec will hit the compile cache once it finishes, instead "
                    "of starting a second one -- do not assume this attempt failed "
                    "outright."
                ) from exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                body = exc.response.text[:300] if exc.response is not None else ""
                retryable_5xx = (
                    retry_5xx
                    and status_code is not None
                    and 500 <= status_code < 600
                    and status_code != 504
                )
                if retryable_5xx and attempt < self.compile_retries:
                    attempt += 1
                    time.sleep(self._COMPILE_RETRY_BACKOFF_S * attempt)
                    continue
                detail = f" Response: {body}" if body else ""
                if status_code is not None and 400 <= status_code < 500:
                    # PAW-DOCTOR advice diagnoses nothing here: a 4xx is a client-side
                    # problem (bad request, invalid key, rate limit), not a service
                    # health issue -- the response body (appended above) carries the
                    # actual reason.
                    raise RuntimeError(
                        f"ProgramAsWeights compile service returned HTTP {status_code} "
                        f"after {attempt + 1} attempt(s).{detail}"
                    ) from exc
                raise RuntimeError(
                    f"ProgramAsWeights compile service returned HTTP {status_code} "
                    f"after {attempt + 1} attempt(s). Run `paw-kit doctor` to check "
                    f"service health.{detail}"
                ) from exc

    #: How many *consecutive* transient poll failures `_wait_for_job` tolerates
    #: before giving up. Unlike the initial compile submission (`_invoke_compile`),
    #: `get_compile_status` is a GET and idempotent -- polling it again after a
    #: failure changes nothing server-side, so it is safe to retry liberally here.
    _MAX_CONSECUTIVE_POLL_FAILURES = 5
    #: Backoff applied after the 1st..5th consecutive poll failure, in order (index 0
    #: for the 1st failure, ... index 4 for the 5th); the 5th value (60s) is reused
    #: for any failure beyond the 5th, but by then `_MAX_CONSECUTIVE_POLL_FAILURES`
    #: has already raised, so in practice this list is never indexed past its end.
    _POLL_FAILURE_BACKOFF_S: "tuple[float, ...]" = (5.0, 10.0, 20.0, 40.0, 60.0)

    def _wait_for_job(self, paw: Any, job: Any) -> tuple[str, Optional[str], str, Any]:
        """Poll `get_compile_status` until the job reaches a terminal state.

        A single transient failure while polling (an httpx timeout, a connect error,
        or a 5xx response) used to propagate as a raw exception and throw away an
        in-progress compile that might be an hour into a finetune. Status polling is
        idempotent (a GET, unlike the compile submission itself), so this tolerates up
        to `_MAX_CONSECUTIVE_POLL_FAILURES` *consecutive* such failures, backing off
        between retries per `_POLL_FAILURE_BACKOFF_S`, and resets the failure count on
        any successful poll. Once that many failures happen in a row, it gives up and
        raises a `RuntimeError` naming `job_id` so the caller can poll the job again
        later by hand -- the compile itself may well still be running. The overall
        `compile_timeout_s` wall-clock cap still applies on top of this.

        A-5: upstream publishes no enum of job states, so `_SUCCESS_STATES` and
        `_FAILED_STATES` are both guesses about a vocabulary that can grow. Anything
        outside them used to fall through to the sleep loop and be polled for the whole
        `compile_timeout_s` -- 720 GETs at the defaults -- while the server's own `error`
        string sat in the response unread. A status *with a non-null `error` and no
        `program_id`* is therefore treated as terminal failure whatever it is called
        (`infrastructure_error`, `redis_unavailable`, or a `completed` that named no
        program), with `error` surfaced verbatim. A status this code does not recognise
        and that carries **no** error is still polled: a server is free to introduce a
        new *transient* state, and failing a compile the user has already paid for
        because its status string is unfamiliar would be strictly worse than waiting.

        When the poll loop itself times out, `cancel_compile` is attempted once
        (best-effort, never masking the timeout) -- the job is queued and billable, and
        nothing else in this process is ever going to come back for it.
        """
        job_id = job.get("job_id") if isinstance(job, dict) else getattr(job, "job_id", None)
        if not job_id:
            raise RuntimeError(f"compile_async returned no job_id: {job!r}")

        deadline = time.monotonic() + self.compile_timeout_s
        consecutive_failures = 0
        while True:
            failure_exc: Optional[Exception] = None
            status_obj: Any = None
            try:
                status_obj = paw.get_compile_status(job_id)
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                failure_exc = exc
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code is not None and 500 <= status_code < 600:
                    failure_exc = exc
                else:
                    raise

            if failure_exc is not None:
                consecutive_failures += 1
                if consecutive_failures > self._MAX_CONSECUTIVE_POLL_FAILURES:
                    raise RuntimeError(
                        f"ProgramAsWeights compile {job_id} could not be polled: "
                        f"{consecutive_failures} consecutive transient failures "
                        f"(last: {type(failure_exc).__name__}: {failure_exc}). The "
                        "compile may still be running server-side -- poll it again "
                        f"later with this job_id: {job_id}"
                    ) from failure_exc
                if time.monotonic() >= deadline:
                    self._cancel_job_best_effort(paw, job_id)
                    raise TimeoutError(
                        f"ProgramAsWeights compile {job_id} still unreachable after "
                        f"{self.compile_timeout_s}s"
                    ) from failure_exc
                backoff_idx = min(consecutive_failures, len(self._POLL_FAILURE_BACKOFF_S)) - 1
                time.sleep(self._POLL_FAILURE_BACKOFF_S[backoff_idx])
                continue

            consecutive_failures = 0
            get = status_obj.get if isinstance(status_obj, dict) else lambda k, d=None: getattr(status_obj, k, d)
            status = str(get("status") or "").lower()
            program_id = get("program_id")
            error = get("error")
            if status in _FAILED_STATES:
                raise RuntimeError(f"ProgramAsWeights finetune compile {job_id} {status}: {error!r}")
            if program_id and (status in _SUCCESS_STATES or get("completed_at")):
                return str(program_id), get("slug"), status, extract_snapshot(status_obj)
            # A-5: the job reported a problem and named no program. Whatever the status
            # string is, there is nothing left to wait for -- fail now, with the
            # server's own explanation, instead of polling until compile_timeout_s.
            if error is not None and not program_id:
                raise RuntimeError(
                    f"ProgramAsWeights finetune compile {job_id} reported status "
                    f"{status!r} with no program_id and an error, so it is treated as "
                    f"terminal rather than polled further: {error!r}"
                )
            if time.monotonic() >= deadline:
                self._cancel_job_best_effort(paw, job_id)
                raise TimeoutError(
                    f"ProgramAsWeights compile {job_id} still {status!r} after {self.compile_timeout_s}s"
                )
            time.sleep(self.poll_interval_s)

    @staticmethod
    def _cancel_job_best_effort(paw: Any, job_id: str) -> None:
        """A-5: ask the service to cancel a job this process has stopped waiting for.

        Strictly best-effort. The caller is already raising, and the reasons this can
        fail are all ones that must not replace the timeout the caller needs to see: an
        SDK too old to expose `cancel_compile`, a 409 because the job has already
        started (upstream documents that response), or the same network fault that
        caused the timeout in the first place.
        """
        cancel = getattr(paw, "cancel_compile", None)
        if not callable(cancel):
            return
        try:
            cancel(job_id)
        except Exception:
            warnings.warn(
                f"ProgramAsWeights compile {job_id} timed out and could not be "
                "cancelled; it may still be running (and billable) server-side.",
                UserWarning,
                stacklevel=2,
            )

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

    def _build_vocabulary(self, fn: Callable[..., str]) -> Vocabulary:
        """Build and verify a `paw_kit.schema.constraint.Vocabulary` from the
        llama_cpp model loaded behind `fn`.

        `fn._llm` is the upstream SDK's private `llama_cpp.Llama` handle
        (`runtime_llamacpp.py`'s `PawFunction._llm`) -- the only place this class
        reaches into an SDK internal, and only to read the loaded model, never to
        drive it directly. Tokens are read via `detokenize()` of every id (byte-fallback
        tokens included, unfiltered -- owner decision 2), EOS via `token_eos()`, and the
        encode callable is bound live to this model's own tokenizer
        (`llm.tokenize(..., add_bos=False, special=False)`, which round-trips the
        `Vocabulary` construction-time probes -- verified live across four measurement
        rounds). Measured at ~0.553 s per model (0.293 s detokenize + 0.260 s
        tokenizer), which is why this is called at most once per model load, not per
        `infer()` call.

        Raises on any failure (`fn` has no `_llm`, the model object is missing an
        expected method, or `Vocabulary`'s own round-trip verification raises
        `PAWSchemaError`) -- the caller is responsible for catching this, warning once,
        and disabling `applies_grammar_constraint` (money route iii); this method never
        does that itself, so it stays usable standalone (e.g. from tests).
        """
        llm = getattr(fn, "_llm", None)
        if llm is None:
            raise RuntimeError(
                "cannot build a grammar-constraint vocabulary: the loaded function "
                "has no '_llm' attribute exposing the underlying llama_cpp model"
            )
        n_vocab = llm.n_vocab()
        tokens = [llm.detokenize([i]) for i in range(n_vocab)]
        eos_token_id = llm.token_eos()

        def _encode(x: Any) -> List[int]:
            if isinstance(x, str):
                x = x.encode("utf-8")
            return llm.tokenize(x, add_bos=False, special=False)

        return Vocabulary(
            tokens=tokens,
            eos_token_id=eos_token_id,
            special_token_ids=(eos_token_id,),
            encode=_encode,
        )

    def _get_function_and_vocabulary(
        self, adapter_path: str
    ) -> "Tuple[Callable[..., str], Optional[Vocabulary]]":
        # PAW-BACKEND-D2: A single os.stat call does double duty -- its FileNotFoundError
        # is the existence check (letting callers fall open via their existing handlers),
        # and its (mtime_ns, size, ino) triple is the staleness component.
        # Known limitation: this staleness check relies on st_ino changing on recompile
        # (true for the normal case since manifests are written via atomic_write_text's
        # os.replace), and degrades to "no worse than before" (not worse, not better) on
        # filesystems where st_ino is unreliable (e.g. some SMB/FUSE mounts) --
        # mtime and size still move in that case.
        st = os.stat(adapter_path)
        stat_identity = (st.st_mtime_ns, st.st_size, st.st_ino)

        with self._lock:
            cached = self._functions.get(adapter_path)
            if cached is not None:
                cached_stat, cached_program_id, cached_fn, cached_vocab = cached
                if cached_stat == stat_identity:
                    self._functions.move_to_end(adapter_path)
                    return cached_fn, cached_vocab

        # Either a cache miss or stat_identity differed. Re-read manifest to resolve program_id.
        manifest = self.read_manifest(adapter_path)
        program_id = manifest.get("program_id") or manifest["slug"]

        with self._lock:
            cached = self._functions.get(adapter_path)
            if cached is not None:
                cached_stat, cached_program_id, cached_fn, cached_vocab = cached
                if cached_stat == stat_identity:
                    self._functions.move_to_end(adapter_path)
                    return cached_fn, cached_vocab
                if cached_program_id == program_id:
                    # program_id unchanged: refresh stat_identity without reloading model.
                    self._functions[adapter_path] = (
                        stat_identity, program_id, cached_fn, cached_vocab,
                    )
                    self._functions.move_to_end(adapter_path)
                    return cached_fn, cached_vocab
                # program_id changed: evict the stale entry.
                #
                # SAFETY INVARIANT (A-11 / F3.3): drop the reference and NOTHING ELSE.
                # No close(), no _cleanup_resources(), no reset(), no teardown of any
                # kind. `infer()` releases `self._lock` before calling the cached
                # callable, so an in-flight caller's own reference is the only thing
                # keeping the underlying llama.cpp model alive once this entry is
                # evicted -- closing/resetting it here would free that model out from
                # under a running call. See bug-hunt-D-money-privacy.md's disposition
                # of A-11 for the full analysis; this comment is intentionally
                # unmissable because D-2 is what turns eviction from rare into
                # routine, which is exactly what makes "let's free it too" look like
                # a plausible next edit.
                self._functions.pop(adapter_path, None)

        kwargs: Dict[str, Any] = {"n_ctx": self.n_ctx, "offline": self.offline}
        if self.n_gpu_layers is not None:
            kwargs["n_gpu_layers"] = self.n_gpu_layers

        paw = self._paw()
        try:
            fn = paw.function(program_id, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"ProgramAsWeights failed to load function for program {program_id!r}: {exc}. "
                "Run `paw-kit doctor` to diagnose environment issues (checks llama_cpp importability, "
                "GPU offload support, and local model cache)."
            ) from exc

        # Money route (iii): a vocabulary that fails to build or verify degrades this
        # INSTANCE (not just this call) to unconstrained decoding, with exactly one
        # UserWarning -- never a raise on every subsequent `infer()` call, and never a
        # silent no-op. Only attempted when the flag is still True: once it has been
        # flipped False (here, or at construction), later model loads on this instance
        # do not re-attempt vocabulary construction.
        vocabulary: Optional[Vocabulary] = None
        if self.applies_grammar_constraint:
            try:
                vocabulary = self._build_vocabulary(fn)
                # P-2: force the engine import and `LLTokenizer` construction HERE,
                # inside this same try, rather than leaving it for `build_constraint`
                # to do lazily on the first `infer()` call. `applies_grammar_constraint`
                # is decided at construction by `find_spec("llguidance")` alone, which
                # only proves the package's on-disk metadata is present -- a present
                # but unimportable engine (an ABI-mismatched or partially-installed
                # native extension, ../docs/install.md's glibc floor) previously
                # surfaced as a plain `PAWSchemaError` from `build_constraint`'s own
                # `import llguidance`, on every single call, with the flag still True
                # and no `UserWarning` -- an unclosed fourth money route. Calling this
                # here lands that failure in the SAME route as a failing vocabulary
                # (iii), with the one warning below. Free: `llguidance_tokenizer()` is
                # cached on `Vocabulary` and would be built on the first `infer()` call
                # for this model regardless -- this only moves that cost earlier, to
                # model load, which is once per model, not once per call.
                vocabulary.llguidance_tokenizer()
            except Exception as exc:
                self.applies_grammar_constraint = False
                warnings.warn(
                    "ProgramAsWeightsBackend could not build/verify the grammar-"
                    f"constraint vocabulary for program {program_id!r} "
                    f"({type(exc).__name__}: {exc}); grammar-constrained decoding is "
                    "disabled for this backend instance from now on. Output is still "
                    "validated after generation by paw_kit.schema.load instead "
                    "(fail-open on mismatch).",
                    UserWarning,
                    stacklevel=2,
                )
                vocabulary = None

        with self._lock:
            self._functions[adapter_path] = (stat_identity, program_id, fn, vocabulary)
            self._functions.move_to_end(adapter_path)
            while len(self._functions) > _MAX_CACHED_FUNCTIONS:
                # SAFETY INVARIANT (A-11 / F3.3): LRU eviction drops the reference only --
                # no teardown or resource release on evicted callable. Cached BESIDE the
                # function, so eviction drops the vocabulary's live, model-bound encoder
                # at the same time as the function it is bound to -- never separately.
                self._functions.popitem(last=False)
        return fn, vocabulary

    def _get_function(self, adapter_path: str) -> Callable[..., str]:
        """Back-compat wrapper: `infer()` uses `_get_function_and_vocabulary()`
        directly (it needs both), but this stays the fn-only entry point other callers
        (tests, `scripts/`) already use."""
        fn, _vocabulary = self._get_function_and_vocabulary(adapter_path)
        return fn

    def _warn_constraint_unavailable_once(self, pattern: str, exc: BaseException) -> None:
        """`infer()`'s degrade-per-schema path for `ConstraintUnavailable` (see
        `constraint.py`'s module docstring and `ConstraintUnavailable`'s own
        docstring): a grammar refused AT CONSTRUCTION is a property of `pattern`, not
        of one call, so re-warning on every call for a schema that can never construct
        would flood the log exactly the way S-14's own warn-once mitigates for a
        mid-generation failure -- except this is a DIFFERENT condition (construction,
        not generation) with a DIFFERENT remedy (skip building the constraint for this
        grammar, don't retry-and-warn on every call), so it gets its own bounded
        warn-once state rather than reusing S-14's per-bound-function one, which lives
        in `loader.py` and knows nothing about grammar patterns.

        Keyed on a hash of the pattern text, not the text itself: the whole point of
        `ConstraintUnavailable` is that the pattern can be enormous (an oversized
        `Literal`/alternation is exactly the refused case), so storing pattern text
        verbatim as a dict key would defeat the purpose of bounding this set at all.
        Bounded to `_MAX_WARNED_CONSTRAINT_UNAVAILABLE_GRAMMARS` distinct grammars,
        LRU-evicted like `_functions`; losing an old entry and warning again later for
        a schema not seen in a while costs one extra log line, never a raise and never
        a teacher call, so correctness does not depend on this cache remembering
        forever. Does not itself retry the build -- the caller decides that -- and a
        retry that raises `ConstraintUnavailable` again is fine: this method just makes
        sure it does not warn a second time for the same (still-cached) pattern.
        """
        key = hashlib.sha256(pattern.encode("utf-8")).hexdigest()
        with self._lock:
            already_warned = key in self._constraint_unavailable_warned
            if already_warned:
                self._constraint_unavailable_warned.move_to_end(key)
            else:
                self._constraint_unavailable_warned[key] = None
                while len(self._constraint_unavailable_warned) > _MAX_WARNED_CONSTRAINT_UNAVAILABLE_GRAMMARS:
                    self._constraint_unavailable_warned.popitem(last=False)
        if already_warned:
            return
        warnings.warn(
            "ProgramAsWeightsBackend: grammar-constrained decoding is unavailable for "
            f"this schema ({type(exc).__name__}: {exc}); this call, and every other "
            "call for this same grammar, proceeds UNCONSTRAINED -- output is still "
            "validated after generation by paw_kit.schema.load (fail-open on "
            "mismatch). Other schemas on this backend instance are unaffected; this "
            "instance's applies_grammar_constraint stays True. Further calls for this "
            "same grammar do not warn again.",
            UserWarning,
            stacklevel=2,
        )

    def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
        fn, vocabulary = self._get_function_and_vocabulary(adapter_path)
        call_kwargs: Dict[str, Any] = {}
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens

        # Re-read `self.applies_grammar_constraint` AFTER `_get_function_and_vocabulary()`
        # returns, not before: `paw.load` reads the attribute once at bind time, so a
        # vocabulary-verification failure that just happened during THIS call's model
        # load (money route iii) must still produce local, unconstrained output on THIS
        # call -- not a grammar-constraint build against a `vocabulary` that is already
        # None, and not a wait until the next call to notice the flag flipped.
        if grammar_constraint is not None and self.applies_grammar_constraint and vocabulary is not None:
            import llama_cpp  # bundled by the SDK; only imported when actually needed

            try:
                constraint = build_constraint(grammar_constraint, vocabulary)
            except ConstraintUnavailable as exc:
                # A construction-time refusal degrades PER SCHEMA, not per call: the
                # instance flag stays True (the engine works; only this grammar is
                # refused), exactly one warning fires for this pattern, and this call
                # (and every other call for the same grammar) runs unconstrained --
                # `call_kwargs` simply never gains a `logits_processor` key. Not
                # catching plain `PAWSchemaError` here: a mid-generation failure must
                # still propagate below and reach `paw.load`'s fallback as itself.
                self._warn_constraint_unavailable_once(grammar_constraint, exc)
            else:
                call_kwargs["logits_processor"] = llama_cpp.LogitsProcessorList([constraint])

        try:
            out = fn(input_text, **call_kwargs)
        except PAWSchemaError:
            # A masking failure (from `constraint`, propagated unwrapped through the
            # SDK's own `guarded_processor` re-raise) must reach `paw.load`'s fail-open
            # fallback AS `PAWSchemaError`, not wrapped below as a `RuntimeError` whose
            # remedy sends the caller to check GPU offload for what is actually a
            # grammar/masking error.
            raise
        except Exception as exc:
            raise RuntimeError(
                f"ProgramAsWeights inference failed for {adapter_path!r}: {exc}. "
                "This is a runtime/environment failure, not a grammar-constraint error "
                "-- those propagate as PAWSchemaError and are never wrapped here. Run "
                "`paw-kit doctor` to diagnose environment issues (checks llama_cpp "
                "importability, GPU offload support, and local model cache)."
            ) from exc
        return out if isinstance(out, str) else str(out)

    def reset(self) -> None:
        """Drop every loaded function (frees the underlying llama.cpp models)."""
        with self._lock:
            self._functions.clear()
