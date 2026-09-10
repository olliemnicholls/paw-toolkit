"""Adapter-vs-adapter comparison: run every suite case through two adapters and diff them.

Lifted from an ad hoc workflow used to produce the "Finetune compiler" section of
`measurements/README.md`: the headline finding there (132/134 byte-identical outputs
between `paw-4b-qwen3-0.6b` and `paw-ft-bs48`, two adversarial-probe outliers) was decided
by a per-case diff, not by the aggregate structural/semantic percentages -- both of which
sat inside the LLM judge's own measured +-2-point run-to-run noise (see
`paw_kit.test.judge`'s module docstring and `conductor/deferred/index.md`, "Semantic judge
is non-deterministic"). This module is that per-case diff, made reusable: it runs both
adapters through the same backend, reuses `evaluate_assertion` (does not duplicate it) to
determine pass/fail, and reports which cases actually differ -- deliberately not just an
aggregate pass-rate delta, which is exactly what buried the finding the first time.
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.runner import evaluate_assertion
from paw_kit.test.suite import TestSuiteConfig

# Mirrors paw_kit.cli's _MAX_MANIFEST_BYTES / _declared_adapter_backend cap for the same
# reason: a manifest is a few hundred bytes to a few KB of JSON, and this is only ever
# read for *display* metadata in a report, not to drive any write.
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024


def read_adapter_manifest(adapter_path: str) -> Dict[str, Any]:
    """Best-effort manifest read for either backend's `.paw` artifact.

    Tries the strict `ProgramAsWeightsBackend.read_manifest` first (the existing manifest
    reader) so a real adapter's `compiler`/`program_id` land in the report unchanged. If
    that raises (not that backend's manifest format -- true of every `MockPAWBackend`
    adapter and any third-party `.paw` file), falls back to a generic bounded JSON read
    so whatever the file *does* declare (e.g. mock's `spec`/`examples_count`) still shows
    up -- but only when it declares a non-empty string `backend` field, the one key every
    manifest either backend writes actually has. Without that check any unrelated JSON
    dict (`{"foo": "bar"}`) read as a "manifest", which is what let `paw-test compare`'s
    CLI gate (`cli.py`'s `Could not read a manifest ... not the expected shape` check)
    wave through a file that plainly isn't one. Returns `{}` when the file cannot be read
    as JSON at all, or is JSON but not shaped like a manifest.
    """
    try:
        return dict(ProgramAsWeightsBackend.read_manifest(adapter_path))
    except (FileNotFoundError, ValueError):
        pass

    path = Path(adapter_path)
    try:
        if not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return {}
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("backend"), str) or not data["backend"]:
        return {}
    return data


# Display-safe manifest fields for a `CompareReport` (finding 11): a manifest can carry
# `examples`/`rules`/`spec` -- for `ProgramAsWeightsBackend`, `spec` is the full spec
# text with every traced example folded in when `compile_on_hit`'s trace-folding ran
# (see `programasweights.py`'s `_render_spec_with_examples`). `compare --json` is meant
# to report which two adapters were compared, not re-export what may be private training
# data through a side channel never intended to carry it. Only these fields, and only the
# ones actually present, make it into the report.
_MANIFEST_DISPLAY_FIELDS = (
    "backend",
    "program_id",
    "compiler",
    "spec_sha256",
    "compiled_at",
    "manifest_version",
)


def _project_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only `_MANIFEST_DISPLAY_FIELDS` of a manifest, dropping `spec`/`examples`/
    `rules`/`default_response`/everything else that isn't meant for display."""
    return {key: manifest[key] for key in _MANIFEST_DISPLAY_FIELDS if key in manifest}


# `"identical"`/`identical_count` above is a byte comparison, deliberately kept: it is
# what the phone-extraction result in `measurements/README.md` rests on
# (132/134 byte-identical outputs). But on the ticket-triage run in that same document,
# 37 of 60 pairs differed *only* in `json.dumps` spacing -- read literally, "0 identical"
# there said "these are two completely different programs" when the truth was "they
# agree on 62% of cases and differ in a separator". `_MATCH_*` below is the second,
# clearly-named comparison this module now also reports: two outputs are "equivalent"
# when they parse as the same JSON value, or (for anything that doesn't parse as JSON on
# both sides) when they're equal after Unicode NFC normalization and whitespace
# collapse. Every byte-identical pair is trivially also equivalent.
_MATCH_BYTE_IDENTICAL = "byte_identical"
_MATCH_EQUIVALENT = "equivalent"
_MATCH_DIFFERENT = "different"


def _parse_json_or_none(text: str) -> Tuple[bool, Any]:
    """`(True, value)` if `text` parses as JSON, `(False, None)` otherwise."""
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        return False, None


def _normalize_whitespace(text: str) -> str:
    """Unicode NFC normalize, then collapse all whitespace runs (including leading/
    trailing) to single spaces -- `str.split()` with no argument already does the
    collapsing half; NFC first so two visually-identical strings encoded differently
    (e.g. composed vs. decomposed accents) don't register as a difference either."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _outputs_match_kind(output_a: str, output_b: str) -> str:
    """Classify one pair of outputs as byte-identical, equivalent-but-not-identical, or
    genuinely different.

    "Equivalent" means: both parse as JSON and their parsed values are equal (so
    `{"a":1}` and `{"a": 1}` match, key order aside -- `json.loads` returns a `dict`,
    and dict equality is order-independent); otherwise, equal after
    `_normalize_whitespace`. If only one side parses as JSON, that is not "both parse"
    -- comparison falls through to the whitespace-normalized text, on the raw strings,
    not a parsed value compared against unparsed text.
    """
    if output_a == output_b:
        return _MATCH_BYTE_IDENTICAL
    a_is_json, parsed_a = _parse_json_or_none(output_a)
    b_is_json, parsed_b = _parse_json_or_none(output_b)
    if a_is_json and b_is_json:
        equivalent = parsed_a == parsed_b
    else:
        equivalent = _normalize_whitespace(output_a) == _normalize_whitespace(output_b)
    return _MATCH_EQUIVALENT if equivalent else _MATCH_DIFFERENT


class CompareRow(BaseModel):
    """One case run through both adapters."""

    input: str
    output_a: str
    output_b: str
    identical: bool
    # Finding 2: which *kind* of match this row is -- `"byte_identical"`,
    # `"equivalent"` (same value/text after normalization, but not byte-for-byte), or
    # `"different"`. `identical` above is exactly `match_kind == "byte_identical"`,
    # kept as its own field for backward-compatible byte-level reporting.
    match_kind: str = _MATCH_DIFFERENT
    pass_a: bool
    pass_b: bool
    failed_rules_a: List[str] = Field(default_factory=list)
    failed_rules_b: List[str] = Field(default_factory=list)
    latency_a_ms: float = 0.0
    latency_b_ms: float = 0.0
    execution_error_a: str | None = None
    execution_error_b: str | None = None


class CompareReport(BaseModel):
    """Structured report produced by `compare_adapters`."""

    __test__ = False

    task_name: str
    adapter_a: str
    adapter_b: str
    manifest_a: Dict[str, Any] = Field(default_factory=dict)
    manifest_b: Dict[str, Any] = Field(default_factory=dict)
    total_cases: int = 0
    identical_count: int = 0
    # Finding 2: outputs that are the same *value* (same JSON, or the same text once
    # Unicode NFC + whitespace are normalized) even when they're not byte-identical --
    # see `_outputs_match_kind`. Includes every byte-identical row too (byte-identical
    # implies equivalent), so `equivalent_count >= identical_count` always. On the
    # ticket-triage run this module's docstring describes, `identical_count` was 0 and
    # `equivalent_count` would have been 37/60 -- the number that actually mattered.
    equivalent_count: int = 0
    a_pass_count: int = 0
    b_pass_count: int = 0
    only_a_pass_count: int = 0
    only_b_pass_count: int = 0
    # A case counts here if the backend raised for adapter A and/or adapter B on it (see
    # `_infer_safely`) -- independent of `identical`/`pass_a`/`pass_b`, which can't be
    # trusted to surface it: two adapters that both raise produce the same
    # "[EXECUTION_ERROR]" placeholder for both, so they're `identical=True` and agree on
    # `pass_a == pass_b`, and never show up in `differing_rows` at all (finding 1).
    errored_count: int = 0
    rows: List[CompareRow] = Field(default_factory=list)

    @property
    def differing_rows(self) -> List[CompareRow]:
        """Rows where the two adapters disagree -- different output, or different pass
        status. This is deliberately the headline view: a per-case diff is what actually
        decided the finetune-compiler comparison, not the aggregate counts below it.

        Unchanged by finding 2: what counts as a "difference" here is still byte-level
        (`not r.identical`), same as before normalization-aware matching existed --
        `equivalent_only_rows` below is the new, separately-reported split of this same
        set, not a redefinition of it.
        """
        return [r for r in self.rows if not r.identical or r.pass_a != r.pass_b]

    @property
    def equivalent_only_rows(self) -> List[CompareRow]:
        """The subset of `differing_rows` that are only a formatting difference: not
        byte-identical, but `match_kind == "equivalent"` (same JSON value, or the same
        text after whitespace normalization) and the two adapters agree on pass/fail.

        This is the "whitespace-only" list finding 2 asks to report separately: `paw-test
        compare`'s CLI lists these under their own, collapsed heading instead of folding
        them into (or silently dropping them from) the main differences listing.
        """
        return [
            r
            for r in self.differing_rows
            if r.match_kind == _MATCH_EQUIVALENT and r.pass_a == r.pass_b
        ]

    @property
    def genuinely_differing_rows(self) -> List[CompareRow]:
        """`differing_rows` minus `equivalent_only_rows` -- real differences, the ones
        the main "Differences" listing should lead with."""
        equivalent_only_ids = {id(r) for r in self.equivalent_only_rows}
        return [r for r in self.differing_rows if id(r) not in equivalent_only_ids]


def _infer_safely(backend: AbstractPAWBackend, adapter_path: str, inp: str) -> tuple[str, float, str | None]:
    """Run one inference, mirroring `TestRunner.run`'s exception handling (PAW-TEST-08):
    a backend failure becomes a placeholder output plus a separate error field, not a
    crash that would abort the whole comparison over one bad case."""
    t0 = time.perf_counter()
    try:
        out = backend.infer(adapter_path, inp)
        error = None
    except Exception as exc:  # noqa: BLE001 -- see TestRunner.run's identical handling
        out = "[EXECUTION_ERROR]"
        error = str(exc)
    latency_ms = (time.perf_counter() - t0) * 1000
    return out, latency_ms, error


def compare_adapters(
    adapter_a: str,
    adapter_b: str,
    suite: TestSuiteConfig,
    backend: AbstractPAWBackend,
    *,
    include_fuzz: bool = True,
) -> CompareReport:
    """Run every standard case (and, by default, every fuzz case) in `suite` through
    both adapters via `backend`, and diff the results.

    `include_fuzz` defaults to **True** so a default call covers the same cases
    `TestRunner.run`/`paw-test check` does (`paw-test compare --no-fuzz` opts back out
    to standard_cases only, e.g. for a quick look).

    Both adapters are held in memory by `backend` for the whole run (two ~600MB
    llama.cpp models under `--backend real`, resident from the first case onward and
    never released) -- so the first row's `latency_a_ms`/`latency_b_ms` include that
    cold load, not steady-state inference time.

    Read-only: this never calls `backend.compile()`. Assertion pass/fail reuses
    `paw_kit.test.runner.evaluate_assertion` with `suite.abstain_value` threaded through
    -- the same function and the same `abstain_value` `TestRunner`/`paw-test check` use
    (`runner.py`'s `TestRunner.run`) -- so a "pass" here means the same thing it means
    there.
    """
    inputs: List[str] = [c.input for c in suite.standard_cases]
    if include_fuzz:
        seed_inputs = [c.input for c in suite.standard_cases]
        inputs.extend(AdversarialFuzzer.generate(suite.fuzzing, base_inputs=seed_inputs))

    rows: List[CompareRow] = []
    identical_count = equivalent_count = a_pass_count = b_pass_count = only_a = only_b = errored_count = 0

    for inp in inputs:
        out_a, lat_a, err_a = _infer_safely(backend, adapter_a, inp)
        out_b, lat_b, err_b = _infer_safely(backend, adapter_b, inp)

        failed_a = []
        for rule in suite.assertions:
            ok, reason = evaluate_assertion(out_a, rule, abstain_value=suite.abstain_value)
            if not ok:
                failed_a.append(f"{rule.rule}: {reason}")
        failed_b = []
        for rule in suite.assertions:
            ok, reason = evaluate_assertion(out_b, rule, abstain_value=suite.abstain_value)
            if not ok:
                failed_b.append(f"{rule.rule}: {reason}")

        pass_a = not failed_a
        pass_b = not failed_b
        identical = out_a == out_b
        match_kind = _outputs_match_kind(out_a, out_b)

        identical_count += int(identical)
        equivalent_count += int(match_kind in (_MATCH_BYTE_IDENTICAL, _MATCH_EQUIVALENT))
        a_pass_count += int(pass_a)
        b_pass_count += int(pass_b)
        if pass_a and not pass_b:
            only_a += 1
        if pass_b and not pass_a:
            only_b += 1
        if err_a or err_b:
            errored_count += 1

        rows.append(
            CompareRow(
                input=inp,
                output_a=out_a,
                output_b=out_b,
                identical=identical,
                match_kind=match_kind,
                pass_a=pass_a,
                pass_b=pass_b,
                failed_rules_a=failed_a,
                failed_rules_b=failed_b,
                latency_a_ms=lat_a,
                latency_b_ms=lat_b,
                execution_error_a=err_a,
                execution_error_b=err_b,
            )
        )

    return CompareReport(
        task_name=suite.task_name,
        adapter_a=str(adapter_a),
        adapter_b=str(adapter_b),
        manifest_a=_project_manifest(read_adapter_manifest(str(adapter_a))),
        manifest_b=_project_manifest(read_adapter_manifest(str(adapter_b))),
        total_cases=len(rows),
        identical_count=identical_count,
        equivalent_count=equivalent_count,
        a_pass_count=a_pass_count,
        b_pass_count=b_pass_count,
        only_a_pass_count=only_a,
        only_b_pass_count=only_b,
        errored_count=errored_count,
        rows=rows,
    )
