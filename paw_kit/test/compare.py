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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.matching import values_equivalent, values_equivalent_unquoted
from paw_kit.test.reporting import ScoredRate, scored_denominator
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
# Quoted-scalar follow-up (measurements/README.md, "Tool feedback"): a pair that
# `values_equivalent` calls different but `values_equivalent_unquoted` calls the same
# -- one side is the other, JSON-string-quoted (e.g. `'"RG-M2"'` vs. `RG-M2`). Kept
# distinct from `_MATCH_EQUIVALENT` rather than folded into it: the strict rule stays
# the one that decides `identical`/`equivalent_count`/pass-fail-shaped reporting, so a
# quoting defect doesn't silently disappear into "equivalent" -- it gets its own count
# and its own (still collapsed, still not a "real" difference) heading instead.
_MATCH_EQUIVALENT_UNQUOTED = "equivalent_unquoted"
_MATCH_DIFFERENT = "different"


def _outputs_match_kind(output_a: str, output_b: str) -> str:
    """Classify one pair of outputs as byte-identical, equivalent-but-not-identical,
    equivalent-only-once-unquoted, or genuinely different.

    "Equivalent" (`values_equivalent`, `paw_kit.test.matching`) means: both parse as
    JSON and their parsed values are equal (so `{"a":1}` and `{"a": 1}` match, key
    order aside -- `json.loads` returns a `dict`, and dict equality is
    order-independent); otherwise, equal after Unicode NFC normalization and
    whitespace collapse, on the raw strings. If only one side parses as JSON, that is
    not "both parse" -- comparison falls through to the whitespace-normalized text,
    not a parsed value compared against unparsed text.

    "Equivalent, unquoted" (`values_equivalent_unquoted`) is strictly weaker: it also
    catches a pair where exactly one side is the other's value as a JSON string scalar
    -- `'"RG-M2"'` vs. `RG-M2` -- which `values_equivalent` deliberately still calls
    different (a quoted output is a real defect for any consumer of the adapter).
    """
    if output_a == output_b:
        return _MATCH_BYTE_IDENTICAL
    if values_equivalent(output_a, output_b):
        return _MATCH_EQUIVALENT
    if values_equivalent_unquoted(output_a, output_b):
        return _MATCH_EQUIVALENT_UNQUOTED
    return _MATCH_DIFFERENT


def adapter_label_pair(adapter_a: str, adapter_b: str) -> Tuple[str, str]:
    """Human-readable labels for two adapters, for display in `paw-test compare`'s
    stdout and JSON report: each path's file stem (`models/a.paw` -> `"a"`), or the
    full path for both when the stems collide (e.g. two same-named adapters compiled
    into different directories) -- telling the two apart takes precedence over
    brevity, and a bare `stem` alone would silently conflate them."""
    stem_a = Path(adapter_a).stem
    stem_b = Path(adapter_b).stem
    if stem_a and stem_b and stem_a != stem_b:
        return stem_a, stem_b
    return str(adapter_a), str(adapter_b)


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
    # H-5: `compare` read `suite.standard_cases` only for `.input` and had no expected
    # field at all, so two adapters of *known different correctness* were
    # indistinguishable: ten cases with `expected`, A 10/10 right and B 0/10, reported
    # `identical=0 equivalent=0 A pass=10/10 B pass=10/10 only_a_pass=0 only_b_pass=0`,
    # exit 0. This is the defect that had already been fixed in `runner.py` and left
    # unfixed in the sibling command.
    #
    # `None` on either `*_expected_match` means "no verdict", exactly as in
    # `TestCaseResult.expected_match`: the case carries no `expected`, or that side
    # abstained, or that side's backend raised. Same precedence rule as `runner.py` --
    # errored beats abstained, decided on the error field, never on the output string,
    # because `_infer_safely` writes the identical "[EXECUTION_ERROR]" placeholder.
    expected: str | None = None
    a_expected_match: bool | None = None
    b_expected_match: bool | None = None


class CompareReport(BaseModel):
    """Structured report produced by `compare_adapters`."""

    __test__ = False

    task_name: str
    adapter_a: str
    adapter_b: str
    # C-5: `backend` is which backend class actually ran the comparison
    # (`type(backend).__name__`); `requested_backend` is the raw `--backend` string
    # the caller asked for. They diverge exactly on a real-to-mock fallback -- the
    # one case a JSON consumer most needs to be able to detect, and previously could
    # not, since the fallback announcement was stdout-only Rich text.
    backend: str = ""
    requested_backend: str = ""
    # Finding (measurements/README.md, "Finetune compiler", tool feedback point 2):
    # display labels for the two adapters -- each one's file stem, or the full path
    # when the stems collide -- kept separate from `adapter_a`/`adapter_b` (the full
    # paths) so existing JSON consumers of those two fields are unaffected.
    label_a: str = "A"
    label_b: str = "B"
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
    # Quoted-scalar follow-up: `equivalent_count` widened to also treat a
    # JSON-string-quoted/bare pair as a match (`values_equivalent_unquoted`) --
    # includes every `equivalent_count` row too, so `equivalent_unquoted_count >=
    # equivalent_count` always, same relationship `equivalent_count` has to
    # `identical_count`. The CLI only prints this when it's strictly greater than
    # `equivalent_count` -- otherwise there's nothing extra to say.
    equivalent_unquoted_count: int = 0
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
    # H-5: answer-key agreement per adapter. `expected_total` counts every case that
    # carries an `expected` value; the `*_matched` counts are over the cases that side
    # actually got a verdict on, and the `*_abstained`/`*_errored` buckets are the rest
    # (see `a_expected_denominator`). A run where `expected_total` is 0 leaves all of
    # these 0 and prints nothing, exactly as before this field existed.
    expected_total: int = 0
    a_expected_matched: int = 0
    b_expected_matched: int = 0
    a_expected_abstained: int = 0
    b_expected_abstained: int = 0
    a_expected_errored: int = 0
    b_expected_errored: int = 0
    rows: List[CompareRow] = Field(default_factory=list)

    def _expected_denominator(self, side: str) -> ScoredRate:
        abstained = self.a_expected_abstained if side == "a" else self.b_expected_abstained
        errored = self.a_expected_errored if side == "a" else self.b_expected_errored
        return scored_denominator(
            total=self.expected_total,
            scored=self.expected_total - abstained - errored,
            excluded={"abstained": abstained, "errored": errored},
            label=f"{side} correct against expected",
        )

    @property
    def a_expected_denominator(self) -> ScoredRate:
        """Adapter A's answer-key rate denominator, with the note naming what it
        excluded. A genuine partition of `expected_total`, so the helper's arithmetic
        check is on."""
        return self._expected_denominator("a")

    @property
    def b_expected_denominator(self) -> ScoredRate:
        """Adapter B's answer-key rate denominator -- see `a_expected_denominator`."""
        return self._expected_denominator("b")

    @property
    def expected_disagreeing_rows(self) -> List[CompareRow]:
        """H-5: rows where exactly one adapter matched the case's own `expected`.

        This is the comparison the command existed to make and could not: it is what
        distinguishes "A is right and B is wrong" from "the two produce different
        text". `measurements/README.md:1633` prints `compare`'s summary line for the
        lookup arms, where one adapter is ~33% correct and another 97.7%.
        """
        return [
            r
            for r in self.rows
            if r.expected is not None and r.a_expected_match != r.b_expected_match
        ]

    @property
    def differing_rows(self) -> List[CompareRow]:
        """Rows where the two adapters disagree -- different output, different pass
        status, or (H-5) different agreement with the case's own `expected`. This is
        deliberately the headline view: a per-case diff is what actually decided the
        finetune-compiler comparison, not the aggregate counts below it.

        Unchanged by finding 2: what counts as an *output* difference here is still
        byte-level (`not r.identical`), same as before normalization-aware matching
        existed. H-5 adds a third disjunct rather than redefining the first:
        byte-identical outputs cannot disagree on `expected`, so in practice this only
        promotes rows that were already differing.
        """
        return [
            r
            for r in self.rows
            if not r.identical
            or r.pass_a != r.pass_b
            or (r.expected is not None and r.a_expected_match != r.b_expected_match)
        ]

    @property
    def equivalent_only_rows(self) -> List[CompareRow]:
        """The subset of `differing_rows` that are only a formatting or quoting
        difference: not byte-identical, but `match_kind` is `"equivalent"` (same JSON
        value, or the same text after whitespace normalization) or
        `"equivalent_unquoted"` (same value once a JSON-string-quoted side is
        unwrapped).

        This is the "whitespace- or quoting-only" list finding 2 (and its quoted-scalar
        follow-up) asks to report separately: `paw-test compare`'s CLI lists these
        under their own, collapsed heading instead of folding them into (or silently
        dropping them from) the main differences listing.

        **G-4:** this used to additionally require `pass_a == pass_b`, so a quoted
        output that *flipped* pass status never reached the collapsed heading -- and
        `compare` contradicted itself five lines apart, listing three rows as full
        `Differences` while summarising them as "3 equivalent output once unwrapped".
        That condition is gone; the CLI annotates the flip inside the collapsed listing
        instead of exiling the row from it.

        The condition that replaces it is H-5's, and it is a different claim: a row
        where the two adapters disagree about the *answer key* is not a formatting
        difference at any level, so it stays in the headline diff.
        """
        return [
            r
            for r in self.differing_rows
            if r.match_kind in (_MATCH_EQUIVALENT, _MATCH_EQUIVALENT_UNQUOTED)
            and not (r.expected is not None and r.a_expected_match != r.b_expected_match)
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
    requested_backend: str = "",
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
    # H-5: parallel to `inputs` -- the case's own answer key, or None for a standard
    # case that doesn't set one and for every fuzz case. Same shape `TestRunner.run`
    # builds, so the two commands grade against the same thing.
    expected_values: List[Optional[str]] = [c.expected for c in suite.standard_cases]
    if include_fuzz:
        seed_inputs = [c.input for c in suite.standard_cases]
        fuzzed = AdversarialFuzzer.generate(suite.fuzzing, base_inputs=seed_inputs)
        inputs.extend(fuzzed)
        expected_values.extend([None] * len(fuzzed))

    rows: List[CompareRow] = []
    identical_count = equivalent_count = equivalent_unquoted_count = 0
    a_pass_count = b_pass_count = only_a = only_b = errored_count = 0
    expected_total = 0
    a_expected_matched = b_expected_matched = 0
    a_expected_abstained = b_expected_abstained = 0
    a_expected_errored = b_expected_errored = 0

    for inp, expected in zip(inputs, expected_values):
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
        equivalent_unquoted_count += int(
            match_kind in (_MATCH_BYTE_IDENTICAL, _MATCH_EQUIVALENT, _MATCH_EQUIVALENT_UNQUOTED)
        )
        a_pass_count += int(pass_a)
        b_pass_count += int(pass_b)
        if pass_a and not pass_b:
            only_a += 1
        if pass_b and not pass_a:
            only_b += 1
        if err_a or err_b:
            errored_count += 1

        # H-5, with `runner.py`'s precedence rule applied per side: errored beats
        # abstained, decided on the error field and never on the output string, because
        # `_infer_safely` writes the same "[EXECUTION_ERROR]" placeholder the runner
        # does and a suite is free to name that string as its `abstain_value`.
        a_expected_match: Optional[bool] = None
        b_expected_match: Optional[bool] = None
        if expected is not None:
            expected_total += 1
            for out, err, side in ((out_a, err_a, "a"), (out_b, err_b, "b")):
                errored = err is not None
                abstained = (
                    not errored
                    and suite.abstain_value is not None
                    and out == suite.abstain_value
                )
                if errored:
                    if side == "a":
                        a_expected_errored += 1
                    else:
                        b_expected_errored += 1
                elif abstained:
                    if side == "a":
                        a_expected_abstained += 1
                    else:
                        b_expected_abstained += 1
                else:
                    matched = values_equivalent(out, expected)
                    if side == "a":
                        a_expected_match = matched
                        a_expected_matched += int(matched)
                    else:
                        b_expected_match = matched
                        b_expected_matched += int(matched)

        rows.append(
            CompareRow(
                input=inp,
                expected=expected,
                a_expected_match=a_expected_match,
                b_expected_match=b_expected_match,
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

    label_a, label_b = adapter_label_pair(str(adapter_a), str(adapter_b))
    return CompareReport(
        task_name=suite.task_name,
        adapter_a=str(adapter_a),
        adapter_b=str(adapter_b),
        backend=type(backend).__name__,
        requested_backend=requested_backend,
        label_a=label_a,
        label_b=label_b,
        manifest_a=_project_manifest(read_adapter_manifest(str(adapter_a))),
        manifest_b=_project_manifest(read_adapter_manifest(str(adapter_b))),
        total_cases=len(rows),
        identical_count=identical_count,
        equivalent_count=equivalent_count,
        equivalent_unquoted_count=equivalent_unquoted_count,
        a_pass_count=a_pass_count,
        b_pass_count=b_pass_count,
        only_a_pass_count=only_a,
        only_b_pass_count=only_b,
        errored_count=errored_count,
        expected_total=expected_total,
        a_expected_matched=a_expected_matched,
        b_expected_matched=b_expected_matched,
        a_expected_abstained=a_expected_abstained,
        b_expected_abstained=b_expected_abstained,
        a_expected_errored=a_expected_errored,
        b_expected_errored=b_expected_errored,
        rows=rows,
    )
