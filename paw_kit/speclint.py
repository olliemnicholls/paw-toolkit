"""paw.speclint: static checks for spec-authoring failure modes seen in real compiles.

Every rule here is derived from a measured failure against a real compiled
ProgramAsWeights adapter, documented in `measurements/README.md` -- this module does
not guess at best practice, it flags the specific shapes of spec that produced a
specific bad outcome on the record. None of it runs automatically today: `compile()`
does not call `lint_spec`, and `@compile_on_hit` does not either. This is a standalone
diagnostic a caller opts into (via the library function or `paw-kit lint-spec`), not a
gate.

Each rule is a small function taking whatever inputs it needs and returning a list of
`Finding`s (usually zero or one). `lint_spec` just concatenates all of them.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import types
from typing import Any, Dict, List, Literal, Optional, Type, Union, get_args, get_origin

from pydantic import BaseModel

Severity = Literal["error", "warn", "info"]

# Upstream's documented compile-time spec size limit (see
# paw_kit/backend/programasweights.py's module docstring / measurements/README.md);
# a spec over this is rejected or truncated by the compile service, not paw-kit.
_MAX_SPEC_CHARS = 16_000
_MIN_SPEC_CHARS = 10


@dataclass(frozen=True)
class Finding:
    """One lint result: `rule_id` identifies which check fired, `severity` is one of
    "error" | "warn" | "info", and `message` explains why, citing the measured
    finding the rule is derived from."""

    rule_id: str
    severity: Severity
    message: str


# ---------------------------------------------------------------------- rule 1

_FORMAT_WORDS = ("format", "consistently", "normalize", "standard")


def _has_concrete_example(spec: str) -> bool:
    """Whether `spec` contains an actual demonstration, not just the word "example".

    H-10: the marker list used to include the bare substring `"example"`, so any spec
    *mentioning* examples ("no examples were given", "for example, dates vary")
    suppressed `output-format-unpinned` entirely -- the rule that exists because a spec
    saying "format it consistently" with nothing showing the format scored 0.0%
    structural pass. A promise of an example is not an example.

    What counts: an `Input:`/`Output:` **pair** (both, not either), an arrow, or a code
    fence. `Output:` alone is a format instruction, not a demonstration.
    """
    lower = spec.lower()
    if "input:" in lower and "output:" in lower:
        return True
    return "->" in spec or "=>" in spec or "```" in spec


def check_output_format_unpinned(spec: str) -> List[Finding]:
    """Flag a spec that asks for consistent/normalized formatting but never pins
    down what that format actually looks like.

    Measured effect (`measurements/README.md`, "Terse, docs-style specs" and "Does
    folding examples into the spec text actually help?"): the spec `"Pull out the
    phone number from this text and format it consistently."`, compiled with zero
    examples folded in, scored **0.0% structural pass** -- not because extraction
    failed, but because the adapter picked *a* consistent format (dash-separated)
    that the test suite's author had assumed would be different (parenthesized).
    Folding in 8 examples that demonstrated the intended format raised that to
    **92.5%** with no other change. The ambiguity, not the model, was the defect.
    """
    lower = spec.lower()
    if not any(word in lower for word in _FORMAT_WORDS):
        return []
    if _has_concrete_example(spec):  # H-10: an actual demonstration, not the word
        return []
    return [
        Finding(
            "output-format-unpinned",
            "warn",
            "Spec asks for consistent/normalized/standard formatting but contains no "
            'concrete example output (no "Example", "Input:"/"Output:", "->"/"=>", or '
            "code fence). A real compiled adapter given an equivalent spec "
            '("...format it consistently") scored 0.0% structural pass with no '
            "examples folded in, and 92.5% once 8 examples pinned down the exact "
            'output format (measurements/README.md, "Does folding examples into the '
            'spec text actually help?"). Add at least one concrete Input:/Output: '
            "example showing the exact output shape.",
        )
    ]


# ---------------------------------------------------------------------- rule 2

_CLOSED_SET_PATTERNS = [
    re.compile(r"\bone of\b", re.IGNORECASE),
    re.compile(r"\beither\b[^.?!\n]*\bor\b", re.IGNORECASE),
    re.compile(r"\bpositive or negative\b", re.IGNORECASE),
]
# A quoted, comma-separated list of two or more labels, e.g. "billing", "technical",
# "sales" -- either quote style, at least one comma between quoted items.
_QUOTED_LIST_RE = re.compile(r"""(["'])[^"'\n]+\1\s*,\s*(["'])[^"'\n]+\2""")
# H-10: an unquoted, parenthesised list -- `priority (low, medium, high, or critical)`.
# The committed triage spec is written exactly that way and produced **zero** hits, and
# that is the very task whose adapter leaked a third label.
_PAREN_LIST_RE = re.compile(r"\(\s*[^()\n]*?,[^()\n]*?\bor\b[^()\n]*?\)", re.IGNORECASE)

_ABSTAIN_TERMS = ("abstain", "unknown", "other", "none", "not applicable", "n/a")
# H-10: whole words, not substrings. As bare substrings, `"other"` matched inside
# `another`/`otherwise` and `"none"` inside `nonetheless`, so ordinary prose silenced
# the rule. `re.escape` because "n/a" carries a slash.
_ABSTAIN_TERM_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _ABSTAIN_TERMS) + r")\b", re.IGNORECASE
)

#: Sentence-ish boundaries. Used to bound the search for an abstain term to the
#: neighbourhood of the enumeration it would have to be an escape hatch *for*.
_SENTENCE_SPLIT_RE = re.compile(r"[.?!\n]")


def _enumeration_windows(spec: str) -> List[str]:
    """The text an abstain term would have to appear in, per enumeration, to count as
    that enumeration's escape hatch.

    Two different window sizes, because the two kinds of enumeration delimit themselves
    differently:

    * A **parenthesised** list is its own window. `priority (low, medium, high, or
      critical) and department (billing, technical, or unknown)` is one sentence with
      two enumerations, and `unknown` is an option for *department* only -- `priority`
      is still a forced choice. Widening to the sentence would call that spec clean.
    * Everything else (`one of ...`, `either X or Y`, a quoted comma list) has no
      closing delimiter, so the sentence is the tightest honest bound.
    """
    windows = [m.group(0) for m in _PAREN_LIST_RE.finditer(spec)]
    for pattern in _CLOSED_SET_PATTERNS:
        windows += [_sentence_around(spec, *m.span()) for m in pattern.finditer(spec)]
    windows += [_sentence_around(spec, *m.span()) for m in _QUOTED_LIST_RE.finditer(spec)]
    return windows


def _sentence_around(spec: str, start: int, end: int) -> str:
    """The text between the sentence boundaries either side of `spec[start:end]`."""
    left = 0
    for m in _SENTENCE_SPLIT_RE.finditer(spec, 0, start):
        left = m.end()
    right_match = _SENTENCE_SPLIT_RE.search(spec, end)
    right = right_match.start() if right_match else len(spec)
    return spec[left:right]


def check_forced_choice_no_abstain(spec: str) -> List[Finding]:
    """Flag a spec that enumerates a closed label set with no escape hatch.

    Measured effect (`measurements/README.md`, "Terse, docs-style specs"): a spec
    restricting sentiment to strictly `"positive or negative"` still had the compiled
    adapter reliably return a third label, `"neutral"`, on empty/gibberish/genuinely
    ambiguous input -- the model found the forced binary unanswerable for that input
    and leaked outside the contract rather than silently picking one. The suite's own
    `not_contains: neutral` assertion (correctly) failed every time.

    **H-10, two fixes, and the first alone is not enough.** The abstain check was a bare
    substring scan of the whole spec, so `"other"` matched inside `another`/`otherwise`
    and `"none"` inside `nonetheless`. Whole-word matching fixes that class -- but not
    the case that was actually measured: this project's own lookup spec matches the
    closed-set pattern, offers no abstain option, and was silent **solely because the
    spec contains the phrase "a city in some other country"**, where `other` is a
    perfectly good whole word. An abstain term is only an escape hatch if it is offered
    *for the enumeration*, so the search is bounded to the sentence containing it.
    """
    windows = _enumeration_windows(spec)
    if not windows:
        return []
    # Fires unless EVERY enumeration has an escape hatch in its own window: a spec that
    # offers "unknown" for `department` and nothing for `priority` still forces a choice
    # on `priority`.
    if all(_ABSTAIN_TERM_RE.search(w) for w in windows):
        return []
    return [
        Finding(
            "forced-choice-no-abstain",
            "warn",
            "Spec enumerates a closed label set (\"one of\", \"either X or Y\", "
            '"positive or negative", a quoted comma list, or a parenthesised '
            '"(a, b, or c)" list) with no '
            "abstain/unknown/other/none/\"not applicable\" option offered alongside "
            "that enumeration. A real compiled "
            'sentiment adapter told strictly "positive or negative" still leaked a '
            'third label, "neutral", on ambiguous input rather than force a coin-flip '
            '(measurements/README.md, "Terse, docs-style specs"). Add an explicit '
            "fallback label the model is allowed to use when no listed choice fits.",
        )
    ]


# ---------------------------------------------------------------------- rule 3

_UNKNOWN_LITERAL_TOKENS = {
    "unknown", "other", "none", "not_applicable", "not applicable", "n/a", "na", "abstain",
}


def _is_optional_annotation(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin not in (Union, getattr(types, "UnionType", Union)):
        return False
    return type(None) in get_args(annotation)


def _literal_has_unknown_value(annotation: Any) -> bool:
    if get_origin(annotation) is not Literal:
        return False
    for arg in get_args(annotation):
        if isinstance(arg, str) and arg.strip().lower() in _UNKNOWN_LITERAL_TOKENS:
            return True
    return False


def check_schema_all_required(schema: Optional[Type[BaseModel]]) -> List[Finding]:
    """Flag a schema where every field is required with no representable "I don't know".

    Measured effect (`measurements/README.md`, "Constrained decoding against the real
    upstream adapter", the 5th-case paragraph): a real grammar-constrained adapter
    given `"no phone number here at all"` -- input with no valid answer -- emitted
    `{"area_code": 0, "number": ""` and then stalled: the FSM would not accept a
    terminator until every required field was filled, and the model had no
    representable way to say "not applicable." Only checked when a schema is
    supplied (`--schema module:Model` on the CLI, or the `schema=` library argument).
    """
    if schema is None:
        return []
    fields = getattr(schema, "model_fields", None)
    if not fields:
        return []
    for field in fields.values():
        if not field.is_required():
            return []  # has a default (Optional or otherwise) -- fine
        annotation = field.annotation
        if _is_optional_annotation(annotation) or _literal_has_unknown_value(annotation):
            return []
    return [
        Finding(
            "schema-all-required",
            "warn",
            f"Every field on {schema.__name__} is required, with no Optional/default "
            "and no Literal value representing \"unknown\"/\"not applicable\". A real "
            'grammar-constrained adapter given input with no valid answer '
            '("no phone number here at all") stalled mid-object because the FSM would '
            "not accept a terminator until every required field was filled "
            '(measurements/README.md, "Constrained decoding against the real upstream '
            'adapter", the 5th-case paragraph). Add an optional field, or an '
            '"unknown"/"not_applicable" Literal value, the model can emit when the '
            "input has no valid answer.",
        )
    ]


# ---------------------------------------------------------------------- rule 4


def check_spec_length(spec: str) -> List[Finding]:
    """Flag a spec too long for the upstream compiler, or too short to be a spec."""
    findings: List[Finding] = []
    length = len(spec)
    if length > _MAX_SPEC_CHARS:
        findings.append(
            Finding(
                "spec-too-long",
                "error",
                f"Spec is {length} characters, over the upstream compile service's "
                f"{_MAX_SPEC_CHARS}-character limit; it will be rejected or silently "
                "truncated. Shorten it.",
            )
        )
    if length < _MIN_SPEC_CHARS:
        findings.append(
            Finding(
                "spec-too-short",
                "error",
                f"Spec is only {length} character(s) -- too short to convey a task to "
                "the compiler. Write a fuller specification.",
            )
        )
    return findings


# ---------------------------------------------------------------------- rule 5


def _punctuation_skeleton(text: str) -> str:
    """Collapse every run of letters/digits to `#`, leaving punctuation/whitespace
    layout as the "shape" of the string (e.g. `"(555) 123-4567"` -> `"(#) #-#"`)."""
    return re.sub(r"[A-Za-z0-9]+", "#", text)


def check_examples_single_form(examples: Optional[List[Dict[str, Any]]]) -> List[Finding]:
    """Flag a set of >=2 examples whose outputs are all the exact same surface shape.

    Measured effect (`measurements/README.md`, "Does folding examples into the spec
    text actually help?"): folding in 8 examples that all shared one output shape
    fixed a format-consistency problem, but the same adapter then regurgitated one of
    those training examples *verbatim* -- including its literal digits -- for a
    genuinely different, out-of-distribution input that shared nothing with it. Only
    checked when examples are supplied (`--examples file.jsonl` on the CLI, or the
    `examples=` library argument); informational, not a warning, since uniform
    examples are often simply what the task calls for.
    """
    if not examples or len(examples) < 2:
        return []
    outputs = [
        ex["output"]
        for ex in examples
        if isinstance(ex, dict) and isinstance(ex.get("output"), str)
    ]
    if len(outputs) < 2:
        return []
    lengths = {len(o) for o in outputs}
    skeletons = {_punctuation_skeleton(o) for o in outputs}
    if len(lengths) != 1 or len(skeletons) != 1:
        return []
    return [
        Finding(
            "examples-single-form",
            "info",
            f"All {len(outputs)} example outputs share one surface pattern (identical "
            "length and punctuation skeleton). Folding same-shaped examples into a "
            "real compile fixed a format-consistency problem but made the adapter "
            "memorize and regurgitate one training example verbatim for a genuinely "
            'different, out-of-distribution input (measurements/README.md, "Does '
            'folding examples into the spec text actually help?", the international-'
            "number case). If the underlying outputs are meant to vary, vary the "
            "examples' surface form (length, punctuation) too.",
        )
    ]


# ---------------------------------------------------------------------- public API


def lint_spec(
    spec: str,
    *,
    examples: Optional[List[Dict[str, Any]]] = None,
    schema: Optional[Type[BaseModel]] = None,
) -> List[Finding]:
    """Run every speclint rule against `spec` (and, where applicable, `examples`/`schema`).

    Args:
        spec: The raw natural-language task specification.
        examples: Optional demonstration pairs (`{"input": ..., "output": ...}`), the
            same shape `AbstractPAWBackend.compile()` takes. Only rule 5 uses this.
        schema: Optional Pydantic model the compiled function's output is validated
            against. Only rule 3 uses this.

    Returns:
        Every finding from every rule, in a fixed order (length checks first, then
        the three spec-text rules, then the examples rule). Empty means clean.
    """
    findings: List[Finding] = []
    findings.extend(check_spec_length(spec))
    findings.extend(check_output_format_unpinned(spec))
    findings.extend(check_forced_choice_no_abstain(spec))
    findings.extend(check_schema_all_required(schema))
    findings.extend(check_examples_single_form(examples))
    return findings
