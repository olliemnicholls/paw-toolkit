"""Tests for paw_kit.speclint -- one positive and one negative case per rule."""

from typing import Literal, Optional

from pydantic import BaseModel

from paw_kit.speclint import Finding, lint_spec


def _rule_ids(findings):
    return {f.rule_id for f in findings}


# ---------------------------------------------------------------- rule 1: output-format-unpinned


def test_output_format_unpinned_fires_without_example():
    findings = lint_spec("Pull out the phone number from this text and format it consistently.")
    assert "output-format-unpinned" in _rule_ids(findings)
    hit = next(f for f in findings if f.rule_id == "output-format-unpinned")
    assert hit.severity == "warn"
    assert "92.5%" in hit.message


def test_output_format_unpinned_does_not_fire_with_example():
    spec = (
        "Pull out the phone number from this text and format it consistently.\n"
        "Example:\nInput: Call 555-123-4567\nOutput: (555) 123-4567"
    )
    findings = lint_spec(spec)
    assert "output-format-unpinned" not in _rule_ids(findings)


def test_output_format_unpinned_does_not_fire_without_format_words():
    findings = lint_spec("Translate this sentence into French.")
    assert "output-format-unpinned" not in _rule_ids(findings)


# ---------------------------------------------------------------- rule 2: forced-choice-no-abstain


def test_forced_choice_no_abstain_fires_on_binary_with_no_escape():
    findings = lint_spec("Decide if this product review is positive or negative.")
    assert "forced-choice-no-abstain" in _rule_ids(findings)
    hit = next(f for f in findings if f.rule_id == "forced-choice-no-abstain")
    assert hit.severity == "warn"
    assert "neutral" in hit.message


def test_forced_choice_no_abstain_does_not_fire_with_abstain_term():
    spec = "Decide if this product review is positive, negative, or unknown."
    findings = lint_spec(spec)
    assert "forced-choice-no-abstain" not in _rule_ids(findings)


def test_forced_choice_no_abstain_does_not_fire_without_a_closed_set():
    findings = lint_spec("Summarize this document in a few sentences.")
    assert "forced-choice-no-abstain" not in _rule_ids(findings)


def test_forced_choice_no_abstain_fires_on_quoted_comma_list():
    spec = 'Classify the department as "billing", "technical", or "sales".'
    findings = lint_spec(spec)
    assert "forced-choice-no-abstain" in _rule_ids(findings)


# ---------------------------------------------------------------- rule 3: schema-all-required


class _AllRequiredContact(BaseModel):
    area_code: int
    number: str
    kind: Literal["mobile", "landline"]


class _ContactWithUnknownLiteral(BaseModel):
    area_code: int
    number: str
    kind: Literal["mobile", "landline", "unknown"]


class _ContactWithOptional(BaseModel):
    area_code: int
    number: Optional[str] = None


def test_schema_all_required_fires_when_every_field_required_and_no_escape():
    findings = lint_spec("Extract a contact from this text.", schema=_AllRequiredContact)
    assert "schema-all-required" in _rule_ids(findings)
    hit = next(f for f in findings if f.rule_id == "schema-all-required")
    assert hit.severity == "warn"
    assert "stalled" in hit.message


def test_schema_all_required_does_not_fire_with_unknown_literal_value():
    findings = lint_spec("Extract a contact from this text.", schema=_ContactWithUnknownLiteral)
    assert "schema-all-required" not in _rule_ids(findings)


def test_schema_all_required_does_not_fire_with_optional_field():
    findings = lint_spec("Extract a contact from this text.", schema=_ContactWithOptional)
    assert "schema-all-required" not in _rule_ids(findings)


def test_schema_all_required_not_checked_when_no_schema_given():
    findings = lint_spec("Extract a contact from this text.")
    assert "schema-all-required" not in _rule_ids(findings)


# ---------------------------------------------------------------- rule 4: spec-too-long / spec-too-short


def test_spec_too_long_fires_over_16000_chars():
    findings = lint_spec("x" * 16_001)
    hit = next(f for f in findings if f.rule_id == "spec-too-long")
    assert hit.severity == "error"


def test_spec_too_short_fires_under_10_chars():
    findings = lint_spec("short")
    hit = next(f for f in findings if f.rule_id == "spec-too-short")
    assert hit.severity == "error"


def test_spec_length_does_not_fire_for_a_reasonable_spec():
    findings = lint_spec("Classify the sentiment of this product review as positive, negative, or unknown.")
    assert "spec-too-long" not in _rule_ids(findings)
    assert "spec-too-short" not in _rule_ids(findings)


# ---------------------------------------------------------------- rule 5: examples-single-form


def test_examples_single_form_fires_when_all_outputs_share_a_shape():
    examples = [
        {"input": "a", "output": "(555) 123-4567"},
        {"input": "b", "output": "(555) 999-0000"},
    ]
    findings = lint_spec("Extract the phone number from this text.", examples=examples)
    assert "examples-single-form" in _rule_ids(findings)
    hit = next(f for f in findings if f.rule_id == "examples-single-form")
    assert hit.severity == "info"
    assert "memorize" in hit.message


def test_examples_single_form_does_not_fire_when_outputs_vary_in_shape():
    examples = [
        {"input": "a", "output": "(555) 123-4567"},
        {"input": "b", "output": "555-999-0000"},
    ]
    findings = lint_spec("Extract the phone number from this text.", examples=examples)
    assert "examples-single-form" not in _rule_ids(findings)


def test_examples_single_form_not_checked_with_fewer_than_two_examples():
    examples = [{"input": "a", "output": "(555) 123-4567"}]
    findings = lint_spec("Extract the phone number from this text.", examples=examples)
    assert "examples-single-form" not in _rule_ids(findings)


def test_examples_single_form_not_checked_when_no_examples_given():
    findings = lint_spec("Extract the phone number from this text.")
    assert "examples-single-form" not in _rule_ids(findings)


# ---------------------------------------------------------------- lint_spec / Finding shape


def test_lint_spec_returns_finding_instances():
    findings = lint_spec("short")
    assert all(isinstance(f, Finding) for f in findings)
    for f in findings:
        assert f.severity in ("error", "warn", "info")


def test_lint_spec_clean_input_yields_no_findings():
    spec = (
        "Classify the priority of this support ticket as one of low, medium, high, "
        "or unknown-cannot-classify.\nExample:\nInput: server is down\nOutput: high"
    )
    findings = lint_spec(spec)
    assert findings == []


# =====================================================================================
# Report section 6, H-10 (bug-hunt-remediation, Track B, Phase B6)
# =====================================================================================

import yaml as _yaml  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

from paw_kit.speclint import (  # noqa: E402
    check_forced_choice_no_abstain,
    check_output_format_unpinned,
)

_REPO = _Path(__file__).parent.parent


def _committed_spec(relpath: str) -> str:
    return _yaml.safe_load((_REPO / relpath).read_text(encoding="utf-8"))["spec"]


def test_h10_forced_choice_fires_on_the_committed_lookup_spec() -> None:
    """H-10: this project's own lookup spec matches the closed-set pattern ("one of
    RG-K7, RG-M2, ...") and offers no abstain option, yet the rule was silent -- solely
    because the spec contains the phrase "a city in some other country", where the bare
    substring `"other"` matched."""
    findings = check_forced_choice_no_abstain(
        _committed_spec("measurements/finetune-lookup-suite-A.yaml")
    )
    assert [f.rule_id for f in findings] == ["forced-choice-no-abstain"]


def test_h10_forced_choice_fires_on_the_committed_triage_spec() -> None:
    """H-10: the triage spec writes its closed sets as unquoted parenthesised lists --
    `priority (low, medium, high, or critical)` -- which no pattern matched, so it
    produced zero hits. That is the very task whose adapter leaked a third label."""
    findings = check_forced_choice_no_abstain(
        _committed_spec("measurements/finetune-triage-suite.yaml")
    )
    assert [f.rule_id for f in findings] == ["forced-choice-no-abstain"]


def test_h10_ordinary_prose_no_longer_suppresses_the_rule() -> None:
    """`another`/`otherwise`/`nonetheless` contain `other`/`none` as substrings. As a
    substring scan, each silenced the rule outright."""
    for prose in (
        "Pick one of red, green or blue. Another run may differ.",
        "Pick one of red, green or blue. Otherwise escalate.",
        "Sentiment is positive or negative. Nonetheless be careful.",
    ):
        assert check_forced_choice_no_abstain(prose), prose


def test_h10_a_real_escape_hatch_still_suppresses_the_rule() -> None:
    """The rule must not become unsuppressable: an abstain option offered alongside the
    enumeration is exactly what it asks for."""
    for spec in (
        "Pick one of red, green or blue, or unknown if none fit.",
        'Return "positive" or "negative", or "not applicable" for empty input.',
        "Classify into department (billing, technical, sales, or unknown).",
    ):
        assert check_forced_choice_no_abstain(spec) == [], spec


def test_h10_abstain_term_must_be_near_the_enumeration_it_escapes() -> None:
    """An abstain word in an unrelated sentence is not an escape hatch for a different
    sentence's forced choice. This is the lookup spec's exact shape, minimised."""
    far = (
        "The city is irrelevant, and it may well be a city in some other country.\n"
        "Output exactly one of RG-K7, RG-M2, RG-Q9."
    )
    assert check_forced_choice_no_abstain(far)

    near = "Output exactly one of RG-K7, RG-M2, RG-Q9, or unknown if no country matches."
    assert check_forced_choice_no_abstain(near) == []


def test_h10_every_enumeration_needs_its_own_escape_hatch() -> None:
    """A spec that offers an abstain option for one field and forces a choice on
    another still forces a choice."""
    spec = (
        "Classify into priority (low, medium, high, or critical) and "
        "department (billing, technical, or unknown)."
    )
    assert check_forced_choice_no_abstain(spec)


def test_h10_has_example_requires_an_actual_example() -> None:
    """H-10, rule 1: the word "example" anywhere suppressed `output-format-unpinned` --
    the rule that exists because "format it consistently" with nothing showing the
    format scored 0.0% structural pass."""
    # A promise of an example is not an example.
    assert check_output_format_unpinned(
        "Extract the phone number and format it consistently. No examples are given."
    )
    assert check_output_format_unpinned(
        "Extract the phone number and format it consistently. For example, be careful."
    )
    # `Output:` alone is a format instruction, not a demonstration.
    assert check_output_format_unpinned(
        "Extract the phone number and format it consistently. Output: the number."
    )
    # An actual demonstration still suppresses it, in each of the accepted forms.
    for demo in (
        "Input: call 5551234\nOutput: (555) 123-4567",
        "555-1234 -> (555) 123-4567",
        "555-1234 => (555) 123-4567",
        "```\n(555) 123-4567\n```",
    ):
        spec = "Extract the phone number and format it consistently.\n" + demo
        assert check_output_format_unpinned(spec) == [], demo
