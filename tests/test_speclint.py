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
