"""Tests for paw.jit's shadow-mode agreement functions (Track 14, Phase T items 24-28)."""

from typing import Any
from pydantic import BaseModel

from paw_kit import default_agreement_fn, field_tolerance_agreement
from paw_kit.jit.agreement import _MAX_AGREEMENT_DEPTH, safe_agreement, stringify_answer


class Triage(BaseModel):
    priority: str
    urgency_score: int


class OtherTriage(BaseModel):
    priority: str
    urgency_score: int


def test_agreement_default_str_normalisation() -> None:
    """Whitespace and NFC normalisation agree; case does not (deliberately not casefolded)."""
    assert default_agreement_fn("High ", "High")
    assert default_agreement_fn("  High\n", "High")
    # "High" vs "high" must DISAGREE -- the default is biased toward reporting
    # disagreement, because a disagreement only delays a hot-swap.
    assert not default_agreement_fn("High", "high")
    # NFC vs NFD spelling of the same character agrees.
    assert default_agreement_fn("café", "café")
    # None vs "" disagrees.
    assert not default_agreement_fn(None, "")
    assert not default_agreement_fn("", None)
    assert default_agreement_fn(None, None)


def test_agreement_default_pydantic_fieldwise() -> None:
    """Field-wise comparison with no partial credit; bool is checked before int."""
    assert default_agreement_fn(
        Triage(priority="high", urgency_score=4), Triage(priority="high", urgency_score=4)
    )
    assert not default_agreement_fn(
        Triage(priority="high", urgency_score=4), Triage(priority="high", urgency_score=5)
    )
    # Same field shape, different model class -> disagree.
    assert not default_agreement_fn(
        Triage(priority="high", urgency_score=4), OtherTriage(priority="high", urgency_score=4)
    )
    # Floats within 1e-9 agree; 1 vs 1.0 agrees.
    assert default_agreement_fn(1.0, 1.0 + 1e-12)
    assert not default_agreement_fn(1.0, 1.01)
    assert default_agreement_fn(1, 1.0)
    # True vs 1 must DISAGREE (bool before int).
    assert not default_agreement_fn(True, 1)
    assert not default_agreement_fn(1, True)
    assert default_agreement_fn(True, True)


def test_agreement_default_collections_and_depth_cap() -> None:
    """Lists are order-sensitive, dicts need identical key sets, deep nesting disagrees."""
    assert default_agreement_fn(["a", "b"], ["a", "b"])
    assert not default_agreement_fn(["a", "b"], ["b", "a"])
    assert not default_agreement_fn(["a"], ["a", "b"])
    assert default_agreement_fn({"a": 1, "b": 2}, {"b": 2, "a": 1})
    assert not default_agreement_fn({"a": 1}, {"a": 1, "b": 2})

    # Nesting past the depth budget disagrees rather than recursing.
    def nest(depth: int) -> Any:
        value: Any = "leaf"
        for _ in range(depth):
            value = {"k": value}
        return value

    shallow = nest(2)
    assert default_agreement_fn(shallow, nest(2))
    deep = nest(_MAX_AGREEMENT_DEPTH + 4)
    assert not default_agreement_fn(deep, nest(_MAX_AGREEMENT_DEPTH + 4))


def test_agreement_default_never_raises() -> None:
    """A value whose comparison raises is a disagreement, not an exception."""

    class Explosive:
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("boom")

        def __hash__(self) -> int:
            raise RuntimeError("boom")

    class ExplodingModel(BaseModel):
        value: int = 1

        def model_dump(self, *args: object, **kwargs: object) -> dict:
            raise RuntimeError("boom")

    assert default_agreement_fn(Explosive(), Explosive()) is False
    assert default_agreement_fn(ExplodingModel(), ExplodingModel()) is False


def test_field_tolerance_agreement_reproduces_measurement_rule() -> None:
    """{"urgency_score": 1} makes 4-vs-5 agree while a differing priority still disagrees."""
    fn = field_tolerance_agreement({"urgency_score": 1})
    assert fn(
        Triage(priority="high", urgency_score=4), Triage(priority="high", urgency_score=5)
    )
    assert not fn(
        Triage(priority="high", urgency_score=4), Triage(priority="high", urgency_score=6)
    )
    assert not fn(
        Triage(priority="high", urgency_score=4), Triage(priority="medium", urgency_score=4)
    )
    # Non-model inputs fall through to the base comparison.
    assert fn("same", "same")
    assert not fn("same", "different")


def test_stringify_answer_handles_tuples_and_falls_back_to_default_str() -> None:
    """Finding 9: the one shared serializer -- str/BaseModel/dict/list/tuple, and a
    `default=str` fallback for anything `json.dumps` cannot handle on its own."""
    assert stringify_answer("already a string") == "already a string"
    assert stringify_answer(Triage(priority="high", urgency_score=4)) == (
        Triage(priority="high", urgency_score=4).model_dump_json()
    )
    assert stringify_answer(("a", 1)) == stringify_answer(["a", 1])

    class Unserializable:
        def __str__(self) -> str:
            return "unserializable-repr"

    # A plain object isn't JSON-serializable on its own -- `default=str` is what
    # keeps this a string instead of a raised TypeError.
    assert stringify_answer([Unserializable()]) == '["unserializable-repr"]'


def test_safe_agreement_converts_a_raising_fn_into_a_recorded_error() -> None:
    """safe_agreement never propagates; a raising agreement_fn is (False, 'ExcName')."""

    def boom(teacher: Any, adapter: Any) -> bool:
        raise ValueError("nope")

    agreed, error_type = safe_agreement(boom, "a", "a")
    assert agreed is False
    assert error_type == "ValueError"

    agreed, error_type = safe_agreement(default_agreement_fn, "a", "a")
    assert agreed is True
    assert error_type is None
