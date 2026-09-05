"""Unit and integration tests for paw.schema: grammar conversion, logits masking, and loader."""

import enum
import time
from typing import Any, List, Literal, Optional
from pydantic import BaseModel, Field
import pytest

from paw_kit import (
    MockPAWBackend,
    PAWSchemaError,
    RegexLogitsProcessor,
    load,
    pydantic_to_regex,
)


class PriorityEnum(str, enum.Enum):
    LOW = "low"
    MED = "med"
    HIGH = "high"


class DetailModel(BaseModel):
    category: str
    confidence: float


class TicketTriage(BaseModel):
    ticket_id: int
    priority: PriorityEnum
    status: Literal["open", "closed"]
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    detail: Optional[DetailModel] = None


def test_pydantic_to_regex_primitives() -> None:
    """Verify regex generation for primitive and complex schema types."""
    pattern = pydantic_to_regex(TicketTriage, anchors=True)
    assert pattern.startswith("^")
    assert pattern.endswith("$")
    assert '"ticket_id"' in pattern
    assert '"priority"' in pattern
    assert '"status"' in pattern
    assert '(?:"low"|"med"|"high")' in pattern
    assert '(?:"open"|"closed")' in pattern


def test_regex_logits_processor_masking() -> None:
    """Verify RegexLogitsProcessor correctly masks invalid tokens at each generation step."""
    # Pattern: {"valid": true}
    pattern = r'\{"valid":\s*true\}'
    vocab = {
        0: '{"valid":',
        1: " true}",
        2: " false}",
        3: "garbage",
        4: "<eos>",
    }

    processor = RegexLogitsProcessor(
        regex_pattern=pattern,
        vocabulary=vocab,
        eos_token_id=4,
    )

    state0 = processor.initial_state
    assert not processor.is_final_state(state0)

    # Step 0: Only token 0 is valid
    allowed0 = processor.get_allowed_tokens(state0)
    assert allowed0 == {0}

    # Verify dense and sparse logit filtering
    logits_dict = {0: 1.0, 1: 5.0, 2: 3.0, 3: 10.0, 4: 2.0}
    filtered_dict = processor.filter_logits(state0, logits_dict)
    assert filtered_dict[0] == 1.0
    assert filtered_dict[1] == -float("inf")
    assert filtered_dict[3] == -float("inf")

    # Step 1: Transition via token 0
    state1 = processor.get_next_state(state0, 0)
    assert state1 is not None
    allowed1 = processor.get_allowed_tokens(state1)
    assert allowed1 == {1}  # Only ' true}' allowed

    # Step 2: Transition via token 1
    state2 = processor.get_next_state(state1, 1)
    assert state2 is not None
    assert processor.is_final_state(state2)
    allowed2 = processor.get_allowed_tokens(state2)
    assert 4 in allowed2  # EOS token is now allowed


def test_logits_processor_latency_guarantee() -> None:
    """Verify logit filtering overhead is <2ms for typical vocabulary sizes."""
    pattern = r'\{"status":\s*"ok"\}'
    # 500 candidate tokens
    vocab = {i: f"tok_{i}" for i in range(500)}
    vocab[0] = '{"status":'
    vocab[1] = ' "ok"}'

    processor = RegexLogitsProcessor(regex_pattern=pattern, vocabulary=vocab)
    logits = [1.0] * 500

    start = time.perf_counter()
    processor.filter_logits(processor.initial_state, logits)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 2.0  # Must be under 2ms per generation step


def test_paw_load_success_validation() -> None:
    """Verify paw.load binds adapter to Pydantic model and returns validated instance."""
    backend = MockPAWBackend()
    adapter_path = "models/triage.paw"
    backend.compile(
        spec="Triage ticket",
        examples=[
            {
                "input": "Crash on login",
                "output": '{"ticket_id": 101, "priority": "high", "status": "open", "tags": ["auth"]}',
            }
        ],
        output_path=adapter_path,
    )

    triage_fn = load(
        adapter_path=adapter_path,
        response_model=TicketTriage,
        backend=backend,
    )

    result = triage_fn("Crash on login")
    assert isinstance(result, TicketTriage)
    assert result.ticket_id == 101
    assert result.priority == PriorityEnum.HIGH
    assert result.status == "open"
    assert result.tags == ["auth"]


def test_paw_load_fail_open_to_fallback() -> None:
    """Verify Fail-Open safety: invalid adapter output falls back transparently to teacher."""
    backend = MockPAWBackend()
    adapter_path = "models/broken.paw"
    # Adapter returns malformed / syntax-violating output
    backend.compile(
        spec="Broken",
        examples=[{"input": "query", "output": "THIS_IS_NOT_VALID_JSON"}],
        output_path=adapter_path,
    )

    fallback_called = False

    def mock_teacher_fallback(inp: str) -> TicketTriage:
        nonlocal fallback_called
        fallback_called = True
        return TicketTriage(
            ticket_id=999,
            priority=PriorityEnum.MED,
            status="closed",
            notes=f"Processed by teacher for: {inp}",
        )

    triage_fn = load(
        adapter_path=adapter_path,
        response_model=TicketTriage,
        backend=backend,
        fallback_provider=mock_teacher_fallback,
    )

    result = triage_fn("query")
    assert fallback_called is True
    assert isinstance(result, TicketTriage)
    assert result.ticket_id == 999
    assert result.notes == "Processed by teacher for: query"


def test_paw_load_raises_schema_error_without_fallback() -> None:
    """Verify that absent fallback, validation failure raises PAWSchemaError."""
    backend = MockPAWBackend()
    adapter_path = "models/broken.paw"
    backend.compile(
        spec="Broken",
        examples=[{"input": "query", "output": "INVALID_OUTPUT"}],
        output_path=adapter_path,
    )

    triage_fn = load(
        adapter_path=adapter_path,
        response_model=TicketTriage,
        backend=backend,
    )

    with pytest.raises(PAWSchemaError, match="Local execution failed validation"):
        triage_fn("query")


def test_schema_grammar_edge_types() -> None:
    """Verify regex generation for integer/bool literals, int Enums, Any, and empty models."""
    class IntEnum(enum.IntEnum):
        ONE = 1
        TWO = 2

    class EmptyModel(BaseModel):
        pass

    class EdgeModel(BaseModel):
        int_lit: Literal[1, 2]
        bool_lit: Literal[True, False]
        none_lit: Literal[None]
        int_enum: IntEnum
        float_val: float
        bool_val: bool
        any_val: Any

    assert pydantic_to_regex(EmptyModel, anchors=True) == r"^\{[ \t\n\r]*\}$"

    pat = pydantic_to_regex(EdgeModel, anchors=False)
    assert "(?:1|2)" in pat
    assert "(?:true|false)" in pat
    assert "null" in pat


def test_paw_load_default_backend_and_return_types() -> None:
    """Verify get_default_backend, set_default_backend, dict/instance outputs, and fallback failures."""
    from paw_kit.schema.loader import get_default_backend, set_default_backend

    custom_backend = MockPAWBackend()
    set_default_backend(custom_backend)
    assert get_default_backend() is custom_backend

    adapter_path = "models/test_load.paw"
    custom_backend.compile(
        spec="Test",
        examples=[],
        output_path=adapter_path,
    )

    # 1. Output as dict
    custom_backend.set_default_response(adapter_path, '{"ticket_id": 1, "priority": "low", "status": "open"}')
    fn = load(adapter_path, TicketTriage)  # tests default backend
    res = fn("test")
    assert res.ticket_id == 1

    # 2. Output is invalid object type from backend
    custom_backend.set_default_response(adapter_path, None)  # type: ignore
    with pytest.raises(PAWSchemaError):
        fn("test")

    # 3. Fallback returns dict
    def fallback_dict(inp: str) -> dict:
        return {"ticket_id": 2, "priority": "med", "status": "closed"}

    fn_fb_dict = load(adapter_path, TicketTriage, backend=custom_backend, fallback_provider=fallback_dict)
    assert fn_fb_dict("test").ticket_id == 2

    # 4. Fallback returns JSON string
    def fallback_str(inp: str) -> str:
        return '{"ticket_id": 3, "priority": "high", "status": "open"}'

    fn_fb_str = load(adapter_path, TicketTriage, backend=custom_backend, fallback_provider=fallback_str)
    assert fn_fb_str("test").ticket_id == 3

    # 5. Fallback throws exception
    def failing_fallback(inp: str) -> str:
        raise RuntimeError("Teacher network timeout")

    fn_failing_fb = load(adapter_path, TicketTriage, backend=custom_backend, fallback_provider=failing_fallback)
    with pytest.raises(PAWSchemaError, match="Both local execution and fallback failed"):
        fn_failing_fb("test")

