"""Unit and integration tests for paw.schema: grammar conversion, logits masking, and loader."""

import datetime as dt
from decimal import Decimal
import enum
import time
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union
import uuid
from pydantic import BaseModel, Field, create_model
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

    # Best of 5 runs to eliminate profiler cold-start and coverage instrumentation jitter
    times = []
    for _ in range(5):
        start = time.perf_counter()
        processor.filter_logits(processor.initial_state, logits)
        times.append((time.perf_counter() - start) * 1000)

    assert min(times) < 2.0  # Must be under 2ms per generation step


def test_paw_load_success_validation(tmp_path) -> None:
    """Verify paw.load binds adapter to Pydantic model and returns validated instance."""
    backend = MockPAWBackend()
    adapter_path = str(tmp_path / "triage.paw")
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


def test_paw_load_fail_open_to_fallback(tmp_path) -> None:
    """Verify Fail-Open safety: invalid adapter output falls back transparently to teacher."""
    backend = MockPAWBackend()
    adapter_path = str(tmp_path / "broken.paw")
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


def test_paw_load_raises_schema_error_without_fallback(tmp_path) -> None:
    """Verify that absent fallback, validation failure raises PAWSchemaError."""
    backend = MockPAWBackend()
    adapter_path = str(tmp_path / "broken.paw")
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


def test_paw_load_default_backend_and_return_types(tmp_path) -> None:
    """Verify get_default_backend, set_default_backend, dict/instance outputs, and fallback failures."""
    from paw_kit.schema.loader import get_default_backend, set_default_backend

    custom_backend = MockPAWBackend()
    set_default_backend(custom_backend)
    assert get_default_backend() is custom_backend

    adapter_path = str(tmp_path / "test_load.paw")
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


# ─── New Tests for Schema Complexity Enhancements ───
import datetime as dt
import uuid
from decimal import Decimal
from typing import Dict, FrozenSet, Set, Tuple


class FixedTupleModel(BaseModel):
    coords: Tuple[str, int]


class VariadicTupleModel(BaseModel):
    tags: Tuple[str, ...]


def test_tuple_fixed_length_generates_array_regex() -> None:
    """Verify tuple[str, int] produces a JSON array regex, not JSON_STRING."""
    pat = pydantic_to_regex(FixedTupleModel, anchors=True)
    assert r"\[" in pat, "Fixed-length tuple should produce array brackets"
    assert r"\]" in pat
    import re as _re
    m = _re.match(pat, '{"coords": ["hello", 42]}')
    assert m is not None, "Fixed-length tuple regex should match valid JSON array"


def test_tuple_variadic_generates_list_regex() -> None:
    """Verify tuple[str, ...] produces a repeating JSON array regex."""
    pat = pydantic_to_regex(VariadicTupleModel, anchors=True)
    assert r"\[" in pat, "Variadic tuple should produce array brackets"
    import re as _re
    m = _re.match(pat, '{"tags": ["a", "b", "c"]}')
    assert m is not None, "Variadic tuple regex should match multi-element array"
    m_empty = _re.match(pat, '{"tags": []}')
    assert m_empty is not None, "Variadic tuple regex should match empty array"


class SetModel(BaseModel):
    unique_tags: Set[str]


class FrozenSetModel(BaseModel):
    immutable_tags: FrozenSet[int]


def test_set_generates_array_regex() -> None:
    """Verify set[str] produces a JSON array regex, not JSON_STRING."""
    pat = pydantic_to_regex(SetModel, anchors=True)
    assert r"\[" in pat, "Set should produce array brackets"
    import re as _re
    m = _re.match(pat, '{"unique_tags": ["alpha", "beta"]}')
    assert m is not None


def test_frozenset_generates_array_regex() -> None:
    """Verify frozenset[int] produces a JSON array regex."""
    pat = pydantic_to_regex(FrozenSetModel, anchors=True)
    assert r"\[" in pat, "FrozenSet should produce array brackets"
    import re as _re
    m = _re.match(pat, '{"immutable_tags": [1, 2, 3]}')
    assert m is not None


class BareListModel(BaseModel):
    items: list


class BareTupleModel(BaseModel):
    items: tuple


class BareSetModel(BaseModel):
    items: set


def test_bare_list_generates_array_regex() -> None:
    """Verify bare list produces a JSON array regex."""
    pat = pydantic_to_regex(BareListModel, anchors=True)
    assert r"\[" in pat, "Bare list should produce array brackets"


def test_bare_tuple_generates_array_regex() -> None:
    """Verify bare tuple produces a JSON array regex."""
    pat = pydantic_to_regex(BareTupleModel, anchors=True)
    assert r"\[" in pat, "Bare tuple should produce array brackets"


def test_bare_set_generates_array_regex() -> None:
    """Verify bare set produces a JSON array regex."""
    pat = pydantic_to_regex(BareSetModel, anchors=True)
    assert r"\[" in pat, "Bare set should produce array brackets"


class SpecializedModel(BaseModel):
    id: uuid.UUID
    created: dt.datetime
    due_date: dt.date
    amount: Decimal


def test_uuid_generates_specific_regex() -> None:
    """Verify UUID produces a hex-formatted regex, not generic JSON_STRING."""
    pat = pydantic_to_regex(SpecializedModel, anchors=True)
    assert "[0-9a-fA-F]" in pat, "UUID should produce hex character class"
    import re as _re
    m = _re.match(pat, (
        '{"id": "550e8400-e29b-41d4-a716-446655440000", '
        '"created": "2024-01-15T10:30:00Z", '
        '"due_date": "2024-01-15", '
        '"amount": "3.14"}'
    ))
    assert m is not None


def test_date_generates_specific_regex() -> None:
    """Verify date produces a YYYY-MM-DD regex."""
    from paw_kit.schema.grammar import _type_to_regex, JSON_DATE
    result = _type_to_regex(dt.date)
    assert result == JSON_DATE


def test_datetime_generates_specific_regex() -> None:
    """Verify datetime produces an ISO 8601 regex."""
    from paw_kit.schema.grammar import _type_to_regex, JSON_DATETIME
    result = _type_to_regex(dt.datetime)
    assert result == JSON_DATETIME


class PatternModel(BaseModel):
    zip_code: str = Field(pattern=r"[0-9]{5}")
    name: str


def test_field_pattern_constraint_used_in_regex() -> None:
    """Verify Field(pattern=...) produces a pattern-specific regex, not generic JSON_STRING."""
    pat = pydantic_to_regex(PatternModel, anchors=True)
    assert "[0-9]{5}" in pat, "Field pattern constraint should appear in regex"
    import re as _re
    m = _re.match(pat, '{"zip_code": "90210", "name": "test"}')
    assert m is not None


class RecursiveNode(BaseModel):
    value: str
    children: Optional[List["RecursiveNode"]] = None


def test_recursive_model_raises_paw_schema_error() -> None:
    """Verify recursive BaseModel raises PAWSchemaError, not RecursionError."""
    with pytest.raises(PAWSchemaError, match="Recursive model detected"):
        pydantic_to_regex(RecursiveNode)


def test_load_wraps_recursion_error_in_paw_schema_error() -> None:
    """Verify load() wraps recursive model errors in PAWSchemaError."""
    with pytest.raises(PAWSchemaError):
        load(
            adapter_path="models/test.paw",
            response_model=RecursiveNode,
        )


def test_load_wraps_unexpected_compilation_error() -> None:
    """Verify load() wraps arbitrary compilation errors in PAWSchemaError."""
    import paw_kit.schema.loader as loader_module
    original = loader_module.pydantic_to_regex

    def _boom(*a, **kw):
        raise TypeError("Unexpected internal error")

    loader_module.pydantic_to_regex = _boom
    try:
        with pytest.raises(PAWSchemaError, match="Failed to compile grammar regex"):
            load(adapter_path="models/test.paw", response_model=TicketTriage)
    finally:
        loader_module.pydantic_to_regex = original


def test_pydantic_to_regex_is_cached() -> None:
    """Verify repeated calls return the same object (cache hit)."""
    pydantic_to_regex.cache_clear()
    r1 = pydantic_to_regex(TicketTriage, anchors=True)
    r2 = pydantic_to_regex(TicketTriage, anchors=True)
    assert r1 is r2, "Cached results should be the same object"
    info = pydantic_to_regex.cache_info()
    assert info.hits >= 1


class OptionalFieldsModel(BaseModel):
    required_field: str
    optional_field: Optional[str] = None


def test_optional_fields_require_explicit_null() -> None:
    """Verify optional fields must be present with explicit null value."""
    import re as _re
    pat = pydantic_to_regex(OptionalFieldsModel, anchors=True)
    # Must match with explicit null
    m = _re.match(pat, '{"required_field": "hello", "optional_field": null}')
    assert m is not None
    # Must NOT match with field omitted (current intentional behavior)
    m2 = _re.match(pat, '{"required_field": "hello"}')
    assert m2 is None, "Optional field omission is intentionally not supported"


# --- PAW-SCHEMA-01: quote-escaping in Literal / Enum / Field(pattern=...) -----------


def test_literal_string_with_quote_cannot_break_out_of_json_boundary_PAW_SCHEMA_01() -> None:
    """A Literal string containing a quote must be JSON-escaped, not left to break out."""
    import re as _re

    class QuoteLiteralModel(BaseModel):
        value: Literal['say "hi"']

    pat = pydantic_to_regex(QuoteLiteralModel, anchors=True)
    # The correctly JSON-escaped form is the only thing that should match.
    assert _re.match(pat, r'{"value": "say \"hi\""}') is not None
    # The pre-fix bug (re.escape doesn't escape '"') would have made the regex accept
    # the raw, unescaped quote as if it were a legitimate JSON string terminator.
    assert _re.match(pat, '{"value": "say "hi""}') is None


def test_enum_string_with_quote_cannot_break_out_of_json_boundary_PAW_SCHEMA_01() -> None:
    """A str-valued Enum member containing a quote must be JSON-escaped, not left to break out."""
    import re as _re

    class QuoteEnum(str, enum.Enum):
        WEIRD = 'a"b'

    class QuoteEnumModel(BaseModel):
        value: QuoteEnum

    pat = pydantic_to_regex(QuoteEnumModel, anchors=True)
    assert _re.match(pat, r'{"value": "a\"b"}') is not None
    assert _re.match(pat, '{"value": "a"b"}') is None


def test_field_pattern_with_bare_quote_is_rejected_PAW_SCHEMA_01() -> None:
    """A Field(pattern=...) containing a bare quote is rejected, not embedded raw."""

    class InjectedPatternModel(BaseModel):
        value: str = Field(pattern=r'safe", "role": "admin", "x": "')

    with pytest.raises(PAWSchemaError, match="double quote characters are forbidden"):
        pydantic_to_regex(InjectedPatternModel)


def test_field_pattern_with_backslash_escaped_quote_is_also_rejected_PAW_SCHEMA_01() -> None:
    """A Field(pattern=...) 'escaping' its quote with a single backslash is also rejected.

    This is the regression test for the audit's own broken suggested fix: a naive
    "reject unescaped quotes, allow backslash-escaped ones" heuristic would let this
    pattern through, since it looks escaped in the pattern SOURCE. But
    `re.compile(r'a\\"b').fullmatch('a"b')` matches -- the single backslash does not
    require a backslash in the matched OUTPUT text at all, so allowing this pattern
    would still let a bare, JSON-string-terminating quote through.
    """

    class FakeEscapedPatternModel(BaseModel):
        value: str = Field(pattern=r'^a\"b$')

    with pytest.raises(PAWSchemaError, match="double quote characters are forbidden"):
        pydantic_to_regex(FakeEscapedPatternModel)


def test_non_ascii_literal_still_matches_raw_utf8_PAW_SCHEMA_01() -> None:
    """A non-ASCII Literal/Enum must still match the raw UTF-8 form a decoder emits.

    Schema Determinism guard on the PAW-SCHEMA-01 fix itself: json.dumps' default
    (`ensure_ascii=True`) would rewrite 'café' as 'caf\\u00e9', so the compiled grammar
    would accept ONLY the escaped form and silently reject the raw UTF-8 string every
    existing non-ASCII enum schema produces today. The escaping fix must be scoped to
    quotes/backslashes/control characters, not to non-ASCII characters.
    """
    import re as _re

    class AccentEnum(str, enum.Enum):
        CAFE = "café"

    class AccentModel(BaseModel):
        literal_value: Literal["café", "naïve"]
        enum_value: AccentEnum

    pat = pydantic_to_regex(AccentModel, anchors=True)
    assert _re.match(pat, '{"literal_value": "café", "enum_value": "café"}') is not None
    # ...while the PAW-SCHEMA-01 breakout protection is unaffected by that scoping.
    assert "café" in pat and "\\u00e9" not in pat


# --- PAW-SCHEMA-02: separate, named collection-nesting depth budget -----------------


def _nested_model_chain(length: int) -> type:
    """Build a chain of `length` distinct BaseModels, each nesting the next.

    Distinct classes (not a self-reference) so the cycle-detection `seen` guard cannot
    fire -- this exercises the `_MAX_RECURSION_DEPTH` budget itself.
    """
    from pydantic import create_model

    model: Any = create_model("ChainLeaf", value=(str, ...))
    for i in range(length):
        model = create_model(f"ChainLevel{i}", child=(model, ...))
    return model


def test_model_nesting_at_the_limit_still_compiles_PAW_SCHEMA_02() -> None:
    """The deepest BaseModel chain within the model-depth budget still compiles.

    The guard is `depth > _MAX_RECURSION_DEPTH` (strict) and the root model is itself
    depth 0, so `_MAX_RECURSION_DEPTH + 1` chain levels is the last accepted shape.
    Asserting the *tight* boundary is the point: the PAW-SCHEMA-02 collection budget
    must not have silently eaten into this one.
    """
    from paw_kit.schema.grammar import _MAX_RECURSION_DEPTH

    pat = pydantic_to_regex(_nested_model_chain(_MAX_RECURSION_DEPTH + 1))
    assert pat  # compiles without raising


def test_model_nesting_past_the_limit_raises_PAW_SCHEMA_02() -> None:
    """One level past that boundary raises, rather than being silently allowed."""
    from paw_kit.schema.grammar import _MAX_RECURSION_DEPTH

    with pytest.raises(PAWSchemaError, match="[Rr]ecursive model detected"):
        pydantic_to_regex(_nested_model_chain(_MAX_RECURSION_DEPTH + 2))


def _nested_list_type(depth: int) -> Any:
    """Build List[List[...[int]...]] nested `depth` levels deep."""
    t: Any = int
    for _ in range(depth):
        t = List[t]
    return t


def test_collection_nesting_at_the_limit_still_compiles_PAW_SCHEMA_02() -> None:
    """A collection nested exactly up to the budget must still compile (not over-eager)."""
    from paw_kit.schema.grammar import _MAX_COLLECTION_DEPTH

    class AtLimitModel(BaseModel):
        value: _nested_list_type(_MAX_COLLECTION_DEPTH)  # type: ignore[valid-type]

    pat = pydantic_to_regex(AtLimitModel)
    assert pat  # compiles without raising


def test_collection_nesting_past_the_limit_raises_PAW_SCHEMA_02() -> None:
    """A collection nested one level past the budget must raise, not blow up memory."""
    from paw_kit.schema.grammar import _MAX_COLLECTION_DEPTH

    class PastLimitModel(BaseModel):
        value: _nested_list_type(_MAX_COLLECTION_DEPTH + 1)  # type: ignore[valid-type]

    with pytest.raises(PAWSchemaError, match="[Cc]ollection nesting"):
        pydantic_to_regex(PastLimitModel)


def test_collection_depth_resets_at_each_nested_model_boundary_PAW_SCHEMA_02() -> None:
    """Collection depth must not accumulate across BaseModel boundaries (separate budget).

    Six BaseModels nested inside each other, each with its own List[Dict[str, ...]]
    field, must still compile: that's well within the per-model collection budget, but
    would incorrectly exceed a *shared* depth counter with the six levels of model
    nesting (regressing Schema Determinism for realistic schemas).
    """

    class Leaf(BaseModel):
        tags: List[Dict[str, int]]

    class Level5(BaseModel):
        tags: List[Dict[str, int]]
        child: Leaf

    class Level4(BaseModel):
        tags: List[Dict[str, int]]
        child: Level5

    class Level3(BaseModel):
        tags: List[Dict[str, int]]
        child: Level4

    class Level2(BaseModel):
        tags: List[Dict[str, int]]
        child: Level3

    class Level1(BaseModel):
        tags: List[Dict[str, int]]
        child: Level2

    pat = pydantic_to_regex(Level1)
    assert pat  # compiles without raising despite 6 levels of model + collection nesting


# --- PAW-SCHEMA-03: bounded FSM compilation (pattern length, timeout, state count) --


def test_compile_fsm_safe_rejects_overlong_pattern_PAW_SCHEMA_03() -> None:
    """A pattern over the length cap is rejected before any compilation is attempted."""
    import paw_kit.schema.logits_processor as lp

    overlong = "a" * (lp._MAX_PATTERN_LENGTH + 1)
    with pytest.raises(PAWSchemaError, match="Pattern length"):
        lp._compile_fsm_safe(overlong)


def test_compile_fsm_safe_rejects_excessive_fsm_states_PAW_SCHEMA_03(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pattern whose compiled FSM exceeds the state cap is rejected."""
    import paw_kit.schema.logits_processor as lp

    monkeypatch.setattr(lp, "_MAX_FSM_STATES", 1)
    with pytest.raises(PAWSchemaError, match="state"):
        lp._compile_fsm_safe("(a|b){3}")  # trivially compiles to more than one state


def test_compile_fsm_safe_timeout_does_not_block_on_runaway_thread_PAW_SCHEMA_03(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow FSM compile must time out promptly, not block until the thread finishes.

    This is the exact bug in the audit's own illustrative fix: running the compile
    inside `with ThreadPoolExecutor(...)` calls `Executor.__exit__` ->
    `shutdown(wait=True)` unconditionally, so even after `future.result()` raises
    `TimeoutError` the `with` block still blocks the caller until the runaway compile
    finishes anyway -- defeating the timeout's entire purpose.
    """
    import paw_kit.schema.logits_processor as lp

    class _SlowParsed:
        def to_fsm(self) -> Any:
            time.sleep(0.3)
            raise AssertionError("should never be reached within the test's timeout")

    monkeypatch.setattr(lp, "_FSM_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(lp.interegular, "parse_pattern", lambda pattern: _SlowParsed())

    start = time.monotonic()
    with pytest.raises(PAWSchemaError, match="timed out"):
        lp._compile_fsm_safe("dummy")
    elapsed = time.monotonic() - start
    assert elapsed < 1.0, "compile_fsm_safe blocked on the runaway thread instead of returning promptly"


class InvoiceLine(BaseModel):
    """One line of the realistic five-field invoice schema S-16 was filed about."""

    sku: str = Field(pattern=r"[A-Z]{3}-[0-9]{4}")
    description: str
    quantity: int
    unit_price: float


class InvoiceModel(BaseModel):
    invoice_id: str = Field(pattern=r"INV-[0-9]{6}")
    issued: dt.date
    customer: DetailModel
    lines: List[InvoiceLine]
    total: float


def test_ordinary_invoice_schema_is_not_refused_by_the_length_cap_S_16() -> None:
    """An ordinary nested invoice schema must compile, not be refused as pathological.

    S-16: `_MAX_PATTERN_LENGTH` was 1,000 *characters*, which a realistic five-field
    nested schema exceeds -- and the refusal blamed the schema for pathology. The cap
    itself is kept (it is the only check that runs before any compilation at all; see
    the module comment), but at 50,000, the same order as where `_MAX_FSM_STATES`
    actually binds.
    """
    import paw_kit.schema.logits_processor as lp

    regex = pydantic_to_regex(InvoiceModel, anchors=False)
    assert len(regex) > 1000, (
        "the S-16 corpus model no longer exceeds the old 1,000-character cap, so this "
        f"test no longer pins the finding (got {len(regex)} characters)"
    )
    fsm = lp._compile_fsm_safe(regex)
    assert fsm.states, "the invoice schema compiled to an empty FSM"


def test_compile_fsm_safe_timeout_does_not_hang_interpreter_exit_S_17() -> None:
    """After a compile timeout the process must still be able to exit (S-17).

    `shutdown(wait=False)` returns to the caller promptly, which is all the existing
    PAW-SCHEMA-03 timeout test checks -- but `concurrent.futures` registers its worker
    threads with `threading._register_atexit`, so interpreter shutdown then *joins* the
    abandoned compile. Executed against that version: the timeout raised at 3.02 s and
    the process never exited (killed externally at 40 s). This is reachable from the
    served path: a `paw-serve` worker that compiles one pathological grammar keeps
    serving and then cannot shut down.

    In-process assertions cannot see this -- the hang is at interpreter exit -- so the
    check has to be a subprocess that is required to terminate.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        """
        import time
        import paw_kit.schema.logits_processor as lp

        class _Runaway:
            def to_fsm(self):
                time.sleep(60)

        lp.interegular.parse_pattern = lambda pattern: _Runaway()
        lp._FSM_TIMEOUT_SECONDS = 0.05
        try:
            lp._compile_fsm_safe("dummy")
        except Exception as exc:
            print("RAISED", type(exc).__name__)
        print("EXITING", flush=True)
        """
    )
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "the interpreter did not exit within 20s after a compile timeout: the "
            "abandoned FSM-compile thread is being joined at shutdown (S-17)"
        )
    elapsed = time.monotonic() - start
    assert "EXITING" in proc.stdout, proc.stderr
    assert "RAISED PAWSchemaError" in proc.stdout, proc.stdout
    assert elapsed < 20, f"process took {elapsed:.1f}s to exit after a 0.05s timeout"


# --- PAW-SCHEMA-04: bounded per-processor caches, avoid full-vocab scan per state ---


def test_logits_processor_transition_cache_bounded_PAW_SCHEMA_04(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _transition_cache evicts oldest entries past its size cap rather than
    growing without limit over a long generation."""
    import paw_kit.schema.logits_processor as lp

    monkeypatch.setattr(lp, "_MAX_TRANSITION_CACHE_ENTRIES", 3)
    vocab = {i: chr(97 + i) for i in range(10)}  # 'a'..'j', single-char tokens
    processor = RegexLogitsProcessor(regex_pattern=r"[a-j]{5}", vocabulary=vocab)

    state = processor.initial_state
    for token_id in range(10):
        processor.get_next_state(state, token_id)

    assert len(processor._transition_cache) <= 3


def test_logits_processor_allowed_tokens_cache_bounded_PAW_SCHEMA_04(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify _allowed_tokens_cache evicts oldest entries past its size cap."""
    import paw_kit.schema.logits_processor as lp

    monkeypatch.setattr(lp, "_MAX_ALLOWED_TOKENS_CACHE_ENTRIES", 2)
    vocab = {i: chr(97 + i) for i in range(5)}
    processor = RegexLogitsProcessor(regex_pattern=r"[a-e]{5}", vocabulary=vocab)

    state = processor.initial_state
    visited_states = {state}
    for token_id in range(5):
        next_state = processor.get_next_state(state, token_id)
        if next_state is not None:
            visited_states.add(next_state)
            processor.get_allowed_tokens(next_state)

    assert len(processor._allowed_tokens_cache) <= 2


def test_logits_processor_skips_full_vocab_scan_for_restrictive_state_PAW_SCHEMA_04() -> None:
    """Verify get_allowed_tokens does not call get_next_state for every vocabulary
    token when only a small fraction of first characters are legal from the current
    state -- the whole point of the first-character bucketing PAW-SCHEMA-04 adds."""
    pattern = r'\{"k":\s*true\}'  # a fixed literal: only one character is ever legal
    # A large vocabulary of single distinct-first-character tokens, only one of which
    # (the one starting with the pattern's first literal character) can ever be legal.
    vocab = {i: chr(33 + i) + "xyz" for i in range(200)}
    vocab[0] = "{" + "xyz"  # ensure the one legal first character is present
    processor = RegexLogitsProcessor(regex_pattern=pattern, vocabulary=vocab)

    call_count = 0
    real_get_next_state = processor.get_next_state

    def spy(state: int, token_id: int):
        nonlocal call_count
        call_count += 1
        return real_get_next_state(state, token_id)

    processor.get_next_state = spy  # type: ignore[method-assign]
    processor.get_allowed_tokens(processor.initial_state)

    # Only tokens whose first character is a legal transition are ever walked --
    # nowhere near the full 200-entry vocabulary.
    assert call_count < len(vocab)


# --- PAW-SCHEMA-05: Tuple[()] compiles strict; bare typing.Tuple stays permissive --


def test_tuple_empty_annotation_compiles_to_strict_empty_array_PAW_SCHEMA_05() -> None:
    """Verify both Tuple[()] and tuple[()] compile to a strict empty-array regex."""
    import re as _re

    class TypingEmptyTupleModel(BaseModel):
        value: Tuple[()]

    class BuiltinEmptyTupleModel(BaseModel):
        value: tuple[()]

    for model in (TypingEmptyTupleModel, BuiltinEmptyTupleModel):
        pat = pydantic_to_regex(model, anchors=True)
        assert _re.match(pat, '{"value": []}') is not None
        assert _re.match(pat, '{"value": [1]}') is None
        assert _re.match(pat, '{"value": [1, 2, 3]}') is None


def test_tuple_empty_annotation_old_permissive_behavior_no_longer_validates_PAW_SCHEMA_05() -> None:
    """Verify the old grammar/Pydantic mismatch is actually closed: a value the
    compiled regex used to accept for Tuple[()] must now be rejected by *both* the
    regex and Pydantic's own validation, restoring the "compiled grammar admits only
    validatable output" invariant rather than merely tightening the regex further."""
    import re as _re

    class EmptyTupleModel(BaseModel):
        value: Tuple[()]

    pat = pydantic_to_regex(EmptyTupleModel, anchors=True)
    non_empty_payload = '{"value": [1, 2, 3]}'
    assert _re.match(pat, non_empty_payload) is None
    with pytest.raises(Exception):
        EmptyTupleModel.model_validate({"value": [1, 2, 3]})


def test_bare_typing_tuple_stays_permissive_PAW_SCHEMA_05() -> None:
    """Verify bare, unsubscripted typing.Tuple ("an array of anything") is unaffected
    by the Tuple[()] fix -- it is indistinguishable from Tuple[()] via
    get_origin/get_args alone, so this must be an identity check, not incidental."""
    import re as _re

    class BareTupleModel(BaseModel):
        value: Tuple

    pat = pydantic_to_regex(BareTupleModel, anchors=True)
    assert _re.match(pat, '{"value": []}') is not None
    assert _re.match(pat, '{"value": [1, 2, 3]}') is not None


# --- PAW-SCHEMA-06: bounded digit runs in JSON_INTEGER and JSON_FLOAT's integer part


def test_json_integer_caps_digit_count_PAW_SCHEMA_06() -> None:
    """Verify an int field's compiled regex rejects a digit run past the cap."""
    import re as _re
    from paw_kit.schema.grammar import _MAX_NUMBER_DIGITS

    class IntModel(BaseModel):
        value: int

    pat = pydantic_to_regex(IntModel, anchors=True)
    at_limit = "9" * _MAX_NUMBER_DIGITS
    over_limit = "9" * (_MAX_NUMBER_DIGITS + 1)
    assert _re.match(pat, f'{{"value": {at_limit}}}') is not None
    assert _re.match(pat, f'{{"value": {over_limit}}}') is None


def test_json_float_integer_part_caps_digit_count_PAW_SCHEMA_06() -> None:
    """Verify a float field's regex rejects an over-cap digit run in the integer part
    -- JSON_FLOAT full-matches a bare, decimal-point-free integer too, so capping only
    JSON_INTEGER would leave the identical DoS reachable through every float field."""
    import re as _re
    from paw_kit.schema.grammar import _MAX_NUMBER_DIGITS

    class FloatModel(BaseModel):
        value: float

    pat = pydantic_to_regex(FloatModel, anchors=True)
    over_limit = "9" * (_MAX_NUMBER_DIGITS + 1)
    assert _re.match(pat, f'{{"value": {over_limit}}}') is None
    assert _re.match(pat, f'{{"value": {over_limit}.5}}') is None
    # A legitimate large-but-in-budget float with a fractional part still compiles.
    at_limit = "9" * _MAX_NUMBER_DIGITS
    assert _re.match(pat, f'{{"value": {at_limit}.5}}') is not None


# --- PAW-SCHEMA-07: content-fingerprint cache keying, not model-identity keying ----


def test_pydantic_to_regex_identical_dynamic_models_share_cache_entry_PAW_SCHEMA_07() -> None:
    """Verify two independently-created, structurally-identical dynamic models (the
    exact `pydantic.create_model` population this finding is about) hit the cache --
    the old identity-keyed lru_cache gave this population a guaranteed 0% hit rate."""
    pydantic_to_regex.cache_clear()
    model_a = create_model("SameShape", x=(str, ...), y=(int, ...))
    model_b = create_model("SameShape", x=(str, ...), y=(int, ...))

    regex_a = pydantic_to_regex(model_a)
    info_after_a = pydantic_to_regex.cache_info()
    regex_b = pydantic_to_regex(model_b)
    info_after_b = pydantic_to_regex.cache_info()

    assert regex_a == regex_b
    assert info_after_b.hits == info_after_a.hits + 1
    assert info_after_b.misses == info_after_a.misses


def test_pydantic_to_regex_reordered_optional_fields_do_not_collide_PAW_SCHEMA_07() -> None:
    """Verify two same-titled dynamic models whose *optional* fields are declared in a
    different order do NOT share a cache entry -- _pydantic_to_regex_impl emits
    fields in model_fields (declaration) order, so a fingerprint that ignored order
    (e.g. the audit's own json.dumps(..., sort_keys=True) suggestion) would silently
    serve one model's regex to the other (Phase 0 Round 1/2)."""
    pydantic_to_regex.cache_clear()
    model_a = create_model("Reordered", a=(Optional[str], None), b=(Optional[int], None))
    model_b = create_model("Reordered", b=(Optional[int], None), a=(Optional[str], None))

    regex_a = pydantic_to_regex(model_a)
    regex_b = pydantic_to_regex(model_b)

    assert regex_a != regex_b


def test_pydantic_to_regex_same_named_differently_shaped_nested_models_do_not_collide_PAW_SCHEMA_07() -> None:
    """Verify two outer models over same-named, differently-shaped dynamic inner
    models do NOT share a cache entry -- str(annotation) (a rejected alternative
    fingerprint) collides here, since two dynamically created classes both named
    "Inner" stringify identically regardless of their actual field shape."""
    inner_a = create_model("Inner", a=(str, ...))
    inner_b = create_model("Inner", b=(int, ...))
    outer_a = create_model("Outer", nested=(inner_a, ...))
    outer_b = create_model("Outer", nested=(inner_b, ...))

    regex_a = pydantic_to_regex(outer_a)
    regex_b = pydantic_to_regex(outer_b)

    assert regex_a != regex_b


def test_pydantic_to_regex_fingerprint_raises_paw_schema_error_not_recursion_error_PAW_SCHEMA_07() -> None:
    """Verify the fingerprint builder itself raises PAWSchemaError (not
    RecursionError) on a recursive model -- it runs *before* _pydantic_to_regex_impl,
    so it is the first code to see one (Phase 0 Round 3, N-16)."""
    from paw_kit.schema.grammar import _fingerprint_model_fields

    with pytest.raises(PAWSchemaError, match="Recursive model detected"):
        _fingerprint_model_fields(RecursiveNode, seen=frozenset(), depth=0)


class _KitchenSinkInner(BaseModel):
    label: str
    score: float


def _kitchen_sink_field_defs() -> Dict[str, Any]:
    """Field definitions exercising every _type_to_regex branch, for the
    PAW-SCHEMA-07 property test below: Union/Optional, Literal, Enum, List, fixed and
    variadic Tuple, Set/FrozenSet, nested BaseModel, Dict, bare list/tuple/set/dict,
    each specialized and primitive type, Field(pattern=...), and Any."""
    return dict(
        union_field=(Union[str, int], ...),
        optional_field=(Optional[str], None),
        literal_field=(Literal["a", "b", 1, True, None], ...),
        enum_field=(PriorityEnum, ...),
        list_field=(List[int], ...),
        tuple_fixed_field=(Tuple[str, int, bool], ...),
        tuple_variadic_field=(Tuple[int, ...], ...),
        tuple_empty_field=(Tuple[()], ...),
        # Bare, unsubscripted `typing.Tuple` is its own `_type_to_regex` branch
        # (grammar.py:279, `elif annotation is Tuple`), distinct from both
        # `Tuple[()]` above and bare builtin `tuple` below -- all three reach branch
        # 5 with `args == ()` and are separated only by identity checks
        # (PAW-SCHEMA-05). Present so the corpus reaches *every* branch, not a
        # representative subset (added at Phase F).
        bare_typing_tuple_field=(Tuple, ...),
        set_field=(Set[str], ...),
        frozenset_field=(FrozenSet[int], ...),
        nested_model_field=(_KitchenSinkInner, ...),
        dict_field=(Dict[str, int], ...),
        bare_list_field=(list, ...),
        bare_tuple_field=(tuple, ...),
        bare_set_field=(set, ...),
        bare_dict_field=(dict, ...),
        uuid_field=(uuid.UUID, ...),
        datetime_field=(dt.datetime, ...),
        date_field=(dt.date, ...),
        decimal_field=(Decimal, ...),
        str_field=(str, ...),
        int_field=(int, ...),
        float_field=(float, ...),
        bool_field=(bool, ...),
        any_field=(Any, ...),
        pattern_field=(str, Field(pattern=r"^[a-z]+$")),
    )


def test_pydantic_to_regex_fingerprint_property_equal_key_implies_equal_regex_PAW_SCHEMA_07() -> None:
    """Property test (Phase 0 Round 3): for a corpus covering every _type_to_regex
    branch, key(a) == key(b) must imply pydantic_to_regex(a) == pydantic_to_regex(b).
    The converse is deliberately not asserted -- a finer key only costs a hit."""
    from paw_kit.schema.grammar import _fingerprint_model_fields

    model_a = create_model("KitchenSinkA", **_kitchen_sink_field_defs())
    model_b = create_model("KitchenSinkB", **_kitchen_sink_field_defs())

    key_a = _fingerprint_model_fields(model_a, seen=frozenset(), depth=0)
    key_b = _fingerprint_model_fields(model_b, seen=frozenset(), depth=0)
    assert key_a == key_b

    assert pydantic_to_regex(model_a) == pydantic_to_regex(model_b)

    # And the regex must actually compile and be usable (exercises every branch for
    # real, not just at the fingerprint level).
    import re as _re

    pat = pydantic_to_regex(model_a, anchors=True)
    payload = (
        '{"union_field": 1, "optional_field": null, "literal_field": "a", '
        '"enum_field": "low", "list_field": [1, 2], '
        '"tuple_fixed_field": ["x", 1, true], "tuple_variadic_field": [1, 2, 3], '
        '"tuple_empty_field": [], "bare_typing_tuple_field": [1, "x"], '
        '"set_field": ["a"], "frozenset_field": [1], '
        '"nested_model_field": {"label": "x", "score": 1.0}, '
        '"dict_field": {"a": 1}, "bare_list_field": [1, "x"], '
        '"bare_tuple_field": [1, "x"], "bare_set_field": [1, "x"], '
        '"bare_dict_field": {"a": 1}, '
        '"uuid_field": "12345678-1234-1234-1234-123456789abc", '
        '"datetime_field": "2026-01-01T00:00:00", "date_field": "2026-01-01", '
        '"decimal_field": "1.5", "str_field": "x", "int_field": 1, '
        '"float_field": 1.5, "bool_field": true, "any_field": "x", '
        '"pattern_field": "abc"}'
    )
    assert _re.match(pat, payload) is not None


# --- S-2: JSON_STRING accepts only the escapes JSON really permits -------------------


def test_json_string_rejects_invalid_escapes_S_2() -> None:
    """`\\q` and a truncated `\\u12` must not satisfy the grammar (S-2).

    The previous `\\\\.` spelling permitted any character after a backslash, so the
    grammar for a one-`str`-field schema accepted strings that fail `json.loads`.
    """
    import json
    import re as _re

    class TextModel(BaseModel):
        text: str

    pat = pydantic_to_regex(TextModel, anchors=True)
    for bad in (r'{"text":"a\qb"}', r'{"text":"\u12"}', r'{"text":"\x41"}', r'{"text":"\ "}'):
        assert _re.match(pat, bad) is None, f"grammar still accepts invalid escape {bad!r}"
        with pytest.raises(json.JSONDecodeError):
            json.loads(bad)


def test_json_string_still_accepts_every_legal_escape_S_2() -> None:
    """The eight legal short escapes and a legal \\uXXXX must stay reachable (S-2)."""
    import re as _re

    class TextModel(BaseModel):
        text: str

    pat = pydantic_to_regex(TextModel, anchors=True)
    for good in (r'{"text":"a\"b"}', r'{"text":"a\\b"}', r'{"text":"a\/b"}',
                 r'{"text":"a\bb"}', r'{"text":"a\fb"}', r'{"text":"a\nb"}',
                 r'{"text":"a\rb"}', r'{"text":"a\tb"}', r'{"text":"aéb"}',
                 r'{"text":"a퟿b"}'):
        assert _re.match(pat, good) is not None, f"grammar rejects legal escape {good!r}"
        TextModel.model_validate_json(good)


def test_json_string_excludes_lone_surrogate_escapes_S_2() -> None:
    """A lone `\\uD800`-`\\uDFFF` escape must not be accepted (S-2).

    `json.loads` tolerates an unpaired surrogate but pydantic's Rust JSON parser does
    not, so admitting it would leave the grammar wider than the validator -- exactly
    the defect S-2 is about, one door along.
    """
    import re as _re

    class TextModel(BaseModel):
        text: str

    pat = pydantic_to_regex(TextModel, anchors=True)
    for surrogate in (r'{"text":"\ud800"}', r'{"text":"\uDFFF"}', r'{"text":"\uD83D"}'):
        assert _re.match(pat, surrogate) is None, f"grammar accepts lone surrogate {surrogate!r}"
        with pytest.raises(Exception):
            TextModel.model_validate_json(surrogate)


# --- S-7: anchors are stripped structurally, honouring backslash escapes -------------


def test_anchor_stripping_honours_backslash_escapes_S_7() -> None:
    """`rstrip("$")` is character-wise, so `r"a\\$"` lost its literal dollar (S-7)."""
    from paw_kit.schema.grammar import _strip_anchors

    assert _strip_anchors(r"^abc$") == "abc"
    assert _strip_anchors(r"a$") == "a"
    assert _strip_anchors(r"a\$") == r"a\$", "an escaped dollar is a literal, not an anchor"
    assert _strip_anchors(r"^\$[0-9]+\.[0-9]{2}$") == r"\$[0-9]+\.[0-9]{2}"
    assert _strip_anchors("a\\\\$") == "a\\\\", "an escaped backslash does not escape the anchor"
    assert _strip_anchors(r"[a$]") == r"[a$]", "a dollar inside a class is not a trailing anchor"
    assert _strip_anchors(r"\$") == r"\$", "a backslash at index 0 still escapes the anchor"
    assert _strip_anchors("^") == "", "a pattern that is nothing but anchors strips to empty"
    assert _strip_anchors("$$") == "", "every unescaped trailing anchor goes, not just the last"


def test_currency_pattern_keeps_its_dollar_sign_S_7() -> None:
    """A currency `Field(pattern=...)` must accept the value with its dollar sign (S-7).

    Before the fix the grammar accepted `{"x": "a"}` (pydantic rejects it) and rejected
    `{"x": "a$"}` (pydantic accepts it) -- wrong in both directions, silently.
    """
    import re as _re

    class CurrencyModel(BaseModel):
        x: str = Field(pattern=r"a\$")

    CurrencyModel.model_validate_json('{"x":"a$"}')  # precondition: pydantic accepts it
    pat = pydantic_to_regex(CurrencyModel, anchors=True)
    assert _re.match(pat, '{"x":"a$"}') is not None, "grammar rejects the legal value"
    assert _re.match(pat, '{"x":"a"}') is None, "grammar still accepts the value pydantic rejects"


# --- S-15: interegular's own exception types are wrapped in PAWSchemaError -----------


@pytest.mark.parametrize("pattern_src", [r"\bfoo\b", r"\p{L}+", r"\Qa.b\E", r"(?<=a)b"])
def test_compile_fsm_safe_wraps_interegular_exceptions_S_15(pattern_src: str) -> None:
    """`Unsupported` and `InvalidSyntax` must surface as PAWSchemaError (S-15)."""
    import paw_kit.schema.logits_processor as lp

    with pytest.raises(PAWSchemaError, match="Cannot compile the pattern into a DFA"):
        lp._compile_fsm_safe(pattern_src)


def test_regex_logits_processor_wraps_interegular_exceptions_S_15() -> None:
    """The public constructor is the real beneficiary: it has no other wrapper (S-15).

    `loader.py` already converts anything that is not a PAWSchemaError, but
    `RegexLogitsProcessor` is a public export constructed directly.
    """
    with pytest.raises(PAWSchemaError, match="Cannot compile the pattern into a DFA"):
        RegexLogitsProcessor(regex_pattern=r"\bword\b", vocabulary={0: "a"}, eos_token_id=1)


# --- S-3 / S-3b: Field(pattern=...) is translated through the AST, not spliced --------


class _DotStar(BaseModel):
    x: str
    y: int


@pytest.mark.parametrize(
    "pattern_src,escape",
    [(r".*", '"'), (r"\S+", '"'), (r"[^a]+", '"'), (r"[\w\W]+", '"'), (r"\D+", '"'),
     (r".*", "\\"), (r"\S+", "\\"), (r"[^a]+", "\\"), (r"\D+", "\\")],
)
def test_character_classes_cannot_match_the_json_string_terminator_S_3(
    pattern_src: str, escape: str
) -> None:
    """No class may match a bare quote or backslash, however it is spelled (S-3).

    `_sanitize_field_pattern` rejected a literal `"` in the pattern *source* and nothing
    else, so every one of these -- including `.*`, the single most common idiom --
    compiled to a grammar that accepted a value breaking out of its own JSON string.
    """
    import re as _re

    model = create_model("Escaper", x=(str, Field(pattern=pattern_src)), y=(int, ...))
    pat = pydantic_to_regex(model, anchors=True)
    broken = '{"x": "' + escape + '", "y": 1}'
    assert _re.match(pat, broken) is None, (
        f"pattern {pattern_src!r} still admits {broken!r}, which is not valid JSON"
    )


def test_shorthand_classes_are_ascii_restricted_where_they_negate_S_3b() -> None:
    """`\\D` must not admit a character pydantic rejects (S-3b).

    interegular's shorthands are ASCII and Python `re`'s / pydantic's are Unicode, so
    `interegular.parse_pattern(r"\\D").to_fsm().accepts("\u0663")` is True while both
    `re.fullmatch(r"\\D", "\u0663")` and pydantic reject it. Splicing `\\D` into the
    grammar therefore made the grammar wider than the validator, and made the exported
    regex and the decoding FSM two different languages.
    """
    import re as _re

    import interegular

    # The premise: the two engines really do disagree about the shorthand itself.
    assert interegular.parse_pattern(r"\D").to_fsm().accepts("\u0663") is True
    assert _re.fullmatch(r"\D", "\u0663") is None

    model = create_model("Shorthand", x=(str, Field(pattern=r"\D+")))
    with pytest.raises(Exception):
        model(x="\u0663")  # pydantic itself rejects it

    pat = pydantic_to_regex(model, anchors=False)
    assert _re.fullmatch(pat, '{"x": "\u0663"}') is None
    # ... and the decoder must agree with the exported regex, which is the S-3b half.
    # (`anchors=False`: interegular refuses `^`/`$` outright, so the decoder only ever
    # sees the unanchored form.)
    assert interegular.parse_pattern(pat).to_fsm().accepts('{"x": "\u0663"}') is False
    assert _re.fullmatch(pat, '{"x": "abc"}') is not None


def test_posix_bracket_classes_are_refused_S_3() -> None:
    """POSIX bracket classes mean different things to the two engines, so they raise."""
    model = create_model("Posix", x=(str, Field(pattern=r"[[:alpha:]]+")))
    with pytest.raises(PAWSchemaError, match="POSIX bracket class"):
        pydantic_to_regex(model)


def test_inline_flag_group_away_from_the_start_is_refused_S_3() -> None:
    """`a(?i)b` means three different things to three engines, so it raises."""
    model = create_model("MidFlag", x=(str, Field(pattern=r"a(?i)b")))
    with pytest.raises(PAWSchemaError, match="inline flag group"):
        pydantic_to_regex(model)


def test_empty_json_safe_intersection_raises_rather_than_matching_nothing_S_3() -> None:
    """Rule 4: a sub-expression with no JSON-safe form raises, never renders nomatch.

    A "match nothing" grammar hands the decoder an all-`-inf` mask at step 0, which is
    the S-12 failure reached by a different door.
    """
    model = create_model("Tabbed", x=(str, Field(pattern=r"\t+")))
    with pytest.raises(PAWSchemaError, match="no JSON-safe form"):
        pydantic_to_regex(model)


@pytest.mark.parametrize("pattern_src", [r"a**", r"a*+", r"a+*?", r"(a*)*", r"(?:a{2}){3}"])
def test_stacked_quantifiers_render_as_a_regex_python_re_accepts_S_3(pattern_src: str) -> None:
    """A quantified atom cannot itself take a quantifier, so it must be wrapped (S-3).

    interegular parses `a**` as a repetition OF a repetition -- its `atom()` consumes
    the first `*` and its `obj()` the second -- so a renderer that treats a quantified
    atom as atomic emits `a**` verbatim, which Python `re` refuses with "multiple
    repeat". That is the same class of failure as S-1: `pydantic_to_regex` returning a
    string that is not a regex.
    """
    import re as _re

    import interegular

    model = create_model("Stacked", x=(str, Field(pattern=pattern_src)))
    pat = pydantic_to_regex(model, anchors=False)
    _re.compile(pat)  # the failure mode is this line raising
    interegular.parse_pattern(pat).to_fsm()  # ... and the decoder must take it too


# The exact rendering, pinned. The compiled regex is a user-facing artefact --
# `examples/pii_scrubber/README.md` tells readers to inspect it, it is what `loader.py`
# hands the backend, and it is embedded verbatim in `measurements/*.json` -- and its
# length is bounded (`logits_processor._MAX_PATTERN_LENGTH`), so "same language,
# different spelling" is not a free pass. Every entry below is a spelling the renderer
# is required to choose, not merely one it happens to produce.
RENDERING_CASES = [
    (r"[abc]", "[a-c]"),               # contiguous runs collapse into a range
    (r"[a-cx]", "[a-cx]"),             # ... and a stray member stays a member
    (r"[0-9]{5}", "[0-9]{5}"),         # tests/test_schema.py:416 depends on this exactly
    (r"\d", "[0-9]"),                  # shorthand -> explicit, read alike by both engines
    (r"a*", "a*"),
    (r"a+", "a+"),
    (r"a?", "a?"),
    (r"x{0,1}", "x?"),
    (r"a{1,1}", "a{1}"),               # min == max collapses
    (r"a{2,}", "a{2,}"),               # open-ended is NOT "+"
    (r"a{0,3}", "a{0,3}"),             # bounded-from-zero is NOT "?"
    (r"(cat|dog)s?", "(?:cat|dog)s?"), # exactly one group, not two
    (r"(cat|dog)+", "(?:cat|dog)+"),   # ... and a quantified group is not re-wrapped
    (r"(?:ab){2}", "(?:ab){2}"),
    (r"[ a]", "[ a]"),                 # a printable space stays a space, not `\x20`
    (r"\s", " "),                      # ... including where it is the whole class
    (r".{3}", '[^\\x00-\\x1f"\\\\\\x7f-\\x9f]{3}'),  # no redundant (?:...) around a class
]


@pytest.mark.parametrize("pattern_src,expected", RENDERING_CASES,
                         ids=[c[0] for c in RENDERING_CASES])
def test_translator_renders_the_documented_spelling_S_3(pattern_src: str, expected: str) -> None:
    """The renderer's output spelling is part of its contract, not an implementation detail."""
    from paw_kit.schema.grammar import _translate_field_pattern

    assert _translate_field_pattern(pattern_src) == expected


def test_translated_regex_carries_no_raw_control_characters_S_3() -> None:
    """Control characters are emitted as `\\xHH`, never raw.

    They reach the output only in the excluded list of a negated class, where a raw
    byte would be legal to both engines but would put unprintable characters into a
    string users read, diff and embed in measurement artefacts.
    """
    from paw_kit.schema.grammar import _translate_field_pattern

    for pattern_src in (r".*", r"[^a]+", r"[^\n]+"):
        rendered = _translate_field_pattern(pattern_src)
        assert all(ord(c) >= 0x20 and not 0x7F <= ord(c) <= 0x9F for c in rendered), (
            f"{pattern_src!r} rendered raw control characters: {rendered!r}"
        )


def test_ascii_restriction_applies_only_to_shorthand_classes_S_3b() -> None:
    """Rule 2 is scoped to `\\d \\D \\w \\W \\s \\S`, and must not leak onto other escapes.

    `[^\\n]` is a negated class built from a non-shorthand escape: all three engines
    agree it matches a non-ASCII character, so ASCII-restricting it would narrow the
    grammar for no reason and make an ordinary accented value unreachable.
    """
    import re as _re

    model = create_model("NotNewline", x=(str, Field(pattern=r"[^\n]+")))
    model(x="café")  # pydantic accepts it
    pat = pydantic_to_regex(model, anchors=False)
    assert _re.fullmatch(pat, '{"x":"café"}') is not None, (
        "the ASCII restriction leaked onto a non-shorthand negated class"
    )


def test_empty_intersection_message_names_the_offending_characters_S_3() -> None:
    """The refusal says which characters JSON forbids, and only mentions quotes for quotes."""
    from paw_kit.schema.grammar import _translate_field_pattern

    with pytest.raises(PAWSchemaError) as quote_exc:
        _translate_field_pattern(r'a"b')
    assert "the literal '\"'" in str(quote_exc.value)
    assert "double quote characters are forbidden" in str(quote_exc.value)

    with pytest.raises(PAWSchemaError) as tab_exc:
        _translate_field_pattern(r"\t+")
    assert "double quote" not in str(tab_exc.value), (
        "a tab-only class is refused for its own reason, not with the quote explanation"
    )


def test_refusal_names_the_construct_it_cannot_compile_S_3() -> None:
    """The success criterion is "raises PAWSchemaError NAMING the construct", not just raises.

    `\b` has two ways of failing: interegular refuses it as a reserved escape outside a
    character class, and *inside* one it is a backspace, whose JSON-safe intersection is
    empty. Both refuse, but only the first tells the author what is wrong with their
    pattern -- so the message, not merely the exception type, is the assertion.
    """
    from paw_kit.schema.grammar import _translate_field_pattern

    with pytest.raises(PAWSchemaError, match=r"Escape \\b is not implemented"):
        _translate_field_pattern(r"\bfoo\b")


def test_interegular_ast_surface_is_still_what_the_translator_expects() -> None:
    """Pin the private `interegular` surface `_translate_field_pattern` is built on.

    The translation reaches into `interegular.patterns` for its node types and its
    parser, because the library exports neither. That is a deliberate trade (the only
    alternative to an AST translation is writing a regex parser of our own), but it is
    a coupling to private names, so it gets a test that fails loudly on an upgrade
    rather than silently mistranslating.
    """
    import interegular
    from interegular.patterns import _CHAR_GROUPS, _CharGroup, _ParsePattern

    # The shorthand singletons the provenance marking keys on, by identity.
    parsed = _ParsePattern(r"\d").parse()
    group = parsed.options[0].parts[0]
    assert isinstance(group, _CharGroup)
    assert group is _CHAR_GROUPS["d"], "escaped() no longer returns the shared singleton"
    assert _CHAR_GROUPS["d"].chars == frozenset("0123456789")
    assert _CHAR_GROUPS["D"].negated is True

    # A bare inline-flag group sets `flags` on the parser and nothing on the node.
    parser = _ParsePattern(r"(?i)a")
    parser.parse()
    assert parser.flags is interegular.patterns.REFlags.CASE_INSENSITIVE

    # The six node types the renderer dispatches on.
    ast = interegular.parse_pattern(r"(?:a|b)*.")
    assert type(ast).__name__ == "Pattern"
    concat = ast.options[0]
    assert type(concat).__name__ == "_Concatenation"
    assert type(concat.parts[0]).__name__ == "_Repeated"
    assert type(concat.parts[1]).__name__ == "__DotCls"


# --- S-1: alternation, named groups and (?i) fall out of the S-3 translation ---------


@pytest.mark.parametrize(
    "pattern_src,legal,truncated",
    [
        (r"cat|dog", ("cat", "dog"), "cat"),
        (r"(cat|dog)s?", ("cats", "dog"), "cat"),
    ],
)
def test_alternation_is_grouped_before_it_is_spliced_S_1(
    pattern_src: str, legal: tuple, truncated: str
) -> None:
    """An un-grouped `|` became top level and rewrote the whole grammar (S-1).

    `Field(pattern="cat|dog")` compiled to `^\\{...\"x\"...:...\"cat|dog\"\\s*\\}$`, which
    means `...\"cat` OR `dog\"\\s*\\}$` -- so it rejected BOTH legal values and accepted
    the truncated, unparseable `{"x": "cat`.
    """
    import re as _re

    model = create_model("Alt", x=(str, Field(pattern=pattern_src)))
    pat = pydantic_to_regex(model, anchors=True)
    for value in legal:
        assert _re.match(pat, '{"x": "%s"}' % value) is not None, value
    assert _re.fullmatch(pat, '{"x": "%s' % truncated) is None, (
        "the grammar still accepts a truncated object that is not JSON at all"
    )


def test_named_groups_across_fields_do_not_collide_S_1() -> None:
    """Two fields sharing a group name made the exported string an invalid regex (S-1).

    Python `re` raises "redefinition of group name" -- i.e. `pydantic_to_regex` returned
    a string that is not a regex. The translation renders every group non-capturing, so
    there is no name left to collide.
    """
    import re as _re

    model = create_model(
        "Named",
        x=(str, Field(pattern=r"(?P<n>a)b")),
        y=(str, Field(pattern=r"(?P<n>c)d")),
    )
    pat = pydantic_to_regex(model, anchors=True)
    assert "(?P<" not in pat
    assert _re.match(_re.compile(pat), '{"x": "ab", "y": "cd"}') is not None


def test_leading_inline_ignorecase_is_honoured_not_refused_S_1() -> None:
    """`(?i)abc` is case-folded into explicit classes rather than rejected (S-1).

    Splicing it produced a string `re.compile()` refuses outright ("global flags not at
    the start of the expression"), because by then the flag group is in the middle of
    the assembled grammar.
    """
    import re as _re

    model = create_model("Ci", x=(str, Field(pattern=r"(?i)abc")))
    pat = pydantic_to_regex(model, anchors=True)
    assert "(?i)" not in pat
    compiled = _re.compile(pat)  # the S-1 failure mode is this line raising
    for value in ("abc", "aBc", "ABC"):
        assert compiled.match('{"x": "%s"}' % value) is not None, value
    assert compiled.match('{"x": "abd"}') is None


# --- S-6: a precompiled Field(pattern=...) compiles its source, not its repr ---------


def test_precompiled_pattern_compiles_its_source_not_its_repr_S_6() -> None:
    """`Field(pattern=re.compile("[a-z]+"))` must compile `[a-z]+` (S-6).

    `_extract_pattern_from_field` returned `str(meta.pattern)`, which for a precompiled
    pattern is `"re.compile('[a-z]+')"` -- so the grammar's accepted language was built
    out of the *repr*. Because that repr happens to contain lowercase letters,
    pydantic's search semantics then accepted it, and the nonsense flowed through
    `paw.load` as a successful result instead of a visible error.
    """
    import re as _re

    model = create_model("Precompiled", x=(str, Field(pattern=_re.compile(r"[a-z]+"))))
    pat = pydantic_to_regex(model, anchors=True)

    assert "compile" not in pat, f"the repr is still in the grammar: {pat!r}"
    assert _re.fullmatch(pat, '{"x": "abc"}') is not None, (
        "the legal value the pattern actually describes must be accepted"
    )
    assert _re.fullmatch(pat, '{"x": "re.compile(\'[a-z]+\')"}') is None, (
        "the grammar still accepts the pattern's own repr as a value"
    )


def test_precompiled_pattern_ignorecase_flag_is_honoured_S_6() -> None:
    """A precompiled pattern's `re.I` is part of the constraint, and pydantic honours it.

    Executed against pydantic 2.13.5: `Field(pattern=re.compile("abc", re.I))`
    validates `"ABC"`. Taking `.pattern` and dropping `.flags` would compile a grammar
    that rejects a value its own model accepts.
    """
    import re as _re

    model = create_model("CiPre", x=(str, Field(pattern=_re.compile(r"abc", _re.I))))
    model(x="ABC")  # precondition: pydantic itself accepts it
    pat = pydantic_to_regex(model, anchors=True)
    for value in ("abc", "ABC", "aBc"):
        assert _re.fullmatch(pat, '{"x": "%s"}' % value) is not None, value
    assert _re.fullmatch(pat, '{"x": "abd"}') is None


def test_precompiled_pattern_dotall_flag_is_accepted_and_subsumed_S_6() -> None:
    """`re.S` is expressed, but the JSON-safe intersection already subsumes it.

    `re.DOTALL` widens `.` to include a newline. A JSON string cannot carry a raw
    newline at all, so `_render_node`'s `.` branch intersects with `_JSON_UNSAFE_CHARS`
    -- which already contains `\\n` -- and the rendered class is byte-identical with and
    without the flag. The flag is therefore *honoured* (passed into the AST walk as
    `REFlags.SINGLE_LINE`) rather than refused, and the identity below is the point:
    the one direction the flag could have gone is widening, and it cannot.
    """
    import re as _re

    dotall = create_model("DotAll", x=(str, Field(pattern=_re.compile(r"a.b", _re.S))))
    plain = create_model("DotPlain", x=(str, Field(pattern=_re.compile(r"a.b"))))

    pat = pydantic_to_regex(dotall, anchors=True)
    assert pat == pydantic_to_regex(plain, anchors=True)
    assert _re.fullmatch(pat, '{"x": "axb"}') is not None
    assert _re.fullmatch(pat, '{"x": "a\nb"}') is None, (
        "a raw newline inside a JSON string is not valid JSON, flag or no flag"
    )


def test_precompiled_pattern_with_an_inexpressible_flag_raises_S_6() -> None:
    """re.VERBOSE changes how the source is tokenised, so it is refused, not dropped."""
    import re as _re

    model = create_model("Verbose", x=(str, Field(pattern=_re.compile(r"a b  # c", _re.X))))
    with pytest.raises(PAWSchemaError, match="precompiled regex carrying the flag"):
        pydantic_to_regex(model)


def test_precompiled_pattern_flags_join_the_cache_fingerprint_S_6_S_4() -> None:
    """Two models differing ONLY in a pattern flag must not share a cache entry.

    The invariant this track works to: any fix that widens what the compiler reads must
    widen the fingerprint in the same commit. Before S-6 the two spellings below were
    told apart only *accidentally* -- the flags showed up in the repr that was being
    (wrongly) compiled. Taking `.pattern` alone would have made them collide while
    compiling differently, which is exactly S-4's bug by another door.

    This is therefore one of the two tests on this branch that **passes at `main` by
    design** (the other pins interegular's private AST surface): it guards the hazard
    the S-6 fix introduces, not a defect that exists before it. `campaign.sh
    red-at-main` on it reports "every named test PASSES" -- expected, not a gate
    failure. Delete the flags from `_extract_pattern_from_field` and it goes red.
    """
    import re as _re

    from paw_kit.schema.grammar import _fingerprint_model_fields

    plain = create_model("FlagFp", x=(str, Field(pattern=_re.compile(r"abc"))))
    folded = create_model("FlagFp", x=(str, Field(pattern=_re.compile(r"abc", _re.I))))

    key_plain = _fingerprint_model_fields(plain, seen=frozenset(), depth=0)
    key_folded = _fingerprint_model_fields(folded, seen=frozenset(), depth=0)
    assert key_plain != key_folded, "the pattern flags are not in the fingerprint"

    pydantic_to_regex.cache_clear()
    assert pydantic_to_regex(plain) != pydantic_to_regex(folded), (
        "equal fingerprints would have served one model the other's grammar"
    )


# --- S-4: the cache fingerprint must not collide values that compare equal -----------


S_4_COLLISION_CASES = [
    ("literal_int_vs_bool", Literal[1], Literal[True], '{"x":1}', '{"x":true}'),
    ("literal_int_vs_float", Literal[1], Literal[1.0], '{"x":1}', '{"x":1.0}'),
    ("literal_zero_vs_false", Literal[0], Literal[False], '{"x":0}', '{"x":false}'),
]


@pytest.mark.parametrize(
    "name,ann_a,ann_b,dump_a,dump_b", S_4_COLLISION_CASES,
    ids=[c[0] for c in S_4_COLLISION_CASES],
)
def test_literal_values_that_compare_equal_do_not_share_a_cache_entry_S_4(
    name: str, ann_a: Any, ann_b: Any, dump_a: str, dump_b: str
) -> None:
    """`1 == True == 1.0` with equal hashes, so `("literal", (1,))` keyed all three.

    The failure is silent and order-dependent: compile `Literal[1]` first and
    `Literal[True]` gets its regex, so `B.model_dump_json()` -- `{"x":true}` -- does
    not match B's own grammar while `{"x":1}` does. The existing property test cannot
    see this, because it compares two *structurally identical* models.
    """
    import re as _re

    from paw_kit.schema.grammar import _fingerprint_model_fields

    model_a = create_model("S4Case", x=(ann_a, ...))
    model_b = create_model("S4Case", x=(ann_b, ...))

    key_a = _fingerprint_model_fields(model_a, seen=frozenset(), depth=0)
    key_b = _fingerprint_model_fields(model_b, seen=frozenset(), depth=0)
    assert key_a != key_b, f"{name}: the two models share a cache fingerprint"

    pydantic_to_regex.cache_clear()
    regex_a = pydantic_to_regex(model_a, anchors=True)  # compiled FIRST, so it wins a collision
    regex_b = pydantic_to_regex(model_b, anchors=True)
    assert regex_a != regex_b, f"{name}: the second model was served the first's regex"
    assert _re.fullmatch(regex_a, dump_a) is not None
    assert _re.fullmatch(regex_b, dump_b) is not None


def test_enum_member_values_that_compare_equal_do_not_share_a_cache_entry_S_4() -> None:
    """The same collision through an Enum's member values rather than a Literal's args."""
    import re as _re

    from paw_kit.schema.grammar import _fingerprint_model_fields

    class IntCode(enum.Enum):
        A = 1

    class BoolCode(enum.Enum):
        A = True

    model_a = create_model("S4Enum", x=(IntCode, ...))
    model_b = create_model("S4Enum", x=(BoolCode, ...))

    key_a = _fingerprint_model_fields(model_a, seen=frozenset(), depth=0)
    key_b = _fingerprint_model_fields(model_b, seen=frozenset(), depth=0)
    assert key_a != key_b, "an int-valued and a bool-valued enum share a fingerprint"

    pydantic_to_regex.cache_clear()
    regex_a = pydantic_to_regex(model_a, anchors=True)
    regex_b = pydantic_to_regex(model_b, anchors=True)
    assert regex_a != regex_b
    assert _re.fullmatch(regex_a, '{"x":1}') is not None
    assert _re.fullmatch(regex_b, '{"x":true}') is not None


# --- S-5: the grammar must require the key pydantic will VALIDATE --------------------


def test_field_alias_produces_the_key_pydantic_validates_S_5() -> None:
    """`Field(alias="fullName")` must compile to `"fullName"`, not `"full_name"` (S-5).

    With pydantic's default configuration the grammar required a key pydantic refuses
    and forbade the one it accepts, so through `paw.load` the local path failed 100% of
    the time and every call silently ran the fallback. The report's own "test needed"
    is this: assert the grammar matches `model_dump_json(by_alias=True)`.
    """
    import re as _re

    from pydantic import ConfigDict

    class Aliased(BaseModel):
        full_name: str = Field(alias="fullName")

    pat = pydantic_to_regex(Aliased, anchors=True)
    dumped = Aliased(fullName="v").model_dump_json(by_alias=True)
    assert dumped == '{"fullName":"v"}'
    assert _re.fullmatch(pat, dumped) is not None, (
        f"the grammar rejects the only key pydantic validates: {pat!r}"
    )
    with pytest.raises(Exception):
        Aliased.model_validate_json('{"full_name":"v"}')
    assert _re.fullmatch(pat, '{"full_name":"v"}') is None, (
        "the grammar still requires a key pydantic rejects"
    )
    assert ConfigDict  # keep the import used by the sibling tests honest


ALIAS_FLAG_CASES = [
    # (config, accepts_alias, accepts_field_name)
    ({}, True, False),
    ({"populate_by_name": True}, True, True),
    ({"validate_by_name": True}, True, True),
    ({"validate_by_alias": False}, False, True),
]


@pytest.mark.parametrize(
    "config,accepts_alias,accepts_name", ALIAS_FLAG_CASES,
    ids=["default", "populate_by_name", "validate_by_name", "validate_by_alias_false"],
)
def test_alias_gating_follows_all_three_config_flags_S_5(
    config: dict, accepts_alias: bool, accepts_name: bool
) -> None:
    """Three flags decide which keys validate, not one (S-5, executed matrix).

    `populate_by_name` is the old spelling; pydantic 2.11 added `validate_by_name` and
    `validate_by_alias`, and `validate_by_alias=False` makes the *alias* the rejected
    key. A one-flag reading of this model gets two of these four rows wrong.

    Three of the four rows are red at `main`. The `validate_by_alias_false` row passes
    there, because it is the one configuration in which the key pydantic validates IS
    the field name -- i.e. the one row `main`'s unconditional "always emit the field
    name" happens to get right. It is kept so the matrix is complete and so a future
    change cannot make that row wrong unnoticed.
    """
    import json as _json
    import re as _re

    from pydantic import ConfigDict, ValidationError

    class Flagged(BaseModel):
        model_config = ConfigDict(**config)
        full_name: str = Field(alias="fullName")

    pat = pydantic_to_regex(Flagged, anchors=True)
    for key, expected in (("fullName", accepts_alias), ("full_name", accepts_name)):
        payload = _json.dumps({key: "v"})
        # Precondition: pydantic's own answer. If this fails the matrix is wrong.
        try:
            Flagged.model_validate_json(payload)
            pydantic_accepts = True
        except ValidationError:
            pydantic_accepts = False
        assert pydantic_accepts is expected, f"{config}: pydantic's answer for {key!r} changed"
        assert (_re.fullmatch(pat, payload) is not None) is expected, (
            f"{config}: the grammar disagrees with pydantic about the key {key!r}"
        )


def test_validation_alias_is_targeted_never_serialization_alias_S_5() -> None:
    """The grammar follows the VALIDATION alias; the serialization one is a trap.

    Executed against pydantic 2.13.5: with `validation_alias="vName",
    serialization_alias="sName"`, pydantic validates `vName` while
    `model_dump_json(by_alias=True)` emits `{"sName": ...}` -- which pydantic then
    **rejects**. A grammar built from the serialization alias is one the validator
    never accepts, so `by_alias=True` output is not the target here.
    """
    import re as _re

    from pydantic import ValidationError

    class Split(BaseModel):
        full_name: str = Field(validation_alias="vName", serialization_alias="sName")

    pat = pydantic_to_regex(Split, anchors=True)
    assert _re.fullmatch(pat, '{"vName":"v"}') is not None
    assert _re.fullmatch(pat, '{"sName":"v"}') is None
    # ... and the reason: pydantic itself will not read back its own by_alias dump.
    dumped = Split.model_validate({"vName": "v"}).model_dump_json(by_alias=True)
    assert dumped == '{"sName":"v"}'
    with pytest.raises(ValidationError):
        Split.model_validate_json(dumped)


def test_alias_choices_compile_to_an_alternation_S_5() -> None:
    """Every choice validates, so every choice is in the grammar."""
    import re as _re

    from pydantic import AliasChoices

    class Choices(BaseModel):
        full_name: str = Field(validation_alias=AliasChoices("a", "b"))

    pat = pydantic_to_regex(Choices, anchors=True)
    for key in ("a", "b"):
        Choices.model_validate({key: "v"})  # precondition
        assert _re.fullmatch(pat, '{"%s":"v"}' % key) is not None, key
    assert _re.fullmatch(pat, '{"full_name":"v"}') is None


def test_alias_choices_drop_an_alias_path_member_rather_than_refusing_S_5() -> None:
    """A flat choice alongside an AliasPath keeps working; the path is simply not emitted.

    Executed: `AliasChoices("a", AliasPath("x", "y"))` validates `{"a": "v"}`, so a
    grammar that emits only `a` is sound. Dropping the path is the narrowing direction.
    """
    import re as _re

    from pydantic import AliasChoices, AliasPath

    class Mixed(BaseModel):
        full_name: str = Field(validation_alias=AliasChoices("a", AliasPath("x", "y")))

    pat = pydantic_to_regex(Mixed, anchors=True)
    assert _re.fullmatch(pat, '{"a":"v"}') is not None


def test_alias_path_raises_rather_than_compiling_a_flat_key_S_5() -> None:
    """An AliasPath addresses a value inside a nested object; the grammar is flat."""
    from pydantic import AliasChoices, AliasPath

    class Pathed(BaseModel):
        full_name: str = Field(validation_alias=AliasPath("outer", "inner"))

    with pytest.raises(PAWSchemaError, match="AliasPath addresses a value nested"):
        pydantic_to_regex(Pathed)

    class OnlyPaths(BaseModel):
        full_name: str = Field(validation_alias=AliasChoices(AliasPath("outer", "inner")))

    with pytest.raises(PAWSchemaError, match="AliasPath addresses a value nested"):
        pydantic_to_regex(OnlyPaths)


def test_alias_generator_needs_no_special_handling_S_5() -> None:
    """`alias_generator` is resolved into each FieldInfo at class build, so it just works."""
    import re as _re

    from pydantic import ConfigDict

    class Generated(BaseModel):
        model_config = ConfigDict(alias_generator=lambda s: s.upper())
        full_name: str

    assert Generated.model_fields["full_name"].alias == "FULL_NAME"
    pat = pydantic_to_regex(Generated, anchors=True)
    assert _re.fullmatch(pat, '{"FULL_NAME":"v"}') is not None
    assert _re.fullmatch(pat, '{"full_name":"v"}') is None


def test_alias_is_json_escaped_like_a_literal_value_S_5() -> None:
    """An alias is an arbitrary string, so it gets PAW-SCHEMA-01 treatment, not re.escape.

    pydantic accepts `Field(alias='fu"ll')`. Splicing that into `f'"{re.escape(name)}"'`
    would close the JSON key's own string early -- the same break-out PAW-SCHEMA-01 is
    about, reached through the key rather than through a Literal value.
    """
    import json as _json
    import re as _re

    class Quoted(BaseModel):
        full_name: str = Field(alias='fu"ll')

    payload = _json.dumps({'fu"ll': "v"})
    Quoted.model_validate_json(payload)  # precondition
    pat = pydantic_to_regex(Quoted, anchors=True)
    assert _re.fullmatch(pat, payload) is not None
    assert _re.compile(pat)


def test_alias_joins_the_cache_fingerprint_S_5() -> None:
    """Two models differing only in an alias -- or only in a config flag -- must not collide."""
    from pydantic import ConfigDict

    from paw_kit.schema.grammar import _fingerprint_model_fields

    class Plain(BaseModel):
        full_name: str

    class Aliased(BaseModel):
        full_name: str = Field(alias="fullName")

    class AliasedBoth(BaseModel):
        model_config = ConfigDict(populate_by_name=True)
        full_name: str = Field(alias="fullName")

    keys = [
        _fingerprint_model_fields(m, seen=frozenset(), depth=0)
        for m in (Plain, Aliased, AliasedBoth)
    ]
    assert len(set(keys)) == 3, "two of these three models share a cache fingerprint"

    pydantic_to_regex.cache_clear()
    regexes = [pydantic_to_regex(m) for m in (Plain, Aliased, AliasedBoth)]
    assert len(set(regexes)) == 3, "a model was served another model's grammar"


def test_alias_on_a_nested_model_is_honoured_S_5() -> None:
    """The alias resolution runs per model, so a nested model uses its OWN config."""
    import re as _re

    class InnerAliased(BaseModel):
        inner_name: str = Field(alias="innerName")

    class OuterPlain(BaseModel):
        detail: InnerAliased

    pat = pydantic_to_regex(OuterPlain, anchors=True)
    payload = '{"detail": {"innerName": "v"}}'
    OuterPlain.model_validate_json(payload)  # precondition
    assert _re.fullmatch(pat, payload) is not None


# --- S-8: an unanchored Field(pattern=...) is still full-matched (DECIDED) -----------


def test_unanchored_pattern_is_still_full_matched_S_8() -> None:
    """The decision, pinned: the grammar full-matches even an unanchored pattern.

    pydantic applies `pattern` with *search* semantics, so
    `Field(pattern=r"[0-9]{5}")` validates `"x90210y"` while the grammar does not. That
    divergence is deliberate. A full match is a *subset* of a search, so this is the
    narrowing direction and cannot produce an unsound grammar; emulating search would
    make the constraint nearly vacuous for a decoder, which is the opposite of the
    point of constraining one.

    **This test passes at `main` by design**, because S-8 is a recorded decision rather
    than a defect -- the grammar full-matches today too. What it adds is that the
    decision is now pinned: the existing
    `test_field_pattern_constraint_used_in_regex` asserts only that `"90210"` matches,
    which is true under either semantics, so nothing stopped a later change from
    quietly switching to search emulation. This is the assertion the bug hunt said was
    missing.
    """
    import re as _re

    from pydantic import ValidationError

    model = create_model("Unanchored", x=(str, Field(pattern=r"[0-9]{5}")))
    pat = pydantic_to_regex(model, anchors=True)

    # pydantic SEARCHES, so these are legal values for the model...
    for value in ("x90210y", "902105", "2026-90210"):
        model(x=value)
        assert _re.fullmatch(pat, '{"x": "%s"}' % value) is None, (
            f"the grammar accepted {value!r}: it is emulating search, not full-matching"
        )

    # ... and the full match is what the grammar emits.
    assert _re.fullmatch(pat, '{"x": "90210"}') is not None
    with pytest.raises(ValidationError):
        model(x="9021")  # too short for either reading


# --- S-9: length and range constraints are honoured, or warned about ----------------


S_9_EXPRESSIBLE = [
    # (id, annotation, Field kwargs, accepted values, rejected values) -- each value is
    # the whole JSON object, and pydantic's own verdict is asserted alongside.
    ("str_both", str, dict(min_length=2, max_length=5),
     ['{"x":"ab"}', '{"x":"abcde"}', '{"x":"a\\u00e9"}'],
     ['{"x":"a"}', '{"x":"abcdef"}', '{"x":""}']),
    ("str_min_only", str, dict(min_length=2),
     ['{"x":"ab"}', '{"x":"abcdefghij"}'], ['{"x":"a"}', '{"x":""}']),
    ("str_max_zero", str, dict(max_length=0), ['{"x":""}'], ['{"x":"a"}']),
    ("list_both", List[int], dict(min_length=1, max_length=3),
     ['{"x":[1]}', '{"x":[1,2,3]}'], ['{"x":[]}', '{"x":[1,2,3,4]}']),
    ("list_max_zero", List[int], dict(max_length=0), ['{"x":[]}'], ['{"x":[1]}']),
    ("list_max_only", List[int], dict(max_length=2),
     ['{"x":[]}', '{"x":[1,2]}'], ['{"x":[1,2,3]}']),
    ("set_min", Set[int], dict(min_length=2), ['{"x":[1,2]}'], ['{"x":[1]}']),
    ("dict_both", Dict[str, int], dict(min_length=1, max_length=2),
     ['{"x":{"a":1}}'], ['{"x":{}}', '{"x":{"a":1,"b":2,"c":3}}']),
    ("tuple_variadic", Tuple[int, ...], dict(min_length=1, max_length=2),
     ['{"x":[1]}', '{"x":[1,2]}'], ['{"x":[]}', '{"x":[1,2,3]}']),
    ("int_ge_le", int, dict(ge=0, le=10),
     ['{"x":0}', '{"x":10}'], ['{"x":11}', '{"x":-1}']),
    ("int_gt_lt", int, dict(gt=0, lt=5), ['{"x":1}', '{"x":4}'], ['{"x":0}', '{"x":5}']),
    ("int_negative", int, dict(ge=-3, le=2),
     ['{"x":-3}', '{"x":2}'], ['{"x":-4}', '{"x":3}']),
]


@pytest.mark.parametrize(
    "name,annotation,kwargs,accepted,rejected", S_9_EXPRESSIBLE,
    ids=[c[0] for c in S_9_EXPRESSIBLE],
)
def test_expressible_constraints_are_enforced_by_the_grammar_S_9(
    name: str, annotation: Any, kwargs: dict, accepted: List[str], rejected: List[str]
) -> None:
    """`Field(ge=0, le=10)` accepted `{"x": 99}`; min/max length were dropped entirely.

    Several of these are straightforwardly expressible as a regex, so the silence was a
    gap and not a limit. The grammar and pydantic must now agree on every value below,
    in both directions -- pydantic's verdict is asserted first, so a wrong row in this
    table fails as a wrong row rather than as a compiler bug.
    """
    import re as _re
    import warnings as _warnings

    from pydantic import ValidationError

    model = create_model("S9_" + name, x=(annotation, Field(**kwargs)))
    pydantic_to_regex.cache_clear()
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # an expressible constraint must NOT warn
        pat = pydantic_to_regex(model, anchors=True)

    for value, expected in [(v, True) for v in accepted] + [(v, False) for v in rejected]:
        try:
            model.model_validate_json(value)
            pydantic_accepts = True
        except ValidationError:
            pydantic_accepts = False
        assert pydantic_accepts is expected, f"{name}: the table's verdict for {value!r} is wrong"
        assert (_re.fullmatch(pat, value) is not None) is expected, (
            f"{name}: the grammar disagrees with pydantic about {value!r}: {pat!r}"
        )


def test_string_length_counts_decoded_characters_not_escape_sequences_S_9() -> None:
    """The `{m,n}` goes on the character alternation, which is what pydantic measures.

    `"a\\u00e9"` is nine characters of JSON and two characters of string. pydantic
    measures the decoded value, and so must the quantifier -- which it does, because
    each repetition of the alternation matches exactly one decoded character however it
    is spelled.
    """
    import re as _re

    model = create_model("S9Escapes", x=(str, Field(min_length=2, max_length=2)))
    pat = pydantic_to_regex(model, anchors=True)
    for value in ('{"x":"a\\u00e9"}', '{"x":"a\\n"}', '{"x":"\\\\\\""}', '{"x":"ab"}'):
        model.model_validate_json(value)  # precondition: exactly two decoded characters
        assert _re.fullmatch(pat, value) is not None, value
    assert _re.fullmatch(pat, '{"x":"a\\u00e9b"}') is None


S_9_WARNED = [
    ("optional_str_length", Optional[str], dict(min_length=2), "min_length/max_length"),
    ("float_range", float, dict(ge=0.0, le=1.0), "ge/le"),
    ("open_ended_int", int, dict(ge=0), "ge"),
    ("too_wide_int", int, dict(ge=0, le=100_000), "ge/le"),
    ("multiple_of", int, dict(multiple_of=3), "MultipleOf"),
    ("decimal_digits", Decimal, dict(max_digits=5, decimal_places=2), "max_digits"),
    ("pattern_plus_length", str, dict(pattern=r"[a-z]+", min_length=2),
     "alongside a pattern="),
]


@pytest.mark.parametrize(
    "name,annotation,kwargs,expected_text", S_9_WARNED, ids=[c[0] for c in S_9_WARNED],
)
def test_inexpressible_constraints_warn_rather_than_vanish_S_9(
    name: str, annotation: Any, kwargs: dict, expected_text: str
) -> None:
    """A constraint the grammar cannot enforce is said out loud, once, naming the field.

    Not a raise: these schemas compile and work today, and the parent track's
    fail-open invariant forbids a new raise on a path the caller depends on. What was
    wrong was the silence -- a caller who writes `Field(multiple_of=3)` and reads
    "syntax compliance guaranteed mathematically" has no way to learn that this
    particular guarantee does not cover their constraint.
    """
    model = create_model("S9Warn_" + name, x=(annotation, Field(**kwargs)))
    pydantic_to_regex.cache_clear()
    with pytest.warns(UserWarning) as caught:
        pydantic_to_regex(model, anchors=True)

    messages = [str(w.message) for w in caught if "does not enforce" in str(w.message)]
    assert len(messages) == 1, f"expected exactly one warning per field, got {messages}"
    assert f"S9Warn_{name}.x" in messages[0], "the warning must name the field"
    assert expected_text in messages[0], f"{expected_text!r} missing from {messages[0]!r}"


def test_a_field_with_several_inexpressible_constraints_warns_once_S_9() -> None:
    """"Warn once per field" means one line listing them, not one line each."""
    model = create_model(
        "S9Multi", x=(int, Field(ge=0, multiple_of=3)), y=(int, Field(le=5, multiple_of=7))
    )
    pydantic_to_regex.cache_clear()
    with pytest.warns(UserWarning) as caught:
        pydantic_to_regex(model, anchors=True)

    messages = [str(w.message) for w in caught if "does not enforce" in str(w.message)]
    assert len(messages) == 2, f"one warning per FIELD, not per constraint: {messages}"
    assert "ge" in messages[0] and "MultipleOf" in messages[0]


def test_every_length_bounded_annotation_actually_changes_the_regex_S_9() -> None:
    """Pin the coupling between `_length_constraint_kind` and `_type_to_regex`.

    `_length_constraint_kind` mirrors a subset of `_type_to_regex`'s dispatch, which is
    a drift risk of exactly the shape `_fingerprint_annotation`'s docstring warns
    about: if the two disagree, a bound this compiler *claims* to honour is threaded
    into a branch that ignores it and is dropped silently -- the very failure S-9 is
    about. So for every annotation the predicate claims, adding a bound must visibly
    change the compiled grammar, and for every annotation it disclaims, it must not.
    """
    from paw_kit.schema.grammar import _length_constraint_kind

    claimed = [
        str, List[int], Set[int], FrozenSet[int], Dict[str, int], Tuple[int, ...],
        list, set, frozenset, dict,
    ]
    disclaimed = [int, float, bool, Optional[str], Tuple[int, str], Decimal, uuid.UUID]

    for annotation in claimed:
        assert _length_constraint_kind(annotation) is not None, annotation
        pydantic_to_regex.cache_clear()
        plain = pydantic_to_regex(create_model("Plain", x=(annotation, ...)))
        bounded = pydantic_to_regex(
            create_model("Bounded", x=(annotation, Field(min_length=1, max_length=2)))
        )
        assert plain != bounded, (
            f"_length_constraint_kind claims {annotation!r} but the bound changed nothing"
        )

    for annotation in disclaimed:
        assert _length_constraint_kind(annotation) is None, annotation


def test_constraints_join_the_cache_fingerprint_S_9() -> None:
    """Two models differing only in a constraint must not share a cache entry.

    Including a constraint the compiler can only *warn* about: `multiple_of=3` and
    `multiple_of=7` compile to the same regex today, but they must not be told apart by
    luck if one of them ever becomes expressible.
    """
    from paw_kit.schema.grammar import _fingerprint_model_fields

    variants = [
        create_model("S9Fp", x=(int, ...)),
        create_model("S9Fp", x=(int, Field(ge=0, le=10))),
        create_model("S9Fp", x=(int, Field(ge=0, le=11))),
        create_model("S9Fp", x=(int, Field(multiple_of=3))),
        create_model("S9Fp", x=(int, Field(multiple_of=7))),
    ]
    keys = [_fingerprint_model_fields(m, seen=frozenset(), depth=0) for m in variants]
    assert len(set(keys)) == len(variants), "two of these models share a cache fingerprint"

    pydantic_to_regex.cache_clear()
    with pytest.warns(UserWarning):
        regexes = [pydantic_to_regex(m) for m in variants]
    assert regexes[1] != regexes[2], "ge=0,le=10 and ge=0,le=11 compiled identically"


def test_int_range_enumeration_is_capped_S_9() -> None:
    """A range wider than the cap is warned about rather than enumerated into the regex."""
    from paw_kit.schema.grammar import _MAX_ENUMERATED_INT_RANGE

    pydantic_to_regex.cache_clear()
    at_cap = create_model("S9AtCap", x=(int, Field(ge=1, le=_MAX_ENUMERATED_INT_RANGE)))
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        pat = pydantic_to_regex(at_cap, anchors=True)
    assert f"|{_MAX_ENUMERATED_INT_RANGE})" in pat

    over_cap = create_model("S9OverCap", x=(int, Field(ge=1, le=_MAX_ENUMERATED_INT_RANGE + 1)))
    with pytest.warns(UserWarning, match="does not enforce"):
        pydantic_to_regex(over_cap, anchors=True)
