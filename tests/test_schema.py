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
