"""Engine tests for `paw_kit.schema.constraint`, the llguidance-backed successor to
`logits_processor.py`'s `RegexLogitsProcessor` (Phase 2 of `constrained-decoding-real-
backend` deletes that module).

`pytest.importorskip("llguidance")` at module top: these are engine tests, not
CI-portable ones -- the engine-absent configuration (backend degrades to unconstrained
decoding) is covered in `tests/test_programasweights_backend.py` instead, which does
not need `llguidance` to be importable.

A small synthetic byte-level vocabulary is built in-process below (`_build_test_tokens`
et al.) rather than requiring the real GGUF model: every byte 0-255 is its own
single-byte token (so any byte sequence is spellable), plus one dedicated multi-byte
token for `'é'` (probe 2) and deliberately NO dedicated token for `'ϩ'`
(probe 3), matching the plan's requirement that probe 3 be spellable only through
byte-fallback.

No red-first claim is made for anything here: the module this replaces is being
deleted, not fixed, so there is no red state at `main` to demonstrate turned green.
"""

from __future__ import annotations

import time
import warnings
from typing import List, Union

import interegular
from interegular.fsm import anything_else
import numpy as np
import pytest

llguidance = pytest.importorskip("llguidance")

from pydantic import BaseModel, create_model
from typing import Literal

from paw_kit.schema.constraint import (
    INITIAL_LEXER_FUEL,
    PROBES,
    ConstraintUnavailable,
    Vocabulary,
    _GTokenizerAdapter,
    build_constraint,
)
from paw_kit.schema.exceptions import PAWSchemaError
from paw_kit.schema.grammar import pydantic_to_regex

from tests.test_schema_spine import SPINE_CASES, SPINE_IDS


# ---------------------------------------------------------------------------------
# Synthetic byte-level vocabulary
# ---------------------------------------------------------------------------------

def _build_test_tokens() -> "tuple[List[bytes], int]":
    tokens: List[bytes] = [bytes([i]) for i in range(256)]
    tokens.append("é".encode("utf-8"))  # id 256: dedicated 'é' token (probe 2)
    eos = len(tokens)
    tokens.append(b"<eos>")
    return tokens, eos


TOKENS, EOS = _build_test_tokens()
N_VOCAB = len(TOKENS)

_BY_BYTES = {}
for _i, _b in enumerate(TOKENS):
    if _b and _b not in _BY_BYTES:
        _BY_BYTES[_b] = _i
_MAXLEN = max(len(b) for b in _BY_BYTES)


def canonical_encode(x: Union[bytes, str]) -> List[int]:
    """Greedy longest-match over TOKENS -- exactly the shape of a real byte-fallback
    BPE tokenizer's encoder, and the encoder every adversarial case below is a
    variation on."""
    if isinstance(x, str):
        x = x.encode("utf-8")
    out: List[int] = []
    i = 0
    while i < len(x):
        for ln in range(min(_MAXLEN, len(x) - i), 0, -1):
            tid = _BY_BYTES.get(x[i:i + ln])
            if tid is not None:
                out.append(tid)
                i += ln
                break
        else:  # pragma: no cover -- unreachable: every single byte has its own token
            i += 1
    return out


def make_vocabulary(encode=canonical_encode) -> Vocabulary:
    return Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=encode)


def tid(b: bytes) -> int:
    return _BY_BYTES[b]


def _allowed_from_masked(out: np.ndarray) -> "set[int]":
    """Recover the allowed-id set from a `_Constraint`-masked output array: llguidance
    sets disallowed logits to -inf (verified against this exact setup), so finite
    positions are exactly the allowed set."""
    return set(np.nonzero(np.isfinite(out))[0].tolist())


class Simple(BaseModel):
    s: str


class Kind(BaseModel):
    kind: Literal["mobile", "landline", "unknown"]


# ---------------------------------------------------------------------------------
# Vocabulary: encoder round-trip verification (K-2 / J-3 table)
# ---------------------------------------------------------------------------------

def test_probe_3_has_no_whole_character_token_in_test_vocabulary() -> None:
    assert "ϩ".encode("utf-8") not in _BY_BYTES


def test_vocabulary_construction_succeeds_with_correct_encoder() -> None:
    vocab = make_vocabulary()
    assert vocab.eos_token_id == EOS
    assert vocab.special_token_ids == (EOS,)


def test_encoder_round_trip_refuses_absent_encoder() -> None:
    with pytest.raises(PAWSchemaError, match="requires an encode callable"):
        Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=None)


def test_encoder_round_trip_refuses_empty_id_list() -> None:
    with pytest.raises(PAWSchemaError, match="empty id list"):
        Vocabulary(
            tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=lambda x: []
        )


def test_encoder_round_trip_refuses_wrong_id() -> None:
    with pytest.raises(PAWSchemaError, match="round-trip"):
        Vocabulary(
            tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=lambda x: [0]
        )


def test_encoder_round_trip_refuses_ascii_ok_junk_on_multibyte() -> None:
    def enc(x: Union[bytes, str]) -> List[int]:
        b = x if isinstance(x, bytes) else x.encode("utf-8")
        if any(byte >= 0x80 for byte in b):
            return [0]  # correct on the ASCII probe, wrong on the multi-byte ones
        return canonical_encode(x)

    with pytest.raises(PAWSchemaError, match="round-trip"):
        Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=enc)


def test_encoder_round_trip_refuses_dropped_last_token() -> None:
    def enc(x: Union[bytes, str]) -> List[int]:
        ids = canonical_encode(x)
        return ids[:-1] if len(ids) > 1 else ids

    with pytest.raises(PAWSchemaError, match="round-trip"):
        Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=enc)


def test_encoder_round_trip_refuses_raising_encoder() -> None:
    def enc(x: Union[bytes, str]) -> List[int]:
        raise ValueError("boom")

    with pytest.raises(PAWSchemaError, match="boom"):
        Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=enc)


def test_encoder_round_trip_passes_with_correct_encoder() -> None:
    # Must not raise.
    Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=canonical_encode)


# ---------------------------------------------------------------------------------
# Construction-time ordering (K-2): compute the initial mask BEFORE reading is_error()
# ---------------------------------------------------------------------------------

def test_construction_time_check_computes_initial_mask_before_reading_is_error() -> None:
    """An encoder that passes the three fixed PROBES (so `Vocabulary.__init__` accepts
    it) but raises on any OTHER input fails only once `llguidance` calls it lazily --
    at `LLTokenizer` construction with `b'test'`, or later resolving a grammar
    literal's forced bytes -- which leaves `is_error()` False immediately after
    construction and True only after the first `compute_bitmask()`. If
    `build_constraint` read `is_error()` before computing that first mask, this
    failure would be silently missed (K-2's central claim, verified live here)."""
    probe_bytes = {p if isinstance(p, bytes) else p.encode("utf-8") for p in PROBES}

    def enc(x: Union[bytes, str]) -> List[int]:
        b = x if isinstance(x, bytes) else x.encode("utf-8")
        if b in probe_bytes:
            return canonical_encode(x)
        raise ValueError(f"boom on unprobed input {b!r}")

    vocab = Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=enc)
    pat = pydantic_to_regex(Kind)
    # ConstraintUnavailable, not plain PAWSchemaError: this is a construction-time
    # refusal (Change 2) -- a property of `pat`/`vocab`, not of one call, and
    # ProgramAsWeightsBackend.infer() catches this subclass specifically to degrade
    # PER SCHEMA rather than propagate a refusal that would recur on every call.
    with pytest.raises(ConstraintUnavailable, match="llguidance grammar construction failed"):
        build_constraint(pat, vocab)


# ---------------------------------------------------------------------------------
# Construction-time EOS-only / empty allowed-set check
# ---------------------------------------------------------------------------------

def test_empty_pattern_refused_as_eos_only_at_construction() -> None:
    """A pattern matching only the empty string admits nothing but EOS as its first
    token -- refused at construction, independent of the encoder-ordering check
    above (round 3's J-3: this check catches a case the encoder check does not). Also
    `ConstraintUnavailable` (Change 2): this is a construction-time refusal, a property
    of the pattern, not of one call."""
    vocab = make_vocabulary()
    with pytest.raises(ConstraintUnavailable, match="EOS as the first token"):
        build_constraint(r"", vocab)


def test_ordinary_pattern_constructs_without_eos_only_false_positive() -> None:
    vocab = make_vocabulary()
    build_constraint(r"x", vocab)  # must not raise


def test_the_llguidance_tokenizer_is_built_once_per_vocabulary_not_once_per_call(
    monkeypatch,
) -> None:
    """RED-FIRST (Phase 4 measurement). `LLTokenizer` construction walks the whole token
    table and is a pure function of the vocabulary, so it belongs with the vocabulary
    object -- which the backend already builds once per model and evicts with the model.

    Measured on the real 151,936-token GGUF vocabulary before this was cached:
    `build_constraint` cost **251.65 ms**, of which **249.13 ms** was this one call;
    the matcher plus its initial mask was 1.35 ms and `grammar_from_regex` 0.01 ms.
    Paid per `infer()` call, that was **13.118 ms** of end-to-end overhead per generated
    token against `roadmap.md`'s `<2 ms` budget. The per-call object that must stay
    per-call is the **matcher**, because an `LLMatcher` dies on error; the tokenizer has
    no such state, and the two are asserted apart below.
    """
    vocab = make_vocabulary()
    built: List[int] = []
    real_tokenizer = llguidance.LLTokenizer

    def _counting(*args, **kwargs):
        built.append(1)
        return real_tokenizer(*args, **kwargs)

    monkeypatch.setattr(llguidance, "LLTokenizer", _counting)

    first = build_constraint(r"x", vocab)
    second = build_constraint(r"x", vocab)

    assert len(built) == 1, "LLTokenizer was rebuilt for the second constraint"
    # ... and the matcher is still strictly per call.
    assert first is not second
    assert first._matcher is not second._matcher


# ---------------------------------------------------------------------------------
# Masking effect (H-3)
# ---------------------------------------------------------------------------------

def test_masking_effect_invocations_and_nonzero_masked_counts() -> None:
    vocab = make_vocabulary()
    pat = pydantic_to_regex(Simple)
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("hello, unrelated model input")
    generated_text = '{"s": "hi"}'
    generated_ids = [canonical_encode(ch)[0] for ch in generated_text]

    gen = list(prompt_ids)
    outputs = []
    for next_id in generated_ids:
        out = c(gen, scores)
        outputs.append(out)
        assert next_id in _allowed_from_masked(out)
        gen.append(next_id)
    # One further call: the mask that allows EOS to close generation.
    outputs.append(c(gen, scores))

    assert c.invocations == len(outputs)
    assert len(c.masked_per_step) == len(outputs)
    assert all(m > 0 for m in c.masked_per_step)
    assert all(o.shape == (N_VOCAB,) for o in outputs)
    assert all(o.dtype == np.float32 for o in outputs)
    # Mask applied to a COPY, never the caller's own buffer.
    assert all(o is not scores for o in outputs)
    assert (scores == 0.0).all()  # caller's original array untouched


# ---------------------------------------------------------------------------------
# Prompt offset (H-2)
# ---------------------------------------------------------------------------------

def test_prompt_offset_first_call_does_not_raise_and_consumes_nothing() -> None:
    vocab = make_vocabulary()
    pat = pydantic_to_regex(Simple)
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("some unrelated prompt text long enough to matter")
    c(prompt_ids, scores)  # must not raise
    assert c.invocations == 1
    assert c._next_index == len(prompt_ids)
    assert c._prompt_len == len(prompt_ids)


# ---------------------------------------------------------------------------------
# consume_token() False / post-mask is_error() (both raise PAWSchemaError)
# ---------------------------------------------------------------------------------

def test_consume_token_false_raises_paw_schema_error_and_writes_rust_warning(capfd) -> None:
    vocab = make_vocabulary()
    pat = pydantic_to_regex(Simple)  # starts with '{'
    c = build_constraint(pat, vocab)  # construction itself succeeds -- not ConstraintUnavailable
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("x")
    c(prompt_ids, scores)  # establishes the prompt offset
    bad_token = tid(b"z")  # illegal first token: the grammar demands '{'
    # Mid-generation (Change 2): a refused token during the walk is NOT a property of
    # the pattern alone (partial output may already exist), so this stays plain
    # PAWSchemaError -- never ConstraintUnavailable, which is reserved for
    # construction-time refusals build_constraint() itself raises.
    with pytest.raises(PAWSchemaError, match="rejected token") as excinfo:
        c(prompt_ids + [bad_token], scores)
    assert not isinstance(excinfo.value, ConstraintUnavailable)

    # llguidance's error state is quiet by construction on stdout/stderr redirection
    # from Python's own `warnings` module -- it is a Rust-side `Warning:` line on fd 2,
    # invisible to `redirect_stderr`. `capfd` (not `capsys`) is required to see it.
    captured = capfd.readouterr()
    assert "Warning:" in captured.err


def test_post_mask_is_error_raises_paw_schema_error_on_a_later_step() -> None:
    """A masking failure is not only a `consume_token()` rejection: the encoder is
    also consulted lazily *during* mask computation (resolving a grammar literal's
    forced bytes). An encoder that is correct everywhere except a specific literal
    ('mobile') passes construction and several ordinary steps, then fails only once
    the walk enters that literal -- and must still raise `PAWSchemaError`, not just
    silently produce a stale/garbage mask."""

    def enc(x: Union[bytes, str]) -> List[int]:
        b = x if isinstance(x, bytes) else x.encode("utf-8")
        if b"mobile" in b:
            raise ValueError("boom on grammar literal")
        return canonical_encode(x)

    vocab = Vocabulary(tokens=TOKENS, eos_token_id=EOS, special_token_ids=(EOS,), encode=enc)
    pat = pydantic_to_regex(Kind)
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("x")
    gen = list(prompt_ids)
    c(gen, scores)

    prefix = '{"kind": "m'  # 'm' is where the encoder is asked to spell "mobile"
    raised = False
    for ch in prefix:
        gen.append(canonical_encode(ch)[0])
        try:
            c(gen, scores)
        except PAWSchemaError as exc:
            raised = True
            assert "error state while computing the mask" in str(exc)
            # Mid-generation (Change 2): not ConstraintUnavailable -- construction
            # already succeeded; this failure surfaces only once the walk reaches the
            # literal the encoder is wrong on, so it is not a property of the pattern
            # alone and must stay plain PAWSchemaError.
            assert not isinstance(exc, ConstraintUnavailable)
            break
    assert raised, "expected a post-mask is_error() to raise PAWSchemaError"


# ---------------------------------------------------------------------------------
# Byte-level property as a sequence (Motivation's central claim)
# ---------------------------------------------------------------------------------

def test_byte_level_property_lead_and_continuation_bytes_at_string_content_state() -> None:
    vocab = make_vocabulary()
    pat = pydantic_to_regex(Simple)
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("x")
    gen = list(prompt_ids)
    out = c(gen, scores)
    for ch in '{"s": "':  # walk to a string-content state
        gen.append(canonical_encode(ch)[0])
        out = c(gen, scores)
    allowed = _allowed_from_masked(out)

    lead = tid(b"\xc3")   # lead byte of a 2-byte UTF-8 sequence
    cont = tid(b"\xa9")   # a bare continuation byte
    quote = tid(b'"')

    assert lead in allowed, "a lone lead byte must be admitted at a string-content state"
    assert cont not in allowed, "a lone continuation byte must be refused"
    assert quote in allowed, "the empty string is legal, so the string can close here"

    gen.append(lead)
    out = c(gen, scores)
    allowed_after_lead = _allowed_from_masked(out)
    assert cont in allowed_after_lead, "after a lead byte, its continuation must be admitted"
    assert lead not in allowed_after_lead, "a second lead byte must not be admitted here"
    assert quote not in allowed_after_lead, (
        "the string must NOT be closable until the character completes"
    )

    gen.append(cont)
    out = c(gen, scores)
    allowed_after_char = _allowed_from_masked(out)
    assert quote in allowed_after_char, "once the character completes, the string can close"


# ---------------------------------------------------------------------------------
# Complete-object / EOS handling
# ---------------------------------------------------------------------------------

def test_only_eos_allowed_after_a_complete_object_and_eos_forbidden_mid_object() -> None:
    vocab = make_vocabulary()
    pat = pydantic_to_regex(Simple)
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("x")
    gen = list(prompt_ids)
    out = c(gen, scores)
    mid_allowed = _allowed_from_masked(out)
    assert EOS not in mid_allowed, "EOS must not be allowed before any object has started"

    for ch in '{"s": "a':  # string content started, not yet closed
        gen.append(canonical_encode(ch)[0])
        out = c(gen, scores)
    mid_allowed = _allowed_from_masked(out)
    assert EOS not in mid_allowed, "EOS must not be allowed mid-object"

    gen.append(tid(b'"'))
    out = c(gen, scores)
    gen.append(tid(b"}"))
    out = c(gen, scores)
    final_allowed = _allowed_from_masked(out)
    assert final_allowed == {EOS}, "after a complete object, only EOS may be allowed"


# ---------------------------------------------------------------------------------
# Pathological patterns: bounded even at the library's own defaults (no fuel override)
# ---------------------------------------------------------------------------------

PATHOLOGICAL_PATTERNS = {
    "wide_any": (r"[\s\S]{0,5000}", "x" * 40),
    "huge_rep": (r"[0-9]{0,100000}", "1" * 40),
    "nested": (r"(a{1,100}){1,100}", "a" * 40),
    "deep_alt_2000way": (
        "(" + "|".join(f"opt{i}" for i in range(2000)) + ")", "opt1999",
    ),
}


@pytest.mark.parametrize(
    "label,case", PATHOLOGICAL_PATTERNS.items(), ids=list(PATHOLOGICAL_PATTERNS)
)
def test_pathological_patterns_build_and_step_bounded_at_library_defaults(label, case) -> None:
    """The pattern class `RegexLogitsProcessor`'s character-level NFA-to-DFA
    construction could blow up on. `llguidance` is lazy and builds no DFA, so all
    four are cheap even with NO `initial_lexer_fuel` override at all (the library's
    own default, ~1,000,000) -- direct evidence for the module docstring's claim that
    this project's 10,000 bound has nothing to usefully refuse in this class. Uses
    the raw `llguidance` API (not `build_constraint`, which always applies this
    project's own bound) specifically to measure the *library's* laziness, not this
    module's choice."""
    pattern, drive_text = case
    tok = llguidance.LLTokenizer(llguidance.TokenizerWrapper(_GTokenizerAdapter(make_vocabulary())))
    t0 = time.perf_counter()
    matcher = llguidance.LLMatcher(tok, llguidance.LLMatcher.grammar_from_regex(pattern))
    build_s = time.perf_counter() - t0
    assert not matcher.is_error(), f"{label}: unexpected construction error: {matcher.get_error()}"
    assert build_s < 2.0, f"{label}: build took {build_s:.3f}s"

    worst_step = 0.0
    for token_id in canonical_encode(drive_text):
        t0 = time.perf_counter()
        matcher.compute_bitmask()
        worst_step = max(worst_step, time.perf_counter() - t0)
        if not matcher.consume_token(token_id):
            break
    assert worst_step < 2.0, f"{label}: worst step took {worst_step:.3f}s"


def test_initial_lexer_fuel_refuses_a_13000_member_pydantic_literal() -> None:
    """This project's own bound (`INITIAL_LEXER_FUEL=100_000`, raised from 10_000 on
    2026-09-15 -- see `constraint.py`'s module docstring) DOES refuse a grammar built
    the expensive way: a 13,000-member `Literal` compiled through `pydantic_to_regex`.
    Re-derived against the new budget on the real 151,936-token vocabulary (identical on
    this module's synthetic one): a 12,000-member `Literal` costs 99,667 fuel and still
    admits; a 12,500-member one costs 103,817 and is refused. 13,000 members (107,967
    fuel) is comfortably past that boundary, so this stays a clean refusal rather than
    a knife-edge one as the exact boundary drifts with any future re-measurement. A
    2,000-member `Literal` (the old bound's refusal case) now easily admits -- it needs
    on the order of 8.3-8.5 fuel/member, ~17,000 total, well under 100,000. Also
    `ConstraintUnavailable` (Change 2): a fuel refusal at construction is a property of
    the pattern, so `ProgramAsWeightsBackend.infer()` catches this subclass to degrade
    that grammar per-schema rather than raising on every call."""
    vals = tuple(f"opt{i}" for i in range(13000))
    model = create_model("L13000", x=(Literal[vals], ...))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pat = pydantic_to_regex(model)
    vocab = make_vocabulary()
    with pytest.raises(ConstraintUnavailable, match="llguidance grammar construction failed"):
        build_constraint(pat, vocab)


def test_bounded_int_models_at_4_and_5_fields_construct_at_the_new_budget() -> None:
    """The concrete money-leak instance Phase 4 found and this track's Change 1 fixes:
    a model with four (or five) `Field(ge=0, le=255)` int fields -- an ordinary schema,
    not a pathological one -- used to be REFUSED by `INITIAL_LEXER_FUEL=10_000` (4
    fields cost 10,334 fuel, just past the old bound) on every `infer()` call. At the
    new `INITIAL_LEXER_FUEL=100_000`, both construct: 4 fields need 10,334 fuel (9.68x
    headroom) and 5 fields need 12,908 (7.75x headroom), both measured on the real
    151,936-token vocabulary and reproduced identically here on the synthetic one."""
    from pydantic import Field

    vocab = make_vocabulary()
    for n_fields in (4, 5):
        model = create_model(
            f"Bounded{n_fields}",
            **{f"f{i}": (int, Field(ge=0, le=255)) for i in range(n_fields)},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pat = pydantic_to_regex(model)
        constraint = build_constraint(pat, vocab)  # must not raise
        assert constraint is not None


def test_initial_lexer_fuel_admits_every_spine_case_pattern() -> None:
    """The same bound that refuses the 2,000-member Literal above admits all 38
    `SPINE_CASES` patterns -- INITIAL_LEXER_FUEL is not simply "too small"."""
    failures = []
    for name, model, _instances in SPINE_CASES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pat = pydantic_to_regex(model)
            except PAWSchemaError:
                continue  # a compiler refusal is a legitimate outcome, not this test's concern
        try:
            build_constraint(pat, make_vocabulary())
        except PAWSchemaError as exc:
            failures.append((name, str(exc)[:200]))
    assert not failures, (
        f"spine cases refused at INITIAL_LEXER_FUEL={INITIAL_LEXER_FUEL}: {failures}"
    )


# ---------------------------------------------------------------------------------
# Known-good sample: every spine case's next token is allowed by the bitmask
# ---------------------------------------------------------------------------------

_SKIPPED_SPINE_CASES: List[str] = []


@pytest.mark.parametrize("name,model,instances", SPINE_CASES, ids=SPINE_IDS)
def test_known_good_sample_next_token_always_allowed_by_bitmask(name, model, instances) -> None:
    """For every spine case, a JSON instance's `model_dump_json()` output -- encoded
    through the (synthetic, byte-level) test vocabulary -- has its next token
    allowed by the bitmask at *every* step. Mask membership, not `consume_token()`
    acceptance: the mask can only be a subset of what `consume_token()` would accept
    (module docstring's residual paragraph), so this is the stronger of the two
    properties and the one Phase T requires."""
    instance = instances[0]
    text = instance.model_dump_json()
    ids = canonical_encode(text)
    if not ids:
        _SKIPPED_SPINE_CASES.append(name)
        pytest.skip(f"{name}: encodes to zero tokens")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pat = pydantic_to_regex(model)
        except PAWSchemaError as exc:
            _SKIPPED_SPINE_CASES.append(name)
            pytest.skip(f"{name}: pydantic_to_regex refused ({exc})")

    c = build_constraint(pat, make_vocabulary())
    scores = np.zeros(N_VOCAB, dtype=np.float32)
    prompt_ids = canonical_encode("unrelated model input")
    gen = list(prompt_ids)
    for step, next_id in enumerate(ids):
        out = c(gen, scores)
        allowed = _allowed_from_masked(out)
        assert next_id in allowed, (
            f"{name}: step {step} token {next_id} ({TOKENS[next_id]!r}) not in the "
            f"bitmask (text so far: {text[: step + 10]!r})"
        )
        gen.append(next_id)


# ---------------------------------------------------------------------------------
# Corpus walk: a string-content state is actually visited, where one exists (Phase T)
# ---------------------------------------------------------------------------------

_STRING_CONTENT_WIDE_THRESHOLD = 40  # allowed-id count (excluding EOS) counted as "wide"


def _pattern_has_a_string_content_region(pattern: str) -> bool:
    """A schema has a free string-content region precisely when its compiled DFA has a
    live state whose transition map uses `interegular`'s `anything_else` catch-all
    symbol -- the shape an unconstrained JSON string body (`[^"\\\\...]`) compiles to.
    A closed `Literal`/enum/fixed-`Field(pattern=...)` field never produces this
    symbol: every transition out of its states names a specific character, never
    "anything else". Reuses `tests/test_schema_spine.py`'s own `interegular`-based
    machinery (that module already imports `interegular` and `anything_else` for
    exactly this kind of DFA inspection) rather than hand-classifying each of the 38
    spine cases.

    `anything_else` is not itself a transition-map key: it is a sentinel passed
    *into* `fsm.alphabet` to look up which concrete symbol id that FSM uses for its
    catch-all bucket (`tests/test_schema_spine.py`'s `accepted_strings` does the same
    lookup) -- `anything_else in transitions` would always be `False` and silently
    detect nothing.
    """
    fsm = interegular.parse_pattern(pattern).to_fsm()
    catch_all_symbol = fsm.alphabet[anything_else]
    for state, transitions in fsm.map.items():
        if fsm.islive(state) and catch_all_symbol in transitions:
            return True
    return False


@pytest.mark.parametrize("name,model,instances", SPINE_CASES, ids=SPINE_IDS)
def test_corpus_walk_reaches_a_string_content_state_where_one_exists(name, model, instances) -> None:
    """For every `SPINE_CASES` regex, walk the real matcher along a generated valid
    instance's tokens (the known-good-sample machinery above) and, where the schema's
    compiled DFA has a free string-content region
    (`_pattern_has_a_string_content_region`), assert the walk actually visits a state
    admitting a WIDE set of next tokens there -- not merely the one character the
    sample happens to emit next.

    This is the region Motivation's spike never entered: `Triage`'s three closed
    `Literal` fields and no free string field meant its FSM walk never needed to visit
    a string-content state at all, which is exactly why the byte-vs-continuation-byte
    hole (Motivation) went unmeasured by the published 15/15 result. Every spine case
    with a genuinely open string body is walked here and required to actually reach
    one; a case with no such region (a closed `Literal`, a fixed `Field(pattern=...)`,
    a plain scalar) is walked too but makes no claim, since the region does not exist
    for it to reach.

    Walks EVERY instance the case provides, not just `instances[0]`: `nullable`'s
    `Optional[str]` field has a free string-content region in its DFA, but its first
    instance is `Nullable(maybe=None)`, whose JSON never enters it -- only
    `Nullable(maybe="x")`, the case's second instance, actually does. "Reached" is
    true for the case as a whole if ANY of its instances' walks visits a wide state.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pat = pydantic_to_regex(model)
        except PAWSchemaError as exc:
            pytest.skip(f"{name}: pydantic_to_regex refused ({exc})")

    if not _pattern_has_a_string_content_region(pat):
        pytest.skip(f"{name}: schema has no free string-content region to reach")

    reached = False
    walked_any = False
    for instance in instances:
        ids = canonical_encode(instance.model_dump_json())
        if not ids:
            continue
        walked_any = True
        c = build_constraint(pat, make_vocabulary())
        scores = np.zeros(N_VOCAB, dtype=np.float32)
        prompt_ids = canonical_encode("unrelated model input")
        gen = list(prompt_ids)
        for next_id in ids:
            out = c(gen, scores)
            allowed = _allowed_from_masked(out)
            if len(allowed - {EOS}) >= _STRING_CONTENT_WIDE_THRESHOLD:
                reached = True
            gen.append(next_id)

    if not walked_any:
        pytest.skip(f"{name}: every instance encodes to zero tokens")
    assert reached, (
        f"{name}: schema has a free string-content region ({pat!r}) but the corpus "
        "walk never visited a state admitting a wide set of next tokens there, "
        f"across any of its {len(instances)} instance(s)"
    )


# ---------------------------------------------------------------------------------
# Lazy-import policy (import paw_kit must work with neither llguidance nor numpy)
# ---------------------------------------------------------------------------------

def test_module_has_no_top_level_llguidance_or_numpy_import() -> None:
    import ast
    import inspect

    import paw_kit.schema.constraint as mod

    tree = ast.parse(inspect.getsource(mod))
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "llguidance" not in top_level_imports
    assert "numpy" not in top_level_imports
