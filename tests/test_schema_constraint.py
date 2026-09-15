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

import numpy as np
import pytest

llguidance = pytest.importorskip("llguidance")

from pydantic import BaseModel, create_model
from typing import Literal

from paw_kit.schema.constraint import (
    INITIAL_LEXER_FUEL,
    PROBES,
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
    with pytest.raises(PAWSchemaError, match="llguidance grammar construction failed"):
        build_constraint(pat, vocab)


# ---------------------------------------------------------------------------------
# Construction-time EOS-only / empty allowed-set check
# ---------------------------------------------------------------------------------

def test_empty_pattern_refused_as_eos_only_at_construction() -> None:
    """A pattern matching only the empty string admits nothing but EOS as its first
    token -- refused at construction, independent of the encoder-ordering check
    above (round 3's J-3: this check catches a case the encoder check does not)."""
    vocab = make_vocabulary()
    with pytest.raises(PAWSchemaError, match="EOS as the first token"):
        build_constraint(r"", vocab)


def test_ordinary_pattern_constructs_without_eos_only_false_positive() -> None:
    vocab = make_vocabulary()
    build_constraint(r"x", vocab)  # must not raise


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
    c = build_constraint(pat, vocab)
    scores = np.zeros(N_VOCAB, dtype=np.float32)

    prompt_ids = canonical_encode("x")
    c(prompt_ids, scores)  # establishes the prompt offset
    bad_token = tid(b"z")  # illegal first token: the grammar demands '{'
    with pytest.raises(PAWSchemaError, match="rejected token"):
        c(prompt_ids + [bad_token], scores)

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


def test_initial_lexer_fuel_refuses_a_2000_member_pydantic_literal() -> None:
    """This project's own bound (`INITIAL_LEXER_FUEL=10_000`) DOES refuse a grammar
    built the expensive way: a 2,000-member `Literal` compiled through
    `pydantic_to_regex`, which costs materially more fuel per member than the bare
    hand-written alternation in `PATHOLOGICAL_PATTERNS` above (module docstring:
    ~8.3-8.5/member for a `Literal` vs ~9.0-9.3/member for a bare alternation --
    close per-member, but a `Literal` of 2,000 members still lands north of 10,000)."""
    vals = tuple(f"opt{i}" for i in range(2000))
    model = create_model("L2000", x=(Literal[vals], ...))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pat = pydantic_to_regex(model)
    vocab = make_vocabulary()
    with pytest.raises(PAWSchemaError, match="llguidance grammar construction failed"):
        build_constraint(pat, vocab)


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
