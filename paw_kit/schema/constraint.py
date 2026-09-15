"""Byte-level grammar-constrained decoding on top of `llguidance`.

This is the successor to `logits_processor.py`'s `RegexLogitsProcessor` (removed in
Phase 2 of `constrained-decoding-real-backend`) and inherits that module's role as the
one public place this project states its masking bounds: PAW-SCHEMA-03 bounded
`RegexLogitsProcessor`'s **character**-level FSM construction with a 50,000-character
pattern-length cap, a 10,000-state cap and a 3-second timeout. Those numbers do not
apply here and are not replaced like-for-like.

`llguidance` is lazy: it builds no DFA up front, so the bound that matters is not "how
big can the compiled automaton get" but "how much work can a single grammar
construction do before it is forced to answer". That is `LLParserLimits.initial_lexer_fuel`,
set explicitly below to **100,000**; the other six `LLParserLimits` fields are left at
their library defaults, so this project's one deliberate bound lives in this project's
own source rather than an upstream default nobody chose.

**Raised from 10,000 to 100,000 on 2026-09-15**, after Phase 4 found the 10,000 value
refusing an ordinary schema. The budget bounds *grammar construction size*, not
per-token work -- `_Constraint.__call__`'s per-step masking cost does not depend on
`initial_lexer_fuel` at all, only construction does, and construction happens once per
`infer()` call (`build_constraint` is never cached across calls). Integer ranges are the
expensive case: `pydantic_to_regex` enumerates a bounded int range member by member
(`grammar.py`'s `_MAX_ENUMERATED_INT_RANGE = 256`), and each enumerated member costs
about 10 fuel, so a *single* `Field(ge=0, le=255)` int field already costs 2,612 --
measured on the real 151,936-token GGUF vocabulary. Four such fields -- an ordinary
schema -- cost 10,334 and were refused by the old 10,000 bound; that boundary case is
exactly what forced this change. Re-derived against the same real vocabulary (the
bounded-int, `Literal` and bare-alternation figures below reproduced identically on
this module's synthetic test vocabulary -- fuel for those pattern shapes is a property
of the pattern's own structure, not the tokenizer; that does not hold for every pattern
shape, e.g. the pathological `(a{1,100}){1,100}` case below needs 9 fuel on the
synthetic vocabulary and 21 on the real one, so it is stated per-vocabulary where it was
actually measured):

- Bounded-int fields (`Field(ge=0, le=255)`), minimum fuel to construct: 1 field 2,612;
  2 fields 5,186; 3 fields 7,760; 4 fields 10,334 (refused at the *old* 10,000 bound,
  admitted at 100,000 with 9.68x headroom); 5 fields 12,908. The marginal cost per extra
  field is a flat 2,574, so the count of such fields the new budget admits is exactly
  computable: 38 fields cost 97,850 and construct; 39 cost 100,424 and are refused.
  **100,000 admits ~38 such fields**, not the 3 the old bound admitted.
- A `Literal` compiled through `pydantic_to_regex` costs roughly 8.3-8.5 fuel per member
  asymptotically; re-derived at the new budget's scale, a 12,000-member `Literal` costs
  99,667 (admitted) and a 12,500-member one costs 103,817 (refused) -- **100,000 admits
  a `Literal` of roughly 12,000 members**, up from ~1,175 at the old bound.
- A bare alternation (not routed through a `Literal`) costs roughly 9.0-9.3 per member,
  slightly more than a `Literal` of the same arity. Re-derived at the new budget's
  scale: an 11,000-way alternation costs 99,113 (admitted), an 11,500-way one costs
  103,618 (refused) -- **the smallest bare alternation refused at 100,000 is between
  11,000 and 11,500 members**, comfortably above what any realistic schema's own
  alternations reach.
- The `Contact` schema used throughout this track's evidence needs 763 to construct --
  unchanged by this module's chosen ceiling, since a schema's own minimum fuel
  requirement depends only on its pattern, never on the ceiling set above it. Measured
  matcher-construction time (build + initial mask, real vocabulary, 5-run minimum) for
  `Contact` is **1.39 ms at `initial_lexer_fuel=10,000`** and **1.40 ms at
  `initial_lexer_fuel=100,000`** -- statistically indistinguishable, because raising the
  ceiling costs nothing for a schema that never approaches the old one. This is the
  direct evidence that the ceiling bounds worst-case construction size, not per-call
  cost: per-call cost tracks the *pattern's own* fuel requirement, which for an ordinary
  schema is far below either ceiling. For a pattern that actually needs the extra
  headroom, construction time scales with the fuel it consumes, not with the ceiling:
  measured on bare alternations sized to need roughly the fuel shown, construction time
  is ~3.0 ms at ~1,000-way, ~11.3 ms at ~5,000-way, ~22.0-23.1 ms at ~10,000-way, ~127-135
  ms at ~50,000-way, and ~262-271 ms at ~100,000-way -- growth that tracks the fuel spent,
  this is the per-call cost of the matcher, since the matcher is built fresh per call
  (`build_constraint`'s docstring).
- All 38 cases in `tests/test_schema_spine.py`'s `SPINE_CASES` still construct at fuel
  100,000; the worst (`WithCollections`) still needs 1,266 -- now 78.99x headroom under
  the chosen bound (was 7.90x at 10,000).

100,000 is still a tenfold tightening of upstream's own `LLParserLimits` default of
1,000,000 -- this project's bound remains materially tighter than "whatever the library
ships with", which was the point of setting it explicitly at all.

**This does not cover the same pattern class PAW-SCHEMA-03 did**, and that is not an
oversight: `llguidance` is lazy and builds no DFA, so the patterns that made
`RegexLogitsProcessor`'s NFA-to-DFA powerset construction blow up --
`[0-9]{0,100000}` (fuel 16 here), `(a{1,100}){1,100}` (fuel 21 here) -- are *cheaper*
than an ordinary schema under this engine, not more expensive, and neither figure moved
with this change (raising `initial_lexer_fuel` cannot make a cheap pattern more
expensive). Measured directly, re-derived at the new budget: all four of the patterns
PAW-SCHEMA-03's own evidence used as pathological cases still need at most 18,023 fuel
to construct (the hand-written 2,000-way alternation) and none of the four costs more
than that, so there remains nothing here for `initial_lexer_fuel` to usefully refuse in
that class at either 10,000 or 100,000. The security-relevant statement is therefore
unchanged by this raise: the old exponential-blowup class is closed by this engine being
lazy, not by this module's fuel bound; `initial_lexer_fuel=100_000` exists to keep an
oversized *schema* (an enormous `Literal`/alternation, now needing on the order of
12,000+ members rather than ~1,175) from doing unbounded construction work, which is a
different failure mode than the one PAW-SCHEMA-03 was written against. See
`conductor/reviews/security-audit-report.md`'s dated PAW-SCHEMA-03 replacement note for
the full disposition.

Two safety properties this module exists to prove, both from the track's "No silent
masking failure" invariant:

1. **The encoder is required and verified by round-trip, not trusted.** `llguidance`
   calls the vocabulary's encoder once with `b'test'` at tokenizer construction and
   thereafter lazily during masking, with the forced byte prefix -- never on the
   grammar's own literals up front. An encoder that raises therefore leaves
   `LLMatcher.is_error()` `False` immediately after construction and `True` only after
   the *first* `compute_bitmask()` call. `build_constraint` computes that first mask
   before reading `is_error()` for exactly this reason: reading it first silently loses
   this line of defence. Before that point is even reached, `Vocabulary.__init__`
   requires a non-`None` encode callable and verifies it against three fixed probes --
   `b'{"k": "v"}'`, the multi-byte character `'é'` (which has its own token in any
   sane byte-fallback vocabulary), and `'ϩ'` (which this module's own construction
   never gives a whole-character token, so a real vocabulary can only spell it through
   single-byte fallback tokens -- an emoji or CJK probe would not exercise that path,
   because those almost always have a dedicated token). Each probe's encoded ids must,
   looked up in the vocabulary's own token table and concatenated, reproduce the
   probe's UTF-8 bytes exactly; any other outcome (an exception, an empty id list, a
   wrong id, a truncated one) raises `PAWSchemaError` at construction, before a single
   token has been generated against it. An encoder that is merely *correct on these
   probes and wrong elsewhere* is a stated residual (non-canonical splitting, junk or
   truncation on other strings): its measured consequence is an over-restricted mask,
   never an admitted illegal token, because the mask can only ever be a subset of what
   the true grammar state allows.
2. **A masking failure is loud, not silent -- and it degrades at the right grain.**
   Two distinct failure shapes, two distinct exceptions. A grammar that cannot even be
   started -- `is_error()` set right after the initial mask (fuel exceeded, or a
   raising encoder surfacing as described in point 1), or an initial allowed token set
   that is empty or EOS-only -- is a property of the *pattern*, not of any one call, so
   `build_constraint` raises `ConstraintUnavailable` (a `PAWSchemaError` subclass) for
   these three cases: `ProgramAsWeightsBackend.infer()` catches it specifically and
   degrades *that grammar* to unconstrained decoding with one warning, rather than
   re-discovering an unfixable refusal on every subsequent call for the same schema
   (the Track-D money-leak class). A failure discovered mid-generation --
   `consume_token()` returning `False`, or `is_error()` becoming set after a later step
   -- is not a property of the pattern alone (partial output may already exist, and a
   different token sequence might not have failed), so it stays plain `PAWSchemaError`,
   propagates through `ProgramAsWeightsBackend.infer()` unwrapped, and reaches
   `paw.load`'s fail-open fallback as itself, per the track's safety invariants.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Union

from paw_kit.schema.exceptions import PAWSchemaError


class ConstraintUnavailable(PAWSchemaError):
    """A `PAWSchemaError` raised for exactly one reason: `build_constraint` could not
    even START this grammar -- fuel/grammar refused at construction (`is_error()` set
    right after the initial mask, which is also how a raising encoder surfaces per the
    module docstring's point 1), or the initial allowed token set is empty or EOS-only.
    These three are properties of the *pattern*, not of any particular call: the same
    grammar refused this way on one call will be refused the same way on every call for
    as long as the schema is unchanged, which is what makes this a per-SCHEMA condition
    rather than a per-call one.

    This is deliberately a DIFFERENT exception from the plain `PAWSchemaError` a
    mid-generation failure raises (`consume_token()` returning `False`, or `is_error()`
    becoming set after a later step) -- see `_Constraint.__call__`, which never raises
    this subclass. `ProgramAsWeightsBackend.infer()` catches this subclass alone to
    degrade *that grammar* to unconstrained decoding with one warning (never a raise on
    every call for a schema that can never construct); it does NOT catch plain
    `PAWSchemaError`, which must keep propagating and failing open through `paw.load`'s
    fallback, because a mid-generation failure may have already produced partial output
    that this call's local path cannot be trusted to have produced correctly.
    """

# The bound this module's docstring documents and justifies (see above). Deliberately
# the only `LLParserLimits` field set: the other six are left at their library
# defaults, so this project's one chosen bound is visible in this project's own source
# rather than mixed in with upstream defaults nobody here chose.
#
# Raised from 10,000 to 100,000 on 2026-09-15 (see module docstring's re-derivation):
# 10,000 admitted at most three `Field(ge=0, le=255)` int fields and refused an
# ordinary four-field schema at construction, on every call, which is the money-leak
# class this track's Change 2 also addresses. 100,000 is still a tenfold tightening of
# upstream's own default of 1,000,000.
INITIAL_LEXER_FUEL = 100_000

# The three fixed round-trip probes (see module docstring, point 1). Kept as module
# constants rather than inlined so a caller inspecting a `PAWSchemaError` message, or a
# future maintainer re-deriving the bound above, has one place to look for exactly what
# was checked. `_PROBE_2` and `_PROBE_3` are given as `str`, not pre-encoded `bytes`,
# because `Vocabulary`'s required encode callable is typed `bytes | str -> list[int]`
# and this exercises the `str` half of that contract; `_PROBE_1` exercises the `bytes`
# half.
_PROBE_1: bytes = b'{"k": "v"}'
_PROBE_2: str = "é"  # 'é' -- expected to have its own token in a real vocabulary.
_PROBE_3: str = "ϩ"  # 'ϩ' -- NOT an emoji/CJK probe on purpose: see module docstring.
PROBES: "tuple[Union[bytes, str], ...]" = (_PROBE_1, _PROBE_2, _PROBE_3)


def _probe_utf8_bytes(probe: Union[bytes, str]) -> bytes:
    return probe if isinstance(probe, bytes) else probe.encode("utf-8")


class Vocabulary:
    """Value object the schema layer consumes and a backend supplies.

    Carries every token's raw bytes (byte-fallback tokens included, unfiltered -- see
    `decisions.md` §2 / owner decision 2, "Restricted string alphabet -- rescinded"),
    the EOS and special-token ids, and a **required** encode callable
    (`bytes | str -> list[int]`) bound to the same tokenizer that produced `tokens`.

    Deliberately holds no reference to `llama_cpp` or any other backend-specific type
    (`llama_cpp` stays out of `paw_kit.schema`, per this track's implementation
    overview) -- `tokens` is any `Sequence[bytes]` indexed by token id, and `encode` is
    any callable with the right shape. The encoder must be a *live* callable bound to
    the model that produced `tokens`, because `llguidance` consults it during masking
    (module docstring, point 1), not just at construction.

    Construction verifies `encode` by round-trip against `PROBES` and raises
    `PAWSchemaError` on any failure -- absence, an exception, an empty id list, a
    wrong id, or a mismatched round-trip. This is the only defence against an encoder
    that is subtly wrong (see module docstring); it is not optional and cannot be
    skipped by a caller.
    """

    __slots__ = ("tokens", "eos_token_id", "special_token_ids", "encode", "_llg_tokenizer")

    def __init__(
        self,
        tokens: Sequence[bytes],
        eos_token_id: int,
        special_token_ids: Sequence[int],
        encode: Optional[Callable[[Union[bytes, str]], List[int]]],
    ) -> None:
        if encode is None:
            raise PAWSchemaError(
                "paw_kit.schema.constraint.Vocabulary requires an encode callable "
                "(bytes|str -> list[int]) bound to the tokenizer that produced "
                "`tokens`; none was given."
            )
        self.tokens = tokens
        self.eos_token_id = eos_token_id
        self.special_token_ids = tuple(special_token_ids)
        self.encode = encode
        # Built lazily on first use by `llguidance_tokenizer()` -- NOT here, because
        # `Vocabulary` must stay constructible (and its encoder verifiable) with
        # `llguidance` absent, which is the engine-absent test configuration.
        self._llg_tokenizer: Any = None
        self._verify_encoder()

    def llguidance_tokenizer(self) -> Any:
        """The `llguidance.LLTokenizer` for this vocabulary, built once and reused.

        This is the ONE piece of llguidance state that is cached rather than rebuilt per
        call, and the distinction is load-bearing. An `LLMatcher` carries the parse in
        progress and **dies on error**, which is why `build_constraint` returns a fresh,
        single-use one every call. An `LLTokenizer` carries no parse state at all: it is
        a pure function of `tokens` + `eos_token_id` + `special_token_ids` + `encode`,
        all of which are immutable for this object's lifetime.

        Building it walks the whole token table, and on the real 151,936-token GGUF
        vocabulary that measured **249.13 ms** -- against **1.35 ms** for the matcher and
        its initial mask. Paid once per `infer()` call, as this was until Phase 4's
        measurement caught it, that alone put end-to-end masking overhead at **13.118 ms
        per generated token** against `roadmap.md`'s `<2 ms` budget. Paid once per
        vocabulary it is a per-model cost, which is where the track's own cost model
        always put it ("0.553 s per model: 0.293 s detokenize + 0.260 s tokenizer").

        Lifetime: `ProgramAsWeightsBackend` caches the `Vocabulary` beside the function
        it was built from and evicts the two together (A-11), so this cache is dropped
        with the model whose tokenizer it wraps and can never outlive it. Two threads
        racing here build two tokenizers and one wins; both are valid, so the race costs
        work and never correctness, which is why there is no lock on a path `infer()`
        deliberately runs without holding one.
        """
        if self._llg_tokenizer is None:
            import llguidance as lg

            self._llg_tokenizer = lg.LLTokenizer(lg.TokenizerWrapper(_GTokenizerAdapter(self)))
        return self._llg_tokenizer

    def _verify_encoder(self) -> None:
        for probe in PROBES:
            expected = _probe_utf8_bytes(probe)
            try:
                ids = self.encode(probe)
            except Exception as exc:  # noqa: BLE001 -- see module docstring, point 1
                raise PAWSchemaError(
                    "paw_kit.schema.constraint.Vocabulary: encoder round-trip check "
                    f"failed on probe {probe!r}: encode() raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not ids:
                raise PAWSchemaError(
                    "paw_kit.schema.constraint.Vocabulary: encoder round-trip check "
                    f"failed on probe {probe!r}: encode() returned an empty id list."
                )
            try:
                got = b"".join(self.tokens[int(tid)] for tid in ids)
            except Exception as exc:  # noqa: BLE001 -- an id table lookup failure
                raise PAWSchemaError(
                    "paw_kit.schema.constraint.Vocabulary: encoder round-trip check "
                    f"failed on probe {probe!r}: looking up ids {ids!r} in the token "
                    f"table raised {type(exc).__name__}: {exc}"
                ) from exc
            if got != expected:
                raise PAWSchemaError(
                    "paw_kit.schema.constraint.Vocabulary: encoder round-trip check "
                    f"failed on probe {probe!r}: ids {ids!r} decode to {got!r}, "
                    f"expected {expected!r}."
                )


class _GTokenizerAdapter:
    """Shapes a `Vocabulary` into the plain object `llguidance.TokenizerWrapper` wants:
    `eos_token_id`, `bos_token_id`, `tokens`, `special_token_ids`, and `__call__` as the
    encoder. `llguidance` never sees `Vocabulary` directly, only this adapter."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        self.eos_token_id = vocabulary.eos_token_id
        self.bos_token_id: Optional[int] = None
        self.tokens = vocabulary.tokens
        self.special_token_ids = list(vocabulary.special_token_ids)
        self._encode = vocabulary.encode

    def __call__(self, s: Union[bytes, str]) -> List[int]:
        return self._encode(s)


def _mask_allowed_ids(raw_mask: Any, n_vocab: int) -> "set[int]":
    """Unpack an `LLMatcher.compute_bitmask()` buffer into the set of allowed ids
    below `n_vocab`. Lazy numpy import -- see module-level import policy."""
    import numpy as np

    arr = np.frombuffer(raw_mask, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")[:n_vocab]
    return set(np.nonzero(bits)[0].tolist())


def _count_masked(raw_mask: Any, n_vocab: int) -> int:
    """How many of the first `n_vocab` bitmask positions are NOT allowed -- the
    "masked logits per step" count Phase T asserts is non-zero at every step."""
    import numpy as np

    arr = np.frombuffer(raw_mask, dtype=np.uint8)
    bits = np.unpackbits(arr, bitorder="little")[:n_vocab]
    return int(n_vocab - int(bits.sum()))


class _Constraint:
    """A fresh, single-use llama-cpp logits-processor callable: `(input_ids, scores)
    -> scores`. Never cache or reuse an instance across calls -- an `LLMatcher` dies on
    error, and `build_constraint` is cheap enough (1.35 ms measured on the real
    151,936-token vocabulary, once its tokenizer is cached per `Vocabulary`) that a fresh one
    per call is the only shape that keeps a stale, already-dead matcher from silently
    doing nothing on a later call.

    Money-route (ii) from the track's safety invariants lives entirely in `__call__`:
    the processor's first invocation carries the whole context, prompt included (the
    SDK's own decode loop, not anything this class assumes), so `__call__` records
    `len(input_ids)` on its first invocation as the prompt length and feeds only ids
    *beyond* that offset to `consume_token()` on every later call. Getting this wrong
    in the "start at zero" direction would feed `<|im_start|>`-style prompt tokens to
    the matcher and fail open on essentially every call.
    """

    def __init__(self, matcher: Any, vocabulary: Vocabulary) -> None:
        self._matcher = matcher
        self._n_vocab = len(vocabulary.tokens)
        self._prompt_len: Optional[int] = None
        self._next_index: int = 0
        self.invocations = 0
        self.masked_per_step: "List[int]" = []

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        import numpy as np
        from llguidance.numpy import apply_token_bitmask_inplace

        self.invocations += 1
        if self._prompt_len is None:
            self._prompt_len = len(input_ids)
            self._next_index = self._prompt_len

        while self._next_index < len(input_ids):
            token_id = int(input_ids[self._next_index])
            ok = self._matcher.consume_token(token_id)
            self._next_index += 1
            if not ok or self._matcher.is_error():
                raise PAWSchemaError(
                    f"llguidance grammar constraint rejected token {token_id} "
                    f"(consume_token()={ok!r}): {self._matcher.get_error()}"
                )

        raw_mask = self._matcher.compute_bitmask()
        # A masking failure is not only a consume_token() rejection: the encoder is
        # also consulted lazily *during* mask computation itself (module docstring,
        # point 1 / K-2) -- e.g. resolving a grammar literal's forced-byte spelling
        # that no consume_token() call has driven yet. is_error() must therefore be
        # checked again here, separately from the loop above, or a failure that only
        # surfaces at mask-computation time would be silently missed.
        if self._matcher.is_error():
            raise PAWSchemaError(
                "llguidance grammar constraint entered an error state while "
                f"computing the mask: {self._matcher.get_error()}"
            )
        self.masked_per_step.append(_count_masked(raw_mask, self._n_vocab))

        # Mask applied to a CONTIGUOUS COPY of the scores -- never the caller's own
        # buffer in place, and the copy (not `scores` itself) is what is returned.
        out = np.array(scores, dtype=np.float32, copy=True)
        mask_arr = np.frombuffer(raw_mask, dtype=np.int32).copy()
        apply_token_bitmask_inplace(out, mask_arr)
        return out


def build_constraint(pattern: str, vocabulary: Vocabulary) -> _Constraint:
    """Build a fresh, single-use llama-cpp logits-processor callable that constrains
    decoding to `pattern` (a regex, e.g. from `pydantic_to_regex`) using `vocabulary`.

    Never cache or reuse the returned callable across calls -- see `_Constraint`'s
    docstring. `llguidance` and `numpy` are imported lazily here (and nowhere at this
    module's top level), so `import paw_kit` and `import paw_kit.schema` work with
    neither installed; their absence raises `PAWSchemaError` naming the `paw` extra
    that provides them.

    At construction: computes the initial bitmask **before** reading `is_error()` (see
    module docstring, point 1 -- reading `is_error()` first would silently lose the one
    check that catches a raising encoder before any token has been generated), then
    raises `ConstraintUnavailable` if `is_error()` is set, and again if the initial
    allowed token set is empty or contains only EOS -- a grammar with no legal first
    token is unusable and, per round 3's measurement, this second check exists
    specifically to catch it independent of the raising-encoder case above.

    All three of these are `ConstraintUnavailable`, not plain `PAWSchemaError`: they are
    properties of `pattern` (and, for the raising-encoder case, of the vocabulary), so a
    grammar refused this way is refused identically on every future call -- the caller
    (`ProgramAsWeightsBackend.infer()`) catches this subclass specifically to degrade
    *that grammar* to unconstrained decoding rather than propagating a refusal that
    would otherwise recur, unwarned differently, on every single call for this schema
    (the Track-D money-leak class this exists to close). A failure discovered
    mid-generation -- `_Constraint.__call__`'s `consume_token()` returning `False`, or
    `is_error()` becoming set after a later step -- is NOT a property of the pattern
    alone (partial output already exists) and stays plain `PAWSchemaError`.
    """
    try:
        import llguidance as lg
    except ImportError as exc:
        raise PAWSchemaError(
            "paw_kit.schema.constraint.build_constraint requires the 'llguidance' "
            "package. Install the 'paw' extra (pip install 'paw-kit[paw]') to enable "
            f"grammar-constrained decoding: {exc}"
        ) from exc

    # Per VOCABULARY (built once, reused), not per call -- see
    # `Vocabulary.llguidance_tokenizer` for why this one object is cached and the
    # matcher below is not.
    tokenizer = vocabulary.llguidance_tokenizer()
    grammar = lg.LLMatcher.grammar_from_regex(pattern)
    limits = lg.LLParserLimits(initial_lexer_fuel=INITIAL_LEXER_FUEL)
    matcher = lg.LLMatcher(tokenizer, grammar, limits=limits)

    # Compute the initial mask BEFORE reading is_error() -- see docstring above and
    # module docstring point 1 (K-2). Do not reorder these two statements.
    initial_mask = matcher.compute_bitmask()
    if matcher.is_error():
        raise ConstraintUnavailable(
            f"llguidance grammar construction failed for pattern {pattern!r}: "
            f"{matcher.get_error()}"
        )

    n_vocab = len(vocabulary.tokens)
    initial_allowed = _mask_allowed_ids(initial_mask, n_vocab)
    if not initial_allowed:
        raise ConstraintUnavailable(
            f"llguidance grammar for pattern {pattern!r} has an empty initial "
            "allowed token set -- nothing could ever be generated."
        )
    if initial_allowed == {vocabulary.eos_token_id}:
        raise ConstraintUnavailable(
            f"llguidance grammar for pattern {pattern!r} allows only EOS as the "
            "first token -- nothing could ever be generated."
        )

    return _Constraint(matcher, vocabulary)
