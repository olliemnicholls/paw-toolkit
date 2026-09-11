"""The differential spine for the schema compiler (Track E, Phase E1).

One property does most of the work of this track:

    every string that `re.fullmatch(pydantic_to_regex(M), s)` accepts must satisfy
    BOTH `json.loads(s)` AND `M.model_validate_json(s)`

Stated the other way round: the compiled grammar's language must be a *subset* of the
language pydantic will validate. That is the direction that makes
grammar-constrained decoding mean anything -- a decoder that can only walk the grammar
can then only emit output the caller's model accepts. The bug hunt's assessment was
that this single test would have caught nine of its fourteen schema findings; it is
built here first, and proven red at `main`, before any of them is fixed.

The test samples the grammar's language *directly*, by enumerating accepted strings out
of the same `interegular` DFA the decoder walks, rather than by guessing candidate
strings. The sample is a seeded random walk of the DFA from the initial state to an accepting
one, biased towards the characters that break a JSON string -- see `accepted_strings`
for why a uniform or breadth-first walk is not a usable detector here.

Three arms, because the property above is one-directional and therefore blind in two
ways:

* `test_spine_*` -- accepted => valid. Catches the grammar being too *wide* (S-2, S-3).
* `test_must_accept_*` -- a curated, enumerated list of values each finding documents as
  legal. Catches the grammar being too *narrow* (S-7). This cannot be a universal
  property: the grammar is deliberately narrower than pydantic, so there is no general
  rule to assert, only a list.
* `test_fsm_and_regex_agree_*` -- the exported regex (compiled by Python `re`, Unicode
  shorthands) and the decoding FSM (compiled by `interegular`, ASCII shorthands) must
  accept the same language. The bug hunt's "the DFA agrees with `re` on 4,004 strings"
  used an ASCII corpus and so could not see S-3b; this arm is non-ASCII on purpose.
"""

import datetime as dt
from decimal import Decimal
import enum
import json
import random
import re
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union
import uuid

import interegular
from interegular.fsm import anything_else
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
import pytest

from paw_kit import PAWSchemaError, pydantic_to_regex


# --- enumeration machinery ------------------------------------------------------------

# Characters a JSON string cannot carry unescaped, plus the two non-ASCII probes that
# sit on the `interegular`-ASCII / `re`-Unicode fault line (S-3b). The sampler below is
# biased towards these: a uniform walk of the DFA spends essentially all of its budget
# on ordinary letters and on `[ \t\n\r]*` whitespace permutations, and would need an
# impractical number of samples to stumble onto the one transition that breaks out of a
# JSON string. Biasing the walk is what makes this test a detector rather than a lottery.
HOSTILE_CHARS = frozenset(
    {'"', "\\", "\x7f", "\u00e9", "\u0663", "\u00a0"} | {chr(c) for c in range(0x20)}
)

# Substituted for interegular's `anything_else` catch-all symbol. A negated class
# collapses every character the pattern never mentions into one transition, so a DFA
# walk reports that transition as a sentinel rather than as any concrete character.
ANYTHING_ELSE_REPS = ('"', "\\", "\x01", "\x7f", "\u00e9", "\u0663", "\u00a0", "e")

_SAMPLES = 600
_HOSTILE_BIAS = 0.75
_STOP_AT_FINAL = 0.3
_MIN_WALK_BUDGET = 24
_FREE_SLACK = 16


def _symbol_sort_key(sym):
    """Deterministic ordering over a transition bucket, which may hold `anything_else`."""
    return (1, "") if sym is anything_else else (0, sym)


def accepted_strings(pattern, samples=_SAMPLES, seed=20260911):
    """Return distinct strings the DFA for `pattern` accepts, sampled by random walk.

    Walks the same `interegular` DFA the decoder walks, from the initial state to an
    accepting one, choosing among *live* transitions only -- so every returned string is
    accepted by construction, and the walk is the decoder's own move. The choice is
    biased towards `HOSTILE_CHARS` (see above) and `anything_else` positions are filled
    from `ANYTHING_ELSE_REPS`.

    Deterministic: seeded, and every set is ordered before it is sampled from.
    """
    rng = random.Random(seed)
    fsm = interegular.parse_pattern(pattern).to_fsm()
    catch_all = fsm.alphabet[anything_else]
    reps = tuple(r for r in ANYTHING_ELSE_REPS if fsm.alphabet[r] == catch_all)

    # symbol -> (all member chars, the hostile subset), ordered, computed once.
    buckets = {}
    for sym, members in fsm.alphabet.by_transition.items():
        ordered = sorted(members, key=_symbol_sort_key)
        concrete = []
        for m in ordered:
            concrete.extend(reps if m is anything_else else [m])
        if not concrete:
            continue
        hostile = [c for c in concrete if c in HOSTILE_CHARS]
        buckets[sym] = (concrete, hostile)

    # `FSM.islive` re-runs a reachability search on every call, so hoist it: compute the
    # live set once and precompute each state's usable transitions, ordered, once.
    live_states = {st for st in fsm.states if fsm.islive(st)}
    moves = {
        st: sorted((sym, nxt) for sym, nxt in trans.items()
                   if nxt in live_states and sym in buckets)
        for st, trans in fsm.map.items()
    }

    # Minimum number of further characters needed to reach an accepting state. A free
    # random walk of a grammar whose shortest accepted string is 50-odd characters long
    # (any model with several fields) almost never lands on an accepting state by
    # chance, so past a soft length budget the walk is restricted to moves that strictly
    # shorten that distance. Every returned sample therefore ends accepted, and lengths
    # still vary because the budget is only a floor on when steering starts.
    dist = _distance_to_final(fsm, moves)
    reachable = dist.get(fsm.initial)
    if reachable is None:
        return []
    # Wander freely for `_FREE_SLACK` characters beyond the shortest accepted string,
    # then steer. The hard limit leaves room for the steered tail to actually close.
    soft_limit = max(_MIN_WALK_BUDGET, reachable + _FREE_SLACK)
    hard_limit = soft_limit + reachable + _FREE_SLACK

    out = set()
    for _ in range(samples):
        state = fsm.initial
        chars = []
        while len(chars) < hard_limit:
            if state in fsm.finals and rng.random() < _STOP_AT_FINAL:
                break
            options = moves.get(state, ())
            if len(chars) >= soft_limit:
                closing = [o for o in options if dist.get(o[1], hard_limit) < dist[state]]
                options = closing or options
            if not options:
                break
            sym, nxt = options[rng.randrange(len(options))]
            concrete, hostile = buckets[sym]
            pool = hostile if (hostile and rng.random() < _HOSTILE_BIAS) else concrete
            chars.append(pool[rng.randrange(len(pool))])
            state = nxt
        if state in fsm.finals:
            out.add("".join(chars))
    return sorted(out)


def _distance_to_final(fsm, moves):
    """Map each state to the minimum number of transitions to an accepting state."""
    reverse = {}
    for st, options in moves.items():
        for _sym, nxt in options:
            reverse.setdefault(nxt, set()).add(st)
    dist = {st: 0 for st in fsm.finals if st in fsm.states}
    frontier = sorted(dist)
    step = 0
    while frontier:
        step += 1
        nxt_frontier = []
        for st in frontier:
            for prev in sorted(reverse.get(st, ())):
                if prev not in dist:
                    dist[prev] = step
                    nxt_frontier.append(prev)
        frontier = nxt_frontier
    return dist


# --- the corpus ----------------------------------------------------------------------
#
# Every supported annotation shape, plus every spelling of Field(pattern=...) the
# findings name. `instances` are values whose `model_dump_json()` output must itself
# match the grammar (the round-trip direction).


class Priority(str, enum.Enum):
    LOW = "low"
    HIGH = "high"


class IntCode(enum.Enum):
    A = 1
    B = 2


class Inner(BaseModel):
    category: str
    score: float


class PlainScalars(BaseModel):
    text: str
    count: int
    ratio: float
    flag: bool


class Nullable(BaseModel):
    maybe: Optional[str]


class WithLiteralStr(BaseModel):
    level: Literal["low", "high"]


class WithLiteralInt(BaseModel):
    code: Literal[1, 2]


class WithLiteralBool(BaseModel):
    on: Literal[True]


class WithLiteralFloat(BaseModel):
    amount: Literal[1.5]


class WithEnumStr(BaseModel):
    priority: Priority


class WithEnumInt(BaseModel):
    code: IntCode


class WithDecimal(BaseModel):
    amount: Decimal


class WithDates(BaseModel):
    day: dt.date
    moment: dt.datetime


class WithUUID(BaseModel):
    ident: uuid.UUID


class WithCollections(BaseModel):
    tags: List[str]
    pair: Tuple[str, int]
    uniq: Set[int]
    frozen: FrozenSet[int]
    mapping: Dict[str, int]


class WithNested(BaseModel):
    detail: Inner


class WithUnion(BaseModel):
    either: Union[str, int]


class WithAny(BaseModel):
    whatever: Any


class PatLiteral(BaseModel):
    x: str = Field(pattern=r"abc")


class PatClass(BaseModel):
    x: str = Field(pattern=r"[0-9]{5}")


class PatDotStar(BaseModel):
    x: str = Field(pattern=r".*")


class PatNegatedClass(BaseModel):
    x: str = Field(pattern=r"[^a]+")


class PatShorthandD(BaseModel):
    x: str = Field(pattern=r"\d{3}")


class PatShorthandNegD(BaseModel):
    x: str = Field(pattern=r"\D+")


class PatShorthandW(BaseModel):
    x: str = Field(pattern=r"\w+")


class PatShorthandNegW(BaseModel):
    x: str = Field(pattern=r"\W+")


class PatShorthandS(BaseModel):
    x: str = Field(pattern=r"\s+")


class PatShorthandNegS(BaseModel):
    x: str = Field(pattern=r"\S+")


class PatAnyClass(BaseModel):
    x: str = Field(pattern=r"[\w\W]+")


class PatAlternation(BaseModel):
    x: str = Field(pattern=r"cat|dog")


class PatAnchored(BaseModel):
    x: str = Field(pattern=r"^[0-9]{3}$")


class PatEscapedDollar(BaseModel):
    x: str = Field(pattern=r"^\$[0-9]+\.[0-9]{2}$")


class PatTrailingEscapedDollar(BaseModel):
    x: str = Field(pattern=r"a\$")


class PatInlineIgnoreCase(BaseModel):
    x: str = Field(pattern=r"(?i)abc")


class PatNamedGroup(BaseModel):
    # S-1's named-group collision is across FIELDS, not within one pattern: pydantic's
    # own rust-regex engine refuses `(?P<n>a)(?P<n>b)` in a single field with
    # "duplicate capture group name", so the report's spelling of it is unreachable.
    # Two fields each carrying the same group name is what makes the *assembled*
    # grammar an invalid Python regex ("redefinition of group name").
    x: str = Field(pattern=r"(?P<n>a)b")
    y: str = Field(pattern=r"(?P<n>c)d")


class PatGroupedAlternation(BaseModel):
    x: str = Field(pattern=r"(cat|dog)s?")


class PatPrecompiled(BaseModel):
    x: str = Field(pattern=re.compile(r"[a-z]+"))


class PatNonAscii(BaseModel):
    x: str = Field(pattern=r"caf[eé]")


class PatAndPlainField(BaseModel):
    zip_code: str = Field(pattern=r"[0-9]{5}")
    name: str


class AliasPopulateByName(BaseModel):
    """S-5, in the one alias configuration all three spine arms can hold for.

    A *strict*-alias model (pydantic's default, `AliasStrict` below) cannot go in this
    corpus, because `test_spine_model_dump_json_matches_the_grammar` would fail on it
    through no fault of the compiler: the grammar must require the key pydantic
    **validates** (`fullName`), while `model_dump_json()` with its default
    `by_alias=False` emits `full_name` -- a string pydantic itself refuses to read
    back. That asymmetry is pydantic's, not the grammar's. With `populate_by_name` both
    keys validate, so the round-trip arm is meaningful again, and the strict case is
    covered in the must-accept arm instead.
    """

    model_config = ConfigDict(populate_by_name=True)
    full_name: str = Field(alias="fullName")
    count: int


class AliasStrict(BaseModel):
    full_name: str = Field(alias="fullName")


class AliasChoicesModel(BaseModel):
    full_name: str = Field(validation_alias=AliasChoices("a", "bb"))


class BoundedString(BaseModel):
    x: str = Field(min_length=2, max_length=5)


class BoundedList(BaseModel):
    x: List[int] = Field(min_length=1, max_length=3)


class BoundedInt(BaseModel):
    x: int = Field(ge=-3, le=10)


SPINE_CASES = [
    ("plain_scalars", PlainScalars, [PlainScalars(text="a b", count=-7, ratio=1.5, flag=True),
                                     PlainScalars(text='q"\\\n\t', count=0, ratio=0.0, flag=False),
                                     PlainScalars(text="café ٣", count=1, ratio=-2.25, flag=True)]),
    ("nullable", Nullable, [Nullable(maybe=None), Nullable(maybe="x")]),
    ("literal_str", WithLiteralStr, [WithLiteralStr(level="low")]),
    ("literal_int", WithLiteralInt, [WithLiteralInt(code=1)]),
    ("literal_bool", WithLiteralBool, [WithLiteralBool(on=True)]),
    ("literal_float", WithLiteralFloat, [WithLiteralFloat(amount=1.5)]),
    ("enum_str", WithEnumStr, [WithEnumStr(priority=Priority.LOW)]),
    ("enum_int", WithEnumInt, [WithEnumInt(code=IntCode.A)]),
    ("dates", WithDates, [WithDates(day=dt.date(2026, 2, 28),
                                    moment=dt.datetime(2026, 2, 28, 13, 14, 15))]),
    ("uuid", WithUUID, [WithUUID(ident=uuid.UUID("12345678-1234-5678-1234-567812345678"))]),
    ("collections", WithCollections, [WithCollections(tags=["a", "b"], pair=("x", 1),
                                                      uniq={1, 2}, frozen=frozenset({3}),
                                                      mapping={"k": 9})]),
    ("nested", WithNested, [WithNested(detail=Inner(category="c", score=0.5))]),
    ("union", WithUnion, [WithUnion(either="s"), WithUnion(either=3)]),
    ("pat_literal", PatLiteral, [PatLiteral(x="abc")]),
    ("pat_class", PatClass, [PatClass(x="90210")]),
    ("pat_dot_star", PatDotStar, [PatDotStar(x="anything")]),
    ("pat_negated_class", PatNegatedClass, [PatNegatedClass(x="bbb")]),
    ("pat_shorthand_d", PatShorthandD, [PatShorthandD(x="123")]),
    ("pat_shorthand_neg_d", PatShorthandNegD, [PatShorthandNegD(x="abc")]),
    ("pat_shorthand_w", PatShorthandW, [PatShorthandW(x="a_1")]),
    ("pat_shorthand_neg_w", PatShorthandNegW, [PatShorthandNegW(x="!!")]),
    ("pat_shorthand_s", PatShorthandS, [PatShorthandS(x="  ")]),
    ("pat_shorthand_neg_s", PatShorthandNegS, [PatShorthandNegS(x="xy")]),
    ("pat_any_class", PatAnyClass, [PatAnyClass(x="zz")]),
    ("pat_alternation", PatAlternation, [PatAlternation(x="cat")]),
    ("pat_anchored", PatAnchored, [PatAnchored(x="123")]),
    ("pat_escaped_dollar", PatEscapedDollar, [PatEscapedDollar(x="$12.34")]),
    ("pat_trailing_escaped_dollar", PatTrailingEscapedDollar, [PatTrailingEscapedDollar(x="a$")]),
    ("pat_inline_ignorecase", PatInlineIgnoreCase, [PatInlineIgnoreCase(x="aBc")]),
    ("pat_named_group", PatNamedGroup, [PatNamedGroup(x="ab", y="cd")]),
    ("pat_grouped_alternation", PatGroupedAlternation, [PatGroupedAlternation(x="dogs")]),
    ("pat_precompiled", PatPrecompiled, [PatPrecompiled(x="abc")]),
    ("pat_non_ascii", PatNonAscii, [PatNonAscii(x="café")]),
    ("pat_and_plain_field", PatAndPlainField, [PatAndPlainField(zip_code="90210", name="n")]),
    ("alias_populate_by_name", AliasPopulateByName,
     [AliasPopulateByName(fullName="v", count=1), AliasPopulateByName(full_name="w", count=2)]),
    # S-9. Only the EXPRESSIBLE constraints belong in this corpus: a constraint the
    # compiler can merely warn about (an open-ended range, `multiple_of`, a length
    # bound on `Optional[str]`) leaves the grammar wider than pydantic by design, so
    # the spine property genuinely does not hold for it and putting one here would
    # assert the opposite of the decision the track took. Those live in
    # `tests/test_schema.py::test_inexpressible_constraints_warn_rather_than_vanish_S_9`.
    ("len_str", BoundedString, [BoundedString(x="ab"), BoundedString(x="caf\u00e9!")]),
    ("len_list", BoundedList, [BoundedList(x=[1]), BoundedList(x=[1, 2, 3])]),
    ("int_range", BoundedInt, [BoundedInt(x=-3), BoundedInt(x=0), BoundedInt(x=10)]),
]

SPINE_IDS = [c[0] for c in SPINE_CASES]


def _compile_or_raise(model):
    """Return the compiled regex, or None if the compiler refused with PAWSchemaError.

    A refusal is an allowed outcome everywhere in this module: "never emit an unsound
    grammar in preference to an error" is a stated invariant of the track, so a case
    that raises `PAWSchemaError` naming its construct has satisfied the property. What
    is never allowed is a grammar that accepts a string pydantic rejects.
    """
    try:
        return pydantic_to_regex(model, anchors=False)
    except PAWSchemaError:
        return None


# --- arm 1: accepted => valid --------------------------------------------------------


@pytest.mark.parametrize("name,model,instances", SPINE_CASES, ids=SPINE_IDS)
def test_spine_every_accepted_string_is_valid_json_and_validates(name, model, instances) -> None:
    """THE SPINE. Every string the compiled grammar accepts must parse and validate."""
    pattern = _compile_or_raise(model)
    if pattern is None:
        pytest.skip(f"{name}: compiler refused with PAWSchemaError (an allowed outcome)")

    # The compiled grammar must be a regex Python `re` will take -- a precondition for
    # the property below to even be checkable, and S-1's own failure mode.
    compiled = re.compile(pattern)

    bad_json = []
    bad_validate = []
    disagree = []
    samples = accepted_strings(pattern)
    assert samples, f"{name}: enumerated no accepted strings at all"
    for s in samples:
        if compiled.fullmatch(s) is None:
            disagree.append(s)
            continue
        try:
            json.loads(s)
        except Exception as exc:
            bad_json.append((s, f"{type(exc).__name__}: {exc}"))
            continue
        try:
            model.model_validate_json(s)
        except Exception as exc:
            bad_validate.append((s, f"{type(exc).__name__}: {str(exc)[:80]}"))

    assert not disagree, (
        f"{name}: the decoding FSM accepts {len(disagree)} strings the exported regex "
        f"rejects, so the two have different languages (S-3b): {disagree[:3]!r}"
    )
    assert not bad_json, (
        f"{name}: the grammar accepts {len(bad_json)} strings that are not valid JSON "
        f"(S-2 / S-3): {bad_json[:3]!r}"
    )
    assert not bad_validate, (
        f"{name}: the grammar accepts {len(bad_validate)} strings "
        f"{model.__name__}.model_validate_json rejects (S-2 / S-3 / S-5): "
        f"{bad_validate[:3]!r}"
    )


@pytest.mark.parametrize("name,model,instances", SPINE_CASES, ids=SPINE_IDS)
def test_spine_model_dump_json_matches_the_grammar(name, model, instances) -> None:
    """The round-trip direction: what the model serialises must satisfy its own grammar."""
    pattern = _compile_or_raise(model)
    if pattern is None:
        pytest.skip(f"{name}: compiler refused with PAWSchemaError (an allowed outcome)")
    compiled = re.compile(pattern)
    for instance in instances:
        dumped = instance.model_dump_json()
        assert compiled.fullmatch(dumped) is not None, (
            f"{name}: {model.__name__}.model_dump_json() produced {dumped!r}, which its "
            f"own compiled grammar rejects"
        )


# --- arm 2: the curated must-accept list --------------------------------------------
#
# One entry per finding's documented legal value. Enumerated, never generated: the
# grammar is deliberately narrower than pydantic, so "pydantic accepts it" is not a
# reason the grammar must. These are the specific values a finding names.

MUST_ACCEPT = [
    # S-7: `rstrip("$")` is character-wise, so `a\$` loses its escape and the grammar
    # both accepts `{"x":"a"}` (pydantic rejects) and rejects `{"x":"a$"}` (pydantic
    # accepts). The legal value is the one with the dollar sign.
    ("S-7", PatTrailingEscapedDollar, '{"x":"a$"}'),
    ("S-7", PatEscapedDollar, '{"x":"$12.34"}'),
    # S-1: alternation is spliced without a group, so the `|` becomes top-level and the
    # grammar rejects BOTH legal values.
    ("S-1", PatAlternation, '{"x":"cat"}'),
    ("S-1", PatAlternation, '{"x":"dog"}'),
    # S-1: two fields sharing a named group, and an inline flag, must not make the
    # exported string an invalid regex.
    ("S-1", PatNamedGroup, '{"x":"ab","y":"cd"}'),
    ("S-1", PatInlineIgnoreCase, '{"x":"aBc"}'),
    ("S-1", PatInlineIgnoreCase, '{"x":"abc"}'),
    ("S-1", PatGroupedAlternation, '{"x":"dogs"}'),
    ("S-1", PatGroupedAlternation, '{"x":"cat"}'),
    # S-3: the translation must not make ordinary non-ASCII values unreachable -- the
    # `ensure_ascii=False` decision in `_json_string_literal_regex` is deliberate and
    # the same reasoning applies to an explicit class.
    ("S-3", PatNonAscii, '{"x":"café"}'),
    ("S-3", PatDotStar, '{"x":"café ٣"}'),
    ("S-3", PatNegatedClass, '{"x":"bé"}'),
    # S-3: the pattern spellings that work today must keep working.
    ("S-3", PatClass, '{"zip_code":"90210"}'.replace("zip_code", "x")),
    ("S-3", PatAnchored, '{"x":"123"}'),
    ("S-3", PatShorthandD, '{"x":"123"}'),
    ("S-3", PatShorthandNegD, '{"x":"abc"}'),
    ("S-3", PatShorthandW, '{"x":"a_1"}'),
    ("S-3", PatShorthandNegS, '{"x":"xy"}'),
    ("S-3", PatAndPlainField, '{"zip_code":"90210","name":"n"}'),
    # S-5: the grammar must require the key pydantic VALIDATES. With the default
    # config that is the alias alone; with populate_by_name it is either. The strict
    # model lives only here, not in SPINE_CASES -- see `AliasPopulateByName`'s
    # docstring for why the round-trip arm cannot hold for it.
    ("S-5", AliasStrict, '{"fullName":"v"}'),
    ("S-5", AliasPopulateByName, '{"fullName":"v","count":1}'),
    ("S-5", AliasPopulateByName, '{"full_name":"v","count":1}'),
    ("S-5", AliasChoicesModel, '{"a":"v"}'),
    ("S-5", AliasChoicesModel, '{"bb":"v"}'),
    # S-2: the escapes JSON really does permit must stay reachable after the escape
    # class is tightened.
    ("S-2", PlainScalars, '{"text":"a\\"b","count":1,"ratio":1.0,"flag":true}'),
    ("S-2", PlainScalars, '{"text":"a\\\\b","count":1,"ratio":1.0,"flag":true}'),
    ("S-2", PlainScalars, '{"text":"a\\nb","count":1,"ratio":1.0,"flag":true}'),
    ("S-2", PlainScalars, '{"text":"a\\u00e9b","count":1,"ratio":1.0,"flag":true}'),
    ("S-2", PlainScalars, '{"text":"a\\/b","count":1,"ratio":1.0,"flag":true}'),
]

MUST_ACCEPT_IDS = [f"{fid}-{m.__name__}-{i}" for i, (fid, m, _) in enumerate(MUST_ACCEPT)]


@pytest.mark.parametrize("finding,model,value", MUST_ACCEPT, ids=MUST_ACCEPT_IDS)
def test_must_accept_documented_legal_value(finding, model, value) -> None:
    """A value a finding documents as legal must be accepted by the compiled grammar.

    This arm exists because the spine test is one-directional and therefore cannot see
    under-acceptance at all.
    """
    # Precondition: pydantic itself accepts it. If this fails the corpus entry is wrong,
    # not the compiler.
    model.model_validate_json(value)

    pattern = pydantic_to_regex(model, anchors=False)
    assert re.fullmatch(pattern, value) is not None, (
        f"{finding}: the grammar for {model.__name__} rejects {value!r}, which pydantic "
        f"accepts and the finding documents as legal"
    )
    # And the decoder must agree with the exported regex about it.
    fsm = interegular.parse_pattern(pattern).to_fsm()
    assert fsm.accepts(value), (
        f"{finding}: the exported regex accepts {value!r} but the decoding FSM does not"
    )


# --- arm 3: non-ASCII FSM/regex agreement (S-3b) ------------------------------------

NON_ASCII_PROBES = ("é", "٣", " ", "😀", "Ω", "·")
ASCII_PROBES = ("a", "Z", "0", "_", " ", "!", "~")


@pytest.mark.parametrize("name,model,instances", SPINE_CASES, ids=SPINE_IDS)
def test_fsm_and_regex_agree_on_non_ascii_values(name, model, instances) -> None:
    """The exported regex and the decoding FSM must accept the same language (S-3b).

    The bug hunt checked this over 4,004 ASCII strings. interegular's `\\d`/`\\w`/`\\s`
    are ASCII while Python `re`'s and pydantic's are Unicode, so an ASCII corpus cannot
    see the divergence; every probe here is chosen to sit on that fault line.
    """
    pattern = _compile_or_raise(model)
    if pattern is None:
        pytest.skip(f"{name}: compiler refused with PAWSchemaError (an allowed outcome)")
    compiled = re.compile(pattern)
    fsm = interegular.parse_pattern(pattern).to_fsm()

    candidates = []
    for instance in instances:
        base = instance.model_dump_json()
        candidates.append(base)
        for probe in NON_ASCII_PROBES + ASCII_PROBES:
            # Splice the probe into each string-valued position of a real serialisation.
            for m in re.finditer(r'"([^"\\]*)"', base):
                candidates.append(base[: m.end() - 1] + probe + base[m.end() - 1 :])
                candidates.append(base[: m.start() + 1] + probe + base[m.start() + 1 :])

    mismatches = [
        c for c in set(candidates)
        if (compiled.fullmatch(c) is not None) != fsm.accepts(c)
    ]
    assert not mismatches, (
        f"{name}: exported regex and decoding FSM disagree on {len(mismatches)} "
        f"non-ASCII candidates (S-3b): "
        + repr([(c, compiled.fullmatch(c) is not None, fsm.accepts(c)) for c in mismatches[:3]])
    )


# --- arm 4: the Edit-7 success criterion -------------------------------------------
#
# "For every pattern pydantic accepts, `pydantic_to_regex` either returns a regex
# `re.compile` accepts, or raises `PAWSchemaError` naming the construct." The original
# criterion ("re.compile succeeds for every pattern pydantic accepts") is impossible:
# pydantic's rust-regex engine accepts `\p{L}+`, which Python `re` does not compile.

PATTERNS_PYDANTIC_ACCEPTS = [
    r"abc",
    r"[0-9]{5}",
    r".*",
    r"[^a]+",
    r"\d{3}",
    r"\D+",
    r"\w+",
    r"\W+",
    r"\s+",
    r"\S+",
    r"[\w\W]+",
    r"cat|dog",
    r"(cat|dog)s?",
    r"^[0-9]{3}$",
    r"a\$",
    r"^\$[0-9]+\.[0-9]{2}$",
    r"(?i)abc",
    r"(?P<n>a)b",
    r"caf[eé]",
    r"[a-z]+",
    r"a{2,4}",
    r"(?:ab)+",
    r"[.]",
    r"[^\D]",
    r"\p{L}+",          # rust-regex only; Python `re` cannot compile it
    r"\b\w+\b",         # interegular Unsupported
    # NOT listed: `(?=a)b`, `(?<=a)b` and `\Qa.b\E`. The report and the track spec
    # cite those as patterns pydantic accepts; executed against pydantic 2.13.5 it
    # REJECTS all three at class-build time with SchemaError, so they cannot reach the
    # compiler through `Field(pattern=...)` at all. They are covered where they are
    # genuinely reachable instead -- `RegexLogitsProcessor`, which takes any string.
    r"[[:alpha:]]+",    # POSIX: pydantic and Python `re` read this differently
    r"a(?i)b",          # rust-regex only
]


@pytest.mark.parametrize("pattern_src", PATTERNS_PYDANTIC_ACCEPTS)
def test_every_pattern_pydantic_accepts_compiles_or_raises_paw_schema_error(pattern_src) -> None:
    """No pattern pydantic accepts may produce a string Python `re` refuses."""
    from pydantic import create_model

    model = create_model("PatCase", x=(str, Field(pattern=pattern_src)))

    try:
        compiled_pattern = pydantic_to_regex(model, anchors=False)
    except PAWSchemaError:
        return  # allowed: refused, naming the construct
    try:
        re.compile(compiled_pattern)
    except re.error as exc:
        pytest.fail(
            f"pattern {pattern_src!r} compiled to a string Python `re` rejects "
            f"({exc}); the compiler must raise PAWSchemaError instead of returning it. "
            f"Got: {compiled_pattern!r}"
        )
    # And the decoder must be able to compile it too -- it is the same string.
    try:
        interegular.parse_pattern(compiled_pattern).to_fsm()
    except (interegular.patterns.Unsupported, interegular.patterns.InvalidSyntax) as exc:
        pytest.fail(
            f"pattern {pattern_src!r} compiled to a string the decoder's own engine "
            f"rejects ({type(exc).__name__}: {exc}): {compiled_pattern!r}"
        )


# --- arm 5: length-bound coherence (H-1, Phase F review) -----------------------------

LENGTH_BOUND_CASES = [
    # (annotation, min_length, max_length, must_raise)
    (str, 5, 2, True),       # H-1: no string can satisfy both
    (str, 3, 3, False),      # boundary: exactly one length, must still compile
    (str, 0, 0, False),      # boundary: only the empty string
    (List[int], 3, 0, True),  # H-1b: max_length=0 silently dropped `low` at main
    (List[int], 0, 0, False),  # boundary: only the empty list
    (List[int], 2, 5, False),  # ordinary case, sanity check
]


@pytest.mark.parametrize("annotation,min_len,max_len,must_raise", LENGTH_BOUND_CASES,
                         ids=[f"{a}_{lo}_{hi}" for a, lo, hi, _ in LENGTH_BOUND_CASES])
def test_incoherent_length_bounds_raise_paw_schema_error_not_a_broken_regex(
    annotation, min_len, max_len, must_raise
) -> None:
    """H-1: `min_length > max_length` (or `max_length == 0 < min_length`) describes a
    field with no legal value. Before this fix, `pydantic_to_regex` returned a string
    Python `re` refuses (`{5,2}`), which then reached `RegexLogitsProcessor` as an
    unwrapped, non-PAWSchemaError exception -- S-1's and S-15's exact failure shapes,
    reopened through the length-bound door. A coherent bound must still compile and
    round-trip normally; only the incoherent ones may raise.
    """
    from pydantic import create_model

    model = create_model(
        "LenCase", x=(annotation, Field(min_length=min_len, max_length=max_len))
    )
    if must_raise:
        with pytest.raises(PAWSchemaError, match="min_length"):
            pydantic_to_regex(model)
        return

    compiled_pattern = pydantic_to_regex(model)
    re.compile(compiled_pattern)  # must not raise re.error
    interegular.parse_pattern(compiled_pattern).to_fsm()  # decoder must accept it too
