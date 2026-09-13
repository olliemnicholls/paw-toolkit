"""Token-level logit masking for grammar and regex constrained autoregressive decoding."""

from collections import OrderedDict
import math
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import interegular
from interegular.fsm import FSM
from interegular.patterns import InvalidSyntax, Unsupported

from paw_kit.schema.exceptions import PAWSchemaError

# PAW-SCHEMA-03: interegular's NFA-to-DFA (Powerset) construction has worst-case
# exponential state complexity -- a pathological pattern (e.g. overlapping repeated
# subexpressions) can pin a CPU core at 100% for many seconds with no way to interrupt
# it, since Python cannot forcibly cancel a running thread.
#
# S-16: the previous comment here called _MAX_PATTERN_LENGTH the *primary* defense,
# on the grounds that it "bounds the work before it starts, for free". It does not,
# and saying so is how S-16 came to be filed against a cap that turned out to be load
# bearing. What is true, and what the ordering below actually buys:
#
#   * _MAX_PATTERN_LENGTH is the only check that happens BEFORE any compilation, so
#     it is the only one that can refuse a pattern for free. That makes it the first
#     line, not the primary one.
#   * It does not bound the work. Executed: `(?:a|b)*a(?:a|b){30}` is TWENTY
#     characters, walks straight past any sane character cap, and its 2**30-state
#     powerset construction does not finish in 3 s. Pattern length and DFA cost are
#     only loosely related.
#   * _MAX_FSM_STATES is checked INSIDE _compile(), i.e. after `to_fsm()` has already
#     returned. It is a post-hoc assertion about a construction that terminated, not a
#     budget on one that might not.
#   * The timeout is what actually bounds *caller latency* for the pathological case.
#     It cannot bound CPU or memory: Python offers no way to cancel a running thread,
#     so the abandoned compile keeps going (see _compile_fsm_safe for what that costs
#     and why the thread must be a daemon).
#
# So: for a pattern whose DFA construction blows up, nothing bounds CPU or memory, and
# the character cap is the only pre-work bound there is. It stays for that reason --
# but at 50,000 rather than 1,000, because at 1,000 it was rejecting ordinary schemas.
# A realistic five-field nested invoice schema compiles to ~1,000 characters (and grew
# past the old cap outright once Field(pattern=...) constraints began being translated
# rather than spliced), while _MAX_FSM_STATES binds at roughly 55 fields. 50,000 is the
# same order as where the state cap binds, so the length cap stops producing false
# positives while still refusing absurd input for free.
_MAX_PATTERN_LENGTH = 50_000
_FSM_TIMEOUT_SECONDS = 3.0
_MAX_FSM_STATES = 10000

# PAW-SCHEMA-04: bounds on RegexLogitsProcessor's two per-instance caches, so a very
# long autoregressive generation (many distinct (state, token_id) pairs and/or many
# distinct states visited) cannot grow either dict without limit. `_MAX_FSM_STATES`
# already bounds distinct FSM states to 10,000; `_MAX_ALLOWED_TOKENS_CACHE_ENTRIES`
# comfortably covers realistic generations without holding an entry for literally
# every state a pathologically long run could visit. `_MAX_TRANSITION_CACHE_ENTRIES`
# is sized per (state, token) pair, so it needs more headroom.
_MAX_TRANSITION_CACHE_ENTRIES = 100_000
_MAX_ALLOWED_TOKENS_CACHE_ENTRIES = 5_000

_UNSET = object()  # cache-miss sentinel; a legitimate cached value can be None.


def _compile_fsm_safe(pattern: str) -> FSM:
    """Compile `pattern` into a DFA, bounded against ReDoS / FSM state explosion.

    Two checks, applied in that order (see the module comment on _MAX_PATTERN_LENGTH
    for what each one does and does not bound):

    1. A pattern-length cap, checked before any compilation is attempted. The only
       check that can refuse a pattern for free -- but it bounds input size, not work.
    2. A timeout on a background thread, which bounds *caller latency* only. The thread
       is deliberately not joined on timeout: doing so would defeat the timeout
       entirely, which is the mistake in the audit's own illustrative fix (it ran the
       compile inside a `with ThreadPoolExecutor(...)` block, and `Executor.__exit__`
       calls `shutdown(wait=True)` unconditionally, so even after `future.result()`
       raised `TimeoutError` the `with` block blocked the caller until the runaway
       compile finished anyway).

    S-17: the thread is a bare `threading.Thread(daemon=True)` rather than a
    `ThreadPoolExecutor` worker with `shutdown(wait=False)`, and the daemon flag is the
    whole point. `concurrent.futures` registers its worker threads with
    `threading._register_atexit`, so abandoning one does not actually abandon it: the
    caller returns promptly, but *interpreter shutdown* then joins the runaway compile.
    Executed against the executor version: `_compile_fsm_safe("(?:a|b)*a(?:a|b){30}")`
    raised the timeout PAWSchemaError at 3.02 s and the process never exited (killed
    externally at 40 s). That is reachable from the served path -- a `paw-serve` worker
    that compiles one pathological grammar raises correctly, keeps serving, and then
    cannot shut down, turning a graceful restart into a forced one. A daemon thread is
    not joined at exit, so the process leaves promptly and the abandoned compile dies
    with it.
    """
    if len(pattern) > _MAX_PATTERN_LENGTH:
        raise PAWSchemaError(
            f"Pattern length ({len(pattern)}) exceeds the maximum of "
            f"{_MAX_PATTERN_LENGTH} characters; refusing to compile it into a DFA."
        )

    def _compile() -> FSM:
        # S-15: interegular raises its own exception types for a construct it cannot
        # express -- `Unsupported` for `\b`, `\p{L}`, lookaround and backreferences,
        # `InvalidSyntax` for e.g. `\Qa.b\E`. Both are raised lazily -- a lookback
        # parses fine and only fails inside `to_fsm()` -- so both calls are wrapped. Letting them escape hands the caller a
        # third-party exception type instead of the documented `PAWSchemaError` that
        # `paw_kit` keys on. `loader.py:74-79` happens to wrap anything that is not a
        # `PAWSchemaError`, so the real beneficiary is `RegexLogitsProcessor.__init__`,
        # which has no such wrapper and is a public export.
        #
        # S-22: `Unsupported`/`InvalidSyntax` do not cover every way this call can fail.
        # A reversed quantifier bound (`a{5,2}`, min > max) parses fine -- interegular's
        # own parser does not reject it -- and fails inside `to_fsm()` with a bare
        # `Exception: Can't multiply an FSM by -3`, escaping both this wrapper and
        # `RegexLogitsProcessor` uncaught. Caught by type below (bare `except Exception`,
        # narrower than `BaseException`, so `KeyboardInterrupt`/`SystemExit` still
        # propagate) rather than by adding a third named exception class: interegular
        # does not export one for this failure mode, and `_compile`'s only job is
        # calling into interegular, so nothing else can reach this branch.
        try:
            fsm = interegular.parse_pattern(pattern).to_fsm()
        except (Unsupported, InvalidSyntax) as exc:
            raise PAWSchemaError(
                f"Cannot compile the pattern into a DFA: {type(exc).__name__}: {exc}. "
                "Grammar-constrained decoding needs a pattern expressible as a finite "
                "automaton; zero-width assertions (\\b, \\B), lookaround, "
                "backreferences and Unicode property classes (\\p{...}) are not."
            ) from exc
        except Exception as exc:
            raise PAWSchemaError(
                f"Cannot compile the pattern into a DFA: {type(exc).__name__}: {exc}. "
                "The pattern parsed but interegular could not turn it into a finite "
                "automaton -- a reversed quantifier bound (e.g. {5,2}, where the "
                "minimum exceeds the maximum) is one known cause."
            ) from exc
        if len(fsm.states) > _MAX_FSM_STATES:
            raise PAWSchemaError(
                f"Compiled FSM exceeds the maximum of {_MAX_FSM_STATES} states "
                f"({len(fsm.states)} states) -- the pattern is too complex to compile safely."
            )
        return fsm

    outcome: List[Any] = []

    def _run() -> None:
        try:
            outcome.append(("ok", _compile()))
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the caller's thread
            outcome.append(("raised", exc))

    # S-17: daemon=True is load bearing -- see the docstring. Without it the process
    # cannot exit while this thread runs.
    worker = threading.Thread(target=_run, name="paw-kit-fsm-compile", daemon=True)
    worker.start()
    worker.join(_FSM_TIMEOUT_SECONDS)
    if worker.is_alive():
        raise PAWSchemaError(
            f"FSM compilation timed out after {_FSM_TIMEOUT_SECONDS}s -- the "
            "pattern is likely pathological (exponential DFA state blowup)."
        )
    kind, payload = outcome[0]
    if kind == "raised":
        raise payload
    return payload


class RegexLogitsProcessor:
    """Masks token logits at each autoregressive step using an FSM compiled from a regex.

    Where it is applied to a sampling loop, every token that would leave the regex's
    language has its logit set to -infinity, so structural validity is a property of the
    FSM rather than a probability (15/15 valid parses against a real model, see
    `measurements/`). No backend shipped in `paw_kit` applies it; `scripts/` show how.
    """

    def __init__(
        self,
        regex_pattern: str,
        vocabulary: Dict[int, str],
        eos_token_id: Optional[int] = None,
    ) -> None:
        """Initialize processor with regex pattern and tokenizer vocabulary.

        Args:
            regex_pattern: Full-match regular expression pattern.
            vocabulary: Mapping of token ID to decoded string token.
            eos_token_id: Token ID denoting end-of-sequence (EOS). Optional, but
                without it `get_allowed_tokens` raises PAWSchemaError as soon as the
                walk reaches an accepting state (S-12): the grammar is satisfied,
                stopping is the only legal move, and no token expresses it.
        """
        # Strip anchors if present, as interegular full-matches by default
        clean_pattern = regex_pattern.lstrip("^").rstrip("$")
        self.pattern = clean_pattern
        self.vocabulary = vocabulary
        self.eos_token_id = eos_token_id

        # Compile regex into deterministic finite state machine (DFA), bounded against
        # ReDoS / FSM state explosion (PAW-SCHEMA-03).
        self.fsm: FSM = _compile_fsm_safe(clean_pattern)

        # PAW-SCHEMA-04: bucket the vocabulary by first character once, up front, so
        # get_allowed_tokens can skip whole buckets whose first character has no legal
        # transition from the current state instead of walking every token in the
        # vocabulary through the FSM for every new state it encounters. This is a
        # partial mitigation, not an asymptotic fix -- a state where "anything else"
        # is a legal transition (interegular's catch-all alphabet bucket) still admits
        # most first characters -- but for the common case of a highly restrictive
        # state (e.g. mid-way through matching a fixed JSON field-name literal), only
        # a small fraction of first-character buckets pass the check below.
        self._tokens_by_first_char: Dict[str, List[int]] = {}
        for tid, token_str in vocabulary.items():
            if token_str:
                self._tokens_by_first_char.setdefault(token_str[0], []).append(tid)

        # PAW-SCHEMA-04: bounded (LRU-evicted) instead of plain dicts -- an unbounded
        # cache keyed on every (state, token_id) pair or every state ever visited can
        # grow without limit over a very long generation. Cache transition results:
        # (state, token_id) -> next_state (or None if invalid).
        self._transition_cache: "OrderedDict[Tuple[int, int], Optional[int]]" = OrderedDict()
        # Cache allowed token sets per state: state -> set of token_ids.
        self._allowed_tokens_cache: "OrderedDict[int, Set[int]]" = OrderedDict()

    @property
    def initial_state(self) -> int:
        """Return the start state of the FSM."""
        return self.fsm.initial

    def is_final_state(self, state: int) -> bool:
        """Return True if the current state is an accepting (final) state."""
        return state in self.fsm.finals

    def _walk_string(self, state: int, text: str) -> Optional[int]:
        """Walk text through FSM character by character. Returns final state or None."""
        curr = state
        for char in text:
            symbol = self.fsm.alphabet[char]
            if symbol is None:
                return None
            curr = self.fsm.map[curr].get(symbol)
            if curr is None or not self.fsm.islive(curr):
                return None
        return curr

    def get_next_state(self, state: int, token_id: int) -> Optional[int]:
        """Compute the next state after emitting token_id from state.

        Returns:
            The new state integer if valid, or None if emitting this token is illegal.
        """
        cache_key = (state, token_id)
        cached = self._transition_cache.get(cache_key, _UNSET)
        if cached is not _UNSET:
            self._transition_cache.move_to_end(cache_key)
            return cached  # type: ignore[return-value]

        if self.eos_token_id is not None and token_id == self.eos_token_id:
            # EOS is only valid if we are already in an accepting state
            next_state = state if self.is_final_state(state) else None
        else:
            token_str = self.vocabulary.get(token_id)
            if token_str is None:
                next_state = None
            else:
                next_state = self._walk_string(state, token_str)

        self._transition_cache[cache_key] = next_state
        self._transition_cache.move_to_end(cache_key)
        if len(self._transition_cache) > _MAX_TRANSITION_CACHE_ENTRIES:
            self._transition_cache.popitem(last=False)
        return next_state

    def get_allowed_tokens(self, state: int) -> Set[int]:
        """Compute and return the set of all valid token IDs from state."""
        cached = self._allowed_tokens_cache.get(state, _UNSET)
        if cached is not _UNSET:
            self._allowed_tokens_cache.move_to_end(state)
            return cached  # type: ignore[return-value]

        allowed: Set[int] = set()
        # PAW-SCHEMA-04: walk only the tokens whose first character is actually a
        # legal transition out of `state`, rather than every token in the vocabulary
        # -- see the bucket built in __init__.
        state_transitions = self.fsm.map.get(state, {})
        for first_char, candidate_ids in self._tokens_by_first_char.items():
            symbol = self.fsm.alphabet[first_char]
            if symbol is None or state_transitions.get(symbol) is None:
                continue
            for token_id in candidate_ids:
                if token_id == self.eos_token_id:
                    continue  # EOS is special-cased below regardless of its bucket
                if self.get_next_state(state, token_id) is not None:
                    allowed.add(token_id)

        # EOS's legality never depends on walking its literal token string through
        # the FSM at all (get_next_state special-cases it on is_final_state alone),
        # so it must be checked independently of the character-bucket pruning above
        # -- a real-world EOS token's string form (e.g. "<eos>") typically doesn't
        # correspond to any in-pattern FSM transition, so the bucket loop would never
        # reach it otherwise, whether or not it also happens to appear in vocabulary.
        if self.eos_token_id is not None and self.get_next_state(state, self.eos_token_id) is not None:
            allowed.add(self.eos_token_id)

        # S-12: at a final state with no EOS token configured there is nothing legal
        # left to emit -- the grammar is satisfied and the only legal move is to stop,
        # but `eos_token_id=None` means no token expresses stopping. The old behaviour
        # was to return an empty set, which `filter_logits` turned into an all-`-inf`
        # mask; softmax of that is NaN in every framework, so the caller's sampler
        # produced garbage (or raised somewhere far away) with no exception, no warning
        # and nothing naming the cause. Every test in the suite passed an explicit EOS
        # id, so nothing covered it.
        #
        # The parameter deliberately stays optional. Making it required would be an API
        # break on a public export and would break four existing constructions that
        # never walk to a final state; raising only in this one state breaks none of
        # them.
        #
        # Only this case raises. An empty set at a NON-final state means something else
        # entirely -- a dead state, or a vocabulary that cannot spell the grammar's next
        # character -- and those are not this finding, so they keep their existing
        # behaviour rather than acquiring a new raise on a live path.
        if not allowed and self.is_final_state(state):
            raise PAWSchemaError(
                f"No token is legal at FSM state {state}. That state is FINAL "
                "(accepting): the pattern is already satisfied, so the only legal move "
                "is to stop -- but this RegexLogitsProcessor was constructed with "
                "eos_token_id=None, so no token expresses stopping and the mask would "
                "be -inf everywhere (softmax of which is NaN). This is not a dead "
                "state and not an over-constrained grammar; it is a missing EOS id. "
                "Pass eos_token_id=<your tokenizer's EOS token id> to "
                "RegexLogitsProcessor."
            )

        self._allowed_tokens_cache[state] = allowed
        self._allowed_tokens_cache.move_to_end(state)
        if len(self._allowed_tokens_cache) > _MAX_ALLOWED_TOKENS_CACHE_ENTRIES:
            self._allowed_tokens_cache.popitem(last=False)
        return allowed

    def filter_logits(
        self,
        state: int,
        logits: Union[Sequence[float], Dict[int, float]],
    ) -> Union[List[float], Dict[int, float]]:
        """Mask forbidden tokens in next_token logits by setting their value to -inf.

        Args:
            state: Current FSM state.
            logits: Logits as a dense sequence (indexed by token_id) or sparse dict.

        Returns:
            Masked logits matching the input type.
        """
        allowed = self.get_allowed_tokens(state)

        if isinstance(logits, dict):
            masked_dict: Dict[int, float] = {}
            for token_id, val in logits.items():
                masked_dict[token_id] = val if token_id in allowed else -float("inf")
            return masked_dict

        # Dense sequence (list or tuple)
        masked_list = list(logits)
        for token_id in range(len(masked_list)):
            if token_id not in allowed:
                masked_list[token_id] = -float("inf")
        return masked_list
