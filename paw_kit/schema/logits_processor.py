"""Token-level logit masking for grammar and regex constrained autoregressive decoding."""

import concurrent.futures
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import interegular
from interegular.fsm import FSM

from paw_kit.schema.exceptions import PAWSchemaError

# PAW-SCHEMA-03: interegular's NFA-to-DFA (Powerset) construction has worst-case
# exponential state complexity -- a pathological pattern (e.g. overlapping repeated
# subexpressions) can pin a CPU core at 100% for many seconds with no way to interrupt
# it, since Python cannot forcibly cancel a running thread. _MAX_PATTERN_LENGTH is the
# *primary* defense (it bounds the work before it starts, for free); the timeout below
# is only a secondary backstop for patterns that are short but still pathological.
_MAX_PATTERN_LENGTH = 1000
_FSM_TIMEOUT_SECONDS = 3.0
_MAX_FSM_STATES = 10000


def _compile_fsm_safe(pattern: str) -> FSM:
    """Compile `pattern` into a DFA, bounded against ReDoS / FSM state explosion.

    Two independent defenses, applied in priority order:

    1. A pattern-length cap, checked before any compilation is attempted. This is the
       primary defense: it rejects known-pathological input sizes for free rather than
       trying to detect blowup after the fact.
    2. A timeout on a background thread, as a secondary backstop. Crucially, the thread
       is *not* joined on timeout: this deliberately avoids the mistake in the audit's
       own illustrative fix, which ran the compile inside a `with
       ThreadPoolExecutor(...)` block -- `Executor.__exit__` calls `shutdown(wait=True)`
       unconditionally, so even after `future.result()` raises `TimeoutError` the
       `with` block still blocks the caller until the runaway compile finishes anyway,
       which defeats the timeout entirely. Calling `executor.shutdown(wait=False)`
       explicitly instead lets the caller return immediately; the abandoned thread
       keeps running in the background (Python has no way to cancel it) until it
       eventually finishes or the process exits.
    """
    if len(pattern) > _MAX_PATTERN_LENGTH:
        raise PAWSchemaError(
            f"Pattern length ({len(pattern)}) exceeds the maximum of "
            f"{_MAX_PATTERN_LENGTH} characters; refusing to compile it into a DFA."
        )

    def _compile() -> FSM:
        fsm = interegular.parse_pattern(pattern).to_fsm()
        if len(fsm.states) > _MAX_FSM_STATES:
            raise PAWSchemaError(
                f"Compiled FSM exceeds the maximum of {_MAX_FSM_STATES} states "
                f"({len(fsm.states)} states) -- the pattern is too complex to compile safely."
            )
        return fsm

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_compile)
        try:
            return future.result(timeout=_FSM_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            raise PAWSchemaError(
                f"FSM compilation timed out after {_FSM_TIMEOUT_SECONDS}s -- the "
                "pattern is likely pathological (exponential DFA state blowup)."
            )
    finally:
        executor.shutdown(wait=False)


class RegexLogitsProcessor:
    """Masks token logits at each autoregressive step using an FSM compiled from a regex.

    Guarantees 0.0% syntax failure by setting the probability of any token that would
    lead to an invalid state to -infinity.
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
            eos_token_id: Token ID denoting end-of-sequence (EOS).
        """
        # Strip anchors if present, as interegular full-matches by default
        clean_pattern = regex_pattern.lstrip("^").rstrip("$")
        self.pattern = clean_pattern
        self.vocabulary = vocabulary
        self.eos_token_id = eos_token_id

        # Compile regex into deterministic finite state machine (DFA), bounded against
        # ReDoS / FSM state explosion (PAW-SCHEMA-03).
        self.fsm: FSM = _compile_fsm_safe(clean_pattern)

        # Cache transition results: (state, token_id) -> next_state (or None if invalid)
        self._transition_cache: Dict[Tuple[int, int], Optional[int]] = {}
        # Cache allowed token sets per state: state -> set of token_ids
        self._allowed_tokens_cache: Dict[int, Set[int]] = {}

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
        if cache_key in self._transition_cache:
            return self._transition_cache[cache_key]

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
        return next_state

    def get_allowed_tokens(self, state: int) -> Set[int]:
        """Compute and return the set of all valid token IDs from state."""
        if state in self._allowed_tokens_cache:
            return self._allowed_tokens_cache[state]

        allowed: Set[int] = set()
        for token_id in self.vocabulary:
            if self.get_next_state(state, token_id) is not None:
                allowed.add(token_id)

        # Check EOS token if separate from vocabulary keys
        if (
            self.eos_token_id is not None
            and self.eos_token_id not in self.vocabulary
            and self.is_final_state(state)
        ):
            allowed.add(self.eos_token_id)

        self._allowed_tokens_cache[state] = allowed
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
