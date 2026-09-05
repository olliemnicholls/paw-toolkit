"""Token-level logit masking for grammar and regex constrained autoregressive decoding."""

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import interegular
from interegular.fsm import FSM


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

        # Compile regex into deterministic finite state machine (DFA)
        self.fsm: FSM = interegular.parse_pattern(clean_pattern).to_fsm()

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
            if char not in self.fsm.alphabet:
                return None
            symbol = self.fsm.alphabet[char]
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
