"""Adversarial mutation fuzzer for neural adapter robustness testing."""

from dataclasses import dataclass, field
from typing import List, Optional
from paw_kit.test.suite import FuzzingConfig

# PAW-TEST-06: Track 09's _MAX_YAML_BYTES (suite.py) already bounds any single seed
# to <=1MB, but `seed * 200` still multiplies that to ~200MB *per seed*, with nothing
# bounding how many seeds there are -- both halves need a cap. `_MAX_PAYLOAD_EXTREME_LENGTH`
# bounds the *repeat count* directly (not the string after building it), so an
# oversized string is never actually constructed in the first place.
# `_MAX_FUZZED_CASES` bounds the total number of cases `generate()` returns overall.
_MAX_PAYLOAD_EXTREME_LENGTH = 50_000
_MAX_FUZZED_CASES = 500

UNICODE_MUTATIONS = [
    "\u200B",  # Zero-width space
    "\u200C",  # Zero-width non-joiner
    "\u202E",  # Right-to-left override
    "\uFEFF",  # Byte-order mark
    "🔥🚀🚨",  # Multi-byte emojis
    "\\u0000",  # Escaped null
    "\x00\x1f",  # Raw control characters
]

WHITESPACE_MUTATIONS = [
    "   ",
    "\t\t\t",
    "\n\n\r\n",
    "   \t  \n  ",
]


@dataclass(frozen=True)
class FuzzGenerationResult:
    """What `AdversarialFuzzer.generate_detailed` produced, including what it dropped.

    H-12: `generate()` returns `unique_fuzzed[:500]` and said nothing about the
    remainder. Custom probes were appended *first*, so a suite with more than 500 of
    them silently truncated away every generated category -- verified by the bug hunt:
    no zero-width space appeared anywhere in the generated set while the suite reported
    `inject_unicode: true`. A whole enabled mutation category ran zero cases and nothing
    said so.
    """

    cases: List[str] = field(default_factory=list)
    #: How many unique cases were generated and then cut by the total cap.
    dropped_count: int = 0

    @property
    def generated_count(self) -> int:
        return len(self.cases)


class AdversarialFuzzer:
    """Generates synthetic adversarial and edge-case inputs for test harnesses."""

    @classmethod
    def generate(
        cls,
        config: FuzzingConfig,
        base_inputs: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate mutated inputs according to the FuzzingConfig rules.

        Args:
            config: Fuzzing configuration settings.
            base_inputs: Optional base seed inputs to mutate.

        Returns:
            List of unique adversarial input strings.

        Kept as the list-returning entry point every existing caller uses. Use
        `generate_detailed` where the dropped count matters (H-12).
        """
        return cls.generate_detailed(config, base_inputs).cases

    @classmethod
    def generate_detailed(
        cls,
        config: FuzzingConfig,
        base_inputs: Optional[List[str]] = None,
    ) -> FuzzGenerationResult:
        """`generate`, plus how many cases the total cap dropped (H-12)."""
        seeds = base_inputs or ["example input"]
        fuzzed: List[str] = []

        # 2. Empty inputs
        if config.empty_inputs:
            fuzzed.append("")

        # 3. Whitespace floods
        if config.whitespace_flood:
            fuzzed.extend(WHITESPACE_MUTATIONS)
            for seed in seeds:
                fuzzed.append(f"   {seed}   \n")
                fuzzed.append(f"\t\t{seed}\t\t")

        # 4. Unicode & corruptions
        if config.inject_unicode:
            for u in UNICODE_MUTATIONS:
                fuzzed.append(u)
                for seed in seeds:
                    fuzzed.append(f"{seed}{u}")
                    fuzzed.append(f"{u}{seed}")

        # 5. Payload length extremes
        if config.payload_extremes:
            fuzzed.append("A" * 5000)
            for seed in seeds:
                # PAW-TEST-06: cap the repeat count itself so `seed * repeat_count`
                # never exceeds _MAX_PAYLOAD_EXTREME_LENGTH -- a long seed (up to
                # suite.py's 1MB YAML cap) multiplied by a fixed 200x previously had
                # no ceiling at all.
                repeat_count = max(1, min(200, _MAX_PAYLOAD_EXTREME_LENGTH // max(1, len(seed))))
                fuzzed.append(seed * repeat_count)

        # 1. Custom domain probes from suite.yaml -- appended LAST (H-12).
        #
        # They used to come first, which interacts badly with the total cap below: a
        # suite with more than _MAX_FUZZED_CASES custom probes consumed the entire
        # budget, and every generated category was truncated away in full while the
        # suite still declared it enabled. The bug hunt verified this -- not one
        # zero-width space appeared in the generated set of a suite with
        # `inject_unicode: true`.
        #
        # Ordering last means the generated categories -- which a suite cannot
        # over-supply, since their count is bounded by the enabled flags and the seed
        # count -- always get their turn, and it is the (unbounded, author-supplied)
        # probe list that gets cut. The cut is now reported rather than silent.
        fuzzed.extend(config.adversarial_probes)

        # Deduplicate while preserving order
        seen = set()
        unique_fuzzed: List[str] = []
        for item in fuzzed:
            if item not in seen:
                seen.add(item)
                unique_fuzzed.append(item)

        # PAW-TEST-06: cap the total case count regardless of source -- an
        # attacker-controlled adversarial_probes list (or a large seed count feeding
        # whitespace_flood/inject_unicode's per-seed multiplication) could otherwise
        # multiply the number of cases TestRunner.run() has to execute without limit.
        kept = unique_fuzzed[:_MAX_FUZZED_CASES]
        return FuzzGenerationResult(
            cases=kept, dropped_count=len(unique_fuzzed) - len(kept)
        )
