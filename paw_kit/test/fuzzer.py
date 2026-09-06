"""Adversarial mutation fuzzer for neural adapter robustness testing."""

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
        """
        seeds = base_inputs or ["example input"]
        fuzzed: List[str] = []

        # 1. Custom domain probes from suite.yaml
        fuzzed.extend(config.adversarial_probes)

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
        return unique_fuzzed[:_MAX_FUZZED_CASES]
