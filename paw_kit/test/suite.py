"""Parser and data models for declarative suite.yaml test specifications."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import yaml

from paw_kit.pathsafety import ensure_contained

_NUMERIC_VALUE_RULES = {"max_length", "min_length"}
_REQUIRED_VALUE_RULES = _NUMERIC_VALUE_RULES | {"exact_match", "not_contains"}
# The complete set `evaluate_assertion` (paw_kit/test/runner.py) actually implements.
# Found 2026-09-09: a suite.yaml naming any *other* rule (e.g. a plausible-sounding
# `contains`, `is_valid_json`, `one_of`) used to load successfully and then silently
# fail every single case at eval time via evaluate_assertion's "Unknown assertion rule"
# fallback -- the same "no signal until you go read the per-case output" shape as the
# two `conductor/deferred/index.md` entries this fix sits alongside. Validated here, at
# suite-load time, matching this class's own established convention (see
# _validate_value's docstring) rather than left for evaluate_assertion's fallback to
# paper over. A directly-constructed or `model_construct()`-bypassed AssertionRule can
# still reach that fallback -- see tests/test_test_harness.py's PAW-TEST-04 section --
# and it stays as the defense-in-depth backstop for that path.
_KNOWN_RULES = _REQUIRED_VALUE_RULES | {"regex_match"}

# PAW-TEST-01: yaml.safe_load already avoids instantiating arbitrary Python objects,
# but standard PyYAML places no limit on anchor/alias expansion -- a "YAML bomb"
# (a handful of nested aliases, each referencing the previous one several times) grows
# exponentially in memory during parsing regardless of which Loader class is used.
_MAX_YAML_ALIASES = 50
_MAX_YAML_BYTES = 1_000_000  # 1 MB

# H-1/H-2: the placeholder `paw_kit.test.runner.TestRunner.run` and
# `paw_kit.test.compare._infer_safely` substitute for a case whose `backend.infer`
# call raised. It lives here, in the leaf module both of those import from, so there
# is exactly one definition -- `TestSuiteConfig` below has to know it in order to
# refuse a suite that claims it as `abstain_value`, and `suite.py` cannot import from
# `runner.py` (runner imports suite; the reverse would be circular).
EXECUTION_ERROR_PLACEHOLDER = "[EXECUTION_ERROR]"


class _BoundedSafeLoader(yaml.SafeLoader):
    """A SafeLoader that aborts once too many alias-expansion events have occurred.

    Overriding `compose_node` (rather than e.g. `construct_*`) means this counts
    aliases as they're encountered during composition, before PyYAML has had a chance
    to actually expand any of them into the exponential in-memory structure a bomb
    relies on.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._alias_count = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.events.AliasEvent):
            self._alias_count += 1
            if self._alias_count > _MAX_YAML_ALIASES:
                raise ValueError(
                    f"YAML contains excessive alias expansions (> {_MAX_YAML_ALIASES}); "
                    "refusing to parse a suite that looks like a YAML bomb."
                )
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> Any:
        """Reject a mapping with a duplicate key (H-13).

        PyYAML inherits last-key-wins from the YAML spec's "recommended" handling, so a
        suite.yaml with two `standard_cases:` blocks parses cleanly and silently runs
        only the second -- the cases in the first are never executed and nothing says
        so. That is the same "no signal until you go read the per-case output" shape as
        `_validate_value`'s unknown-rule check, and it is worse here because the
        *count* looks plausible either way.

        Overridden on this existing subclass rather than registered as a new
        constructor: this loader already exists for the alias-bomb cap, and a second
        loader class would be one more thing to keep in sync with `load_suite`.
        """
        mapping = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hashable = key in mapping
            except TypeError:  # an unhashable key -- let the base class report it
                continue
            if hashable:
                raise ValueError(
                    f"Duplicate key {key!r} in the suite YAML (line "
                    f"{key_node.start_mark.line + 1}). YAML silently keeps only the last "
                    "one, so the earlier block would never run -- two `standard_cases:` "
                    "blocks means half the suite is dead. Merge them."
                )
            mapping.add(key)
        return super().construct_mapping(node, deep=deep)


class AssertionRule(BaseModel):
    """Rule constraining adapter generation outputs."""

    rule: str
    pattern: Optional[str] = None
    value: Optional[Any] = None

    @model_validator(mode="after")
    def _validate_value(self) -> "AssertionRule":
        """Fail fast at suite-load time instead of crashing (or silently mismatching) mid test-run."""
        if self.rule not in _KNOWN_RULES:
            raise ValueError(
                f"Unknown assertion rule {self.rule!r}; evaluate_assertion only implements "
                f"{sorted(_KNOWN_RULES)}. A typo here would otherwise load successfully and "
                "silently fail every case at eval time."
            )
        if self.rule in _REQUIRED_VALUE_RULES and self.value is None:
            raise ValueError(f"'{self.rule}' assertion requires a 'value' field")
        if self.rule in _NUMERIC_VALUE_RULES:
            try:
                int(self.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'{self.rule}' assertion value must be an integer, got {self.value!r}"
                ) from exc
        return self


class FuzzingConfig(BaseModel):
    """Adversarial fuzzer generation settings."""

    inject_unicode: bool = False
    empty_inputs: bool = False
    whitespace_flood: bool = False
    payload_extremes: bool = False
    adversarial_probes: List[str] = Field(default_factory=list)


# PAW-TEST-07: each active-learning iteration can trigger a full recompile and a
# round of teacher queries (often a paid API), so an unbounded max_iterations is a
# denial-of-service / denial-of-wallet risk, not just a slow suite.
_MAX_ACTIVE_LEARNING_ITERATIONS = 20
_DEFAULT_MAX_QUERIES_PER_ITERATION = 50


class ActiveLearningConfig(BaseModel):
    """Configuration for the teacher query and auto-recompilation loop."""

    auto_recompile: bool = True
    teacher_model: str = "claude-3-5-sonnet-20241022"
    max_iterations: int = 3
    # PAW-TEST-07: caps how many of an iteration's failing cases get queried against
    # the teacher -- Track 09's PAW-TEST-05 fix already made each individual query
    # injection-safe and label-validated; this bounds how *many* queries a single
    # iteration can rack up, independent of how many cases happen to be failing.
    max_queries_per_iteration: int = _DEFAULT_MAX_QUERIES_PER_ITERATION

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ActiveLearningConfig":
        """Fail fast at suite-load time, matching AssertionRule's own convention."""
        if self.max_iterations < 1:
            raise ValueError("active_learning.max_iterations must be at least 1")
        if self.max_iterations > _MAX_ACTIVE_LEARNING_ITERATIONS:
            raise ValueError(
                f"active_learning.max_iterations ({self.max_iterations}) exceeds the "
                f"maximum of {_MAX_ACTIVE_LEARNING_ITERATIONS} -- each iteration can "
                "trigger a full recompile plus a round of teacher queries (PAW-TEST-07)."
            )
        if self.max_queries_per_iteration < 1:
            raise ValueError("active_learning.max_queries_per_iteration must be at least 1")
        return self


class StandardTestCase(BaseModel):
    """In-distribution standard evaluation case."""

    input: str
    expected: Optional[str] = None


class TestSuiteConfig(BaseModel):
    """Root configuration model for paw.test suites."""

    __test__ = False

    task_name: str
    spec: str
    adapter_path: str
    standard_cases: List[StandardTestCase] = Field(default_factory=list)
    assertions: List[AssertionRule] = Field(default_factory=list)
    fuzzing: FuzzingConfig = Field(default_factory=FuzzingConfig)
    active_learning: ActiveLearningConfig = Field(default_factory=ActiveLearningConfig)
    # Track "Active-learning stuck signal" (conductor/deferred/index.md): the real
    # 2026-09-08 run's 11 failures were all whitespace/control-character garbage with no
    # legal ISO-date answer, so every teacher label was correctly rejected by the suite's
    # own date-shaped assertions -- there was no valid target to train toward. Setting
    # abstain_value gives a suite an explicit "I don't know" escape hatch: any assertion
    # (regardless of rule) passes automatically when the output equals this value exactly,
    # so a model can be trained to abstain on genuinely unanswerable input instead of
    # being forced to hallucinate a plausible-looking answer just to satisfy the suite.
    abstain_value: Optional[str] = None

    @model_validator(mode="after")
    def _reject_reserved_abstain_value(self) -> "TestSuiteConfig":
        """H-1/H-2 defence in depth: `abstain_value` may not claim the placeholder the
        runner substitutes for a case whose backend call raised.

        `TestRunner.run` and `compare._infer_safely` both write the literal
        `"[EXECUTION_ERROR]"` into `output` when `backend.infer` raises. A suite
        declaring that same string as its abstain value makes every crashed case look
        like a deliberate "I don't know" -- and it is reachable from an ordinary
        suite.yaml. Executed against pre-fix source: an all-raising backend with this
        abstain_value reported `Pass rate: 100.0% (10/10)` at exit 0.

        The run loop's precedence rule (errored beats abstained, decided on
        `execution_error is not None` rather than on the output string) is the actual
        fix and holds without this. This makes the placeholder un-claimable anyway, so
        the collision cannot be re-opened by a later refactor of that loop.

        Declared on the model rather than in `load_suite` so a directly-constructed
        `TestSuiteConfig` gets the same guarantee -- `run_active_learning_loop` and
        `compare_adapters` both accept configs that never went through the loader.
        """
        if self.abstain_value is not None and self.abstain_value == EXECUTION_ERROR_PLACEHOLDER:
            raise ValueError(
                f"abstain_value must not be {EXECUTION_ERROR_PLACEHOLDER!r}: that is the "
                "placeholder the test runner substitutes for a case whose backend call "
                "raised, and claiming it would make every crashed case read as a "
                "deliberate abstention. Choose any other string."
            )
        return self


def load_suite(path_or_yaml: Union[str, Path]) -> TestSuiteConfig:
    """Load and parse a test suite configuration from a YAML file or raw string.

    Args:
        path_or_yaml: Filesystem path to suite.yaml or raw YAML string content.

    Returns:
        Validated TestSuiteConfig instance.
    """
    content: str
    if isinstance(path_or_yaml, Path) or (
        isinstance(path_or_yaml, str) and "\n" not in path_or_yaml and Path(path_or_yaml).exists()
    ):
        with open(path_or_yaml, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = str(path_or_yaml)

    # PAW-TEST-01: applied here, after the two entry points above have already
    # converged on a single `content` string, so the cap covers both the file-path and
    # the raw-YAML-string branch alike -- a cap placed on only one of them (e.g. via
    # os.path.getsize before reading the file) is trivially bypassed via the other.
    content_bytes = len(content.encode("utf-8"))
    if content_bytes > _MAX_YAML_BYTES:
        raise ValueError(
            f"Suite YAML content ({content_bytes} bytes) exceeds the maximum of "
            f"{_MAX_YAML_BYTES} bytes; refusing to parse it."
        )

    parsed_data = yaml.load(content, Loader=_BoundedSafeLoader)  # PAW-TEST-01
    if not isinstance(parsed_data, dict):
        raise ValueError("Invalid suite specification: root must be a YAML mapping.")

    # Support 'adversarial_dates' alias from paper examples
    if "fuzzing" in parsed_data and isinstance(parsed_data["fuzzing"], dict):
        if "adversarial_dates" in parsed_data["fuzzing"] and "adversarial_probes" not in parsed_data["fuzzing"]:
            parsed_data["fuzzing"]["adversarial_probes"] = parsed_data["fuzzing"].pop("adversarial_dates")

    config = TestSuiteConfig.model_validate(parsed_data)

    # PAW-TEST-02: an untrusted suite.yaml's adapter_path eventually drives a write
    # (run_active_learning_loop -> backend.compile(..., output_path=adapter_path)) if
    # assertions fail and auto_recompile fires. Validated here -- at the suite loader,
    # not only in the CLI's own recompile-triggering path (paw_kit.cli's PAW-CLI-02
    # check) -- so any caller that loads a suite via load_suite() gets the same
    # guarantee, regardless of how it goes on to use the resulting config.
    ensure_contained(config.adapter_path, Path.cwd(), label="suite.yaml's adapter_path")

    return config
