"""Parser and data models for declarative suite.yaml test specifications."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import yaml

_NUMERIC_VALUE_RULES = {"max_length", "min_length"}
_REQUIRED_VALUE_RULES = _NUMERIC_VALUE_RULES | {"exact_match", "not_contains"}


class AssertionRule(BaseModel):
    """Rule constraining adapter generation outputs."""

    rule: str
    pattern: Optional[str] = None
    value: Optional[Any] = None

    @model_validator(mode="after")
    def _validate_value(self) -> "AssertionRule":
        """Fail fast at suite-load time instead of crashing (or silently mismatching) mid test-run."""
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


class ActiveLearningConfig(BaseModel):
    """Configuration for the teacher query and auto-recompilation loop."""

    auto_recompile: bool = True
    teacher_model: str = "claude-3-5-sonnet-20241022"
    max_iterations: int = 3


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

    parsed_data = yaml.safe_load(content)
    if not isinstance(parsed_data, dict):
        raise ValueError("Invalid suite specification: root must be a YAML mapping.")

    # Support 'adversarial_dates' alias from paper examples
    if "fuzzing" in parsed_data and isinstance(parsed_data["fuzzing"], dict):
        if "adversarial_dates" in parsed_data["fuzzing"] and "adversarial_probes" not in parsed_data["fuzzing"]:
            parsed_data["fuzzing"]["adversarial_probes"] = parsed_data["fuzzing"].pop("adversarial_dates")

    return TestSuiteConfig.model_validate(parsed_data)
