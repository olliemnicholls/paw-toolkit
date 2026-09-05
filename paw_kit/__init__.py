"""paw-kit: Production Runtime & Reliability Toolkit for Program-as-Weights (PAW)."""

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.jit.decorator import compile_on_hit
from paw_kit.schema.exceptions import PAWSchemaError, PAWSyntaxError
from paw_kit.schema.grammar import pydantic_to_regex
from paw_kit.schema.loader import load
from paw_kit.schema.logits_processor import RegexLogitsProcessor
from paw_kit.test.active import ActiveLearningReport, run_active_learning_loop
from paw_kit.test.fuzzer import AdversarialFuzzer
from paw_kit.test.runner import TestRunReport, TestRunner, evaluate_assertion
from paw_kit.test.suite import (
    ActiveLearningConfig,
    AssertionRule,
    FuzzingConfig,
    StandardTestCase,
    TestSuiteConfig,
    load_suite,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "AbstractPAWBackend",
    "ActiveLearningConfig",
    "ActiveLearningReport",
    "AdversarialFuzzer",
    "AssertionRule",
    "BackgroundCompiler",
    "FuzzingConfig",
    "MockPAWBackend",
    "PAWSchemaError",
    "PAWSyntaxError",
    "RegexLogitsProcessor",
    "StandardTestCase",
    "TestRunReport",
    "TestRunner",
    "TestSuiteConfig",
    "TraceDB",
    "compile_on_hit",
    "evaluate_assertion",
    "load",
    "load_suite",
    "pydantic_to_regex",
    "run_active_learning_loop",
]
