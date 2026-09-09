"""paw-kit: reliability and migration harness for Program-as-Weights (PAW) neural functions."""

from paw_kit.backend.base import AbstractPAWBackend
from paw_kit.backend.mock import MockPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.jit.compiler import BackgroundCompiler
from paw_kit.jit.db import TraceDB
from paw_kit.jit.decorator import compile_on_hit
from paw_kit.schema.exceptions import PAWSchemaError, PAWSyntaxError
from paw_kit.schema.grammar import pydantic_to_regex
from paw_kit.schema.loader import load
from paw_kit.schema.logits_processor import RegexLogitsProcessor
from paw_kit.serve.docker import export_docker_scaffold
from paw_kit.serve.server import create_app, serve_adapter
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
    "ProgramAsWeightsBackend",
    "RegexLogitsProcessor",
    "StandardTestCase",
    "TestRunReport",
    "TestRunner",
    "TestSuiteConfig",
    "TraceDB",
    "compile_on_hit",
    "create_app",
    "evaluate_assertion",
    "export_docker_scaffold",
    "load",
    "load_suite",
    "pydantic_to_regex",
    "run_active_learning_loop",
    "serve_adapter",
]
