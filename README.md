# paw-toolkit (PAW-Kit)

> **The Production Runtime & Reliability Toolkit for Program-as-Weights (PAW)**  
> *Translating natural language specifications into local, deterministic, zero-marginal-cost neural functions.*

Based on * Compile by Training: Turning Natural-Language Specifications into Local Neural Functions* (Deng et al., arXiv:2609.04199).

## Overview

paw-toolkit provides enterprise-grade developer tooling for Program-as-Weights:
- **paw.jit**: @compile_on_hit JIT decorator with automatic SQLite call-tracing and transparent background compilation.
- **paw.schema**: Grammar-constrained decoding for 0.0% Pydantic/JSON schema syntax errors.
- **paw.test**: Adversarial test runner with active-learning auto-repair loops.
