#!/usr/bin/env python3
"""Mutation testing for paw_kit: does the suite actually notice when behaviour changes?

Line coverage says a line ran. It does not say an assertion would have failed had the
line been wrong. This tool answers the second question: it makes a small, definitely
semantic change to the source (flip a comparison operator, swap and/or, negate a bool,
bump an int), runs the tests, and records whether anything failed.

    SURVIVED  -- the suite passed with the mutated code. The behaviour on that line is
                 unasserted. This is a finding.
    KILLED    -- some test failed, AND that failure was attributed to the mutant (see
                 `confirm_kill`). That line's behaviour is pinned by the suite.

    Two further statuses both count as survivors, via `is_survivor()`, because neither is
    evidence that the mutant was detected:

    SURVIVED_TIMEOUT -- the run did not finish. Not a kill: "did not finish" says nothing
                 about detection. Reported on its own line, and a hard refusal to write a
                 baseline, since a timed-out mutant's true status is unknown.
    SURVIVED (unattributed) -- the suite failed but the failing test could not be parsed out
                 of pytest's summary, so `confirm_kill`'s two runs could not be performed.
                 Carries a note saying the kill was not verified.

See tools/README-mutate.md for what to do with survivors and how to regenerate the
recorded baseline.

THE CONTROL GATE -- read this before changing anything in this file
-------------------------------------------------------------------
Every mutant is produced by `ast.parse` -> mutate one node -> `ast.unparse`, so a mutant
file differs from the original in two ways: the intended semantic change, and the
incidental reformatting that `ast.unparse` does to the whole file. Before any mutant is
run, this tool writes an unparse round-trip of each target file with *no* semantic change
and requires that file's tests to pass. That is the control.

This is not ceremony. The first version of this harness reported 548 mutations and a
100% kill rate in 20 seconds, which is impossible -- it was passing `--timeout=60` to a
pytest without `pytest-timeout` installed, so every run exited rc=4 (usage error) and
every rc!=0 was being scored as "killed". A harness that cannot run the tests at all
reports perfect test quality. The control runs are what caught it, which is why:

  * control failure is a HARD ABORT of the whole run, not a warning;
  * stdout *and* stderr of a failing control are printed in full;
  * any exit code other than 0 (pass) or 1 (tests failed) is scored as ERROR and
    reported separately -- never silently folded into "killed".

A suspiciously high kill rate is the known failure mode of this kind of tool. Distrust it.

Kills are attributed, not assumed. A phase-2 run executes the whole suite, so any failure
anywhere would otherwise score as a kill -- including one the mutant cannot have caused.
See confirm_kill() below; without it this tool reported a regression on an unchanged tree.

Provenance
----------
`paw-toolkit/.venv` contains `_editable_impl_paw_kit.pth` hardcoding the MAIN checkout's
absolute path, so a stray `import paw_kit` from a git worktree can load the main
checkout's source instead. Mutants would then be applied to a file nobody imports and
every mutant would "survive"... or, worse, the reverse. So before the controls run, this
tool asserts that `paw_kit` imported inside a mutant copy resolves inside that copy.

Usage
-----
    uv run --no-sync python tools/mutate.py                       # full default set
    uv run --no-sync python tools/mutate.py --modules runner.py   # one module
    uv run --no-sync python tools/mutate.py --baseline tools/mutation-baseline.json
    uv run --no-sync python tools/mutate.py --write-baseline tools/mutation-baseline.json

Exit codes
----------
    0  ran cleanly; if --baseline was given, no module regressed
    1  REGRESSION: a targeted module has more survivors than the baseline records
       (or, with --strict-identity, a survivor the baseline does not know about)
    2  CONTROL GATE FAILED or provenance failed -- results are meaningless, nothing ran
    3  harness errors (unexpected pytest exit codes) -- results are untrustworthy

    A run containing a TIMEOUT does NOT set a non-zero exit on its own: the timeout is
    already counted as a survivor, which makes the gate conservative, and failing the run
    outright would block merges on a slow or loaded machine. It prints a WARNING naming each
    timed-out mutant, and it refuses --write-baseline. If you lowered --timeout, raise it and
    re-run for a decisive answer. Note that phase-1 timeouts now flow into phase 2, one
    full-suite run each, so a too-low --timeout on a large module costs real time.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue

# --------------------------------------------------------------------------------------
# What gets mutated
# --------------------------------------------------------------------------------------
# `tests` is the fast per-module selection used for phase 1. A mutant that survives it is
# re-run against the FULL suite in phase 2, because a test outside the selection may still
# kill it; only full-suite survivors are counted. `skip_kinds` drops a mutation operator
# that produces mostly noise for that module. `default` is whether the module is in the
# set that the recorded baseline covers.


@dataclass(frozen=True)
class Target:
    tests: str
    default: bool = True
    skip_kinds: tuple[str, ...] = ()
    note: str = ""


TARGETS: dict[str, Target] = {
    # --- the default set: the modules where a wrong answer is silent -------------------
    "paw_kit/test/matching.py": Target(
        "tests/test_matching.py tests/test_test_harness.py tests/test_compare.py"
    ),
    "paw_kit/test/runner.py": Target("tests/test_test_harness.py tests/test_cli.py"),
    "paw_kit/test/compare.py": Target("tests/test_compare.py tests/test_cli.py"),
    "paw_kit/test/active.py": Target("tests/test_test_harness.py"),
    "paw_kit/schema/grammar.py": Target(
        "tests/test_schema.py",
        # grammar.py's integer literals are overwhelmingly regex/quantifier/cap tuning
        # constants whose exact value is a tuning choice, not asserted behaviour. Mutating
        # them produced 48 mutants that say nothing useful about test quality.
        skip_kinds=("int",),
    ),
    "paw_kit/schema/logits_processor.py": Target("tests/test_schema.py"),
    # --- available via --modules, NOT in the baseline ---------------------------------
    "paw_kit/jit/agreement.py": Target(
        "tests/test_jit_agreement.py tests/test_jit_shadow.py", default=False
    ),
    "paw_kit/jit/shadow.py": Target(
        "tests/test_jit_shadow.py tests/test_jit.py tests/test_jit_persistence.py "
        "tests/test_jit_fail_open.py tests/test_jit_deadline_integration.py "
        "tests/test_jit_shadow_raw_values.py tests/test_jit_shadow_drops.py",
        default=False,
        note="slow (~23s per run) and shares the known test_jit_shadow flake. Track C "
        "(bug-hunt-remediation) added the four tests/test_jit_*.py files after "
        "test_jit_persistence.py above: without them a shadow.py run misses every "
        "J-2/J-4/J-10/M-4b test and reports survivors those tests do in fact kill "
        "(confirmed by execution: 3 mutants -- the pool-size constant, and two "
        "drop-warning boundary conditions -- surfaced as false regressions the first "
        "time this track ran the gate before this line was updated).",
    ),
    "paw_kit/jit/decorator.py": Target(
        "tests/test_jit.py tests/test_jit_shadow.py tests/test_jit_persistence.py "
        "tests/test_jit_fail_open.py tests/test_jit_deadline_integration.py "
        "tests/test_jit_shadow_raw_values.py tests/test_jit_shadow_drops.py",
        default=False,
        note="M-4 (the ~1-in-12 test_jit_shadow flake that made this module's control gate "
        "intermittent) was fixed on track-instruments. Left out of the default set and "
        "the baseline until a clean multi-run measurement is recorded; re-enable with "
        "--modules paw_kit/jit/decorator.py. Track C added the four tests/test_jit_*.py "
        "files after test_jit_persistence.py above, for the same reason as shadow.py's "
        "entry just above.",
    ),
    "paw_kit/jit/db.py": Target(
        "tests/test_jit.py tests/test_jit_shadow.py tests/test_jit_persistence.py "
        "tests/test_jit_fail_open.py tests/test_jit_deadline_integration.py "
        "tests/test_jit_shadow_raw_values.py",
        default=False,
        note="slow (~37s per run). tests/test_jit_persistence.py joins the selection "
        "for Track G: without it a db.py run misses every D-1/D-5/D-7/D-9 test and "
        "reports survivors those tests do in fact kill. Track C added the three "
        "tests/test_jit_*.py files after it, for J-1's real DB-fault injection and "
        "J-2/J-10's pool_exhausted/teacher_error exclusion in get_agreement_stats.",
    ),
    "paw_kit/atomicio.py": Target(
        "tests/test_cli.py tests/test_jit.py tests/test_jit_persistence.py",
        default=False,
        note="added for Track G (D-3/D-8); had no TARGETS entry at all before, so "
        "--modules paw_kit/atomicio.py used to hard-error in resolve_modules.",
    ),
    "paw_kit/jit/compiler.py": Target(
        "tests/test_jit.py tests/test_jit_persistence.py",
        default=False,
        note="added for Track D (D-2, J-5, J-6): trigger_compilation's cross-process "
        "compiling-guard and sync_compile's duplicate-compile guard both live here.",
    ),
    "paw_kit/backend/programasweights.py": Target(
        "tests/test_programasweights_backend.py",
        default=False,
        note="added for Track D (A-1, A-2, A-3, A-6, A-9): the 5xx-retry guard, the "
        "manifest's requested-vs-confirmed public field, the precheck contract and "
        "the folded-example count all live here.",
    ),
    "paw_kit/backend/manifest_lineage.py": Target(
        "tests/test_manifest_lineage.py tests/test_programasweights_backend.py "
        "tests/test_mock_backend.py",
        default=False,
        note="added for Track D (D-4): the history-sidecar field filter lives here, "
        "and both shipped backends' lineage writes go through it, hence both test "
        "files -- a mock-only fix would miss the real backend's own filter call. "
        "tests/test_manifest_lineage.py joins the selection as Track D implements D-4: "
        "it did not exist when this entry was written and it holds *every* one of D-4's "
        "tests (allow-list, evolution guard, rotation, concurrent-append-across-rotation, "
        "fsync scoping), so without it a run here misses all of them and reports "
        "survivors those tests do in fact kill -- the same trap Track G recorded for "
        "db.py and tests/test_jit_persistence.py.",
    ),
    "paw_kit/serve/server.py": Target(
        "tests/test_serve.py",
        default=False,
        note="added for Track F (X-1..X-9): auth ordering, middleware stack order, "
        "the rate limiter's keying and eviction, and the inference-slot admission "
        "path all live here.",
    ),
    "paw_kit/serve/docker.py": Target(
        "tests/test_serve.py",
        default=False,
        note="added for Track F (X-10): the generated requirements.txt pin list and "
        "the base-image tags live here. Not exercised through test_cli.py -- the "
        "CLI's `export docker` has no dedicated test file of its own; the exporter "
        "is tested directly via test_serve.py.",
    ),
}

FULL_SUITE = ""  # empty selection == run everything

# Directories never copied into a mutant workspace. __pycache__ is excluded deliberately:
# see _ENV below.
COPY_IGNORE = shutil.ignore_patterns(
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    "htmlcov", ".coverage", "dist", "build", "*.egg-info",
)

# --------------------------------------------------------------------------------------
# Mutation operators
# --------------------------------------------------------------------------------------

CMP_SWAP = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}
CMP_NAME = {
    ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
    ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in", ast.NotIn: "not in",
}
INT_LIMIT = 10_000  # don't mutate magic-looking huge literals (hashes, seeds, ports)
SNIPPET = 100


def _snip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= SNIPPET else text[: SNIPPET - 1] + "…"


def collect(src: str, skip_kinds: tuple[str, ...] = ()) -> list[dict]:
    """Enumerate the mutations available in `src`.

    Each mutation is identified by (kind, line, col) -- the position of the node in the
    ORIGINAL source -- plus the old and new token. `expr_before`/`expr_after` are the
    unparsed node either side of the mutation, and `source_line` the original line; those
    three are what lets a later run say *which* mutant newly survived rather than only
    how many did.
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    out: list[dict] = []

    def record(kind, node, old, new, mutated_node):
        if kind in skip_kinds:
            return
        out.append(
            {
                "kind": kind,
                "line": node.lineno,
                "col": node.col_offset,
                "old": old,
                "new": new,
                "expr_before": _snip(ast.unparse(node)),
                "expr_after": _snip(ast.unparse(mutated_node)),
                "source_line": _snip(lines[node.lineno - 1]) if node.lineno <= len(lines) else "",
            }
        )

    class Collector(ast.NodeVisitor):
        def visit_Compare(self, node):
            if len(node.ops) == 1 and type(node.ops[0]) in CMP_SWAP:
                op = type(node.ops[0])
                mutated = copy.deepcopy(node)
                mutated.ops = [CMP_SWAP[op]()]
                record("cmp", node, CMP_NAME[op], CMP_NAME[CMP_SWAP[op]], mutated)
            self.generic_visit(node)

        def visit_BoolOp(self, node):
            old = "and" if isinstance(node.op, ast.And) else "or"
            mutated = copy.deepcopy(node)
            mutated.op = ast.Or() if old == "and" else ast.And()
            record("bool", node, old, "or" if old == "and" else "and", mutated)
            self.generic_visit(node)

        def visit_Constant(self, node):
            if isinstance(node.value, bool):
                mutated = copy.deepcopy(node)
                mutated.value = not node.value
                record("const", node, str(node.value), str(not node.value), mutated)
            elif isinstance(node.value, int) and abs(node.value) < INT_LIMIT:
                mutated = copy.deepcopy(node)
                mutated.value = node.value + 1
                record("int", node, str(node.value), str(node.value + 1), mutated)
            self.generic_visit(node)

    Collector().visit(tree)
    return out


def apply_mutation(src: str, mut: dict) -> str | None:
    """Return `src` with `mut` applied, or None if the node could not be found.

    The whole file goes through ast.unparse, which is why the control gate exists.
    """
    kind, line, col, old = mut["kind"], mut["line"], mut["col"], mut["old"]
    tree = ast.parse(src)
    done: list[int] = []

    class T(ast.NodeTransformer):
        def visit_Compare(self, node):
            self.generic_visit(node)
            if kind == "cmp" and node.lineno == line and node.col_offset == col:
                op = type(node.ops[0])
                if CMP_NAME.get(op) == old:
                    node.ops = [CMP_SWAP[op]()]
                    done.append(1)
            return node

        def visit_BoolOp(self, node):
            self.generic_visit(node)
            if kind == "bool" and node.lineno == line and node.col_offset == col:
                node.op = ast.Or() if old == "and" else ast.And()
                done.append(1)
            return node

        def visit_Constant(self, node):
            if kind in ("const", "int") and node.lineno == line and node.col_offset == col:
                if kind == "const" and isinstance(node.value, bool) and str(node.value) == old:
                    node.value = not node.value
                    done.append(1)
                elif (
                    kind == "int"
                    and isinstance(node.value, int)
                    and not isinstance(node.value, bool)
                    and str(node.value) == old
                ):
                    node.value = node.value + 1
                    done.append(1)
            return node

    new_tree = T().visit(tree)
    if not done:
        return None
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def mut_id(module: str, mut: dict) -> str:
    return f"{module}:{mut['line']}:{mut['col']}:{mut['kind']}:{mut['old']}>{mut['new']}"


def mut_fingerprint(module: str, mut: dict) -> str:
    """Line-independent identity, so a survivor is still recognised after the file moves.

    Used as a fallback when the exact id does not match, which happens whenever unrelated
    edits shift line numbers.
    """
    return f"{module}|{mut['kind']}|{mut['old']}>{mut['new']}|{mut['expr_before']}"


# --------------------------------------------------------------------------------------
# Running tests in isolated copies
# --------------------------------------------------------------------------------------

# PYTHONDONTWRITEBYTECODE: a .pyc is invalidated on (source mtime truncated to seconds,
# source size). An `int` mutation n -> n+1 usually changes neither, so successive mutants
# of one file inside one second can hit a stale .pyc -- the unmutated code runs, the tests
# pass, and the mutant is scored SURVIVED. Byte-compilation is therefore off, and
# __pycache__ is not copied into the workspaces.
_ENV_BASE = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


@dataclass
class Outcome:
    rc: object
    tail: str
    stdout: str = ""
    stderr: str = ""


@dataclass
class Runner:
    python: str
    timeout: int
    full_timeout: int
    extra: list[str] = field(default_factory=list)

    def run(self, wd: pathlib.Path, tests: str) -> Outcome:
        full = not tests.strip()
        cmd = [self.python, "-m", "pytest"]
        if tests.strip():
            cmd += tests.split()
        # -rf: the short failure summary names the killing test, which the phase-2
        # kill confirmation below needs. Reporting flags only; no effect on verdicts.
        cmd += ["-x", "-q", "-rf", "-p", "no:randomly", *self.extra]
        env = {**os.environ, **_ENV_BASE, "PYTHONPATH": str(wd)}
        try:
            r = subprocess.run(
                cmd, cwd=wd, capture_output=True, text=True, env=env,
                timeout=self.full_timeout if full else self.timeout,
            )
        except subprocess.TimeoutExpired as e:
            # A hung *process* is not the same thing as an *undecided* test result.
            # Some mutations (e.g. a `daemon=True` flag flipped to `False` on a
            # worker thread) make pytest report a definitive result -- including a
            # kill -- and only then hang forever at interpreter shutdown, joining a
            # thread nothing will ever finish, long after the outcome was already
            # decided. `subprocess.run`'s `TimeoutExpired` carries whatever
            # stdout/stderr had already been captured before the process was
            # killed (verified by execution: a pytest run that fails in 0.5s and
            # then hangs still has "1 failed" sitting in `e.stdout` at the 10s
            # mark) -- as *bytes*, even though this call passes `text=True`; that
            # decoding happens on the success path inside `subprocess.run` itself,
            # not on this exception, so it must be done here too.
            #
            # A definitive "N failed" line is real evidence of a kill, not a guess
            # -- unlike a genuine timeout with no result recorded at all, which
            # still returns SURVIVED_TIMEOUT exactly as before.
            def _decode(b: object) -> str:
                if isinstance(b, bytes):
                    return b.decode("utf-8", errors="replace")
                return b or ""

            stdout, stderr = _decode(e.stdout), _decode(e.stderr)
            combined = stdout + stderr
            if re.search(r"^\d+ failed", combined, re.M):
                tail = combined.strip().splitlines()[-1] if combined.strip() else ""
                return Outcome(1, tail, stdout, stderr)
            return Outcome("TIMEOUT", "timed out", stdout, stderr)
        combined = (r.stdout or "") + (r.stderr or "")
        tail = combined.strip().splitlines()[-1] if combined.strip() else ""
        return Outcome(r.returncode, tail, r.stdout or "", r.stderr or "")


FAILED_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.M)


def confirm_kill(wd: pathlib.Path, path: pathlib.Path, mutant_src: str, orig_src: str,
                 out: Outcome, runner: Runner) -> tuple[str, str]:
    """Decide whether a full-suite failure is really attributable to the mutant.

    WHY THIS EXISTS. A phase-2 run executes the whole suite with one mutated module, so
    *any* failure anywhere scores as a kill -- including a failure the mutant cannot
    possibly have caused. Measured: running this tool twice on an unmutated tree gave
    compare.py 16 survivors and then 18, and the recorded baseline had a `paw_kit/test/
    compare.py` mutant "killed" by a test in a subsystem it does not touch. With six
    suites running in parallel, order- and contention-sensitive tests fail occasionally;
    each such failure silently converts a survivor into a kill. A baseline built from
    those is a false-alarm generator: the gate then reports a REGRESSION on a tree nobody
    changed, which is exactly what happened before this function existed.

    So a kill must be attributable. Two cheap single-test runs:

      1. Re-run just the failing test(s) with the mutant still applied. If they pass, the
         full-suite failure was not caused by the mutant -> SURVIVED.
      2. Re-run them with the mutant removed. If they fail there too, the test is flaky or
         broken independently of the mutant -> SURVIVED, and said so loudly.

    Only "fails with the mutant, passes without it" is a kill. The bias is deliberately
    towards SURVIVED: over-reporting survivors makes the ratchet conservative, whereas
    over-reporting kills makes the gate fire on innocent branches.
    """
    ids = list(dict.fromkeys(FAILED_RE.findall((out.stdout or "") + (out.stderr or ""))))[:5]
    if not ids:
        # Unattributable: the suite failed but no test name could be parsed out, so the
        # two confirmation runs below cannot be performed. Scoring this KILLED would be
        # exactly the unverified kill this function exists to prevent, so it is a survivor
        # with a loud note. (Phase F review, 2026-09-11.)
        return "SURVIVED", "killing test not named in output; kill NOT verified, counted as survived"
    sel = " ".join(ids)
    mutated = runner.run(wd, sel)
    if mutated.rc == 0:
        return "SURVIVED", f"full-suite failure did not reproduce in isolation ({sel})"
    if mutated.rc != 1:
        return "ERROR", f"confirmation run exited rc={mutated.rc} on {sel}"
    path.write_text(orig_src)
    try:
        clean = runner.run(wd, sel)
    finally:
        path.write_text(mutant_src)
    if clean.rc == 0:
        return "KILLED", f"confirmed: {sel} passes unmutated, fails mutated"
    return "SURVIVED", (f"FLAKY TEST: {sel} fails unmutated too (rc={clean.rc}); "
                        f"this kill is not attributable to the mutant")


def is_survivor(status: str) -> bool:
    """True for every status that must be counted as a survivor.

    A single predicate on purpose: there is more than one way to fail to kill a mutant
    (the suite passed; the kill could not be attributed; the run timed out), and an
    `== "SURVIVED"` comparison at each call site silently dropped the others from the
    survivor list -- which under-counts survivors and lets the gate pass. Added after the
    Wave 0 Phase F review.
    """
    return status.startswith("SURVIVED")


def classify(rc: object) -> str:
    """pytest exit codes: 0 ok, 1 tests failed, 2 interrupted, 3 internal, 4 usage, 5 none
    collected. Only 0 means survived and only 1 means killed; everything else is a broken
    harness and must be visible."""
    if rc == "TIMEOUT":
        # NOT a kill. A timeout says "this run did not finish", which is not evidence that
        # the mutant was detected -- and `run_phase` never routes it through `confirm_kill`,
        # so it would be an unattributed kill. Scoring it KILLED let one slow run erase a
        # genuine survivor, report "kill rate 100% / improved", and exit 0 (demonstrated in
        # the Wave 0 Phase F review with `--timeout 1`). Counted as a survivor, and tracked
        # separately so the run says plainly that it happened.
        return "SURVIVED_TIMEOUT"
    if rc == 0:
        return "SURVIVED"
    if rc == 1:
        return "KILLED"
    return "ERROR"


def assert_provenance(wd: pathlib.Path, python: str) -> None:
    """Refuse to run if `paw_kit` does not import from inside the mutant copy."""
    env = {**os.environ, **_ENV_BASE, "PYTHONPATH": str(wd)}
    r = subprocess.run(
        [python, "-c", "import paw_kit, os; print(os.path.realpath(paw_kit.__file__))"],
        cwd=wd, capture_output=True, text=True, env=env, timeout=120,
    )
    got = (r.stdout or "").strip().splitlines()[-1] if (r.stdout or "").strip() else ""
    if not got:
        die(2, f"provenance check could not import paw_kit in {wd}\n{r.stderr}")
    root = str(wd.resolve())
    if not got.startswith(root + os.sep):
        die(
            2,
            "PROVENANCE FAILURE: paw_kit resolved to\n"
            f"  {got}\n"
            f"which is outside the mutant workspace\n  {root}\n"
            "Mutants would be applied to a file nobody imports. Refusing to run.",
        )


def die(code: int, msg: str) -> None:
    print(f"\nmutate.py: {msg}", file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------------------


def control_gate(workers: list[pathlib.Path], modules: list[str], runner: Runner,
                 recheck: bool) -> dict:
    """Unparse round-trip with NO semantic change; the suite MUST still pass.

    Aborts the entire run on any failure. Do not downgrade this to a warning.
    """
    wd = workers[0]
    controls: dict[str, dict] = {}
    failures: list[tuple[str, Outcome]] = []

    for rel in modules:
        p = wd / rel
        orig = p.read_text()
        try:
            p.write_text(ast.unparse(ast.parse(orig)))
            out = runner.run(wd, TARGETS[rel].tests)
        finally:
            p.write_text(orig)
        controls[rel] = {"rc": out.rc, "tail": out.tail}
        status = "OK" if out.rc == 0 else "FAIL"
        print(f"  control {status:4} {rel}: rc={out.rc}  {out.tail}", flush=True)
        if out.rc != 0:
            failures.append((rel, out))

    if recheck:
        # The phase-2 verdict comes from a full-suite run, so that invocation needs its own
        # control. This is the run that caught the rc=4 usage error.
        originals = {rel: (wd / rel).read_text() for rel in modules}
        try:
            for rel, orig in originals.items():
                (wd / rel).write_text(ast.unparse(ast.parse(orig)))
            out = runner.run(wd, FULL_SUITE)
        finally:
            for rel, orig in originals.items():
                (wd / rel).write_text(orig)
        controls["<full suite>"] = {"rc": out.rc, "tail": out.tail}
        print(f"  control {'OK' if out.rc == 0 else 'FAIL':4} <full suite>: "
              f"rc={out.rc}  {out.tail}", flush=True)
        if out.rc != 0:
            failures.append(("<full suite>", out))

    if failures:
        msg = [
            "CONTROL GATE FAILED -- aborting.",
            "",
            "A semantics-preserving ast.unparse round-trip did not pass the suite, so this",
            "run cannot distinguish a mutant the tests caught from a harness that cannot",
            "run the tests at all. Every number it would print would be fiction.",
            "",
            "Causes seen in practice: an unsupported pytest flag (rc=4 usage error), a",
            "missing pytest plugin, no tests collected (rc=5), or a genuinely flaky test in",
            "the selection (see paw_kit/jit/decorator.py's note in TARGETS).",
            "",
        ]
        for rel, out in failures:
            msg += [f"--- {rel}: rc={out.rc} " + "-" * 40, "",
                    "stdout:", out.stdout.strip()[-4000:] or "(empty)", "",
                    "stderr:", out.stderr.strip()[-4000:] or "(empty)", ""]
        die(2, "\n".join(msg))
    return controls


def run_phase(label: str, jobs: list[dict], workers: list[pathlib.Path], runner: Runner,
              tests_for, progress_every: int, confirm: bool = False) -> list[dict]:
    pool: Queue = Queue()
    for w in workers:
        pool.put(w)
    t0 = time.time()
    results: list[dict] = []

    def work(job: dict) -> dict:
        wd = pool.get()
        try:
            p = wd / job["module"]
            orig = p.read_text()
            try:
                newsrc = apply_mutation(orig, job)
                if newsrc is None:
                    return {**job, "status": "NOT_APPLIED", "tail": "node not found"}
                p.write_text(newsrc)
                out = runner.run(wd, tests_for(job))
                status, note = classify(out.rc), ""
                if confirm and status == "KILLED":
                    status, note = confirm_kill(wd, p, newsrc, orig, out, runner)
                return {**job, "status": status, "rc": str(out.rc), "tail": out.tail,
                        "note": note,
                        "stderr": out.stderr.strip()[-2000:] if status == "ERROR" else ""}
            finally:
                p.write_text(orig)
        finally:
            pool.put(wd)

    with ThreadPoolExecutor(max_workers=len(workers)) as ex:
        for i, r in enumerate(ex.map(work, jobs)):
            results.append(r)
            if progress_every and (i + 1) % progress_every == 0:
                surv = sum(1 for x in results if is_survivor(x["status"]))
                print(f"  {label} {i + 1}/{len(jobs)}  survived={surv}  "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"  {label} done in {time.time() - t0:.0f}s", flush=True)
    return results


# --------------------------------------------------------------------------------------
# Baseline comparison
# --------------------------------------------------------------------------------------


def build_baseline(repo: pathlib.Path, modules: list[str], per_module: dict) -> dict:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:
        commit = ""
    return {
        "schema": 1,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": commit,
        "note": (
            "Surviving mutants per module, measured by tools/mutate.py. A survivor is a "
            "semantic change the suite does not notice. Counts are full-suite survivors: "
            "a mutant that survives its module's fast test selection is re-run against "
            "the whole suite, and only then counted. Regenerate with "
            "`uv run --no-sync python tools/mutate.py --write-baseline "
            "tools/mutation-baseline.json`; see tools/README-mutate.md."
        ),
        "totals": {
            "mutations": sum(m["mutations"] for m in per_module.values()),
            "survivors": sum(m["survivors"] for m in per_module.values()),
        },
        "modules": {rel: per_module[rel] for rel in modules},
    }


def compare_baseline(baseline: dict, modules: list[str], per_module: dict,
                     strict_identity: bool) -> bool:
    """Print the comparison; return True if it is a regression."""
    base_modules = baseline.get("modules", {})
    regressed = False
    print("\n--- baseline comparison " + "-" * 46)
    for rel in modules:
        now = per_module[rel]
        base = base_modules.get(rel)
        if base is None:
            print(f"  {rel}: {now['survivors']} survivors (NOT IN BASELINE -- advisory only)")
            continue
        delta = now["survivors"] - base["survivors"]
        flag = "REGRESSION" if delta > 0 else ("improved" if delta < 0 else "ok")
        if delta > 0:
            regressed = True
        print(f"  {rel}: {now['survivors']} survivors vs baseline "
              f"{base['survivors']}  ({delta:+d})  {flag}")

        known_ids = {s["id"] for s in base.get("survivors_detail", [])}
        known_fps = {s.get("fingerprint") for s in base.get("survivors_detail", [])}
        new = [s for s in now.get("survivors_detail", [])
               if s["id"] not in known_ids and s.get("fingerprint") not in known_fps]
        for s in new:
            print(f"      NEW survivor  line {s['line']}  {s['kind']} "
                  f"{s['old']} -> {s['new']}   {s['source_line']}")
        if new and strict_identity:
            regressed = True
        gone_ids = {s["id"] for s in now.get("survivors_detail", [])}
        gone_fps = {s.get("fingerprint") for s in now.get("survivors_detail", [])}
        for s in base.get("survivors_detail", []):
            if s["id"] not in gone_ids and s.get("fingerprint") not in gone_fps:
                print(f"      now killed    line {s['line']}  {s['kind']} "
                      f"{s['old']} -> {s['new']}   {s.get('source_line', '')}")
    print("-" * 70)
    return regressed


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def resolve_modules(requested: list[str] | None) -> list[str]:
    if not requested:
        return [rel for rel, t in TARGETS.items() if t.default]
    out: list[str] = []
    for want in requested:
        want = want.strip()
        hits = [rel for rel in TARGETS if rel == want or rel.endswith("/" + want.lstrip("/"))]
        if not hits:
            die(2, f"unknown module {want!r}. Known:\n  " + "\n  ".join(TARGETS))
        if len(hits) > 1:
            die(2, f"ambiguous module {want!r}: {hits}")
        if hits[0] not in out:
            out.append(hits[0])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repo", default=None,
                    help="repo root to mutate (default: parent of this file's directory)")
    ap.add_argument("--modules", nargs="+", metavar="PATH",
                    help="modules to mutate; a bare filename is matched by suffix. "
                         "Default: every module marked default in TARGETS.")
    ap.add_argument("--workers", type=int, default=6,
                    help="parallel mutant workspaces / pytest processes (default 6)")
    ap.add_argument("--baseline", metavar="PATH",
                    help="compare against this baseline and exit 1 if a targeted module "
                         "has more survivors than it records (this is the campaign gate)")
    ap.add_argument("--write-baseline", metavar="PATH",
                    help="write the measured result as a new baseline file")
    ap.add_argument("--strict-identity", action="store_true",
                    help="also fail when a survivor appears that the baseline does not "
                         "know about, even if the count did not rise")
    ap.add_argument("--no-recheck", action="store_true",
                    help="skip the phase-2 full-suite recheck of survivors. Faster, but "
                         "over-reports: some survivors of a module's own tests are killed "
                         "by a test elsewhere. Not comparable to the recorded baseline.")
    ap.add_argument("--kinds", nargs="+", choices=["cmp", "bool", "const", "int"],
                    help="restrict to these mutation operators")
    ap.add_argument("--timeout", type=int, default=180, help="per-run timeout, phase 1")
    ap.add_argument("--full-timeout", type=int, default=600, help="per-run timeout, phase 2")
    ap.add_argument("--python", default=sys.executable, help="interpreter for pytest")
    ap.add_argument("--workdir", default=None,
                    help="where the mutant copies live (default: a temp dir, removed after)")
    ap.add_argument("--keep-workdir", action="store_true", help="do not delete --workdir")
    ap.add_argument("--json", metavar="PATH", help="write the full per-mutant result here")
    ap.add_argument("--list", action="store_true",
                    help="print the mutation plan and exit without running anything")
    args = ap.parse_args(argv)

    repo = pathlib.Path(args.repo).resolve() if args.repo \
        else pathlib.Path(__file__).resolve().parent.parent
    if not (repo / "paw_kit").is_dir():
        die(2, f"{repo} does not look like the paw-toolkit repo (no paw_kit/)")
    modules = resolve_modules(args.modules)

    # ---- plan ----------------------------------------------------------------------
    jobs: list[dict] = []
    per_module: dict[str, dict] = {}
    for rel in modules:
        t = TARGETS[rel]
        src = (repo / rel).read_text()
        muts = collect(src, t.skip_kinds)
        if args.kinds:
            muts = [m for m in muts if m["kind"] in args.kinds]
        for m in muts:
            jobs.append({**m, "module": rel, "id": mut_id(rel, m),
                         "fingerprint": mut_fingerprint(rel, m)})
        per_module[rel] = {"mutations": len(muts), "survivors": 0, "survivors_detail": []}
        extra = f"   [{t.note}]" if t.note else ""
        print(f"  {rel}: {len(muts)} mutations"
              + (f" (skipping {', '.join(t.skip_kinds)})" if t.skip_kinds else "")
              + extra)
    print(f"\nrepo      {repo}")
    print(f"modules   {len(modules)}")
    print(f"mutations {len(jobs)}")
    print(f"workers   {args.workers}")
    print(f"recheck   {'no (NOT baseline-comparable)' if args.no_recheck else 'full suite'}")
    if args.list:
        return 0
    if not jobs:
        die(2, "no mutations to run")

    # ---- workspaces ----------------------------------------------------------------
    tmp = None
    if args.workdir:
        workdir = pathlib.Path(args.workdir).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.mkdtemp(prefix="paw-mutate-")
        workdir = pathlib.Path(tmp)
    print(f"\npreparing {args.workers} mutant copies in {workdir} ...", flush=True)
    workers: list[pathlib.Path] = []
    for i in range(args.workers):
        d = workdir / f"w{i}"
        if not d.exists():
            shutil.copytree(repo, d, symlinks=True, ignore=COPY_IGNORE)
        workers.append(d)

    rc_final = 0
    try:
        runner = Runner(args.python, args.timeout, args.full_timeout)

        # ---- provenance, then the control gate -------------------------------------
        print("\nasserting provenance ...", flush=True)
        assert_provenance(workers[0], args.python)
        print("  paw_kit imports from inside the mutant copy: OK")

        print("\ncontrol gate (ast.unparse round-trip, no semantic change -- MUST pass):",
              flush=True)
        controls = control_gate(workers, modules, runner, not args.no_recheck)

        # ---- phase 1: the module's own tests ---------------------------------------
        print(f"\nphase 1: {len(jobs)} mutants against their module's test selection",
              flush=True)
        p1 = run_phase("p1", jobs, workers, runner,
                       lambda j: TARGETS[j["module"]].tests, 25)

        errors = [r for r in p1 if r["status"] == "ERROR"]
        not_applied = [r for r in p1 if r["status"] == "NOT_APPLIED"]
        survivors = [r for r in p1 if is_survivor(r["status"])]
        # Every timeout seen in any phase. A timeout is scored SURVIVED (see `classify`), but
        # its true status is unknown, so it is recorded here the moment it happens rather than
        # read back off the final result set -- phase 2 can re-resolve a phase-1 timeout and
        # erase the evidence that the run was ever indecisive. Used to refuse a baseline write.
        timed_out = [r for r in p1 if r["status"] == "SURVIVED_TIMEOUT"]
        print(f"  phase 1: {len(p1)} mutants, {len(survivors)} survived, "
              f"{sum(1 for r in p1 if r['status'] == 'KILLED')} killed, "
              f"{sum(1 for r in p1 if r['status'] == 'SURVIVED_TIMEOUT')} timed out (counted as survived), "
              f"{len(errors)} harness errors, {len(not_applied)} not applied")

        # ---- phase 2: full-suite recheck of survivors ------------------------------
        final = {r["id"]: r for r in p1}
        if not args.no_recheck and survivors:
            print(f"\nphase 2: re-running {len(survivors)} phase-1 survivors against the "
                  f"FULL suite", flush=True)
            p2 = run_phase("p2", survivors, workers, runner, lambda j: FULL_SUITE, 20,
                           confirm=True)
            for r in p2:
                if r["status"] == "KILLED":
                    r["status"] = "KILLED_BY_OTHER_TESTS"
                final[r["id"]] = r
            errors += [r for r in p2 if r["status"] == "ERROR"]
            timed_out += [r for r in p2 if r["status"] == "SURVIVED_TIMEOUT"]
            print(f"  phase 2: {sum(1 for r in p2 if is_survivor(r['status']))} of "
                  f"{len(p2)} survive the full suite")
            for r in p2:
                if r["status"] == "KILLED_BY_OTHER_TESTS":
                    print(f"    killed outside its own selection: {r['id']}\n"
                          f"      {r.get('note', '')}")
            unattributable = [r for r in p2 if is_survivor(r["status"]) and r.get("note")]
            if unattributable:
                print(f"    {len(unattributable)} full-suite failures were NOT attributable "
                      f"to the mutant and are scored SURVIVED:")
                for r in unattributable:
                    print(f"      {r['id']}\n        {r['note']}")

        if timed_out:
            n_timed_out = len({r["id"] for r in timed_out})
            print(f"\n  WARNING: {n_timed_out} mutant(s) timed out and were counted as "
                  f"SURVIVED; a timeout is not evidence of detection. Raise --timeout for a "
                  f"decisive answer. A baseline cannot be written from this run:")
            for mid in sorted({r["id"] for r in timed_out}):
                print(f"    {mid}")

        results = [final[j["id"]] for j in jobs]

        # ---- tally -----------------------------------------------------------------
        for r in results:
            if is_survivor(r["status"]):
                m = per_module[r["module"]]
                m["survivors"] += 1
                m["survivors_detail"].append(
                    {k: r[k] for k in ("id", "fingerprint", "line", "col", "kind", "old",
                                       "new", "expr_before", "expr_after", "source_line")}
                )
        total_surv = sum(m["survivors"] for m in per_module.values())

        print("\n" + "=" * 70)
        print(f"RESULT: {total_surv} of {len(results)} mutations survive"
              + ("" if args.no_recheck else " the full suite"))
        print("=" * 70)
        for rel in modules:
            m = per_module[rel]
            pct = 100.0 * (m["mutations"] - m["survivors"]) / m["mutations"] if m["mutations"] else 0
            print(f"  {rel:40} {m['survivors']:3}/{m['mutations']:<4} survive "
                  f"(kill rate {pct:.0f}%)")
        if errors:
            print(f"\n!! {len(errors)} HARNESS ERRORS -- unexpected pytest exit codes. "
                  f"These are NOT killed mutants; the numbers above are incomplete.")
            for r in errors[:10]:
                print(f"   {r['id']}  rc={r.get('rc')}  {r.get('tail', '')}")
                if r.get("stderr"):
                    print("     stderr: " + r["stderr"].replace("\n", "\n     ")[-800:])
            rc_final = 3
        if not_applied:
            print(f"\n!! {len(not_applied)} mutations could not be applied "
                  f"(AST node not found) -- a generator/transformer mismatch.")
            for r in not_applied[:10]:
                print(f"   {r['id']}")
            rc_final = max(rc_final, 3)

        # ---- outputs ---------------------------------------------------------------
        payload = {
            "repo": str(repo), "modules": modules, "controls": controls,
            "recheck": not args.no_recheck,
            "totals": {"mutations": len(results), "survivors": total_surv,
                       "errors": len(errors)},
            "per_module": per_module, "results": results,
        }
        if args.json:
            pathlib.Path(args.json).write_text(json.dumps(payload, indent=2))
            print(f"\nwrote {args.json}")
        if args.write_baseline:
            if args.no_recheck:
                die(2, "refusing to write a baseline from a --no-recheck run: its survivor "
                       "counts are inflated and not comparable to a normal run.")
            if errors or not_applied:
                die(2, "refusing to write a baseline from a run with harness errors.")
            if timed_out:
                die(2, f"refusing to write a baseline from a run with {len(timed_out)} "
                       "timeout(s): a timed-out mutant's true status is unknown, so the "
                       "baseline would record a guess. Raise --timeout and re-run.")
            base = build_baseline(repo, modules, per_module)
            pathlib.Path(args.write_baseline).write_text(json.dumps(base, indent=2) + "\n")
            print(f"wrote baseline {args.write_baseline} "
                  f"({base['totals']['survivors']}/{base['totals']['mutations']})")

        # ---- gate ------------------------------------------------------------------
        if args.baseline:
            baseline = json.loads(pathlib.Path(args.baseline).read_text())
            if compare_baseline(baseline, modules, per_module, args.strict_identity):
                print("\nGATE FAILED: surviving mutants increased on a targeted module.")
                return 1
            print("\nGATE PASSED: no module has more survivors than the baseline records.")
        return rc_final
    finally:
        if tmp and not args.keep_workdir:
            shutil.rmtree(tmp, ignore_errors=True)
        elif args.keep_workdir:
            print(f"\nkept mutant copies in {workdir}")


if __name__ == "__main__":
    sys.exit(main())
