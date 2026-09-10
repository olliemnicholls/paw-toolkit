"""Typer and Rich command-line interface for paw-toolkit."""

import importlib
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
import typer

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.pathsafety import ensure_contained
from paw_kit.serve.server import backend_label
from paw_kit.speclint import Finding, lint_spec
from paw_kit.test.active import run_active_learning_loop
from paw_kit.test.compare import CompareReport, compare_adapters, read_adapter_manifest
from paw_kit.test.judge import (
    JudgeInputRow,
    JudgeReport,
    anthropic_judge,
    diff_verdicts,
    judge_disagreements,
    judge_outputs,
)
from paw_kit.test.runner import TestRunner
from paw_kit.test.suite import load_suite

console = Console()


def _e(value: object) -> str:
    """Escape a value for interpolation into a Rich markup string.

    Rich parses square brackets as markup tags, so an unescaped interpolation either
    silently deletes the bracketed span or raises MarkupError and takes the command down.
    Both were live bugs here: an install hint printed `pip install 'paw-kit'` because
    Rich ate "[real]", and an adapter path like `[v2]model.paw` crashed `paw-inspect`.
    Paths, filenames, exception text and model output are all attacker- or
    environment-controlled; route every one of them through this.

    `tests/test_cli.py::test_no_unescaped_console_interpolations` enforces its use.
    """
    return escape(str(value))

# PAW-CLI-06: cap on how much of a candidate adapter file `paw-inspect` will attempt
# to parse as JSON. See `inspect()` below.
_MAX_INSPECT_FILE_BYTES = 50 * 1024 * 1024

app = typer.Typer(help="paw-kit: reliability and migration harness for Program-as-Weights (PAW) neural functions")
test_app = typer.Typer(help="paw-test: Test runner and active-learning self-healing suite")
test_app.__test__ = False  # Prevent pytest from treating Typer instance as a test suite


@test_app.callback()
def test_app_main() -> None:
    """paw-test: Test runner and active-learning self-healing suite."""


# PAW-CLI-06 applies the same reasoning to `paw-inspect`: a `.paw` path is user-supplied
# and may point at anything, so never read one unbounded. This cap is smaller because a
# manifest is a few hundred bytes of JSON -- an adapter file larger than this is not one.
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024


def _declared_adapter_backend(adapter_path: str) -> Optional[str]:
    """Return the `backend` an existing .paw manifest declares, or None.

    None means "no opinion": the file is unreadable, oversized, not JSON, or carries no
    usable `backend` key. It does **not** mean "safe to overwrite" -- callers must gate on
    whether the file *exists* (see `check()`), because an absent adapter is the legitimate
    first-compile case while a present-but-unreadable one is not.
    """
    try:
        path = Path(adapter_path)
        if not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
            return None
        # UnicodeDecodeError is a ValueError, not an OSError -- a binary .paw used to
        # escape this function as an uncaught traceback on the default invocation.
        raw = path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    declared = manifest.get("backend")
    return declared if isinstance(declared, str) else None


def _resolve_cli_backend(backend_type: str) -> Any:
    """Resolve the --backend flag to a concrete backend.

    `real` resolves to `ProgramAsWeightsBackend` -- the official upstream SDK, and the
    only backend in paw-kit proven end to end against a real model (see
    `measurements/`). It previously resolved to `RealPAWBackend` (an in-process PyTorch
    placeholder, deleted in Track 13), whose `compile()` and `infer()` raised
    `NotImplementedError`, so this function returned `MockPAWBackend()`
    on *every* path: `paw-test check --backend real` printed a "falling back" notice and
    then tested a dictionary lookup. That made the CLI structurally incapable of
    exercising a real compiled adapter, which is the one thing `paw-test` exists to do.

    Falling back to the mock is still possible (the SDK is an optional dependency), but
    it is now the exception and it is always announced, never the silent default.
    """
    resolved = backend_type.strip().lower()

    if resolved == "mock":
        return MockPAWBackend()

    if resolved != "real":
        raise typer.BadParameter(
            f"Unknown backend {backend_type!r}. Expected 'mock' or 'real'.",
            param_hint="--backend",
        )

    backend = ProgramAsWeightsBackend()
    try:
        sdk_present = backend.is_available()
    except Exception:
        # find_spec can raise on a corrupted/shadowed package rather than returning
        # False. Treat that as absent: this whole branch exists to degrade loudly.
        sdk_present = False
    if not sdk_present:
        console.print(
            "[bold yellow]Warning:[/bold yellow] --backend real needs the official upstream SDK.\n"
            # Escape the [ so Rich does not parse "[real]" as a markup tag and print
            # `pip install 'paw-kit'` -- wrong advice, and silently wrong.
            # paw-kit is not on PyPI yet, so name the SDK directly as well: the extra
            # only works from a source checkout today.
            "  To install: [cyan]pip install programasweights[/cyan] "
            "(or [cyan]uv sync --extra real[/cyan] from a source checkout)\n"
            "  Falling back to [green]MockPAWBackend[/green] -- results below come from a "
            "dictionary lookup, not a model."
        )
        return MockPAWBackend()

    # `is_available()` only calls find_spec -- it never imports. A package that is present
    # but unimportable (mismatched llama_cpp, half-finished install, a broken CUDA build)
    # therefore passes the check above and fails later, inside infer(), where TestRunner's
    # blanket `except Exception` turns it into "[EXECUTION_ERROR]" and a 0% pass rate that
    # reads as the *model* failing the user's assertions. Force the import here instead, so
    # infrastructure failure surfaces as infrastructure failure and the announced-fallback
    # contract above still applies to it.
    try:
        backend._paw()
    except Exception as exc:
        console.print(
            "[bold yellow]Warning:[/bold yellow] the upstream SDK is installed but could "
            f"not be loaded: {escape(str(exc))}\n"
            "  Falling back to [green]MockPAWBackend[/green] -- results below come from a "
            "dictionary lookup, not a model."
        )
        return MockPAWBackend()

    if not backend.has_api_key():
        # Not fatal: inference against an already-cached program needs no key. Only
        # compilation does, and the backend raises its own clear error if one is needed.
        console.print(
            "[bold yellow]Note:[/bold yellow] PAW_API_KEY is not set. Inference on an "
            "already-compiled adapter will work; compiling a new one will not.\n"
            "  Get a key at [cyan]https://programasweights.com/settings[/cyan]"
        )

    console.print(
        f"[green]Backend:[/green] ProgramAsWeightsBackend "
        f"([dim]compiler={_e(backend.compiler)}[/dim])"
    )
    return backend


def _run_triage_demo() -> None:
    """Inline ticket triage demo — zero external dependencies."""
    import pydantic
    from paw_kit.jit.decorator import compile_on_hit

    class TriageResult(pydantic.BaseModel):
        priority: str = pydantic.Field(description="Priority: low, medium, high, critical")
        department: str = pydantic.Field(description="Department: billing, technical, sales, general")
        urgency_score: int = pydantic.Field(description="Urgency 1-5")

    console.print(Panel.fit(
        "[bold cyan]⚡ PAW-Kit Demo: Support Ticket Triage[/bold cyan]\n\n"
        "Watch [bold]@compile_on_hit[/bold] trace remote teacher calls, trigger background\n"
        "compilation, and hot-swap to a simulated local adapter (MockPAWBackend,\n"
        "no model) — all running locally with [green]zero GPU and zero external API keys[/green].",
        border_style="cyan",
    ))

    tickets = [
        "Charged twice on credit card for invoice #INV-9821, need refund!",
        "Production API returning 502 errors across all regions!",
        "Interested in enterprise contract for 250 seats.",
        "Can't find the upload button in settings.",
        "Database replication lag exceeding 45 minutes!",
        "iOS app crashes on startup since latest update.",
    ]

    # PAW-CLI-08: a context manager guarantees cleanup on every exit path -- the
    # previous manual mkdtemp()+rmtree() pair skipped cleanup entirely on an
    # unhandled exception or Ctrl+C (KeyboardInterrupt) mid-demo, leaking a temp
    # directory holding trace data on every abnormal exit, not just the happy path.
    with tempfile.TemporaryDirectory(prefix="paw_demo_") as temp_dir:
        class TriageMockBackend(MockPAWBackend):
            def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
                lower = input_text.lower()
                if "502" in lower or "crash" in lower or "lag" in lower:
                    res = {"priority": "critical", "department": "technical", "urgency_score": 5}
                elif "charged" in lower or "refund" in lower:
                    res = {"priority": "high", "department": "billing", "urgency_score": 4}
                elif "contract" in lower or "enterprise" in lower:
                    res = {"priority": "medium", "department": "sales", "urgency_score": 3}
                else:
                    res = {"priority": "low", "department": "general", "urgency_score": 1}
                return json.dumps(res)

        backend = TriageMockBackend()

        @compile_on_hit(
            spec="Classify support ticket into priority, department, urgency_score.",
            threshold=3,
            response_model=TriageResult,
            backend=backend,
            cache_dir=temp_dir,
            # Shadow mode is on by default; this 6-ticket demo has nowhere near
            # enough calls to fill an agreement window, so turn it off to keep the
            # hot-swap deterministic.
            shadow_window=0,
            sync_compile=True,
        )
        def triage_ticket(ticket_body: str) -> TriageResult:
            time.sleep(0.05)  # Simulate remote teacher latency
            lower = ticket_body.lower()
            if "502" in lower or "crash" in lower or "lag" in lower:
                return TriageResult(priority="critical", department="technical", urgency_score=5)
            elif "charged" in lower or "refund" in lower:
                return TriageResult(priority="high", department="billing", urgency_score=4)
            elif "contract" in lower or "enterprise" in lower:
                return TriageResult(priority="medium", department="sales", urgency_score=3)
            else:
                return TriageResult(priority="low", department="general", urgency_score=1)

        # Fixed column widths are sized to fit an 80-column terminal (the common
        # default for xterm/gnome-terminal/tmux/CI logs) with a few columns of
        # margin: content widths sum to 58, +12 for per-column padding, +7 for
        # borders = 77 <= 80. A "Ticket Preview" column used to sit here too, but
        # at 80 columns Rich had no room left to render it and silently collapsed
        # it to zero width, gluing the table's right border into a doubled "┃┃".
        table = Table(title="Live JIT Ticket Triage Results", show_lines=True)
        table.add_column("#", style="dim", width=2)
        table.add_column("Mode", width=16)
        table.add_column("Latency", justify="right", width=10)
        table.add_column("Priority", width=10)
        table.add_column("Department", width=12)
        table.add_column("Urgency", justify="center", width=8)

        for i, ticket in enumerate(tickets, 1):
            is_local = triage_ticket.is_compiled()
            mode = "[green]LOCAL (mock)[/green]" if is_local else "[yellow]REMOTE TEACHER[/yellow]"

            t0 = time.perf_counter()
            result = triage_ticket(ticket)
            ms = (time.perf_counter() - t0) * 1000

            table.add_row(
                str(i), mode, f"{ms:.1f}ms",
                result.priority, result.department,
                str(result.urgency_score),
            )

            if i == 3:
                console.print("\n[bold cyan]>>> Hit threshold (3 calls) reached! Background compilation hot-swapped adapter.[/bold cyan]\n")

        console.print(table)
        console.print("\n[bold green]✓[/bold green] Calls 1–3 via remote teacher (logged to SQLite trace DB)")
        console.print("[bold green]✓[/bold green] Calls 4–6 via simulated local adapter (MockPAWBackend, no model) — <1ms, no inference cost")
        console.print("[bold green]✓[/bold green] Zero GPU • Zero API keys • Zero configuration\n")


def _run_pii_demo() -> None:
    """Inline PII scrubber demo — zero external dependencies."""
    import pydantic
    from paw_kit.schema.loader import load

    class PIIEntity(pydantic.BaseModel):
        entity_type: str
        value: str

    class PIIScrubResult(pydantic.BaseModel):
        sanitized_text: str
        entities: List[PIIEntity]
        total_redacted: int

    console.print(Panel.fit(
        "[bold cyan]⚡ PAW-Kit Demo: High-Throughput PII Scrubber[/bold cyan]\n\n"
        "Watch [bold]paw.load[/bold] bind an adapter to a strict Pydantic schema, validate\n"
        "every output, and fall back on failure. The adapter here is MockPAWBackend (no model,\n"
        "no token-level grammar masking); this shows the control flow, not model quality.",
        border_style="cyan",
    ))

    samples = [
        "Contact alice@enterprise.org or call 555-019-2834 regarding order #42.",
        "SSN is 000-12-3456, and alternate email is secret_user@gmail.com.",
        "Please charge corporate card 4111-2222-3333-4444 before expiration.",
        "No sensitive data here! Just asking about the documentation.",
    ]

    class PIIMock(MockPAWBackend):
        def infer(self, adapter_path: str, input_text: str, grammar_constraint: Optional[str] = None) -> str:
            entities = []
            sanitized = input_text
            for m in re.finditer(r"[\w\.-]+@[\w\.-]+\.\w+", input_text):
                entities.append({"entity_type": "EMAIL", "value": m.group(0)})
                sanitized = sanitized.replace(m.group(0), "[REDACTED_EMAIL]")
            for m in re.finditer(r"\b\d{3}-\d{3}-\d{4}\b", input_text):
                entities.append({"entity_type": "PHONE", "value": m.group(0)})
                sanitized = sanitized.replace(m.group(0), "[REDACTED_PHONE]")
            for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", input_text):
                entities.append({"entity_type": "SSN", "value": m.group(0)})
                sanitized = sanitized.replace(m.group(0), "[REDACTED_SSN]")
            for m in re.finditer(r"\b\d{4}-\d{4}-\d{4}-\d{4}\b", input_text):
                entities.append({"entity_type": "CREDIT_CARD", "value": m.group(0)})
                sanitized = sanitized.replace(m.group(0), "[REDACTED_CARD]")
            return json.dumps({"sanitized_text": sanitized, "entities": entities, "total_redacted": len(entities)})

    # PAW-CLI-08: see the matching comment in _run_triage_demo -- a context manager
    # guarantees cleanup even if scrub_fn or an assertion inside the loop raises.
    with tempfile.TemporaryDirectory(prefix="paw_pii_") as temp_dir:
        adapter_path = Path(temp_dir) / "pii.paw"
        backend = PIIMock()
        backend.compile(spec="Extract and redact PII", examples=[{"input": "test", "output": "{}"}], output_path=str(adapter_path))

        scrub_fn = load(adapter_path=str(adapter_path), response_model=PIIScrubResult, backend=backend)

        for i, text in enumerate(samples, 1):
            t0 = time.perf_counter()
            result = scrub_fn(text)
            ms = (time.perf_counter() - t0) * 1000
            console.print(f"\n[bold]Item #{i}[/bold] [{ms:.2f}ms] — {result.total_redacted} entity(ies) redacted")
            console.print(f"  [dim]Raw:[/dim]       {_e(text)}")
            console.print(f"  [green]Sanitized:[/green] {_e(result.sanitized_text)}")
            if result.entities:
                for e in result.entities:
                    console.print(f"  [yellow]  → {_e(e.entity_type)}:[/yellow] {_e(e.value)}")

        console.print("\n[bold green]✓[/bold green] Every output validated against the Pydantic schema, fail-open on mismatch")
        console.print("[bold green]✓[/bold green] Simulated adapter (MockPAWBackend, no model) — timings are not a benchmark\n")


@app.command(name="demo")
def demo_cmd(
    scenario: str = typer.Option("triage", "--scenario", "-s", help="Demo scenario: triage | pii"),
) -> None:
    """Run an interactive zero-hardware demo showcasing JIT compilation or schema enforcement."""
    if scenario.lower() == "triage":
        _run_triage_demo()
    elif scenario.lower() == "pii":
        _run_pii_demo()
    else:
        console.print(f"[bold red]Unknown scenario:[/bold red] '{_e(scenario)}'. Choose 'triage' or 'pii'.")
        raise typer.Exit(code=1)


def _write_json_report(json_out: Optional[Path], data: dict) -> None:
    """Shared `--json PATH` writer for `check`/`compare`: a plain `json.dumps`, not
    routed through Rich -- machine-readable output must not be subject to Rich's
    markup parsing or terminal-width line wrapping."""
    if json_out is None:
        return
    json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[dim]Wrote JSON report to {_e(json_out)}[/dim]")


@test_app.command(name="check")
@app.command(name="check")
def check(
    suite_path: Path = typer.Argument(..., help="Path to declarative suite.yaml specification"),
    backend_type: str = typer.Option("mock", "--backend", "-b", help="Backend engine: mock | real"),
    auto_recompile: Optional[bool] = typer.Option(
        None, "--auto-recompile/--no-auto-recompile", help="Enable active learning (overrides suite.yaml if set)"
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Write the run's TestRunReport as JSON to this path (input for `paw-test judge`)"
    ),
) -> None:
    """Run test suite assertions and active-learning self-healing loop on a .paw adapter."""
    if not suite_path.exists():
        console.print(f"[bold red]Error:[/bold red] Suite file '{_e(suite_path)}' does not exist.")
        raise typer.Exit(code=1)

    try:
        config = load_suite(str(suite_path))
    except Exception as exc:
        console.print(f"[bold red]Error parsing suite:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)

    if auto_recompile is not None:
        config.active_learning.auto_recompile = auto_recompile

    # Select backend safely
    backend = _resolve_cli_backend(backend_type)
    is_real = not isinstance(backend, MockPAWBackend)
    actual_backend = type(backend).__name__

    # A real backend's compile() is a paid remote submission that overwrites
    # config.adapter_path *in place*. `auto_recompile` defaults to True (suite.py) and the
    # shipped example suite sets it true, so before this guard existed, adding one flag to
    # the command the README prints -- `paw-test check examples/date_normalizer/suite.yaml
    # --backend real` -- would submit up to max_iterations-1 real compiles, built on labels
    # invented by cli_teacher below (a demo stub that answers "2026-01-01" to almost
    # anything), and destroy the adapter it was asked to test. Never fire that implicitly.
    if is_real and config.active_learning.auto_recompile:
        if auto_recompile is not True:
            config.active_learning.auto_recompile = False
            console.print(
                "[bold yellow]Note:[/bold yellow] auto-recompile is disabled for "
                f"--backend {_e(backend_type.strip().lower())}. Recompiling would submit a "
                "paid upstream "
                f"compile and overwrite [cyan]{escape(config.adapter_path)}[/cyan] in place.\n"
                "  Running assertions read-only. Pass [cyan]--auto-recompile[/cyan] "
                "explicitly to allow recompilation."
            )
        else:
            # Explicit opt-in still must not feed a stub teacher into a paid compile.
            raise typer.BadParameter(
                "--auto-recompile with a real backend needs a real teacher; the CLI's "
                "built-in teacher is a demo stub that answers '2026-01-01' to almost any "
                "input, and its labels would become training signal for a paid compile "
                "that overwrites your adapter. Drive the loop from code instead: "
                "run_active_learning_loop(config, backend, teacher_provider=...).",
                param_hint="--auto-recompile",
            )

    # The mock is NOT a safe place to recompile either, which the guard above originally
    # claimed it was. `MockPAWBackend.compile()` writes a real file (`backend/mock.py`,
    # atomic_write_text), so `paw-test check suite.yaml` with no flags at all -- default
    # backend, default auto_recompile=True -- overwrites whatever `adapter_path` points at
    # with a mock stub whose examples are cli_teacher's fabricated labels. Reproduced
    # against a real programasweights manifest: it was replaced wholesale. Nothing about
    # that is specific to --backend real; it is the *default* invocation.
    if config.active_learning.auto_recompile and Path(config.adapter_path).exists():
        # Gate on existence, not on the return value. An *absent* adapter is the normal
        # first-compile case and must stay allowed; a file that is present but whose
        # manifest cannot be read is the dangerous case -- it may be a third-party
        # AbstractPAWBackend adapter, a foreign format, or binary weights, none of which
        # the mock may silently replace. Anything present that does not positively
        # identify itself as "mock" is protected.
        existing_backend = _declared_adapter_backend(config.adapter_path)
        if existing_backend != "mock":
            config.active_learning.auto_recompile = False
            console.print(
                "[bold yellow]Note:[/bold yellow] auto-recompile is disabled: "
                f"[cyan]{escape(config.adapter_path)}[/cyan] is "
                + (
                    f"a [bold]{escape(existing_backend)}[/bold] adapter"
                    if existing_backend
                    else "not a readable mock manifest"
                )
                + ", and recompiling would replace it with a mock stub built from this "
                "CLI's demo teacher.\n"
                "  Running assertions read-only. Recompile it with the backend that "
                "produced it, from code."
            )

    console.print(
        f"[bold cyan]Running paw.test check on:[/bold cyan] {_e(config.task_name)} "
        f"([dim]{escape(config.adapter_path)}[/dim]) [dim](backend: {_e(actual_backend)})[/dim]"
    )

    # PAW-CLI-02: adapter_path containment used to be checked here directly, but
    # load_suite() (paw_kit.test.suite, PAW-TEST-02) now validates it unconditionally
    # for every suite -- since `config` above can only have come from load_suite(), a
    # duplicate check here is unreachable dead code, not defense in depth.

    # If active learning auto-recompile is disabled, just run once
    if not config.active_learning.auto_recompile:
        runner = TestRunner(backend=backend)
        report = runner.run(config)
        for res in report.results:
            if res.passed:
                console.print(
                    f"  [green][PASS][/green] Input: {escape(repr(res.input[:50]))} -> "
                    f"{escape(repr(res.output[:40]))}"
                )
            else:
                console.print(
                    f"  [red][FAIL][/red] Input: {escape(repr(res.input[:50]))} -> "
                    f"{escape(repr(res.output[:40]))}"
                )
                for reason in res.failed_rules:
                    console.print(f"         [dim red]{escape(reason)}[/dim red]")
                # TestRunner catches backend exceptions into "[EXECUTION_ERROR]" and stashes
                # the message here. Without printing it, a backend that cannot run at all is
                # indistinguishable from a model failing the user's assertions.
                if res.execution_error:
                    console.print(f"         [dim red]backend error: {escape(res.execution_error)}[/dim red]")

        console.print(
            f"\n[bold]Pass rate:[/bold] {report.pass_rate:.1f}% "
            f"({report.passed_cases}/{report.total_cases}) [dim](backend: {_e(actual_backend)})[/dim]"
        )
        _write_json_report(json_out, report.model_dump())
        if not report.is_success:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    # Demo-only stub teacher for active-learning auto-repair in CLI check. This is NOT a
    # frontier model -- it is a two-branch lookup. The guards above keep its fabricated
    # labels away from a paid compile (real backends never reach here) and away from any
    # adapter that was not produced by the mock. They do NOT make it harmless: reaching
    # here still rewrites a mock adapter file on disk with these labels folded in.
    def cli_teacher(inp: str) -> str:
        console.print(
            f"  [yellow][ACTION][/yellow] Active learning: Querying frontier teacher "
            f"for '{escape(inp[:40])}'..."
        )
        # If input was an invalid date, return canonical gold label
        if "February 30" in inp or "32" in inp:
            return "INVALID"
        return "2026-01-01"

    al_report = run_active_learning_loop(
        config=config,
        backend=backend,
        teacher_provider=cli_teacher,
    )

    for i, rep in enumerate(al_report.iteration_reports, 1):
        console.print(f"\n[bold]Iteration {i}:[/bold] {rep.passed_cases}/{rep.total_cases} passed ({rep.pass_rate:.1f}%)")
        if not rep.is_success and i < len(al_report.iteration_reports):
            console.print("  [cyan][ACTION][/cyan] Recompiling adapter with augmented edge-case pairs...")

    # PAW-CLI-09: `--json` always writes the *last* iteration's plain `TestRunReport`
    # (not the ActiveLearningReport wrapper), whichever code path got here -- so a
    # consumer like `paw-test judge` sees the same {task_name, results: [...]} shape
    # regardless of whether auto-recompile ran.
    _write_json_report(json_out, al_report.iteration_reports[-1].model_dump())

    if al_report.is_success:
        console.print(f"\n[bold green][SUCCESS][/bold green] All assertions passed! (Iterations: {al_report.iterations_run})")
        if al_report.recompiled:
            console.print(f"  [cyan][UPDATE][/cyan] Recompiled and updated artifact: {_e(config.adapter_path)}")
        raise typer.Exit(code=0)
    else:
        console.print(f"\n[bold red][FAIL][/bold red] Assertions failed after {al_report.iterations_run} iteration(s).")
        raise typer.Exit(code=1)


@test_app.command(name="compare")
def compare_cmd(
    adapter_a: Path = typer.Argument(..., help="First compiled .paw adapter"),
    adapter_b: Path = typer.Argument(..., help="Second compiled .paw adapter"),
    suite_path: Path = typer.Argument(..., help="Path to declarative suite.yaml specification"),
    backend_type: str = typer.Option("mock", "--backend", "-b", help="Backend engine: mock | real"),
    fuzz: bool = typer.Option(
        True,
        "--fuzz/--no-fuzz",
        help="Also run the suite's fuzz cases through both adapters, not just "
        "standard_cases -- on by default, so a default `compare` run covers the same "
        "cases `paw-test check` does. Pass --no-fuzz for standard_cases only.",
    ),
    json_out: Optional[Path] = typer.Option(
        None, "--json", help="Write the full CompareReport as JSON to this path"
    ),
) -> None:
    """Run every suite case through two compiled adapters and diff the results, per case.

    Read-only against both adapters: `--backend real` never compiles. A per-case diff is
    the point -- this is the technique that actually decided whether `paw-ft-bs48` was a
    different compile from `paw-4b-qwen3-0.6b` (132/134 byte-identical outputs, see
    `measurements/README.md`'s "Finetune compiler" section), where the aggregate pass-rate
    delta alone sat inside the LLM judge's own measured run-to-run noise.

    Both adapters are held in memory for the whole run (two ~600MB llama.cpp models
    under `--backend real`, never released) -- the first row's latencies include that
    cold load, not steady-state inference time.
    """
    for label, adapter_path_arg in (("A", adapter_a), ("B", adapter_b)):
        if not adapter_path_arg.exists():
            console.print(f"[bold red]Error:[/bold red] Adapter {_e(label)} '{_e(adapter_path_arg)}' does not exist.")
            raise typer.Exit(code=1)
    if not suite_path.exists():
        console.print(f"[bold red]Error:[/bold red] Suite file '{_e(suite_path)}' does not exist.")
        raise typer.Exit(code=1)

    try:
        suite = load_suite(str(suite_path))
    except Exception as exc:
        console.print(f"[bold red]Error parsing suite:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)

    for label, adapter_path_arg in (("A", adapter_a), ("B", adapter_b)):
        if not read_adapter_manifest(str(adapter_path_arg)):
            console.print(
                f"[bold red]Error:[/bold red] Could not read a manifest from adapter {_e(label)} "
                f"'{_e(adapter_path_arg)}' -- not JSON, or not the expected shape. Refusing to compare."
            )
            raise typer.Exit(code=1)

    backend = _resolve_cli_backend(backend_type)
    report: CompareReport = compare_adapters(
        str(adapter_a), str(adapter_b), suite, backend, include_fuzz=fuzz
    )

    # PAW-TEST-08-style backend failures (`_infer_safely`) never raise -- a backend
    # that errors on every case still produces a "complete" report, byte-identical
    # "[EXECUTION_ERROR]" output for both adapters, and no differing rows at all. Print
    # every errored case's own error text, separately from the differences section
    # below, so that isn't mistaken for a clean comparison that measured nothing.
    errored_rows = [row for row in report.rows if row.execution_error_a or row.execution_error_b]
    if errored_rows:
        console.print(f"[bold red]Errors ({_e(len(errored_rows))}/{_e(report.total_cases)}):[/bold red]\n")
        for row in errored_rows:
            console.print(f"  [bold]Input:[/bold] {_e(row.input[:80])}")
            if row.execution_error_a:
                console.print(f"    [red]A error:[/red] {_e(row.execution_error_a)}")
            if row.execution_error_b:
                console.print(f"    [red]B error:[/red] {_e(row.execution_error_b)}")
        console.print()

    differing = report.differing_rows
    if differing:
        console.print(f"[bold]Differences ({_e(len(differing))}/{_e(report.total_cases)}):[/bold]\n")
        for row in differing:
            console.print(f"  [bold]Input:[/bold] {_e(row.input[:80])}")
            console.print(f"    A -> {_e(row.output_a[:80])}")
            console.print(f"    B -> {_e(row.output_b[:80])}")
            if row.pass_a != row.pass_b:
                console.print(f"    [yellow]pass differs:[/yellow] A={_e(row.pass_a)} B={_e(row.pass_b)}")
    elif not errored_rows:
        # Suppressed whenever any case errored -- both adapters raising on every case is
        # exactly the "0 differing rows" shape this green line used to paper over.
        console.print("[bold green]No differences[/bold green] -- identical output and pass status on every case.")

    console.print(
        f"\n[bold]Summary:[/bold] {_e(report.total_cases)} cases, {_e(report.identical_count)} identical output, "
        f"A pass {_e(report.a_pass_count)}/{_e(report.total_cases)}, "
        f"B pass {_e(report.b_pass_count)}/{_e(report.total_cases)}, "
        f"only-A-pass {_e(report.only_a_pass_count)}, only-B-pass {_e(report.only_b_pass_count)}, "
        f"errored {_e(report.errored_count)}/{_e(report.total_cases)}"
    )

    _write_json_report(json_out, report.model_dump())
    if errored_rows:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _load_judge_verdicts_file(path: Path) -> Dict[str, JudgeReport]:
    """Load a `judge --out` file for `--diff`.

    Accepts both shapes `--out` can write: a bare `JudgeReport` (judging a `check`
    report) -- returned as `{"": report}` -- or the `{"adapter_a": ..., "adapter_b":
    ...}` wrapper (judging a `compare` report) -- returned as `{"A": ..., "B": ...}`.
    Before this, `--diff` on a compare-shaped `--out` file validated the whole file as
    one `JudgeReport` and failed outright (finding 4).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "adapter_a" in data and "adapter_b" in data:
        return {
            "A": JudgeReport.model_validate(data["adapter_a"]),
            "B": JudgeReport.model_validate(data["adapter_b"]),
        }
    return {"": JudgeReport.model_validate(data)}


# Above this share of a judge run's cases parsing as "unparseable" (finding 6), warn --
# a run where most responses didn't even follow the requested YES/NO format is a broken
# judge/prompt, not a genuinely bad pass rate, and the plain pass-rate number alone
# reads identically to both.
_UNPARSEABLE_WARN_THRESHOLD = 0.2


def _warn_on_unparseable(report: JudgeReport, side: Optional[str] = None) -> None:
    """Print a stderr warning when more than `_UNPARSEABLE_WARN_THRESHOLD` of a judge
    run's verdicts came from an unparseable response (see `JudgeReport.unparseable_count`
    and `parse_verdict`)."""
    if not report.total_cases:
        return
    rate = report.unparseable_count / report.total_cases
    if rate <= _UNPARSEABLE_WARN_THRESHOLD:
        return
    label = f" ({side})" if side else ""
    print(
        f"[paw-test judge] WARNING: {report.unparseable_count}/{report.total_cases} "
        f"({rate * 100:.0f}%) judge responses{label} were unparseable "
        f"(judge_id={report.judge_id}) -- these count as a fail, not a reliable "
        "pass/fail signal. Check the judge's raw responses or prompt format.",
        file=sys.stderr,
    )


@test_app.command(name="judge")
def judge_cmd(
    report: Optional[Path] = typer.Argument(
        None, help="A `compare --json` or `check --json` report to judge (omit when using --diff)"
    ),
    spec: Optional[str] = typer.Option(None, "--spec", help="Task spec to show the judge"),
    suite_path: Optional[Path] = typer.Option(
        None, "--suite", help="suite.yaml to read the spec from, instead of --spec"
    ),
    judge_name: str = typer.Option(
        "anthropic:claude-haiku-4-5",
        "--judge",
        help="provider:model for the judge, e.g. anthropic:claude-haiku-4-5",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Write verdicts JSON to this path"),
    diff: Optional[Tuple[Path, Path]] = typer.Option(
        None, "--diff", help="Diff two verdicts JSON files (OLD NEW) instead of judging a report"
    ),
) -> None:
    """Score a `compare`/`check` report's outputs with an LLM judge, or diff two prior
    verdict runs to check reproducibility.

    Temperature is pinned to 0 (see `paw_kit.test.judge.anthropic_judge`): a
    default-temperature judge call flipped its YES/NO verdict on byte-identical
    input/output 4.5% of the time (6/134), measured in `measurements/README.md`'s
    "Finetune compiler" section. `--diff` is how to check whether that is still true for a
    given judge and prompt.

    Every case's spec, input, and output leaves this machine: each is sent to the
    configured judge provider (Anthropic by default) as part of the judge prompt.
    Judging a `compare` report costs 2x cases in judge calls, since both adapters'
    outputs are judged separately.
    """
    if diff is not None:
        old_path, new_path = diff
        for diff_path in (old_path, new_path):
            if not diff_path.exists():
                console.print(f"[bold red]Error:[/bold red] Verdicts file '{_e(diff_path)}' does not exist.")
                raise typer.Exit(code=1)
        try:
            old_reports = _load_judge_verdicts_file(old_path)
            new_reports = _load_judge_verdicts_file(new_path)
        except Exception as exc:
            console.print(f"[bold red]Error reading verdicts:[/bold red] {_e(exc)}")
            raise typer.Exit(code=1)

        if set(old_reports) != set(new_reports):
            console.print(
                "[bold red]Error:[/bold red] old and new verdicts files are different shapes -- "
                "one is a single judge report, the other a `compare`-report A/B wrapper."
            )
            raise typer.Exit(code=1)

        # "" alone for a bare JudgeReport; "A" then "B" for a compare-report wrapper --
        # diffs A against A and B against B, and prints both (finding 4).
        for i, side in enumerate(sorted(old_reports)):
            if i > 0:
                console.print()
            old_report = old_reports[side]
            new_report = new_reports[side]
            diff_report = diff_verdicts(old_report, new_report)
            suffix = f" ({side})" if side else ""
            if diff_report.flips:
                console.print(
                    f"[bold]Flipped verdicts{_e(suffix)} "
                    f"({_e(diff_report.flipped_count)}/{_e(diff_report.compared_cases)}):[/bold]\n"
                )
                for flip in diff_report.flips:
                    console.print(f"  [bold]Input:[/bold] {_e(flip.input[:80])}")
                    console.print(f"    Output: {_e(flip.output[:80])}")
                    console.print(
                        f"    {_e(old_report.judge_id)}: {_e(flip.old_verdict)} ({_e(flip.old_reason)})  ->  "
                        f"{_e(new_report.judge_id)}: {_e(flip.new_verdict)} ({_e(flip.new_reason)})"
                    )
            else:
                console.print(f"[bold green]No flips{_e(suffix)}[/bold green] -- every comparable verdict matched.")
            console.print(
                f"\n[bold]Flip rate{_e(suffix)}:[/bold] {_e(round(diff_report.flip_rate, 1))}% "
                f"({_e(diff_report.flipped_count)}/{_e(diff_report.compared_cases)})"
            )
        raise typer.Exit(code=0)

    if report is None:
        console.print("[bold red]Error:[/bold red] REPORT is required unless --diff is given.")
        raise typer.Exit(code=1)
    if not report.exists():
        console.print(f"[bold red]Error:[/bold red] Report file '{_e(report)}' does not exist.")
        raise typer.Exit(code=1)

    resolved_spec = spec
    if resolved_spec is None and suite_path is not None:
        if not suite_path.exists():
            console.print(f"[bold red]Error:[/bold red] Suite file '{_e(suite_path)}' does not exist.")
            raise typer.Exit(code=1)
        try:
            resolved_spec = load_suite(str(suite_path)).spec
        except Exception as exc:
            console.print(f"[bold red]Error parsing suite:[/bold red] {_e(exc)}")
            raise typer.Exit(code=1)
    if resolved_spec is None:
        console.print("[bold red]Error:[/bold red] --spec or --suite is required to judge a report.")
        raise typer.Exit(code=1)

    try:
        report_data = json.loads(report.read_text(encoding="utf-8"))
    except Exception as exc:
        console.print(f"[bold red]Error reading report:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)

    provider, _sep, model = judge_name.partition(":")
    model = model or "claude-haiku-4-5"
    if provider != "anthropic":
        console.print(
            f"[bold red]Error:[/bold red] Unknown judge provider {_e(provider)!r}; only 'anthropic' is built in."
        )
        raise typer.Exit(code=2)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] No judge is configured: ANTHROPIC_API_KEY is not set. "
            "Set it, or pass a different --judge. Refusing to silently skip judging."
        )
        raise typer.Exit(code=2)
    try:
        judge_fn = anthropic_judge(model=model)
    except ImportError as exc:
        console.print(f"[bold red]Error:[/bold red] {_e(exc)}")
        raise typer.Exit(code=2)

    temperature_note = (
        "temperature=0.0 -- default-temperature judging measured a 4.5% (6/134) "
        "verdict-flip rate on byte-identical input/output (measurements/README.md, "
        "'Finetune compiler')."
    )
    # "/" separators, not ":" -- Rich's emoji shortcode syntax (":word:") can match
    # across an f-string's literal ": " and a colon-separated id sitting next to it
    # (verified: "anthropic:x:temperature=0.0:A" prints with a stray emoji in place of
    # ":x:"). `_e()`/`escape()` only guards Rich's `[markup]` brackets, not this, so the
    # id itself stays colon-free rather than relying on escaping to catch it.
    judge_id = f"anthropic/{model}/temperature=0.0"

    if "rows" in report_data and "adapter_a" in report_data:
        rows_a = [
            JudgeInputRow(input=r["input"], output=r["output_a"], rule_passed=r.get("pass_a"))
            for r in report_data["rows"]
        ]
        rows_b = [
            JudgeInputRow(input=r["input"], output=r["output_b"], rule_passed=r.get("pass_b"))
            for r in report_data["rows"]
        ]
        report_a = judge_outputs(
            rows_a, judge_fn, spec=resolved_spec, temperature_note=temperature_note, judge_id=f"{judge_id}/A"
        )
        report_b = judge_outputs(
            rows_b, judge_fn, spec=resolved_spec, temperature_note=temperature_note, judge_id=f"{judge_id}/B"
        )

        diffs = [(va, vb) for va, vb in zip(report_a.verdicts, report_b.verdicts) if va.verdict != vb.verdict]
        if diffs:
            console.print(
                f"[bold]Verdict differs between A and B ({_e(len(diffs))}/{_e(report_a.total_cases)}):[/bold]\n"
            )
            for va, vb in diffs:
                console.print(f"  [bold]Input:[/bold] {_e(va.input[:80])}")
                console.print(f"    A: {_e(va.output[:80])} -> {_e(va.verdict)} ({_e(va.reason)})")
                console.print(f"    B: {_e(vb.output[:80])} -> {_e(vb.verdict)} ({_e(vb.reason)})")
        else:
            console.print("[bold green]No verdict differences[/bold green] between A and B.")
        # Finding 5: assertions passing/failing and the judge's own verdict are two
        # separate signals -- print where they disagree instead of only ever showing
        # the judge's own YES/NO.
        for side_label, side_report in (("A", report_a), ("B", report_b)):
            disagreements = judge_disagreements(side_report)
            if disagreements:
                console.print(
                    f"\n[bold yellow]judge disagrees with assertions ({_e(side_label)}) "
                    f"({_e(len(disagreements))}/{_e(side_report.total_cases)}):[/bold yellow]"
                )
                for v in disagreements:
                    console.print(
                        f"  {_e(v.case_id)}: rule_passed={_e(v.rule_passed)} "
                        f"judge={_e(v.verdict)} ({_e(v.reason)})"
                    )
        console.print(
            f"\n[bold]Summary:[/bold] A pass rate {_e(round(report_a.pass_rate, 1))}% "
            f"({_e(report_a.pass_count)}/{_e(report_a.total_cases)}), "
            f"unparseable A {_e(report_a.unparseable_count)}, errored A {_e(report_a.error_count)}, "
            f"B pass rate {_e(round(report_b.pass_rate, 1))}% "
            f"({_e(report_b.pass_count)}/{_e(report_b.total_cases)}), "
            f"unparseable B {_e(report_b.unparseable_count)}, errored B {_e(report_b.error_count)}"
        )
        _warn_on_unparseable(report_a, side="A")
        _warn_on_unparseable(report_b, side="B")
        if out is not None:
            out.write_text(
                json.dumps({"adapter_a": report_a.model_dump(), "adapter_b": report_b.model_dump()}, indent=2),
                encoding="utf-8",
            )
            console.print(f"[dim]Wrote verdicts to {_e(out)}[/dim]")
        raise typer.Exit(code=0)

    if "results" in report_data:
        rows = [
            JudgeInputRow(input=r["input"], output=r["output"], rule_passed=r.get("passed"))
            for r in report_data["results"]
        ]
        jreport = judge_outputs(
            rows, judge_fn, spec=resolved_spec, temperature_note=temperature_note, judge_id=judge_id
        )
        failing = [v for v in jreport.verdicts if not v.verdict]
        if failing:
            console.print(f"[bold]Judge said NO ({_e(len(failing))}/{_e(jreport.total_cases)}):[/bold]\n")
            for v in failing:
                console.print(f"  [bold]Input:[/bold] {_e(v.input[:80])}")
                console.print(f"    Output: {_e(v.output[:80])}")
                console.print(f"    [red]Reason:[/red] {_e(v.reason)}")
        else:
            console.print("[bold green]Judge said YES on every case.[/bold green]")
        # Finding 5: the case that matters most is assertions-pass-judge-says-NO (or the
        # reverse), and it's invisible in the "Judge said NO" list above whenever the
        # assertions themselves failed too.
        disagreements = judge_disagreements(jreport)
        if disagreements:
            console.print(
                f"\n[bold yellow]judge disagrees with assertions "
                f"({_e(len(disagreements))}/{_e(jreport.total_cases)}):[/bold yellow]"
            )
            for v in disagreements:
                console.print(
                    f"  {_e(v.case_id)}: rule_passed={_e(v.rule_passed)} judge={_e(v.verdict)} ({_e(v.reason)})"
                )
        console.print(
            f"\n[bold]Pass rate:[/bold] {_e(round(jreport.pass_rate, 1))}% "
            f"({_e(jreport.pass_count)}/{_e(jreport.total_cases)}), "
            f"unparseable {_e(jreport.unparseable_count)}, errored {_e(jreport.error_count)}"
        )
        _warn_on_unparseable(jreport)
        if out is not None:
            out.write_text(jreport.model_dump_json(indent=2), encoding="utf-8")
            console.print(f"[dim]Wrote verdicts to {_e(out)}[/dim]")
        raise typer.Exit(code=0)

    console.print(
        "[bold red]Error:[/bold red] Unrecognized report shape; expected a `paw-test compare --json` "
        "or `paw-test check --json` report."
    )
    raise typer.Exit(code=1)

# Preferred, stable ordering for manifest fields `inspect()` prints -- covers every
# key either backend's compile() writes (manifest_version 1 and 2, both
# ProgramAsWeightsBackend and MockPAWBackend). Any manifest key not named here still
# prints (a third-party AbstractPAWBackend's own fields, or a future addition) --
# sorted alphabetically, after every key below and before "spec", which is always
# last regardless of where it falls in this list.
_INSPECT_FIELD_ORDER = [
    "manifest_version",
    "backend",
    "program_id",
    "slug",
    "compiler",
    "status",
    "spec_sha256",
    "full_spec_sha256",
    "examples_count",
    "examples_folded_into_spec",
    "folded_example_ids",
    "public",
    "ephemeral",
    "cache_hit",
    "parent_program_id",
    "parent_manifest_sha256",
    "compile_wall_s",
    "compiler_snapshot",
    "compiled_at",
    "rules",
    "default_response",
    "examples",
]


def _prettify_field_name(key: str) -> str:
    """`"examples_count"` -> `"Examples Count"` -- matches this command's pre-existing
    row labels for the fields that already had one, extended to every manifest key."""
    return key.replace("_", " ").title()


@app.command(name="inspect")
def inspect(
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
    json_output: bool = typer.Option(
        False, "--json", help="Print the raw manifest JSON instead of a formatted table"
    ),
) -> None:
    """Inspect metadata, task spec, and file properties of a .paw adapter artifact."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{_e(adapter_path)}' does not exist.")
        raise typer.Exit(code=1)

    stat = adapter_path.stat()
    size_bytes = stat.st_size

    # PAW-CLI-06: json.load() reads its entire input into memory before it can even
    # attempt to parse it, so pointing this at a huge file or a stream with no
    # end-of-file at all (a named pipe, /dev/zero) previously hung the process or
    # exhausted memory rather than failing fast. is_file() rejects anything that
    # isn't a regular file outright; the size cap skips the parse attempt (falling
    # back to the "Binary / Raw Weights" branch below) rather than reading a file
    # too large to plausibly be adapter metadata.
    metadata: dict = {}
    is_json = False
    if adapter_path.is_file() and size_bytes <= _MAX_INSPECT_FILE_BYTES:
        try:
            with open(adapter_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                metadata = loaded
                is_json = True
        except Exception:
            pass

    if json_output:
        if not is_json:
            console.print(
                f"[bold red]Error:[/bold red] '{_e(adapter_path)}' is not JSON adapter metadata."
            )
            raise typer.Exit(code=1)
        console.print_json(data=metadata)
        return

    table = Table(title=f"PAW Adapter: {adapter_path.name}")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("File Path", _e(adapter_path.resolve()))
    table.add_row("File Size", f"{size_bytes} bytes ({size_bytes / 1024:.2f} KB)")
    table.add_row("Format", "JSON Simulation" if is_json else "Binary / Raw Weights")

    if is_json:
        remaining = sorted(k for k in metadata if k not in _INSPECT_FIELD_ORDER and k != "spec")
        ordered_keys = [k for k in _INSPECT_FIELD_ORDER if k in metadata] + remaining
        for key in ordered_keys:
            table.add_row(_e(_prettify_field_name(key)), _e(metadata[key]))
        # The spec is printed last, always -- it is typically the longest value and
        # the one most useful to see uninterrupted at the bottom of the table.
        table.add_row("Specification", _e(metadata.get("spec", "N/A")))

    console.print(table)


# Sidecar history logs are append-only JSONL, one line per compile -- unbounded over
# an adapter's lifetime in principle, but each line is a manifest-sized JSON object
# (kilobytes, not megabytes). This is a generous cap consistent with the rest of the
# codebase's "never read an externally-writable path unbounded" posture (PAW-CLI-06),
# not a realistic ceiling for legitimate use.
_MAX_HISTORY_FILE_BYTES = 10 * 1024 * 1024


def _history_path(adapter_path: Path) -> Path:
    return Path(str(adapter_path) + ".history.jsonl")


@app.command(name="history")
def history(
    adapter_path: Path = typer.Argument(..., help="Path to a compiled .paw adapter artifact"),
) -> None:
    """Print the append-only compile lineage log for an adapter.

    Every `compile()` call (on any `AbstractPAWBackend` shipped by paw-kit) appends
    one line to `<adapter>.history.jsonl` -- the compiled manifest minus its spec
    text. A manifest itself only ever points at its immediate parent (an overwritten
    file cannot be read back), so this sidecar is the only place the full compile
    lineage of a repeatedly-recompiled adapter survives.
    """
    log_path = _history_path(adapter_path)
    if not log_path.is_file():
        console.print(f"[bold red]Error:[/bold red] no history log at '{_e(log_path)}'.")
        raise typer.Exit(code=1)
    if log_path.stat().st_size > _MAX_HISTORY_FILE_BYTES:
        console.print(
            f"[bold red]Error:[/bold red] history log '{_e(log_path)}' exceeds "
            f"{_e(_MAX_HISTORY_FILE_BYTES)} bytes."
        )
        raise typer.Exit(code=1)

    entries: List[dict] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)

    if not entries:
        console.print(f"[dim]No lineage entries recorded in '{_e(log_path)}'.[/dim]")
        raise typer.Exit(code=0)

    table = Table(title=f"Compile history: {adapter_path.name}")
    table.add_column("#", style="dim", justify="right")
    table.add_column("Compiled At")
    table.add_column("Backend")
    table.add_column("Program ID")
    table.add_column("Compiler")
    table.add_column("Compile (s)", justify="right")
    table.add_column("Parent Program ID")

    for i, entry in enumerate(entries, 1):
        compile_wall_s = entry.get("compile_wall_s")
        wall_str = f"{compile_wall_s:.3f}" if isinstance(compile_wall_s, (int, float)) else "-"
        table.add_row(
            str(i),
            _e(entry.get("compiled_at") or "-"),
            _e(entry.get("backend") or "-"),
            _e(entry.get("program_id") or "-"),
            _e(entry.get("compiler") or "-"),
            _e(wall_str),
            _e(entry.get("parent_program_id") or "-"),
        )

    console.print(table)


_LINT_SEVERITY_COLOR = {"error": "red", "warn": "yellow", "info": "cyan"}


@app.command(name="lint-spec")
def lint_spec_cmd(
    spec_text: Optional[str] = typer.Argument(
        None, help="Spec text to lint. Omit and use --file to read it from a file instead."
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", help="Read the spec from this file instead of the positional argument"
    ),
    schema: Optional[str] = typer.Option(
        None,
        "--schema",
        help="Pydantic response model to check, as 'module:ClassName' (enables the "
        "schema-all-required rule)",
    ),
    examples_file: Optional[Path] = typer.Option(
        None,
        "--examples",
        help="JSONL file of {\"input\": ..., \"output\": ...} pairs (enables the "
        "examples-single-form rule)",
    ),
    json_output: bool = typer.Option(False, "--json", help="Print findings as JSON"),
) -> None:
    """Lint a spec for failure modes measured against real compiled PAW adapters.

    Exits 1 only if a finding's severity is "error" (spec-too-long / spec-too-short);
    "warn" and "info" findings are printed but do not fail the command.
    """
    if spec_text is not None and file is not None:
        console.print("[bold red]Error:[/bold red] pass spec text or --file, not both.")
        raise typer.Exit(code=1)
    if file is not None:
        if not file.exists():
            console.print(f"[bold red]Error:[/bold red] spec file '{_e(file)}' does not exist.")
            raise typer.Exit(code=1)
        spec_content = file.read_text(encoding="utf-8")
    elif spec_text is not None:
        spec_content = spec_text
    else:
        console.print("[bold red]Error:[/bold red] provide spec text, or --file.")
        raise typer.Exit(code=1)

    schema_model = None
    if schema is not None:
        if ":" not in schema:
            console.print(
                "[bold red]Error:[/bold red] --schema must be 'module:ClassName', e.g. "
                "'myapp.models:Contact'."
            )
            raise typer.Exit(code=1)
        module_name, _, class_name = schema.partition(":")
        try:
            module = importlib.import_module(module_name)
            schema_model = getattr(module, class_name)
        except Exception as exc:
            console.print(f"[bold red]Error loading --schema:[/bold red] {_e(exc)}")
            raise typer.Exit(code=1)
        if not hasattr(schema_model, "model_fields"):
            console.print(
                f"[bold red]Error:[/bold red] {_e(schema)} is not a Pydantic BaseModel."
            )
            raise typer.Exit(code=1)

    examples_list: Optional[List[dict]] = None
    if examples_file is not None:
        if not examples_file.exists():
            console.print(
                f"[bold red]Error:[/bold red] examples file '{_e(examples_file)}' does not exist."
            )
            raise typer.Exit(code=1)
        examples_list = []
        for line in examples_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                examples_list.append(parsed)

    findings = lint_spec(spec_content, examples=examples_list, schema=schema_model)

    if json_output:
        console.print_json(
            data=[
                {"rule_id": f.rule_id, "severity": f.severity, "message": f.message}
                for f in findings
            ]
        )
    elif not findings:
        console.print("[bold green]No issues found.[/bold green]")
    else:
        for f in findings:
            color = _LINT_SEVERITY_COLOR[f.severity]
            # The "[rule_id]" brackets are literal text, not a markup tag -- run
            # through _e() as one unit (rather than escaping f.rule_id alone and
            # writing the brackets as bare f-string text) so Rich never sees them as
            # an attempted, unrecognized style tag and silently deletes the span
            # (exactly the class of bug _e()'s own docstring describes).
            console.print(
                f"[bold {_e(color)}]{_e(f.severity.upper())}[/bold {_e(color)}] "
                f"{_e('[' + f.rule_id + ']')} {_e(f.message)}"
            )

    if any(f.severity == "error" for f in findings):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


@app.command(name="clean")
def clean(
    cache_dir: Path = typer.Option(Path("./.paw"), "--cache-dir", "-c", help="Directory containing traces and adapters"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview files to delete without removing"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt (for scripted/CI use)"),
) -> None:
    """Purge local trace database and compiled adapter cache."""
    # PAW-CLI-01: cache_dir came straight from a CLI flag with no containment check at
    # all -- `paw-clean -c .` (deletes the caller's own project) or `paw-clean -c
    # /etc/my_app` (deletes an unrelated directory) would both unlink files with no
    # relation to a paw-kit cache (CWE-22/CWE-73). Restores Track 05's declared-but-
    # never-implemented invariant: never delete files outside the designated cache dir.
    try:
        resolved_cache = ensure_contained(cache_dir, Path.cwd(), label="--cache-dir")
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)

    if not resolved_cache.exists():
        console.print(f"[dim]Cache directory '{_e(cache_dir)}' does not exist. Nothing to clean.[/dim]")
        raise typer.Exit(code=0)

    files_to_remove = list(resolved_cache.glob("*"))
    if not files_to_remove:
        console.print(f"[dim]No cached artifacts found in '{_e(cache_dir)}'.[/dim]")
        raise typer.Exit(code=0)

    console.print(f"[bold yellow]{'Dry run: would remove' if dry_run else 'Purging'}[/bold yellow] {len(files_to_remove)} files in '{_e(cache_dir)}':")
    for file in files_to_remove:
        console.print(f"  - {_e(file.name)}")

    if dry_run:
        raise typer.Exit(code=0)

    # PAW-CLI-01, second half of the audit's remediation: require an explicit
    # confirmation before unlinking anything, so an accidental/automated invocation
    # (e.g. a Makefile or CI step) doesn't silently destroy files.
    if not yes and not typer.confirm(f"Permanently delete these {len(files_to_remove)} file(s)?"):
        console.print("[dim]Aborted -- no files were deleted.[/dim]")
        raise typer.Exit(code=0)

    for file in files_to_remove:
        try:
            if file.is_file():
                file.unlink()
        except Exception as exc:
            console.print(f"    [red]Failed to delete {_e(file.name)}: {_e(exc)}[/red]")

    console.print("[bold green]Cache cleaned successfully.[/bold green]")


@app.command(name="doctor")
def doctor(
    adapter: Optional[Path] = typer.Option(
        None, "--adapter", help="A .paw manifest to also check offline-readiness for"
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Skip every check that touches the network"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print results as a JSON list instead of a table"
    ),
) -> None:
    """Diagnose the local environment for running --backend real (ProgramAsWeightsBackend).

    Checks the SDK and llama_cpp installs, GPU visibility, PAW_API_KEY, upstream service
    health, the local base-model cache, and cached compiled programs -- each PASS / WARN
    / FAIL with a one-line remedy. Exits 0 if nothing FAILed, 1 otherwise.
    """
    from paw_kit.doctor import run_checks

    results = run_checks(offline=offline, adapter_path=str(adapter) if adapter else None)

    if json_output:
        # Deliberately the bare `print`, not `console.print`: this output is meant to be
        # machine-parsed, so it must not be subject to Rich's markup parsing (a detail
        # string containing "[" would otherwise be silently mangled) or its terminal-width
        # line wrapping.
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        table = Table(title="paw-kit doctor")
        table.add_column("Check", style="cyan", no_wrap=True)
        table.add_column("Status")
        table.add_column("Detail")
        table.add_column("Remedy", overflow="fold")
        status_style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}
        for res in results:
            style = status_style.get(res.status, "white")
            table.add_row(
                _e(res.name),
                f"[{style}]{_e(res.status)}[/{style}]",
                _e(res.detail),
                _e(res.remedy) if res.remedy else "[dim]-[/dim]",
            )
        console.print(table)

    if any(res.status == "FAIL" for res in results):
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


export_app = typer.Typer(help="Export PAW adapters and traces to external formats")
app.add_typer(export_app, name="export")


@app.command(name="serve")
def serve(
    ctx: typer.Context,
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    backend_type: str = typer.Option("mock", "--backend", "-b", help="Backend engine: mock | real"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        "-k",
        envvar="PAW_API_KEY",
        help="Optional secret bearer token for authentication. Prefer setting the "
        "PAW_API_KEY environment variable instead of this flag where possible.",
    ),
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Disable authentication entirely (PAW-SERVE-01). Without this flag, an "
        "ephemeral bearer token is generated and printed to stderr if no --api-key "
        "or PAW_API_KEY is configured.",
    ),
    warm: bool = typer.Option(
        False,
        "--warm/--no-warm",
        help="Run one inference before binding so /ready is 200 from the start, "
        "instead of 503 until the first real request completes (a cold Qwen3-0.6B "
        "load can take up to ~110s -- see measurements/README.md).",
    ),
) -> None:
    """Serve an adapter over HTTP with OpenAI- and Anthropic-compatible endpoints."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{_e(adapter_path)}' does not exist.")
        raise typer.Exit(code=1)

    # PAW-CLI-07: a value passed on the command line lands in argv, which is visible
    # to any other local user via /proc/<pid>/cmdline or `ps aux`, and often ends up
    # recorded in shell history too. `envvar="PAW_API_KEY"` above lets the flag be
    # skipped entirely in favor of the environment; this only warns -- --api-key must
    # keep working for scripted/CI callers that can't set process environment.
    # Compared by enum *name*, not identity/equality against click.core.ParameterSource
    # directly: Typer vendors its own fork of click (typer._click), so ctx here yields
    # a distinct (if identically-named) ParameterSource enum class from the one a
    # top-level `import click` would resolve to.
    api_key_source = ctx.get_parameter_source("api_key")
    if api_key and api_key_source is not None and api_key_source.name == "COMMANDLINE":
        console.print(
            "[bold yellow]Warning:[/bold yellow] --api-key was passed on the command line; "
            "it is visible to other local users (ps aux, /proc/<pid>/cmdline) and may be "
            "recorded in shell history. Prefer setting the PAW_API_KEY environment "
            "variable instead."
        )

    backend = _resolve_cli_backend(backend_type)
    actual_type = backend_label(backend)
    console.print(f"[bold green]Launching PAW microservice on http://{_e(host)}:{port}[/bold green]")
    console.print(f"  [cyan]Adapter:[/cyan] {_e(adapter_path)}")
    console.print(f"  [cyan]Backend:[/cyan] {_e(actual_type)}")
    if api_key:
        console.print("  [yellow]Authentication:[/yellow] Bearer token active")
    elif allow_anonymous:
        console.print("  [bold red]Authentication:[/bold red] DISABLED (--allow-anonymous)")
    else:
        console.print("  [yellow]Authentication:[/yellow] ephemeral token (see stderr on startup)")
    console.print("  [dim]Endpoints: /v1/chat/completions, /v1/messages, /invoke, /health, /ready, /metrics[/dim]")

    from paw_kit.serve.server import serve_adapter

    serve_adapter(
        adapter_path=adapter_path,
        host=host,
        port=port,
        backend=backend,
        api_key=api_key,
        allow_anonymous=allow_anonymous,
        warm=warm,
    )


@export_app.command(name="docker")
def export_docker_cmd(
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
    out_dir: Path = typer.Option(Path("./docker"), "--out-dir", "-o", help="Output directory for Docker assets"),
    backend_type: str = typer.Option(
        "real",
        "--backend",
        "-b",
        help="Backend the generated container actually serves: mock | real. Threaded "
        "into the generated CMD's --backend flag, requirements.txt (real pulls in the "
        "SDK extra), and the HEALTHCHECK/comments, so none of them describe a backend "
        "the container doesn't run. Defaults to real -- a container that only ever "
        "serves the mock is a demo, not a deployment.",
    ),
) -> None:
    """Generate a Dockerfile and docker-compose scaffold for serving an adapter."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{_e(adapter_path)}' does not exist.")
        raise typer.Exit(code=1)

    resolved_backend = backend_type.strip().lower()
    if resolved_backend not in ("mock", "real"):
        console.print(
            f"[bold red]Error:[/bold red] Unknown --backend {_e(repr(backend_type))}. Expected 'mock' or 'real'."
        )
        raise typer.Exit(code=1)

    from paw_kit.serve.docker import export_docker_scaffold

    try:
        dest = export_docker_scaffold(adapter_path=adapter_path, output_dir=out_dir, backend=resolved_backend)
        console.print(f"[bold green]Docker deployment assets successfully generated in:[/bold green] {_e(dest.resolve())}")
        console.print("  - Dockerfile")
        console.print("  - .dockerignore")
        console.print("  - docker-compose.yml")
        console.print("  - README.md")
    except Exception as exc:
        console.print(f"[bold red]Error exporting Docker assets:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)


@export_app.command(name="dataset")
def export_dataset_cmd(
    db_path: Path = typer.Option(Path("./.paw/traces.db"), "--db", help="Path to SQLite trace database"),
    out_file: Path = typer.Option(Path("traces.jsonl"), "--out", "-o", help="Path to output JSONL file"),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite --out without prompting for confirmation"
    ),
) -> None:
    """Export traced SQLite teacher-student interaction pairs to standard JSONL format."""
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Trace database '{_e(db_path)}' does not exist.")
        raise typer.Exit(code=1)

    # PAW-CLI-03: refuse a destination that doesn't even look like the format we're
    # about to write, and confirm before silently clobbering an existing file --
    # neither was checked before.
    if out_file.suffix != ".jsonl":
        console.print(f"[bold red]Error:[/bold red] --out must have a '.jsonl' extension: {_e(out_file)}")
        raise typer.Exit(code=1)
    if out_file.exists() and not force:
        if not typer.confirm(f"{_e(out_file)} already exists. Overwrite?"):
            console.print("[dim]Aborted -- file was not overwritten.[/dim]")
            raise typer.Exit(code=0)

    import sqlite3

    # PAW-CLI-04: the query previously named columns ("input", "output") that no
    # TraceDB schema has ever had -- the real `traces` table (jit/db.py) has
    # input_payload/teacher_output. `cursor.execute` raised sqlite3.OperationalError
    # against any real trace DB, empty or populated (the column is resolved at
    # prepare time), always caught by the bare `except Exception` below and reported
    # as a generic export failure -- `paw export dataset` had never once worked
    # against production data. Fixed to match the actual schema.
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT input_payload, teacher_output FROM traces ORDER BY timestamp ASC;")
            rows = cursor.fetchall()
    except Exception as exc:
        console.print(f"[bold red]Error exporting dataset:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)

    if not rows:
        console.print(f"[yellow]Warning:[/yellow] No traces found in {_e(db_path)}.")
        # PAW-CLI-04: this Exit used to sit inside the write's try/except below.
        # typer.Exit subclasses RuntimeError, so the bare `except Exception` there
        # swallowed it and re-raised as exit 1 -- an empty trace DB printed this
        # warning *and* "Error exporting dataset:" and still exited 1. Raised here,
        # outside any try, an empty DB now exits 0 as the message implies.
        raise typer.Exit(code=0)

    try:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        # PAW-CLI-05: 0600 from the moment of creation via os.open's mode, not a bare
        # open(..., "w") -- which is subject to the process umask (commonly 0644,
        # world-readable) and would leave exported trace content (potentially
        # unredacted prompts/PII, see PAW-JIT-02) briefly world-readable before any
        # later chmod.
        fd = os.open(str(out_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for inp, out in rows:
                record = {
                    "messages": [
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": out},
                    ]
                }
                f.write(json.dumps(record) + "\n")

        console.print(f"[bold green]Successfully exported {len(rows)} traces to:[/bold green] {_e(out_file.resolve())}")
    except Exception as exc:
        console.print(f"[bold red]Error exporting dataset:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)




def _short_timestamp(value: Optional[str]) -> str:
    """Trim an ISO timestamp to seconds for the report table, or em-dash if absent."""
    if not value:
        return "-"
    return str(value)[:19].replace("T", " ")


def _format_agreement(agreement: dict) -> str:
    """`0.85 (17/20)` or `-` when nothing has been compared yet."""
    if agreement.get("rate") is None:
        return "-"
    return f"{agreement['rate']:.2f} ({agreement['agree']}/{agreement['samples']})"


def _render_report_table(reports: List[dict]) -> Table:
    """Build the one-row-per-task report table. Factored out so it is unit-testable."""
    table = Table(title="paw-kit task report")
    table.add_column("Task", style="dim", no_wrap=True)
    table.add_column("State")
    table.add_column("Calls", justify="right")
    table.add_column("Agreement")
    table.add_column("Fail-open", justify="right")
    table.add_column("Promoted")
    table.add_column("Demoted")

    state_styles = {"ready": "green", "shadow": "yellow", "failed": "red", "compiling": "cyan"}
    for report in reports:
        agreement = report.get("agreement") or {}
        style = state_styles.get(report.get("status", ""), "white")
        window = agreement.get("window") or 0
        phase = agreement.get("phase")
        agreement_cell = _format_agreement(agreement)
        if phase and window:
            agreement_cell = f"{agreement_cell} {phase}/{window}"
        table.add_row(
            _e(str(report.get("task_id", ""))[:12] + "..."),
            f"[{style}]{_e(report.get('status', 'tracing'))}[/{style}]",
            str(report.get("call_count", 0)),
            _e(agreement_cell),
            str(report.get("fail_open_count", 0)),
            _e(_short_timestamp(report.get("promoted_at"))),
            _e(_short_timestamp(report.get("demoted_at"))),
        )
    return table


@app.command(name="report")
def report_cmd(
    db_path: Path = typer.Option(Path("./.paw/traces.db"), "--db", help="Path to SQLite trace database"),
    task: Optional[str] = typer.Option(None, "--task", help="Show only this task_id"),
    limit: int = typer.Option(5, "--disagreements", "-n", help="Disagreements to show per task"),
    json_out: bool = typer.Option(False, "--json", help="Machine-readable JSON instead of a table"),
) -> None:
    """Report @compile_on_hit task state, shadow-mode agreement and recent disagreements.

    NOTE: this command OPENS the trace database with the library's own TraceDB, which
    migrates a pre-v2 file to the current schema in place. It therefore *writes*. That
    is deliberate: the alternative is a hand-rolled read-only connection duplicating
    schema knowledge that would rot.
    """
    import sqlite3

    from paw_kit.jit.db import TraceDB

    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Trace database '{_e(db_path)}' does not exist.")
        raise typer.Exit(code=1)

    db = None
    try:
        db = TraceDB(db_path=str(db_path))
        task_ids = [task] if task else db.list_task_ids()
        reports = [db.get_task_report(task_id) for task_id in task_ids]
        for report in reports:
            report["last_disagreements"] = db.get_recent_disagreements(report["task_id"], limit)
    except sqlite3.OperationalError as exc:
        # A read-only filesystem or a locked database: the schema migration above (and
        # its PRAGMA user_version write) cannot run.
        console.print(
            f"[bold red]Error reading trace database:[/bold red] {_e(exc)}\n"
            "[dim]`paw-kit report` opens the database for writing, because opening it "
            "migrates it to the current schema.[/dim]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        console.print(f"[bold red]Error reading trace database:[/bold red] {_e(exc)}")
        raise typer.Exit(code=1)
    finally:
        if db is not None:
            db.close()

    if json_out:
        typer.echo(json.dumps({"db": str(db_path), "tasks": reports}, indent=2, default=str))
        return

    if not reports:
        console.print(f"[yellow]No tasks recorded in {_e(db_path)}.[/yellow]")
        return

    console.print(_render_report_table(reports))

    for report in reports:
        disagreements = report.get("last_disagreements") or []
        if not disagreements:
            continue
        lines = []
        for entry in disagreements:
            verdict = entry.get("verdict")
            detail = f" ({entry.get('error_type')})" if entry.get("error_type") else ""
            lines.append(
                f"[bold]{_e(verdict)}{_e(detail)}[/bold] [dim]{_e(entry.get('phase'))} "
                f"{_e(_short_timestamp(entry.get('timestamp')))}[/dim]\n"
                f"  in:      {_e(_truncate(entry.get('input_payload')))}\n"
                f"  teacher: {_e(_truncate(entry.get('teacher_output')))}\n"
                f"  adapter: {_e(_truncate(entry.get('adapter_output')))}"
            )
        console.print(
            Panel(
                "\n\n".join(lines),
                title=f"Last disagreements: {_e(str(report['task_id'])[:12])}...",
                border_style="yellow",
            )
        )


def _truncate(value: Optional[str], length: int = 120) -> str:
    """Clip persisted (possibly redacted) text for display."""
    if value is None:
        return "-"
    text = str(value)
    return text if len(text) <= length else text[:length] + "..."


def inspect_cli() -> None:
    """Direct entrypoint for paw-inspect command."""
    typer.run(inspect)


def clean_cli() -> None:
    """Direct entrypoint for paw-clean command."""
    typer.run(clean)


def serve_cli() -> None:
    """Direct entrypoint for paw-serve command."""
    typer.run(serve)


if __name__ == "__main__":
    app()
