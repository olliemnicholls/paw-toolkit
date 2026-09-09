"""Typer and Rich command-line interface for paw-toolkit."""

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, List, Optional
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
import typer

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.backend.programasweights import ProgramAsWeightsBackend
from paw_kit.pathsafety import ensure_contained
from paw_kit.serve.server import backend_label
from paw_kit.test.active import run_active_learning_loop
from paw_kit.test.runner import TestRunner
from paw_kit.test.suite import load_suite

console = Console()

# PAW-CLI-06: cap on how much of a candidate adapter file `paw-inspect` will attempt
# to parse as JSON. See `inspect()` below.
_MAX_INSPECT_FILE_BYTES = 50 * 1024 * 1024

app = typer.Typer(help="PAW-Kit: Production Runtime & Reliability Toolkit for Program-as-Weights")
test_app = typer.Typer(help="paw-test: Test runner and active-learning self-healing suite")
test_app.__test__ = False  # Prevent pytest from treating Typer instance as a test suite


@test_app.callback()
def test_app_main() -> None:
    """paw-test: Test runner and active-learning self-healing suite."""


def _declared_adapter_backend(adapter_path: str) -> Optional[str]:
    """Return the `backend` an existing .paw manifest declares, or None.

    None means "no opinion": the file is absent, unreadable, not JSON, or carries no
    `backend` key. Callers must treat that as "unknown", never as "mock" -- this is used
    to decide whether overwriting the file is safe, so an unreadable manifest must not
    read as permission.
    """
    try:
        raw = Path(adapter_path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    declared = manifest.get("backend")
    return declared if isinstance(declared, str) else None


def _resolve_cli_backend(backend_type: str) -> Any:
    """Resolve the --backend flag to a concrete backend.

    `real` resolves to `ProgramAsWeightsBackend` -- the official upstream SDK, and the
    only backend in paw-kit proven end to end against a real model (see
    `measurements/`). It previously resolved to `RealPAWBackend`, whose `compile()` and
    `infer()` raise `NotImplementedError`, so this function returned `MockPAWBackend()`
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
        f"([dim]compiler={backend.compiler}[/dim])"
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
            console.print(f"  [dim]Raw:[/dim]       {text}")
            console.print(f"  [green]Sanitized:[/green] {result.sanitized_text}")
            if result.entities:
                for e in result.entities:
                    console.print(f"  [yellow]  → {e.entity_type}:[/yellow] {e.value}")

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
        console.print(f"[bold red]Unknown scenario:[/bold red] '{scenario}'. Choose 'triage' or 'pii'.")
        raise typer.Exit(code=1)


@test_app.command(name="check")
@app.command(name="check")
def check(
    suite_path: Path = typer.Argument(..., help="Path to declarative suite.yaml specification"),
    backend_type: str = typer.Option("mock", "--backend", "-b", help="Backend engine: mock | real"),
    auto_recompile: Optional[bool] = typer.Option(
        None, "--auto-recompile/--no-auto-recompile", help="Enable active learning (overrides suite.yaml if set)"
    ),
) -> None:
    """Run test suite assertions and active-learning self-healing loop on a .paw adapter."""
    if not suite_path.exists():
        console.print(f"[bold red]Error:[/bold red] Suite file '{suite_path}' does not exist.")
        raise typer.Exit(code=1)

    try:
        config = load_suite(str(suite_path))
    except Exception as exc:
        console.print(f"[bold red]Error parsing suite:[/bold red] {exc}")
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
                f"--backend {backend_type}. Recompiling would submit a paid upstream "
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
    if config.active_learning.auto_recompile:
        existing_backend = _declared_adapter_backend(config.adapter_path)
        if existing_backend is not None and existing_backend != "mock":
            config.active_learning.auto_recompile = False
            console.print(
                "[bold yellow]Note:[/bold yellow] auto-recompile is disabled: "
                f"[cyan]{escape(config.adapter_path)}[/cyan] is a "
                f"[bold]{escape(existing_backend)}[/bold] adapter, and recompiling would "
                "replace it with a mock stub built from this CLI's demo teacher.\n"
                "  Running assertions read-only. Recompile it with the backend that "
                "produced it, from code."
            )

    console.print(
        f"[bold cyan]Running paw.test check on:[/bold cyan] {config.task_name} "
        f"([dim]{escape(config.adapter_path)}[/dim]) [dim](backend: {actual_backend})[/dim]"
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
            f"({report.passed_cases}/{report.total_cases}) [dim](backend: {actual_backend})[/dim]"
        )
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

    if al_report.is_success:
        console.print(f"\n[bold green][SUCCESS][/bold green] All assertions passed! (Iterations: {al_report.iterations_run})")
        if al_report.recompiled:
            console.print(f"  [cyan][UPDATE][/cyan] Recompiled and updated artifact: {config.adapter_path}")
        raise typer.Exit(code=0)
    else:
        console.print(f"\n[bold red][FAIL][/bold red] Assertions failed after {al_report.iterations_run} iteration(s).")
        raise typer.Exit(code=1)


@app.command(name="inspect")
def inspect(
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
) -> None:
    """Inspect metadata, task spec, and file properties of a .paw adapter artifact."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{adapter_path}' does not exist.")
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
    metadata = {}
    is_json = False
    if adapter_path.is_file() and size_bytes <= _MAX_INSPECT_FILE_BYTES:
        try:
            with open(adapter_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                is_json = True
        except Exception:
            pass

    table = Table(title=f"PAW Adapter: {adapter_path.name}")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("File Path", str(adapter_path.resolve()))
    table.add_row("File Size", f"{size_bytes} bytes ({size_bytes / 1024:.2f} KB)")
    table.add_row("Format", "JSON Simulation" if is_json else "Binary / Raw Weights")

    if is_json:
        table.add_row("Specification", metadata.get("spec", "N/A"))
        table.add_row("Backend", metadata.get("backend", "mock"))
        table.add_row("Examples Count", str(metadata.get("examples_count", len(metadata.get("examples", [])))))

    console.print(table)


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
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not resolved_cache.exists():
        console.print(f"[dim]Cache directory '{cache_dir}' does not exist. Nothing to clean.[/dim]")
        raise typer.Exit(code=0)

    files_to_remove = list(resolved_cache.glob("*"))
    if not files_to_remove:
        console.print(f"[dim]No cached artifacts found in '{cache_dir}'.[/dim]")
        raise typer.Exit(code=0)

    console.print(f"[bold yellow]{'Dry run: would remove' if dry_run else 'Purging'}[/bold yellow] {len(files_to_remove)} files in '{cache_dir}':")
    for file in files_to_remove:
        console.print(f"  - {file.name}")

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
            console.print(f"    [red]Failed to delete {file.name}: {exc}[/red]")

    console.print("[bold green]Cache cleaned successfully.[/bold green]")


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
) -> None:
    """Launch high-performance OpenAI & Anthropic compatible HTTP microservice."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{adapter_path}' does not exist.")
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
    console.print(f"[bold green]Launching PAW microservice on http://{host}:{port}[/bold green]")
    console.print(f"  [cyan]Adapter:[/cyan] {adapter_path}")
    console.print(f"  [cyan]Backend:[/cyan] {actual_type}")
    if api_key:
        console.print("  [yellow]Authentication:[/yellow] Bearer token active")
    elif allow_anonymous:
        console.print("  [bold red]Authentication:[/bold red] DISABLED (--allow-anonymous)")
    else:
        console.print("  [yellow]Authentication:[/yellow] ephemeral token (see stderr on startup)")
    console.print("  [dim]Endpoints: /v1/chat/completions, /v1/messages, /invoke, /health, /metrics[/dim]")

    from paw_kit.serve.server import serve_adapter

    serve_adapter(
        adapter_path=adapter_path,
        host=host,
        port=port,
        backend=backend,
        api_key=api_key,
        allow_anonymous=allow_anonymous,
    )


@export_app.command(name="docker")
def export_docker_cmd(
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
    out_dir: Path = typer.Option(Path("./docker"), "--out-dir", "-o", help="Output directory for Docker assets"),
) -> None:
    """Generate production-ready Dockerfile and docker-compose deployment assets."""
    if not adapter_path.exists():
        console.print(f"[bold red]Error:[/bold red] Adapter file '{adapter_path}' does not exist.")
        raise typer.Exit(code=1)

    from paw_kit.serve.docker import export_docker_scaffold

    try:
        dest = export_docker_scaffold(adapter_path=adapter_path, output_dir=out_dir)
        console.print(f"[bold green]Docker deployment assets successfully generated in:[/bold green] {dest.resolve()}")
        console.print("  - Dockerfile")
        console.print("  - .dockerignore")
        console.print("  - docker-compose.yml")
        console.print("  - README.md")
    except Exception as exc:
        console.print(f"[bold red]Error exporting Docker assets:[/bold red] {exc}")
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
        console.print(f"[bold red]Error:[/bold red] Trace database '{db_path}' does not exist.")
        raise typer.Exit(code=1)

    # PAW-CLI-03: refuse a destination that doesn't even look like the format we're
    # about to write, and confirm before silently clobbering an existing file --
    # neither was checked before.
    if out_file.suffix != ".jsonl":
        console.print(f"[bold red]Error:[/bold red] --out must have a '.jsonl' extension: {out_file}")
        raise typer.Exit(code=1)
    if out_file.exists() and not force:
        if not typer.confirm(f"{out_file} already exists. Overwrite?"):
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
        console.print(f"[bold red]Error exporting dataset:[/bold red] {exc}")
        raise typer.Exit(code=1)

    if not rows:
        console.print(f"[yellow]Warning:[/yellow] No traces found in {db_path}.")
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

        console.print(f"[bold green]Successfully exported {len(rows)} traces to:[/bold green] {out_file.resolve()}")
    except Exception as exc:
        console.print(f"[bold red]Error exporting dataset:[/bold red] {exc}")
        raise typer.Exit(code=1)


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
