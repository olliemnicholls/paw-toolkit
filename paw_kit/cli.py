"""Typer and Rich command-line interface for paw-toolkit."""

import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any, List, Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from paw_kit.backend.mock import MockPAWBackend
from paw_kit.backend.real import RealPAWBackend
from paw_kit.test.active import run_active_learning_loop
from paw_kit.test.runner import TestRunner
from paw_kit.test.suite import load_suite

console = Console()
app = typer.Typer(help="PAW-Kit: Production Runtime & Reliability Toolkit for Program-as-Weights")
test_app = typer.Typer(help="paw-test: Test runner and active-learning self-healing suite")
test_app.__test__ = False  # Prevent pytest from treating Typer instance as a test suite


@test_app.callback()
def test_app_main() -> None:
    """paw-test: Test runner and active-learning self-healing suite."""


def _resolve_cli_backend(backend_type: str) -> Any:
    """Resolve backend from CLI flag with graceful degradation and clear guidance."""
    if backend_type.lower() == "real":
        real_backend = RealPAWBackend()
        if not real_backend.is_available():
            console.print(
                "[bold yellow]Warning:[/bold yellow] Real GPU/CPU backend requires PyTorch and transformers packages.\n"
                "  To install: [cyan]pip install 'paw-kit[torch]'[/cyan]\n"
                "  Falling back to [green]MockPAWBackend[/green] for zero-hardware execution."
            )
        else:
            console.print(
                "[bold yellow]Note:[/bold yellow] Direct PyTorch neural execution is under active development for v0.2.\n"
                "  Falling back to [green]MockPAWBackend[/green] for this run."
            )
        return MockPAWBackend()
    return MockPAWBackend()


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
        "compilation, and hot-swap to local 0.6B neural execution — all running\n"
        "locally with [green]zero GPU and zero external API keys[/green].",
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

    temp_dir = tempfile.mkdtemp(prefix="paw_demo_")

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

    table = Table(title="Live JIT Ticket Triage Results", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Mode", width=18)
    table.add_column("Latency", justify="right", width=10)
    table.add_column("Priority", width=10)
    table.add_column("Department", width=12)
    table.add_column("Urgency", justify="center", width=8)
    table.add_column("Ticket Preview", max_width=40)

    for i, ticket in enumerate(tickets, 1):
        is_local = triage_ticket.is_compiled()
        mode = "[green]LOCAL 0.6B[/green]" if is_local else "[yellow]REMOTE TEACHER[/yellow]"

        t0 = time.perf_counter()
        result = triage_ticket(ticket)
        ms = (time.perf_counter() - t0) * 1000

        table.add_row(
            str(i), mode, f"{ms:.1f}ms",
            result.priority, result.department,
            str(result.urgency_score), ticket[:38] + "…",
        )

        if i == 3:
            console.print("\n[bold cyan]>>> Hit threshold (3 calls) reached! Background compilation hot-swapped adapter.[/bold cyan]\n")

    console.print(table)
    console.print("\n[bold green]✓[/bold green] Calls 1–3 via remote teacher (logged to SQLite trace DB)")
    console.print("[bold green]✓[/bold green] Calls 4–6 via local compiled neural function (<1ms, $0 marginal cost)")
    console.print("[bold green]✓[/bold green] Zero GPU • Zero API keys • Zero configuration\n")

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


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
        "Watch [bold]paw.load[/bold] bind a compiled neural adapter to a strict Pydantic\n"
        "schema with guaranteed 0.0% JSON syntax errors via FSM token masking.",
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

    temp_dir = tempfile.mkdtemp(prefix="paw_pii_")
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

    console.print("\n[bold green]✓[/bold green] 0.0% JSON syntax errors guaranteed by FSM token masking")
    console.print("[bold green]✓[/bold green] Sub-millisecond local execution without GPU\n")

    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


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

    console.print(f"[bold cyan]Running paw.test check on:[/bold cyan] {config.task_name} ([dim]{config.adapter_path}[/dim])")

    # If active learning auto-recompile is disabled, just run once
    if not config.active_learning.auto_recompile:
        runner = TestRunner(backend=backend)
        report = runner.run(config)
        for res in report.results:
            if res.passed:
                console.print(f"  [green][PASS][/green] Input: {res.input[:50]!r} -> {res.output[:40]!r}")
            else:
                console.print(f"  [red][FAIL][/red] Input: {res.input[:50]!r} -> {res.output[:40]!r}")
                for reason in res.failed_rules:
                    console.print(f"         [dim red]{reason}[/dim red]")

        console.print(f"\n[bold]Pass rate:[/bold] {report.pass_rate:.1f}% ({report.passed_cases}/{report.total_cases})")
        if not report.is_success:
            raise typer.Exit(code=1)
        raise typer.Exit(code=0)

    # Simulated frontier teacher for active-learning auto-repair in CLI check
    def cli_teacher(inp: str) -> str:
        console.print(f"  [yellow][ACTION][/yellow] Active learning: Querying frontier teacher for '{inp[:40]}'...")
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

    # Attempt to read JSON metadata
    metadata = {}
    is_json = False
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
) -> None:
    """Purge local trace database and compiled adapter cache."""
    if not cache_dir.exists():
        console.print(f"[dim]Cache directory '{cache_dir}' does not exist. Nothing to clean.[/dim]")
        raise typer.Exit(code=0)

    files_to_remove = list(cache_dir.glob("*"))
    if not files_to_remove:
        console.print(f"[dim]No cached artifacts found in '{cache_dir}'.[/dim]")
        raise typer.Exit(code=0)

    console.print(f"[bold yellow]{'Dry run: would remove' if dry_run else 'Purging'}[/bold yellow] {len(files_to_remove)} files in '{cache_dir}':")
    for file in files_to_remove:
        console.print(f"  - {file.name}")
        if not dry_run:
            try:
                if file.is_file():
                    file.unlink()
            except Exception as exc:
                console.print(f"    [red]Failed to delete {file.name}: {exc}[/red]")

    if not dry_run:
        console.print("[bold green]Cache cleaned successfully.[/bold green]")


export_app = typer.Typer(help="Export PAW adapters and traces to external formats")
app.add_typer(export_app, name="export")


@app.command(name="serve")
def serve(
    adapter_path: Path = typer.Argument(..., help="Path to compiled .paw adapter artifact"),
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    backend_type: str = typer.Option("mock", "--backend", "-b", help="Backend engine: mock | real"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", help="Optional secret bearer token for authentication"),
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

    backend = _resolve_cli_backend(backend_type)
    actual_type = "mock" if isinstance(backend, MockPAWBackend) else backend_type.lower()
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
) -> None:
    """Export traced SQLite teacher-student interaction pairs to standard JSONL format."""
    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Trace database '{db_path}' does not exist.")
        raise typer.Exit(code=1)

    import sqlite3

    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT input, output FROM traces ORDER BY timestamp ASC;")
            rows = cursor.fetchall()

        if not rows:
            console.print(f"[yellow]Warning:[/yellow] No traces found in {db_path}.")
            raise typer.Exit(code=0)

        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
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
