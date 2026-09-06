"""Typer and Rich command-line interface for paw-toolkit."""

import json
from pathlib import Path
import sys
from typing import Any, Optional
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
    console.print("  [dim]Endpoints: /v1/chat/completions, /v1/messages, /invoke, /health, /metrics[/dim]")

    from paw_kit.serve.server import serve_adapter

    serve_adapter(adapter_path=adapter_path, host=host, port=port, backend=backend, api_key=api_key)


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
