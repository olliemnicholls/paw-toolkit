"""Smoke tests for runnable examples."""

from pathlib import Path
import pytest
from typer.testing import CliRunner

from paw_kit.cli import app, test_app as paw_test_app
import examples.triage_ticket.run as triage_demo
import examples.pii_scrubber.run as pii_demo
import examples.date_normalizer.run as date_demo

runner = CliRunner()


def test_triage_ticket_example_smoke() -> None:
    """Verify triage_ticket runnable example executes cleanly."""
    triage_demo.main()


def test_pii_scrubber_example_smoke() -> None:
    """Verify pii_scrubber runnable example executes cleanly."""
    pii_demo.main()


def test_date_normalizer_example_smoke() -> None:
    """Verify date_normalizer runnable example executes cleanly."""
    date_demo.main()


def test_date_normalizer_cli_invocation() -> None:
    """Verify paw-test check runs on date_normalizer suite.yaml via CLI."""
    suite_path = Path(__file__).parent.parent / "examples" / "date_normalizer" / "suite.yaml"
    result = runner.invoke(paw_test_app, ["check", str(suite_path)])
    assert result.exit_code == 0
    assert "All assertions passed" in result.output
