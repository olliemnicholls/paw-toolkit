"""Golden-output snapshots for the paw-kit CLI.

WHAT THIS IS
------------
One checked-in file per case under `tests/golden/`, holding the exact stdout (and
`--json` artifact, where a command writes one) that today's CLI produces for a fixed
fixture. Each case is re-run here and compared byte for byte.

These snapshots exist because `paw_kit/test/runner.py`, `judge.py`, `compare.py` and
`cli.py` produce output that is very hard to review by eye: a summary line that quietly
loses a column, or a `--json` artifact that loses a field, passes a diff review
unnoticed. A snapshot turns that into a test failure with a readable diff.

**THESE SNAPSHOTS RECORD CURRENT BEHAVIOUR INCLUDING ITS KNOWN DEFECTS.** That is
deliberate. The 2026-09-11 bug hunt filed several findings that are *about* misleading
CLI output, so some of the text pinned here is text a later track is expected to change.
A snapshot is not an endorsement. When a track intentionally changes an output it
regenerates that snapshot in its own diff, where a reviewer sees the before and the
after side by side -- which is the whole point. Do not "fix" a snapshot to match a
change you made without looking at the diff; that is the one way to make this suite
worthless. Where a snapshot captures something that looks wrong, there is a comment on
its case below naming it. Those comments are observations, not TODOs for this file.

REGENERATING
------------
    PAW_GOLDEN_UPDATE=1 <campaign.sh test <track>> tests/test_golden_cli.py

See `tests/golden/README.md`.

DETERMINISM
-----------
Every case runs in-process through Typer's `CliRunner` against `MockPAWBackend` or
hand-written fixture files: no GPU, no network, no API key, no subprocess. What genuine
nondeterminism remains is removed by the narrowly-scoped scrubbers in the
"scrubbers" section below -- each one documents exactly what it hides and why. Nothing
else is normalised: if a pass rate, a column, a field name, an ordering or an exit code
changes, these tests fail.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

import pytest
from typer.testing import CliRunner

# Imported under an alias: pytest tries to *collect* a module-level name starting
# with `test_`, and warns that a Typer app is not a function.
from paw_kit.cli import app
from paw_kit.cli import test_app as paw_test_app

GOLDEN_DIR = Path(__file__).parent / "golden"

#: Set `PAW_GOLDEN_UPDATE=1` to rewrite every snapshot from the current behaviour
#: instead of asserting against it. Read at import time on purpose, so a half-updated
#: run is not possible.
UPDATE_SNAPSHOTS = os.environ.get("PAW_GOLDEN_UPDATE") == "1"


# --------------------------------------------------------------------------------------
# Deterministic environment
# --------------------------------------------------------------------------------------

#: Console width for every case. Pinned wide enough that no fixture path or spec string
#: wraps, which keeps line-wrapping out of the scrubbers entirely.
#:
#: It has to be pinned in two places, because Rich and Typer read it at different times:
#:
#: * `--help` is rendered by `typer.rich_utils`, which builds a fresh `Console` per call
#:   and so picks up `COLUMNS` from the environment override on `invoke()`.
#: * every other command prints through `paw_kit.cli.console`, a module-level `Console()`
#:   built at *import* time. Rich resolves `COLUMNS` into `Console._width` inside
#:   `__init__` (verified against rich 15.0), so by the time a test sets the variable that
#:   console's width is already fixed to whatever the developer's terminal was. The
#:   `pinned_console` fixture below replaces it instead.
#:
#: Verified: the whole suite passes unchanged under `COLUMNS=40 LINES=10 FORCE_COLOR=1
#: ANTHROPIC_API_KEY=... TZ=Asia/Tokyo`.
_COLUMNS = 120

#: Every case's working directory is padded to exactly this many characters (see
#: `golden_workdir`). Rich sizes table columns from the *length* of the longest cell, so
#: a working-directory path whose length varies between runs (pytest's tmp dir counter
#: rolling from 9 to 10, a different $TMPDIR on CI) would move table borders even after
#: the path text itself was scrubbed. Pinning the length pins the borders; the borders
#: then still move if a *real* column-width change happens, which is what we want to see.
_WORKDIR_PATH_LEN = 64

_BASE_ENV: Dict[str, Optional[str]] = {
    "COLUMNS": str(_COLUMNS),
    "LINES": "40",
    # Rich emits SGR codes when FORCE_COLOR is set even for a non-tty stream. Ask for
    # none; `strip_ansi` below is the belt to this braces.
    "NO_COLOR": "1",
    "FORCE_COLOR": None,
    # `paw-test judge` branches on ANTHROPIC_API_KEY's presence, and `_resolve_cli_backend`
    # on PAW_API_KEY's. A developer with either exported must get the same snapshots as
    # CI, so both are removed unless a case explicitly sets one.
    "ANTHROPIC_API_KEY": None,
    "PAW_API_KEY": None,
}

runner = CliRunner()


@pytest.fixture(autouse=True)
def pinned_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `paw_kit.cli.console` with one of a pinned width.

    See `_COLUMNS`: the real module-level console baked the developer's terminal width in
    at import time, so without this the snapshots record the window they were generated
    in. Only the *width* and colour are pinned -- everything else is a default `Console`,
    and every `console.print` call in `cli.py` (markup, tables, panels, `print_json`) runs
    for real.
    """
    from rich.console import Console

    monkeypatch.setattr("paw_kit.cli.console", Console(width=_COLUMNS, height=40, no_color=True))


@pytest.fixture()
def golden_workdir(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fixed-length, resolved, empty directory that is also the process CWD.

    Fixed length: see `_WORKDIR_PATH_LEN`. Resolved: `paw-inspect` and `paw-kit export`
    print `Path.resolve()`d paths, and resolving a symlinked $TMPDIR after the fact would
    defeat the scrubber. CWD: several commands (`load_suite`'s containment check,
    `--adapter`, `clean`) require their paths to sit under the CWD, and running from
    inside the fixture directory lets every case pass *relative* paths, which keeps
    absolute paths out of almost every snapshot in the first place.
    """
    base = tmp_path_factory.mktemp("g").resolve()
    pad = _WORKDIR_PATH_LEN - len(str(base)) - 1
    if pad < 1:
        raise AssertionError(
            f"pytest tmp base {base!r} is {len(str(base))} chars, too long to pad to "
            f"{_WORKDIR_PATH_LEN}. Raise _WORKDIR_PATH_LEN and regenerate the snapshots."
        )
    workdir = base / ("w" * pad)
    workdir.mkdir()
    assert len(str(workdir)) == _WORKDIR_PATH_LEN
    monkeypatch.chdir(workdir)
    return workdir


# --------------------------------------------------------------------------------------
# Scrubbers
#
# Each one hides exactly one source of run-to-run variation. Read the docstrings before
# adding another: the value of this whole suite is in what it still notices, so a
# scrubber that is one character broader than it needs to be is a hole, not a
# convenience. In particular there is deliberately NO scrubber for numbers in general,
# for pass rates, for counts, for case ordering, or for whitespace inside a line.
# --------------------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

#: `YYYY-MM-DD HH:MM:SS`, the shape `paw_kit.cli._short_timestamp` renders into the
#: `report` table and its disagreement panels.
_SHORT_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

#: The placeholder is exactly as long as what it replaces (19 chars), so a scrubbed
#: timestamp cell keeps its column width and the surrounding table borders stay put.
_SHORT_TS_PLACEHOLDER = "0000-00-00 00:00:00"

#: JSON keys whose values are wall-clock durations in milliseconds or seconds. Scrubbed
#: by key, never by value pattern -- a float that happens to look like a duration but
#: lives under a different key is left alone.
_DURATION_KEYS = frozenset({"latency_ms", "latency_a_ms", "latency_b_ms", "compile_wall_s"})

#: JSON keys whose values are timestamps. Same rule: by key only.
_TIMESTAMP_KEYS = frozenset(
    {
        "timestamp",
        "created_at",
        "updated_at",
        "compiled_at",
        "promoted_at",
        "demoted_at",
        "shadow_started_at",
    }
)

#: A full ISO-8601 datetime. Used only as a tripwire (`assert_no_unscrubbed_timestamps`),
#: never as a substitution: a *date* alone is left strictly alone, because "2026-09-11"
#: is a legitimate fixture output value in half the cases here.
_ISO_DATETIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


def strip_ansi(text: str) -> str:
    """Remove ANSI SGR escape sequences.

    HIDES: colour and text style only (Rich's `[bold red]` becomes plain "Error:").
    A later track that changed a message from red to green, and nothing else, would not
    be caught here. Everything the user actually reads is still compared character for
    character. `NO_COLOR=1` above normally means there is nothing to strip; this exists
    so a developer with FORCE_COLOR exported gets the same result as CI.
    """
    return _ANSI_RE.sub("", text)


def rstrip_lines(text: str) -> str:
    """Right-trim every line.

    HIDES: trailing space padding. Rich pads `--help` panels and Usage lines out to the
    full console width, which produces checked-in files full of trailing whitespace that
    editors and pre-commit hooks silently eat, causing spurious failures. The padding
    carries no information: a panel's actual width is still visible in its box-drawing
    borders, which are NOT trimmed, so a genuine column-width change still fails.
    """
    return "\n".join(line.rstrip() for line in text.splitlines())


#: Exactly `_WORKDIR_PATH_LEN` characters, so substituting it for the fixture root
#: leaves a Rich table cell the same visual width it was rendered at and the table's
#: borders still line up in the checked-in file.
_WORKDIR_TOKEN = "<WORKDIR" + "." * (_WORKDIR_PATH_LEN - 9) + ">"


def scrub_workdir(text: str, workdir: Path) -> str:
    """Replace the absolute fixture directory with a fixed same-length token.

    HIDES: where the test happened to run. Only the fixture root is replaced, so every
    path *component below it* (`report.json`, `docker/`, `adapter.paw`) is still
    compared -- a command that started writing to a different filename still fails.
    See `_WORKDIR_PATH_LEN` and `_WORKDIR_TOKEN` for why both the length and the text
    are pinned: a shorter replacement would silently knock every table row that
    contains a path out of alignment in the snapshot.
    """
    return text.replace(str(workdir), _WORKDIR_TOKEN)


def scrub_short_timestamps(text: str) -> str:
    """Replace `YYYY-MM-DD HH:MM:SS` with a same-length placeholder.

    HIDES: the wall-clock time at which a fixture trace database row was written. The
    *presence*, position and column of the timestamp are all still compared -- a
    `report` table that stopped printing `promoted_at` would still fail, because the
    placeholder would be missing.
    """
    return _SHORT_TS_RE.sub(_SHORT_TS_PLACEHOLDER, text)


def scrub_json(value: Any) -> Any:
    """Recursively replace duration and timestamp *values* in a parsed JSON artifact.

    HIDES: the value under one of the keys listed in `_DURATION_KEYS` /
    `_TIMESTAMP_KEYS`, and nothing else. The key itself stays, so a dropped or renamed
    field still fails. A duration key whose value changes *type* (float to null, say)
    is likewise still caught, since the placeholder is only substituted for a number.

    DOES NOT HIDE, and this is the limit worth knowing: a duration *value* collapsing to
    a constant. `latency_ms = 0.0` on every case passes here, because the value is
    replaced either way. That is the likelier defect -- the harness quietly stopping
    measuring -- so it is stated rather than implied away; pinning it needs an ordinary
    assertion, not a snapshot. (Wave 0 Phase F review, 2026-09-11.)
    """
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _DURATION_KEYS and isinstance(item, (int, float)) and not isinstance(item, bool):
                out[key] = "<DURATION>"
            elif key in _TIMESTAMP_KEYS and isinstance(item, str):
                out[key] = "<TIMESTAMP>"
            else:
                out[key] = scrub_json(item)
        return out
    if isinstance(value, list):
        return [scrub_json(item) for item in value]
    return value


def scrub_stdout(text: str, workdir: Path) -> str:
    """The full stdout pipeline, in the order the scrubbers must run."""
    return scrub_short_timestamps(scrub_workdir(rstrip_lines(strip_ansi(text)), workdir))


def assert_no_leaked_paths(text: str, workdir: Path) -> None:
    """Fail if any fragment of the fixture directory survived scrubbing.

    `scrub_workdir` is a plain substring replace, so it misses a path that Rich wrapped
    across two lines -- and the fragment left behind is exactly the kind of per-run text
    that makes a snapshot flap a week later, in someone else's branch. The fixture
    directory's own name is a run of "w"s chosen for this check: it cannot appear in any
    fixture content, so finding it after scrubbing means a path leaked. Widen the case's
    COLUMNS (see check_adapter_outside_cwd) rather than broadening a scrubber.
    """
    marker = workdir.name
    if len(marker) >= 6:
        assert marker not in text, (
            f"a fixture path fragment ({marker!r}) survived scrubbing -- Rich almost "
            "certainly wrapped the path across lines. Widen this case's COLUMNS."
        )
    assert "pytest-of-" not in text, "an unscrubbed pytest tmp path leaked into the snapshot"


def assert_no_unscrubbed_timestamps(json_body: str) -> None:
    """Fail if a JSON artifact still contains a wall-clock ISO datetime.

    `scrub_json` works from an explicit key list, so a field added upstream under a name
    nobody thought of (this caught `shadow_started_at`) would quietly make a snapshot
    flap once a day. This turns that into an immediate, explanatory failure: add the key
    to `_TIMESTAMP_KEYS`. Only applied to JSON artifacts -- a .txt snapshot may legitimately
    contain a hand-written fixture timestamp (see the `history_table` case).
    """
    match = _ISO_DATETIME_RE.search(json_body)
    assert match is None, (
        f"unscrubbed timestamp {match.group(0)!r} in a JSON artifact -- add its key to "
        "_TIMESTAMP_KEYS, do not widen the scrubber to match by value."
    )


# --------------------------------------------------------------------------------------
# Fixture builders
# --------------------------------------------------------------------------------------

#: A .paw manifest as MockPAWBackend's `infer()` reads it, written by hand rather than
#: produced by `compile()`: a compiled manifest carries `compile_wall_s`, which would be
#: one more thing to scrub, and hand-writing keeps the fixture visible in this file.
def write_adapter(
    path: Path,
    rules: Dict[str, str],
    *,
    spec: str = "Convert a natural-language date to ISO-8601, or INVALID.",
    backend: str = "mock",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    manifest: Dict[str, Any] = {
        "backend": backend,
        "manifest_version": 2,
        "spec": spec,
        "examples": [],
        "examples_count": 0,
        "rules": rules,
        "default_response": None,
    }
    if extra:
        manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


SUITE_YAML = """\
task_name: golden_date_normalizer
spec: "Convert a natural-language date to ISO-8601, or INVALID."
adapter_path: "{adapter}"

standard_cases:
  - input: "today"
    expected: "2026-09-11"
  - input: "new year"
    expected: "2026-01-01"

assertions:
  - rule: regex_match
    pattern: '^(\\d{{4}}-\\d{{2}}-\\d{{2}}|INVALID)$'

fuzzing:
  adversarial_probes:
    - "February 30th"

active_learning:
  auto_recompile: {auto_recompile}
  max_iterations: 2
"""


def write_suite(workdir: Path, adapter: str = "good.paw", *, auto_recompile: bool = False, name: str = "suite.yaml") -> None:
    (workdir / name).write_text(
        SUITE_YAML.format(adapter=adapter, auto_recompile=str(auto_recompile).lower()),
        encoding="utf-8",
    )


#: Same fixture cases, no assertions at all: used by the one `compare` case that needs
#: both adapters to agree on pass/fail so the "quoting-only" branch is reachable.
LOOSE_SUITE_YAML = """\
task_name: golden_date_normalizer_loose
spec: "Convert a natural-language date to ISO-8601, or INVALID."
adapter_path: "alpha.paw"

standard_cases:
  - input: "today"
  - input: "new year"

fuzzing:
  adversarial_probes:
    - "February 30th"

active_learning:
  auto_recompile: false
"""


GOOD_RULES = {"today": "2026-09-11", "new year": "2026-01-01", "February 30th": "INVALID"}
#: "tomorrow" fails the suite's regex assertion *and* the per-case `expected`.
BAD_RULES = {"today": "tomorrow", "new year": "2026-01-01", "February 30th": "INVALID"}
#: Every answer is right, but wrapped in JSON string quotes -- the shape that scored
#: 0/300 on the fast compiler's lookup adapter (measurements/README.md).
QUOTED_RULES = {"today": '"2026-09-11"', "new year": '"2026-01-01"', "February 30th": '"INVALID"'}
#: Every answer is the suite's `abstain_value` -- H-2's reproduction. True correctness
#: is 0/2; before the fix this reported "Correct against expected: 2/2 (100.0%)".
ABSTAIN_RULES = {"today": "UNPARSEABLE", "new year": "UNPARSEABLE", "February 30th": "UNPARSEABLE"}


def _verdicts(judge_id: str, pairs: List[tuple]) -> Dict[str, Any]:
    """A `paw-test judge --out` file, hand-written, for the `--diff` cases."""
    from paw_kit.test.judge import case_id_for

    verdicts = [
        {
            "case_id": case_id_for(inp, out),
            "input": inp,
            "output": out,
            "verdict": verdict,
            "reason": reason,
            "rule_passed": None,
            "judge_error": None,
        }
        for inp, out, verdict, reason in pairs
    ]
    passed = sum(1 for v in verdicts if v["verdict"])
    return {
        "judge_id": judge_id,
        "spec": "Convert a natural-language date to ISO-8601, or INVALID.",
        "temperature_note": "temperature=0.0",
        "total_cases": len(verdicts),
        "pass_count": passed,
        "pass_rate": (passed / len(verdicts) * 100.0) if verdicts else 0.0,
        "unparseable_count": 0,
        "error_count": 0,
        "verdicts": verdicts,
    }


def seed_trace_db(workdir: Path) -> str:
    """A trace DB with one promoted task, one fail-open and one recorded disagreement.

    Mirrors `tests/test_cli.py::_seed_shadow_db`; duplicated rather than imported so a
    change to that helper cannot silently rewrite these snapshots.
    """
    from paw_kit.jit.db import TraceDB

    db_file = workdir / ".paw" / "traces.db"
    task_id = "a" * 64
    db = TraceDB(str(db_file))
    db.sync_shadow_config(
        task_id,
        {"shadow_window": 2, "shadow_threshold": 0.5, "audit_window": 2, "demote_threshold": 0.4},
    )
    db.record_trace(task_id, "hello", "teacher:hello", 1.0)
    db.record_trace(task_id, "world", "teacher:world", 2.0)
    db.set_shadow_started(task_id, str(workdir / "adapter.paw"))
    epoch = db.get_task_routing(task_id)[2]
    db.record_shadow_pair(task_id, epoch, "shadow", "hello", "teacher:hello", "teacher:hello", "agree")
    db.record_shadow_pair(task_id, epoch, "shadow", "world", "teacher:world", "[v2] wrong", "disagree")
    db.try_promote(task_id, epoch, 0.5, 2)
    db.increment_fail_open(task_id)
    db.close()
    return task_id


class _StubJudge:
    """A deterministic stand-in for `anthropic_judge`, so the judge cases need no key.

    Says NO to anything that is not an ISO date, YES otherwise, with a fixed reason
    string. This replaces only the network call: `judge_outputs`, `parse_verdict`,
    `judge_disagreements` and every line of `cli.judge_cmd`'s own rendering are the real
    ones, which is what the snapshot is for.

    GAP, by design: the prompt `judge.py` builds is consumed here but not snapshotted, so
    a change to `JUDGE_PROMPT`'s wording is not caught by this suite. The assert below is
    the partial backstop -- a change to its `<model_output>` *delimiters* fails loudly
    instead of quietly turning every verdict into NO.
    """

    def __call__(self, prompt: str) -> str:
        match = re.search(r"<model_output>\n(.*?)\n    </model_output>", prompt, re.DOTALL)
        assert match is not None, (
            "judge.JUDGE_PROMPT no longer delimits the model output the way this stub "
            "parses it. Update _StubJudge -- without this assert the stub would silently "
            "judge an empty string and every golden verdict would flip to NO."
        )
        output = match.group(1).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", output):
            return "YES: matches the ISO-8601 shape the spec asks for"
        return "NO: not an ISO-8601 date"


def patch_stub_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("paw_kit.cli.anthropic_judge", lambda model="": _StubJudge())


# --------------------------------------------------------------------------------------
# Case table
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One snapshotted CLI invocation.

    `name` is the snapshot filename stem. `prog` and `argv` are exactly what the snapshot
    header shows, so a reader can reproduce it by hand. `artifact` names a file the
    command writes, snapshotted alongside the stdout as `<name>.json`.
    """

    name: str
    prog: str
    argv: List[str]
    #: Builds the fixture inside the (already current) working directory. Its return
    #: value is ignored, so a one-liner may be a lambda over an expression.
    setup: Optional[Callable[[Path], Any]] = None
    patch: Optional[Callable[[pytest.MonkeyPatch], None]] = None
    env: Dict[str, Optional[str]] = field(default_factory=dict)
    #: Fed to the command on stdin; only the interactive-confirm cases need it.
    stdin: Optional[str] = None
    #: True when the command writes its machine-readable artifact to stdout instead of
    #: to a `--json PATH` file (`paw-kit report --json`). The artifact is then snapshotted
    #: as `<name>.json`, scrubbed like any other artifact, and the `.txt` file records
    #: only the invocation and exit code -- snapshotting the raw stdout as text as well
    #: would duplicate it AND smuggle in unscrubbed ISO timestamps.
    stdout_is_json: bool = False
    artifact: Optional[str] = None
    #: A note explaining *why* this case is here, or what is odd about what it captures.
    note: str = ""

    @property
    def app(self) -> Any:
        return paw_test_app if self.prog == "paw-test" else app


def _cases() -> List[Case]:
    cases: List[Case] = []

    # -- Help text -----------------------------------------------------------------
    # Cheap, and they pin the whole CLI surface: a renamed flag, a dropped subcommand,
    # a reworded help string or a changed default all land in these files.
    cases += [
        Case("help_paw_kit", "paw-kit", ["--help"]),
        Case("help_paw_test", "paw-test", ["--help"]),
        Case("help_check", "paw-test", ["check", "--help"]),
        Case("help_compare", "paw-test", ["compare", "--help"]),
        Case("help_judge", "paw-test", ["judge", "--help"]),
        Case("help_report", "paw-kit", ["report", "--help"]),
        Case("help_inspect", "paw-kit", ["inspect", "--help"]),
        Case("help_history", "paw-kit", ["history", "--help"]),
        Case("help_lint_spec", "paw-kit", ["lint-spec", "--help"]),
        Case("help_clean", "paw-kit", ["clean", "--help"]),
        Case("help_doctor", "paw-kit", ["doctor", "--help"]),
        Case("help_demo", "paw-kit", ["demo", "--help"]),
        Case("help_serve", "paw-kit", ["serve", "--help"]),
        Case("help_export", "paw-kit", ["export", "--help"]),
        Case("help_export_docker", "paw-kit", ["export", "docker", "--help"]),
        Case("help_export_dataset", "paw-kit", ["export", "dataset", "--help"]),
    ]

    # -- check ---------------------------------------------------------------------
    def setup_good(wd: Path) -> None:
        write_adapter(wd / "good.paw", GOOD_RULES)
        write_suite(wd)

    def setup_bad(wd: Path) -> None:
        write_adapter(wd / "bad.paw", BAD_RULES)
        write_suite(wd, "bad.paw")

    def setup_quoted(wd: Path) -> None:
        write_adapter(wd / "quoted.paw", QUOTED_RULES)
        write_suite(wd, "quoted.paw")

    def setup_foreign(wd: Path) -> None:
        write_adapter(wd / "foreign.paw", GOOD_RULES, backend="programasweights")
        write_suite(wd, "foreign.paw", auto_recompile=True)

    def setup_unreadable(wd: Path) -> None:
        (wd / "opaque.paw").write_bytes(b"\x00\x01not json at all")
        write_suite(wd, "opaque.paw", auto_recompile=True)

    def setup_al(wd: Path) -> None:
        write_adapter(wd / "al.paw", BAD_RULES)
        write_suite(wd, "al.paw", auto_recompile=True)

    # bug-hunt-remediation Track B, Phase B5 (M-1, reporting half).
    def setup_al_recompiled(wd: Path) -> None:
        """A suite whose answer key matches what the CLI's demo teacher returns, so the
        teacher's label survives H-8(a)'s check and a recompile actually happens -- the
        one situation in which M-1's circularity warning must fire."""
        write_adapter(wd / "al2.paw", {"today": "tomorrow", "new year": "2026-01-01",
                                       "February 30th": "INVALID"})
        (wd / "suite.yaml").write_text(
            SUITE_YAML.format(adapter="al2.paw", auto_recompile="true").replace(
                '    expected: "2026-09-11"', '    expected: "2026-01-01"'
            ),
            encoding="utf-8",
        )

    # bug-hunt-remediation Track B, Phase B2.
    def setup_abstain(wd: Path) -> None:
        """H-2: an adapter that answers `abstain_value` to every case."""
        write_adapter(wd / "abstain.paw", ABSTAIN_RULES)
        (wd / "suite.yaml").write_text(
            SUITE_YAML.format(adapter="abstain.paw", auto_recompile="false")
            + 'abstain_value: "UNPARSEABLE"\n',
            encoding="utf-8",
        )

    def setup_keyless(wd: Path) -> None:
        """H-3: `expected:` present but empty on one of the two standard cases."""
        write_adapter(wd / "good.paw", GOOD_RULES)
        (wd / "suite.yaml").write_text(
            SUITE_YAML.format(adapter="good.paw", auto_recompile="false").replace(
                '  - input: "new year"\n    expected: "2026-01-01"\n',
                '  - input: "new year"\n    expected:\n',
            ),
            encoding="utf-8",
        )

    cases += [
        Case(
            "check_pass", "paw-test", ["check", "suite.yaml", "--json", "report.json"],
            setup=setup_good, artifact="report.json",
            note="Every case passes. The artifact is the input shape `paw-test judge` consumes.",
        ),
        Case(
            "check_fail", "paw-test", ["check", "suite.yaml", "--json", "report.json"],
            setup=setup_bad, artifact="report.json",
            note="One failing case; exit 1. Note the per-case reason text and its truncation.",
        ),
        Case(
            "check_quoted_scalar", "paw-test", ["check", "suite.yaml"],
            setup=setup_quoted,
            note=(
                "Captures the 'Correct after unquoting a JSON string' second line. Worth "
                "reading next to the 'Pass rate' line below it: pass rate is 0% while "
                "'correct after unquoting' is 100%, and the two numbers are printed with "
                "equal weight and no explanation of which one a reader should believe."
            ),
        ),
        Case(
            "check_auto_recompile_refused_foreign", "paw-test", ["check", "suite.yaml"],
            setup=setup_foreign,
            note="auto_recompile=true against a non-mock manifest: the guard declines and runs read-only.",
        ),
        Case(
            "check_auto_recompile_refused_unreadable", "paw-test", ["check", "suite.yaml"],
            setup=setup_unreadable,
            note=(
                "Same guard, unreadable manifest. The adapter is not JSON, so MockPAWBackend "
                "cannot infer from it either: every case reports the mock's fallback string, "
                "and the run reads as 'the model was wrong' rather than 'the adapter is "
                "unreadable'. Snapshotted as-is."
            ),
        ),
        Case(
            "check_active_learning_fail", "paw-test", ["check", "suite.yaml"],
            setup=setup_al,
            note=(
                "SUPERSEDED 2026-09-13 (bug-hunt-remediation Track H, C-2). This case's "
                "fixture pre-writes a mock-declared adapter (`al.paw`) and sets "
                "auto_recompile=true, which used to reach the active-learning path this "
                "case was written to pin (the per-iteration lines, H-8(a)'s poisoned-label "
                "rejection, the honest 1/2). C-2 closed the guard's mock-adapter exemption: "
                "*any* existing adapter now blocks auto-recompile, not only non-mock ones, "
                "so this exact fixture now hits that guard first and runs read-only instead "
                "-- never reaching the active-learning branch at all. Still a correct, even "
                "stronger outcome (the existing adapter is protected two ways now: H-8 would "
                "reject the poisoned label, and C-2 never lets it try), but the per-iteration "
                "text this case used to snapshot is not exercised by this fixture shape "
                "anymore. The original notes are kept below for the historical record.\n"
                "Original: the active-learning path: the CLI's demo-stub teacher runs and "
                "the run still fails. Pins the per-iteration lines and the [FAIL] summary.\n"
                "  * G-1 -- the [ACTION] line read \"Querying frontier teacher for 'You "
                "are an authoritative labeling teache'...\", the first 40 characters of "
                "the *teacher prompt*, identical for every case. It now quotes the case "
                "input ('today').\n"
                "  * H-8(a)/M-1 -- 'Correct against expected: 2/2 (100.0%)' printed "
                "directly above '[FAIL] Assertions failed'. That 2/2 was circular: the "
                "demo teacher's fabricated '2026-01-01' was accepted as gold, the adapter "
                "was recompiled from a dataset seeded with the suite's own answer key, "
                "and it then scored full marks against that key. The label now fails the "
                "answer-key check, no recompile happens, and the honest 1/2 shows."
            ),
        ),
        Case(
            "check_active_learning_recompiled", "paw-test", ["check", "suite.yaml"],
            setup=setup_al_recompiled,
            note=(
                "SUPERSEDED 2026-09-13 (Track H, C-2) -- same cause as "
                "check_active_learning_fail above: the pre-written mock-declared adapter "
                "now blocks auto-recompile outright, so the active-learning path (and "
                "M-1's circularity note this case existed to pin) never runs; this is now "
                "a read-only run against the pre-seeded adapter. CLI-level golden coverage "
                "of M-1's circularity-note text is lost by this fixture shape; the "
                "mechanism itself remains covered at the `run_active_learning_loop` unit "
                "level (Track B's own tests) and by `paw_kit/cli.py`'s "
                "`test_check_mock_backend_recompiles_freely_when_no_adapter_exists_yet` "
                "(a first-compile scenario with no pre-existing adapter, added by Track H). "
                "Filed as a gap rather than silently dropped -- see that track's file.\n"
                "Original note: M-1's reporting half: the adapter was recompiled during "
                "this run from a dataset seeded with the suite's own `expected` values, so "
                "any agreement with `expected` afterwards is circular. Report M-1 filed "
                "exactly this as a High finding -- a 2-case suite reporting 'Correct "
                "against expected: 2/2 (100.0%)' and [SUCCESS] at exit 0 against an "
                "adapter built from its own answer key seconds earlier."
            ),
        ),
        Case(
            "check_adapter_override", "paw-test", ["check", "suite.yaml", "--adapter", "bad.paw"],
            setup=lambda wd: (setup_good(wd), write_adapter(wd / "bad.paw", BAD_RULES)),
            note="--adapter overrides the suite's own adapter_path.",
        ),
        # -- bug-hunt-remediation Track B, Phase B2 (H-1, H-2, H-3, G-5) ------------
        Case(
            "check_missing_adapter", "paw-test", ["check", "suite.yaml"],
            setup=lambda wd: write_suite(wd, "absent.paw"),
            note=(
                "H-1: the adapter-existence gate `compare` already had. With "
                "recompilation off there is nothing to run the suite against, so this "
                "exits 1 instead of running every case against MockPAWBackend's "
                "fallback string and reporting on a placeholder. Scoped to the "
                "read-only path on purpose: whether `check` may *create* an adapter is "
                "M-1, an open policy decision belonging to Track H."
            ),
        ),
        Case(
            "check_abstain_all", "paw-test", ["check", "suite.yaml"],
            setup=setup_abstain,
            note=(
                "H-2: an adapter abstaining on every case. Before the fix this printed "
                "'Correct against expected: 2/2 (100.0%)' at a true correctness of 0/2, "
                "because an output equal to `abstain_value` was scored as a *match*. It "
                "now reports 0/0 with the denominator naming the two abstentions, plus "
                "an 'Abstained:' line of its own. The assertions still pass -- that "
                "escape hatch is deliberate and unchanged."
            ),
        ),
        Case(
            "check_keyless_expected", "paw-test", ["check", "suite.yaml"],
            setup=setup_keyless,
            note=(
                "H-3: one of the two standard cases was authored as `expected:` with "
                "nothing after the colon -- valid YAML, and the key looks present. The "
                "old output was 'Correct against expected: 1/1 (100.0%)' with nothing "
                "saying a case had silently left the only correctness number printed."
            ),
        ),
        Case("check_missing_suite", "paw-test", ["check", "nope.yaml"]),
        Case(
            "check_invalid_yaml", "paw-test", ["check", "bad.yaml"],
            setup=lambda wd: (wd / "bad.yaml").write_text("- not a mapping\n", encoding="utf-8"),
        ),
        Case(
            "check_unknown_rule", "paw-test", ["check", "suite.yaml"],
            setup=lambda wd: (wd / "suite.yaml").write_text(
                'task_name: t\nspec: s\nadapter_path: "a.paw"\nassertions:\n  - rule: contains\n    value: x\n',
                encoding="utf-8",
            ),
            note="suite.py validates the rule name at load time; this is the message it produces.",
        ),
        Case(
            "check_unknown_backend", "paw-test", ["check", "suite.yaml", "--backend", "gpt"],
            setup=setup_good,
            note="typer.BadParameter -- goes to stderr with exit 2, unlike every other error here.",
        ),
        Case(
            "check_adapter_outside_cwd", "paw-test", ["check", "suite.yaml", "--adapter", "/etc/escape.paw"],
            setup=setup_good, env={"COLUMNS": "240"},
            note=(
                "PAW-TEST-02 containment on the --adapter override. Deliberately an absolute "
                "path outside the tmp tree, not '../escape.paw': the error text quotes the "
                "resolved path, and a path under the fixture's *parent* would vary in length "
                "between runs. COLUMNS is widened for this case so the message (which embeds "
                "the 64-char fixture root) does not wrap mid-path, which would defeat "
                "scrub_workdir."
            ),
        ),
    ]

    # -- compare -------------------------------------------------------------------
    def setup_compare_same(wd: Path) -> None:
        write_adapter(wd / "alpha.paw", GOOD_RULES)
        write_adapter(wd / "beta.paw", GOOD_RULES)
        write_suite(wd)
        write_adapter(wd / "good.paw", GOOD_RULES)

    def setup_compare_diff(wd: Path) -> None:
        write_adapter(wd / "alpha.paw", GOOD_RULES)
        write_adapter(wd / "beta.paw", BAD_RULES)
        write_suite(wd)
        write_adapter(wd / "good.paw", GOOD_RULES)

    def setup_compare_quoted(wd: Path) -> None:
        write_adapter(wd / "alpha.paw", GOOD_RULES)
        write_adapter(wd / "beta.paw", QUOTED_RULES)
        write_suite(wd)
        write_adapter(wd / "good.paw", GOOD_RULES)

    def setup_compare_quoted_loose(wd: Path) -> None:
        """Same adapters, but a suite with no assertions, so both sides "pass"."""
        setup_compare_quoted(wd)
        (wd / "loose.yaml").write_text(LOOSE_SUITE_YAML, encoding="utf-8")

    cases += [
        Case(
            "compare_identical", "paw-test", ["compare", "alpha.paw", "beta.paw", "suite.yaml"],
            setup=setup_compare_same,
            note="The 'No differences' branch and the full Summary line.",
        ),
        Case(
            "compare_differences", "paw-test",
            ["compare", "alpha.paw", "beta.paw", "suite.yaml", "--json", "compare.json"],
            setup=setup_compare_diff, artifact="compare.json",
            note="Per-case diff listing plus the CompareReport artifact shape.",
        ),
        Case(
            "compare_quoted_pass_differs", "paw-test", ["compare", "alpha.paw", "beta.paw", "suite.yaml"],
            setup=setup_compare_quoted,
            note=(
                "Quoted-scalar outputs that also flip pass status. `equivalent_only_rows` "
                "requires pass_a == pass_b, so these are listed as full Differences even "
                "though the summary line simultaneously calls them '3 equivalent output once "
                "unwrapped'. Two readings of the same three rows, five lines apart."
            ),
        ),
        Case(
            "compare_quoting_only", "paw-test", ["compare", "alpha.paw", "beta.paw", "loose.yaml"],
            setup=setup_compare_quoted_loose,
            note=(
                "The collapsed 'Whitespace- or quoting-only differences' heading, which only "
                "appears when the two adapters agree on pass/fail -- hence the assertion-free "
                "suite. Note the summary still reports 0 identical and 0 equivalent for rows "
                "the heading above calls 'not a real disagreement'."
            ),
        ),
        Case(
            "compare_no_fuzz", "paw-test", ["compare", "alpha.paw", "beta.paw", "suite.yaml", "--no-fuzz"],
            setup=setup_compare_diff,
            note="--no-fuzz drops the adversarial probe, changing the denominators.",
        ),
        Case(
            "compare_missing_adapter", "paw-test", ["compare", "alpha.paw", "gone.paw", "suite.yaml"],
            setup=setup_compare_diff,
        ),
        Case(
            "compare_unreadable_manifest", "paw-test", ["compare", "alpha.paw", "opaque.paw", "suite.yaml"],
            setup=lambda wd: (setup_compare_diff(wd), (wd / "opaque.paw").write_bytes(b"\x00nope")),
        ),
        Case(
            "compare_missing_suite", "paw-test", ["compare", "alpha.paw", "beta.paw", "nope.yaml"],
            setup=setup_compare_diff,
        ),
    ]

    # -- judge ---------------------------------------------------------------------
    # No network: `paw_kit.cli.anthropic_judge` is replaced by `_StubJudge` for the cases
    # that actually judge, and the `--diff` cases read hand-written verdict files and
    # never touch a provider at all.
    def setup_judge_check(wd: Path) -> None:
        setup_bad(wd)
        result = runner.invoke(
            paw_test_app, ["check", "suite.yaml", "--json", "check.json"], env=dict(_BASE_ENV), prog_name="paw-test"
        )
        assert (wd / "check.json").exists(), result.output

    def setup_judge_compare(wd: Path) -> None:
        setup_compare_diff(wd)
        result = runner.invoke(
            paw_test_app,
            ["compare", "alpha.paw", "beta.paw", "suite.yaml", "--json", "compare.json"],
            env=dict(_BASE_ENV),
            prog_name="paw-test",
        )
        assert (wd / "compare.json").exists(), result.output

    def setup_diff_files(wd: Path, *, flipped: bool) -> None:
        old = _verdicts(
            "anthropic/claude-haiku-4-5/temperature=0.0",
            [("today", "2026-09-11", True, "correct"), ("new year", "tomorrow", False, "not a date")],
        )
        new = _verdicts(
            "anthropic/claude-haiku-4-5/temperature=0.0",
            [
                ("today", "2026-09-11", not flipped, "flipped" if flipped else "correct"),
                ("new year", "tomorrow", False, "not a date"),
            ],
        )
        (wd / "old.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
        (wd / "new.json").write_text(json.dumps(new, indent=2), encoding="utf-8")

    # bug-hunt-remediation Track B, Phase B4 (H-7).
    def setup_diff_disjoint(wd: Path) -> None:
        """Two runs that share no `case_id`. A `case_id` hashes input AND output, so
        this is what an ordinary adapter change produces -- and it used to read as a
        perfect reproducibility result."""
        old = _verdicts(
            "anthropic/claude-haiku-4-5/temperature=0.0",
            [("today", "2026-09-11", True, "correct"), ("new year", "2026-01-01", True, "correct")],
        )
        new = _verdicts(
            "anthropic/claude-haiku-4-5/temperature=0.0",
            [("today", "11/09/2026", True, "correct"), ("new year", "01/01/2026", True, "correct")],
        )
        (wd / "old.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
        (wd / "new.json").write_text(json.dumps(new, indent=2), encoding="utf-8")

    cases += [
        Case(
            "judge_check_report", "paw-test",
            ["judge", "check.json", "--suite", "suite.yaml", "--out", "verdicts.json"],
            setup=setup_judge_check, patch=patch_stub_judge, artifact="verdicts.json",
            env={"ANTHROPIC_API_KEY": "golden-not-a-real-key"},
            note=(
                "The `check`-report branch: 'Judge said NO' listing, disagreement block, pass rate.\n"
                "LOOKS WRONG, NOT FIXED HERE: the 'judge disagrees with assertions' block "
                "identifies its cases by the 64-character `case_id` hash, while every other "
                "listing in the same command prints the input and output. The block the "
                "docstring calls 'the case that matters most' is the one a reader cannot "
                "identify without grepping the artifact."
            ),
        ),
        Case(
            "judge_compare_report", "paw-test",
            ["judge", "compare.json", "--suite", "suite.yaml", "--out", "verdicts.json"],
            setup=setup_judge_compare, patch=patch_stub_judge, artifact="verdicts.json",
            env={"ANTHROPIC_API_KEY": "golden-not-a-real-key"},
            note="The `compare`-report branch: A/B verdict differences and the A/B wrapper artifact.",
        ),
        Case(
            "judge_diff_no_flips", "paw-test", ["judge", "--diff", "old.json", "new.json"],
            setup=lambda wd: setup_diff_files(wd, flipped=False),
        ),
        Case(
            "judge_diff_flips", "paw-test", ["judge", "--diff", "old.json", "new.json"],
            setup=lambda wd: setup_diff_files(wd, flipped=True),
            note="The reproducibility check the judge docstring points at.",
        ),
        Case(
            "judge_diff_disjoint", "paw-test", ["judge", "--diff", "old.json", "new.json"],
            setup=setup_diff_disjoint,
            note=(
                "H-7: two runs sharing no comparable case. This printed 'No flips -- "
                "every comparable verdict matched. Flip rate: 0.0% (0/0)' and exited 0 "
                "-- and since a `case_id` hashes the input *and* the output, any change "
                "to the adapter produces exactly this, while docs/results.md offers the "
                "command as the check that temperature-0 pinning held."
            ),
        ),
        Case(
            "judge_diff_missing_file", "paw-test", ["judge", "--diff", "old.json", "gone.json"],
            setup=lambda wd: setup_diff_files(wd, flipped=False),
        ),
        Case("judge_missing_report", "paw-test", ["judge", "gone.json", "--spec", "s"]),
        Case(
            "judge_no_report_argument", "paw-test", ["judge", "--spec", "s"],
        ),
        Case(
            "judge_no_spec", "paw-test", ["judge", "check.json"],
            setup=setup_judge_check,
        ),
        Case(
            "judge_unknown_provider", "paw-test",
            ["judge", "check.json", "--spec", "s", "--judge", "openai:gpt-4"],
            setup=setup_judge_check,
            note="Exit 2. The provider check runs before the API-key check.",
        ),
        Case(
            "judge_no_api_key", "paw-test", ["judge", "check.json", "--spec", "s"],
            setup=setup_judge_check,
            note="Exit 2 rather than silently skipping. ANTHROPIC_API_KEY is cleared by _BASE_ENV.",
        ),
        Case(
            "judge_unrecognised_shape", "paw-test", ["judge", "weird.json", "--spec", "s"],
            setup=lambda wd: (wd / "weird.json").write_text('{"hello": "world"}', encoding="utf-8"),
            patch=patch_stub_judge,
            env={"ANTHROPIC_API_KEY": "golden-not-a-real-key"},
        ),
    ]

    # -- report --------------------------------------------------------------------
    def setup_stalled_db(wd: Path) -> None:
        from paw_kit.jit.db import TraceDB
        from paw_kit.jit.shadow import _SHADOW_STALL_FACTOR

        db_file = wd / ".paw" / "traces.db"
        task_id = "b" * 64
        window = 2
        db = TraceDB(str(db_file))
        db.sync_shadow_config(
            task_id,
            {"shadow_window": window, "shadow_threshold": 0.8, "audit_window": 2, "demote_threshold": 0.6},
        )
        db.record_trace(task_id, "hello", "teacher:hello", 1.0)
        db.set_shadow_started(task_id, str(wd / "adapter.paw"))
        epoch = db.get_task_routing(task_id)[2]
        for n in range(_SHADOW_STALL_FACTOR * window + 1):
            db.record_shadow_pair(task_id, epoch, "shadow", f"in{n}", f"teacher:{n}", "wrong", "disagree")
        db.close()

    cases += [
        Case(
            "report_table", "paw-kit", ["report"],
            setup=lambda wd: seed_trace_db(wd),
            note="The task table plus the 'Last disagreements' panel.",
        ),
        Case(
            "report_json", "paw-kit", ["report", "--json"], stdout_is_json=True,
            setup=lambda wd: seed_trace_db(wd),
            note=(
                "stdout IS the artifact here (typer.echo, not --json PATH), so it is "
                "snapshotted as the .json file for this case."
            ),
        ),
        Case(
            "report_stalled", "paw-kit", ["report"],
            setup=setup_stalled_db,
            note="The `(stalled)` marker. Rich truncates it in the narrow State column.",
        ),
        Case("report_missing_db", "paw-kit", ["report"]),
        Case(
            "report_empty_db", "paw-kit", ["report"],
            setup=lambda wd: __import__("paw_kit.jit.db", fromlist=["TraceDB"]).TraceDB(
                str(wd / ".paw" / "traces.db")
            ).close(),
        ),
    ]

    # -- inspect -------------------------------------------------------------------
    def setup_inspect(wd: Path) -> None:
        write_adapter(
            wd / "adapter.paw",
            {"today": "2026-09-11"},
            extra={
                "program_id": "prog_golden_0001",
                "compiler": "golden-fixture",
                "spec_sha256": "0" * 64,
                "examples_folded_into_spec": 2,
                "folded_example_ids": ["ex-0", "ex-1"],
                "public": True,
                "unknown_third_party_field": "kept, sorted after the known ones",
            },
        )

    cases += [
        Case(
            "inspect_table", "paw-kit", ["inspect", "adapter.paw"],
            setup=setup_inspect,
            note=(
                "Pins _INSPECT_FIELD_ORDER and the File Path/Size/Format rows. `public: True` "
                "is printed as a plain row with no emphasis, which is the hub-visibility "
                "default the 2026-09-10 note is about."
            ),
        ),
        Case("inspect_json", "paw-kit", ["inspect", "adapter.paw", "--json"], setup=setup_inspect),
        Case("inspect_missing", "paw-kit", ["inspect", "gone.paw"]),
        Case(
            "inspect_not_json", "paw-kit", ["inspect", "weights.bin", "--json"],
            setup=lambda wd: (wd / "weights.bin").write_bytes(b"\x00\x01\x02binary"),
        ),
        Case(
            "inspect_binary_table", "paw-kit", ["inspect", "weights.bin"],
            setup=lambda wd: (wd / "weights.bin").write_bytes(b"\x00\x01\x02binary"),
        ),
    ]

    # -- history -------------------------------------------------------------------
    def setup_history(wd: Path) -> None:
        write_adapter(wd / "adapter.paw", GOOD_RULES)
        lines = [
            {
                "compiled_at": "2026-09-01T10:00:00",
                "backend": "programasweights",
                "program_id": "prog_0001",
                "compiler": "fast",
                "compile_wall_s": 12.3456,
                "parent_program_id": None,
            },
            {
                "compiled_at": "2026-09-02T11:30:00",
                "backend": "programasweights",
                "program_id": "prog_0002",
                "compiler": "finetune",
                "compile_wall_s": 640.5,
                "parent_program_id": "prog_0001",
            },
        ]
        (wd / "adapter.paw.history.jsonl").write_text(
            "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
        )

    cases += [
        Case(
            "history_table", "paw-kit", ["history", "adapter.paw"], setup=setup_history,
            note=(
                "compiled_at is printed raw (ISO with a 'T'), unlike `report`'s table which "
                "runs it through _short_timestamp. Inconsistent, snapshotted as-is."
            ),
        ),
        Case("history_missing", "paw-kit", ["history", "adapter.paw"]),
        Case(
            "history_empty", "paw-kit", ["history", "adapter.paw"],
            setup=lambda wd: (wd / "adapter.paw.history.jsonl").write_text("\n\n", encoding="utf-8"),
            note="Blank/unparseable lines are skipped; an all-blank log exits 0.",
        ),
    ]

    # -- lint-spec -----------------------------------------------------------------
    # Under speclint's 16k spec-too-long ceiling on purpose: this case is here for the
    # `output-format-unpinned` warning and its measurement citation, which is the longest
    # single message the linter can print and therefore the one most likely to be
    # reflowed by an unrelated change.
    _LONG_SPEC = "Normalize the input. " * 400
    # Over the ceiling, for the `error`-severity branch and its exit 1.
    _OVERLONG_SPEC = "Normalize the input consistently. " * 500
    cases += [
        Case("lint_spec_clean", "paw-kit", ["lint-spec", "--file", "spec.txt"],
             setup=lambda wd: (wd / "spec.txt").write_text(
                 "Convert a natural-language date to ISO-8601 (YYYY-MM-DD). If the date is "
                 "not a real calendar date, output exactly INVALID. Output nothing else.\n"
                 "Examples: 'today' -> 2026-09-11. 'February 30th' -> INVALID.\n",
                 encoding="utf-8")),
        Case("lint_spec_too_short", "paw-kit", ["lint-spec", "do it"],
             note="Severity 'error' -> exit 1."),
        Case("lint_spec_format_warning", "paw-kit", ["lint-spec", "--file", "spec.txt"],
             setup=lambda wd: (wd / "spec.txt").write_text(_LONG_SPEC, encoding="utf-8"),
             note="A `warn` finding only -- note it still exits 0."),
        Case("lint_spec_too_long", "paw-kit", ["lint-spec", "--file", "spec.txt"],
             setup=lambda wd: (wd / "spec.txt").write_text(_OVERLONG_SPEC, encoding="utf-8"),
             note="Severity 'error' -> exit 1."),
        Case("lint_spec_json", "paw-kit", ["lint-spec", "do it", "--json"]),
        Case("lint_spec_both_sources", "paw-kit", ["lint-spec", "text", "--file", "spec.txt"],
             setup=lambda wd: (wd / "spec.txt").write_text("x", encoding="utf-8")),
        Case("lint_spec_no_source", "paw-kit", ["lint-spec"]),
    ]

    # -- clean ---------------------------------------------------------------------
    cases += [
        Case(
            "clean_dry_run", "paw-kit", ["clean", "--dry-run"],
            setup=lambda wd: [
                (wd / ".paw").mkdir(),
                (wd / ".paw" / "a.paw").write_text("{}", encoding="utf-8"),
            ],
            note=(
                "Exactly one cached file on purpose: `clean` lists `Path.glob('*')` in "
                "filesystem order, not sorted, so a two-file fixture would be a snapshot that "
                "depends on the filesystem -- which is also a real (cosmetic) finding about "
                "the command.\n"
                "LOOKS WRONG, NOT FIXED HERE: 'would remove 1 files'."
            ),
        ),
        Case("clean_no_cache_dir", "paw-kit", ["clean"]),
        Case(
            "clean_empty_cache_dir", "paw-kit", ["clean"],
            setup=lambda wd: (wd / ".paw").mkdir(),
        ),
        Case(
            "clean_outside_cwd", "paw-kit", ["clean", "--cache-dir", "/etc"], env={"COLUMNS": "240"},
            note="PAW-CLI-01 containment: refuses a directory outside the CWD. See check_adapter_outside_cwd for the COLUMNS override.",
        ),
        Case(
            "clean_confirm_declined", "paw-kit", ["clean"],
            setup=lambda wd: [
                (wd / ".paw").mkdir(),
                (wd / ".paw" / "a.paw").write_text("{}", encoding="utf-8"),
            ],
            stdin="n\n",
            note="Answering 'n' at the confirmation prompt: nothing is deleted, exit 0.",
        ),
    ]

    # -- export --------------------------------------------------------------------
    def setup_traces_db(wd: Path) -> None:
        from paw_kit.jit.db import TraceDB

        db = TraceDB(str(wd / ".paw" / "traces.db"))
        db.record_trace("a" * 64, "hello", "teacher:hello", 1.0)
        db.record_trace("a" * 64, "world", "teacher:world", 2.0)
        db.close()

    cases += [
        Case(
            "export_docker", "paw-kit", ["export", "docker", "adapter.paw"],
            setup=lambda wd: write_adapter(wd / "adapter.paw", GOOD_RULES),
        ),
        Case("export_docker_missing_adapter", "paw-kit", ["export", "docker", "gone.paw"]),
        Case(
            "export_docker_bad_backend", "paw-kit", ["export", "docker", "adapter.paw", "-b", "gpt"],
            setup=lambda wd: write_adapter(wd / "adapter.paw", GOOD_RULES),
        ),
        Case(
            "export_dataset", "paw-kit", ["export", "dataset", "-o", "traces.jsonl"],
            setup=setup_traces_db, artifact=None,
        ),
        Case("export_dataset_missing_db", "paw-kit", ["export", "dataset", "-o", "traces.jsonl"]),
        Case(
            "export_dataset_bad_suffix", "paw-kit", ["export", "dataset", "-o", "traces.txt"],
            setup=setup_traces_db,
        ),
        Case(
            "export_dataset_empty_db", "paw-kit", ["export", "dataset", "-o", "traces.jsonl"],
            setup=lambda wd: __import__("paw_kit.jit.db", fromlist=["TraceDB"]).TraceDB(
                str(wd / ".paw" / "traces.db")
            ).close(),
            note="An empty trace DB warns and exits 0 (PAW-CLI-04).",
        ),
    ]

    return cases


CASES = _cases()
assert len({c.name for c in CASES}) == len(CASES), "duplicate golden case name"


# --------------------------------------------------------------------------------------
# Runner + comparison
# --------------------------------------------------------------------------------------


def _render(case: Case, result: Any, workdir: Path) -> str:
    """The `.txt` snapshot body: the command, its exit code, then its streams.

    The command line and exit code are part of the snapshot on purpose. Several of the
    hunt's findings are about exit codes rather than text, and a header makes each file
    reproducible by hand without reading this module.
    """
    parts = [
        f"$ {case.prog} {' '.join(case.argv)}",
        f"exit code: {result.exit_code}",
        "--- stdout ---",
        (
            f"(stdout is the JSON artifact; see {case.name}.json)"
            if case.stdout_is_json
            else scrub_stdout(result.stdout, workdir)
        ),
    ]
    stderr = result.stderr if result.stderr is not None else ""
    if stderr:
        parts += ["--- stderr ---", scrub_stdout(stderr, workdir)]
    return "\n".join(parts).rstrip() + "\n"


def _compare_or_write(path: Path, actual: str) -> None:
    if UPDATE_SNAPSHOTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual, encoding="utf-8")
        return
    if not path.exists():
        raise AssertionError(
            f"missing golden snapshot {path.name}. Regenerate with PAW_GOLDEN_UPDATE=1 "
            "and review the new file before committing it."
        )
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, (
        f"\nGolden snapshot {path.name} does not match current CLI output.\n"
        "If you changed this output on purpose, rerun with PAW_GOLDEN_UPDATE=1 and put "
        "the regenerated snapshot in the SAME diff as the change, so a reviewer sees "
        "the before and after. Do not regenerate to make a red test green.\n"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_golden_cli_output(case: Case, golden_workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run one CLI case and compare its stdout (and artifact) against the snapshot."""
    if case.setup is not None:
        case.setup(golden_workdir)
    if case.patch is not None:
        case.patch(monkeypatch)

    env: Dict[str, Optional[str]] = dict(_BASE_ENV)
    env.update(case.env)
    if "COLUMNS" in case.env and case.env["COLUMNS"]:
        # A case that widens the console (see check_adapter_outside_cwd) has to widen both
        # halves of the width story, not just the environment one.
        from rich.console import Console

        monkeypatch.setattr(
            "paw_kit.cli.console", Console(width=int(case.env["COLUMNS"]), height=40, no_color=True)
        )
    result = runner.invoke(
        case.app, case.argv, env=env, input=case.stdin, prog_name=case.prog, catch_exceptions=False
    )

    rendered = _render(case, result, golden_workdir)
    assert_no_leaked_paths(rendered, golden_workdir)
    _compare_or_write(GOLDEN_DIR / f"{case.name}.txt", rendered)

    if case.artifact is not None:
        artifact_path = golden_workdir / case.artifact
        assert artifact_path.exists(), f"{case.name}: expected artifact {case.artifact}"
        payload = scrub_json(json.loads(artifact_path.read_text(encoding="utf-8")))
        payload = json.loads(scrub_workdir(json.dumps(payload), golden_workdir))
        body = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        assert_no_leaked_paths(body, golden_workdir)
        assert_no_unscrubbed_timestamps(body)
        _compare_or_write(GOLDEN_DIR / f"{case.name}.json", body)
    elif case.stdout_is_json:
        # `report --json` writes its artifact to stdout rather than to a path.
        payload = scrub_json(json.loads(strip_ansi(result.stdout)))
        payload = json.loads(scrub_workdir(json.dumps(payload), golden_workdir))
        body = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        assert_no_leaked_paths(body, golden_workdir)
        assert_no_unscrubbed_timestamps(body)
        _compare_or_write(GOLDEN_DIR / f"{case.name}.json", body)


def test_no_orphan_snapshots() -> None:
    """Every file in tests/golden/ belongs to a live case.

    A deleted case that leaves its snapshot behind turns the directory into a museum of
    output nothing checks any more, which is exactly how a golden suite rots.
    """
    expected = {f"{c.name}.txt" for c in CASES}
    expected |= {f"{c.name}.json" for c in CASES if c.artifact is not None}
    expected |= {f"{c.name}.json" for c in CASES if c.stdout_is_json}
    expected.add("README.md")
    actual = {p.name for p in GOLDEN_DIR.iterdir() if p.is_file()}
    assert actual == expected, (
        f"orphaned: {sorted(actual - expected)}; missing: {sorted(expected - actual)}"
    )


def test_snapshots_are_not_being_updated_in_ci() -> None:
    """PAW_GOLDEN_UPDATE must never be set for a normal run.

    With it set, every assertion above rewrites its snapshot and passes unconditionally.
    This test is the tripwire: it fails loudly rather than letting an exported variable
    turn the whole suite into a no-op.
    """
    assert not UPDATE_SNAPSHOTS, (
        "PAW_GOLDEN_UPDATE=1 is set: the snapshots were REWRITTEN, not checked. "
        "Review `git diff tests/golden/` before committing, then unset it."
    )
