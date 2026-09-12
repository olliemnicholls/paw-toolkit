"""D-4: what the `.history.jsonl` lineage sidecar is allowed to retain, and for how long.

The defect: `append_history_entry` filtered the manifest with
`{k: v for k, v in manifest.items() if k != "spec"}` -- a **deny-list of one key**. The
default (mock) backend puts the traced input/output pairs in `examples`, which walked
straight past it, so three mock compiles of 500 examples produced a 216 KB sidecar of raw
production rows. Append-only, no cap, no pruning, no fsync, forever.

**The retention decision this file pins** (D-4 is a DECISION-bucket item in the parent
track, not just a patch -- a fix with an unstated policy does not terminate):

- *What may be retained*: an **allow-list** of lineage fields -- ids, hashes, counts,
  visibility provenance, parent pointers, timings. Anything a backend writes that is not
  on it is dropped, and the two sensitive keys (`spec`, `examples`) are additionally named
  explicitly so that dropping them is a recorded decision rather than an accident of
  omission. A deny-list cannot be right here: it has to enumerate every *future* way a
  backend might carry raw traffic, and it got that wrong the first time.
- *How much*: `_HISTORY_MAX_BYTES` per file, with `_HISTORY_ROTATIONS` generations kept,
  so total retention is bounded at roughly twice the cap rather than growing forever.
- *How it is trimmed*: by `os.replace` to a `.1` sidecar, never by rewriting the live
  file. Report §12 measured this sidecar clean under 6 processes x 10 appends of 1.6 MB
  lines (0 corrupt lines), and that holds *because* each append is one `write()` under
  `O_APPEND`. "Oldest-first rotation" as the finding first described it would have turned
  it back into read-modify-write and regressed a verified-clean property.
- *Durability*: one `fsync` per append. A compile takes seconds to minutes; one fsync is
  free at that scale.
"""

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from paw_kit.backend.mock import MockPAWBackend
import paw_kit.backend.manifest_lineage as lineage
from paw_kit.backend.programasweights import ProgramAsWeightsBackend

from tests.test_programasweights_backend import FakeSDK


TRACED_INPUT = "ticket-body-INV-9821-customer-was-double-charged"
TRACED_OUTPUT = "billing/urgent"
EXAMPLES = [{"input": TRACED_INPUT, "output": TRACED_OUTPUT}]


def _history_lines(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _compile_both_backends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """Compile the same traced examples through both shipped backends."""
    monkeypatch.setenv("PAW_API_KEY", "paw_sk_test")
    mock_out = tmp_path / "mock.paw"
    MockPAWBackend().compile("Triage tickets.", EXAMPLES, str(mock_out))
    paw_out = tmp_path / "paw.paw"
    ProgramAsWeightsBackend(sdk=FakeSDK(), max_spec_examples=8).compile(
        "Triage tickets.", EXAMPLES, str(paw_out)
    )
    return {"mock": mock_out, "programasweights": paw_out}


@pytest.mark.parametrize("backend_name", ["mock", "programasweights"])
def test_history_line_retains_no_traced_text_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """The finding's own required assertion: no value in a history line contains a traced
    input/output string, for **both** shipped backends.

    The mock one is the one that was broken -- and the mock backend is the default, so
    this was the path anybody following the README took.
    """
    out = _compile_both_backends(tmp_path, monkeypatch)[backend_name]
    entries = _history_lines(Path(str(out) + ".history.jsonl"))
    assert entries, "no history line was written at all"
    blob = json.dumps(entries)
    assert TRACED_INPUT not in blob, (
        "a traced production input survived into the append-only sidecar, which has no "
        "cap and is never pruned"
    )
    assert TRACED_OUTPUT not in blob
    assert "spec" not in entries[0]
    assert "examples" not in entries[0]


@pytest.mark.parametrize("backend_name", ["mock", "programasweights"])
def test_history_allow_list_classifies_every_key_a_shipped_backend_writes_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend_name: str
) -> None:
    """The evolution guard, without which the allow-list silently drops the *next*
    lineage field anyone adds -- Pattern 5 from the other direction.

    Adding a manifest key must force a decision: allow-list it, or name it as
    deliberately excluded with a reason. Failing here is not a bug in this test; it means
    a new manifest key needs classifying in `manifest_lineage.py`.
    """
    out = _compile_both_backends(tmp_path, monkeypatch)[backend_name]
    manifest = json.loads(Path(out).read_text(encoding="utf-8"))
    classified = set(lineage._HISTORY_ALLOWED_FIELDS) | set(lineage._HISTORY_EXCLUDED_FIELDS)
    unclassified = sorted(set(manifest) - classified)
    assert unclassified == [], (
        f"{backend_name} writes manifest key(s) {unclassified} that are neither "
        "allow-listed for the lineage sidecar nor named as deliberately excluded. "
        "Classify them in manifest_lineage.py rather than letting them be dropped "
        "(or retained) by accident."
    )


def test_history_keeps_the_lineage_fields_it_exists_for_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An allow-list that drops the lineage too would "pass" the leak test vacuously."""
    out = _compile_both_backends(tmp_path, monkeypatch)["programasweights"]
    entry = _history_lines(Path(str(out) + ".history.jsonl"))[0]
    for key in (
        "program_id", "spec_sha256", "full_spec_sha256", "examples_count",
        "examples_folded_into_spec", "folded_example_ids", "parent_program_id",
        "parent_manifest_sha256", "compiled_at", "public_requested", "public_confirmed",
    ):
        assert key in entry, f"lineage field {key!r} was dropped by the allow-list"


def test_sensitive_fields_are_named_not_merely_absent_D_4() -> None:
    """`spec` and `examples` are classified *explicitly*, each with a reason.

    The old filter dropped `spec` and kept `examples` by omission. Naming both is what
    makes the next reader (and the guard test above) able to tell a decision from an
    oversight.
    """
    for key in ("spec", "examples"):
        assert key in lineage._HISTORY_EXCLUDED_FIELDS
        assert "sensitive" in lineage._HISTORY_EXCLUDED_FIELDS[key]
    assert not set(lineage._HISTORY_ALLOWED_FIELDS) & {"spec", "examples"}


def test_history_append_is_fsynced_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One fsync per append. A compile is slow; the fsync is free at that scale, and
    without it the lineage record of a compile can be lost to a power cut that the
    compile itself survived."""
    synced: List[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    out = tmp_path / "a.paw"
    lineage.append_history_entry(str(out), {"backend": "mock", "program_id": "p"})
    assert synced, "the history append was never fsynced"


# --- retention: bounded, and trimmed without read-modify-write --------------------


def _append_n(path: Path, n: int, payload_len: int = 400) -> None:
    for i in range(n):
        lineage.append_history_entry(
            str(path),
            {"backend": "mock", "program_id": f"p{i}", "slug": "s" * payload_len},
        )


def test_sidecar_rotates_by_replace_and_never_rewrites_the_live_file_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rotation moves the file aside with `os.replace`; it does not rewrite it.

    This is the property Report §12 measured clean (0 corrupt lines under 6 concurrent
    appending processes) and it holds *because* every append is one `write()` under
    `O_APPEND`. Rewriting the live file to drop its oldest lines -- "oldest-first
    rotation" as the finding first put it -- would reintroduce read-modify-write on a
    file several processes append to.
    """
    monkeypatch.setattr(lineage, "_HISTORY_MAX_BYTES", 2_000)
    out = tmp_path / "a.paw"
    live = Path(str(out) + ".history.jsonl")
    rotated = Path(str(live) + ".1")

    # Append one at a time up to the first rotation, so `before` is exactly what the live
    # file held at the instant it was moved aside.
    before: List[Dict[str, Any]] = []
    for i in range(100):
        if rotated.exists():
            break
        before = _history_lines(live) if live.exists() else []
        lineage.append_history_entry(
            str(out), {"backend": "mock", "program_id": f"p{i}", "slug": "s" * 400}
        )
    assert rotated.exists(), "the cap was passed and nothing rotated"
    assert before, "rotated on the very first append"
    # Byte for byte what the live file held: moved, not rewritten, nothing dropped.
    assert _history_lines(rotated) == before
    assert live.stat().st_size <= lineage._HISTORY_MAX_BYTES
    assert oct(rotated.stat().st_mode)[-3:] == "600", "the rotated file must stay 0600"


def test_sidecar_total_retention_is_bounded_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of the cap: total bytes stop growing. One generation is kept, so the
    ceiling is about twice the per-file cap however many compiles happen."""
    monkeypatch.setattr(lineage, "_HISTORY_MAX_BYTES", 2_000)
    out = tmp_path / "a.paw"
    live = Path(str(out) + ".history.jsonl")
    _append_n(out, 200)
    total = sum(
        p.stat().st_size for p in (live, Path(str(live) + ".1")) if p.exists()
    )
    assert total <= 2 * lineage._HISTORY_MAX_BYTES + 1_000, (
        f"{total} bytes retained after 200 compiles: the sidecar is still unbounded"
    )
    assert lineage._HISTORY_ROTATIONS == 1
    # Exactly two generations, never a .2 -- an unbounded rotation depth is an
    # unbounded sidecar wearing a different shape.
    assert not Path(str(live) + ".2").exists()
    # The most recent compiles are the ones kept.
    assert _history_lines(live)[-1]["program_id"] == "p199"


def _w_append_history(path: str, n: int) -> None:
    """Module-level so it is picklable. Sets the cap itself rather than relying on a
    monkeypatch surviving the process boundary."""
    import paw_kit.backend.manifest_lineage as mod

    mod._HISTORY_MAX_BYTES = 2_000
    for i in range(n):
        mod.append_history_entry(
            path, {"backend": "mock", "program_id": f"{os.getpid()}-{i}", "slug": "s" * 400}
        )


def test_concurrent_appends_across_a_rotation_corrupt_no_lines_D_4(tmp_path: Path) -> None:
    """6 processes appending while the file rotates under them: 0 corrupt lines.

    This is the arm Report §12 already verified for the *unrotated* sidecar, re-run now
    that rotation exists -- because rotation is the change most likely to have broken it.
    A process that already holds the descriptor keeps appending into the rotated-away
    inode, which is why those lines end up in the `.1` file rather than being lost or
    interleaved half-written.
    """
    out = str(tmp_path / "a.paw")
    live = Path(out + ".history.jsonl")
    workers = [
        multiprocessing.Process(target=_w_append_history, args=(out, 20))
        for _ in range(6)
    ]
    for p in workers:
        p.start()
    for p in workers:
        p.join(timeout=120)
        assert p.exitcode == 0

    corrupt: List[str] = []
    kept = 0
    for path in (live, Path(str(live) + ".1")):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                json.loads(line)
                kept += 1
            except ValueError:
                corrupt.append(line)
    assert corrupt == [], f"{len(corrupt)} corrupt line(s) written across a rotation"
    assert kept > 0
