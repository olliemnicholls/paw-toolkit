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


# --- D-4's fsync is scoped to compiles where it is actually free ------------------


def _count_fsyncs(monkeypatch: pytest.MonkeyPatch) -> List[int]:
    seen: List[int] = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (seen.append(fd), real(fd))[1])
    return seen


def test_real_backend_compile_fsyncs_its_lineage_line_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the finding's premise holds, the fsync happens.

    A ProgramAsWeights compile is a billed, irreversible server-side event that takes
    seconds (fast compiler) to minutes (finetune). One ~11 ms disk sync against that is
    free, and the thing it protects -- the only local record that the compile happened --
    cannot be recreated. `append_history_entry` therefore fsyncs by default.
    """
    monkeypatch.setenv("PAW_API_KEY", "paw_sk_test")
    out = tmp_path / "paw.paw"
    before = len(_count_fsyncs(monkeypatch))
    seen = _count_fsyncs(monkeypatch)
    ProgramAsWeightsBackend(sdk=FakeSDK()).compile("Spec.", EXAMPLES, str(out))
    assert len(seen) > before
    assert Path(str(out) + ".history.jsonl").is_file()


def test_mock_backend_compile_does_not_fsync_its_lineage_line_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the premise fails, the fsync is declined -- explicitly, not by oversight.

    Phase 0 cleared D-4's fsync as free because "compiles are slow". Measured, that is
    **false for the default backend**: a `MockPAWBackend` compile is ~0.2 ms of in-memory
    work, so an fsync is a ~160x slowdown on it, and `@compile_on_hit(sync_compile=True)`
    runs that compile on the caller's own request thread. What the fsync would protect is
    also worthless there: a simulated compile is reproducible for free, so losing its
    lineage line to a power cut costs nothing, unlike losing the record of a paid one.

    `tests/test_mock_backend.py::test_mock_compilation_creates_artifact` asserts the
    documented "fast deterministic compile guarantee" (< 50 ms) and is what makes this a
    regression rather than a preference. See also addendum D-ADD-5: `atomic_write_text`
    already spends ~23 ms of that budget on two fsyncs of its own (Track G's D-3), which
    is why the remaining headroom is not this track's to spend.
    """
    out = tmp_path / "mock.paw"
    seen = _count_fsyncs(monkeypatch)
    MockPAWBackend().compile("Spec.", EXAMPLES, str(out))
    history = Path(str(out) + ".history.jsonl")
    assert history.is_file(), "the lineage line is still written -- only its fsync is declined"
    assert _history_lines(history), "and it still has content"
    # Track G's `atomic_write_text` fsyncs the manifest and its parent directory, so the
    # count is not zero. What must not appear is a *third* fsync for the sidecar.
    assert len(seen) == 2, (
        f"{len(seen)} fsyncs on a mock compile (expected 2, both from atomic_write_text): "
        "the sidecar append must not add one to a compile that does 0.2 ms of work"
    )


def test_mock_compile_stays_inside_its_documented_latency_budget_D_4(tmp_path: Path) -> None:
    """The guarantee itself, asserted where this track can see it.

    Deliberately a tighter ceiling than `test_mock_compilation_creates_artifact`'s 50 ms
    and measured over several compiles, because a single-shot wall-clock assertion at 50 ms
    is what let ~23 ms of pre-existing fsync hide until a loaded machine surfaced it
    (D-ADD-5). This fails loudly if anything puts another disk sync on the default
    backend's compile path.

    **Median, not worst-of-5, and a 25 ms ceiling rather than 50 ms.** The first version
    took the slowest of five samples against 50 ms, which made the assertion a detector of
    two different things at once: a new sync on the compile path, and whoever else happened
    to be scheduled on the machine. On a shared CI runner the second one fires by itself
    (observed: 74.1 ms with no code change), so the test was red for reasons it was never
    asserting about. The statistic is now the one that actually matches the defect: a disk
    sync costs its ~23 ms on *every* compile, so it moves the median, while a noisy
    neighbour only ever moves the tail. Measured locally over 30 compiles: min 5.97 ms,
    median 6.47 ms, max 13.69 ms -- so a single added fsync lands the median near 29 ms and
    trips a 25 ms ceiling, which worst-of-5-against-50 ms would have missed unless the
    machine was already loaded. The guard is strictly tighter than before, not looser.
    """
    import statistics
    import time

    samples = []
    for i in range(9):
        out = tmp_path / f"m{i}.paw"
        t0 = time.perf_counter()
        MockPAWBackend().compile("Spec.", EXAMPLES, str(out))
        samples.append((time.perf_counter() - t0) * 1000)
    median = statistics.median(samples)
    assert median < 25.0, (
        f"median of 9 mock compiles took {median:.1f} ms against a 25 ms budget "
        f"(samples: {', '.join(f'{s:.1f}' for s in sorted(samples))})"
    )


def test_retention_policy_constants_are_the_decided_values_D_4() -> None:
    """D-4 is a DECISION-bucket item, so the decided numbers are asserted, not just used.

    A cap and a rotation depth that nothing pins are a policy that can drift to any value
    without a reviewer seeing it -- and "bounded" is only a guarantee once the bound is
    written down somewhere that fails when it changes. The reasoning for these particular
    values is in this module's docstring and in the track file's status log.

    (Kills the two `_HISTORY_MAX_BYTES` int mutants the gate-3 run found alive.)
    """
    assert lineage._HISTORY_MAX_BYTES == 1024 * 1024, "the per-file cap is 1 MiB"
    assert lineage._HISTORY_ROTATIONS == 1, (
        "one generation is kept, so total retention is bounded at ~2x the cap; an "
        "unbounded rotation depth is an unbounded sidecar wearing a different shape"
    )


def test_rotation_happens_when_the_line_would_exceed_the_cap_not_reach_it_D_4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary itself: a line that lands *exactly* on the cap does not rotate; the
    first byte past it does.

    Asserted directly on `_rotate_history_if_full` rather than through `append_history_entry`
    because a JSON line's length is not something a test should have to control to the byte
    to say what it means.

    (Kills `manifest_lineage.py cmp <=-><`, alive at gate 3.)
    """
    monkeypatch.setattr(lineage, "_HISTORY_MAX_BYTES", 1_000)
    live = tmp_path / "a.paw.history.jsonl"
    rotated = Path(str(live) + ".1")
    live.write_bytes(b"x" * 900)

    lineage._rotate_history_if_full(str(live), 100)  # 900 + 100 == the cap exactly
    assert not rotated.exists(), "a line that exactly fills the cap must not rotate"
    assert live.stat().st_size == 900

    lineage._rotate_history_if_full(str(live), 101)  # one byte past
    assert rotated.exists(), "a line that would exceed the cap must rotate"
    assert not live.exists(), "rotation moves the live file aside; it does not copy it"
