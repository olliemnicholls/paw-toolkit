"""Shared manifest-lineage helpers used by every `AbstractPAWBackend.compile()`.

Both `ProgramAsWeightsBackend` and `MockPAWBackend` write a `.paw` manifest that
today records *how many* examples were folded into a compile but not *which* ones,
carries no schema version or spec hash, and has no link back to whatever adapter it
just overwrote. This module is the one place that logic lives, so the two backends'
manifests stay in lockstep rather than drifting.

`read_parent_lineage` deliberately runs *before* `atomic_write_text` replaces
`output_path`: once that call returns, the previous manifest -- and the only record
of what `parent_program_id`/`parent_manifest_sha256` should be -- is gone. Recompiling
into a fresh path (no pre-existing file) is the normal first-compile case, so "no
readable manifest there yet" returns `(None, None)` rather than raising.

The append-only sidecar (`append_history_entry`) is what actually preserves lineage
across repeated overwrites: a manifest only ever points at its immediate parent, but
`<adapter>.paw.history.jsonl` accumulates one line per compile, so `paw-kit history`
can show the full chain even though each individual manifest cannot.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# History sidecar lines are opened with this mode on creation (0600, not subject to
# the process umask) -- same rationale as PAW-CLI-05 in paw_kit.cli's dataset export:
# a compile manifest's spec text (folded into the corresponding history line minus
# the spec) can carry traced production input/output pairs, which is not something
# to leave world-readable by default.
_HISTORY_FILE_MODE = 0o600

# --- D-4: the lineage sidecar's retention policy ---------------------------------
#
# This used to be a one-key **deny-list** (`if k != "spec"`). The default backend --
# `MockPAWBackend`, the one the README's own example uses -- retains every traced
# input/output pair verbatim in `examples`, which walked straight past it: three mock
# compiles of 500 examples produced a 216 KB sidecar of raw production rows, append-only,
# uncapped and never pruned.
#
# A deny-list is the wrong shape for this, not just wrongly populated. It has to
# enumerate every *future* way some backend might carry raw traffic, and whoever adds the
# next such field is not thinking about this file. An allow-list fails the other way --
# towards dropping a new lineage field rather than publishing a new sensitive one -- and
# `tests/test_manifest_lineage.py` turns that remaining failure mode into a test that
# makes adding a manifest key force a decision.
#
# What belongs here: identifiers, hashes, counts, visibility provenance, parent pointers
# and timings. What does not: anything derived from the traced traffic itself.
_HISTORY_ALLOWED_FIELDS = frozenset(
    {
        # identity and schema
        "backend",
        "manifest_version",
        "program_id",
        "slug",
        "cached_program_id",
        "compiler",
        "compiler_snapshot",
        "status",
        # content fingerprints -- a hash of the spec, never the spec
        "spec_sha256",
        "full_spec_sha256",
        # what was folded, by count and by id (ids are SHA-256 digests)
        "examples_count",
        "examples_folded_into_spec",
        "folded_example_ids",
        # visibility provenance (A-2). `public` is the pre-A-2 name, kept so a sidecar
        # line written by an older paw-kit still carries the value it recorded.
        "public_requested",
        "public_confirmed",
        "public_confirmed_reason",
        "public",
        "ephemeral",
        "cache_hit",
        # lineage
        "parent_program_id",
        "parent_manifest_sha256",
        # timings
        "compile_wall_s",
        "compiled_at",
    }
)

# Named, not merely omitted. The old filter dropped `spec` explicitly and kept
# `examples` by oversight; enumerating both -- with a reason -- is what lets a reader
# (and the allow-list evolution test) tell a decision from an accident.
_HISTORY_EXCLUDED_FIELDS = {
    "spec": (
        "sensitive: the spec text, which ProgramAsWeightsBackend.compile() folds traced "
        "production input/output pairs into before sending. Its hash is retained instead."
    ),
    "examples": (
        "sensitive: the raw traced input/output pairs themselves. MockPAWBackend retains "
        "them verbatim in the manifest as live inference signal; the sidecar records "
        "`examples_count` and `folded_example_ids` instead."
    ),
    "rules": "not lineage: backend configuration, not a record of this compile.",
    "default_response": "not lineage: backend configuration.",
}

#: Per-file cap on the sidecar. One line is a few hundred bytes of lineage, so 1 MiB is
#: on the order of a thousand compiles of history -- far more than `paw-kit history` has
#: ever needed to show, and a bound instead of no bound.
_HISTORY_MAX_BYTES = 1024 * 1024

#: Generations kept beside the live file (`<sidecar>.1`). Total retention is therefore
#: bounded at roughly `(_HISTORY_ROTATIONS + 1) * _HISTORY_MAX_BYTES`. An unbounded
#: rotation depth would be an unbounded sidecar wearing a different shape.
_HISTORY_ROTATIONS = 1


def sha256_text(text: str) -> str:
    """SHA-256 hex digest of `text`, encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def example_id(example: Dict[str, Any]) -> Optional[str]:
    """SHA-256 of `input + "\\x1f" + output` for a well-formed example dict.

    Returns None for anything that isn't a dict with string `input` and `output`
    keys -- callers filter those out rather than hash a placeholder for them, so a
    malformed/foreign example never silently gets a lineage id.
    """
    if not isinstance(example, dict):
        return None
    inp, out = example.get("input"), example.get("output")
    if not isinstance(inp, str) or not isinstance(out, str):
        return None
    return sha256_text(inp + "\x1f" + out)


def select_folded_examples(
    examples: List[Dict[str, Any]], limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """The subset of `examples` actually folded into a compile, in order.

    Mirrors `programasweights._render_spec_with_examples`'s own "usable" filter
    (dict, with both `input` and `output` present) so `folded_example_ids` always
    names exactly the examples that ended up in the rendered spec text, not a
    superset of them. `limit=None` means "no cap" (used by the mock backend, which
    does not fold examples into spec text at all and so has no analogous limit).
    """
    usable = [ex for ex in examples if isinstance(ex, dict) and "input" in ex and "output" in ex]
    if limit is None:
        return usable
    if limit <= 0:
        return []
    return usable[:limit]


def folded_example_ids(examples: List[Dict[str, Any]], limit: Optional[int] = None) -> List[str]:
    """`example_id` for each example `select_folded_examples` selects, in order."""
    ids = []
    for ex in select_folded_examples(examples, limit):
        eid = example_id(ex)
        if eid is not None:
            ids.append(eid)
    return ids


def read_parent_lineage(output_path: Union[str, Path], max_bytes: int) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort `(parent_program_id, parent_manifest_sha256)` from an existing
    manifest at `output_path`, read *before* it is overwritten.

    Both are None when there is nothing there yet (the ordinary first-compile case),
    when the existing file is not a regular file, or when it exceeds `max_bytes` --
    mirroring the size-guard pattern the rest of this codebase uses before parsing an
    externally-writable path (PAW-CLI-06, PAW-BACKEND-04). `parent_manifest_sha256`
    is computed from the raw file bytes even if the content turns out not to be valid
    JSON (or not a dict, or has no `program_id`) -- the hash still identifies exactly
    what was there, which is the point of recording it; only `parent_program_id`
    additionally requires a parseable `program_id` string.
    """
    path = Path(output_path)
    try:
        if not path.is_file() or path.stat().st_size > max_bytes:
            return None, None
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None, None

    parent_manifest_sha256 = sha256_text(raw)
    parent_program_id: Optional[str] = None
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    if isinstance(data, dict):
        pid = data.get("program_id")
        if isinstance(pid, str):
            parent_program_id = pid
    return parent_program_id, parent_manifest_sha256


def _rotate_history_if_full(history_path: str, incoming_bytes: int) -> None:
    """Move the sidecar aside when the next line would take it past its cap (D-4).

    `os.replace` to `<sidecar>.1`, deliberately -- **never** a rewrite of the live file.
    Report §12 measured this sidecar clean under 6 concurrent appending processes (0
    corrupt lines), and that holds precisely *because* each append is one `write()` under
    `O_APPEND` rather than a read-modify-write. Dropping the oldest lines in place, which
    is how the finding first described "rotation", would have reintroduced exactly the
    shape that measurement proved safe to avoid.

    A process that already holds a descriptor on the rotated-away inode keeps appending
    into it; those lines live on in the `.1` file. Nothing is interleaved or truncated.

    Best-effort: a sidecar that cannot be rotated is still appended to. Retention is a
    policy, and failing a compile over it would be worse than exceeding it.
    """
    try:
        size = os.stat(history_path).st_size
    except OSError:
        return  # nothing there yet, or unreadable -- either way, nothing to rotate
    if size + incoming_bytes <= _HISTORY_MAX_BYTES:
        return
    try:
        os.replace(history_path, f"{history_path}.{_HISTORY_ROTATIONS}")
    except OSError:
        pass


def append_history_entry(output_path: Union[str, Path], manifest: Dict[str, Any]) -> None:
    """Append one line of **lineage** to `<output_path>.history.jsonl`.

    The line is `manifest` filtered through `_HISTORY_ALLOWED_FIELDS` -- ids, hashes,
    counts, visibility provenance, parent pointers, timings. See that constant for why
    this is an allow-list and not the deny-list it used to be (D-4): a one-key deny-list
    dropped `spec` and let the mock backend's `examples`, the traced input/output pairs
    themselves, through untouched.

    Append-only by construction (`os.O_APPEND`) rather than read-modify-write, so
    concurrent compiles against distinct adapters never contend, and a compile that
    dies mid-write leaves every prior line intact. `0o600` is set at creation time via
    `os.open`'s mode argument (not a later `chmod`), the same reasoning as
    `atomic_write_text`/PAW-CLI-05 elsewhere in this codebase: a bare `open(..., "a")`
    is subject to the process umask (commonly 0644) and would leave a first-ever line
    briefly world-readable. `os.open`'s mode argument only governs permissions at
    *creation*; it has no effect on a file that already exists, which is the expected
    case for every compile after the first.

    Two additions from D-4 beyond the filter: the file is rotated rather than grown
    without limit (see `_rotate_history_if_full`), and the append is `fsync`ed. A compile
    takes seconds to minutes, so one fsync costs nothing measurable at this call site, and
    without it the lineage record of a compile can be lost to a power cut the compile
    itself survived.
    """
    history_path = str(output_path) + ".history.jsonl"
    entry = {k: v for k, v in manifest.items() if k in _HISTORY_ALLOWED_FIELDS}
    line = json.dumps(entry) + "\n"
    _rotate_history_if_full(history_path, len(line.encode("utf-8")))
    fd = os.open(history_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, _HISTORY_FILE_MODE)
    try:
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        # os.fdopen took ownership of fd; on the (unlikely) chance it fails before
        # that handoff completes, avoid leaking the raw descriptor.
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def extract_snapshot(obj: Any) -> Any:
    """Best-effort compiler snapshot off an upstream compiler response object.

    The upstream `Program` dataclass (programasweights 0.4.4, `client.py`) names it
    `compiler_snapshot` (e.g. `paw-4b-qwen3-0.6b-20260407`); `snapshot` is accepted as
    a fallback. None if `obj` is None or carries neither.
    """
    if obj is None:
        return None
    for key in ("compiler_snapshot", "snapshot"):
        if isinstance(obj, dict):
            value = obj.get(key)
        else:
            value = getattr(obj, key, None)
        if value is not None:
            return value
    return None
