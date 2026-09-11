"""Shared atomic file-write helper.

PAW-JIT-05: closes the TOCTOU window between "a compiled adapter file's contents are
still being written" and "another process/thread reads it" -- without this, a
concurrent inference call (or a `paw-inspect` run, or the `os.stat` a caching reader
does to detect staleness) could observe a partially-written adapter file mid-compile.
Writing to a temp file in the same directory and then atomically replacing the target
means the destination is always either the complete old file or the complete new one,
never anything in between.

Same-directory placement matters for two reasons: `os.replace` cannot atomically
rename across filesystems, and it is what guarantees the destination path is always
given a *fresh* inode on every write -- the target's old inode is still linked at that
path while the temp file (a distinct, not-yet-linked-there inode) is being written, so
they can never be equal. `paw_kit.jit.decorator`'s adapter-callable cache (PAW-JIT-05)
depends on exactly this property: it keys on `(task_id, adapter_path, os.stat
identity)`, and this is what makes a recompile over the same deterministic path always
change that identity, including across separate processes.

D-3: "atomic" here now means durable as well as indivisible. Before, the data was
written, closed and renamed with **no fsync of the file and none of the parent
directory** (`strace` counted `fsync calls: 0`), so on ext4 `data=ordered` the rename
can reach disk before the data blocks and a machine crash can leave a zero-length or
garbage-tailed file while the previous good one is already unlinked. The guarantee held
for a process crash, not for a power loss. Note what is and is not claimed: the fsyncs
below are the standard write-fsync-rename-fsync(dir) sequence, and the tests assert
that they happen; the torn-file outcome itself was never reproduced and needs failure
injection to demonstrate either way.

D-8: `os.replace` does not follow symlinks, so a symlinked destination used to be
*replaced by a regular file* -- the real target kept its stale content and every other
reader of it saw the old program. The link is resolved first, the temp file is created
beside the **resolved** target (or `os.replace` fails with `EXDEV` when the two are on
different filesystems), and an existing destination's mode is reapplied to the temp
file **before** the rename, not after, so there is no window in which the destination
sits at `mkstemp`'s `0o600`.
"""

import os
from pathlib import Path
import stat
import tempfile
from typing import Union


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename into it is durable. Best-effort.

    Guarded: opening a directory for reading is a POSIX thing, and fails on Windows
    and on some network mounts. A durability optimisation must never be the reason a
    write fails.
    """
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically and durably (see module docstring).

    Creates `path`'s parent directory if it doesn't already exist. A symlinked `path`
    is followed: its target is rewritten and the link is left in place. An existing
    destination's permission bits are preserved. On any failure (including a
    `KeyboardInterrupt` mid-write) the temp file is removed and the target is left
    untouched -- never partially written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # D-8: resolve before choosing the temp directory, and only when the destination
    # really is a symlink -- resolving unconditionally would also resolve symlinked
    # *parents*, silently moving where every write lands.
    resolved = Path(os.path.realpath(target)) if target.is_symlink() else target
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing_mode = stat.S_IMODE(os.stat(resolved).st_mode)
    except OSError:
        existing_mode = None

    fd, tmp_path = tempfile.mkstemp(
        dir=str(resolved.parent), prefix=f".{resolved.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
            # D-3: the data must be on disk before the rename that publishes it.
            f.flush()
            os.fsync(f.fileno())
        if existing_mode is not None:
            # D-8: before the rename, not after. Afterwards leaves a window in which
            # the published file sits at mkstemp's 0o600.
            os.chmod(tmp_path, existing_mode)
        os.replace(tmp_path, resolved)
        # D-3: and the rename itself must be on disk, or the directory entry can be
        # lost while the data blocks survive.
        _fsync_dir(resolved.parent)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
