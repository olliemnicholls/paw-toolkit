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
"""

import os
from pathlib import Path
import tempfile
from typing import Union


def atomic_write_text(path: Union[str, Path], content: str, *, encoding: str = "utf-8") -> None:
    """Write `content` to `path` atomically (see module docstring).

    Creates `path`'s parent directory if it doesn't already exist. On any failure
    (including a `KeyboardInterrupt` mid-write) the temp file is removed and the
    target is left untouched -- never partially written.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, target)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
