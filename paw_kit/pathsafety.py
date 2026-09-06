"""Shared filesystem-containment helper.

Closes the CWE-22 / CWE-73 ("External Control of File Name or Path") class of
findings from the security audit: a filesystem path sourced from a CLI flag or a
YAML config file (`--cache-dir`, `suite.yaml`'s `adapter_path`) used directly for
deletion or write, with no check that it stays inside the directory the caller
actually intended. Used by `paw_kit.cli` (`PAW-CLI-01`, `PAW-CLI-02`) and, in a
later phase, `paw_kit.test.suite` / `paw_kit.test.active` (`PAW-TEST-02`).
"""

from pathlib import Path
from typing import Union


def ensure_contained(path: Union[str, Path], root: Union[str, Path], *, label: str = "path") -> Path:
    """Resolve `path` and verify it is strictly contained within `root`.

    "Strictly" means `path` must resolve to a real descendant of `root` -- `root`
    itself does not count as contained within itself. This deliberately blocks the
    audit's own attack scenarios: pointing `--cache-dir` at `.` (would resolve equal
    to `root`) or at an unrelated directory like `/etc/my_app` (would resolve outside
    `root` entirely) are both rejected; a genuine subdirectory of `root` is not.

    Neither `path` nor `root` need to exist on disk -- resolution is purely lexical
    for the non-existent tail of a path, so this is safe to call before creating a
    cache directory or before a write that will create its parent directories.

    Raises:
        ValueError: if the resolved path is not a strict descendant of the resolved
            root. `label` identifies the offending value in the message (e.g.
            "--cache-dir" or "suite.yaml's adapter_path").
    """
    resolved_root = Path(root).resolve()
    resolved_path = Path(path).resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(
            f"{label} {str(path)!r} resolves to {resolved_path}, which is not contained "
            f"within the expected root {resolved_root}. Refusing to touch it."
        )
    return resolved_path
