"""Engine-absent test configuration shim.

Add this directory to `PYTHONPATH` (e.g. `PYTHONPATH=tests/_no_llguidance`) before
running pytest to force the "llguidance not installed" configuration, without
actually uninstalling the package from `.venv`. This matters because this checkout's
`.venv` has `llguidance` 1.8.0 installed directly and it is not, and cannot be,
made absent by `uv sync`/`uv lock` alone: `llguidance` lives behind the `paw` extra
(`pyproject.toml`), and neither command uninstalls a package that is already present
but simply not part of the currently-resolved extra set. So a plain
`uv run pytest` / `.venv/bin/python -m pytest` here always exercises the
engine-**present** configuration; CI's `uv sync --dev` (no extras) is the only thing
that naturally exercises the engine-**absent** one (`decisions.md` §3, H-10). This
shim forces the absent configuration locally too, so both configurations can be
verified on the same machine before either one reaches CI.

`sitecustomize.py` is imported automatically by Python's `site` module at
interpreter startup, before user code runs, whenever its containing directory is on
`sys.path` -- which `PYTHONPATH` puts there ahead of the installed-package
directories, so this shim's `sitecustomize.py` wins over any other same-named module
on the path. Setting `sys.modules["llguidance"] = None` here makes every subsequent
`import llguidance` raise `ImportError` and every `importlib.util.find_spec
("llguidance")` return `None` too (both are documented consequences of a `None`
entry in `sys.modules` -- see the Python language reference on the import system),
reproducing CI's no-extras install byte for byte, at every call site that checks for
the engine's presence, rather than monkeypatching only the one or two call sites a
test happens to exercise.

Usage, from the repository root:

    PYTHONPATH=tests/_no_llguidance .venv/bin/python -m pytest -q -p no:cacheprovider

See `CONTRIBUTING.md` for the two configurations this project's suite is run in, and
`.github/workflows/ci.yml` for how CI exercises both without this shim (the
engine-absent job is `uv sync --dev` with no extras; the engine-present job installs
only `llguidance`/`numpy` on top of that, never the whole `paw` extra).
"""

import sys

sys.modules["llguidance"] = None
