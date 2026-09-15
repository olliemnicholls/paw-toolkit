# Contributing to paw-kit

This is for people changing paw-kit's own code, not for people using it — see
[README.md](README.md) and [docs/](docs/) for that. It holds the dev-only process
pieces (internal test-suite configuration, tooling) that don't belong on a
user-facing page.

## Running the test suite in both engine configurations

paw-kit's own test suite (`.venv/bin/python -m pytest -q -p no:cacheprovider`) needs to
pass in two configurations, because grammar-constrained decoding's engine
(`llguidance`) is optional: with it importable, and with it absent (the backend then
degrades to post-hoc validation only, with one warning — see
[real-backend](docs/real-backend.md)). A checkout with `llguidance` already installed in
`.venv` — as this project's own dev environment has it, for local iteration on
`paw_kit.schema.constraint` — exercises the engine-present configuration by default,
so the engine-absent one has to be forced rather than assumed:

```bash
PYTHONPATH=tests/_no_llguidance .venv/bin/python -m pytest -q -p no:cacheprovider
```

`tests/_no_llguidance/sitecustomize.py` sets `sys.modules["llguidance"] = None` before
any test module imports, which is what a genuinely absent `llguidance` looks like to
every call site that checks for it (`import llguidance` and
`importlib.util.find_spec("llguidance")` both fail identically) — reproducing the
no-extras install CI's own engine-absent job runs, without uninstalling anything from
`.venv`. Tests that need the engine skip cleanly in this configuration
(`pytest.importorskip("llguidance")`); nothing should fail in either configuration.

Both configurations run in CI (`ci.yml`'s "engine absent" and "engine present" jobs).
