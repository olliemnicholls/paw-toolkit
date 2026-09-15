"""Contract tests against the *real* upstream `programasweights` SDK.

Every other test in this suite runs against a fake SDK (`tests/test_programasweights_backend.py`)
or the mock backend, deliberately, so the suite needs no network, no GPU and no API key. That
means nothing in CI notices if upstream renames a function, reorders a parameter, or drops a
keyword `ProgramAsWeightsBackend` passes -- users would find out first, at runtime.

These tests close that gap. They are skipped entirely when the SDK is not installed (the
normal case), and run in the scheduled `upstream-contract` workflow, which installs the
latest release on a weekly cron. They assert only the surface `paw_kit` actually calls, and
they never compile, never infer, and never need `PAW_API_KEY` -- signature inspection only,
so the job costs nothing and cannot be flaky.

When one of these fails it does NOT mean paw-kit is broken for existing users: their pinned
range (`programasweights>=0.4.4,<0.5.0` in pyproject.toml) still holds. It means the next
release is going to break them, with lead time to react.
"""

import inspect

import pytest

paw = pytest.importorskip(
    "programasweights",
    reason="upstream SDK not installed; run the scheduled upstream-contract workflow "
    "or `uv pip install 'paw-kit[real]'` to exercise these",
)


def _params(fn):
    return inspect.signature(fn).parameters


# `ProgramAsWeightsBackend` calls exactly these module-level functions. If a name
# disappears, `_paw().<name>` raises AttributeError at runtime, inside a user's compile.
#
# A-3: `precheck_compile` was missing from this list, which is how an upstream rename
# of it could have silently disabled the cache-hit leak warning forever with nothing in
# CI noticing. It belongs here (and not with `get_program_meta` below) because it *does*
# have a module-level wrapper -- `programasweights/__init__.py`.
# A-5: `cancel_compile` joins it, for the poll-loop timeout path.
@pytest.mark.parametrize(
    "name",
    [
        "compile",
        "compile_async",
        "get_compile_status",
        "function",
        "precheck_compile",
        "cancel_compile",
    ],
)
def test_sdk_still_exposes_called_function(name):
    assert hasattr(paw, name), (
        f"upstream SDK no longer exposes `programasweights.{name}`, which "
        f"paw_kit/backend/programasweights.py calls directly."
    )


def test_compile_accepts_the_public_keyword_A_3():
    """A-3: nothing asserted that `public=` is even an accepted keyword.

    `compile()` forwards `public=self.public` to both submission functions, and
    upstream's own default is `public=True`. If the keyword were renamed or dropped,
    `paw.compile(..., public=False)` would raise TypeError -- or, worse, if it became
    `**kwargs`-absorbed, every paw-kit compile would silently revert to upstream's
    public default. That is the exact leak `ProgramAsWeightsBackend`'s `public=False`
    default exists to prevent, so it gets an assertion of its own.
    """
    assert "public" in _params(paw.compile), "`public=` keyword dropped from compile()"
    assert "public" in _params(paw.compile_async), (
        "`public=` keyword dropped from compile_async()"
    )


def test_precheck_compile_accepts_spec_and_compiler_A_3():
    """`paw.precheck_compile(full_spec, compiler=self.compiler)` -- the cache-hit check."""
    p = _params(paw.precheck_compile)
    assert "spec" in p or list(p)[0] == "spec"
    assert "compiler" in p, "`compiler=` keyword dropped from precheck_compile()"


def test_paw_client_still_exposes_get_program_meta_A_2():
    """A-2's `verify_visibility` reaches the server's confirmed visibility through
    `PAWClient.get_program_meta`.

    Deliberately a separate assertion from the parametrisation above, which covers
    *module-level* functions only: `get_program_meta` has no module-level wrapper, and
    `PAWClient` is not even in `programasweights.__all__`, so it is reached as
    `programasweights.client.PAWClient`. If that moves, visibility silently stops being
    confirmable -- and `public_confirmed` would go permanently `None`, which is the
    honest failure but still one worth lead time on.
    """
    from programasweights.client import PAWClient

    assert hasattr(PAWClient, "get_program_meta"), (
        "PAWClient.get_program_meta is gone; ProgramAsWeightsBackend(verify_visibility=True) "
        "can no longer confirm a compiled program's visibility."
    )
    assert len(_params(PAWClient.get_program_meta)) >= 2  # self + program_id


def test_compile_accepts_spec_and_compiler():
    """`paw.compile(full_spec, compiler=self.compiler)` -- programasweights.py:170."""
    p = _params(paw.compile)
    assert "spec" in p or list(p)[0] == "spec"
    assert "compiler" in p, "`compiler=` keyword dropped from compile()"


def test_compile_async_accepts_spec_and_compiler():
    """`paw.compile_async(full_spec, compiler=self.compiler)` -- programasweights.py:167."""
    p = _params(paw.compile_async)
    assert "spec" in p or list(p)[0] == "spec"
    assert "compiler" in p, "`compiler=` keyword dropped from compile_async()"


def test_get_compile_status_accepts_job_id():
    """`paw.get_compile_status(job_id)` -- programasweights.py:204."""
    assert len(_params(paw.get_compile_status)) >= 1


def test_function_accepts_the_kwargs_backend_passes():
    """`paw.function(program_id, n_ctx=, offline=, n_gpu_layers=)` -- programasweights.py:252.

    These are built in `_get_function` and splatted in; a dropped keyword is a TypeError
    on the first real inference call, not at import.
    """
    p = _params(paw.function)
    for kwarg in ("n_ctx", "offline", "n_gpu_layers"):
        assert kwarg in p, f"`{kwarg}=` keyword dropped from function(); _get_function passes it"


def test_compiled_function_is_callable_with_max_tokens():
    """`fn(input_text, max_tokens=...)` -- programasweights.py:275.

    Checked on the class rather than a live program so this needs no API key and no
    cached weights.
    """
    from programasweights.runtime_llamacpp import PawFunction

    p = _params(PawFunction.__call__)
    assert "input_text" in p, "PawFunction.__call__ no longer takes `input_text`"
    assert "max_tokens" in p, "`max_tokens=` keyword dropped from PawFunction.__call__"


def test_compiled_function_accepts_logits_processor():
    """`fn(input_text, logits_processor=...)` -- the public 0.4.6 hook
    `ProgramAsWeightsBackend.infer()` depends on to apply grammar-constrained decoding
    (constrained-decoding-real-backend, Phase 3). A dropped keyword is a TypeError on
    the first grammar-constrained inference call, not at import.
    """
    from programasweights.runtime_llamacpp import PawFunction

    p = _params(PawFunction.__call__)
    assert "logits_processor" in p, (
        "`logits_processor=` keyword dropped from PawFunction.__call__; "
        "ProgramAsWeightsBackend.infer() can no longer apply grammar-constrained decoding."
    )


def test_private_sampling_hook_still_present_for_measurements():
    """Soft check on the *unshipped* constrained-decoding path.

    `scripts/measure_constrained_decoding_upstream.py` reaches `PawFunction._llm` and
    monkeypatches its `sample()` to inject `RegexLogitsProcessor`
    (`measurements/README.md`). That is a private attribute with no stability contract,
    is deliberately not used by anything in `paw_kit/`, and is documented as expected to
    break on upstream churn.

    So this is `xfail(strict=False)`, not a gate: when it starts failing, the measurement
    script's claims need re-running, but no user is affected and CI should not go red.
    """
    from programasweights.runtime_llamacpp import PawFunction

    assert "_llm" in inspect.getsource(PawFunction.__init__) or hasattr(PawFunction, "_llm")


test_private_sampling_hook_still_present_for_measurements = pytest.mark.xfail(
    strict=False,
    reason="private attribute, no stability contract; informational only -- see "
    "conductor/deferred/index.md 'Upstream logits_processor passthrough'",
)(test_private_sampling_hook_still_present_for_measurements)
