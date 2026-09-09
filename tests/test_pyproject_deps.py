"""Structural regression coverage for pyproject.toml's dependency version ranges."""

from pathlib import Path
import re
import tomllib

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(_PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def _all_dependency_specifiers(data: dict) -> list:
    """Every dependency string across [project.dependencies],
    [project.optional-dependencies] and [dependency-groups]."""
    specs = list(data["project"].get("dependencies", []))
    for group in data["project"].get("optional-dependencies", {}).values():
        specs.extend(group)
    for group in data.get("dependency-groups", {}).values():
        specs.extend(group)
    return specs


def test_pyproject_dependencies_are_all_upper_bounded_PAW_DEPS_01() -> None:
    """Verify every runtime/optional/dev dependency specifier carries both a lower
    (>=) and an upper (<) bound -- a floor-only specifier lets a fresh install
    silently resolve whatever major version is latest on PyPI at install time,
    including one released after this package and never tested against it."""
    specs = _all_dependency_specifiers(_load_pyproject())
    assert specs, "expected at least one dependency to check"

    for spec in specs:
        if spec.startswith("paw-kit"):
            # Self-reference (the `real` extra pulling in the `paw` extra) -- not
            # an external supply-chain dependency.
            continue
        assert ">=" in spec, f"{spec!r} has no lower bound"
        assert "<" in spec, f"{spec!r} has no upper bound (PAW-DEPS-01)"


def test_pyproject_interegular_is_upper_bounded_PAW_DEPS_03() -> None:
    """Verify interegular specifically carries an upper bound -- its
    algorithmic-complexity half is closed by PAW-SCHEMA-03 (Track 09), so this is the
    supply-chain half only."""
    specs = _all_dependency_specifiers(_load_pyproject())
    interegular_specs = [s for s in specs if re.match(r"^interegular\b", s)]
    assert interegular_specs, "expected an interegular dependency entry"
    assert all("<" in s for s in interegular_specs)


def test_pyproject_build_backend_is_upper_bounded_PAW_DEPS_02() -> None:
    """Regression guard for Track 09's PAW-DEPS-02 fix: the build-system requirement
    must stay upper-bounded too, not just the runtime dependencies."""
    data = _load_pyproject()
    build_requires = data.get("build-system", {}).get("requires", [])
    assert build_requires
    assert all("<" in r for r in build_requires)
