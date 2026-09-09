"""Structural regression coverage for pyproject.toml's release metadata.

Separate from `test_pyproject_deps.py`, which covers version *ranges*. This module
covers the metadata a published package is judged by: author, license, URLs,
classifiers, keywords -- plus one guard against a TOML-shaped foot-gun that actually
fired while this metadata was being added.
"""

from pathlib import Path
import tomllib

_PYPROJECT_PATH = Path(__file__).resolve().parent.parent / "pyproject.toml"

_EXPECTED_RUNTIME_DEPS = {
    "pydantic",
    "typer",
    "pyyaml",
    "interegular",
    "fastapi",
    "uvicorn",
    "httpx",
}


def _load_pyproject() -> dict:
    with open(_PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def test_runtime_dependencies_are_not_absorbed_into_a_later_table() -> None:
    """`[project].dependencies` must survive as a list of the expected packages.

    This is not hypothetical. Adding `[project.urls]` in the middle of the `[project]`
    section silently absorbed the `dependencies` array that followed it -- everything
    after a TOML table header belongs to that table -- so the package briefly declared
    *zero* runtime dependencies. `uv lock` believed it and stripped 205 lines of
    resolved packages out of the lockfile; a release built in that state would have
    installed nothing and failed at first import.

    `test_pyproject_deps.py` could not catch this: it reads
    `data["project"].get("dependencies", [])` and then asserts the combined list is
    non-empty, so the optional and dev groups alone keep it green while every runtime
    dependency vanishes.
    """
    project = _load_pyproject()["project"]

    assert isinstance(project.get("dependencies"), list), (
        "[project].dependencies is missing or not a list -- check that no [project.*] "
        "sub-table (urls, optional-dependencies, scripts) was inserted above it"
    )
    names = {
        # Strip the version specifier: "pydantic>=2.6,<3.0.0" -> "pydantic".
        dep.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()
        for dep in project["dependencies"]
    }
    assert names == _EXPECTED_RUNTIME_DEPS, (
        f"runtime dependency set changed: {names ^ _EXPECTED_RUNTIME_DEPS}. "
        "Update _EXPECTED_RUNTIME_DEPS deliberately if this is intended."
    )


def test_release_metadata_is_complete() -> None:
    """The fields a PyPI page is built from must all be present.

    Absent these, the package publishes with no author, no license shown, no links back
    to the source, and no classifiers -- which is what `paw-kit` would have shipped as
    of Track 13.
    """
    project = _load_pyproject()["project"]

    for field in ("name", "version", "description", "readme", "requires-python",
                  "authors", "license", "license-files", "keywords", "classifiers"):
        assert project.get(field), f"[project].{field} is missing or empty"

    assert Path(_PYPROJECT_PATH.parent / "LICENSE").is_file(), (
        "license-files names LICENSE but the file does not exist"
    )

    urls = project.get("urls", {})
    for key in ("Homepage", "Repository", "Issues"):
        assert key in urls, f"[project.urls] is missing {key!r}"
    for key, value in urls.items():
        assert value.startswith("https://"), f"{key} URL is not https: {value!r}"


def test_license_uses_pep639_expression_without_a_conflicting_classifier() -> None:
    """PyPI rejects an upload carrying both a PEP 639 `license` expression and a
    legacy `License ::` classifier. Declaring the expression is the modern form, so the
    classifier must stay out -- an easy thing to re-add from habit, and it fails only
    at upload time, after everything else has passed.
    """
    project = _load_pyproject()["project"]

    assert isinstance(project["license"], str), (
        "expected a PEP 639 SPDX string, not the deprecated {text = ...} table"
    )
    offenders = [c for c in project["classifiers"] if c.startswith("License ::")]
    assert not offenders, f"remove the legacy license classifier(s): {offenders}"


def test_build_backend_floor_supports_the_license_expression() -> None:
    """PEP 639 couples the metadata to the build backend's version.

    `license = "MIT"` as a string emits `License-Expression` and metadata 2.4+, which
    hatchling only supports from 1.27.0. The floor was 1.25.0 when the expression was
    introduced, so a build anywhere in the declared range could have failed. Keep the
    two in step.
    """
    data = _load_pyproject()
    requires = data["build-system"]["requires"]
    hatchling = next(r for r in requires if r.startswith("hatchling"))

    floor = hatchling.split(">=")[1].split(",")[0]
    major, minor = (int(part) for part in floor.split(".")[:2])
    assert (major, minor) >= (1, 27), (
        f"hatchling floor {floor} predates PEP 639 support; either raise it to 1.27.0 "
        "or stop using the SPDX string form of [project].license"
    )
