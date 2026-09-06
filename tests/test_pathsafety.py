"""Unit tests for the shared filesystem-containment helper (PAW-CLI-01/02, PAW-TEST-02)."""

from pathlib import Path
import pytest

from paw_kit.pathsafety import ensure_contained


def test_ensure_contained_accepts_direct_child(tmp_path: Path) -> None:
    """A direct child of root is contained."""
    child = tmp_path / "child.txt"
    resolved = ensure_contained(child, tmp_path, label="test path")
    assert resolved == child.resolve()


def test_ensure_contained_accepts_nested_descendant(tmp_path: Path) -> None:
    """A deeply nested descendant of root is contained."""
    nested = tmp_path / "a" / "b" / "c.txt"
    resolved = ensure_contained(nested, tmp_path, label="test path")
    assert resolved == nested.resolve()


def test_ensure_contained_rejects_root_itself(tmp_path: Path) -> None:
    """Root itself does not count as contained within itself."""
    with pytest.raises(ValueError, match="not contained within"):
        ensure_contained(tmp_path, tmp_path, label="test path")


def test_ensure_contained_rejects_sibling(tmp_path: Path) -> None:
    """A path outside root entirely is rejected."""
    root = tmp_path / "root"
    root.mkdir()
    sibling = tmp_path / "sibling.txt"
    with pytest.raises(ValueError, match="not contained within"):
        ensure_contained(sibling, root, label="test path")


def test_ensure_contained_rejects_traversal(tmp_path: Path) -> None:
    """A '..'-traversal path that lexically escapes root is rejected."""
    root = tmp_path / "root"
    root.mkdir()
    traversal = root / ".." / "escaped.txt"
    with pytest.raises(ValueError, match="not contained within"):
        ensure_contained(traversal, root, label="test path")


def test_ensure_contained_error_names_the_label(tmp_path: Path) -> None:
    """The raised error identifies which value was rejected."""
    with pytest.raises(ValueError, match="--cache-dir"):
        ensure_contained(tmp_path, tmp_path, label="--cache-dir")


def test_ensure_contained_does_not_require_existence(tmp_path: Path) -> None:
    """Neither path nor root need to exist yet (safe to call before creation/write)."""
    root = tmp_path / "not_yet_created"
    candidate = root / "also_not_yet_created.paw"
    resolved = ensure_contained(candidate, root, label="test path")
    assert resolved == candidate.resolve()
