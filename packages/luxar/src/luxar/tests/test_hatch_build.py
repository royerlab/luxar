"""Tests for Luxar's custom Hatch wheel build hook."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Pytest imports this file through the ``luxar.tests`` package, so the repository
# root (which owns hatch_build.py) is not otherwise on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from hatch_build import LuxarBuildHook  # noqa: E402


def make_hook(root: Path) -> LuxarBuildHook:
    """Construct the hook with inert Hatchling collaborators."""
    return LuxarBuildHook(
        root=str(root),
        config={},
        build_config=MagicMock(),
        metadata=MagicMock(),
        directory=str(root / "build"),
        target_name="wheel",
    )


def test_editable_build_replaces_viewer_force_include(tmp_path: Path) -> None:
    """Editable builds use a non-empty marker map instead of viewer ``dist``."""
    build_data: dict[str, object] = {}

    make_hook(tmp_path).initialize("editable", build_data)

    assert build_data == {
        "force_include_editable": {
            "hatch_build.py": "_luxar_editable_build_marker",
        }
    }


def test_standard_build_requires_viewer_index(tmp_path: Path) -> None:
    """Standard wheels fail early when the production viewer is absent."""
    with pytest.raises(FileNotFoundError, match=r"make build-viewer"):
        make_hook(tmp_path).initialize("standard", {})


def test_standard_build_accepts_viewer_index(tmp_path: Path) -> None:
    """Standard wheels proceed once the production viewer index exists."""
    viewer_index = tmp_path / "packages" / "luxar-viewer" / "dist" / "index.html"
    viewer_index.parent.mkdir(parents=True)
    viewer_index.touch()
    build_data: dict[str, object] = {}

    make_hook(tmp_path).initialize("standard", build_data)

    assert build_data == {}


def test_unknown_build_version_is_rejected(tmp_path: Path) -> None:
    """Unexpected Hatchling build modes fail instead of bypassing validation."""
    with pytest.raises(ValueError, match=r"expected 'editable' or 'standard'"):
        make_hook(tmp_path).initialize("unexpected", {})
