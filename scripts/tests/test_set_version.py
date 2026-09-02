"""Tests for release version stamping, consistency, and package metadata.

These release mechanisms run exactly once per release — so a defect surfaces
on launch day, in front of everyone, with no earlier signal. That asymmetry is
why they are tested here rather than trusted.

The gate tests deliberately assert that it *fails*: a consistency check that
cannot go red is indistinguishable from no check at all.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SET_VERSION = REPO / "scripts/set_version.py"
CHECK_VERSIONS = REPO / "scripts/check_version_consistency.py"
RELEASE = REPO / "scripts/release.sh"


def test_legacy_npm_package_name_is_absent_from_tracked_files() -> None:
    legacy_name = "@royerlab" + "/luxar-viewer"
    result = subprocess.run(
        ["git", "grep", "-n", "-F", "--", legacy_name],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1, result.stdout or result.stderr


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def set_version_module() -> ModuleType:
    return _load(SET_VERSION, "set_version")


@pytest.fixture
def check_module() -> ModuleType:
    return _load(CHECK_VERSIONS, "check_version_consistency")


def _checkout(tmp_path: Path, version: str = "2026.06.05") -> dict[str, Path]:
    """A miniature repo carrying the three version representations."""
    init = tmp_path / "__init__.py"
    init.write_text(f'"""Luxar."""\n\n__version__ = "{version}"\n\n__all__ = []\n')

    pkg_json = tmp_path / "package.json"
    semver = ".".join(str(int(p)) for p in version.split("."))
    pkg_json.write_text(
        json.dumps({"name": "@luxar/viewer", "version": semver}, indent=2) + "\n"
    )

    citation = tmp_path / "CITATION.cff"
    citation.write_text(
        "cff-version: 1.2.0\n"
        'title: "Luxar"\n'
        f'version: "{version}"\n'
        f'date-released: "{version.replace(".", "-")}"\n'
    )
    return {"init": init, "pkg_json": pkg_json, "citation": citation}


def _point_at(
    module: ModuleType,
    paths: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "INIT", paths["init"])
    monkeypatch.setattr(module, "PKG_JSON", paths["pkg_json"])
    monkeypatch.setattr(module, "CITATION", paths["citation"])


# --------------------------------------------------------------------------- #
# set_version.py
# --------------------------------------------------------------------------- #


def test_stamps_all_three_representations(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0

    assert '__version__ = "2026.09.15"' in paths["init"].read_text()
    # npm/semver forbids leading zeros; the same release, spelled differently.
    assert json.loads(paths["pkg_json"].read_text())["version"] == "2026.9.15"
    cff = paths["citation"].read_text()
    assert 'version: "2026.09.15"' in cff
    assert 'date-released: "2026-09-15"' in cff


def test_zero_padding_is_preserved_where_it_matters(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-digit month/day is the case the two spellings diverge on."""
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.01.02"]) == 0

    assert '__version__ = "2026.01.02"' in paths["init"].read_text()
    assert json.loads(paths["pkg_json"].read_text())["version"] == "2026.1.2"
    assert 'version: "2026.01.02"' in paths["citation"].read_text()
    assert 'date-released: "2026-01-02"' in paths["citation"].read_text()


@pytest.mark.parametrize(
    "bad",
    [
        "2026.6.5",
        "2026-06-05",
        "v2026.06.05",
        "1.2.3",
        "2026.06",
        "2026.02.31",
        "2026.13.01",
        "",
    ],
)
def test_rejects_non_calver(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad: str,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)
    before = {name: path.read_text() for name, path in paths.items()}

    assert set_version_module.main(["set_version.py", bad]) == 2
    assert {name: path.read_text() for name, path in paths.items()} == before


def test_reports_a_citation_missing_its_stampable_lines(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Silently skipping would reintroduce exactly the drift this closes."""
    paths = _checkout(tmp_path)
    paths["citation"].write_text('cff-version: 1.2.0\ntitle: "Luxar"\n')
    _point_at(set_version_module, paths, tmp_path, monkeypatch)
    before = {name: path.read_text() for name, path in paths.items()}

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 1
    assert {name: path.read_text() for name, path in paths.items()} == before


def test_stamps_citation_lines_without_spaces_after_colons(
    set_version_module: ModuleType,
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    paths["citation"].write_text(
        "cff-version: 1.2.0\nversion:2026.06.05\ndate-released:2026-06-05\n"
    )
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    assert 'version: "2026.09.15"' in paths["citation"].read_text()
    assert 'date-released: "2026-09-15"' in paths["citation"].read_text()


def test_is_idempotent(
    set_version_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(set_version_module, paths, tmp_path, monkeypatch)

    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    once = {k: p.read_text() for k, p in paths.items()}
    assert set_version_module.main(["set_version.py", "2026.09.15"]) == 0
    assert {k: p.read_text() for k, p in paths.items()} == once


# --------------------------------------------------------------------------- #
# check_version_consistency.py — each assertion is that the gate goes RED
# --------------------------------------------------------------------------- #


def test_gate_passes_on_a_consistent_checkout(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0


def test_gate_catches_a_stale_viewer_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _checkout(tmp_path)
    paths["pkg_json"].write_text(json.dumps({"version": "2026.5.1"}) + "\n")
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1
    assert "make set-version DATE=2026.06.05" in capsys.readouterr().err


@pytest.mark.parametrize("version", ["2026.6.5", "2026.02.31"])
def test_gate_rejects_an_invalid_python_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    version: str,
) -> None:
    paths = _checkout(tmp_path, version)
    _point_at(check_module, paths, tmp_path, monkeypatch)

    assert check_module.main() == 2
    assert str(paths["init"]) in capsys.readouterr().err


def test_gate_catches_a_stale_citation_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _checkout(tmp_path)
    cff = paths["citation"].read_text().replace("2026.06.05", "2026.05.01")
    paths["citation"].write_text(cff)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1
    assert "make set-version DATE=2026.06.05" in capsys.readouterr().err


def test_gate_catches_a_date_that_disagrees_with_the_version(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The subtle one: right version, wrong release date."""
    paths = _checkout(tmp_path)
    cff = paths["citation"].read_text().replace("2026-06-05", "2026-06-06")
    paths["citation"].write_text(cff)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 1


def test_gate_reports_a_citation_missing_its_fields(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _checkout(tmp_path)
    paths["citation"].write_text('cff-version: 1.2.0\ntitle: "Luxar"\n')
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 2


@pytest.mark.parametrize(
    "citation",
    [
        "cff-version: 1.2.0\nversion: 2026.06.05\ndate-released: 2026-06-05\n",
        "cff-version: 1.2.0\nversion: '2026.06.05'\ndate-released: '2026-06-05'\n",
        'cff-version: 1.2.0\nversion: "2026.06.05"  # tag v2026.06.05\n'
        'date-released: "2026-06-05"\n',
    ],
)
def test_gate_accepts_valid_yaml_scalar_spellings(
    check_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    citation: str,
) -> None:
    """The gate must accept ordinary YAML quoting and inline comments."""
    paths = _checkout(tmp_path)
    paths["citation"].write_text(citation)
    _point_at(check_module, paths, tmp_path, monkeypatch)
    assert check_module.main() == 0


# --------------------------------------------------------------------------- #
# The real repo
# --------------------------------------------------------------------------- #


def test_the_committed_tree_is_consistent(check_module: ModuleType) -> None:
    """Guards the actual files, not a fixture — this is what CI runs for."""
    assert check_module.main() == 0
    citation = yaml.safe_load((REPO / "CITATION.cff").read_text())
    init_text = (REPO / "packages/luxar/src/luxar/__init__.py").read_text()
    version_match = check_module.CALVER_RE.search(init_text)
    assert version_match is not None
    version = version_match.group(1)
    assert isinstance(citation["version"], str)
    assert isinstance(citation["date-released"], str)
    assert citation["version"] == version
    assert citation["date-released"] == version.replace(".", "-")


def test_release_remedy_passes_the_version_as_a_make_variable() -> None:
    assert "make set-version DATE=$VERSION" in RELEASE.read_text()


def test_set_version_and_the_gate_agree_on_where_the_files_are() -> None:
    """A path that drifts between the writer and the checker fails silently."""
    writer = _load(SET_VERSION, "set_version_paths")
    checker = _load(CHECK_VERSIONS, "check_versions_paths")
    for attr in ("REPO", "INIT", "PKG_JSON", "CITATION"):
        assert getattr(writer, attr) == getattr(checker, attr), attr
