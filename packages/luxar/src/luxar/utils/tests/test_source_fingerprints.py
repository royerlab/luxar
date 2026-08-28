"""Tests for local import-graph source fingerprints."""

from __future__ import annotations

from pathlib import Path

from luxar.utils.source_fingerprints import (
    fingerprint_imported_sources,
    imported_source_files,
)


def _source_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "repo"
    package_root = root / "src"
    examples_root = root / "examples"
    writer = package_root / "luxar/io/writer.py"
    helper = examples_root / "_helper.py"
    script = examples_root / "example.py"
    writer.parent.mkdir(parents=True)
    examples_root.mkdir(parents=True)
    (package_root / "luxar/__init__.py").write_text("")
    (writer.parent / "__init__.py").write_text("")
    writer.write_text("FORMAT = 1\n")
    helper.write_text("VALUE = 1\n")
    script.write_text("from _helper import VALUE\nfrom luxar.io import writer\n")
    return root, package_root, examples_root, script


def test_imported_source_files_support_multiple_local_roots(tmp_path: Path) -> None:
    root, package_root, examples_root, script = _source_tree(tmp_path)

    sources = imported_source_files(script, (package_root, examples_root))

    assert {path.relative_to(root).as_posix() for path in sources} == {
        "examples/_helper.py",
        "examples/example.py",
        "src/luxar/__init__.py",
        "src/luxar/io/__init__.py",
        "src/luxar/io/writer.py",
    }


def test_import_fingerprint_ignores_unrelated_files(tmp_path: Path) -> None:
    root, package_root, examples_root, script = _source_tree(tmp_path)
    unrelated = package_root / "luxar/unrelated.py"
    unrelated.write_text("VALUE = 1\n")

    before = fingerprint_imported_sources(root, script, (package_root, examples_root))
    unrelated.write_text("VALUE = 2\n")
    after_unrelated_change = fingerprint_imported_sources(
        root, script, (package_root, examples_root)
    )
    (examples_root / "_helper.py").write_text("VALUE = 2\n")
    after_helper_change = fingerprint_imported_sources(
        root, script, (package_root, examples_root)
    )

    assert after_unrelated_change == before
    assert after_helper_change != before
