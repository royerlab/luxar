"""Tests for ``utils/atomic_copy.py`` (CL-1)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from luxar.utils.atomic_copy import atomic_copytree


class TestAtomicCopytree:
    """Verify atomic semantics: destination either exists in full or not at all."""

    def test_happy_path_copies_all_files(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("alpha")
        (src / "sub").mkdir()
        (src / "sub" / "b.txt").write_text("beta")

        dst = tmp_path / "dst"
        atomic_copytree(src, dst)

        assert (dst / "a.txt").read_text() == "alpha"
        assert (dst / "sub" / "b.txt").read_text() == "beta"

    def test_existing_destination_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.txt").write_text("alpha")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "preexisting.txt").write_text("do not clobber")

        with pytest.raises(FileExistsError):
            atomic_copytree(src, dst)

        # Pre-existing content untouched.
        assert (dst / "preexisting.txt").read_text() == "do not clobber"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            atomic_copytree(tmp_path / "nope", tmp_path / "dst")

    def test_source_not_a_directory_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("not a dir")
        with pytest.raises(NotADirectoryError):
            atomic_copytree(f, tmp_path / "dst")

    def test_failure_mid_copy_leaves_no_partial_destination(
        self, tmp_path: Path
    ) -> None:
        # CL-1 contract: a failure inside copytree must NOT leave the
        # destination as a half-written tree. The atomic rename happens only
        # after copytree completes successfully.
        src = tmp_path / "src"
        src.mkdir()
        for i in range(5):
            (src / f"f{i}.txt").write_text(f"content {i}")

        dst = tmp_path / "dst"

        original_copytree = shutil.copytree

        def failing_copytree(s, d, **kwargs):
            # Run the real copytree first so the temp dir exists with
            # partial content, then raise to simulate a mid-copy failure.
            original_copytree(s, d, **kwargs)
            raise OSError("simulated mid-copy failure")

        with patch(
            "luxar.utils.atomic_copy.shutil.copytree", side_effect=failing_copytree
        ):
            with pytest.raises(OSError, match="simulated mid-copy failure"):
                atomic_copytree(src, dst)

        # Destination must NOT exist.
        assert not dst.exists(), f"atomic_copytree leaked partial destination: {dst}"

        # No leaked .tmp_* sibling either.
        leaked = list(tmp_path.glob(".tmp_dst_*"))
        assert leaked == [], f"leaked temp directories: {leaked}"
