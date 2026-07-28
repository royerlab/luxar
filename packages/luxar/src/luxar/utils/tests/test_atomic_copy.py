"""Tests for ``utils/atomic_copy.py`` (CL-1)."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from luxar.utils.atomic_copy import atomic_copy_file, atomic_copytree


def _partial_then_fail(_src: object, dst_arg: object) -> None:
    """Stand-in for ``shutil.copy2`` that writes PARTIAL bytes to whatever
    destination path it is handed (the temp sibling under correct code, the
    canonical dst under mutation), then raises — modelling a real mid-write
    interruption (disk full, SIGKILL)."""
    Path(dst_arg).write_bytes(b"partial-truncated")  # type: ignore[arg-type]
    raise OSError("disk full mid-write")


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

    # [Python-R2/D-W5] Strengthen the mid-copy failure scenario: simulate
    # a REAL mid-copytree failure where the tmp dir is HALF-populated at
    # the moment of the exception (not the prior test's "succeeded then
    # raised" model). The cleanup path must still clear the tmp dir.
    def test_failure_during_copytree_clears_half_populated_tmp(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        for i in range(5):
            (src / f"f{i}.txt").write_text(f"content {i}")

        dst = tmp_path / "dst"

        def half_populating_failing_copytree(s, d, **kwargs):
            # Manually create the destination directory and populate it
            # with one file, then raise — emulating shutil.copytree that
            # crashed after writing some but not all entries.
            d_path = Path(d)
            d_path.mkdir(parents=True, exist_ok=True)
            (d_path / "f0.txt").write_text("partial")
            raise OSError("disk full mid-copy")

        with patch(
            "luxar.utils.atomic_copy.shutil.copytree",
            side_effect=half_populating_failing_copytree,
        ):
            with pytest.raises(OSError, match="disk full mid-copy"):
                atomic_copytree(src, dst)

        # The half-populated tmp must be gone; the dst must never have
        # been created.
        assert not dst.exists()
        leaked = list(tmp_path.glob(".tmp_dst_*"))
        assert leaked == [], f"half-populated temp dirs leaked: {leaked}"


class TestAtomicCopyFile:
    """File-level counterpart to atomic_copytree (cache-refresh semantics)."""

    def test_copies_content_and_preserves_mtime(self, tmp_path: Path) -> None:
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dst = tmp_path / "sub" / "dst.bin"

        out = atomic_copy_file(src, dst)

        assert out == dst
        assert dst.read_bytes() == b"payload"
        assert dst.stat().st_mtime == pytest.approx(src.stat().st_mtime, abs=1e-3)

    def test_existing_destination_is_replaced(self, tmp_path: Path) -> None:
        """Unlike atomic_copytree, overwrite is the point — this refreshes a cache."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"new")
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"old-and-longer")

        atomic_copy_file(src, dst)

        assert dst.read_bytes() == b"new"

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            atomic_copy_file(tmp_path / "nope", tmp_path / "dst")

    def test_failure_mid_write_preserves_existing_dst_and_cleans_tmp(
        self, tmp_path: Path
    ) -> None:
        """A copy that fails AFTER writing partial bytes must leave a
        pre-existing ``dst`` untouched and leak no ``.tmp_*`` sibling.

        The patched ``copy2`` writes partial bytes to whatever path it is
        handed (the temp sibling under correct code, or the canonical ``dst``
        under a mutation that skips the temp+rename), then raises. This pins
        the atomicity property: it kills both (a) dropping the temp cleanup
        (would leak a partial ``.tmp_*``) and (b) copying straight to ``dst``
        (would clobber ``dst`` with partial bytes).
        """
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dst = tmp_path / "dst.bin"
        dst.write_bytes(b"previous")

        with patch("shutil.copy2", side_effect=_partial_then_fail):
            with pytest.raises(OSError, match="disk full mid-write"):
                atomic_copy_file(src, dst)

        # Canonical name must keep its prior content (kills mutation b).
        assert dst.read_bytes() == b"previous"
        # Exactly the two originals remain — no partial temp sibling and no
        # stray file under any prefix (kills mutation a even if it renames).
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.bin", "src.bin"]

    def test_failure_mid_write_leaves_no_dst_when_absent(self, tmp_path: Path) -> None:
        """Same partial-write failure, but ``dst`` did NOT pre-exist: the
        canonical name must never appear with partial bytes, and no ``.tmp_*``
        sibling may leak."""
        src = tmp_path / "src.bin"
        src.write_bytes(b"payload")
        dst = tmp_path / "dst.bin"  # does NOT exist

        with patch("shutil.copy2", side_effect=_partial_then_fail):
            with pytest.raises(OSError, match="disk full mid-write"):
                atomic_copy_file(src, dst)

        # Canonical name must never appear with partial bytes (kills mutation b).
        assert not dst.exists()
        # Only the source remains — no partial temp sibling and no stray file
        # under any prefix (kills mutation a even if it renames).
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src.bin"]
