"""Tests for safe zip-member extraction."""

import zipfile
from pathlib import Path

import pytest

from luxar.demos._support.downloads.zip_safety import (
    _safe_extract_zip_member,
    _validate_zip_member_path,
)


class TestSafeZipExtraction:
    """Security tests for demo bundle zip extraction helpers."""

    @pytest.mark.parametrize(
        "member",
        ["../evil.zarr.zip", "/absolute/evil.zarr.zip", "dir/../../evil", "dir\\evil"],
    )
    def test_validate_zip_member_rejects_traversal(self, member: str) -> None:
        """Unsafe archive member paths are rejected before extraction."""
        with pytest.raises(ValueError, match="Unsafe|Invalid"):
            _validate_zip_member_path(member)

    def test_safe_extract_zip_member_flattens_safe_member(self, tmp_path: Path) -> None:
        """Safe members can be copied into a controlled cache filename."""
        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("nested/data.gsplats.zarr.zip", b"payload")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with zipfile.ZipFile(bundle, "r") as zf:
            extracted = _safe_extract_zip_member(
                zf,
                "nested/data.gsplats.zarr.zip",
                cache_dir,
                target_name="data.gsplats.zarr.zip",
            )

        assert extracted == cache_dir / "data.gsplats.zarr.zip"
        assert extracted.read_bytes() == b"payload"

    def test_safe_extract_zip_member_rejects_unsafe_target(
        self, tmp_path: Path
    ) -> None:
        """Even a safe archive member cannot be written outside the cache root."""
        bundle = tmp_path / "bundle.zip"
        with zipfile.ZipFile(bundle, "w") as zf:
            zf.writestr("data.gsplats.zarr.zip", b"payload")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with zipfile.ZipFile(bundle, "r") as zf:
            with pytest.raises(ValueError, match="Unsafe extraction target"):
                _safe_extract_zip_member(
                    zf,
                    "data.gsplats.zarr.zip",
                    cache_dir,
                    target_name="../escape.gsplats.zarr.zip",
                )
