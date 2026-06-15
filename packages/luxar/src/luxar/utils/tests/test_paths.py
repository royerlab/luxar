"""Unit tests for path utilities (``luxar.utils.paths``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from luxar.utils.paths import normalize_zarr_path


class TestNormalizeZarrPath:
    """``normalize_zarr_path`` enforces a canonical zarr suffix."""

    def test_appends_when_no_suffix(self):
        assert normalize_zarr_path("scene", ".luxar.zarr") == Path("scene.luxar.zarr")

    def test_corrects_bare_zarr(self):
        # plain .zarr -> canonical .luxar.zarr (the "correct if needed" case)
        assert normalize_zarr_path("scene.zarr", ".luxar.zarr") == Path(
            "scene.luxar.zarr"
        )

    def test_idempotent_on_canonical(self):
        assert normalize_zarr_path("scene.luxar.zarr", ".luxar.zarr") == Path(
            "scene.luxar.zarr"
        )

    def test_rewrites_other_canonical_suffix(self):
        # a .gsplats.zarr name normalized for scenes becomes .luxar.zarr
        assert normalize_zarr_path("blob.gsplats.zarr", ".luxar.zarr") == Path(
            "blob.luxar.zarr"
        )

    def test_gsplats_canonical(self):
        assert normalize_zarr_path("blob", ".gsplats.zarr") == Path("blob.gsplats.zarr")
        assert normalize_zarr_path("blob.zarr", ".gsplats.zarr") == Path(
            "blob.gsplats.zarr"
        )
        assert normalize_zarr_path("blob.gsplats.zarr", ".gsplats.zarr") == Path(
            "blob.gsplats.zarr"
        )

    def test_preserves_parent_directory(self):
        result = normalize_zarr_path(Path("/tmp/out/scene.zarr"), ".luxar.zarr")
        assert result == Path("/tmp/out/scene.luxar.zarr")
        assert result.parent == Path("/tmp/out")

    def test_preserves_dots_in_stem(self):
        # only a trailing recognized suffix is stripped; interior dots stay
        assert normalize_zarr_path("v1.2.scene.zarr", ".luxar.zarr") == Path(
            "v1.2.scene.luxar.zarr"
        )

    def test_accepts_str_and_path(self):
        assert normalize_zarr_path(Path("scene"), ".luxar.zarr") == Path(
            "scene.luxar.zarr"
        )

    def test_raises_without_filename(self):
        with pytest.raises(ValueError):
            normalize_zarr_path(".", ".luxar.zarr")
