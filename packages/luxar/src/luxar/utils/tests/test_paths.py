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


class TestDatasetsDirResolution:
    """``get_datasets_dir`` must work from an installed wheel, not only a checkout (#3097)."""

    @pytest.fixture(autouse=True)
    def _clear_datasets_override(self, monkeypatch):
        monkeypatch.delenv("LUXAR_DATASETS_DIR", raising=False)

    def test_uses_repo_datasets_when_root_owns_luxar(self):
        # In this editable checkout the discovered root owns the imported luxar,
        # so output stays in <repo>/datasets (unchanged behaviour).
        from luxar.utils import paths

        got = paths.get_datasets_dir(create=False)
        assert got.name == "datasets"
        assert (got.parent / "packages" / "luxar" / "src" / "luxar").is_dir()

    def test_falls_back_to_user_cache_when_no_project_root(self, tmp_path, monkeypatch):
        # A plain `pip install` has no pyproject.toml ancestor, so get_project_root
        # raises. Before the fix this propagated and the demo could not resolve an
        # output dir at all; now it falls back to ~/.cache/luxar/datasets.
        from luxar.utils import paths

        def _no_root() -> Path:
            raise RuntimeError("no project root")

        monkeypatch.setattr(paths, "get_project_root", _no_root)
        monkeypatch.setenv("HOME", str(tmp_path))
        got = paths.get_datasets_dir(create=True)
        assert got == tmp_path / ".cache" / "luxar" / "datasets"
        assert got.is_dir()

    def test_falls_back_when_root_does_not_own_luxar(self, tmp_path, monkeypatch):
        # A non-editable install inside a user's own uv/poetry project finds the
        # USER's pyproject.toml, which does not own luxar — must not write there.
        from luxar.utils import paths

        foreign = tmp_path / "someones_project"
        foreign.mkdir()
        monkeypatch.setattr(paths, "get_project_root", lambda: foreign)
        monkeypatch.setenv("HOME", str(tmp_path))
        got = paths.get_datasets_dir(create=False)
        assert got == tmp_path / ".cache" / "luxar" / "datasets"
        assert foreign not in got.parents

    def test_demos_output_dir_under_user_cache_on_wheel(self, tmp_path, monkeypatch):
        from luxar.utils import paths

        def _no_root() -> Path:
            raise RuntimeError("no project root")

        monkeypatch.setattr(paths, "get_project_root", _no_root)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert (
            paths.get_demos_output_dir(create=True)
            == tmp_path / ".cache" / "luxar" / "datasets" / "demos"
        )

    def test_fallback_matches_demo_cache_root(self, monkeypatch):
        from luxar.demos.registry import DEMO_CACHE_ROOT
        from luxar.utils import paths

        def _no_root() -> Path:
            raise RuntimeError("no project root")

        monkeypatch.setattr(paths, "get_project_root", _no_root)
        assert paths.get_datasets_dir(create=False) == DEMO_CACHE_ROOT / "datasets"

    def test_override_wins_in_checkout_and_respects_create(self, tmp_path, monkeypatch):
        from luxar.utils import paths

        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("LUXAR_DATASETS_DIR", "~/large_disk/scenes")
        expected = tmp_path / "large_disk" / "scenes"
        assert paths.get_datasets_dir(create=False) == expected
        assert not expected.exists()
        assert paths.get_demos_output_dir() == expected / "demos"
        assert (expected / "demos").is_dir()
