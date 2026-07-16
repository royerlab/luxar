"""Pure-helper smoke tests for the four gsplats interop demos.

Exercises the shared scene-building scaffolding (``_interop_common``) on a
synthetic ``.splat`` fixture — no network, no GPU — plus the per-demo config
wiring (scene tables, member paths). Each demo's ``main()`` is covered
separately by ``test_all_demos_import``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.interop.tests._synthetic import (
    make_ground_truth,
    write_antimatter_splat,
)

_DEMOS_DIR = Path(__file__).resolve().parents[1]


def _load(name: str):  # noqa: ANN202
    path = _DEMOS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_luxar_{name}_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def splat_fixture(tmp_path: Path) -> Path:
    src = tmp_path / "scene.splat"
    write_antimatter_splat(src, make_ground_truth(n=64, seed=4))
    return src


class TestInteropCommon:
    def test_stream_recipe_builds_scene(
        self, splat_fixture: Path, tmp_path: Path
    ) -> None:
        cache = build_gsplats_cache(
            splat_fixture, tmp_path / "s.gsplats.zarr", recipe="stream", n_lods=2
        )
        assert cache.exists()
        # Cache is a valid, non-empty gsplats store.
        assert GSplatData.load(cache).n_splats == 64
        scene = build_interop_scene(
            cache,
            tmp_path / "scene.luxar.zarr",
            title="t",
            layer_name="x",
            credit="c",
        )
        assert scene.exists()

    def test_tiles_recipe_builds_scene(
        self, splat_fixture: Path, tmp_path: Path
    ) -> None:
        cache = build_gsplats_cache(
            splat_fixture,
            tmp_path / "t.gsplats.zarr",
            recipe="tiles",
            max_elements=16,
            n_lods=2,
        )
        scene = build_interop_scene(
            cache,
            tmp_path / "scene_t.luxar.zarr",
            title="t",
            layer_name="x",
            credit="c",
        )
        assert scene.exists()

    def test_cache_is_reused(self, splat_fixture: Path, tmp_path: Path) -> None:
        out = tmp_path / "c.gsplats.zarr"
        build_gsplats_cache(splat_fixture, out)
        mtime = out.stat().st_mtime_ns
        # Second call without recompute must not rewrite the store.
        build_gsplats_cache(splat_fixture, out)
        assert out.stat().st_mtime_ns == mtime

    def test_no_recipe_writes_plain_leaf(
        self, splat_fixture: Path, tmp_path: Path
    ) -> None:
        cache = build_gsplats_cache(splat_fixture, tmp_path / "leaf.gsplats.zarr")
        data = GSplatData.load(cache)
        assert data.n_splats == 64
        assert data.colors is not None


class TestDemoConfig:
    def test_mipnerf_scene_table(self) -> None:
        m = _load("demo_gsplats_interop_mipnerf_garden")
        assert set(m.SCENES) == {"garden", "bicycle"}
        assert m.SCENES["garden"]["size"] == 186_713_088
        assert m._scene_arg.__call__  # helper exists

    def test_observatory_urls_are_release_assets(self) -> None:
        m = _load("demo_gsplats_interop_observatory")
        for spec in m.SCENES.values():
            assert spec["url"].startswith(
                "https://github.com/khyron/Gaussian-Splatting/releases/download"
            )

    def test_spz_scene_table(self) -> None:
        m = _load("demo_gsplats_interop_spz_scaniverse")
        assert set(m.SCENES) == {"hornedlizard", "racoonfamily"}
        assert m.BASE_URL.endswith("nianticlabs/spz/main/samples")

    def test_inria_member_path_and_size(self) -> None:
        m = _load("demo_gsplats_interop_inria_garden")
        assert m.MEMBER == "garden/point_cloud/iteration_30000/point_cloud.ply"
        assert m.MEMBER_SIZE == 1_447_027_964
        assert m.MODELS_ZIP.endswith("pretrained/models.zip")

    def test_unknown_scene_raises(self) -> None:
        m = _load("demo_gsplats_interop_mipnerf_garden")
        with pytest.raises(ValueError, match="Unknown scene"):
            m.build_scene("not-a-scene")
