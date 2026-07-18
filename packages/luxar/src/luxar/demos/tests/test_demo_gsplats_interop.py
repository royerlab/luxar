"""Pure-helper smoke tests for the gsplats interop demos.

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
import zarr

from luxar.demos._interop_common import build_gsplats_cache, build_interop_scene
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.interop.tests._synthetic import (
    make_ground_truth,
    write_antimatter_splat,
)


def _find_gsplats_blending_mode(scene_path: Path, layer_name: str) -> str | None:
    """Return the ``blending_mode`` attr stored on a scene's gsplats layer node."""
    root = zarr.open_group(str(scene_path), mode="r")
    return dict(root[layer_name].attrs).get("blending_mode")


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

    def test_tiled_scene_propagates_normal_blending_to_parts(
        self, splat_fixture: Path, tmp_path: Path
    ) -> None:
        # A tiled import is a kind=partition: the leaf writer stamps a DEFAULT
        # blending_mode="additive" on each part, which the viewer reads per-part
        # and which shadows the wrapper's "normal". Without the graft propagating
        # blending_mode to children, tiled photogrammetric imports would render
        # as additive GLOW instead of the surface-like alpha-over they need.
        # Assert every part carries the wrapper's "normal" (not "additive").
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
            layer_name="lizard",
            credit="c",
        )
        root = zarr.open_group(str(scene), mode="r")
        layer = root["lizard"]
        part_names = [k for k in layer.group_keys() if k.startswith("part_")]
        assert part_names, "expected a kind=partition with part_* children"
        for pn in part_names:
            assert dict(layer[pn].attrs).get("blending_mode") == "normal", (
                f"part {pn} must inherit the layer's normal blending, not additive"
            )

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

    def test_scene_uses_normal_blending_mode(
        self, splat_fixture: Path, tmp_path: Path
    ) -> None:
        # R10a-critical invariant: classical/photogrammetric captures are
        # surface-like and MUST composite via ``normal`` (alpha-over) blending,
        # which only renders correctly once depth sorting is on. If a refactor
        # silently reverted this to additive glow the demos would look wrong,
        # so pin it explicitly (the config smoke tests never check the scene).
        cache = build_gsplats_cache(
            splat_fixture, tmp_path / "s.gsplats.zarr", recipe="stream", n_lods=2
        )
        scene = build_interop_scene(
            cache,
            tmp_path / "scene.luxar.zarr",
            title="t",
            layer_name="lizard",
            credit="c",
        )
        assert _find_gsplats_blending_mode(scene, "lizard") == "normal"


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

    def test_clusterfly_member_and_license(self) -> None:
        m = _load("demo_gsplats_interop_macro_clusterfly")
        # Range-extracts the L-level PLY member from the GitHub release zip.
        assert m.MEMBER == "cluster fly L.ply"
        assert m.MEMBER_SIZE == 71_263_622
        assert m.ARCHIVE_URL.startswith(
            "https://github.com/danybittel/splats/releases/download"
        )
        # CC BY 4.0 attribution must be surfaced in the in-scene credit
        # OVERLAY specifically (the artifact-level credit baked into the
        # scene) — not merely somewhere in the source, which the console-only
        # provenance print would also satisfy. Assert on the CREDIT constant
        # that is actually passed to build_interop_scene.
        assert "CC BY 4.0" in m.CREDIT and "danybittel" in m.CREDIT
        # And that CREDIT is the value wired into the scene builder.
        import inspect

        assert "credit=CREDIT" in inspect.getsource(m.build_scene)

    def test_sog_matrixcity_bundle_and_files(self) -> None:
        m = _load("demo_gsplats_interop_sog_matrixcity")
        assert m.BASE_URL.startswith("https://") and "ace6e5b0" in m.BASE_URL
        # meta.json + the 5 DC-group files; the shN palette images are NOT
        # fetched (Luxar bakes DC only), so the bundle stays lean.
        assert m.SOG_FILES[0] == "meta.json"
        assert "sh0.webp" in m.SOG_FILES
        assert not any("shN" in f for f in m.SOG_FILES)
        # It must build the large-scale hierarchy via the overview recipe.
        import inspect

        assert 'recipe="overview"' in inspect.getsource(m.build_scene)

    def test_unknown_scene_raises(self) -> None:
        m = _load("demo_gsplats_interop_mipnerf_garden")
        with pytest.raises(ValueError, match="Unknown scene"):
            m.build_scene("not-a-scene")
