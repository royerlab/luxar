"""Tests for substitutive-LOD on Points (coarse levels = synthesised gsplats).

Covers:

- The resolver ``resolve_substitutive_axis_points`` (value vocabulary).
- End-to-end ``add_points(..., substitutive_lod=...)`` → a ``kind=lod`` Group
  whose finest child is the original Points node and whose coarser children are
  synthesised GSplats (round-tripped through the writer).
- The render-light conservation guarantee (no zoom-out dimming).
- Mutual exclusion with ``additive_lod`` and the scalar+colormap limitation.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.points import resolve_substitutive_axis_points
from luxar.gsplats.lift import (
    coarse_substitutive_levels,
    lift_points_to_gsplats,
    render_light,
)
from luxar.io.compiler import LuxarZarrCompiler

# ────────────────────────────────────────────────────────────────────────
# Resolver
# ────────────────────────────────────────────────────────────────────────


class TestResolveSubstitutiveAxisPoints:
    def test_none_and_false_are_noops(self) -> None:
        assert resolve_substitutive_axis_points(None) is None
        assert resolve_substitutive_axis_points(False) is None

    def test_true_and_empty_dict_give_defaults(self) -> None:
        for spec in (True, {}):
            r = resolve_substitutive_axis_points(spec)
            assert r is not None
            assert r["compression_factor"] == 4
            assert r["levels"] == 3
            assert r["method"] == "auto"
            assert r["truncation_radius"] == 3.0

    def test_aliases_K_and_n_lods(self) -> None:
        r = resolve_substitutive_axis_points(dict(K=8, n_lods=2))
        assert r["compression_factor"] == 8
        assert r["levels"] == 2

    def test_method_hyphen_normalized(self) -> None:
        r = resolve_substitutive_axis_points(dict(method="kmeans-lloyd"))
        assert r["method"] == "kmeans_lloyd"

    def test_explicit_min_pixel_sizes(self) -> None:
        r = resolve_substitutive_axis_points(dict(min_pixel_sizes=[0, 10, 20, 30]))
        assert r["min_pixel_sizes"] == [0.0, 10.0, 20.0, 30.0]

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized keys"):
            resolve_substitutive_axis_points(dict(bogus=1))

    def test_bad_values_raise(self) -> None:
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(compression_factor=1))
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(levels=0))
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(method="nope"))
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(truncation_radius=0))

    def test_non_dict_raises(self) -> None:
        with pytest.raises(TypeError):
            resolve_substitutive_axis_points(5)


# ────────────────────────────────────────────────────────────────────────
# End-to-end through the writer
# ────────────────────────────────────────────────────────────────────────


def _build(tmp_path, *, n=6000, levels=3, **kw):
    out = tmp_path / "t.luxar.zarr"
    rng = np.random.RandomState(0)
    pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    radii = rng.uniform(0.5, 1.5, n).astype(np.float32)
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            pos,
            colors=colors,
            radii=radii,
            substitutive_lod=dict(compression_factor=4, levels=levels, device="cpu",
                                  seed=0, **kw),
        )
    return zarr.open(str(out), mode="r")["cloud"], n


class TestAddPointsSubstitutiveLod:
    def test_group_is_kind_lod_points(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        assert grp.attrs["selector"] == "pixel_size"
        assert grp.attrs["default_level"] == 0

    def test_child_count_is_levels_gsplats_plus_points(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=3)
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        # 3 coarse gsplat levels (level 0 dropped) + 1 points node.
        assert children == ["child_0", "child_1", "child_2", "child_3"]

    def test_finest_is_points_coarse_are_gsplats(self, tmp_path) -> None:
        grp, n = _build(tmp_path, levels=3)
        types = [grp[f"child_{i}"].attrs["type"] for i in range(4)]
        assert types[:3] == ["gsplats", "gsplats", "gsplats"]
        assert types[3] == "points"
        # finest child carries the full cloud
        assert grp["child_3"].attrs["n_points"] == n

    def test_counts_increase_coarsest_to_finest(self, tmp_path) -> None:
        grp, n = _build(tmp_path, levels=3)
        counts = [
            grp[f"child_{i}"].attrs.get("n_splats")
            or grp[f"child_{i}"].attrs.get("n_points")
            for i in range(4)
        ]
        assert counts == sorted(counts)  # ascending coarsest -> finest
        assert counts[-1] == n

    def test_min_pixel_size_monotone_coarsest_zero(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=3)
        mps = [float(grp[f"child_{i}"].attrs["min_pixel_size"]) for i in range(4)]
        assert mps[0] == 0.0
        assert all(mps[i] < mps[i + 1] for i in range(len(mps) - 1))

    def test_position_bounds_backfilled(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert "position_bounds" in grp.attrs

    def test_render_light_conserved_across_coarse_levels(self) -> None:
        # The whole point: zooming out must not dim. ``coarse_substitutive_levels``
        # rescales each coarse level so its render-light (sum a * sigma_geo^3)
        # equals the finest (lifted) level's. Tested in-memory to avoid the
        # writer's uint16 amplitude quantisation.
        rng = np.random.RandomState(0)
        pos = rng.normal(0, 20, (6000, 3)).astype(np.float32)
        radii = rng.uniform(0.5, 1.5, 6000).astype(np.float32)
        lifted = lift_points_to_gsplats(pos, radii, colors=None, truncation_radius=3.0)
        target = render_light(lifted)
        coarse = coarse_substitutive_levels(
            lifted, compression_factor=4, levels=3, device="cpu", seed=0
        )
        assert len(coarse) == 3
        for lvl in coarse:
            assert render_light(lvl) == pytest.approx(target, rel=1e-4)

    def test_explicit_min_pixel_sizes_override(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=3, min_pixel_sizes=[0.0, 5.0, 25.0, 100.0])
        mps = [float(grp[f"child_{i}"].attrs["min_pixel_size"]) for i in range(4)]
        assert mps == [0.0, 5.0, 25.0, 100.0]


class TestSubstitutiveLodGuards:
    def test_mutually_exclusive_with_additive(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="mutually exclusive"):
                scene.add_points(
                    "pts", pos, additive_lod=True, substitutive_lod=True
                )

    def test_scalars_without_colors_raises(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        pos = rng.rand(200, 3).astype(np.float32)
        scalars = rng.rand(200).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(NotImplementedError, match="scalars"):
                scene.add_points(
                    "pts", pos, scalars=scalars, colormap="viridis",
                    substitutive_lod=True,
                )

    def test_tiny_input_builds_valid_group_without_crashing(self, tmp_path) -> None:
        # Very small N still reduces (each coarse level may be 1 splat); the
        # builder must produce a valid kind=lod group with the points node finest.
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).rand(5, 3).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", pos, radii=1.0,
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["pts"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert grp[children[-1]].attrs["type"] == "points"  # finest is points
