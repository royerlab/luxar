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

    def test_non_ascending_min_pixel_sizes_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_points(dict(min_pixel_sizes=[0.0, 50.0, 10.0]))

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

    def test_base_pixel_size(self) -> None:
        assert resolve_substitutive_axis_points(dict(base_pixel_size=25.0))[
            "base_pixel_size"
        ] == 25.0
        with pytest.raises(ValueError, match="base_pixel_size"):
            resolve_substitutive_axis_points(dict(base_pixel_size=0.0))


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

    def test_uint8_colors_render_sdr_on_coarse_levels(self, tmp_path) -> None:
        # CRITICAL regression: uint8 colors must NOT become HDR on the coarse
        # gsplat levels (which would render ~255x too bright vs the SDR points
        # child at the LOD seam). Post-fix the gsplat child colors are SDR (uint8
        # encoding) and no HDR warning fires.
        import warnings

        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        colors_u8 = rng.integers(0, 256, (6000, 3)).astype(np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("cloud", pos, colors=colors_u8, radii=1.0,
                                 substitutive_lod=dict(levels=2, device="cpu", seed=0))
            hdr = [w for w in caught if "HDR" in str(w.message)]
        grp = zarr.open(str(out), mode="r")["cloud"]
        # child_0 is a coarse gsplat level; its colors must be SDR (uint8), not
        # the blown-out float32-HDR the pre-fix lift produced.
        assert np.asarray(grp["child_0"]["colors"]).dtype == np.uint8
        assert len(hdr) == 0, f"unexpected HDR colour warning(s): {[str(w.message) for w in hdr]}"

    def test_opacity_rides_on_group_not_baked_into_amplitudes(self, tmp_path) -> None:
        # opacity is a compositing attr on the kind=lod group; the lift always uses
        # opacity=1, so the baked gsplat amplitudes are INDEPENDENT of the node
        # opacity (it is applied once, at composite, to all children — not twice).
        def build(op):
            out = tmp_path / f"op{op}.luxar.zarr"
            rng = np.random.default_rng(0)
            pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points("cloud", pos, radii=1.0, opacity=op,
                                 substitutive_lod=dict(levels=2, device="cpu", seed=0))
            return zarr.open(str(out), mode="r")["cloud"]

        g_half, g_full = build(0.5), build(1.0)
        assert float(g_half.attrs["opacity"]) == pytest.approx(0.5)
        assert float(g_full.attrs["opacity"]) == pytest.approx(1.0)
        # Amplitudes (and their stored range) must be identical regardless of opacity.
        a_half = np.asarray(g_half["child_0"]["amplitudes"])
        a_full = np.asarray(g_full["child_0"]["amplitudes"])
        np.testing.assert_array_equal(a_half, a_full)
        assert (g_half["child_0"].attrs["amplitude_range"]["max"]
                == pytest.approx(g_full["child_0"].attrs["amplitude_range"]["max"]))

    def test_render_light_survives_writer_quantization(self, tmp_path) -> None:
        # End-to-end: the render-light rescale must survive the writer's uint16
        # amplitude quantization (each level stores its own amplitude_range, so a
        # per-level scalar rescale is preserved through encode+decode).
        from luxar.gsplats.utils.trils import unpack_tril

        grp, _ = _build(tmp_path, levels=3)
        lights = []
        for i in range(3):  # gsplat children
            child = grp[f"child_{i}"]
            u_raw = np.asarray(child["amplitudes"])
            amax = float(child.attrs["amplitude_range"]["max"])
            # The encoder adaptively picks uint8 OR uint16 per node; dequantize
            # against the stored array's own dtype max (writer: u = a/max*dtype_max).
            qmax = float(np.iinfo(u_raw.dtype).max)
            a = u_raw.astype(np.float64) / qmax * amax
            L = unpack_tril(np.asarray(child["cholesky_factors"], dtype=np.float64), 3)
            det = np.abs(np.linalg.det(L))
            lights.append(float(np.sum(a * det)))
        for li in lights[1:]:
            assert li == pytest.approx(lights[0], rel=0.03)


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

    def test_partition_and_substitutive_raises(self, tmp_path) -> None:
        # Must not silently drop the substitutive ladder when partition= is set.
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="partition.*substitutive_lod"):
                scene.add_points("pts", pos, partition=True, substitutive_lod=True)

    def test_substitutive_takes_precedence_over_auto_partition(self, tmp_path) -> None:
        # With compiler auto-partition enabled, substitutive_lod must still win
        # (build the LOD group, not silently get auto-partitioned away).
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out, auto_partition_max_elements=2000) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("cloud", pos, radii=1.0,
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["cloud"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"

    def test_scalars_without_colormap_raises(self, tmp_path) -> None:
        # scalars need a colormap to map to colour — without one, raise clearly.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        pos = rng.rand(200, 3).astype(np.float32)
        scalars = rng.rand(200).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="colormap"):
                scene.add_points("pts", pos, scalars=scalars, substitutive_lod=True)

    def test_uniform_scalar_plus_colormap_broadcasts(self, tmp_path) -> None:
        # A scalar-valued (uniform) `scalars` must broadcast to n_points, not crash
        # with a 'Colors count 1 doesn't match centers count N' error.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("cloud", pos, scalars=0.5, colormap="viridis",
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["cloud"]
        assert grp.attrs["kind"] == "lod"

    def test_scalars_plus_colormap_bakes_colors_on_coarse_levels(self, tmp_path) -> None:
        # scalars+colormap now WORKS: coarse gsplat levels carry baked SDR colours
        # from the colormap; the finest Points child keeps scalars+colormap.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        scalars = rng.uniform(0, 1, 6000).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("cloud", pos, scalars=scalars, colormap="viridis",
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["cloud"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        # Coarse gsplat children carry baked colours (SDR, not blown-out HDR).
        c0 = grp[children[0]]
        assert c0.attrs["type"] == "gsplats"
        assert bool(c0.attrs.get("has_colors")) is True
        # Finest child stays scalar-driven + colormapped.
        finest = grp[children[-1]]
        assert finest.attrs["type"] == "points"
        assert bool(finest.attrs.get("has_scalars")) is True
        assert finest.attrs.get("colormap") == "viridis"

    def test_image_labels_forwarded_to_finest_points_child(self, tmp_path) -> None:
        # image_labels must NOT be silently dropped on the substitutive path; they
        # ride to the finest Points child (the real N-point node).
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (3000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("cloud", pos, radii=1.0, image_labels=[b"x"] * 3000,
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["cloud"]
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        finest = grp[children[-1]]
        assert finest.attrs["type"] == "points"
        assert finest.attrs.get("has_image_labels") is True

    def test_empty_cloud_falls_through_to_canonical_path(self, tmp_path) -> None:
        # N=0 must route through the canonical path (the n_points>0 guard), which
        # rejects empty input with its normal "empty points" error — NOT a
        # confusing substitutive-internal 'coarsest child must have >=1' error.
        out = tmp_path / "t.luxar.zarr"
        empty = np.zeros((0, 3), np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="empty points") as exc:
                scene.add_points("pts", empty, radii=1.0,
                                 substitutive_lod=dict(levels=2, device="cpu"))
            assert "coarsest child" not in str(exc.value)

    def test_all_zero_radius_falls_through_to_flat_points(self, tmp_path) -> None:
        # Regression: an all-zero-radius cloud lifts to 0 splats, so every coarse
        # level is empty (coarse[-1].n_splats == 0). The degenerate guard must
        # route this to a flat Points node — pre-fix, `if not coarse:` missed the
        # reachable empty-coarsest case and crashed building a 0-element gsplat
        # child ('coarsest child must have at least 1 element, got 0').
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).normal(0, 20, (4000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points("pts", pos, radii=0.0,  # every point degenerate
                             substitutive_lod=dict(levels=2, device="cpu", seed=0))
        grp = zarr.open(str(out), mode="r")["pts"]
        assert grp.attrs.get("kind") != "lod"  # flat fallback, not a 1-child LOD
        assert grp.attrs.get("type") == "points"

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
