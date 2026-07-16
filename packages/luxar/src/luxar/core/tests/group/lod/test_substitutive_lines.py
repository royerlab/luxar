"""Tests for substitutive-LOD on Lines (coarse levels = synthesised gsplats).

Mirror of ``test_substitutive_points.py``: each segment is lifted to isotropic
"bead" gaussians, reduced by the gsplat substitutive pipeline, and assembled as
a ``kind=lod`` Group whose finest child is the original Lines node.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group.lod.lines import resolve_substitutive_axis_lines
from luxar.gsplats.lift import (
    coarse_substitutive_levels,
    lift_lines_to_gsplats,
    render_light,
)
from luxar.io.compiler import LuxarZarrCompiler


def _segments(n_seg, seed=0):
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(0, 60, (n_seg, 3))
    d = rng.normal(0, 5, (n_seg, 3))
    verts = np.empty((n_seg * 2, 3), np.float32)
    verts[0::2] = p0
    verts[1::2] = p0 + d
    return verts


def _build(tmp_path, *, n_seg=1500, line_type="segments", levels=2, widths=0.8, **kw):
    out = tmp_path / "t.luxar.zarr"
    verts = _segments(n_seg)
    colors = (
        np.random.default_rng(1).uniform(0, 1, (verts.shape[0], 3)).astype(np.float32)
    )
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_lines(
            "curves",
            verts,
            widths,
            colors=colors,
            line_type=line_type,
            substitutive_lod=dict(levels=levels, device="cpu", seed=0, **kw),
        )
    return zarr.open(str(out), mode="r")["curves"], verts.shape[0]


class TestResolveSubstitutiveAxisLines:
    def test_shares_points_vocabulary(self) -> None:
        assert resolve_substitutive_axis_lines(None) is None
        assert resolve_substitutive_axis_lines(False) is None
        r = resolve_substitutive_axis_lines(True)
        assert (
            r["compression_factor"] == 4 and r["levels"] == 3 and r["method"] == "auto"
        )

    def test_max_aspect_key_resolved(self) -> None:
        assert resolve_substitutive_axis_lines(True)["max_aspect"] == 3.0
        assert resolve_substitutive_axis_lines(dict(max_aspect=5))["max_aspect"] == 5.0
        assert resolve_substitutive_axis_lines(dict(max_aspect=None))["max_aspect"] is (
            None
        )
        with pytest.raises(ValueError, match="max_aspect"):
            resolve_substitutive_axis_lines(dict(max_aspect=0.5))
        assert (
            resolve_substitutive_axis_lines(dict(K=8, n_lods=2))["compression_factor"]
            == 8
        )

    def test_unknown_key_mentions_lines(self) -> None:
        with pytest.raises(ValueError, match="Lines"):
            resolve_substitutive_axis_lines(dict(bogus=1))

    def test_non_ascending_coverage_fractions_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 0.5, 0.1]))

    def test_coverage_fractions_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 2.0]))


class TestAddLinesSubstitutiveLod:
    def test_group_is_kind_lod_lines(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        assert grp.attrs["selector"] == "coverage"
        assert grp.attrs["default_level"] == 0

    def test_finest_is_lines_coarse_are_gsplats(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=2)
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert children == ["child_0", "child_1", "child_2"]  # 2 gsplat + 1 lines
        types = [grp[c].attrs["type"] for c in children]
        assert types[:2] == ["gsplats", "gsplats"]
        assert types[2] == "lines"

    def test_coverage_fraction_monotone_coarsest_zero(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=2)
        cf = [float(grp[f"child_{i}"].attrs["coverage_fraction"]) for i in range(3)]
        assert cf[0] == 0.0
        assert cf[-1] == 1.0
        assert all(cf[i] < cf[i + 1] for i in range(len(cf) - 1))

    def test_finest_carries_all_vertices(self, tmp_path) -> None:
        grp, n_verts = _build(tmp_path, levels=2)
        finest = grp["child_2"]
        assert finest.attrs["type"] == "lines"
        assert finest.attrs["n_vertices"] == n_verts

    @pytest.mark.parametrize("line_type", ["segments", "polyline", "loop"])
    def test_line_types_build_a_group(self, tmp_path, line_type) -> None:
        grp, _ = _build(tmp_path, n_seg=400, line_type=line_type, levels=2)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"

    def test_position_bounds_backfilled(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert "position_bounds" in grp.attrs


class TestSubstitutiveLinesGuards:
    def test_mutually_exclusive_with_additive(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="mutually exclusive"):
                scene.add_lines(
                    "c",
                    verts,
                    1.0,
                    line_type="segments",
                    additive_lod=True,
                    substitutive_lod=True,
                )

    def test_mutually_exclusive_with_partition(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="partition.*substitutive_lod"):
                scene.add_lines(
                    "c",
                    verts,
                    1.0,
                    line_type="segments",
                    partition=True,
                    substitutive_lod=True,
                )

    def test_scalars_plus_colormap_bakes_colors(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(1500)
        scalars = (
            np.random.default_rng(0).uniform(0, 1, verts.shape[0]).astype(np.float32)
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "c",
                verts,
                0.8,
                scalars=scalars,
                colormap="viridis",
                line_type="segments",
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["c"]
        assert grp.attrs["kind"] == "lod" and grp.attrs["display_type"] == "lines"
        assert grp["child_0"].attrs["type"] == "gsplats"
        assert bool(grp["child_0"].attrs.get("has_colors")) is True
        finest = grp["child_2"]
        assert bool(finest.attrs.get("has_scalars")) is True
        assert finest.attrs.get("colormap") == "viridis"

    def test_scalars_without_colormap_raises(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(50)
        scalars = (
            np.random.default_rng(0).uniform(0, 1, verts.shape[0]).astype(np.float32)
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="colormap"):
                scene.add_lines(
                    "c",
                    verts,
                    1.0,
                    scalars=scalars,
                    line_type="segments",
                    substitutive_lod=True,
                )

    def test_uniform_scalar_plus_colormap_works(self, tmp_path) -> None:
        # A scalar-valued (uniform) `scalars` must broadcast, not crash.
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(1500)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "c",
                verts,
                0.8,
                scalars=0.5,
                colormap="viridis",
                line_type="segments",
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["c"]
        assert grp.attrs["kind"] == "lod"

    def test_edgeless_input_delegates_to_flat_not_substitutive_crash(
        self, tmp_path
    ) -> None:
        # A 1-vertex polyline is genuinely invalid (a flat add_lines rejects it
        # too). The substitutive builder must DELEGATE to the flat path and raise
        # the SAME normal validation error — not a cryptic substitutive-internal
        # error (e.g. 'coarsest child must have >=1 element') from running the
        # pipeline on a 0-bead lift.
        out = tmp_path / "t.luxar.zarr"
        verts = np.array([[1.0, 2.0, 3.0]], np.float32)  # 1 vertex -> 0 segments
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="at least 2 vertices"):
                scene.add_lines(
                    "c",
                    verts,
                    1.0,
                    line_type="polyline",
                    substitutive_lod=dict(levels=2, device="cpu"),
                )

    def test_finest_count_uses_bead_currency(self, tmp_path) -> None:
        # The coverage fractions normalise by the lifted BEAD count, not
        # n_vertices. Because the bead count exceeds n_vertices, the intermediate
        # coarse gsplat child gets a SMALLER fraction than a (wrong) vertex-currency
        # ladder would produce — keeping every fraction in [0, 1] with the lines
        # node anchored at the 1.0 top.
        from luxar.core.group.lod.group import coverage_fractions

        grp, n_verts = _build(tmp_path, levels=2)
        n = len(sorted(k for k in grp.keys() if k.startswith("child_")))
        coarse_counts = [int(grp[f"child_{i}"].attrs["n_splats"]) for i in range(n - 1)]
        # Intermediate coarse gsplat child (index 1): its actual bead-currency fraction.
        actual = float(grp["child_1"].attrs["coverage_fraction"])
        # An n_vertices-based ladder (the wrong currency) would give a LARGER fraction.
        wrong = coverage_fractions(coarse_counts + [n_verts])[1]
        assert actual < wrong
        assert float(grp[f"child_{n - 1}"].attrs["coverage_fraction"]) == 1.0


class TestSubstitutiveLinesConservationAndSymmetry:
    """Lines mirror of the Points conservation/symmetry guarantees."""

    def test_render_light_conserved_across_coarse_levels(self) -> None:
        # The headline guarantee, mirrored for Lines: zooming out must not dim.
        # ``coarse_substitutive_levels`` rescales each coarse level so its
        # render-light (sum a * sigma_geo^3) equals the finest (lifted bead)
        # level's. Tested in-memory to avoid the writer's amplitude quantisation.
        verts = _segments(2000, seed=3)
        lifted = lift_lines_to_gsplats(
            verts, 0.8, line_type="segments", truncation_radius=3.0
        )
        target = render_light(lifted)
        coarse = coarse_substitutive_levels(
            lifted, compression_factor=4, levels=3, device="cpu", seed=0
        )
        assert len(coarse) == 3
        for lvl in coarse:
            assert render_light(lvl) == pytest.approx(target, rel=1e-4)

    @staticmethod
    def _aspects(lvl) -> np.ndarray:
        from luxar.gsplats.utils.trils import unpack_tril

        L = unpack_tril(np.asarray(lvl.cholesky_factors, np.float64), int(lvl.ndim))
        ev = np.linalg.eigvalsh(L @ np.swapaxes(L, 1, 2))
        return np.sqrt(ev[:, -1] / np.maximum(ev[:, 0], 1e-30))

    def test_coarse_level_aspect_capped(self) -> None:
        # THE brightness/hue-pop fix: the merge of a 1D bead string elongates
        # representatives level over level (aspect ~1.9/6.6/25 uncapped), and an
        # elongated gaussian's ray integral flares ~aspect x when viewed end-on
        # — per-splat, per-orientation, per-level flares = haphazard pops. The
        # default max_aspect=3 bounds every coarse splat's aspect.
        verts = _segments(2000, seed=3)
        lifted = lift_lines_to_gsplats(verts, 0.8, line_type="segments")
        coarse = coarse_substitutive_levels(
            lifted, compression_factor=4, levels=3, device="cpu", seed=0
        )
        for lvl in coarse:
            assert float(self._aspects(lvl).max()) <= 3.0 * (1 + 1e-4)

    def test_max_aspect_none_disables_cap(self) -> None:
        # Guard that the knob is live: uncapped coarse levels of a bead string
        # DO exceed aspect 3 (otherwise the capped test above proves nothing).
        verts = _segments(2000, seed=3)
        lifted = lift_lines_to_gsplats(verts, 0.8, line_type="segments")
        coarse = coarse_substitutive_levels(
            lifted,
            compression_factor=4,
            levels=3,
            device="cpu",
            seed=0,
            max_aspect=None,
        )
        assert float(self._aspects(coarse[-1]).max()) > 3.0

    def test_colored_light_per_channel_conserved_across_levels(self) -> None:
        # Hue coherence: per-channel colored light (sum a*|det L|*c_ch) must
        # match the lifted level's on EVERY coarse level. Exact per bin with
        # amplitude="mass" (mass-weighted mean colors x mass-preserving
        # amplitudes); the aspect cap and render-light rescale preserve it.
        verts = _segments(2000, seed=3)
        rng = np.random.default_rng(7)
        colors = rng.uniform(0.05, 1.0, (verts.shape[0], 3)).astype(np.float32)
        lifted = lift_lines_to_gsplats(verts, 0.8, line_type="segments", colors=colors)
        from luxar.gsplats.utils.trils import unpack_tril

        def colored_light(d):
            flat = d.flattened()
            L = unpack_tril(
                np.asarray(flat.cholesky_factors, np.float64), int(flat.ndim)
            )
            det = np.abs(np.linalg.det(L))
            a = np.asarray(flat.amplitudes, np.float64)
            c = np.asarray(flat.colors, np.float64)
            return (a * det) @ c

        target = colored_light(lifted)
        coarse = coarse_substitutive_levels(
            lifted, compression_factor=4, levels=3, device="cpu", seed=0
        )
        for lvl in coarse:
            np.testing.assert_allclose(colored_light(lvl), target, rtol=1e-4)

    def test_opacity_rides_on_group_not_baked_into_amplitudes(self, tmp_path) -> None:
        # opacity is a compositing attr on the kind=lod group; the lift always uses
        # opacity=1, so the baked gsplat amplitudes are INDEPENDENT of node opacity
        # (applied once at composite, never twice).
        def build(op):
            out = tmp_path / f"op{op}.luxar.zarr"
            verts = _segments(1500)
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "c",
                    verts,
                    0.8,
                    line_type="segments",
                    opacity=op,
                    substitutive_lod=dict(levels=2, device="cpu", seed=0),
                )
            return zarr.open(str(out), mode="r")["c"]

        g_half, g_full = build(0.5), build(1.0)
        assert float(g_half.attrs["opacity"]) == pytest.approx(0.5)
        a_half = np.asarray(g_half["child_0"]["amplitudes"])
        a_full = np.asarray(g_full["child_0"]["amplitudes"])
        np.testing.assert_array_equal(a_half, a_full)
        assert g_half["child_0"].attrs["amplitude_range"]["max"] == pytest.approx(
            g_full["child_0"].attrs["amplitude_range"]["max"]
        )

    def test_explicit_coverage_fractions_override(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=2, coverage_fractions=[0.0, 0.25, 1.0])
        cf = [float(grp[f"child_{i}"].attrs["coverage_fraction"]) for i in range(3)]
        assert cf == [0.0, 0.25, 1.0]

    def test_uint8_colors_render_sdr_on_coarse_levels(self, tmp_path) -> None:
        # Lines mirror of the CRITICAL points regression: uint8 line colours must
        # NOT become HDR on the coarse gsplat levels (~255x too bright vs the SDR
        # lines child at the LOD seam). The coarse child colours stay SDR (uint8)
        # and no HDR warning fires.
        import warnings

        out = tmp_path / "t.luxar.zarr"
        verts = _segments(1500)
        rng = np.random.default_rng(0)
        colors_u8 = rng.integers(0, 256, (verts.shape[0], 3)).astype(np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "c",
                    verts,
                    0.8,
                    colors=colors_u8,
                    line_type="segments",
                    substitutive_lod=dict(levels=2, device="cpu", seed=0),
                )
            hdr = [w for w in caught if "HDR" in str(w.message)]
        grp = zarr.open(str(out), mode="r")["c"]
        assert np.asarray(grp["child_0"]["colors"]).dtype == np.uint8
        assert not hdr, (
            f"unexpected HDR colour warning(s): {[str(w.message) for w in hdr]}"
        )

    def test_image_labels_forwarded_to_finest_lines_child(self, tmp_path) -> None:
        # image_labels must NOT be dropped on the substitutive path; they ride to
        # the finest Lines child (the real line node).
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(1500)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "c",
                verts,
                0.8,
                line_type="segments",
                image_labels=[b"x"] * verts.shape[0],
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["c"]
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        finest = grp[children[-1]]
        assert finest.attrs["type"] == "lines"
        assert finest.attrs.get("has_image_labels") is True

    def test_tiny_input_builds_valid_group_without_crashing(self, tmp_path) -> None:
        # A handful of valid segments still reduces; the builder must produce a
        # valid kind=lod group with the lines node finest (not crash, not flat).
        out = tmp_path / "t.luxar.zarr"
        verts = _segments(6)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "c",
                verts,
                0.8,
                line_type="segments",
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["c"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert grp[children[-1]].attrs["type"] == "lines"


# ────────────────────────────────────────────────────────────────────────
# coarsen_dims — barrier-aware coarsening (parallels the Points tests)
# ────────────────────────────────────────────────────────────────────────


class TestCoarsenDimsLines:
    def _build_4d(self, tmp_path, *, coarsen_dims="__unset__", n_groups=3):
        # Segments stacked at categorical (display=False) coloring values 0..G-1,
        # sharing the same xyz so a barrier-unaware coarsening would blend them.
        rng = np.random.default_rng(0)
        n_seg = 1200
        xyz = rng.normal(0, 5, (2 * n_seg, 3)).astype(np.float32)
        parts = [
            np.column_stack([np.full(2 * n_seg, g, np.float32), xyz])
            for g in range(n_groups)
        ]
        verts = np.vstack(parts).astype(np.float32)
        dims = Dimensions(
            [
                Dimension(
                    "coloring",
                    categories=[str(g) for g in range(n_groups)],
                    display=False,
                ),
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
            ]
        )
        out = tmp_path / "l4d.luxar.zarr"
        kw = {} if coarsen_dims == "__unset__" else {"coarsen_dims": coarsen_dims}
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_lines(
                "curves",
                verts,
                0.8,
                line_type="segments",
                substitutive_lod=dict(levels=3, device="cpu", **kw),
            )
        return zarr.open(str(out), mode="r")["curves"]

    @staticmethod
    def _purity(grp) -> float:
        worst = 0.0
        for k in grp.keys():
            if k.startswith("child_") and grp[k].attrs.get("type") == "gsplats":
                from luxar.encoding import ArrayDecoder

                c0 = ArrayDecoder().decode(grp[k]["centers"], grp)[:, 0]
                worst = max(worst, float(np.abs(c0 - np.round(c0)).max()))
        return worst

    def test_auto_default_groups_by_non_displayed(self, tmp_path) -> None:
        assert self._purity(self._build_4d(tmp_path)) < 1e-4

    def test_all_dims_blends(self, tmp_path) -> None:
        grp = self._build_4d(tmp_path, n_groups=4, coarsen_dims="all")
        assert self._purity(grp) > 0.05

    def test_resolver_passthrough(self) -> None:
        assert resolve_substitutive_axis_lines(dict(coarsen_dims=["x", "y", "z"]))[
            "coarsen_dims"
        ] == ["x", "y", "z"]
