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

import warnings

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group.lod.group import (
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
    partitioned_coverage_fractions,
)
from luxar.core.group.lod.points import resolve_substitutive_axis_points
from luxar.gsplats.lift import (
    LIFT_TRUNCATION_RADIUS,
    coarse_point_levels,
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
            # The resolver's default feeds the point/line lift, whose T is a
            # profile-matching parameter (3.0) — deliberately NOT the render
            # default DEFAULT_TRUNCATION_RADIUS (2.75). See lift.py.
            assert r["truncation_radius"] == LIFT_TRUNCATION_RADIUS

    def test_aliases_K_and_n_lods(self) -> None:
        r = resolve_substitutive_axis_points(dict(K=8, n_lods=2))
        assert r["compression_factor"] == 8
        assert r["levels"] == 2

    def test_points_coarse_subsample_vocabulary(self) -> None:
        r = resolve_substitutive_axis_points(
            dict(coarse="points", brightness_compensation=2.5)
        )
        assert r["coarse"] == "points"
        assert r["brightness_compensation"] == 2.5

    @pytest.mark.parametrize(
        "key", ["truncation_radius", "max_aspect", "method", "device", "coarsen_dims"]
    )
    def test_points_coarse_refuses_lift_only_keys(self, key: str) -> None:
        with pytest.raises(ValueError, match=rf"{key!r} does not apply"):
            resolve_substitutive_axis_points(dict(coarse="points", **{key: 2.0}))

    @pytest.mark.parametrize("value", [None, "nonsense"])
    def test_bad_brightness_compensation_has_stable_error(self, value) -> None:
        with pytest.raises(ValueError, match="must be 'auto' or a finite value"):
            resolve_substitutive_axis_points(
                dict(coarse="points", brightness_compensation=value)
            )

    def test_unrecognized_key_lists_points_only_keys(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            resolve_substitutive_axis_points(dict(coarse="points", coarse_mode="x"))
        message = str(exc_info.value)
        assert "coarse" in message
        assert "brightness_compensation" in message

    def test_max_aspect_key_resolved(self) -> None:
        # Mirror of the Lines resolver test (shared implementation, but the
        # per-geometry wrapper must expose the key identically).
        assert resolve_substitutive_axis_points(True)["max_aspect"] == 3.0
        assert resolve_substitutive_axis_points(dict(max_aspect=5))["max_aspect"] == 5.0
        assert (
            resolve_substitutive_axis_points(dict(max_aspect=None))["max_aspect"]
            is None
        )
        with pytest.raises(ValueError, match="max_aspect"):
            resolve_substitutive_axis_points(dict(max_aspect=0.5))

    def test_max_aspect_threaded_spec_to_lift(self, tmp_path, monkeypatch) -> None:
        # Mirror of the Lines spy test: the Points adder must forward the
        # USER'S max_aspect to coarse_substitutive_levels (the signature
        # default masks a dropped forward from output-based tests).
        import luxar.gsplats.lift as lift_mod

        seen: list = []
        real = lift_mod.coarse_substitutive_levels

        def spy(*args, **kwargs):
            seen.append(kwargs.get("max_aspect"))
            return real(*args, **kwargs)

        monkeypatch.setattr(lift_mod, "coarse_substitutive_levels", spy)
        out = tmp_path / "spy.luxar.zarr"
        rng = np.random.default_rng(0)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                rng.uniform(0, 40, (400, 3)).astype(np.float32),
                radii=0.5,
                substitutive_lod=dict(levels=1, device="cpu", seed=0, max_aspect=5),
            )
        assert seen == [5.0]

    def test_method_hyphen_normalized(self) -> None:
        r = resolve_substitutive_axis_points(dict(method="kmeans-lloyd"))
        assert r["method"] == "kmeans_lloyd"

    def test_explicit_coverage_fractions(self) -> None:
        r = resolve_substitutive_axis_points(
            dict(coverage_fractions=[0, 0.25, 0.5, 1.0])
        )
        assert r["coverage_fractions"] == [0.0, 0.25, 0.5, 1.0]

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unrecognized keys"):
            resolve_substitutive_axis_points(dict(bogus=1))

    def test_non_ascending_coverage_fractions_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_points(dict(coverage_fractions=[0.0, 0.5, 0.1]))

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

    def test_coverage_fractions_out_of_range_raises(self) -> None:
        # The ceiling is MAX_COVERAGE_FRACTION == SCREEN_FILL_DIAGONAL_RATIO /
        # FILL_FACTOR == 4.0 (roughly the metric
        # a screen-filling object produces), not 1.0.
        with pytest.raises(ValueError, match=r"\[0, 4\]"):
            resolve_substitutive_axis_points(dict(coverage_fractions=[0.0, 4.5]))

    def test_coverage_fractions_above_one_accepted(self) -> None:
        # Above the auto-derived 1.0 anchor but within the ceiling (inclusive) —
        # the escape hatch for a level that must hold until the object is LARGER
        # than half the fitted screen axis (e.g. a spatially tiled layer).
        r = resolve_substitutive_axis_points(dict(coverage_fractions=[0.0, 1.5]))
        assert r["coverage_fractions"] == [0.0, 1.5]
        r = resolve_substitutive_axis_points(dict(coverage_fractions=[0.0, 4.0]))
        assert r["coverage_fractions"] == [0.0, 4.0]

    def test_empty_coverage_fractions_raises_clean_error(self) -> None:
        # An empty explicit list must raise an actionable ValueError, NOT an
        # IndexError from the [0]/[-1] range check (regression: deep-double-check).
        with pytest.raises(ValueError, match="non-empty"):
            resolve_substitutive_axis_points(dict(coverage_fractions=[]))


# ────────────────────────────────────────────────────────────────────────
# End-to-end through the writer
# ────────────────────────────────────────────────────────────────────────


def _build(tmp_path, *, n=6000, levels=3, radius_scale=1.0, **kw):
    out = tmp_path / "t.luxar.zarr"
    rng = np.random.RandomState(0)
    pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
    colors = rng.uniform(0, 1, (n, 3)).astype(np.float32)
    radii = rng.uniform(0.5, 1.5, n).astype(np.float32) * radius_scale
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_points(
            "cloud",
            pos,
            colors=colors,
            radii=radii,
            substitutive_lod=dict(
                compression_factor=4, levels=levels, device="cpu", seed=0, **kw
            ),
        )
    return zarr.open(str(out), mode="r")["cloud"], n


class TestAddPointsSubstitutiveLod:
    def test_points_coarse_levels_stay_points_and_compensate_hdr(
        self, tmp_path
    ) -> None:
        from luxar.encoding import ArrayDecoder

        out = tmp_path / "points-coarse.luxar.zarr"
        positions = np.stack(np.meshgrid(np.arange(4), np.arange(4), np.arange(4)), -1)
        positions = positions.reshape(-1, 3).astype(np.float32)
        colors = np.tile(np.array([[64, 128, 255]], dtype=np.uint8), (64, 1))
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                colors=colors,
                substitutive_lod=dict(
                    coarse="points", compression_factor=4, levels=2, seed=7
                ),
            )

        group = zarr.open(str(out), mode="r")["cloud"]
        assert [group[f"child_{i}"].attrs["type"] for i in range(3)] == [
            "points",
            "points",
            "points",
        ]
        assert [group[f"child_{i}"].attrs["n_points"] for i in range(3)] == [
            4,
            16,
            64,
        ]
        decoder = ArrayDecoder()
        expected = colors[0].astype(np.float32) / 255.0
        for child_index, gain in ((0, 16.0), (1, 4.0)):
            child = group[f"child_{child_index}"]
            decoded = decoder.decode(child["colors"], group)
            assert decoded.dtype == np.float32
            np.testing.assert_allclose(
                decoded, np.broadcast_to(expected * gain, decoded.shape), rtol=1e-5
            )

    def test_points_coarse_auto_does_not_brighten_normal_blending(
        self, tmp_path
    ) -> None:
        from luxar.encoding import ArrayDecoder

        out = tmp_path / "points-normal.luxar.zarr"
        positions = np.stack(np.meshgrid(np.arange(4), np.arange(4), np.arange(4)), -1)
        positions = positions.reshape(-1, 3).astype(np.float32)
        colors = np.full((64, 3), 0.25, dtype=np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                colors=colors,
                blending_mode="normal",
                substitutive_lod=dict(
                    coarse="points", compression_factor=4, levels=1, seed=7
                ),
            )

        group = zarr.open(str(out), mode="r")["cloud"]
        decoded = ArrayDecoder().decode(group["child_0"]["colors"], group)
        np.testing.assert_allclose(decoded, 0.25, rtol=1e-5)

    def test_points_coarse_inherits_normal_blending_without_widening(
        self, tmp_path
    ) -> None:
        from luxar.encoding import ArrayDecoder

        out = tmp_path / "points-inherited-normal.luxar.zarr"
        positions = np.stack(np.meshgrid(np.arange(4), np.arange(4), np.arange(4)), -1)
        positions = positions.reshape(-1, 3).astype(np.float32)
        colors = np.full((64, 3), 64, dtype=np.uint8)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            parent = scene.add_group("parent", blending_mode="normal")
            parent.add_points(
                "cloud",
                positions,
                colors=colors,
                substitutive_lod=dict(coarse="points", compression_factor=4, levels=1),
            )

        group = zarr.open(str(out), mode="r")["parent/cloud"]
        decoded = ArrayDecoder().decode(group["child_0"]["colors"], group)
        assert decoded.dtype == np.uint8
        np.testing.assert_array_equal(decoded, 64)

    def test_points_coarse_accepts_uniform_tuple_color(self, tmp_path) -> None:
        from luxar.encoding import ArrayDecoder

        out = tmp_path / "points-broadcast.luxar.zarr"
        positions = np.random.default_rng(0).uniform(0, 1, (64, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                colors=(1.0, 0.0, 0.0),
                substitutive_lod=dict(coarse="points", levels=1),
            )
        group = zarr.open(str(out), mode="r")["cloud"]
        decoded = ArrayDecoder().decode(group["child_0"]["colors"], group)
        np.testing.assert_allclose(decoded[:, 0], 4.0)
        np.testing.assert_allclose(decoded[:, 1:], 0.0)

    def test_points_coarse_keeps_every_hidden_slice(self, tmp_path) -> None:
        out = tmp_path / "points-slices.luxar.zarr"
        n_slices = 40
        points_per_slice = 32
        rng = np.random.default_rng(0)
        positions = np.column_stack(
            [
                np.repeat(np.arange(n_slices), points_per_slice),
                rng.uniform(0, 1, (n_slices * points_per_slice, 3)),
            ]
        ).astype(np.float32)
        dims = Dimensions(
            [
                Dimension(
                    "time", categories=list(map(str, range(n_slices))), display=False
                ),
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
            ]
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "cloud",
                positions,
                substitutive_lod=dict(coarse="points", compression_factor=4, levels=2),
                additive_lod=False,
            )
        group = zarr.open(str(out), mode="r")["cloud"]
        from luxar.encoding import ArrayDecoder

        coarse_positions = ArrayDecoder().decode(
            group["child_0/positions"], group["child_0"]
        )
        assert set(coarse_positions[:, 0]) == set(range(n_slices))

    def test_points_coarse_refuses_more_hidden_slices_than_coarse_points(
        self, tmp_path
    ) -> None:
        positions = np.column_stack(
            [
                np.repeat(np.arange(20), 2),
                np.zeros((40, 3), dtype=np.float32),
            ]
        ).astype(np.float32)
        dims = Dimensions(
            [
                Dimension("time", categories=list(map(str, range(20))), display=False),
                Dimension("x", display=True),
                Dimension("y", display=True),
                Dimension("z", display=True),
            ]
        )
        with pytest.raises(ValueError, match="coarsest level has 10 points for 20"):
            with LuxarZarrCompiler(tmp_path / "starved.luxar.zarr") as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "cloud",
                    positions,
                    substitutive_lod=dict(
                        coarse="points", compression_factor=4, levels=1
                    ),
                )

    def test_points_coarse_sorted_sheet_remains_spatially_balanced(
        self, tmp_path
    ) -> None:
        from luxar.encoding import ArrayDecoder

        side = 224
        positions = np.column_stack(
            [
                np.repeat(np.linspace(0, 1, side), side),
                np.tile(np.linspace(0, 1, side), side),
                np.zeros(side * side),
            ]
        ).astype(np.float32)
        out = tmp_path / "sorted-sheet.luxar.zarr"
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                substitutive_lod=dict(
                    coarse="points", compression_factor=4, levels=3, seed=7
                ),
                additive_lod=False,
            )

        group = zarr.open(str(out), mode="r")["cloud"]
        coarse = ArrayDecoder().decode(group["child_0/positions"], group["child_0"])
        first_quarter = np.mean(coarse[:, 0] < 0.25)
        last_quarter = np.mean(coarse[:, 0] >= 0.75)
        assert 0.18 < first_quarter < 0.32
        assert 0.18 < last_quarter < 0.32

    def test_points_coarse_seed_controls_order(self, tmp_path) -> None:
        from luxar.encoding import ArrayDecoder

        decoder = ArrayDecoder()
        positions = np.column_stack(
            [np.arange(2000), np.zeros(2000), np.zeros(2000)]
        ).astype(np.float32)

        def build(seed: int, suffix: str) -> np.ndarray:
            out = tmp_path / f"seed-{suffix}.luxar.zarr"
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "cloud",
                    positions,
                    substitutive_lod=dict(
                        coarse="points", compression_factor=4, levels=1, seed=seed
                    ),
                    additive_lod=False,
                )
            group = zarr.open(str(out), mode="r")["cloud"]
            return decoder.decode(group["child_0/positions"], group["child_0"])

        first = build(7, "first")
        repeated = build(7, "repeated")
        different = build(8, "different")
        np.testing.assert_array_equal(first, repeated)
        assert not np.array_equal(first, different)

    def test_points_coarse_allows_continuous_hidden_axis(self, tmp_path) -> None:
        rng = np.random.default_rng(0)
        positions = np.column_stack(
            [rng.normal(size=(4000, 3)), np.linspace(0, 1, 4000)]
        ).astype(np.float32)
        dims = Dimensions(
            [
                Dimension("x"),
                Dimension("y"),
                Dimension("z"),
                Dimension("w", display=False, spatial=True),
            ]
        )
        out = tmp_path / "continuous-hidden.luxar.zarr"
        with pytest.warns(RuntimeWarning, match="continuous hidden dimensions"):
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=dims)
                scene.add_points(
                    "cloud",
                    positions,
                    substitutive_lod=dict(
                        coarse="points", compression_factor=4, levels=2
                    ),
                    additive_lod=False,
                )

        group = zarr.open(str(out), mode="r")["cloud"]
        assert group["child_0"].attrs["n_points"] == 250

    def test_numeric_compensation_is_exponentiated_per_level(self, tmp_path) -> None:
        from luxar.encoding import ArrayDecoder

        out = tmp_path / "points-explicit-gain.luxar.zarr"
        positions = np.random.default_rng(0).uniform(0, 1, (64, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                colors=np.ones((64, 3), dtype=np.float32),
                substitutive_lod=dict(
                    coarse="points",
                    compression_factor=4,
                    levels=2,
                    brightness_compensation=2,
                ),
            )
        group = zarr.open(str(out), mode="r")["cloud"]
        decoder = ArrayDecoder()
        np.testing.assert_allclose(decoder.decode(group["child_0/colors"], group), 4.0)
        np.testing.assert_allclose(decoder.decode(group["child_1/colors"], group), 2.0)

    def test_points_coarse_partitioned_finest_and_scalar_bake(self, tmp_path) -> None:
        out = tmp_path / "points-partition-scalars.luxar.zarr"
        rng = np.random.default_rng(0)
        positions = rng.uniform(0, 20, (400, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                positions,
                scalars=np.linspace(0, 1, len(positions), dtype=np.float32),
                colormap="viridis",
                partition={"max_elements": 100},
                substitutive_lod=dict(coarse="points", levels=1),
                additive_lod=False,
            )
        group = zarr.open(str(out), mode="r")["cloud"]
        assert group["child_0"].attrs["type"] == "points"
        assert group["child_0"].attrs.get("has_colors") is True
        assert group["child_1"].attrs["kind"] == "partition"

    def test_compensated_levels_do_not_warn_and_labels_stay_finest(
        self, tmp_path
    ) -> None:
        out = tmp_path / "points-labels.luxar.zarr"
        positions = np.random.default_rng(0).uniform(0, 1, (600, 3)).astype(np.float32)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "cloud",
                    positions,
                    image_labels=[b"x"] * len(positions),
                    substitutive_lod=dict(coarse="points", levels=2),
                )
        assert not [record for record in records if "HDR colors" in str(record.message)]
        group = zarr.open(str(out), mode="r")["cloud"]
        assert group["child_0"].attrs.get("has_image_labels") is not True
        assert group["child_1"].attrs.get("has_image_labels") is not True
        assert group["child_2"].attrs.get("has_image_labels") is True

    def test_group_is_kind_lod_points(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        # A DERIVED ladder stamps the screen-area selector (thresholds are
        # literal screen-area fractions).
        assert grp.attrs["selector"] == "screen-area"
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

    def test_coverage_fraction_monotone_coarsest_zero(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=3)
        cf = [float(grp[f"child_{i}"].attrs["coverage_fraction"]) for i in range(4)]
        assert cf[0] == 0.0
        assert cf[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
        assert all(cf[i] < cf[i + 1] for i in range(len(cf) - 1))

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

    def test_coarse_level_aspect_capped(self) -> None:
        # Points mirror of the Lines anisotropy cap (default max_aspect=3):
        # coarse merged splats of a lifted (isotropic) cloud must stay near-
        # isotropic so their rendered brightness is not view-dependent.
        from luxar.gsplats.utils.trils import unpack_tril

        rng = np.random.RandomState(1)
        # An elongated filament-like cloud, the worst case for bin elongation.
        pos = np.zeros((4000, 3), np.float32)
        pos[:, 2] = np.linspace(0, 400, 4000)
        pos[:, :2] = rng.normal(0, 0.5, (4000, 2))
        lifted = lift_points_to_gsplats(pos, 0.6, colors=None)
        coarse = coarse_substitutive_levels(
            lifted, compression_factor=4, levels=3, device="cpu", seed=0
        )
        for lvl in coarse:
            L = unpack_tril(np.asarray(lvl.cholesky_factors, np.float64), 3)
            ev = np.linalg.eigvalsh(L @ np.swapaxes(L, 1, 2))
            aspects = np.sqrt(ev[:, -1] / np.maximum(ev[:, 0], 1e-30))
            assert float(aspects.max()) <= 3.0 * (1 + 1e-4)

    def test_explicit_coverage_fractions_override(self, tmp_path) -> None:
        grp, _ = _build(tmp_path, levels=3, coverage_fractions=[0.0, 0.05, 0.25, 1.0])
        cf = [float(grp[f"child_{i}"].attrs["coverage_fraction"]) for i in range(4)]
        assert cf == [0.0, 0.05, 0.25, 1.0]

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
                scene.add_points(
                    "cloud",
                    pos,
                    colors=colors_u8,
                    radii=1.0,
                    substitutive_lod=dict(levels=2, device="cpu", seed=0),
                )
            hdr = [w for w in caught if "HDR" in str(w.message)]
        grp = zarr.open(str(out), mode="r")["cloud"]
        # child_0 is a coarse gsplat level; its colors must be SDR (uint8), not
        # the blown-out float32-HDR the pre-fix lift produced.
        assert np.asarray(grp["child_0"]["colors"]).dtype == np.uint8
        assert len(hdr) == 0, (
            f"unexpected HDR colour warning(s): {[str(w.message) for w in hdr]}"
        )

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
                scene.add_points(
                    "cloud",
                    pos,
                    radii=1.0,
                    opacity=op,
                    substitutive_lod=dict(levels=2, device="cpu", seed=0),
                )
            return zarr.open(str(out), mode="r")["cloud"]

        g_half, g_full = build(0.5), build(1.0)
        assert float(g_half.attrs["opacity"]) == pytest.approx(0.5)
        assert float(g_full.attrs["opacity"]) == pytest.approx(1.0)
        # Amplitudes (and their stored range) must be identical regardless of opacity.
        a_half = np.asarray(g_half["child_0"]["amplitudes"])
        a_full = np.asarray(g_full["child_0"]["amplitudes"])
        np.testing.assert_array_equal(a_half, a_full)
        assert g_half["child_0"].attrs["amplitude_range"]["max"] == pytest.approx(
            g_full["child_0"].attrs["amplitude_range"]["max"]
        )

    def test_render_light_survives_writer_quantization(self, tmp_path) -> None:
        # End-to-end: the render-light rescale must survive the writer's lossy
        # encodings (uint16 amplitudes AND the per-channel log/signed-log Cholesky
        # quantization). Decode each array through ArrayDecoder so the per-channel
        # Cholesky dequant + amplitude dequant are applied exactly as on load.
        from luxar.encoding import ArrayDecoder
        from luxar.gsplats.utils.trils import merge_tril, unpack_tril

        grp, _ = _build(tmp_path, levels=3, radius_scale=10.0)
        dec = ArrayDecoder()
        lights = []
        for i in range(3):  # gsplat children
            child = grp[f"child_{i}"]
            assert "amplitude_normalization_factor" not in child.attrs
            a = dec.decode(child["amplitudes"], grp).astype(np.float64)
            # v3.1: Cholesky stored split (diag + offdiag), each per-channel
            # quantized; decode then recombine into packed L.
            diag = dec.decode(child["cholesky_factors_diag"], grp).astype(np.float64)
            off = dec.decode(child["cholesky_factors_offdiag"], grp).astype(np.float64)
            L = unpack_tril(merge_tril(diag, off, 3), 3)
            det = np.abs(np.linalg.det(L))
            lights.append(float(np.sum(a * det)))
        for li in lights[1:]:
            assert li == pytest.approx(lights[0], rel=0.03)


class TestCoarsePointLevels:
    def test_validation_and_degenerate_inputs(self) -> None:
        with pytest.raises(ValueError, match="order has 2 entries"):
            coarse_point_levels(
                np.array([0, 1]),
                np.array([1.0]),
                compression_factor=4,
                levels=2,
            )
        assert (
            coarse_point_levels(
                np.array([0]),
                np.array([1.0]),
                compression_factor=4,
                levels=2,
            )
            == []
        )

    def test_duplicate_counts_and_zero_energy_fallback(self) -> None:
        levels = coarse_point_levels(
            np.arange(3),
            np.zeros(3),
            compression_factor=2,
            levels=4,
        )
        assert len(levels) == 1
        np.testing.assert_array_equal(levels[0][0], [0])
        assert levels[0][1] == 1.0


class TestSubstitutiveComposedWithAdditive:
    """``additive_lod`` composes with ``substitutive_lod`` (it used to raise).

    Substitutive chooses WHICH level renders at the current zoom; additive
    describes HOW each level streams in. Without the composition the finest level
    of a substitutive Points ladder was the one node in the system that could not
    paint progressively — it committed all-or-nothing however large it was, which
    is what made the 9.75M-point DESI demo freeze the main thread for ~85s.

    Every build here runs a real CPU k-means, so unlike the rest of this file
    these tests share ONE module-scoped scene and assert many things against it.
    """

    N = 4000

    @pytest.fixture(scope="class")
    def composed(self, tmp_path_factory) -> tuple:
        out = tmp_path_factory.mktemp("composed") / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        pos = rng.normal(0, 20, (self.N, 3)).astype(np.float32)
        colors = rng.uniform(0.1, 1.0, (self.N, 3)).astype(np.float32)
        radii = rng.uniform(0.5, 1.5, self.N).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                colors=colors,
                radii=radii,
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
                additive_lod=dict(counts="stream:400", method="random", seed=0),
            )
        return zarr.open(str(out), mode="r")["cloud"], self.N

    @staticmethod
    def _children(grp) -> list:
        return sorted(k for k in grp.keys() if k.startswith("child_"))

    def test_finest_child_carries_a_ladder(self, composed) -> None:
        grp, n = composed
        finest = grp[self._children(grp)[-1]]

        assert finest.attrs["type"] == "points"
        n_sub = int(finest.attrs["n_additive_sublods"])
        assert n_sub > 1
        subs = sorted(k for k in finest.keys() if k.startswith("additive_"))
        assert subs == [f"additive_{i}" for i in range(n_sub)]
        assert sum(int(finest[s].attrs["n_points"]) for s in subs) == n

    def test_default_ladder_uses_real_hidden_slice_count(
        self, tmp_path, monkeypatch
    ) -> None:
        from luxar.core.group.lod import group as lod_group

        monkeypatch.setattr(lod_group, "DEFAULT_LADDER_TARGET_MS", 0.1)
        n_slices = 30
        points_per_slice = 20
        n = n_slices * points_per_slice
        rng = np.random.default_rng(7)
        positions = np.column_stack(
            [
                rng.normal(size=(n, 3)),
                np.repeat(np.arange(n_slices), points_per_slice),
            ]
        ).astype(np.float32)
        dims = Dimensions(
            [
                Dimension("x"),
                Dimension("y"),
                Dimension("z"),
                Dimension("time", display=False, discrete=True),
            ]
        )
        out = tmp_path / "sliced.luxar.zarr"
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "cloud",
                positions,
                substitutive_lod=dict(
                    compression_factor=20,
                    levels=1,
                    method="greedy",
                    device="cpu",
                    seed=0,
                ),
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        coarse = grp[self._children(grp)[0]]
        finest = grp[self._children(grp)[-1]]
        assert int(coarse.attrs.get("n_additive_sublods", 1)) > 1
        assert int(finest.attrs.get("n_additive_sublods", 1)) > 1
        assert int(finest["additive_0"].attrs["n_points"]) == 75

    def test_coarse_levels_are_laddered_too(self, composed) -> None:
        # Full symmetry with the gsplat pyramid, which ladders every level.
        # A coarse level smaller than one stream chunk stays a flat leaf —
        # that fallback is what keeps tiny levels from growing useless subgroups.
        grp, _ = composed
        children = self._children(grp)

        for name in children[:-1]:
            child = grp[name]
            assert child.attrs["type"] == "gsplats"
            n_splats = int(child.attrs.get("n_splats", 0) or 0)
            if int(child.attrs.get("n_additive_sublods", 1)) > 1:
                assert any(k.startswith("additive_") for k in child.keys())
            else:
                # Only legitimate when the level is too small to split.
                assert n_splats <= 400 or n_splats == 0

    def test_ladder_is_not_degenerate(self, composed) -> None:
        # The trap this guards: global_rivers' terrain ladder puts 99.98% of its
        # 8M points in the LAST level, so it streams in name only. A ladder whose
        # biggest level is most of the data does not fix anything.
        grp, n = composed
        finest = grp[self._children(grp)[-1]]
        n_sub = int(finest.attrs["n_additive_sublods"])

        sizes = [int(finest[f"additive_{i}"].attrs["n_points"]) for i in range(n_sub)]
        assert max(sizes) / n <= 0.6, sizes

    def test_first_level_is_one_stream_chunk(self, composed) -> None:
        grp, _ = composed
        finest = grp[self._children(grp)[-1]]

        # The finest level has a coarser sibling, so the sibling-aware rule
        # raises its first chunk to ceil(n/(2K)) = 4000/8 = 500 > the 400 asked.
        assert int(finest["additive_0"].attrs["n_points"]) == 500

    def test_lod_group_invariants_survive_composition(self, composed) -> None:
        grp, _ = composed
        children = self._children(grp)

        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        # Derived ladder → screen-area selector survives the composition.
        assert grp.attrs["selector"] == "screen-area"
        assert int(grp.attrs["default_level"]) == 0
        assert "position_bounds" in grp.attrs
        cf = [float(grp[c].attrs["coverage_fraction"]) for c in children]
        assert cf[0] == 0.0
        assert cf[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
        assert all(a < b for a, b in zip(cf, cf[1:])), cf

    def test_every_level_is_energy_stamped(self, composed) -> None:
        # Both halves of the viewer's display-gate contract, end to end: the
        # per-prefix fraction and the leaf's total. Missing either makes the gate
        # fall back to counting elements.
        grp, _ = composed
        finest = grp[self._children(grp)[-1]]
        n_sub = int(finest.attrs["n_additive_sublods"])

        fracs = [
            finest[f"additive_{i}"].attrs["lod_stats"]["energy_fraction_cum"]
            for i in range(n_sub)
        ]
        assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
        assert fracs[-1] == pytest.approx(1.0)
        assert dict(finest.attrs["level_stats"])["reference_energy"] > 0

        # The pairing is both-or-neither for EVERY child of the composed lod
        # group — including the coarse synthesised-gsplat children. A child that
        # ladders (n_additive_sublods > 1) and stamps energy_fraction_cum on its
        # sub-LODs MUST ride the paired reference_energy on its own leaf. Coarse
        # levels too small to split stay a single flat leaf and don't stream, so
        # they carry no fraction stamps and are skipped.
        children = self._children(grp)
        finest_name = children[-1]
        coarse_checked = 0  # coarse (non-finest) laddered children actually asserted
        for name in children:
            child = grp[name]
            child_n_sub = int(child.attrs.get("n_additive_sublods", 1))
            if child_n_sub <= 1:
                continue
            sub_fracs = [
                child[f"additive_{i}"]
                .attrs.get("lod_stats", {})
                .get("energy_fraction_cum")
                for i in range(child_n_sub)
            ]
            if not all(f is not None for f in sub_fracs):
                continue  # not fraction-stamped → no pairing obligation
            assert dict(child.attrs["level_stats"])["reference_energy"] > 0, name
            if name != finest_name:
                coarse_checked += 1
        # Vacuity floor: at least one COARSE laddered child was genuinely checked,
        # so the loop can never silently assert nothing if stamps disappear.
        assert coarse_checked >= 1

    def test_sublods_keep_their_spatial_index(self, composed) -> None:
        # Losing the per-sub-LOD index would regress frustum-culled range reads
        # on nD-sliced scenes.
        grp, _ = composed
        finest = grp[self._children(grp)[-1]]

        for i in range(int(finest.attrs["n_additive_sublods"])):
            sub = finest[f"additive_{i}"]
            assert "chunk_bounds" in sub
            assert sub.attrs["ordering"] == "hilbert"

    def test_default_on_without_an_explicit_additive_lod(self, tmp_path) -> None:
        # The behaviour that fixes the existing demos with no demo edits.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(1)
        pos = rng.normal(0, 20, (3000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=np.full(3000, 1.0, dtype=np.float32),
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        finest = grp[sorted(k for k in grp.keys() if k.startswith("child_"))[-1]]
        # 3000 points against the default 39062-element first chunk: the whole
        # level fits in one chunk, so it correctly stays a flat leaf.
        assert finest.attrs["type"] == "points"
        assert int(finest.attrs.get("n_additive_sublods", 1)) == 1

    def test_default_on_ladders_a_finest_child_above_one_chunk(self, tmp_path) -> None:
        # The real regression guard for the default composition: N well ABOVE the
        # default stream chunk (39062), so the finest child MUST carry a
        # multi-sublod ladder. This FAILS if default composition is removed
        # (unlike the 3,000-point sibling above, which fits one chunk and passes
        # even with composition deleted).
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(4)
        n = 80_000
        pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=np.full(n, 1.0, dtype=np.float32),
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        finest = grp[sorted(k for k in grp.keys() if k.startswith("child_"))[-1]]
        assert finest.attrs["type"] == "points"
        assert int(finest.attrs.get("n_additive_sublods", 1)) > 1

    def test_additive_false_opts_out(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(2)
        pos = rng.normal(0, 20, (3000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=np.full(3000, 1.0, dtype=np.float32),
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
                additive_lod=False,
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        for name in (k for k in grp.keys() if k.startswith("child_")):
            assert "n_additive_sublods" not in grp[name].attrs

    def test_degenerate_input_still_gets_its_ladder(self, tmp_path) -> None:
        # Too small to synthesise coarse levels -> flat Points node. It used to
        # drop the caller's ladder on this path, silently.
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(3).rand(300, 3).astype(np.float32) * 0.001
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_points(
                "cloud",
                pos,
                radii=np.full(300, 1e-6, dtype=np.float32),
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
                additive_lod=dict(counts=[50, 150]),
            )
        assert node is not None

        grp = zarr.open(str(out), mode="r")["cloud"]
        # The caller's ladder (counts=[50, 150] over 300 points -> cuts
        # [50, 150, 300]) must survive whether the build took the degenerate
        # flat-Points path (grp is the leaf) or synthesised a one-level LOD group
        # (grp is the kind=lod group, and the finest child carries the ladder).
        # Assert the ladder UNCONDITIONALLY on whichever finest leaf is produced.
        if dict(grp.attrs).get("type") == "points":
            finest = grp  # degenerate flat path: the leaf itself
        else:
            finest = grp[sorted(k for k in grp.keys() if k.startswith("child_"))[-1]]
        assert int(finest.attrs["n_additive_sublods"]) == 3


class TestAdditiveLevelStatsPairing:
    """A caller-supplied ``level_stats`` must not break the energy pairing.

    ``level_stats`` is a free-form key a caller can thread through the public
    ``add_points(**attrs)``. When the additive ladder is fraction-stamped, the
    parent must still receive the paired ``reference_energy`` (both-or-neither),
    while the caller's own keys survive intact.
    """

    def test_caller_level_stats_without_reference_energy_gets_paired(
        self, tmp_path
    ) -> None:
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(7)
        n = 80_000
        pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=np.full(n, 1.0, dtype=np.float32),
                additive_lod=dict(counts="stream:5000", method="random", seed=0),
                level_stats={"caller_key": 123},
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        assert int(grp.attrs["n_additive_sublods"]) > 1
        stored = dict(grp.attrs["level_stats"])
        # Caller's own key is preserved …
        assert stored["caller_key"] == 123
        # … and the missing half of the pairing is restored.
        assert stored["reference_energy"] > 0

    def test_non_finite_caller_reference_energy_is_replaced(self, tmp_path) -> None:
        # A NaN rides the caller dict straight into .zattrs as a bare NaN token
        # (invalid strict JSON — the viewer's JSON.parse then rejects the whole
        # attrs document), so it must be dropped and the computed finite value
        # must show through instead.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(7)
        n = 80_000
        pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=np.full(n, 1.0, dtype=np.float32),
                additive_lod=dict(counts="stream:5000", method="random", seed=0),
                level_stats={"caller_key": 123, "reference_energy": float("nan")},
            )

        grp = zarr.open(str(out), mode="r")["cloud"]
        stored = dict(grp.attrs["level_stats"])
        assert stored["caller_key"] == 123
        assert np.isfinite(stored["reference_energy"])
        assert stored["reference_energy"] > 0


# ────────────────────────────────────────────────────────────────────────
# Partition-bound anchor (hand-built kind=partition of per-part ladders)
# ────────────────────────────────────────────────────────────────────────


def _partitioned_points(
    tmp_path, *, n=600, levels=2, partitioned=True, wrap_in_group=False, **kw
):
    """Hand-build a ``kind=partition`` wrapper and one ladder per part.

    ``partitioned=False`` places the same ladders at the scene ROOT instead (the
    over-trigger control), so both variants get identical geometry and only the
    insertion point differs.

    The adaptive shape ``demo_biodiversity_planetary_scale`` builds: the combined
    ``add_points(partition=..., substitutive_lod=...)`` spelling authors a global
    overview cap, so a caller who wants one substitutive ladder per tile creates
    the partition wrapper itself and calls the adder once per part.

    ``method="kmeans_lloyd"`` instead of the default ``"auto"``: at these sizes
    ``auto`` routes to the submodular ``greedy`` Runnalls path, whose sparse-Gram
    build scales with OVERLAP DENSITY and costs tens of seconds per part on a
    lifted cloud. Every assertion below reads per-level COUNTS
    (``compression_factor``/``levels`` decide those, identically for either
    method) and never reduction quality, so the cheap O(N log N) reduction is
    sound here — the same reasoning as ``_acceptance_params`` in
    ``test_gsplats.py``. Pre-existing tests keep ``auto``.
    """
    out = tmp_path / "p.luxar.zarr"
    rng = np.random.RandomState(0)
    pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
    radii = rng.uniform(0.5, 1.5, n).astype(np.float32)
    halves = [np.arange(n) % 2 == 0, np.arange(n) % 2 == 1]
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        wrapper = (
            scene.add_partition_group(
                "tiled", display_type="points", max_elements=n // 2
            )
            if partitioned
            else scene
        )
        for i, idx in enumerate(halves):
            target = wrapper.add_group(f"holder_{i}") if wrap_in_group else wrapper
            target.add_points(
                f"part_{i}",
                pos[idx],
                radii=radii[idx],
                substitutive_lod=dict(
                    compression_factor=4,
                    levels=levels,
                    method="kmeans_lloyd",
                    device="cpu",
                    seed=0,
                    **kw,
                ),
            )
    root = zarr.open(str(out), mode="r")
    return (root["tiled"] if partitioned else root), len(halves)


def _child_coverage(lod_group) -> list:
    """Per-child ``coverage_fraction``, coarsest→finest."""
    names = sorted(
        (k for k in lod_group.keys() if k.startswith("child_")),
        key=lambda k: int(k.split("_")[1]),
    )
    return [float(lod_group[k].attrs["coverage_fraction"]) for k in names]


def _child_counts(lod_group) -> list:
    names = sorted(
        (k for k in lod_group.keys() if k.startswith("child_")),
        key=lambda k: int(k.split("_")[1]),
    )
    return [
        int(lod_group[k].attrs.get("n_splats") or lod_group[k].attrs.get("n_points"))
        for k in names
    ]


class TestPartitionBoundAnchorPoints:
    """A hand-built partition of per-part ladders gets the fills-screen anchor.

    Before ``derive_coverage_fractions`` detected the ``kind=partition`` ancestor,
    this path always auto-derived the WHOLE-OBJECT ladder (finest 0.5 — half
    the screen area) — so every tile sat on its finest level at the opening
    whole-object framing. The expected finest is ``PARTITION_FINEST_AREA``
    (1.0 — the tile alone fills the screen).

    Three tests here REGRESS without the fix (``test_every_part_ladder_is_partition
    _anchored``, ``test_finest_is_exactly_the_ceiling``,
    ``test_plain_group_between_partition_and_ladder_still_anchored``). The other two
    are CONTROLS that pass either way and pin what must NOT change: no
    over-triggering outside a partition, and an explicit list still winning.
    """

    def test_every_part_ladder_is_partition_anchored(self, tmp_path) -> None:
        wrapper, n_parts = _partitioned_points(tmp_path)
        assert wrapper.attrs["kind"] == "partition"
        for i in range(n_parts):
            part = wrapper[f"part_{i}"]
            assert part.attrs["kind"] == "lod"
            counts = _child_counts(part)
            # The finest count IS the point count for Points (the lift drops no
            # positive-radius point), so the on-disk counts reproduce the ladder.
            assert _child_coverage(part) == pytest.approx(
                partitioned_coverage_fractions(counts)
            )

    def test_finest_is_exactly_the_ceiling(self, tmp_path) -> None:
        wrapper, n_parts = _partitioned_points(tmp_path)
        for i in range(n_parts):
            assert _child_coverage(wrapper[f"part_{i}"])[-1] == pytest.approx(
                PARTITION_FINEST_AREA
            )

    def test_scene_root_still_gets_the_whole_object_anchor(self, tmp_path) -> None:
        """CONTROL (passes pre-fix): the SAME ladders at the scene root must keep
        the whole-object anchor — the over-trigger guard."""
        root, n_parts = _partitioned_points(tmp_path, partitioned=False)
        for i in range(n_parts):
            assert _child_coverage(root[f"part_{i}"])[-1] == pytest.approx(
                WHOLE_OBJECT_FINEST_ANCHOR
            )

    def test_plain_group_between_partition_and_ladder_still_anchored(
        self, tmp_path
    ) -> None:
        wrapper, n_parts = _partitioned_points(tmp_path, wrap_in_group=True)
        for i in range(n_parts):
            part = wrapper[f"holder_{i}"][f"part_{i}"]
            assert _child_coverage(part)[-1] == pytest.approx(PARTITION_FINEST_AREA)

    def test_explicit_coverage_fractions_still_win_under_a_partition(
        self, tmp_path
    ) -> None:
        """CONTROL (passes pre-fix): the biodiversity demo's contract — it
        hand-tunes the list itself, and that must keep winning verbatim."""
        explicit = [0.0, 3.52, 4.0]
        wrapper, n_parts = _partitioned_points(
            tmp_path, levels=2, coverage_fractions=explicit
        )
        for i in range(n_parts):
            assert _child_coverage(wrapper[f"part_{i}"]) == pytest.approx(explicit)

    def test_one_part_partition_also_gets_the_tile_anchor_documented_blind_spot(
        self, tmp_path
    ) -> None:
        """CHARACTERIZATION of the documented blind spot, not an endorsement.

        A hand-built ``kind=partition`` holding exactly ONE part is not a tiling —
        that part IS the whole object — so the fills-screen anchor is a factor of 2
        too coarse for it (in screen-area units: tile 1.0 vs whole-object 0.5). ``derive_coverage_fractions`` gives it the tile anchor
        (``PARTITION_FINEST_AREA`` = 1.0) anyway, and cannot do otherwise: part 0's
        ladder is derived at ``add_points`` time, when the sibling count does not
        exist yet (part 1 may never be added). That is the caveat spelled out in
        ``partitioned_coverage_fractions``.

        Nothing rewrites it — an explicit ``coverage_fractions=`` list is
        indistinguishable from a derived one on disk — but the compiler's finalize
        pass DOES see the final sibling count and warns
        (``io/_compiler/finalize/lod_backfill.py::warn_one_part_partition_anchors``).
        This test exists so the code and that documented caveat cannot drift apart.
        """
        out = tmp_path / "one_part.luxar.zarr"
        rng = np.random.RandomState(0)
        n = 1000
        pos = rng.normal(0, 20, (n, 3)).astype(np.float32)
        radii = rng.uniform(0.5, 1.5, n).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            wrapper = scene.add_partition_group(
                "tiled", display_type="points", max_elements=n
            )
            wrapper.add_points(
                "part_0",
                pos,
                radii=radii,
                substitutive_lod=dict(
                    compression_factor=4,
                    levels=2,
                    method="kmeans_lloyd",
                    device="cpu",
                    seed=0,
                ),
            )
        tiled = zarr.open(str(out), mode="r")["tiled"]
        assert len([k for k in tiled.keys() if k.startswith("part_")]) == 1
        assert _child_coverage(tiled["part_0"])[-1] == pytest.approx(
            PARTITION_FINEST_AREA
        ), "the lone part still takes the tile anchor — the documented blind spot"


class TestSubstitutiveLodGuards:
    def test_partition_and_substitutive_builds_overview(self, tmp_path) -> None:
        """The combined spelling authors a coarse cap over fine spatial parts."""
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        pos = rng.uniform(0, 40, (600, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                radii=1.0,
                partition={"max_elements": 200},
                substitutive_lod={
                    "compression_factor": 4,
                    "levels": 1,
                    "method": "kmeans_lloyd",
                    "device": "cpu",
                    "seed": 0,
                },
                additive_lod=False,
            )

        group = zarr.open(str(out), mode="r")["pts"]
        assert group.attrs["kind"] == "lod"
        assert group.attrs["selector"] == "screen-area"
        assert group.attrs["display_type"] == "points"
        assert set(group.keys()) == {"child_0", "child_1"}

        coarse = group["child_0"]
        fine = group["child_1"]
        assert coarse.attrs["type"] == "gsplats"
        assert coarse.attrs["coverage_fraction"] == 0.0
        assert fine.attrs["kind"] == "partition"
        assert fine.attrs["display_type"] == "points"
        assert fine.attrs["coverage_fraction"] == PARTITION_FINEST_AREA
        part_names = list(fine.keys())
        assert len(part_names) > 1
        assert sum(int(fine[name].attrs["n_points"]) for name in part_names) == len(pos)
        assert all(fine[name].attrs["type"] == "points" for name in part_names)

    def test_overview_fine_parts_keep_streaming_ladders(
        self, tmp_path, monkeypatch
    ) -> None:
        """Each fine part sizes its additive ladder from its own point count."""
        from luxar.core.group.lod import group as lod_group

        monkeypatch.setattr(lod_group, "DEFAULT_LADDER_TARGET_MS", 0.1)
        out = tmp_path / "t.luxar.zarr"
        n_slices = 30
        points_per_slice = 20
        n = n_slices * points_per_slice
        spatial = np.random.RandomState(0).uniform(0, 40, (n, 3)).astype(np.float32)
        pos = np.column_stack(
            [spatial, np.repeat(np.arange(n_slices), points_per_slice)]
        ).astype(np.float32)
        dims = Dimensions(
            [
                Dimension("x"),
                Dimension("y"),
                Dimension("z"),
                Dimension("time", display=False, discrete=True),
            ]
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_points(
                "pts",
                pos,
                radii=1.0,
                partition={"max_elements": 50},
                substitutive_lod={
                    "compression_factor": 20,
                    "levels": 1,
                    "method": "kmeans_lloyd",
                    "device": "cpu",
                    "seed": 0,
                },
            )

        fine = zarr.open(str(out), mode="r")["pts/child_1"]
        part_names = list(fine.keys())
        assert len(part_names) > 1
        assert all(
            int(fine[name].attrs.get("n_additive_sublods", 1)) > 1
            for name in part_names
        )

    def test_overview_fine_parts_keep_explicit_streaming_ladders(
        self, tmp_path
    ) -> None:
        """An explicit ladder remains independently resolved on every fine part."""
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).uniform(0, 40, (1200, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                radii=1.0,
                partition={"max_elements": 200},
                substitutive_lod={
                    "compression_factor": 4,
                    "levels": 1,
                    "method": "kmeans_lloyd",
                    "device": "cpu",
                    "seed": 0,
                },
                additive_lod={
                    "method": "random",
                    "counts": "stream:100",
                    "seed": 0,
                },
            )

        fine = zarr.open(str(out), mode="r")["pts/child_1"]
        part_names = list(fine.keys())
        assert len(part_names) > 1
        assert all(
            int(fine[name].attrs.get("n_additive_sublods", 1)) == 2
            for name in part_names
        )

    def test_one_part_combination_falls_back_to_whole_object_lod(
        self, tmp_path
    ) -> None:
        """A partition request that does not split must not take the overview anchor."""
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).rand(100, 3).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                radii=1.0,
                partition={"max_elements": 200},
                substitutive_lod={
                    "levels": 1,
                    "method": "kmeans_lloyd",
                    "device": "cpu",
                    "seed": 0,
                },
                additive_lod=False,
            )

        group = zarr.open(str(out), mode="r")["pts"]
        assert group.attrs["kind"] == "lod"
        assert group["child_1"].attrs["type"] == "points"
        assert group["child_1"].attrs["coverage_fraction"] == WHOLE_OBJECT_FINEST_ANCHOR

    def test_substitutive_takes_precedence_over_auto_partition(self, tmp_path) -> None:
        # With compiler auto-partition enabled, substitutive_lod must still win
        # (build the LOD group, not silently get auto-partitioned away).
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out, auto_partition_max_elements=2000) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=1.0,
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
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
            scene.add_points(
                "cloud",
                pos,
                scalars=0.5,
                colormap="viridis",
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["cloud"]
        assert grp.attrs["kind"] == "lod"

    def test_scalars_plus_colormap_bakes_colors_on_coarse_levels(
        self, tmp_path
    ) -> None:
        # scalars+colormap now WORKS: coarse gsplat levels carry baked SDR colours
        # from the colormap; the finest Points child keeps scalars+colormap.
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (6000, 3)).astype(np.float32)
        scalars = rng.uniform(0, 1, 6000).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                scalars=scalars,
                colormap="viridis",
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
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
            scene.add_points(
                "cloud",
                pos,
                radii=1.0,
                image_labels=[b"x"] * 3000,
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["cloud"]
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        finest = grp[children[-1]]
        assert finest.attrs["type"] == "points"
        assert finest.attrs.get("has_image_labels") is True

    def test_image_labels_only_flatten_finest_additive_ladder(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        rng = np.random.default_rng(0)
        pos = rng.uniform(0, 40, (600, 3)).astype(np.float32)
        with pytest.warns(UserWarning, match="reveal_center is not applied"):
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "cloud",
                    pos,
                    radii=1.0,
                    image_labels=[b"x"] * len(pos),
                    substitutive_lod=dict(
                        compression_factor=2, levels=1, device="cpu", seed=0
                    ),
                    additive_lod={"method": "radial", "counts": "stream:50"},
                )

        grp = zarr.open(str(out), mode="r")["cloud"]
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert int(grp[children[0]].attrs.get("n_additive_sublods", 1)) > 1
        assert grp[children[-1]].attrs.get("has_image_labels") is True
        assert "n_additive_sublods" not in grp[children[-1]].attrs

    def test_image_labels_degenerate_fallback_uses_finest_policy(
        self, tmp_path
    ) -> None:
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.default_rng(0).uniform(0, 1, (300, 3)).astype(np.float32)
        with pytest.warns(UserWarning, match="where one applies"):
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_points(
                    "cloud",
                    pos,
                    radii=0.0,
                    image_labels=[b"x"] * len(pos),
                    substitutive_lod=dict(levels=2, device="cpu", seed=0),
                    additive_lod={"counts": "stream:50"},
                )

        node = zarr.open(str(out), mode="r")["cloud"]
        assert node.attrs["type"] == "points"
        assert node.attrs.get("has_image_labels") is True
        assert "n_additive_sublods" not in node.attrs

    def test_empty_cloud_falls_through_to_canonical_path(self, tmp_path) -> None:
        # N=0 must route through the canonical path (the n_points>0 guard), which
        # rejects empty input with its normal "empty points" error — NOT a
        # confusing substitutive-internal 'coarsest child must have >=1' error.
        out = tmp_path / "t.luxar.zarr"
        empty = np.zeros((0, 3), np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="empty points") as exc:
                scene.add_points(
                    "pts",
                    empty,
                    radii=1.0,
                    substitutive_lod=dict(levels=2, device="cpu"),
                )
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
            scene.add_points(
                "pts",
                pos,
                radii=0.0,  # every point degenerate
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["pts"]
        assert grp.attrs.get("kind") != "lod"  # flat fallback, not a 1-child LOD
        assert grp.attrs.get("type") == "points"

    def test_all_zero_radius_overview_falls_back_to_partition(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).normal(0, 20, (600, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                radii=0.0,
                partition={"max_elements": 200},
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
                additive_lod=False,
                opacity=0.5,
            )

        wrapper = zarr.open(str(out), mode="r")["pts"]
        part_names = list(wrapper.keys())
        assert wrapper.attrs["kind"] == "partition"
        assert wrapper.attrs["opacity"] == pytest.approx(0.5)
        assert len(part_names) > 1
        assert sum(int(wrapper[name].attrs["n_points"]) for name in part_names) == len(
            pos
        )
        assert all(
            float(wrapper[name].attrs.get("opacity", 1.0)) == pytest.approx(1.0)
            for name in part_names
        )

    def test_tiny_input_builds_valid_group_without_crashing(self, tmp_path) -> None:
        # Very small N still reduces (each coarse level may be 1 splat); the
        # builder must produce a valid kind=lod group with the points node finest.
        out = tmp_path / "t.luxar.zarr"
        pos = np.random.RandomState(0).rand(5, 3).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                pos,
                radii=1.0,
                substitutive_lod=dict(levels=2, device="cpu", seed=0),
            )
        grp = zarr.open(str(out), mode="r")["pts"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "points"
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert grp[children[-1]].attrs["type"] == "points"  # finest is points


# ────────────────────────────────────────────────────────────────────────
# coarsen_dims — barrier-aware coarsening (Auto default = group by non-display)
# ────────────────────────────────────────────────────────────────────────


def _build_4d(
    tmp_path,
    *,
    n_per=1500,
    n_groups=3,
    coarsen_dims="__unset__",
    dim_order=None,
):
    """A 4D scene: categorical (display=False) `coloring` dim 0 + xyz displayed.

    The same xyz is stacked at each coloring value, so a barrier-unaware
    coarsening would merge across colorings.
    """
    rng = np.random.default_rng(0)
    xyz = rng.normal(0, 5, (n_per, 3)).astype(np.float32)
    parts = [
        np.column_stack([np.full(n_per, g, np.float32), xyz]) for g in range(n_groups)
    ]
    pos = np.vstack(parts).astype(np.float32)
    if dim_order is not None:
        pos = pos[:, [1, 2, 3, 0]]
    palette = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    colors = np.vstack([np.tile(palette[g % 3], (n_per, 1)) for g in range(n_groups)])
    dims = Dimensions(
        [
            Dimension(
                "coloring", categories=[str(g) for g in range(n_groups)], display=False
            ),
            Dimension("x", display=True),
            Dimension("y", display=True),
            Dimension("z", display=True),
        ]
    )
    out = tmp_path / "t4d.luxar.zarr"
    kw = {} if coarsen_dims == "__unset__" else {"coarsen_dims": coarsen_dims}
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points(
            "cloud",
            pos,
            colors=colors.astype(np.float32),
            radii=np.full(len(pos), 0.5, np.float32),
            dim_order=dim_order,
            substitutive_lod=dict(compression_factor=4, levels=3, device="cpu", **kw),
        )
    return zarr.open(str(out), mode="r")["cloud"]


def _coarse_barrier_purity(grp) -> float:
    """Max |fractional offset| of the coarse gsplat children's dim-0 centers."""
    worst = 0.0
    for k in grp.keys():
        if not k.startswith("child_"):
            continue
        child = grp[k]
        if child.attrs.get("type") != "gsplats":
            continue
        from luxar.encoding import ArrayDecoder

        # AUTO centers are uint16 per-axis fixed-point — decode so the sub-integer
        # offset metric is meaningful (raw codes are trivially integers).
        c0 = ArrayDecoder().decode(child["centers"], grp)[:, 0]
        worst = max(worst, float(np.abs(c0 - np.round(c0)).max()))
    return worst


class TestCoarsenDimsPoints:
    def test_auto_default_groups_by_non_displayed(self, tmp_path) -> None:
        # No coarsen_dims passed → Auto = coarsen displayed (x,y,z), group by the
        # non-displayed categorical dim. Coarse splats stay on integer values.
        grp = _build_4d(tmp_path)
        assert _coarse_barrier_purity(grp) < 1e-4

    def test_explicit_names(self, tmp_path) -> None:
        grp = _build_4d(tmp_path, coarsen_dims=["x", "y", "z"])
        assert _coarse_barrier_purity(grp) < 1e-4

    def test_auto_default_after_nonidentity_dim_order(self, tmp_path) -> None:
        grp = _build_4d(
            tmp_path,
            n_per=300,
            dim_order=["x", "y", "z", "coloring"],
        )
        assert _coarse_barrier_purity(grp) < 1e-4
        assert int(grp["child_0"].attrs["n_splats"]) < int(
            grp["child_3"].attrs["n_points"]
        )

    def test_all_dims_blends_across_barrier(self, tmp_path) -> None:
        # Opting out of grouping ("all") reproduces the cross-coloring blend.
        grp = _build_4d(tmp_path, n_groups=4, coarsen_dims="all")
        assert _coarse_barrier_purity(grp) > 0.05

    def test_pure_3d_scene_unaffected(self, tmp_path) -> None:
        # No non-displayed dims → Auto has nothing to group by → builds fine.
        out = tmp_path / "t3d.luxar.zarr"
        rng = np.random.default_rng(1)
        pos = rng.normal(0, 5, (4000, 3)).astype(np.float32)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "cloud",
                pos,
                radii=1.0,
                substitutive_lod=dict(compression_factor=4, levels=2, device="cpu"),
            )
        grp = zarr.open(str(out), mode="r")["cloud"]
        assert grp.attrs["kind"] == "lod"


class TestResolveCoarsenDims:
    def test_default_none(self) -> None:
        assert resolve_substitutive_axis_points({})["coarsen_dims"] is None

    def test_names_passthrough(self) -> None:
        r = resolve_substitutive_axis_points(dict(coarsen_dims=["x", "y", 2]))
        assert r["coarsen_dims"] == ["x", "y", 2]

    def test_display_and_all_sentinels(self) -> None:
        assert (
            resolve_substitutive_axis_points(dict(coarsen_dims="display"))[
                "coarsen_dims"
            ]
            == "display"
        )
        assert (
            resolve_substitutive_axis_points(dict(coarsen_dims="all"))["coarsen_dims"]
            == "all"
        )

    def test_bad_values_raise(self) -> None:
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(coarsen_dims=[]))
        with pytest.raises(ValueError):
            resolve_substitutive_axis_points(dict(coarsen_dims="nope"))
