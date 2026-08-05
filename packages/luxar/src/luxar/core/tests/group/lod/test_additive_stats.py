"""On-disk quality stamps for Points and Lines additive ladders.

The viewer's never-downgrade display gate can release a coarse->fine LOD swap as
soon as the committed prefix carries enough of a level's energy, rather than
waiting for the element count to pass the coarser sibling. That needs two stamps,
and it needs BOTH — ``lod-display-gate.ts``'s ``foldProgress`` poisons a whole
subtree's aggregate energy to null if either is missing on any visible leaf:

* ``lod_stats.energy_fraction_cum`` on each ``additive_<i>/`` subgroup;
* ``level_stats.reference_energy`` on the leaf itself.

GSplats have stamped these for a while; these tests pin the Points and Lines
equivalents, including the deliberate no-stamp behaviour on degenerate input.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimensions
from luxar.core.group.lod.group import additive_level_stats
from luxar.io.compiler import LuxarZarrCompiler


def _sublod_stamps(grp) -> list[dict]:
    n = int(grp.attrs["n_additive_sublods"])
    return [dict(grp[f"additive_{i}"].attrs.get("lod_stats", {})) for i in range(n)]


class TestAdditiveLevelStatsFunction:
    """Direct unit tests for the pure ``additive_level_stats`` helper.

    The writer-path classes below reach this only through the compiler and cover
    just the usable/all-zero cases; these pin the function directly, including
    the non-finite (NaN, +Inf) totals that must refuse to stamp rather than emit
    a fabricated or non-JSON-safe energy.
    """

    _stats = staticmethod(additive_level_stats)

    _KW = dict(
        method="self_energy",
        breakpoints_kind="counts",
        energy_kind="points-luminance-volume",
    )

    def test_normal_ladder_stamps_fractions_and_reference(self) -> None:
        # The happy path: fractions are cumulative, end at 1.0, and the kwargs
        # round-trip verbatim into the stamps.
        per_level, ref, parent = self._stats([1.0, 2.0, 1.0], [10, 20, 10], **self._KW)

        fracs = [s["energy_fraction_cum"] for s in per_level]
        assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
        assert fracs[-1] == pytest.approx(1.0)
        assert ref == pytest.approx(4.0)
        assert parent["reference_energy"] == pytest.approx(4.0)
        assert [s["lod_cumulative_n"] for s in per_level] == [10, 30, 40]
        assert parent["lod_n_lods"] == 3
        assert parent["energy_kind"] == "points-luminance-volume"
        for i, s in enumerate(per_level):
            assert s["lod_method"] == "self_energy"
            assert s["lod_level"] == i
            assert s["lod_breakpoints_kind"] == "counts"
        assert [s["lod_n_elements"] for s in per_level] == [10, 20, 10]

    def test_zero_total_stamps_no_energy_but_keeps_counts(self) -> None:
        # All-black / zero-radius input: no fabricated energy stamps, but the
        # count/method bookkeeping still lands.
        per_level, ref, parent = self._stats([0.0, 0.0], [5, 7], **self._KW)

        assert ref is None
        assert "reference_energy" not in parent
        for s in per_level:
            assert "energy_fraction_cum" not in s
        assert [s["lod_cumulative_n"] for s in per_level] == [5, 12]
        assert all(s["lod_method"] == "self_energy" for s in per_level)

    def test_nan_total_stamps_no_energy(self) -> None:
        # A NaN energy poisons the total; the guard must refuse to stamp.
        per_level, ref, parent = self._stats([1.0, float("nan")], [5, 7], **self._KW)

        assert ref is None
        assert "reference_energy" not in parent
        for s in per_level:
            assert "energy_fraction_cum" not in s

    def test_inf_total_stamps_no_energy(self) -> None:
        # A +Inf energy makes fractions meaningless; the guard refuses to stamp.
        per_level, ref, parent = self._stats([1.0, float("inf")], [5, 7], **self._KW)

        assert ref is None
        assert "reference_energy" not in parent
        for s in per_level:
            assert "energy_fraction_cum" not in s

    def test_single_level_ladder_fraction_is_one(self) -> None:
        # A lone level owns all the energy, so its cumulative fraction is 1.0.
        per_level, ref, parent = self._stats([5.0], [7], **self._KW)

        assert per_level[0]["energy_fraction_cum"] == pytest.approx(1.0)
        assert ref == pytest.approx(5.0)
        assert parent["reference_energy"] == pytest.approx(5.0)

    def test_length_mismatch_raises(self) -> None:
        # Mismatched lengths are an internal bug, not a degenerate input.
        with pytest.raises(ValueError):
            self._stats([1.0, 2.0], [10], **self._KW)


class TestPointsAdditiveStamps:
    @staticmethod
    def _build(tmp_path, *, n=2000, radii=None, colors="ramp", **additive):
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        positions = rng.rand(n, 3).astype(np.float32)
        if isinstance(colors, str) and colors == "ramp":
            # A luminance ramp so energy is genuinely non-uniform.
            ramp = np.linspace(0.05, 1.0, n, dtype=np.float32)
            colors_arr = np.stack([ramp, ramp, ramp], axis=1)
        else:
            colors_arr = colors
        if radii is None:
            radii = np.linspace(0.5, 2.0, n).astype(np.float32)

        spec = dict(counts="stream:250", method="random", seed=0)
        spec.update(additive)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts", positions, colors=colors_arr, radii=radii, additive_lod=spec
            )
        return zarr.open(str(output), mode="r")["pts"], n

    def test_every_sublod_carries_a_cumulative_energy_fraction(self, tmp_path) -> None:
        grp, _ = self._build(tmp_path)

        stamps = _sublod_stamps(grp)
        assert stamps, "expected a laddered node"
        for i, s in enumerate(stamps):
            assert "energy_fraction_cum" in s, f"additive_{i} is unstamped"

    def test_fractions_are_non_decreasing_and_end_at_one(self, tmp_path) -> None:
        grp, _ = self._build(tmp_path)

        fracs = [s["energy_fraction_cum"] for s in _sublod_stamps(grp)]
        assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
        assert fracs[-1] == pytest.approx(1.0)

    def test_cumulative_counts_track_the_levels(self, tmp_path) -> None:
        grp, n = self._build(tmp_path)

        stamps = _sublod_stamps(grp)
        assert stamps[-1]["lod_cumulative_n"] == n
        per_level = [s["lod_n_elements"] for s in stamps]
        assert sum(per_level) == n

    def test_breakpoints_kind_and_method_are_recorded(self, tmp_path) -> None:
        grp, _ = self._build(tmp_path)

        stamps = _sublod_stamps(grp)
        assert {s["lod_breakpoints_kind"] for s in stamps} == {"stream"}
        assert {s["lod_method"] for s in stamps} == {"random"}
        assert [s["lod_level"] for s in stamps] == list(range(len(stamps)))

    def test_leaf_carries_reference_energy(self, tmp_path) -> None:
        # The other half of the gate contract: without this the viewer cannot
        # weight this leaf against its siblings and falls back to counts.
        grp, _ = self._build(tmp_path)

        level_stats = dict(grp.attrs["level_stats"])
        assert level_stats["reference_energy"] > 0
        assert level_stats["energy_kind"] == "points-luminance-volume"
        assert level_stats["lod_n_lods"] == int(grp.attrs["n_additive_sublods"])

    def test_energy_ordering_front_loads_the_first_level(self, tmp_path) -> None:
        # With salience+energy ordering the first prefix should carry a larger
        # share of the energy than of the elements — that is the whole point of
        # the stamp, since it is what lets the gate release early.
        grp, n = self._build(
            tmp_path, method="salience", salience_kind="energy", counts=[400, 900]
        )

        stamps = _sublod_stamps(grp)
        first = stamps[0]
        assert first["energy_fraction_cum"] > first["lod_n_elements"] / n

    def test_degenerate_energy_stamps_nothing_rather_than_lying(self, tmp_path) -> None:
        # All-black colors -> zero luminance -> zero energy everywhere (radii
        # must stay positive, which the writer enforces). Omitting the stamps
        # makes the viewer fall back to its count rule; a fabricated 0.0 would
        # make it release swaps on content carrying no energy at all.
        grp, _ = self._build(tmp_path, colors=np.zeros((2000, 3), dtype=np.float32))

        for s in _sublod_stamps(grp):
            assert "energy_fraction_cum" not in s
        assert "reference_energy" not in dict(grp.attrs.get("level_stats", {}))

    def test_explicit_level_stats_is_not_overwritten(self, tmp_path) -> None:
        output = tmp_path / "t.luxar.zarr"
        positions = np.random.RandomState(1).rand(500, 3).astype(np.float32)

        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_points(
                "pts",
                positions,
                additive_lod=dict(n_lods=3),
                level_stats={"reference_energy": 42.0},
            )

        grp = zarr.open(str(output), mode="r")["pts"]
        assert dict(grp.attrs["level_stats"])["reference_energy"] == 42.0


class TestLinesAdditiveStamps:
    @staticmethod
    def _build(tmp_path, *, n_seg=400, widths=0.8, **additive):
        output = tmp_path / "t.luxar.zarr"
        rng = np.random.RandomState(0)
        verts = rng.rand(n_seg * 2, 3).astype(np.float32)
        ramp = np.linspace(0.05, 1.0, n_seg * 2, dtype=np.float32)
        colors = np.stack([ramp, ramp, ramp], axis=1)

        spec = dict(counts="stream:100", method="random", seed=0)
        spec.update(additive)
        with LuxarZarrCompiler(output) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "ln",
                verts,
                widths,
                colors=colors,
                line_type="segments",
                additive_lod=spec,
            )
        return zarr.open(str(output), mode="r")["ln"], n_seg * 2

    def test_every_sublod_carries_a_cumulative_energy_fraction(self, tmp_path) -> None:
        grp, _ = self._build(tmp_path)

        stamps = _sublod_stamps(grp)
        assert stamps
        for i, s in enumerate(stamps):
            assert "energy_fraction_cum" in s, f"additive_{i} is unstamped"

    def test_fractions_are_non_decreasing_and_end_at_one(self, tmp_path) -> None:
        grp, _ = self._build(tmp_path)

        fracs = [s["energy_fraction_cum"] for s in _sublod_stamps(grp)]
        assert all(a <= b for a, b in zip(fracs, fracs[1:])), fracs
        assert fracs[-1] == pytest.approx(1.0)

    def test_counts_are_in_vertices_and_sum_to_the_total(self, tmp_path) -> None:
        grp, n_vertices = self._build(tmp_path)

        stamps = _sublod_stamps(grp)
        assert sum(s["lod_n_elements"] for s in stamps) == n_vertices
        assert stamps[-1]["lod_cumulative_n"] == n_vertices

    def test_leaf_carries_reference_energy_in_the_lines_currency(
        self, tmp_path
    ) -> None:
        grp, _ = self._build(tmp_path)

        level_stats = dict(grp.attrs["level_stats"])
        assert level_stats["reference_energy"] > 0
        # Deliberately a different currency from points/gsplats; the key
        # documents that so nobody compares them across geometries.
        assert level_stats["energy_kind"] == "lines-tube-volume"
