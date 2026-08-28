"""Tests for substitutive-LOD on Lines (coarse levels = synthesised gsplats).

Mirror of ``test_substitutive_points.py``: each segment is lifted to isotropic
"bead" gaussians, reduced by the gsplat substitutive pipeline, and assembled as
a ``kind=lod`` Group whose finest child is the original Lines node.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import zarr

from luxar.core.dimensions import Dimension, Dimensions
from luxar.core.group.lod.group import PARTITION_FINEST_AREA, WHOLE_OBJECT_FINEST_ANCHOR
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


def _build(
    tmp_path,
    *,
    n_seg=1500,
    line_type="segments",
    levels=2,
    widths=0.8,
    additive_lod=None,
    **kw,
):
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
            additive_lod=additive_lod,
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

    def test_max_aspect_threaded_spec_to_lift(self, tmp_path, monkeypatch) -> None:
        # The adder must forward the USER'S max_aspect to
        # coarse_substitutive_levels — the signature default would silently
        # mask a dropped forward (a mutation no output-based test catches).
        import luxar.gsplats.lift as lift_mod

        seen: list = []
        real = lift_mod.coarse_substitutive_levels

        def spy(*args, **kwargs):
            seen.append(kwargs.get("max_aspect"))
            return real(*args, **kwargs)

        monkeypatch.setattr(lift_mod, "coarse_substitutive_levels", spy)
        _build(tmp_path, n_seg=300, levels=1, max_aspect=5)
        assert seen == [5.0]

    def test_unknown_key_mentions_lines(self) -> None:
        with pytest.raises(ValueError, match="Lines"):
            resolve_substitutive_axis_lines(dict(bogus=1))

    def test_non_ascending_coverage_fractions_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 0.5, 0.1]))

    def test_coverage_fractions_out_of_range_raises(self) -> None:
        # The ceiling is MAX_COVERAGE_FRACTION == SCREEN_FILL_DIAGONAL_RATIO /
        # FILL_FACTOR == 4.0 (roughly the metric
        # a screen-filling object produces), not 1.0.
        with pytest.raises(ValueError, match=r"\[0, 4\]"):
            resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 4.5]))

    def test_coverage_fractions_above_one_accepted(self) -> None:
        # Above the auto-derived 1.0 anchor but within the ceiling (inclusive) —
        # the escape hatch for a level that must hold until the object is LARGER
        # than half the fitted screen axis (e.g. a spatially tiled layer).
        r = resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 1.5]))
        assert r["coverage_fractions"] == [0.0, 1.5]
        r = resolve_substitutive_axis_lines(dict(coverage_fractions=[0.0, 4.0]))
        assert r["coverage_fractions"] == [0.0, 4.0]


class TestAddLinesSubstitutiveLod:
    def test_group_is_kind_lod_lines(self, tmp_path) -> None:
        grp, _ = _build(tmp_path)
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        # A DERIVED ladder stamps the screen-area selector (thresholds are
        # literal screen-area fractions).
        assert grp.attrs["selector"] == "screen-area"
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
        assert cf[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
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

    def test_render_light_survives_writer_quantization(self, tmp_path) -> None:
        from luxar.encoding import ArrayDecoder
        from luxar.gsplats.utils.trils import merge_tril, unpack_tril

        grp, _ = _build(tmp_path, n_seg=2000, levels=3, widths=0.05, additive_lod=False)
        decoder = ArrayDecoder()
        lights = []
        for i in range(3):
            child = grp[f"child_{i}"]
            assert "amplitude_normalization_factor" not in child.attrs
            amplitudes = decoder.decode(child["amplitudes"], grp).astype(np.float64)
            diagonal = decoder.decode(child["cholesky_factors_diag"], grp).astype(
                np.float64
            )
            off_diagonal = decoder.decode(
                child["cholesky_factors_offdiag"], grp
            ).astype(np.float64)
            factors = unpack_tril(merge_tril(diagonal, off_diagonal, 3), 3)
            lights.append(float(np.sum(amplitudes * np.abs(np.linalg.det(factors)))))

        for light in lights[1:]:
            assert light == pytest.approx(lights[0], rel=0.03)

    def test_float_indices_rejected_before_partial_lod_write(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]],
            dtype=np.float32,
        )
        indices = np.array([0.9, 1.9, 2.9, 3.9], dtype=np.float64)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="integer array"):
                scene.add_lines(
                    "curves",
                    vertices,
                    1.0,
                    line_type="indexed",
                    indices=indices,
                    substitutive_lod=dict(
                        compression_factor=2,
                        levels=1,
                        device="cpu",
                        seed=0,
                    ),
                    additive_lod=False,
                )

        store = zarr.open(str(out), mode="r")
        assert "curves" not in store

    @pytest.mark.parametrize(
        ("indices", "message"),
        [
            (np.array([-1, 0, 2, 3]), r"Index -1 < 0"),
            (np.array([0, 4]), r"Index 4 >= n_vertices 4"),
        ],
    )
    def test_out_of_bounds_indices_rejected_before_partial_lod_write(
        self, tmp_path, indices, message
    ) -> None:
        out = tmp_path / "t.luxar.zarr"
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [10, 0, 0], [11, 0, 0]],
            dtype=np.float32,
        )
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match=message):
                scene.add_lines(
                    "curves",
                    vertices,
                    1.0,
                    line_type="indexed",
                    indices=indices,
                    substitutive_lod=dict(
                        compression_factor=2,
                        levels=1,
                        device="cpu",
                        seed=0,
                    ),
                    additive_lod=False,
                )

        store = zarr.open(str(out), mode="r")
        assert "curves" not in store


class TestSubstitutiveLinesComposedWithAdditive:
    """``additive_lod`` composes with ``substitutive_lod`` on Lines too.

    The Points twin carries the bulk of the assertions; these cover what is
    specific to Lines — whole-polyline levels, and the single-polyline line
    types where a ladder is a no-op rather than a win.
    """

    N_SEG = 1500

    @pytest.fixture(scope="class")
    def composed(self, tmp_path_factory) -> tuple:
        out = tmp_path_factory.mktemp("composed_lines") / "t.luxar.zarr"
        verts = _segments(self.N_SEG, seed=0)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "c",
                verts,
                0.8,
                line_type="segments",
                substitutive_lod=dict(
                    compression_factor=4, levels=2, device="cpu", seed=0
                ),
                additive_lod=dict(counts="stream:300", method="random", seed=0),
            )
        return zarr.open(str(out), mode="r")["c"], self.N_SEG * 2

    @staticmethod
    def _children(grp) -> list:
        return sorted(k for k in grp.keys() if k.startswith("child_"))

    def test_finest_lines_child_carries_a_ladder(self, composed) -> None:
        grp, n_vertices = composed
        finest = grp[self._children(grp)[-1]]

        assert finest.attrs["type"] == "lines"
        n_sub = int(finest.attrs["n_additive_sublods"])
        assert n_sub > 1
        assert (
            sum(int(finest[f"additive_{i}"].attrs["n_vertices"]) for i in range(n_sub))
            == n_vertices
        )

    def test_every_level_holds_whole_polylines(self, composed) -> None:
        # The invariant that makes a partial load renderable: a level must never
        # contain half a polyline, or its segment indices dangle.
        grp, _ = composed
        finest = grp[self._children(grp)[-1]]

        for i in range(int(finest.attrs["n_additive_sublods"])):
            sub = finest[f"additive_{i}"]
            n_v = int(sub.attrs["n_vertices"])
            assert n_v % 2 == 0, "segments polylines are vertex pairs"
            if "indices" in sub:
                idx = np.asarray(sub["indices"][:])
                assert idx.size == 0 or int(idx.max()) < n_v

    def test_lod_group_invariants_survive_composition(self, composed) -> None:
        grp, _ = composed
        children = self._children(grp)

        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        # Derived ladder → screen-area selector survives the composition.
        assert grp.attrs["selector"] == "screen-area"
        cf = [float(grp[c].attrs["coverage_fraction"]) for c in children]
        assert cf[0] == 0.0
        assert cf[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)
        assert all(a < b for a, b in zip(cf, cf[1:])), cf

    def test_every_level_is_energy_stamped(self, composed) -> None:
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

    @pytest.mark.parametrize("line_type", ["polyline", "loop"])
    def test_single_polyline_types_skip_the_ladder_cleanly(
        self, tmp_path, line_type
    ) -> None:
        # One polyline cannot be split without breaking segment topology, so the
        # ladder is suppressed BEFORE the builder runs — no UserWarning, no
        # error, and the substitutive group is still built normally.
        out = tmp_path / "t.luxar.zarr"
        verts = np.random.RandomState(0).normal(0, 20, (2000, 3)).astype(np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "c",
                    verts,
                    0.8,
                    line_type=line_type,
                    substitutive_lod=dict(
                        compression_factor=4, levels=2, device="cpu", seed=0
                    ),
                )

        grp = zarr.open(str(out), mode="r")["c"]
        finest = grp[sorted(k for k in grp.keys() if k.startswith("child_"))[-1]]
        assert finest.attrs["type"] == "lines"
        assert int(finest.attrs.get("n_additive_sublods", 1)) == 1


class TestSubstitutiveLinesIndexedVerifiesAdditive:
    """The ``indexed``-lines branch of ``add_lines_substitutive_lod_wrapper_impl``.

    The additive multi-LOD writer does not carry an edge list: it rebuilds one by
    chaining each connected component in ascending vertex order. That is faithful
    exactly when every component already IS an ascending simple path, so the
    branch TESTS the data rather than refusing ``line_type="indexed"`` outright —
    real tractography and streamline sets qualify and used to lose their ladder
    for nothing.

    Both arms are driven end-to-end through a real indexed geometry node, so this
    proves the branch in ``adders/lines.py`` actually fires; the
    ``test_lod_group.py`` tests only hand-feed a reason string into
    ``compose_additive_under_substitutive`` directly.
    """

    @staticmethod
    def _indexed_verts_and_edges(
        n_seg: int = 300, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """QUALIFYING data: disjoint 2-vertex components, each trivially a chain."""
        verts = _segments(n_seg, seed=seed)
        # One edge per vertex pair: [0, 1, 2, 3, …] — the same topology as
        # `segments`, expressed as an explicit integer edge list.
        indices = np.arange(verts.shape[0], dtype=np.int64)
        return verts, indices

    @staticmethod
    def _forked_verts_and_edges(
        n_seg: int = 300, seed: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """NON-QUALIFYING data: one component branches, so a chain would lie.

        Note it takes TWO extra edges to build a fork here. Adding only ``(1, 4)``
        to the disjoint pairs would join ``{0,1,4,5}`` into the path 0-1-4-5,
        whose edges ARE its consecutive-vertex pairs — safe, and correctly
        accepted. A fork needs a vertex of degree 3.
        """
        verts = _segments(n_seg, seed=seed)
        pairs = np.arange(verts.shape[0], dtype=np.int64).reshape(-1, 2)
        # Vertex 1 gains neighbours 2 AND 4 on top of 0, so its component
        # {0..5} is a Y. Edge (1, 4) spans three positions in ascending order,
        # which is exactly what the check rejects.
        fork = np.asarray([[1, 2], [1, 4]], dtype=np.int64)
        return verts, np.vstack([pairs, fork]).reshape(-1)

    @staticmethod
    def _assert_no_additive_ladder(grp) -> None:
        children = sorted(k for k in grp.keys() if k.startswith("child_"))
        assert children, "substitutive group must have children"
        for name in children:
            child = grp[name]
            assert int(child.attrs.get("n_additive_sublods", 1)) == 1, name
            assert not any(k.startswith("additive_") for k in child.keys()), name

    def test_chain_shaped_indexed_data_gets_its_ladder(self, tmp_path) -> None:
        # The positive arm, and the behaviour change: this topology is safe, so
        # the ladder is BUILT and no warning is emitted. Explicit `counts` are
        # needed because the composed default (stream:39062 vertices) would
        # collapse to one leaf at 600 vertices and prove nothing.
        out = tmp_path / "t.luxar.zarr"
        verts, indices = self._indexed_verts_and_edges()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "curves",
                    verts,
                    0.8,
                    line_type="indexed",
                    indices=indices,
                    substitutive_lod=dict(
                        compression_factor=2, levels=1, device="cpu", seed=0
                    ),
                    additive_lod=dict(counts=[100, 300], method="random", seed=0),
                )

        grp = zarr.open(str(out), mode="r")["curves"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        finest = grp[sorted(k for k in grp.keys() if k.startswith("child_"))[-1]]
        assert int(finest.attrs.get("n_additive_sublods", 1)) > 1
        assert any(k.startswith("additive_") for k in finest.keys())

    def test_explicit_additive_on_non_chain_warns_and_suppresses(
        self, tmp_path
    ) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts, indices = self._forked_verts_and_edges()
        with pytest.warns(UserWarning, match="not a simple path in ascending"):
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "curves",
                    verts,
                    0.8,
                    line_type="indexed",
                    indices=indices,
                    substitutive_lod=dict(
                        compression_factor=2, levels=1, device="cpu", seed=0
                    ),
                    additive_lod=True,
                )

        grp = zarr.open(str(out), mode="r")["curves"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        self._assert_no_additive_ladder(grp)

    def test_default_additive_on_non_chain_is_suppressed_quietly(
        self, tmp_path
    ) -> None:
        # With additive_lod omitted, an unsafe node's ladder is skipped QUIETLY
        # (an aprint info line, NOT a UserWarning). The load-bearing assertion
        # here is the ABSENCE of a UserWarning (enforced by simplefilter below).
        out = tmp_path / "t.luxar.zarr"
        verts, indices = self._forked_verts_and_edges()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "curves",
                    verts,
                    0.8,
                    line_type="indexed",
                    indices=indices,
                    substitutive_lod=dict(
                        compression_factor=2, levels=1, device="cpu", seed=0
                    ),
                )

        grp = zarr.open(str(out), mode="r")["curves"]
        assert grp.attrs["kind"] == "lod"
        assert grp.attrs["display_type"] == "lines"
        self._assert_no_additive_ladder(grp)

    def test_direct_additive_path_raises_on_non_chain(self, tmp_path) -> None:
        # WITHOUT a substitutive wrapper there is no level to fall back to, and
        # this path previously had no guard at all: it fabricated a chain per
        # component and wrote a scene whose edges were quietly wrong. It must
        # raise, not warn.
        out = tmp_path / "t.luxar.zarr"
        verts, indices = self._forked_verts_and_edges()
        with pytest.raises(ValueError, match="simple path in ascending vertex order"):
            with LuxarZarrCompiler(out) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_lines(
                    "curves",
                    verts,
                    0.8,
                    line_type="indexed",
                    indices=indices,
                    additive_lod=dict(counts=[100, 300], method="random", seed=0),
                )

    def test_direct_additive_path_ladders_chain_shaped_data(self, tmp_path) -> None:
        out = tmp_path / "t.luxar.zarr"
        verts, indices = self._indexed_verts_and_edges()
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "curves",
                verts,
                0.8,
                line_type="indexed",
                indices=indices,
                additive_lod=dict(counts=[100, 300], method="random", seed=0),
            )

        grp = zarr.open(str(out), mode="r")["curves"]
        assert any(k.startswith("additive_") for k in grp.keys())

    def test_real_tractography_index_layout_qualifies(self) -> None:
        # The third arm: the layout the demos actually build, straight from the
        # demo's own index builder rather than a hand-written fixture.
        from luxar.core.group.lod.lines import indexed_components_are_chains
        from luxar.demos.demo_dmri_tractography import polyline_segment_indices

        n_paths, n_steps = 40, 28
        idx = polyline_segment_indices(n_paths, n_steps)
        assert indexed_components_are_chains(
            n_paths * n_steps, np.asarray(idx, dtype=np.intp).reshape(-1, 2)
        )


class TestAdditiveLevelStatsPairingLines:
    """A caller-supplied ``level_stats`` must not break the energy pairing.

    The Lines twin of ``TestAdditiveLevelStatsPairing`` in the Points file.
    ``level_stats`` is a free-form key a caller threads through the public
    ``add_lines(**attrs)``. When the additive ladder is fraction-stamped, the
    parent must still receive the paired ``reference_energy`` (both-or-neither),
    while the caller's own keys survive intact. This exercises the ``elif``
    merge branch in ``add_lines_multi_lod_wrapper_impl``.
    """

    def test_caller_level_stats_without_reference_energy_gets_paired(
        self, tmp_path
    ) -> None:
        out = tmp_path / "t.luxar.zarr"
        # 1500 segment-polylines (3000 vertices) + positive widths gives a real
        # tube-volume energy, and stream:300 splits it into >1 sub-LOD.
        verts = _segments(1500, seed=0)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "curves",
                verts,
                0.8,
                line_type="segments",
                additive_lod=dict(counts="stream:300", method="random", seed=0),
                level_stats={"caller_key": 123},
            )

        grp = zarr.open(str(out), mode="r")["curves"]
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
        verts = _segments(1500, seed=0)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_lines(
                "curves",
                verts,
                0.8,
                line_type="segments",
                additive_lod=dict(counts="stream:300", method="random", seed=0),
                level_stats={"caller_key": 123, "reference_energy": float("nan")},
            )

        grp = zarr.open(str(out), mode="r")["curves"]
        stored = dict(grp.attrs["level_stats"])
        assert stored["caller_key"] == 123
        assert np.isfinite(stored["reference_energy"])
        assert stored["reference_energy"] > 0


# ────────────────────────────────────────────────────────────────────────
# Partition-bound anchor (hand-built kind=partition of per-part ladders)
# ────────────────────────────────────────────────────────────────────────


def _lines_ladder_coverage(
    tmp_path, store_name, *, n_seg=250, levels=2, partitioned, wrap_in_group=False, **kw
):
    """Build TWO Lines ladders — a real 2-tile partition, or the same pair at root.

    Returns one per-child ``coverage_fraction`` list per part, coarsest→finest.
    Two parts on purpose: a ONE-part ``kind=partition`` is the degenerate shape
    ``partitioned_coverage_fractions`` documents as 4x too coarse (its "tile" is
    the whole object), so it must not be the fixture that motivates the rule.

    Both variants get byte-identical per-part geometry (the segment list is split
    on whole-segment boundaries), so their per-level bead counts — and therefore
    the derived ladders before anchoring — are identical; the only thing that can
    differ is which anchor was chosen.

    ``method="kmeans_lloyd"`` instead of the default ``"auto"``: at these sizes
    ``auto`` routes to the submodular ``greedy`` Runnalls path, whose sparse-Gram
    build scales with OVERLAP DENSITY — the worst case for a lifted bead string —
    and costs tens of seconds. Every assertion here reads per-level COUNTS
    (``compression_factor``/``levels`` decide those, identically for either method)
    and never reduction quality, so the cheap O(N log N) reduction is sound — the
    same reasoning as ``_acceptance_params`` in ``test_gsplats.py``. Pre-existing
    tests keep ``auto``.
    """
    out = tmp_path / store_name
    verts = _segments(n_seg)
    half = n_seg // 2  # split on a whole-SEGMENT boundary (2 vertices each)
    halves = [verts[: 2 * half], verts[2 * half :]]
    spec = dict(levels=levels, method="kmeans_lloyd", device="cpu", seed=0, **kw)
    with LuxarZarrCompiler(out) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        if partitioned:
            wrapper = scene.add_partition_group(
                "tiled", display_type="lines", max_elements=half
            )
        else:
            wrapper = scene
        for i, part_verts in enumerate(halves):
            target = (
                wrapper.add_group(f"holder_{i}")
                if (partitioned and wrap_in_group)
                else wrapper
            )
            target.add_lines(f"part_{i}", part_verts, 0.8, substitutive_lod=spec)

    root = zarr.open(str(out), mode="r")
    node = root["tiled"] if partitioned else root
    out_lists = []
    for i in range(len(halves)):
        holder = node[f"holder_{i}"] if (partitioned and wrap_in_group) else node
        lod = holder[f"part_{i}"]
        assert lod.attrs["kind"] == "lod"
        names = sorted(
            (k for k in lod.keys() if k.startswith("child_")),
            key=lambda k: int(k.split("_")[1]),
        )
        out_lists.append([float(lod[k].attrs["coverage_fraction"]) for k in names])
    return out_lists


class TestPartitionBoundAnchorLines:
    """A hand-built partition of per-part Lines ladders gets the fills-screen anchor.

    Before ``derive_coverage_fractions`` detected the ``kind=partition`` ancestor,
    this path always auto-derived the WHOLE-OBJECT ladder, so every tile sat on
    its finest level at the opening whole-object framing.

    Three tests here REGRESS without the fix
    (``test_partition_ladder_is_the_root_ladder_times_the_ceiling``,
    ``test_finest_is_exactly_the_ceiling``,
    ``test_plain_group_between_partition_and_ladder_still_anchored``). The other two
    are CONTROLS that pass either way and pin what must NOT change.

    The fixture is a real TWO-tile partition, and every assertion covers BOTH
    parts. The Lines ladder is keyed on the lifted BEAD count, which is not on
    disk, so the expected lists are obtained by building the identical per-part
    geometry at the scene root and rescaling by ×2 (exact — a power of two) onto
    the tile anchor ``PARTITION_FINEST_AREA`` = 1.0.
    """

    def test_partition_ladders_are_the_root_ladders_times_the_ceiling(
        self, tmp_path
    ) -> None:
        root = _lines_ladder_coverage(tmp_path, "root.luxar.zarr", partitioned=False)
        tiled = _lines_ladder_coverage(tmp_path, "tiled.luxar.zarr", partitioned=True)
        assert len(tiled) == len(root) == 2, "must be a real 2-tile partition"
        for i, (tile, whole) in enumerate(zip(tiled, root)):
            assert len(tile) >= 3
            # fills-screen anchor = whole-object ladder × (tile / whole-object
            # anchor) = ×2 in area units.
            assert tile == pytest.approx(
                [
                    f * (PARTITION_FINEST_AREA / WHOLE_OBJECT_FINEST_ANCHOR)
                    for f in whole
                ]
            ), f"part_{i}"

    def test_finest_is_exactly_the_ceiling_for_every_part(self, tmp_path) -> None:
        tiled = _lines_ladder_coverage(tmp_path, "tiled.luxar.zarr", partitioned=True)
        assert len(tiled) == 2
        for i, tile in enumerate(tiled):
            assert tile[-1] == pytest.approx(PARTITION_FINEST_AREA), f"part_{i}"

    def test_scene_root_still_gets_the_whole_object_anchor(self, tmp_path) -> None:
        """CONTROL: no partition ancestor → finest stays at the half-screen-area
        whole-object anchor (0.5) — the over-trigger guard."""
        root = _lines_ladder_coverage(tmp_path, "root.luxar.zarr", partitioned=False)
        assert len(root) == 2
        for i, whole in enumerate(root):
            assert whole[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR), f"part_{i}"

    def test_plain_group_between_partition_and_ladder_still_anchored(
        self, tmp_path
    ) -> None:
        tiled = _lines_ladder_coverage(
            tmp_path, "tiled.luxar.zarr", partitioned=True, wrap_in_group=True
        )
        assert len(tiled) == 2
        for i, tile in enumerate(tiled):
            assert tile[-1] == pytest.approx(PARTITION_FINEST_AREA), f"part_{i}"

    def test_explicit_coverage_fractions_still_win_under_a_partition(
        self, tmp_path
    ) -> None:
        """CONTROL (passes pre-fix): an explicit list must keep winning verbatim."""
        explicit = [0.0, 3.52, 4.0]
        tiled = _lines_ladder_coverage(
            tmp_path,
            "tiled.luxar.zarr",
            partitioned=True,
            coverage_fractions=explicit,
        )
        assert len(tiled) == 2
        for i, tile in enumerate(tiled):
            assert tile == pytest.approx(explicit), f"part_{i}"


class TestSubstitutiveLinesGuards:
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

    def test_false_substitutive_lod_does_not_block_partition(self, tmp_path) -> None:
        out = tmp_path / "partition.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "c",
                verts,
                1.0,
                line_type="segments",
                partition=dict(max_elements=10),
                substitutive_lod=False,
            )

        assert node.attrs["kind"] == "partition"

    def test_false_partition_does_not_block_substitutive_lod(self, tmp_path) -> None:
        out = tmp_path / "lod.luxar.zarr"
        verts = _segments(50)
        with LuxarZarrCompiler(out) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            node = scene.add_lines(
                "c",
                verts,
                1.0,
                line_type="segments",
                partition=False,
                substitutive_lod=dict(levels=1, device="cpu", seed=0),
            )

        assert node.attrs["kind"] == "lod"

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

    def test_thresholds_are_count_currency_independent(self, tmp_path) -> None:
        # Screen-occupancy halving reads only the ladder LENGTH, so the old
        # bead-vs-vertex count-currency question cannot reach the thresholds:
        # any 3-level lines ladder pins the same halving list.
        grp, _n_verts = _build(tmp_path, levels=2)
        n = len(sorted(k for k in grp.keys() if k.startswith("child_")))
        cf = [float(grp[f"child_{i}"].attrs["coverage_fraction"]) for i in range(n)]
        # Area halving: [0, 0.25, 0.5] — /2 per coarser level from the
        # half-screen anchor.
        assert cf == pytest.approx(
            [0.0, WHOLE_OBJECT_FINEST_ANCHOR / 2.0, WHOLE_OBJECT_FINEST_ANCHOR]
        )


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
