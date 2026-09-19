"""Decimation must produce a correct coarser SURFACE, not merely a smaller array.

Every assertion here is about a property the renderer or the LOD group depends on:
in-range indices (or the writer rejects it), no NaN (or the shader draws nothing),
monotonically falling counts (or `coverage_fractions` is non-monotonic), and
geometry that still occupies the same space as the original (or the level is not a
stand-in for anything).
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from .. import qem
from ..decimate import (
    _ORPHAN_DISTANCE_BLOCK_PAIR_BUDGET,
    _ORPHAN_NEAREST_PAIR_BUDGET,
    QEM_AUTO_VERTEX_LIMIT,
    _normalized_collapse_error,
    decimate,
    decimate_cluster,
    decimate_ladder,
    resolve_decimation_method,
)
from ..qem import (
    _edge_target,
    _face_quadrics,
    _solve_system,
    decimate_qem,
    decimate_qem_ladder,
)


def octasphere(subdivisions: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A closed, boundary-free, manifold sphere: chi=2 with no special cases.

    Built by subdividing an octahedron and renormalizing, so the ground truth is
    exact — every vertex is at radius 1 — which is what lets the tests below assert
    that decimation preserves the SHAPE and not just the vertex count.
    """
    v = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float
    )
    f = np.array(
        [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
         [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]], np.uint32,
    )  # fmt: skip
    for _ in range(subdivisions):
        mid: dict[tuple[int, int], int] = {}
        nv = list(v)
        nf = []

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in mid:
                p = nv[a] + nv[b]
                mid[key] = len(nv)
                nv.append(p / np.linalg.norm(p))
            return mid[key]

        for tri in f:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            nf += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        v = np.array(nv)
        f = np.array(nf, np.uint32)
    return v.astype(np.float32), f


def edge_audit(n_vertices: int, faces: np.ndarray) -> tuple[int, int, int]:
    """``(unique_edges, boundary_edges, nonmanifold_edges)`` counted for real.

    Deliberately not the ``E = 3F/2`` shortcut: that identity only holds for a
    closed manifold, so using it to TEST for one assumes the conclusion. It read
    chi=32 on a level that actually has chi=2.
    """
    e = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
    )
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return len(uniq), int((counts == 1).sum()), int((counts > 2).sum())


class TestDecimateCluster:
    def test_attribute_barrier_prevents_boundary_smearing(self) -> None:
        v, f = octasphere(3)
        side = v[:, 0] + 0.31 * v[:, 1] >= 0
        colors = np.zeros((len(v), 3), np.uint8)
        colors[side, 0] = 255
        colors[~side, 2] = 255

        default = decimate_cluster(v, f, target_vertices=40, colors=colors)
        preserved = decimate_cluster(
            v, f, target_vertices=40, colors=colors, attribute_weight=1.0
        )

        assert default.colors is not None and preserved.colors is not None
        assert np.any((default.colors[:, 0] > 0) & (default.colors[:, 2] > 0))
        assert not np.any((preserved.colors[:, 0] > 0) & (preserved.colors[:, 2] > 0))
        edges, boundary, nonmanifold = edge_audit(
            len(preserved.vertices), preserved.faces
        )
        assert len(preserved.vertices) - edges + len(preserved.faces) == 2
        assert boundary == 0 and nonmanifold == 0

    def test_reports_normalized_collapse_error(self) -> None:
        v, f = octasphere(3)
        coarse = decimate_cluster(v, f, target_vertices=40)
        unchanged = decimate_cluster(v, f, target_vertices=len(v))

        assert 0.0 < coarse.geometric_error < 1.0
        assert unchanged.geometric_error == 0.0

    def test_orphan_vertices_use_their_own_nearest_coarse_vertex(
        self, monkeypatch
    ) -> None:
        module = importlib.import_module("luxar.mesh.decimate")
        monkeypatch.setattr(module, "_ORPHAN_DISTANCE_BLOCK_PAIR_BUDGET", 2)
        source = np.array([[-5, 0, 0], [5, 0, 0]], dtype=np.float64)
        output = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)
        inverse = np.array([-1, -1], dtype=np.int64)

        error = _normalized_collapse_error(source, output, inverse, (0, 1, 2))

        assert error == pytest.approx(0.4)

    def test_large_orphan_search_uses_a_conservative_shared_representative(
        self, monkeypatch
    ) -> None:
        module = importlib.import_module("luxar.mesh.decimate")
        monkeypatch.setattr(module, "_ORPHAN_NEAREST_PAIR_BUDGET", 3)
        source = np.array([[-5, 0, 0], [5, 0, 0]], dtype=np.float64)
        output = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)
        inverse = np.array([-1, -1], dtype=np.int64)

        error = _normalized_collapse_error(source, output, inverse, (0, 1, 2))

        assert error == pytest.approx(0.6)

    def test_many_disconnected_components_bound_orphan_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        n_islands = 5_000
        offsets = np.random.default_rng(7).random((n_islands, 3), dtype=np.float32)
        tetra = np.array(
            [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
            dtype=np.float32,
        )
        vertices = (offsets[:, None, :] + tetra[None, :, :]).reshape(-1, 3)
        tetra_faces = np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.uint32
        )
        faces = (
            tetra_faces[None, :, :]
            + (4 * np.arange(n_islands, dtype=np.uint32))[:, None, None]
        ).reshape(-1, 3)
        norm_sizes: list[int] = []
        norm = np.linalg.norm

        def recording_norm(values: Any, *args: Any, **kwargs: Any) -> Any:
            norm_sizes.append(np.asarray(values).size)
            return norm(values, *args, **kwargs)

        monkeypatch.setattr(np.linalg, "norm", recording_norm)

        coarse = decimate_cluster(vertices, faces, target_vertices=len(vertices) // 4)

        assert 0 < len(coarse.vertices) < len(vertices)
        assert 0.0 < coarse.geometric_error < 1.0
        assert max(norm_sizes) <= 3 * _ORPHAN_DISTANCE_BLOCK_PAIR_BUDGET
        assert sum(norm_sizes) <= 2 * _ORPHAN_NEAREST_PAIR_BUDGET

    def test_a_ladder_of_levels_is_strictly_coarser_and_still_a_sphere(self) -> None:
        v, f = octasphere(4)
        previous = len(v)
        for target in (400, 150, 40):
            r = decimate_cluster(v, f, target_vertices=target)

            assert len(r.vertices) < previous, "each level must be strictly coarser"
            previous = len(r.vertices)
            assert len(r.faces) > 0
            assert int(r.faces.max()) < len(r.vertices), "face index out of range"
            assert np.isfinite(r.vertices).all(), "NaN/inf renders as nothing"
            # Still a sphere: every representative sits on the unit surface. A
            # decimator that merely returned fewer vertices (say, a prefix) would
            # keep radius 1 too, which is why the count and topology assertions
            # above carry the rest of the weight.
            radii = np.linalg.norm(r.vertices, axis=1)
            assert 0.9 < radii.min() and radii.max() < 1.15

    def test_euler_characteristic_survives(self) -> None:
        # chi = V - E + F = 2 for any closed surface of genus 0. Preserving it means
        # the coarse level is topologically still a sphere rather than a bag of
        # disconnected shards that happens to have the right vertex count.
        v, f = octasphere(4)
        for target in (400, 150, 40):
            r = decimate_cluster(v, f, target_vertices=target)
            edges, _, _ = edge_audit(len(r.vertices), r.faces)
            assert len(r.vertices) - edges + len(r.faces) == 2

    def test_duplicate_triangles_are_removed(self) -> None:
        # Two fine faces can collapse onto the same three representatives. Keeping
        # both corrupts the Euler characteristic and z-fights; this pins the removal
        # so a future "simplification" cannot quietly drop it.
        v, f = octasphere(4)
        r = decimate_cluster(v, f, target_vertices=150)
        canonical = np.sort(r.faces, axis=1)
        assert len(np.unique(canonical, axis=0)) == len(canonical)

    def test_no_degenerate_triangles(self) -> None:
        v, f = octasphere(3)
        r = decimate_cluster(v, f, target_vertices=60)
        a, b, c = r.faces[:, 0], r.faces[:, 1], r.faces[:, 2]
        assert np.all((a != b) & (b != c) & (a != c))

    def test_normals_are_recomputed_from_the_COARSE_surface(self) -> None:
        """Not averaged from the fine one, which would describe the wrong surface."""
        v, f = octasphere(4)
        r = decimate_cluster(
            v, f, target_vertices=150, normals=np.zeros_like(v), normal_dims=(0, 1, 2)
        )
        assert r.normals is not None
        np.testing.assert_allclose(np.linalg.norm(r.normals, axis=1), 1.0, atol=1e-4)
        # On a sphere the outward normal IS the position, so agreement here shows the
        # normals describe the coarse geometry. Averaging the (zero) input normals
        # would have produced zeros, and passing them through would fail the unit
        # check above — two different bugs, both caught.
        #
        # Compared on the MEDIAN, not the mean, and deliberately so. Clustering
        # leaves a few degree-1 corners, and a vertex with one incident face
        # correctly takes that face's normal — which on a curved surface is roughly
        # perpendicular to the radius. Those are honest values, not errors, but they
        # drag a mean below any threshold worth asserting. The degree-1 population is
        # bounded separately below, where it belongs.
        cosine = np.einsum(
            "ij,ij->i",
            r.normals,
            r.vertices / np.linalg.norm(r.vertices, axis=1, keepdims=True),
        )
        assert float(np.median(cosine)) > 0.98

    def test_ragged_degree_one_corners_stay_a_small_minority(self) -> None:
        """Clustering leaves dangling corners; this bounds how many.

        A degree-1 vertex is a triangle corner nothing else touches — the visible
        edge of the non-manifold region clustering can produce. They are legal and
        they render, but a level that was mostly dangles would be a bad stand-in for
        the fine surface, and nothing else in this file would notice.
        """
        v, f = octasphere(4)
        r = decimate_cluster(v, f, target_vertices=150)
        degree = np.bincount(r.faces.reshape(-1), minlength=len(r.vertices))
        assert int((degree == 0).sum()) == 0, "unreferenced vertices must be compacted"
        assert (degree == 1).mean() < 0.15

    def test_colors_are_averaged_within_each_cluster(self) -> None:
        v, f = octasphere(3)
        # Northern hemisphere red, southern blue: after clustering, representatives
        # near the equator should blend and the poles should stay pure.
        colors = np.where(
            (v[:, 2] > 0)[:, None], np.array([255, 0, 0]), np.array([0, 0, 255])
        ).astype(np.uint8)
        r = decimate_cluster(v, f, target_vertices=80, colors=colors)
        assert r.colors is not None and r.colors.shape == (len(r.vertices), 3)
        north = r.vertices[:, 2] > 0.8
        assert int(r.colors[north][:, 0].min()) > 200, "the pole must stay red"

    @pytest.mark.parametrize(
        "dtype, full",
        [(np.uint8, 255.0), (np.uint16, 65535.0), (np.float32, 1.0)],
    )
    def test_colors_come_back_in_their_INPUT_dtype(self, dtype, full) -> None:
        """Mesh colours are uint8, uint16 OR float32, and all three must survive.

        The averaging is dtype-agnostic float64, so only the cast at the end knew
        anything about the input — and it was hardcoded to uint8 with a [0, 255]
        clip. Float32 SDR colours in [0, 1] truncated to 0, so every coarse level
        rendered BLACK against a coloured finest one; uint16 clipped at 255, which
        on a 65535 scale is black too. Both are silent: the arrays are the right
        shape and the level loads fine.

        The uint8 row is a BASELINE, not coverage: it passed before the fix too.
        Only the uint16 and float32 rows discriminate.
        """
        v, f = octasphere(3)
        colors = np.zeros((len(v), 3), dtype)
        colors[:, 0] = dtype(0.8 * full)
        r = decimate_cluster(v, f, target_vertices=80, colors=colors)
        assert r.colors is not None
        assert r.colors.dtype == dtype, "the level must not change the colour scale"
        # A constant field: every cluster mean is the exact input value, so this
        # is an equality and not a tolerance. Rounding bias has nowhere to hide.
        assert float(r.colors[:, 0].min()) == pytest.approx(0.8 * full, abs=1e-6)
        assert float(r.colors[:, 0].max()) == pytest.approx(0.8 * full, abs=1e-6)
        assert float(r.colors[:, 1].max()) == 0.0, "an untouched channel stays 0"

    def test_an_HDR_float_colour_is_not_clipped(self) -> None:
        # The other half of the dtype rule: an integer dtype clips to its own
        # range, a float one does not clip at all, because a float colour above
        # 1.0 is legitimate HDR data (docs/guides/user/HDR_GUIDE.md) and clipping
        # it would quietly tone-map the coarse levels only.
        v, f = octasphere(3)
        colors = np.full((len(v), 3), 3.5, np.float32)
        r = decimate_cluster(v, f, target_vertices=80, colors=colors)
        assert r.colors is not None
        assert float(r.colors.max()) == pytest.approx(3.5, abs=1e-6)

    def test_colors_and_scalars_are_MEANS_not_a_representative_pick(self) -> None:
        """The discriminator between averaging and picking one member per cluster.

        Every other assertion here (dtype, range, compaction, a pure pole) holds
        for a decimator that simply keeps one vertex's value per cluster. A
        two-valued field is what separates them: a cluster straddling the two
        populations must come back with a THIRD value that is in neither input.
        """
        v, f = octasphere(3)
        northern = v[:, 2] > 0
        scalars = northern.astype(np.float32)
        colors = np.zeros((len(v), 3), np.uint8)
        colors[northern, 0] = 255
        r = decimate_cluster(v, f, target_vertices=80, colors=colors, scalars=scalars)
        assert r.scalars is not None and r.colors is not None
        blended_scalars = (r.scalars > 1e-6) & (r.scalars < 1.0 - 1e-6)
        assert int(blended_scalars.sum()) > 0, (
            f"no cluster blended the two scalar values: {np.unique(r.scalars)}"
        )
        red = r.colors[:, 0].astype(np.int64)
        assert int(((red > 0) & (red < 255)).sum()) > 0, (
            f"no cluster blended the two colours: {np.unique(red)}"
        )

    def test_an_integer_cluster_mean_ROUNDS_for_colours(self) -> None:
        """Round, not truncate — and the difference is a whole LSB per cluster.

        Exercised on the helper directly because it needs a cluster of known
        composition: two of three members at 1 averages to 0.667, which rounds
        to 1 and truncates to 0. A grid cannot be asked for that cluster.
        """
        from ..decimate import _average_per_cluster

        values = np.array([[0], [1], [1]], np.uint8)
        inverse = np.zeros(3, np.int64)
        counts = np.array([3.0])
        out = _average_per_cluster(values, inverse, counts, 1, quantize=True)
        assert out.dtype == np.uint8
        assert int(out[0, 0]) == 1, "truncation would give 0"

    def test_integer_SCALARS_keep_their_fractional_mean(self) -> None:
        """Scalars are not quantized by their input dtype; colours are.

        `write_scalars` casts to float32 unconditionally, so an integer scalars
        array never reaches disk as an integer — rounding it would only throw the
        cluster mean away, and `np.rint` is half-to-even, so a 0/1 field
        averaging to 0.5 would round DOWN to 0. The coarse levels would render
        hard-classified against a blended finest level: the pop again.
        """
        v, f = octasphere(3)
        scalars = (v[:, 2] > 0).astype(np.int32)
        r = decimate_cluster(v, f, target_vertices=80, scalars=scalars)
        assert r.scalars is not None
        assert r.scalars.dtype == np.float32, "float32 is the on-disk dtype anyway"
        fractional = (r.scalars > 1e-6) & (r.scalars < 1.0 - 1e-6)
        assert int(fractional.sum()) > 0, (
            f"integer scalars were re-quantized: {np.unique(r.scalars)}"
        )

    def test_scalars_are_averaged_and_compacted_alongside_colors(self) -> None:
        """A coarse level without scalars carries a colormap it cannot use.

        `colormap` rides on every child of the ladder, and the viewer maps only
        when `has_scalars` — so a level that dropped them renders unmapped while
        the finest renders mapped, which is a visible pop at every LOD switch.
        """
        v, f = octasphere(3)
        scalars = v[:, 2].astype(np.float32)  # z, exactly [-1, 1] on the sphere
        colors = np.zeros((len(v), 3), np.uint8)
        r = decimate_cluster(v, f, target_vertices=80, colors=colors, scalars=scalars)
        assert r.scalars is not None
        # Compacted with the vertices, not left at the pre-compaction cluster
        # count — a length mismatch is a hard write-side rejection.
        assert r.scalars.shape == (len(r.vertices),)
        assert r.scalars.dtype == np.float32
        # A mean of values in [-1, 1] stays in [-1, 1]; nothing may be invented.
        assert float(r.scalars.min()) >= -1.0 - 1e-6
        assert float(r.scalars.max()) <= 1.0 + 1e-6
        # And the field still tracks the geometry: the north pole keeps a high z.
        north = r.vertices[:, 2] > 0.8
        assert float(r.scalars[north].min()) > 0.7

    def test_a_uniform_scalar_is_passed_through_untouched(self) -> None:
        # Not per-vertex data: it applies to every vertex, so there is nothing to
        # merge and the coarse level must carry the same value.
        v, f = octasphere(3)
        assert decimate_cluster(v, f, target_vertices=80, scalars=0.25).scalars == 0.25

    def test_a_barrier_dimension_never_merges(self) -> None:
        """Two timepoints at identical positions must stay separate vertices.

        This is the mesh analog of `--coarsen-dims`: a 4th column that is categorical
        must act as a hard barrier, or a coarse level would blend two timepoints into
        one surface that existed at neither.
        """
        v3, f = octasphere(3)
        v = np.concatenate(
            [np.hstack([v3, np.zeros((len(v3), 1), np.float32)]),
             np.hstack([v3, np.ones((len(v3), 1), np.float32)])]
        )  # fmt: skip
        faces = np.concatenate([f, f + len(v3)]).astype(np.uint32)
        r = decimate_cluster(v, faces, target_vertices=120, spatial_dims=(0, 1, 2))
        t = r.vertices[:, 3]
        assert set(np.unique(t).tolist()) == {0.0, 1.0}, "timepoints must not blend"
        # And each timepoint keeps its own surface rather than one being absorbed.
        assert (t == 0).sum() > 10 and (t == 1).sum() > 10

    def test_input_already_below_target_is_returned_unchanged(self) -> None:
        v, f = octasphere(1)
        scalars = v[:, 0].astype(np.float32)
        r = decimate_cluster(v, f, target_vertices=10_000, scalars=scalars)
        np.testing.assert_array_equal(r.vertices, v)
        np.testing.assert_array_equal(r.faces, f)
        # The early return must forward the optional channels too, or a level
        # that happened to need no reduction would lose its scalars.
        np.testing.assert_array_equal(r.scalars, scalars)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"target_vertices": 2}, "at least 4"),
            ({"target_vertices": 100, "spatial_dims": (0, 9)}, "out of range"),
        ],
    )
    def test_bad_arguments_are_named(self, kwargs: dict, match: str) -> None:
        v, f = octasphere(2)
        with pytest.raises(ValueError, match=match):
            decimate_cluster(v, f, **kwargs)

    def test_a_faceless_input_is_refused(self) -> None:
        # Without this the 50 vertices come back as ~8 orphan representatives that
        # no face references — a "surface" with nothing to draw, which then blows up
        # downstream in coverage_fractions with no clue which mesh caused it.
        v = np.zeros((50, 3), np.float32)
        with pytest.raises(ValueError, match="no faces"):
            decimate_cluster(v, np.zeros((0, 3), np.uint32), target_vertices=10)

    def test_coincident_vertices_are_refused_not_silently_emptied(self) -> None:
        """Every triangle collapsing must raise, not return a face-less level.

        Returning it would satisfy "a DecimatedMesh came back" while producing a
        level a kind=lod group cannot use at all: `coverage_fractions` raises on a
        zero-count level, so the failure would surface later and further away.
        """
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.uint32)
        with pytest.raises(ValueError, match="no surface|degenerate"):
            decimate_cluster(np.zeros((9, 3), np.float32), faces, target_vertices=4)

    def test_collinear_input_keeps_its_index_valid_faces(self) -> None:
        """The refusal is INDEX degeneracy, deliberately not zero AREA.

        Collinear vertices stay distinct at a fine enough spacing, so their faces
        survive with three different corners and zero area. That is left alone on
        purpose: nothing downstream breaks (a zero-area triangle rasterizes to
        nothing, which is the truth about a line), and an area threshold would be a
        new way to wrongly reject legitimately thin geometry.
        """
        v = np.array([[i, 0, 0] for i in range(9)], np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.uint32)
        r = decimate_cluster(v, faces, target_vertices=4)
        assert len(r.faces) > 0
        assert all(len(set(t.tolist())) == 3 for t in r.faces)

    def test_every_returned_level_has_at_least_one_triangle(self) -> None:
        v, f = octasphere(4)
        for target in (2000, 400, 100, 20, 4):
            assert len(decimate_cluster(v, f, target_vertices=target).faces) > 0

    def test_the_search_does_not_burn_its_whole_iteration_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each bisection step is a full O(V log V) reclustering, so passes cost.

        A small target cannot be hit within the 10% window a grid allows (20
        vertices permits a window of 2), so before the bracket-convergence exit this
        ran all 24 iterations and returned exactly the same mesh as 12.
        """
        # ``luxar.mesh.decimate`` the ATTRIBUTE is the re-exported function; the
        # module object is only reachable through ``sys.modules``.
        module = importlib.import_module("luxar.mesh.decimate")

        calls = {"n": 0}
        original = module._cluster_once

        def counting(*args: object, **kwargs: object) -> object:
            calls["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "_cluster_once", counting)
        v, f = octasphere(4)
        decimate_cluster(v, f, target_vertices=20)
        assert calls["n"] < 24, f"search used the full budget ({calls['n']} passes)"

    def test_out_of_range_face_index_is_refused(self) -> None:
        v, f = octasphere(2)
        bad = f.copy()
        bad[0, 0] = len(v) + 5
        with pytest.raises(ValueError, match="out of range"):
            decimate_cluster(v, bad, target_vertices=20)


def test_the_cell_search_is_robust_to_non_monotone_cluster_counts() -> None:
    """The search is a heuristic, and its RESULT is validated rather than assumed.

    A comment here once justified bisection by claiming the occupied-cell count
    falls monotonically as `cell` grows. It does not: `floor(p / cell)` grids at
    different spacings are not nested, so 10.1 and 10.9 share a cell at 2.0 (both
    floor to 5) and split at 2.1 (4 and 5) — a COARSER grid yielding MORE
    clusters. Bisection can therefore skip an interval holding a tighter fit.

    What must survive that is the CONTRACT, not the tightness: whatever comes back
    is a real surface with at least `target_vertices` vertices. This asserts the
    contract across a sweep of targets, on geometry with the near-coincident
    coordinate pairs that make the grids disagree.
    """
    v, f = octasphere(4)
    # Nudge one axis so many vertex pairs sit close enough to straddle a cell
    # boundary at some spacings and share one at others.
    v = v.copy()
    v[::3, 0] += 1e-3

    for target in (20, 50, 120, 300, 700):
        r = decimate_cluster(v, f, target_vertices=target)
        assert r.vertices.shape[0] >= target, (
            f"target {target}: got {r.vertices.shape[0]} vertices — the search may "
            "return a LOOSER fit than optimal, but never one below the target"
        )
        assert r.faces.shape[0] > 0, f"target {target}: no surviving triangle"
        assert int(r.faces.max()) < r.vertices.shape[0], (
            f"target {target}: face index out of range"
        )


class TestDecimateQEM:
    def test_attribute_quadric_preserves_a_colour_boundary(self) -> None:
        v, f = octasphere(3)
        side = v[:, 0] + 0.31 * v[:, 1] >= 0
        colors = np.zeros((len(v), 3), np.uint8)
        colors[side, 0] = 255
        colors[~side, 2] = 255

        default = decimate_qem(v, f, target_vertices=40, colors=colors)
        preserved = decimate_qem(
            v, f, target_vertices=40, colors=colors, attribute_weight=1.0
        )

        assert default.colors is not None and preserved.colors is not None
        assert np.any((default.colors[:, 0] > 0) & (default.colors[:, 2] > 0))
        assert not np.any((preserved.colors[:, 0] > 0) & (preserved.colors[:, 2] > 0))
        edges, boundary, nonmanifold = edge_audit(
            len(preserved.vertices), preserved.faces
        )
        assert len(preserved.vertices) - edges + len(preserved.faces) == 2
        assert boundary == 0 and nonmanifold == 0
        assert 0.0 < preserved.geometric_error < 1.0

    def test_mid_weight_stays_in_the_partial_colour_tradeoff_regime(self) -> None:
        v, f = octasphere(3)
        side = v[:, 0] + 0.31 * v[:, 1] >= 0
        colors = np.zeros((len(v), 3), np.uint8)
        colors[side, 0] = 255
        colors[~side, 2] = 255

        default = decimate_qem(v, f, target_vertices=40, colors=colors)
        partial = decimate_qem(
            v, f, target_vertices=40, colors=colors, attribute_weight=0.05
        )

        assert default.colors is not None and partial.colors is not None
        default_mixed = np.count_nonzero(
            (default.colors[:, 0] > 0) & (default.colors[:, 2] > 0)
        )
        partial_mixed = np.count_nonzero(
            (partial.colors[:, 0] > 0) & (partial.colors[:, 2] > 0)
        )
        assert 0 < partial_mixed < default_mixed

    def test_attribute_quadric_preserves_a_scalar_boundary(self) -> None:
        v, f = octasphere(3)
        scalars = (v[:, 0] + 0.31 * v[:, 1] >= 0).astype(np.float32)

        default = decimate_qem(v, f, target_vertices=40, scalars=scalars)
        preserved = decimate_qem(
            v, f, target_vertices=40, scalars=scalars, attribute_weight=1.0
        )

        assert default.scalars is not None and preserved.scalars is not None
        assert np.any((default.scalars > 0) & (default.scalars < 1))
        assert set(np.unique(preserved.scalars)) == {0.0, 1.0}

    @pytest.mark.parametrize("value", [-1.0, np.inf, np.nan])
    def test_attribute_weight_must_be_nonnegative_and_finite(
        self, value: float
    ) -> None:
        v, f = octasphere(1)
        with pytest.raises(ValueError, match="attribute_weight"):
            decimate_qem(v, f, target_vertices=10, attribute_weight=value)

    def test_boundary_and_incident_face_caches_stay_exact_after_each_collapse(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        segments = 64
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        inner = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(segments)))
        outer = inner * np.array([2.0, 2.0, 1.0])
        vertices = np.concatenate((inner, outer)).astype(np.float32)
        faces = []
        for index in range(segments):
            following = (index + 1) % segments
            faces.extend(
                (
                    (index, segments + index, segments + following),
                    (index, segments + following, following),
                )
            )

        apply_collapse = qem._apply_collapse
        collapse_count = 0

        def checked_apply_collapse(*args: Any, **kwargs: Any) -> None:
            nonlocal collapse_count
            apply_collapse(*args, **kwargs)
            collapse_count += 1
            work_faces = kwargs["work_faces"]
            active_faces = kwargs["active_faces"]
            alive = kwargs["alive"]
            vertex_faces = kwargs["vertex_faces"]
            boundary_vertices = kwargs["boundary_vertices"]

            expected_faces = [set() for _ in range(len(alive))]
            edge_counts: dict[tuple[int, int], int] = {}
            for face_index in np.flatnonzero(active_faces):
                face = work_faces[face_index]
                for vertex in face:
                    expected_faces[int(vertex)].add(int(face_index))
                for first, second in (
                    (face[0], face[1]),
                    (face[1], face[2]),
                    (face[2], face[0]),
                ):
                    edge = tuple(sorted((int(first), int(second))))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

            expected_boundary = np.zeros(len(alive), dtype=bool)
            for (first, second), count in edge_counts.items():
                if count == 1:
                    expected_boundary[[first, second]] = True

            for vertex in np.flatnonzero(alive):
                assert vertex_faces[vertex] == expected_faces[vertex]
            np.testing.assert_array_equal(
                boundary_vertices[alive], expected_boundary[alive]
            )

        monkeypatch.setattr(qem, "_apply_collapse", checked_apply_collapse)

        result = decimate_qem(
            vertices, np.asarray(faces, dtype=np.uint32), target_vertices=48
        )

        assert collapse_count > 0
        assert len(result.vertices) == 48

    @pytest.mark.parametrize("ndim", [3, 4])
    def test_small_system_solver_matches_numpy_and_rejects_rank_deficiency(
        self, ndim: int
    ) -> None:
        """The tolerance has to follow the CONDITIONING, not sit at a constant.

        ``_solve_system`` takes two different routes. At ndim != 3 it delegates
        to ``np.linalg.solve``, so this comparison is against itself and the
        error is exactly 0 — that arm pins the delegation, not any arithmetic.
        At ndim == 3 it uses a closed-form adjugate inverse to avoid a LAPACK
        call on a tiny matrix, and an adjugate loses accuracy faster than LU:
        measured over 400 random draws per rung, its relative disagreement with
        LAPACK is 4e-16 at kappa=1, 7.4e-12 at 1e4, but reaches 8.6e-6 at 1e8 —
        roughly ``kappa**1.5 * eps`` rather than LU's ``kappa * eps``.

        A flat ``rtol=1e-6`` therefore failed the kappa=1e8 rung for a large
        share of seeds (median 3.3e-7, p99 6.5e-6). It survived on Linux and
        failed on macOS only because the two LAPACKs build a different ``qr``
        basis from the same seed — platform roulette, not a platform bug. The
        bound below tracks the algorithm's real error growth and still sits
        orders of magnitude below anything a broken solver would produce.
        """
        eps = np.finfo(np.float64).eps
        rng = np.random.default_rng(1798 + ndim)
        for condition in (1.0, 1e4, 1e8):
            basis, _ = np.linalg.qr(rng.normal(size=(ndim, ndim)))
            eigenvalues = np.geomspace(1.0, 1.0 / condition, ndim)
            matrix = basis @ np.diag(eigenvalues) @ basis.T
            rhs = rng.normal(size=ndim)

            solved = _solve_system(matrix, rhs)

            assert solved is not None
            # ndim != 3 delegates, so hold it to exactness; the adjugate path
            # gets the conditioning-aware bound, floored so kappa=1 stays tight.
            rtol = 0.0 if ndim != 3 else max(1e-9, condition**1.5 * eps)
            atol = 0.0 if ndim != 3 else 1e-8
            np.testing.assert_allclose(
                solved, np.linalg.solve(matrix, rhs), rtol=rtol, atol=atol
            )

        rank_deficient = np.eye(ndim, dtype=np.float64)
        rank_deficient[-1] = rank_deficient[-2]
        assert _solve_system(rank_deficient, np.ones(ndim)) is None

    def test_non_three_dimensional_solver_rejects_near_singularity(self) -> None:
        matrix = np.diag([1.0, 1.0, 1.0, 1e-15])
        rhs = np.ones(4)

        assert np.linalg.matrix_rank(matrix, tol=1e-12) == 3
        assert np.linalg.norm(np.linalg.solve(matrix, rhs)) > 1e14
        assert _solve_system(matrix, rhs) is None

    def test_a_flat_four_dimensional_ladder_stays_inside_its_input_bounds(
        self,
    ) -> None:
        side = 20
        vertices = np.array(
            [
                (x, y, 0, 0)
                for y in np.linspace(0, 1, side)
                for x in np.linspace(0, 1, side)
            ],
            dtype=np.float32,
        )
        faces = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                b, c, d = a + 1, a + side, a + side + 1
                faces.extend(((a, b, d), (a, d, c)))

        levels = decimate_qem_ladder(
            vertices,
            np.asarray(faces, dtype=np.uint32),
            target_vertices=[25, 100],
            spatial_dims=(0, 1, 2, 3),
        )

        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        for level in levels:
            assert np.all(level.vertices >= lower)
            assert np.all(level.vertices <= upper)

    def test_a_ladder_reuses_one_collapse_sequence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        vertices, faces = octasphere(4)
        colors = np.round((vertices + 1.0) * 127.5).astype(np.uint8)
        scalars = np.arange(len(vertices), dtype=np.float32)
        build_heap = qem._build_heap
        heap_builds = 0

        def counted_build_heap(*args: object, **kwargs: object) -> object:
            nonlocal heap_builds
            heap_builds += 1
            return build_heap(*args, **kwargs)

        monkeypatch.setattr(qem, "_build_heap", counted_build_heap)

        targets = [40, 120, 400]
        levels = decimate_ladder(
            vertices,
            faces,
            target_vertices=targets,
            method="qem",
            colors=colors,
            scalars=scalars,
        )

        assert heap_builds == 1
        assert [len(level.vertices) for level in levels] == targets
        for target, level in zip(targets, levels, strict=True):
            independent = decimate_qem(
                vertices,
                faces,
                target_vertices=target,
                colors=colors,
                scalars=scalars,
            )
            np.testing.assert_array_equal(level.vertices, independent.vertices)
            np.testing.assert_array_equal(level.faces, independent.faces)
            np.testing.assert_array_equal(level.colors, independent.colors)
            np.testing.assert_array_equal(level.scalars, independent.scalars)
            edges, boundary, nonmanifold = edge_audit(len(level.vertices), level.faces)
            assert len(level.vertices) - edges + len(level.faces) == 2
            assert boundary == 0
            assert nonmanifold == 0

    def test_cluster_ladder_dispatch_matches_independent_levels(self) -> None:
        vertices, faces = octasphere(3)
        targets = [40, 80]

        levels = decimate_ladder(
            vertices, faces, target_vertices=targets, method="cluster"
        )

        for target, level in zip(targets, levels, strict=True):
            independent = decimate_cluster(vertices, faces, target_vertices=target)
            np.testing.assert_array_equal(level.vertices, independent.vertices)
            np.testing.assert_array_equal(level.faces, independent.faces)

    def test_edge_target_does_not_run_an_svd_per_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        quadrics = np.zeros((2, 4, 4), dtype=np.float64)
        quadrics[:, :3, :3] = np.eye(3)
        quadrics[0, :3, 3] = quadrics[0, 3, :3] = [-0.25, 0.0, 0.0]
        quadrics[1, :3, 3] = quadrics[1, 3, :3] = [-0.75, 0.0, 0.0]

        def reject_svd(*args: object, **kwargs: object) -> None:
            raise AssertionError(
                "the per-edge target path must not compute matrix rank"
            )

        monkeypatch.setattr(np.linalg, "matrix_rank", reject_svd)

        _, target = _edge_target(0, 1, positions, quadrics)

        np.testing.assert_allclose(target, [0.5, 0.0, 0.0])

    def test_edge_target_rejects_an_out_of_envelope_solve(self) -> None:
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        quadrics = np.zeros((2, 4, 4), dtype=np.float64)
        quadrics[:, :3, :3] = np.diag([1.0, 1.0, 1e-13]) / 2.0
        quadrics[:, :3, 3] = quadrics[:, 3, :3] = np.array([-0.5, 0.0, 1e-3]) / 2.0

        solved = _solve_system(
            (quadrics[0] + quadrics[1])[:3, :3],
            -(quadrics[0] + quadrics[1])[:3, 3],
        )
        assert solved is not None
        assert solved[2] == pytest.approx(-1e10)

        _, target = _edge_target(0, 1, positions, quadrics)

        np.testing.assert_array_equal(target, [0.5, 0.0, 0.0])

    def test_an_at_target_mesh_is_returned_unchanged(self) -> None:
        vertices, faces = octasphere(1)
        result = decimate_qem(vertices, faces, target_vertices=len(vertices))
        np.testing.assert_array_equal(result.vertices, vertices)
        np.testing.assert_array_equal(result.faces, faces)

    def test_single_and_ladder_at_target_paths_accept_a_degenerate_surface(
        self,
    ) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float32
        )
        faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32)

        single = decimate_qem(vertices, faces, target_vertices=10)
        ladder = decimate_qem_ladder(vertices, faces, target_vertices=[10])[0]

        np.testing.assert_array_equal(single.vertices, vertices)
        np.testing.assert_array_equal(single.faces, faces)
        np.testing.assert_array_equal(ladder.vertices, vertices)
        np.testing.assert_array_equal(ladder.faces, faces)

    def test_link_condition_preserves_the_closed_sphere_at_every_level(self) -> None:
        v, f = octasphere(5)
        cluster = decimate_cluster(v, f, target_vertices=500)
        _, cluster_boundary, cluster_nonmanifold = edge_audit(
            len(cluster.vertices), cluster.faces
        )
        assert (cluster_boundary, cluster_nonmanifold) == (120, 60)

        for target in (1000, 500, 150):
            result = decimate_qem(v, f, target_vertices=target)
            edges, boundary, nonmanifold = edge_audit(
                len(result.vertices), result.faces
            )
            assert len(result.vertices) == target
            assert len(result.vertices) - edges + len(result.faces) == 2
            assert boundary == 0
            assert nonmanifold == 0

    def test_attributes_are_aggregated_and_normals_are_recomputed(self) -> None:
        v, f = octasphere(3)
        colors = np.round((v + 1.0) * 127.5).astype(np.uint8)
        scalars = np.arange(len(v), dtype=np.float32)
        result = decimate_qem(
            v,
            f,
            target_vertices=50,
            normals=np.zeros_like(v),
            normal_dims=(0, 1, 2),
            colors=colors,
            scalars=scalars,
        )
        assert result.colors is not None and result.colors.dtype == np.uint8
        assert result.colors.shape == (50, 3)
        assert result.scalars.shape == (50,)
        assert result.normals is not None
        np.testing.assert_allclose(
            np.linalg.norm(result.normals, axis=1), 1.0, atol=1e-5
        )

    def test_an_open_surface_preserves_its_orientation_area_and_boundary(self) -> None:
        side = 8
        vertices = np.array(
            [(x, y, 0) for y in range(side) for x in range(side)], np.float32
        )
        faces = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                b, c, d = a + 1, a + side, a + side + 1
                faces.extend(((a, b, d), (a, d, c)))
        result = decimate_qem(
            vertices, np.asarray(faces, np.uint32), target_vertices=24
        )
        triangles = result.vertices[result.faces]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        edges, boundary, nonmanifold = edge_audit(len(result.vertices), result.faces)
        assert len(result.vertices) - edges + len(result.faces) == 1
        assert np.all(cross[:, 2] > 0), "every coarse face must keep the input winding"
        assert np.linalg.norm(cross, axis=1).sum() / 2 >= 0.9 * (side - 1) ** 2
        assert boundary >= 2 * (side - 1)
        assert nonmanifold == 0

    def test_an_annulus_does_not_fold_over_itself(self) -> None:
        radial_count, angular_count = 12, 40
        radii = np.linspace(0.5, 1.0, radial_count)
        angles = np.linspace(0.0, 2 * np.pi, angular_count, endpoint=False)
        vertices = np.array(
            [
                (radius * np.cos(angle), radius * np.sin(angle), 0.0)
                for radius in radii
                for angle in angles
            ],
            np.float32,
        )
        faces = []
        for radial_index in range(radial_count - 1):
            for angular_index in range(angular_count):
                next_angle = (angular_index + 1) % angular_count
                a = radial_index * angular_count + angular_index
                b = radial_index * angular_count + next_angle
                c = (radial_index + 1) * angular_count + angular_index
                d = (radial_index + 1) * angular_count + next_angle
                faces.extend(((a, b, d), (a, d, c)))
        result = decimate_qem(
            vertices, np.asarray(faces, np.uint32), target_vertices=100
        )

        triangles = result.vertices[result.faces]
        oriented_area = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )[:, 2]
        assert len(result.vertices) >= 100
        assert np.all(oriented_area < 0), (
            "every coarse face must keep the input winding"
        )
        assert np.abs(oriented_area).sum() == pytest.approx(
            abs(oriented_area.sum()), rel=1e-6
        )

    def test_small_coordinate_scales_keep_a_valid_cost_function(self) -> None:
        vertices, faces = octasphere(2)

        for scale in (1e-7, 1e-13):
            quadrics = _face_quadrics((vertices * scale).astype(np.float64)[faces])
            assert np.all(np.abs(quadrics).max(axis=(1, 2)) > 0.0)

            scaled = decimate_qem(vertices * scale, faces, target_vertices=30)
            assert len(scaled.vertices) == 30
            assert np.isfinite(scaled.vertices).all()
            radii = np.linalg.norm(scaled.vertices / scale, axis=1)
            assert radii.min() > 0.9
            assert radii.max() < 1.15

    def test_distant_component_does_not_zero_local_face_quadrics(self) -> None:
        vertices, faces = octasphere(2)
        far = np.array([[1e9, 0, 0], [1e9 + 1, 0, 0], [1e9, 1, 0]], dtype=np.float64)
        positions = np.concatenate([vertices.astype(np.float64), far])
        all_faces = np.concatenate(
            [faces, np.array([[len(vertices), len(vertices) + 1, len(vertices) + 2]])]
        )

        quadrics = _face_quadrics(positions[all_faces])

        assert np.all(np.abs(quadrics[: len(faces)]).max(axis=(1, 2)) > 0.0)

    def test_nonspatial_columns_are_hard_collapse_barriers(self) -> None:
        vertices, faces = octasphere(3)
        barrier = (vertices[:, 2] >= 0).astype(np.float32)[:, None]
        stacked = np.concatenate([barrier, vertices], axis=1)
        result = decimate_qem(
            stacked,
            faces,
            target_vertices=80,
            spatial_dims=(1, 2, 3),
            scalars=barrier[:, 0],
        )
        groups, counts = np.unique(result.vertices[:, 0], return_counts=True)

        assert len(result.vertices) < len(vertices), (
            "collapses must happen within groups"
        )
        assert groups.tolist() == [0.0, 1.0], "barrier values must not blend"
        assert np.all(counts > 10), "each barrier group must keep its own surface"
        assert result.scalars is not None
        np.testing.assert_array_equal(result.scalars, result.vertices[:, 0])

    def test_unreferenced_vertices_do_not_consume_the_target_budget(self) -> None:
        vertices, faces = octasphere(3)
        with_strays = np.concatenate(
            [vertices, np.full((200, 3), 7.0, dtype=np.float32)]
        )

        above_surface = decimate_qem(with_strays, faces, target_vertices=300)
        clustered = decimate_cluster(with_strays, faces, target_vertices=300)
        assert len(above_surface.vertices) == len(clustered.vertices) == len(vertices)

        reduced = decimate_qem(with_strays, faces, target_vertices=114)
        assert len(reduced.vertices) == 114
        assert len(reduced.faces) > 0

    def test_a_detached_tetrahedron_component_is_not_annihilated(self) -> None:
        sphere_vertices, sphere_faces = octasphere(4)
        tetra_vertices = np.array(
            [[3, 0, 0], [4, 0, 0], [3.5, 1, 0], [3.5, 0.5, 1]],
            dtype=np.float32,
        )
        tetra_faces = np.array(
            [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.uint32
        ) + len(sphere_vertices)
        vertices = np.concatenate([sphere_vertices, tetra_vertices])
        faces = np.concatenate([sphere_faces, tetra_faces])

        result = decimate_qem(vertices, faces, target_vertices=20)

        assert len(result.vertices) == 20
        assert int((result.vertices[:, 0] > 2).sum()) == 4

    def test_a_detached_open_component_is_not_annihilated(self) -> None:
        sphere_vertices, sphere_faces = octasphere(4)
        triangle_vertices = np.array(
            [[3, 0, 0], [4, 0, 0], [3.5, 1, 0]], dtype=np.float32
        )
        triangle_faces = np.array([[0, 1, 2]], dtype=np.uint32) + len(sphere_vertices)
        vertices = np.concatenate([sphere_vertices, triangle_vertices])
        faces = np.concatenate([sphere_faces, triangle_faces])

        result = decimate_qem(vertices, faces, target_vertices=20)

        assert len(result.vertices) == 20
        assert int((result.vertices[:, 0] > 2).sum()) == 3

    def test_an_attached_triangle_patch_is_not_annihilated(self) -> None:
        sphere_vertices, sphere_faces = octasphere(4)
        patch_vertices = np.array([[3, 0, 0], [3.5, 1, 0]], dtype=np.float32)
        patch_face = np.array(
            [[0, len(sphere_vertices), len(sphere_vertices) + 1]], dtype=np.uint32
        )
        vertices = np.concatenate([sphere_vertices, patch_vertices])
        faces = np.concatenate([sphere_faces, patch_face])

        result = decimate_qem(vertices, faces, target_vertices=20)

        assert len(result.vertices) == 20
        assert int((result.vertices[:, 0] > 2).sum()) == 2

    def test_two_dimensional_auto_falls_back_but_explicit_qem_is_refused(
        self,
    ) -> None:
        side = 8
        vertices = np.array(
            [(x, y) for y in range(side) for x in range(side)], np.float32
        )
        faces = []
        for y in range(side - 1):
            for x in range(side - 1):
                a = y * side + x
                b, c, d = a + 1, a + side, a + side + 1
                faces.extend(((a, b, d), (a, d, c)))
        face_array = np.asarray(faces, np.uint32)

        automatic = decimate(
            vertices,
            face_array,
            target_vertices=24,
            method="auto",
            spatial_dims=(0, 1),
        )
        clustered = decimate_cluster(
            vertices, face_array, target_vertices=24, spatial_dims=(0, 1)
        )

        np.testing.assert_array_equal(automatic.vertices, clustered.vertices)
        np.testing.assert_array_equal(automatic.faces, clustered.faces)
        with pytest.raises(ValueError, match="requires at least 3 coarsening"):
            decimate(
                vertices,
                face_array,
                target_vertices=24,
                method="qem",
                spatial_dims=(0, 1),
            )
        with pytest.raises(ValueError, match="requires at least 3 coarsening"):
            decimate_qem(
                vertices,
                face_array,
                target_vertices=24,
                spatial_dims=(0, 1),
            )

    def test_a_degenerate_surface_is_refused_instead_of_returned_empty(self) -> None:
        vertices = np.zeros((10, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=np.uint32)

        with pytest.raises(ValueError, match="input .* has no triangle spanning"):
            decimate_qem(vertices, faces, target_vertices=4)

    def test_auto_uses_qem_only_inside_its_measured_envelope(self) -> None:
        assert (
            resolve_decimation_method("auto", QEM_AUTO_VERTEX_LIMIT, announce=False)
            == "qem"
        )
        assert (
            resolve_decimation_method("auto", QEM_AUTO_VERTEX_LIMIT + 1, announce=False)
            == "cluster"
        )
        assert resolve_decimation_method("qem", 1_000, announce=False) == "qem"
