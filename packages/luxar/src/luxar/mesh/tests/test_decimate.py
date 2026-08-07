"""Decimation must produce a correct coarser SURFACE, not merely a smaller array.

Every assertion here is about a property the renderer or the LOD group depends on:
in-range indices (or the writer rejects it), no NaN (or the shader draws nothing),
monotonically falling counts (or `coverage_fractions` is non-monotonic), and
geometry that still occupies the same space as the original (or the level is not a
stand-in for anything).
"""

from __future__ import annotations

import numpy as np
import pytest

from ..decimate import decimate_cluster


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
        from .. import decimate as module

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
