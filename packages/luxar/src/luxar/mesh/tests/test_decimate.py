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
        r = decimate_cluster(v, f, target_vertices=150, normals=np.zeros_like(v))
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
        r = decimate_cluster(v, f, target_vertices=10_000)
        np.testing.assert_array_equal(r.vertices, v)
        np.testing.assert_array_equal(r.faces, f)

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
