"""Mesh decimation — the producer a substitutive LOD ladder needs.

A ``kind=lod`` group selects ONE child at a time by ``coverage_fraction``, so each
level must be an independently renderable stand-in for the finer one. For points,
lines and gsplats the coarse level is a *subset or merge of independent elements*.
A surface has no such freedom: dropping triangles punches holes, and dropping
vertices without repairing the faces that index them corrupts the topology
outright. Decimation is the operation that produces a genuinely coarser SURFACE,
and its absence — not any structural objection — is the only reason mesh had no
substitutive ladder (``docs/specs/MESH_NODE_SPEC.md`` §9).

``cluster`` (this module) is quadric-weighted vertex clustering: snap vertices to a
grid, collapse each occupied cell to one representative, reindex the faces, drop
the triangles that collapsed to a line. It is O(V log V), fully vectorized, and
cannot fail — properties that matter because the writer's cap is 2**27 vertices and
a Python edge-collapse loop is unusable at that scale.

The quadric is what makes it worth more than a centroid snap. Placing each
representative at the minimizer of the summed squared distance to its cell's
incident planes keeps creases and thin sheets where a centroid would round them
off — and creases are exactly what Luxar's target data (isosurfaces, segmentation
boundaries) is made of.

**Clustering does not preserve manifoldness, and cannot.** When a cell swallows two
sheets that were separate on the fine surface, their triangles land on the same
representatives and the result has edges shared by more than two faces. Measured on
a closed unit sphere (4098 vertices, boundary-free, chi=2), the coarse levels come
back chi=2 with 0 boundary edges at most cell sizes but hit 120 boundary and 60
non-manifold edges at one. That is a property of the algorithm rather than a defect
here, it renders correctly either way (a non-manifold edge is still just triangles),
and it is the concrete reason to keep an edge-collapse tier around rather than a
vague appeal to quality: edge collapse can refuse a collapse that would break the
link condition, and clustering has no such veto to exercise.

Related: dropping the DUPLICATE triangles clustering produces is not cosmetic. With
them the sphere above measures chi=86 with 96 non-manifold edges; without them,
chi=2 with 60. The duplicates visually seal the degenerate region while corrupting
the topology underneath it, and two co-planar opaque triangles z-fight.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class DecimatedMesh(NamedTuple):
    """One coarser level: a complete, independently renderable surface."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.uint32]
    normals: NDArray[np.float32] | None
    colors: NDArray[np.uint8] | None


def _face_quadrics(
    vertices: NDArray[np.float64], faces: NDArray[np.uint32]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-face plane quadric ``K = p pᵀ`` (as its 10 unique entries) and area.

    ``vertices`` must be the SPATIAL columns only, shape ``(V, 3)``. A plane is a
    3D notion, and a mesh's vertex array is nD — passing all of it here is how a 4D
    (timelapse) mesh reaches ``np.cross`` with four components and raises.

    ``p = [nx, ny, nz, d]`` with ``n`` unit-length and ``d = -n·v0``, so ``pᵀ[v,1]``
    is the signed distance from ``v`` to the face's plane and ``[v,1]ᵀ K [v,1]`` is
    its square. Summing K over the faces around a vertex gives the standard
    Garland-Heckbert quadric; the minimizer of that sum is where the representative
    belongs.

    Faces are weighted by AREA. Without it a dense cluster of slivers outvotes the
    one large triangle that actually defines the surface there.
    """
    v0 = vertices[faces[:, 0]]
    e1 = vertices[faces[:, 1]] - v0
    e2 = vertices[faces[:, 2]] - v0
    cross = np.cross(e1, e2)
    norm = np.linalg.norm(cross, axis=1)
    area = 0.5 * norm
    # A degenerate face has no plane; give it a zero normal so its quadric is zero
    # and it contributes nothing rather than producing NaN.
    safe = np.where(norm > 0, norm, 1.0)
    n = cross / safe[:, None]
    n[norm <= 0] = 0.0
    d = -np.einsum("ij,ij->i", n, v0)
    p = np.concatenate([n, d[:, None]], axis=1)
    return p, area


# Index pairs of the upper triangle of a symmetric 4x4, in row-major order. Storing
# 10 numbers instead of 16 halves the accumulation traffic, which is the hot loop.
_TRI_I, _TRI_J = np.triu_indices(4)


def _accumulate_vertex_quadrics(
    n_vertices: int,
    faces: NDArray[np.uint32],
    p: NDArray[np.float64],
    area: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum each face's area-weighted quadric onto its three corners -> ``(V, 10)``."""
    packed = area[:, None] * (p[:, _TRI_I] * p[:, _TRI_J])
    out = np.zeros((n_vertices, 10), dtype=np.float64)
    for corner in range(3):
        np.add.at(out, faces[:, corner], packed)
    return out


def _unpack_quadrics(packed: NDArray[np.float64]) -> NDArray[np.float64]:
    """``(M, 10)`` upper-triangle storage -> ``(M, 4, 4)`` symmetric matrices."""
    full = np.zeros((packed.shape[0], 4, 4), dtype=np.float64)
    full[:, _TRI_I, _TRI_J] = packed
    full[:, _TRI_J, _TRI_I] = packed
    return full


def _solve_representatives(
    quadrics: NDArray[np.float64], fallback: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Place each representative at its quadric's minimizer, else the centroid.

    The minimizer solves ``A x = -b`` with ``A = Q[:3,:3]``, ``b = Q[:3,3]``. ``A`` is
    singular whenever the cell's planes do not pin all three axes — a flat patch
    (one plane), a straight crease (two), an isolated vertex (none) — which is
    common, not exceptional. Those cells fall back to the centroid, which is the
    right answer for them precisely because the quadric expresses no preference
    along the unconstrained directions.

    Solved as one batched ``np.linalg.solve`` over the well-conditioned cells rather
    than a Python loop; ``np.linalg.cond`` is checked instead of catching
    ``LinAlgError`` because a *nearly* singular system returns a huge finite answer
    that would fling a vertex far outside the mesh.
    """
    out = fallback.copy()
    a = quadrics[:, :3, :3]
    b = quadrics[:, :3, 3]
    with np.errstate(all="ignore"):
        cond = np.linalg.cond(a)
    ok = np.isfinite(cond) & (cond < 1e8)
    if np.any(ok):
        # `b` must be a stack of COLUMN vectors: numpy's solve reads a trailing
        # (k, 3) as one 3-column matrix per system, not k right-hand sides.
        solved = np.linalg.solve(a[ok], -b[ok][..., None])[..., 0]
        # The caller additionally clamps each representative to its own cluster's
        # bounding box: a well-conditioned quadric on a near-flat patch is legitimate
        # but can still place a vertex far outside the local geometry, which reads as
        # a spike on the coarse level.
        out[ok] = solved
    result: NDArray[np.float64] = out
    return result


def _cluster_keys(
    vertices: NDArray[np.float64],
    spatial_dims: tuple[int, ...],
    cell: float,
) -> NDArray[np.int64]:
    """Grid cell per vertex over ``spatial_dims``; every OTHER dim is a barrier.

    Non-spatial dimensions enter the key at full precision rather than being
    snapped, so vertices that merely share a location at different timepoints or
    channels never merge. That is the same rule ``--coarsen-dims`` expresses for the
    gsplat reductions, applied here as the default instead of an option: a mesh's
    non-spatial axes are categorical in every dataset Luxar writes today.
    """
    n, d = vertices.shape
    spatial = np.asarray(spatial_dims, dtype=np.intp)
    grid = np.floor(vertices[:, spatial] / cell).astype(np.int64)
    barrier = [i for i in range(d) if i not in set(spatial_dims)]
    if not barrier:
        return grid
    # Barrier columns are compared exactly; float bits are fine as an equality key.
    exact = vertices[:, barrier].view(np.int64).reshape(n, len(barrier))
    return np.concatenate([grid, exact], axis=1)


def decimate_cluster(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    target_vertices: int,
    normals: NDArray[np.float32] | None = None,
    colors: NDArray[np.uint8] | None = None,
    spatial_dims: tuple[int, ...] | None = None,
    max_iterations: int = 24,
) -> DecimatedMesh:
    """Reduce ``vertices`` toward ``target_vertices`` by quadric vertex clustering.

    Args:
        vertices: ``(V, D)`` positions, D >= 2.
        faces: ``(F, 3)`` indices into ``vertices``.
        target_vertices: Desired vertex count of the result. Approximate — the grid
            cannot hit an arbitrary count exactly, so the search stops at the
            coarsest spacing that still leaves at least this many vertices.
        normals: Optional ``(V, 3)``. Recomputed from the coarse geometry rather
            than averaged: an averaged normal describes the FINE surface and would
            light the coarse one wrongly at exactly the creases clustering moved.
        colors: Optional ``(V, C)`` uint8, averaged within each cluster.
        spatial_dims: Which columns are spatial. Defaults to the first ``min(3, D)``.
        max_iterations: Bisection budget for the cell-size search.

    Returns:
        A :class:`DecimatedMesh`. Never empty: if the search cannot reach the target
        without collapsing the surface away, the finest achievable level is
        returned instead, because a `kind=lod` group with an empty level cannot
        derive coverage fractions at all.

    Raises:
        ValueError: If inputs are malformed or ``target_vertices`` < 4.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError(f"vertices must be (V, D>=2), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    if target_vertices < 4:
        raise ValueError(
            f"target_vertices must be at least 4 (a tetrahedron is the smallest "
            f"closed surface), got {target_vertices}"
        )
    if faces.size and int(faces.max()) >= vertices.shape[0]:
        raise ValueError(
            f"face index {int(faces.max())} is out of range for "
            f"{vertices.shape[0]} vertices"
        )

    d = vertices.shape[1]
    if spatial_dims is None:
        spatial_dims = tuple(range(min(3, d)))
    if not spatial_dims or any(i < 0 or i >= d for i in spatial_dims):
        raise ValueError(f"spatial_dims {spatial_dims} out of range for {d} dims")

    if vertices.shape[0] <= target_vertices:
        return DecimatedMesh(vertices, faces, normals, colors)

    v64 = vertices.astype(np.float64)
    spatial = np.asarray(spatial_dims, dtype=np.intp)
    if len(spatial_dims) == 3:
        p, area = _face_quadrics(v64[:, spatial], faces)
        vertex_quadrics = _accumulate_vertex_quadrics(vertices.shape[0], faces, p, area)
    else:
        # A plane quadric needs exactly three spatial axes. With two (a planar mesh)
        # there is no normal direction to preserve, so every cell falls back to its
        # centroid — which for a flat surface is what the quadric would pick anyway.
        vertex_quadrics = np.zeros((vertices.shape[0], 10), dtype=np.float64)
    extent = float(
        np.max(v64[:, spatial].max(axis=0) - v64[:, spatial].min(axis=0)) or 1.0
    )

    # Bisect the cell size. Coarser cells -> fewer vertices, monotonically, which is
    # what makes bisection valid here. Seeded from the ideal cubic packing so the
    # first probe is usually within a factor of two.
    ratio = max(target_vertices / vertices.shape[0], 1e-9)
    lo, hi = extent * 1e-6, extent
    guess = extent * (ratio ** (1.0 / len(spatial_dims)))
    best: DecimatedMesh | None = None
    for _ in range(max_iterations):
        candidate = _cluster_once(
            v64, faces, vertex_quadrics, spatial_dims, guess, normals, colors
        )
        count = candidate.vertices.shape[0]
        if count >= target_vertices:
            best = candidate
            if count <= target_vertices * 1.1:
                break
            lo = guess
        else:
            hi = guess
        guess = 0.5 * (lo + hi)

    if best is None:
        # Every probe overshot. Return the least-reduced one we can still build.
        best = _cluster_once(
            v64, faces, vertex_quadrics, spatial_dims, lo, normals, colors
        )
    return best


def _cluster_once(
    v64: NDArray[np.float64],
    faces: NDArray[np.uint32],
    vertex_quadrics: NDArray[np.float64],
    spatial_dims: tuple[int, ...],
    cell: float,
    normals: NDArray[np.float32] | None,
    colors: NDArray[np.uint8] | None,
) -> DecimatedMesh:
    """One clustering pass at a fixed cell size."""
    keys = _cluster_keys(v64, spatial_dims, max(cell, 1e-12))
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_clusters = int(inverse.max()) + 1 if inverse.size else 0

    # Centroid per cluster — both the fallback representative and the clamp target.
    counts = np.bincount(inverse, minlength=n_clusters).astype(np.float64)
    centroid = np.zeros((n_clusters, v64.shape[1]), dtype=np.float64)
    for col in range(v64.shape[1]):
        centroid[:, col] = np.bincount(
            inverse, weights=v64[:, col], minlength=n_clusters
        )
    centroid /= np.maximum(counts, 1)[:, None]

    summed = np.zeros((n_clusters, 10), dtype=np.float64)
    np.add.at(summed, inverse, vertex_quadrics)
    spatial = np.asarray(spatial_dims, dtype=np.intp)
    placed = _solve_representatives(_unpack_quadrics(summed), centroid[:, spatial])

    new_v = centroid.copy()
    new_v[:, spatial] = placed
    # Keep every representative inside its cluster's own bounding box. A quadric
    # minimizer on a near-flat patch is legitimate but can sit far outside the local
    # geometry, which shows up as a spike on the coarse level.
    for axis, col in enumerate(spatial):
        lo = np.full(n_clusters, np.inf)
        hi = np.full(n_clusters, -np.inf)
        np.minimum.at(lo, inverse, v64[:, col])
        np.maximum.at(hi, inverse, v64[:, col])
        new_v[:, col] = np.clip(new_v[:, col], lo, hi)

    new_f = inverse[faces.reshape(-1)].reshape(faces.shape).astype(np.uint32)
    a, b, c = new_f[:, 0], new_f[:, 1], new_f[:, 2]
    new_f = new_f[(a != b) & (b != c) & (a != c)]
    # Clustering also creates DUPLICATE triangles (two fine faces collapsing onto
    # the same coarse corners). They render identically but inflate the face count
    # the LOD thresholds are derived from, so drop them.
    if new_f.shape[0]:
        canonical = np.sort(new_f, axis=1)
        _, keep = np.unique(canonical, axis=0, return_index=True)
        new_f = new_f[np.sort(keep)]

    # Average the colours over the FULL cluster set, before any compaction below,
    # so every contributing fine vertex is counted exactly once.
    new_colors = None
    if colors is not None:
        acc = np.zeros((n_clusters, colors.shape[1]), dtype=np.float64)
        for col in range(colors.shape[1]):
            acc[:, col] = np.bincount(
                inverse,
                weights=colors[:, col].astype(np.float64),
                minlength=n_clusters,
            )
        new_colors = np.clip(acc / np.maximum(counts, 1)[:, None], 0, 255).astype(
            np.uint8
        )

    # Drop representatives no surviving face references. They are not harmless
    # padding: `coverage_fractions` derives the LOD switch thresholds from element
    # COUNTS, so phantom vertices shift every threshold — and a vertex with no
    # incident face has no defined normal, so it would take the arbitrary fallback
    # and drag the level's shading statistics with it.
    if new_f.shape[0]:
        referenced = np.unique(new_f)
        if referenced.size != new_v.shape[0]:
            remap = np.zeros(new_v.shape[0], dtype=np.int64)
            remap[referenced] = np.arange(referenced.size)
            new_f = remap[new_f].astype(np.uint32)
            new_v = new_v[referenced]
            if new_colors is not None:
                new_colors = new_colors[referenced]

    new_normals = None
    if normals is not None:
        new_normals = _recompute_normals(new_v[:, spatial], new_f, new_v.shape[0])

    return DecimatedMesh(
        vertices=new_v.astype(np.float32),
        faces=new_f,
        normals=new_normals,
        colors=new_colors,
    )


def _recompute_normals(
    positions: NDArray[np.float64], faces: NDArray[np.uint32], n_vertices: int
) -> NDArray[np.float32]:
    """Area-weighted vertex normals of the COARSE surface.

    Area weighting falls out of not normalizing the cross product before
    accumulation, which is the standard trick and the right weighting: a big
    triangle should dominate the shading of a corner it shares with a sliver.
    """
    out = np.zeros((n_vertices, 3), dtype=np.float64)
    if faces.shape[0]:
        v0 = positions[faces[:, 0]]
        face_n = np.cross(positions[faces[:, 1]] - v0, positions[faces[:, 2]] - v0)
        for corner in range(3):
            np.add.at(out, faces[:, corner], face_n)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    # An isolated vertex (every incident face dropped as degenerate) has no defined
    # normal; +Z is arbitrary but finite, and a zero normal would blow up the
    # shader's normalize().
    out = np.where(norm > 0, out / np.maximum(norm, 1e-20), np.array([0.0, 0.0, 1.0]))
    return out.astype(np.float32)


__all__ = ["DecimatedMesh", "decimate_cluster"]
