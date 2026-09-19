"""Mesh decimation — the producer a substitutive LOD ladder needs.

A ``kind=lod`` group selects ONE child at a time by ``coverage_fraction``, so each
level must be an independently renderable stand-in for the finer one. For points,
lines and gsplats the coarse level is a *subset or merge of independent elements*.
A surface has no such freedom: dropping triangles punches holes, and dropping
vertices without repairing the faces that index them corrupts the topology
outright. Decimation is the operation that produces a genuinely coarser SURFACE,
and its absence — not any structural objection — is the only reason mesh had no
substitutive ladder (``docs/specs/MESH_NODE_SPEC.md`` §9).

The dispatcher offers ``cluster`` and ``qem``. ``cluster`` is the vectorized
implementation in this module: snap vertices to a grid, collapse
each occupied cell to its centroid, reindex the faces, drop the triangles that
collapsed to a line. It is O(V log V) per pass and fully vectorized, which matters
because the writer's cap is 2**27 vertices and a Python edge-collapse loop is
unusable at that scale. It has no quality-driven failure mode — no seeding, no
convergence criterion, nothing to diverge — and refuses only input that is not a
surface at all (no faces, or every vertex coincident).

The grid spacing is found by bisection, so the cost is several passes rather than
one. Measured ~0.2 MV/s per pass, with the bracket-convergence exit keeping it to a
handful of passes rather than the full iteration budget.

**The clustering representative is the plain centroid; a Garland-Heckbert quadric
placement inside each grid cell was tried and removed.** The theory says the quadric minimizer preserves
creases a centroid rounds off, and the algebra does work — three orthogonal planes
solve to their exact corner. It moved vertices (up to 0.067 on a unit sphere, most
of them by something) but improved no measurable quality: same vertex counts, same
face counts, same topology, different positions that were not better positions. On
a closed sphere the centroid was marginally BETTER (mean radial error 0.00659
against 0.00667); on a sharp wedge the two agreed to five decimals, because a
crease is two
planes, giving a rank-2 system that is singular and falls back to the centroid
anyway; on a cube the two were bit-identical, because a cube's corner is a lone
vertex in its cell and the centroid already lands on it exactly. Rank-3 cells, the
only case where the quadric can differ, are rare enough on real sampled surfaces
that nothing observable changed. It is recorded here so the next person does not
re-derive it: the win is real in the literature and absent at this grid resolution,
where the cell is already smaller than the features being preserved.

**Clustering does not preserve manifoldness, and cannot.** When a cell swallows two
sheets that were separate on the fine surface, their triangles land on the same
representatives and the result has edges shared by more than two faces. Measured on
a closed unit sphere (4098 vertices, boundary-free, chi=2), the coarse levels come
back chi=2 with 0 boundary edges at most cell sizes but hit 120 boundary and 60
non-manifold edges at one. That is a property of the algorithm rather than a defect
here, it renders correctly either way (a non-manifold edge is still just triangles),
and it is the concrete reason to keep an edge-collapse tier around rather than a
vague appeal to quality: edge collapse can refuse a collapse that would break the
link condition, and clustering has no such veto to exercise. That argument stands
independently of the quadric note above — it is about topology, not placement.

Related: dropping the DUPLICATE triangles clustering produces is not cosmetic. With
them the sphere above measures chi=86 with 96 non-manifold edges; without them,
chi=2 with 60. The duplicates visually seal the degenerate region while corrupting
the topology underneath it, and two co-planar opaque triangles z-fight.
"""

from __future__ import annotations

from typing import Any, Literal, NamedTuple, Sequence, cast

import numpy as np
from arbol import aprint
from numpy.typing import NDArray

# Worst-case 3-level K=4 open-annulus ladder: 9,882 vertices takes 45.9 s.
# Keep the automatic tier below the one-minute boundary.
QEM_AUTO_VERTEX_LIMIT = 10_000
DECIMATION_METHODS = frozenset({"cluster", "qem"})
# 4M exact pairs cost ~0.15 s per call, or ~3 s across a 25-pass bisection.
_ORPHAN_NEAREST_PAIR_BUDGET = 4_000_000
# 1M pairs cap each 3D distance temporary at 3M float64 elements.
_ORPHAN_DISTANCE_BLOCK_PAIR_BUDGET = 1_000_000


class DecimatedMesh(NamedTuple):
    """One coarser level: a complete, independently renderable surface."""

    vertices: NDArray[np.float32]
    faces: NDArray[np.uint32]
    normals: NDArray[np.float32] | None
    # Whatever dtype came in — uint8 / uint16 / float32 are all valid mesh
    # colours, and the averaging round-trips the input dtype rather than
    # forcing one (see :func:`_average_per_cluster`).
    colors: NDArray[Any] | None
    # Appended rather than slotted next to `colors` because this is a
    # NamedTuple: an earlier position would silently reassign every positional
    # construction site.
    scalars: Any = None
    # Maximum source-vertex displacement to its coarse representative,
    # normalized by the source spatial bounding-box diagonal.
    geometric_error: float = 0.0


def _appearance_features(
    colors: NDArray[Any] | None,
    scalars: Any,
    n_vertices: int,
) -> NDArray[np.float64] | None:
    """Normalized per-vertex appearance columns, omitting constant channels."""
    columns: list[NDArray[np.float64]] = []
    if colors is not None and colors.ndim == 2 and colors.shape[:1] == (n_vertices,):
        columns.append(np.asarray(colors, dtype=np.float64).reshape(n_vertices, -1))
    if isinstance(scalars, np.ndarray) and scalars.shape[:1] == (n_vertices,):
        columns.append(np.asarray(scalars, dtype=np.float64).reshape(n_vertices, -1))
    if not columns:
        return None
    values = np.concatenate(columns, axis=1)
    spans = np.ptp(values, axis=0)
    varying = spans > 0
    if not np.any(varying):
        return None
    values = values[:, varying]
    return (values - values.min(axis=0)) / spans[varying]


def _appearance_classes(
    colors: NDArray[Any] | None,
    scalars: Any,
    n_vertices: int,
) -> NDArray[np.int64] | None:
    """Compact exact-equivalence keys for cluster attribute barriers."""
    classes: list[NDArray[np.int64]] = []
    if colors is not None and colors.ndim == 2 and colors.shape[:1] == (n_vertices,):
        _, inverse = np.unique(colors, axis=0, return_inverse=True)
        classes.append(np.asarray(inverse, dtype=np.int64))
    if isinstance(scalars, np.ndarray) and scalars.shape[:1] == (n_vertices,):
        values = np.asarray(scalars).reshape(n_vertices, -1)
        _, inverse = np.unique(values, axis=0, return_inverse=True)
        classes.append(np.asarray(inverse, dtype=np.int64))
    if not classes:
        return None
    return np.column_stack(classes)


def _normalized_collapse_error(
    source_vertices: NDArray[np.float64],
    output_vertices: NDArray[np.float64],
    inverse: NDArray[np.int64],
    spatial_dims: tuple[int, ...],
) -> float:
    """Upper-bound source-to-coarse surface error from collapse representatives."""
    if not len(inverse):
        return 0.0
    spatial = np.asarray(spatial_dims, dtype=np.intp)
    source_spatial = source_vertices[:, spatial]
    diagonal = float(np.linalg.norm(np.ptp(source_spatial, axis=0)))
    if diagonal <= 0.0:
        return 0.0
    representative_indices = inverse.copy()
    missing = representative_indices < 0
    if np.any(missing):
        # Components whose every face collapsed have no surviving root. They still
        # belong to the source surface, so omitting them would understate the error
        # exactly when a small feature disappeared. Keep the tighter per-orphan bound
        # while its pairwise search is cheap, but fall back to the previous shared
        # representative for many-component meshes rather than making error reporting
        # quadratic in both memory and time.
        output_spatial = output_vertices[:, spatial]
        orphan_spatial = source_spatial[missing]
        pair_count = len(orphan_spatial) * len(output_spatial)
        if pair_count > _ORPHAN_NEAREST_PAIR_BUDGET:
            orphan_centroid = orphan_spatial.mean(axis=0)
            shared_index = int(
                np.argmin(np.linalg.norm(output_spatial - orphan_centroid, axis=1))
            )
            representative_indices[missing] = shared_index
        else:
            block_size = max(
                1, _ORPHAN_DISTANCE_BLOCK_PAIR_BUDGET // len(output_spatial)
            )
            missing_indices = np.flatnonzero(missing)
            for start in range(0, len(orphan_spatial), block_size):
                stop = min(start + block_size, len(orphan_spatial))
                distances = np.linalg.norm(
                    orphan_spatial[start:stop, None, :] - output_spatial[None, :, :],
                    axis=2,
                )
                representative_indices[missing_indices[start:stop]] = np.argmin(
                    distances, axis=1
                )
    representatives = output_vertices[representative_indices][:, spatial]
    return float(
        np.linalg.norm(source_spatial - representatives, axis=1).max() / diagonal
    )


def _cluster_keys(
    vertices: NDArray[np.float64],
    spatial_dims: tuple[int, ...],
    cell: float,
    attribute_classes: NDArray[np.int64] | None = None,
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
    parts = [grid]
    # Barrier columns are compared exactly; float bits are fine as an equality key.
    if barrier:
        exact = vertices[:, barrier].view(np.int64).reshape(n, len(barrier))
        parts.append(exact)
    if attribute_classes is not None:
        parts.append(attribute_classes)
    return np.concatenate(parts, axis=1)


def _validate_normal_frame(
    normals: NDArray[np.float32] | None,
    normal_dims: tuple[int, ...] | None,
    ndim: int,
) -> None:
    """Require a 3-axis frame whenever normals are supplied.

    Its own function because it is the one guard here that is about a CONCEPT
    rather than an array shape: `normal_dims` is not interchangeable with
    `spatial_dims`, and conflating them failed silently rather than loudly (a
    2-dim coarsening made `np.cross` return scalars).
    """
    if normals is None:
        return
    if normal_dims is None or len(tuple(normal_dims)) != 3:
        raise ValueError(
            "normals require normal_dims naming exactly 3 dimensions (got "
            f"{normal_dims!r}). They are recomputed from the COARSE surface, "
            "which needs the frame they are defined in — spatial_dims cannot "
            "stand in for it, since a grid may coarsen over any number of axes."
        )
    if any(i < 0 or i >= ndim for i in normal_dims):
        raise ValueError(
            f"normal_dims {tuple(normal_dims)} out of range for {ndim} dims"
        )
    # Distinct, for the same reason the writer's `validate_normal_dims_for_writing`
    # requires it: a repeated axis is not a frame. `(0, 0, 1)` would pass every
    # check above and then have `_recompute_normals` cross two vectors confined to
    # a degenerate plane, producing normals that point nowhere in particular — with
    # nothing anywhere saying so. Checked HERE and not left to the writer because
    # decimation runs BEFORE the ladder's first child is written: without this the
    # refusal arrives with a `kind=lod` group already on disk.
    if len(set(int(i) for i in normal_dims)) != 3:
        raise ValueError(
            f"normal_dims {tuple(normal_dims)} must name three DISTINCT dimensions; "
            "a repeated axis is a degenerate frame, and the normals recomputed in "
            "it would be meaningless rather than merely wrong."
        )


def _validate_decimate_inputs(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    target_vertices: int,
    normals: NDArray[np.float32] | None,
    normal_dims: tuple[int, ...] | None,
    spatial_dims: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Reject malformed input and resolve the default ``spatial_dims``.

    Split out of :func:`decimate_cluster` purely for readability — it is seven
    guards in a row with no interleaved logic, and inline they were most of that
    function's branching. Returns the resolved ``spatial_dims`` because the
    default (the first ``min(3, D)`` columns) can only be computed once the
    vertex width is known, and the caller needs the resolved value.
    """
    if vertices.ndim != 2 or vertices.shape[1] < 2:
        raise ValueError(f"vertices must be (V, D>=2), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {faces.shape}")
    if faces.shape[0] == 0:
        raise ValueError(
            "cannot decimate a mesh with no faces: there is no surface to coarsen, "
            "and every vertex would come back unreferenced"
        )
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
    _validate_normal_frame(normals, normal_dims, d)
    return spatial_dims


def decimate_cluster(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    target_vertices: int,
    normals: NDArray[np.float32] | None = None,
    normal_dims: tuple[int, ...] | None = None,
    colors: NDArray[Any] | None = None,
    scalars: Any = None,
    spatial_dims: tuple[int, ...] | None = None,
    attribute_weight: float = 0.0,
    max_iterations: int = 24,
) -> DecimatedMesh:
    """Reduce ``vertices`` toward ``target_vertices`` by vertex clustering.

    Args:
        vertices: ``(V, D)`` positions, D >= 2.
        faces: ``(F, 3)`` indices into ``vertices``.
        target_vertices: Desired vertex count of the result. Approximate — the grid
            cannot hit an arbitrary count exactly, so the search stops at the
            coarsest spacing that still leaves at least this many vertices.
        normals: Optional ``(V, 3)``. Recomputed from the coarse geometry rather
            than averaged: an averaged normal describes the FINE surface and would
            light the coarse one wrongly at exactly the creases clustering moved.
        normal_dims: The three dimension indices ``normals`` describes. **Required
            with** ``normals``, and NOT interchangeable with ``spatial_dims`` — the
            two answer different questions. ``spatial_dims`` is which axes the grid
            merges over, and may be any number of them; ``normal_dims`` is the
            3-axis frame a normal vector lives in. Deriving one from the other
            silently produced garbage: a 2-dim coarsening made ``np.cross`` return
            scalars (which then broke the accumulation), and a 4-dim one made it
            raise.
        colors: Optional ``(V, C)`` per-vertex colours — uint8, uint16 or float32
            are all valid mesh colours — averaged within each cluster. The input
            dtype round-trips: integers are rounded and clipped to their own
            range, floats are left unclipped (an HDR colour exceeds 1.0
            legitimately).
        scalars: Optional per-vertex ``(V,)`` (or ``(V, 1)``) values, averaged
            within each cluster like ``colors`` but NOT re-quantized: an integral
            input comes back float32, which is what ``write_scalars`` stores
            anyway, so a 0/1 field keeps its fractional cluster means instead of
            being hard-classified on the coarse levels only. A coarse level that
            dropped scalars entirely would carry the caller's ``colormap`` with
            nothing to map, so it would render unmapped while the finest level is
            mapped — a visible pop at every LOD switch. A non-array (uniform
            broadcast) value is passed through untouched: it applies to every
            vertex, so there is nothing to merge.
        spatial_dims: Which columns are spatial. Defaults to the first ``min(3, D)``.
        max_iterations: Bisection budget for the cell-size search.

    Returns:
        A :class:`DecimatedMesh` with at least one triangle. The search prefers the
        coarsest spacing that still leaves ``target_vertices``, and falls back to the
        least-reduced level it can build rather than returning something emptier.

    Raises:
        ValueError: If inputs are malformed, ``target_vertices`` < 4, or the input
            surface is degenerate enough that no triangle survives. That last case
            is an error rather than an empty result on purpose: the
            substitutive-LOD wrapper keeps a decimated level only when its vertex
            count strictly exceeds the previous one's, so an empty return (count
            0) would be dropped in silence — a degenerate mesh would come out as a
            plain leaf with nothing naming which mesh caused it.
    """
    if not np.isfinite(attribute_weight) or attribute_weight < 0:
        raise ValueError(
            f"attribute_weight must be finite and >= 0, got {attribute_weight}"
        )
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    spatial_dims = _validate_decimate_inputs(
        vertices, faces, target_vertices, normals, normal_dims, spatial_dims
    )

    if vertices.shape[0] <= target_vertices:
        return DecimatedMesh(vertices, faces, normals, colors, scalars)

    v64 = vertices.astype(np.float64)
    attribute_classes = (
        _appearance_classes(colors, scalars, len(vertices))
        if attribute_weight > 0
        else None
    )
    spatial = np.asarray(spatial_dims, dtype=np.intp)
    extent = float(
        np.max(v64[:, spatial].max(axis=0) - v64[:, spatial].min(axis=0)) or 1.0
    )

    # Search the cell size by bisection, seeded from the ideal cubic packing so the
    # first probe is usually within a factor of two.
    #
    # The occupied-cell count trends downward as `cell` grows but is NOT pointwise
    # monotone, so this is a heuristic search rather than a proof-carrying binary
    # search — an earlier comment here claimed monotonicity and was wrong.
    # `floor(p / cell)` grids at different spacings are not nested: 10.1 and 10.9
    # share a cell at 2.0 (both floor to 5) and split at 2.1 (4 and 5), so a
    # COARSER grid can yield MORE clusters. Bisection can therefore skip an
    # interval holding a tighter fit.
    #
    # That costs fit quality, not correctness, and the difference matters: every
    # candidate is VALIDATED before it can become `best` (at least `target_vertices`
    # vertices, at least one surviving triangle), and the caller additionally
    # requires each ladder level to be strictly coarser than the last. So a missed
    # interval yields a level that is less tight than it could have been, never one
    # that is wrong.
    ratio = max(target_vertices / vertices.shape[0], 1e-9)
    lo, hi = extent * 1e-6, extent
    guess = extent * (ratio ** (1.0 / len(spatial_dims)))
    best: DecimatedMesh | None = None
    for _ in range(max_iterations):
        candidate = _cluster_once(
            v64,
            faces,
            spatial_dims,
            guess,
            normals,
            normal_dims,
            colors,
            scalars,
            attribute_classes,
        )
        count = candidate.vertices.shape[0]
        # A candidate with no surviving triangle is not a usable level regardless of
        # its vertex count, so it must not become `best` — it would be returned as a
        # surface with nothing to draw.
        if candidate.faces.shape[0] and count >= target_vertices:
            best = candidate
            if count <= target_vertices * 1.1:
                break
            lo = guess
        else:
            hi = guess
        # Stop once the bracket itself has converged, not just when the count lands
        # inside the 10% window. A grid cannot hit a SMALL target within 10% at all
        # (20 vertices allows a window of 2), so without this the search burns the
        # whole budget refining a spacing that no longer changes the result — and
        # every pass is a full O(V log V) reclustering, which is minutes at the
        # 2**27 vertices this function exists to handle.
        if hi - lo <= lo * 1e-3:
            break
        guess = 0.5 * (lo + hi)

    if best is None:
        # Every probe overshot. Return the least-reduced one we can still build.
        best = _cluster_once(
            v64,
            faces,
            spatial_dims,
            lo,
            normals,
            normal_dims,
            colors,
            scalars,
            attribute_classes,
        )
    if best.faces.shape[0] == 0:
        # Reachable only for input that has no surface to begin with — every
        # triangle collinear, or every vertex coincident — since any real triangle
        # survives at a fine enough spacing and `lo` is 1e-6 of the extent.
        raise ValueError(
            f"decimation collapsed every triangle of a {vertices.shape[0]}-vertex, "
            f"{faces.shape[0]}-face mesh, leaving no surface. The input is "
            "degenerate (collinear or coincident vertices) rather than merely fine."
        )
    return best


def resolve_decimation_method(
    method: str,
    n_vertices: int,
    *,
    spatial_ndim: int | None = None,
    announce: bool = True,
) -> Literal["cluster", "qem"]:
    """Resolve ``auto`` once from source size and QEM's dimensional envelope."""
    normalized = str(method).replace("-", "_")
    if normalized == "qem" and spatial_ndim is not None and spatial_ndim < 3:
        raise ValueError(
            "mesh decimation method 'qem' requires at least 3 coarsening "
            f"dimensions; got {spatial_ndim}"
        )
    if normalized == "auto":
        qem_has_normal_space = spatial_ndim is None or spatial_ndim >= 3
        normalized = (
            "qem"
            if qem_has_normal_space and n_vertices <= QEM_AUTO_VERTEX_LIMIT
            else "cluster"
        )
        if announce:
            if not qem_has_normal_space:
                aprint(
                    "  📐 Mesh decimation auto-selected 'cluster': "
                    f"{spatial_ndim} coarsening dimensions is below QEM's "
                    "3-dimension minimum."
                )
            else:
                comparison = "at or below" if normalized == "qem" else "above"
                aprint(
                    f"  📐 Mesh decimation auto-selected '{normalized}': "
                    f"{n_vertices:,} vertices is {comparison} the "
                    f"{QEM_AUTO_VERTEX_LIMIT:,}-vertex QEM limit."
                )
    if normalized not in DECIMATION_METHODS:
        raise ValueError(
            "mesh decimation method must be one of ['auto', 'cluster', 'qem']; "
            f"got {method!r}"
        )
    return cast(Literal["cluster", "qem"], normalized)


def decimate(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    target_vertices: int,
    method: str = "auto",
    **kwargs: Any,
) -> DecimatedMesh:
    """Dispatch to the selected mesh decimator."""
    return decimate_ladder(
        vertices,
        faces,
        target_vertices=[target_vertices],
        method=method,
        **kwargs,
    )[0]


def decimate_ladder(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    target_vertices: Sequence[int],
    method: str = "auto",
    **kwargs: Any,
) -> list[DecimatedMesh]:
    """Dispatch a target sequence, sharing one collapse sequence for QEM."""
    spatial_dims = kwargs.get("spatial_dims")
    spatial_ndim = (
        len(spatial_dims) if spatial_dims is not None else min(3, vertices.shape[1])
    )
    resolved = resolve_decimation_method(
        method, len(vertices), spatial_ndim=spatial_ndim
    )
    if resolved == "cluster":
        return [
            decimate_cluster(vertices, faces, target_vertices=target, **kwargs)
            for target in target_vertices
        ]
    from .qem import decimate_qem_ladder

    return decimate_qem_ladder(
        vertices, faces, target_vertices=target_vertices, **kwargs
    )


def _average_per_cluster(
    values: NDArray[Any],
    inverse: NDArray[np.int64],
    counts: NDArray[np.float64],
    n_clusters: int,
    *,
    quantize: bool,
) -> NDArray[Any]:
    """Cluster-mean of a per-vertex attribute, cast back to a sensible dtype.

    The mean itself is float64 whatever came in; the CAST is what has to respect
    the input, and hardcoding ``uint8`` there was wrong in both directions. Mesh
    colours are uint8, uint16 or float32 (``MESH_NODE_SPEC.md`` §2), so a float32
    SDR colour in [0, 1] truncated to 0 — every coarse level black — and a uint16
    one was clipped at 255, i.e. black again against a 65535-scale sibling.

    ``quantize`` says whether the input dtype is the STORAGE dtype:

    * Colours pass ``True``. An integer colour dtype really is the on-disk scale,
      so the mean is rounded (truncation biases every average downward) and
      clipped to that dtype's range.
    * Scalars pass ``False``. ``write_scalars`` casts to float32 unconditionally,
      so an integer scalars array never reaches disk as an integer and rounding
      it only throws the cluster mean away — and ``np.rint`` is half-to-even, so
      a 0/1 field averaging to exactly 0.5 rounds DOWN. The coarse levels would
      come back hard-classified against a blended finest level: the same pop
      again. Integral input therefore becomes float32 (its on-disk dtype) and
      floating input keeps its own.

    Floats are never clipped: an HDR colour above 1.0 is legitimate data and a
    scalar field has no bounded range to clip to.

    Shape-agnostic beyond the first axis, so it serves both ``(V, C)`` colours
    and ``(V,)`` / ``(V, 1)`` scalars.
    """
    flat = values.reshape(values.shape[0], -1)
    acc = np.zeros((n_clusters, flat.shape[1]), dtype=np.float64)
    for col in range(flat.shape[1]):
        acc[:, col] = np.bincount(
            inverse, weights=flat[:, col].astype(np.float64), minlength=n_clusters
        )
    mean = acc / np.maximum(counts, 1)[:, None]
    integral = np.issubdtype(values.dtype, np.integer)
    if integral and quantize:
        info = np.iinfo(values.dtype)
        mean = np.clip(np.rint(mean), info.min, info.max)
    out_dtype = np.dtype(np.float32) if integral and not quantize else values.dtype
    return mean.astype(out_dtype).reshape((n_clusters, *values.shape[1:]))


def _cluster_once(
    v64: NDArray[np.float64],
    faces: NDArray[np.uint32],
    spatial_dims: tuple[int, ...],
    cell: float,
    normals: NDArray[np.float32] | None,
    normal_dims: tuple[int, ...] | None,
    colors: NDArray[Any] | None,
    scalars: Any = None,
    attribute_classes: NDArray[np.int64] | None = None,
) -> DecimatedMesh:
    """One clustering pass at a fixed cell size."""
    keys = _cluster_keys(
        v64, spatial_dims, max(cell, 1e-12), attribute_classes=attribute_classes
    )
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

    new_v = centroid
    new_f = inverse[faces.reshape(-1)].reshape(faces.shape).astype(np.uint32)
    a, b, c = new_f[:, 0], new_f[:, 1], new_f[:, 2]
    new_f = new_f[(a != b) & (b != c) & (a != c)]
    # Clustering also creates DUPLICATE triangles (two fine faces collapsing onto
    # the same coarse corners). They draw the same pixels twice — wasted index
    # bytes and rasterization, and an inflated face count for the level — so drop
    # them.
    if new_f.shape[0]:
        canonical = np.sort(new_f, axis=1)
        _, keep = np.unique(canonical, axis=0, return_index=True)
        new_f = new_f[np.sort(keep)]

    # Average the colours and the scalars over the FULL cluster set, before any
    # compaction below, so every contributing fine vertex is counted exactly once.
    new_colors = None
    if colors is not None:
        new_colors = _average_per_cluster(
            colors, inverse, counts, n_clusters, quantize=True
        )

    # A scalar field is merged exactly like a colour is — it IS the colour, one
    # colormap lookup later. Only a genuinely per-vertex array is averaged; a
    # uniform broadcast value applies to every vertex, so it is forwarded as-is
    # (the same distinction the adder draws for a uniform colour).
    scalars_per_vertex = isinstance(scalars, np.ndarray) and scalars.shape[:1] == (
        v64.shape[0],
    )
    new_scalars = scalars
    if scalars_per_vertex:
        new_scalars = _average_per_cluster(
            scalars, inverse, counts, n_clusters, quantize=False
        )

    # Drop representatives no surviving face references. They are not harmless
    # padding: a vertex with no incident face has no defined normal, so it would
    # take the arbitrary fallback and drag the level's shading statistics with it
    # — and it still costs vertex-buffer bytes on every upload.
    if new_f.shape[0]:
        referenced = np.unique(new_f)
        if referenced.size != new_v.shape[0]:
            remap = np.full(new_v.shape[0], -1, dtype=np.int64)
            remap[referenced] = np.arange(referenced.size)
            new_f = remap[new_f].astype(np.uint32)
            new_v = new_v[referenced]
            if new_colors is not None:
                new_colors = new_colors[referenced]
            if scalars_per_vertex:
                new_scalars = new_scalars[referenced]
            inverse = remap[inverse]

    geometric_error = _normalized_collapse_error(v64, new_v, inverse, spatial_dims)

    new_normals = None
    if normals is not None and normal_dims is not None:
        # The NORMAL frame, not the coarsening axes — see `decimate_cluster`.
        frame = np.asarray(tuple(normal_dims), dtype=np.intp)
        new_normals = _recompute_normals(new_v[:, frame], new_f, new_v.shape[0])

    return DecimatedMesh(
        vertices=new_v.astype(np.float32),
        faces=new_f,
        normals=new_normals,
        colors=new_colors,
        scalars=new_scalars,
        geometric_error=geometric_error,
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


__all__ = [
    "DECIMATION_METHODS",
    "DecimatedMesh",
    "QEM_AUTO_VERTEX_LIMIT",
    "decimate",
    "decimate_cluster",
    "decimate_ladder",
    "resolve_decimation_method",
]
