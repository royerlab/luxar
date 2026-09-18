"""Topology-preserving Garland-Heckbert mesh decimation."""

from __future__ import annotations

import heapq
from typing import Any, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray

from .decimate import (
    DecimatedMesh,
    _appearance_features,
    _average_per_cluster,
    _normalized_collapse_error,
    _recompute_normals,
    _validate_decimate_inputs,
)

# Dimensionless: large enough to dominate the few face quadrics at a rim vertex.
_BOUNDARY_QUADRIC_WEIGHT = 1000.0
_RELATIVE_GEOMETRY_TOLERANCE = 1e-12


def _coordinate_extent(positions: NDArray[np.float64]) -> float:
    """Largest coordinate span, used to scale geometric tolerances."""
    flattened = positions.reshape(-1, positions.shape[-1])
    return float(np.ptp(flattened, axis=0).max(initial=0.0))


def _face_quadrics(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """One homogeneous squared-distance quadric per triangle."""
    n_faces, _, ndim = positions.shape
    quadrics = np.zeros((n_faces, ndim + 1, ndim + 1), dtype=np.float64)
    if ndim == 3:
        edge1 = positions[:, 1] - positions[:, 0]
        edge2 = positions[:, 2] - positions[:, 0]
        normal = np.cross(edge1, edge2)
        length = np.linalg.norm(normal, axis=1)
        valid = (
            length
            > (np.linalg.norm(edge1, axis=1) * np.linalg.norm(edge2, axis=1))
            * _RELATIVE_GEOMETRY_TOLERANCE
        )
        normal[valid] /= length[valid, None]
        plane = np.concatenate(
            [normal, -np.einsum("ij,ij->i", normal, positions[:, 0])[:, None]], axis=1
        )
        quadrics[valid] = np.einsum("fi,fj->fij", plane[valid], plane[valid])
        return quadrics

    extent = _coordinate_extent(positions)
    identity = np.eye(ndim)
    for index, triangle in enumerate(positions):
        edges = (triangle[1:] - triangle[0]).T
        basis, singular, _ = np.linalg.svd(edges, full_matrices=False)
        rank = int(np.count_nonzero(singular > extent * _RELATIVE_GEOMETRY_TOLERANCE))
        projector = identity - basis[:, :rank] @ basis[:, :rank].T
        offset = -projector @ triangle[0]
        quadrics[index, :ndim, :ndim] = projector
        quadrics[index, :ndim, ndim] = offset
        quadrics[index, ndim, :ndim] = offset
        quadrics[index, ndim, ndim] = float(triangle[0] @ projector @ triangle[0])
    return quadrics


def _vertex_quadrics(
    positions: NDArray[np.float64], faces: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Sum incident face and boundary-line quadrics onto each vertex."""
    face_quadrics = _face_quadrics(positions[faces])
    out = np.zeros((len(positions), positions.shape[1] + 1, positions.shape[1] + 1))
    for corner in range(3):
        np.add.at(out, faces[:, corner], face_quadrics)

    edges = np.sort(
        np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]
    edge_vectors = positions[boundary_edges[:, 1]] - positions[boundary_edges[:, 0]]
    lengths = np.linalg.norm(edge_vectors, axis=1)
    valid = lengths > _coordinate_extent(positions) * _RELATIVE_GEOMETRY_TOLERANCE
    for (u, v), edge_vector, length in zip(
        boundary_edges[valid], edge_vectors[valid], lengths[valid], strict=True
    ):
        direction = edge_vector / length
        projector = np.eye(positions.shape[1]) - np.outer(direction, direction)
        point = positions[u]
        offset = -projector @ point
        quadric = np.zeros_like(out[0])
        quadric[:-1, :-1] = projector
        quadric[:-1, -1] = offset
        quadric[-1, :-1] = offset
        quadric[-1, -1] = float(point @ projector @ point)
        out[u] += _BOUNDARY_QUADRIC_WEIGHT * quadric
        out[v] += _BOUNDARY_QUADRIC_WEIGHT * quadric
    return out


def _require_nondegenerate_surface(
    positions: NDArray[np.float64], faces: NDArray[np.uint32]
) -> None:
    """Refuse input with no triangle spanning a two-dimensional surface."""
    triangles = positions[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    tolerance = _coordinate_extent(positions) * _RELATIVE_GEOMETRY_TOLERANCE
    if np.any(np.linalg.matrix_rank(edges, tol=tolerance) >= 2):
        return
    raise ValueError(
        f"the input {positions.shape[0]}-vertex, {faces.shape[0]}-face mesh has no "
        "triangle spanning a surface; its vertices are collinear or coincident"
    )


def _edge_target(
    u: int,
    v: int,
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Return the minimum-cost endpoint, midpoint, or solved QEM placement."""
    quadric = quadrics[u] + quadrics[v]
    matrix = quadric[:-1, :-1]
    rhs = -quadric[:-1, -1]
    candidates = [positions[u], positions[v], 0.5 * (positions[u] + positions[v])]
    solved = _solve_system(matrix, rhs)
    if solved is not None and np.isfinite(solved).all():
        # An ill-conditioned solve can escape far beyond the edge, especially in nD
        # where there is no face-orientation veto. Keeping every accepted placement
        # inside this envelope also keeps every decimated level inside the input bbox.
        lower = np.minimum(positions[u], positions[v])
        upper = np.maximum(positions[u], positions[v])
        if np.all(solved >= lower) and np.all(solved <= upper):
            candidates.append(solved)
    costs = [_quadric_cost(candidate, quadric) for candidate in candidates]
    best = min(range(len(costs)), key=costs.__getitem__)
    return max(costs[best], 0.0), np.asarray(candidates[best]).copy()


def _solve_system(
    matrix: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Solve the common 3x3 QEM system without a small-array LAPACK call."""
    if matrix.shape != (3, 3):
        if np.linalg.matrix_rank(matrix, tol=1e-12) < matrix.shape[0]:
            return None
        try:
            return np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)
        except np.linalg.LinAlgError:
            return None
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    cofactor00 = e * i - f * h
    cofactor01 = f * g - d * i
    cofactor02 = d * h - e * g
    determinant = a * cofactor00 + b * cofactor01 + c * cofactor02
    scale = float(np.max(np.abs(matrix)))
    if scale == 0.0 or abs(determinant) <= np.finfo(np.float64).eps * scale**3:
        return None
    inverse = np.array(
        [
            [cofactor00, c * h - b * i, b * f - c * e],
            [cofactor01, a * i - c * g, c * d - a * f],
            [cofactor02, b * g - a * h, a * e - b * d],
        ]
    )
    return np.asarray(inverse @ rhs / determinant, dtype=np.float64)


def _quadric_cost(position: NDArray[np.float64], quadric: NDArray[np.float64]) -> float:
    """Evaluate a homogeneous QEM quadric without allocating the trailing one."""
    matrix = quadric[:-1, :-1]
    linear = quadric[:-1, -1]
    return float(
        position @ matrix @ position + 2.0 * linear @ position + quadric[-1, -1]
    )


def _edge_cost(
    u: int,
    v: int,
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
    attribute_sums: NDArray[np.float64] | None = None,
    attribute_counts: NDArray[np.float64] | None = None,
    attribute_scale: float = 0.0,
) -> float:
    """Cheap heap-ordering cost evaluated at the edge midpoint."""
    midpoint = 0.5 * (positions[u] + positions[v])
    cost = max(_quadric_cost(midpoint, quadrics[u] + quadrics[v]), 0.0)
    if attribute_sums is not None and attribute_counts is not None:
        count_u = attribute_counts[u]
        count_v = attribute_counts[v]
        mean_delta = attribute_sums[u] / count_u - attribute_sums[v] / count_v
        cost += (
            attribute_scale
            * count_u
            * count_v
            / (count_u + count_v)
            * float(mean_delta @ mean_delta)
        )
    return cost


def _edge_faces(u: int, v: int, vertex_faces: list[set[int]]) -> set[int]:
    """Triangles incident to both endpoints; ``vertex_faces`` is active-only."""
    return vertex_faces[u] & vertex_faces[v]


def _link_condition(
    u: int,
    v: int,
    faces: NDArray[np.int64],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
    boundary_vertices: NDArray[np.bool_],
) -> bool:
    """Veto collapses that change manifold topology or cross a boundary."""
    incident = _edge_faces(u, v, vertex_faces)
    if len(incident) not in (1, 2):
        return False
    opposite = {
        int(vertex)
        for face_index in incident
        for vertex in faces[face_index]
        if vertex != u and vertex != v
    }
    if (neighbors[u] - {v}) & (neighbors[v] - {u}) != opposite:
        return False
    endpoint_faces = vertex_faces[u] | vertex_faces[v]
    if endpoint_faces == incident:
        return False
    if any(vertex_faces[vertex] <= incident for vertex in opposite):
        return False

    boundary_u = bool(boundary_vertices[u])
    boundary_v = bool(boundary_vertices[v])
    if boundary_u or boundary_v:
        return len(incident) == 1 and boundary_u and boundary_v
    if len(neighbors[u] | neighbors[v] | {u, v}) == 4:
        return False
    return len(incident) == 2


def _rebuild_neighbors(
    vertices: set[int],
    faces: NDArray[np.int64],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
) -> None:
    """Recompute local one-rings after replacing one endpoint in nearby faces."""
    for vertex in vertices:
        adjacent: set[int] = set()
        for face_index in vertex_faces[vertex]:
            adjacent.update(int(value) for value in faces[face_index])
        adjacent.discard(vertex)
        neighbors[vertex] = adjacent


def _refresh_boundary_vertices(
    vertices: set[int],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
    boundary_vertices: NDArray[np.bool_],
) -> None:
    """Refresh boundary membership after a local topology rewrite."""
    for vertex in vertices:
        boundary_vertices[vertex] = any(
            len(_edge_faces(vertex, other, vertex_faces)) == 1
            for other in neighbors[vertex]
        )


def _aggregate_attributes(
    inverse: NDArray[np.int64],
    n_output: int,
    colors: NDArray[Any] | None,
    scalars: Any,
) -> tuple[NDArray[Any] | None, Any]:
    """Average per-input-vertex channels over the final collapse components."""
    counts = np.bincount(inverse, minlength=n_output).astype(np.float64)
    new_colors = (
        _average_per_cluster(colors, inverse, counts, n_output, quantize=True)
        if colors is not None
        else None
    )
    scalars_per_vertex = isinstance(scalars, np.ndarray) and scalars.shape[:1] == (
        len(inverse),
    )
    new_scalars = scalars
    if scalars_per_vertex:
        new_scalars = _average_per_cluster(
            scalars, inverse, counts, n_output, quantize=False
        )
    return new_colors, new_scalars


HeapEntry = tuple[float, int, int, int, int, int]


class _CollapseState(NamedTuple):
    heap: list[HeapEntry]
    serial: int
    parent: NDArray[np.int64]
    remaining: int


def _build_topology(
    faces: NDArray[np.int64], n_vertices: int
) -> tuple[list[set[int]], list[set[int]], NDArray[np.bool_]]:
    """Build incident-face and one-ring tables for the collapse loop."""
    vertex_faces: list[set[int]] = [set() for _ in range(n_vertices)]
    neighbors: list[set[int]] = [set() for _ in range(n_vertices)]
    for face_index, face in enumerate(faces):
        a, b, c = (int(value) for value in face)
        vertex_faces[a].add(face_index)
        vertex_faces[b].add(face_index)
        vertex_faces[c].add(face_index)
        neighbors[a].update((b, c))
        neighbors[b].update((a, c))
        neighbors[c].update((a, b))
    boundary_vertices = np.zeros(n_vertices, dtype=bool)
    _refresh_boundary_vertices(
        set(range(n_vertices)), neighbors, vertex_faces, boundary_vertices
    )
    return vertex_faces, neighbors, boundary_vertices


def _push_edge(
    heap: list[HeapEntry],
    serial: int,
    u: int,
    v: int,
    *,
    alive: NDArray[np.bool_],
    barrier_dims: tuple[int, ...],
    vertices: NDArray[np.float32],
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
    versions: NDArray[np.int64],
    attribute_sums: NDArray[np.float64] | None = None,
    attribute_counts: NDArray[np.float64] | None = None,
    attribute_scale: float = 0.0,
) -> int:
    """Push one current edge, returning the next stable tie-break serial."""
    if u == v or not alive[u] or not alive[v]:
        return serial
    u, v = sorted((u, v))
    if barrier_dims and not np.array_equal(
        vertices[u, barrier_dims], vertices[v, barrier_dims]
    ):
        return serial
    serial += 1
    heapq.heappush(
        heap,
        (
            _edge_cost(
                u,
                v,
                positions,
                quadrics,
                attribute_sums,
                attribute_counts,
                attribute_scale,
            ),
            u,
            v,
            int(versions[u]),
            int(versions[v]),
            serial,
        ),
    )
    return serial


def _build_heap(
    neighbors: list[set[int]],
    *,
    alive: NDArray[np.bool_],
    barrier_dims: tuple[int, ...],
    vertices: NDArray[np.float32],
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
    versions: NDArray[np.int64],
    attribute_sums: NDArray[np.float64] | None = None,
    attribute_counts: NDArray[np.float64] | None = None,
    attribute_scale: float = 0.0,
) -> tuple[list[HeapEntry], int]:
    """Build the initial edge heap and its last serial number."""
    heap: list[HeapEntry] = []
    serial = 0
    for u, adjacent in enumerate(neighbors):
        for v in adjacent:
            if u < v:
                serial = _push_edge(
                    heap,
                    serial,
                    u,
                    v,
                    alive=alive,
                    barrier_dims=barrier_dims,
                    vertices=vertices,
                    positions=positions,
                    quadrics=quadrics,
                    versions=versions,
                    attribute_sums=attribute_sums,
                    attribute_counts=attribute_counts,
                    attribute_scale=attribute_scale,
                )
    return heap, serial


def _apply_collapse(
    u: int,
    v: int,
    target: NDArray[np.float64],
    *,
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
    work_faces: NDArray[np.int64],
    active_faces: NDArray[np.bool_],
    alive: NDArray[np.bool_],
    parent: NDArray[np.int64],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
    boundary_vertices: NDArray[np.bool_],
    attribute_sums: NDArray[np.float64] | None = None,
    attribute_counts: NDArray[np.float64] | None = None,
) -> None:
    """Apply one accepted edge collapse and rebuild its local topology."""
    affected = neighbors[u] | neighbors[v] | {u, v}
    changed_faces = vertex_faces[u] | vertex_faces[v]
    for face_index in changed_faces:
        assert active_faces[face_index]
        old_face = work_faces[face_index].copy()
        for vertex in old_face:
            vertex_faces[int(vertex)].discard(face_index)
        work_faces[face_index][work_faces[face_index] == v] = u
        if len(set(int(value) for value in work_faces[face_index])) < 3:
            active_faces[face_index] = False
            continue
        for vertex in work_faces[face_index]:
            vertex_faces[int(vertex)].add(face_index)

    positions[u] = target
    quadrics[u] += quadrics[v]
    if attribute_sums is not None and attribute_counts is not None:
        attribute_sums[u] += attribute_sums[v]
        attribute_counts[u] += attribute_counts[v]
    alive[v] = False
    parent[v] = u
    _rebuild_neighbors(affected, work_faces, neighbors, vertex_faces)
    _refresh_boundary_vertices(affected, neighbors, vertex_faces, boundary_vertices)


def _preserves_face_orientation(
    u: int,
    v: int,
    target: NDArray[np.float64],
    *,
    positions: NDArray[np.float64],
    work_faces: NDArray[np.int64],
    vertex_faces: list[set[int]],
) -> bool:
    """Whether every surviving incident triangle keeps its orientation."""
    if positions.shape[1] != 3:
        return True
    incident = _edge_faces(u, v, vertex_faces)
    changed_indices = [
        index for index in vertex_faces[u] | vertex_faces[v] if index not in incident
    ]
    if not changed_indices:
        return True
    faces = work_faces[changed_indices]
    before = positions[faces]
    after = before.copy()
    after[(faces == u) | (faces == v)] = target
    before_edge1 = before[:, 1] - before[:, 0]
    before_edge2 = before[:, 2] - before[:, 0]
    after_edge1 = after[:, 1] - after[:, 0]
    after_edge2 = after[:, 2] - after[:, 0]
    # Binet-Cauchy: cross(e1, e2) dot cross(e1', e2') without two cross products.
    orientation = np.einsum("ij,ij->i", before_edge1, after_edge1) * np.einsum(
        "ij,ij->i", before_edge2, after_edge2
    ) - np.einsum("ij,ij->i", before_edge1, after_edge2) * np.einsum(
        "ij,ij->i", before_edge2, after_edge1
    )
    return bool(np.all(orientation > 0.0))


def _collapse_to_target(
    target_vertices: int,
    *,
    vertices: NDArray[np.float32],
    positions: NDArray[np.float64],
    barrier_dims: tuple[int, ...],
    work_faces: NDArray[np.int64],
    active_faces: NDArray[np.bool_],
    alive: NDArray[np.bool_],
    versions: NDArray[np.int64],
    quadrics: NDArray[np.float64],
    vertex_faces: list[set[int]],
    neighbors: list[set[int]],
    boundary_vertices: NDArray[np.bool_],
    state: _CollapseState | None = None,
    attribute_sums: NDArray[np.float64] | None = None,
    attribute_counts: NDArray[np.float64] | None = None,
    attribute_scale: float = 0.0,
) -> _CollapseState:
    """Collapse valid edges until the referenced surface reaches its target."""
    if state is None:
        heap, serial = _build_heap(
            neighbors,
            alive=alive,
            barrier_dims=barrier_dims,
            vertices=vertices,
            positions=positions,
            quadrics=quadrics,
            versions=versions,
            attribute_sums=attribute_sums,
            attribute_counts=attribute_counts,
            attribute_scale=attribute_scale,
        )
        parent = np.arange(len(vertices), dtype=np.int64)
        remaining = int(np.unique(work_faces).size)
    else:
        heap, serial, parent, remaining = state
    while remaining > target_vertices and heap:
        _, u, v, version_u, version_v, _ = heapq.heappop(heap)
        if not alive[u] or not alive[v]:
            continue
        if (
            versions[u] != version_u
            or versions[v] != version_v
            or v not in neighbors[u]
        ):
            continue
        if not _link_condition(
            u, v, work_faces, neighbors, vertex_faces, boundary_vertices
        ):
            continue
        _, target = _edge_target(u, v, positions, quadrics)
        if not _preserves_face_orientation(
            u,
            v,
            target,
            positions=positions,
            work_faces=work_faces,
            vertex_faces=vertex_faces,
        ):
            continue
        _apply_collapse(
            u,
            v,
            target,
            positions=positions,
            quadrics=quadrics,
            work_faces=work_faces,
            active_faces=active_faces,
            alive=alive,
            parent=parent,
            neighbors=neighbors,
            vertex_faces=vertex_faces,
            boundary_vertices=boundary_vertices,
            attribute_sums=attribute_sums,
            attribute_counts=attribute_counts,
        )
        remaining -= 1
        # Only ``u`` acquired a new position and quadric. Costs for edges between
        # its neighbours are unchanged; their link condition is checked against
        # CURRENT adjacency when they eventually leave the heap.
        versions[u] += 1
        versions[v] += 1
        for other in neighbors[u]:
            serial = _push_edge(
                heap,
                serial,
                u,
                other,
                alive=alive,
                barrier_dims=barrier_dims,
                vertices=vertices,
                positions=positions,
                quadrics=quadrics,
                versions=versions,
                attribute_sums=attribute_sums,
                attribute_counts=attribute_counts,
                attribute_scale=attribute_scale,
            )
    return _CollapseState(heap, serial, parent, remaining)


def _compact_output(
    vertices: NDArray[np.float32],
    input_faces: NDArray[np.uint32],
    *,
    positions: NDArray[np.float64],
    spatial_dims: tuple[int, ...],
    work_faces: NDArray[np.int64],
    active_faces: NDArray[np.bool_],
    parent: NDArray[np.int64],
    normals: NDArray[np.float32] | None,
    normal_dims: tuple[int, ...] | None,
    colors: NDArray[Any] | None,
    scalars: Any,
) -> DecimatedMesh:
    """Compact active faces, collapse roots, and per-vertex attributes."""
    for index in range(len(parent) - 1, -1, -1):
        root = index
        while parent[root] != root:
            root = int(parent[root])
        parent[index] = root

    active_output_faces = work_faces[active_faces]
    if not len(active_output_faces):
        raise ValueError(
            f"decimation collapsed every triangle of a {vertices.shape[0]}-vertex, "
            f"{input_faces.shape[0]}-face mesh, leaving no surface. The input is "
            "degenerate (collinear or coincident vertices) rather than merely fine."
        )
    referenced = np.unique(active_output_faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[referenced] = np.arange(len(referenced))
    output_faces = np.asarray(remap[active_output_faces], dtype=np.uint32)
    root_to_output = np.full(len(vertices), -1, dtype=np.int64)
    root_to_output[referenced] = np.arange(len(referenced))
    inverse = root_to_output[parent]
    contributing = inverse >= 0

    output_vertices = vertices[referenced].astype(np.float64)
    output_vertices[:, spatial_dims] = positions[referenced]
    output_colors, output_scalars = _aggregate_attributes(
        inverse[contributing],
        len(referenced),
        colors[contributing] if colors is not None else None,
        scalars[contributing]
        if isinstance(scalars, np.ndarray) and scalars.shape[:1] == (len(vertices),)
        else scalars,
    )
    output_normals = None
    if normals is not None and normal_dims is not None:
        output_normals = _recompute_normals(
            output_vertices[:, normal_dims], output_faces, len(output_vertices)
        )
    return DecimatedMesh(
        output_vertices.astype(np.float32),
        output_faces,
        output_normals,
        output_colors,
        output_scalars,
        _normalized_collapse_error(
            vertices.astype(np.float64),
            output_vertices,
            inverse,
            spatial_dims,
        ),
    )


def decimate_qem(
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
) -> DecimatedMesh:
    """Reduce a mesh by quadric edge collapse without violating the link condition.

    Args:
        vertices: ``(N, D)`` vertex coordinates.
        faces: ``(F, 3)`` triangle indices.
        target_vertices: Approximate maximum vertex count for the result. On an open
            near-planar surface the orientation veto can stop well above this target,
            which may remove a requested coarse ladder level; use ``cluster`` when
            closely hitting the count matters more than topology preservation.
        normals: Optional per-vertex normals.
        normal_dims: Three coordinate columns defining the normal frame.
        colors: Optional per-vertex colours.
        scalars: Optional per-vertex scalars or a uniform scalar value.
        spatial_dims: Coordinate columns QEM may coarsen across. At least three
            are required; use :func:`decimate` for automatic clustering fallback.

    Raises:
        ValueError: If the inputs are invalid or the surface collapses completely.
    """
    return decimate_qem_ladder(
        vertices,
        faces,
        target_vertices=[target_vertices],
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
        spatial_dims=spatial_dims,
        attribute_weight=attribute_weight,
    )[0]


def decimate_qem_ladder(
    vertices: NDArray[np.float32],
    faces: NDArray[np.uint32],
    *,
    target_vertices: Sequence[int],
    normals: NDArray[np.float32] | None = None,
    normal_dims: tuple[int, ...] | None = None,
    colors: NDArray[Any] | None = None,
    scalars: Any = None,
    spatial_dims: tuple[int, ...] | None = None,
    attribute_weight: float = 0.0,
) -> list[DecimatedMesh]:
    """Build several QEM levels from one collapse sequence.

    Results follow ``target_vertices`` order. Each snapshot aggregates attributes
    from the original vertices rather than averaging an already-coarsened level.
    """
    targets = [int(target) for target in target_vertices]
    if not targets:
        return []
    if not np.isfinite(attribute_weight) or attribute_weight < 0:
        raise ValueError(
            f"attribute_weight must be finite and >= 0, got {attribute_weight}"
        )
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    input_faces = np.ascontiguousarray(faces, dtype=np.uint32)
    spatial_dims = _validate_decimate_inputs(
        vertices, input_faces, min(targets), normals, normal_dims, spatial_dims
    )
    if len(spatial_dims) < 3:
        raise ValueError(
            "QEM mesh decimation requires at least 3 coarsening dimensions; "
            f"got {len(spatial_dims)}"
        )

    positions = vertices[:, spatial_dims].astype(np.float64)
    if min(targets) < len(vertices):
        _require_nondegenerate_surface(positions, input_faces)
    barrier_dims = tuple(
        index for index in range(vertices.shape[1]) if index not in spatial_dims
    )
    work_faces = input_faces.astype(np.int64)
    active_faces = np.ones(len(work_faces), dtype=bool)
    alive = np.ones(len(vertices), dtype=bool)
    versions = np.zeros(len(vertices), dtype=np.int64)
    quadrics = _vertex_quadrics(positions, work_faces)
    attribute_features = (
        _appearance_features(colors, scalars, len(vertices))
        if attribute_weight > 0
        else None
    )
    attribute_sums = (
        attribute_features.copy() if attribute_features is not None else None
    )
    attribute_counts = (
        np.ones(len(vertices), dtype=np.float64)
        if attribute_features is not None
        else None
    )
    attribute_scale = (
        attribute_weight**2 * max(_coordinate_extent(positions), 1e-12) ** 2
    )
    vertex_faces, neighbors, boundary_vertices = _build_topology(
        work_faces, len(vertices)
    )
    state: _CollapseState | None = None
    levels: dict[int, DecimatedMesh] = {}
    for target in sorted(set(targets), reverse=True):
        if target >= len(vertices):
            levels[target] = DecimatedMesh(
                vertices, input_faces, normals, colors, scalars
            )
            continue
        state = _collapse_to_target(
            target,
            vertices=vertices,
            positions=positions,
            barrier_dims=barrier_dims,
            work_faces=work_faces,
            active_faces=active_faces,
            alive=alive,
            versions=versions,
            quadrics=quadrics,
            vertex_faces=vertex_faces,
            neighbors=neighbors,
            boundary_vertices=boundary_vertices,
            state=state,
            attribute_sums=attribute_sums,
            attribute_counts=attribute_counts,
            attribute_scale=attribute_scale,
        )
        levels[target] = _compact_output(
            vertices,
            input_faces,
            positions=positions,
            spatial_dims=spatial_dims,
            work_faces=work_faces,
            active_faces=active_faces,
            parent=state.parent,
            normals=normals,
            normal_dims=normal_dims,
            colors=colors,
            scalars=scalars,
        )
    return [levels[target] for target in targets]


__all__ = ["decimate_qem", "decimate_qem_ladder"]
