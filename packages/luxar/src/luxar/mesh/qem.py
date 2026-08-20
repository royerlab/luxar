"""Topology-preserving Garland-Heckbert mesh decimation."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .decimate import (
    DecimatedMesh,
    _average_per_cluster,
    _recompute_normals,
    _validate_decimate_inputs,
)


def _face_quadrics(positions: NDArray[np.float64]) -> NDArray[np.float64]:
    """One homogeneous squared-distance quadric per triangle."""
    n_faces, _, ndim = positions.shape
    quadrics = np.zeros((n_faces, ndim + 1, ndim + 1), dtype=np.float64)
    if ndim == 3:
        normal = np.cross(
            positions[:, 1] - positions[:, 0], positions[:, 2] - positions[:, 0]
        )
        length = np.linalg.norm(normal, axis=1)
        valid = length > 1e-15
        normal[valid] /= length[valid, None]
        plane = np.concatenate(
            [normal, -np.einsum("ij,ij->i", normal, positions[:, 0])[:, None]], axis=1
        )
        quadrics[valid] = np.einsum("fi,fj->fij", plane[valid], plane[valid])
        return quadrics

    identity = np.eye(ndim)
    for index, triangle in enumerate(positions):
        edges = (triangle[1:] - triangle[0]).T
        basis, singular, _ = np.linalg.svd(edges, full_matrices=False)
        rank = int(np.count_nonzero(singular > 1e-12))
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
    """Sum every incident face quadric onto each vertex."""
    face_quadrics = _face_quadrics(positions[faces])
    out = np.zeros((len(positions), positions.shape[1] + 1, positions.shape[1] + 1))
    for corner in range(3):
        np.add.at(out, faces[:, corner], face_quadrics)
    return out


def _require_nondegenerate_surface(
    positions: NDArray[np.float64], faces: NDArray[np.uint32]
) -> None:
    """Refuse input with no triangle spanning a two-dimensional surface."""
    triangles = positions[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    if np.any(np.linalg.matrix_rank(edges, tol=1e-12) >= 2):
        return
    raise ValueError(
        f"decimation collapsed every triangle of a {positions.shape[0]}-vertex, "
        f"{faces.shape[0]}-face mesh, leaving no surface. The input is "
        "degenerate (collinear or coincident vertices) rather than merely fine."
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
    if np.linalg.matrix_rank(matrix, tol=1e-12) == matrix.shape[0]:
        solved = np.linalg.solve(matrix, rhs)
        if np.isfinite(solved).all():
            candidates.append(solved)
    homogeneous = np.concatenate(
        [np.asarray(candidates), np.ones((len(candidates), 1), dtype=np.float64)],
        axis=1,
    )
    costs = np.einsum("ni,ij,nj->n", homogeneous, quadric, homogeneous)
    best = int(np.argmin(costs))
    return max(float(costs[best]), 0.0), np.asarray(candidates[best]).copy()


def _edge_cost(
    u: int,
    v: int,
    positions: NDArray[np.float64],
    quadrics: NDArray[np.float64],
) -> float:
    """Cheap heap-ordering cost evaluated at the edge midpoint."""
    midpoint = 0.5 * (positions[u] + positions[v])
    homogeneous = np.append(midpoint, 1.0)
    return max(float(homogeneous @ (quadrics[u] + quadrics[v]) @ homogeneous), 0.0)


def _edge_faces(
    u: int, v: int, vertex_faces: list[set[int]], active_faces: NDArray[np.bool_]
) -> set[int]:
    """Active triangles incident to both endpoints of an edge."""
    return {index for index in vertex_faces[u] & vertex_faces[v] if active_faces[index]}


def _boundary_vertex(
    vertex: int,
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
    active_faces: NDArray[np.bool_],
) -> bool:
    """Whether any active incident edge belongs to only one triangle."""
    return any(
        len(_edge_faces(vertex, other, vertex_faces, active_faces)) == 1
        for other in neighbors[vertex]
    )


def _link_condition(
    u: int,
    v: int,
    faces: NDArray[np.int64],
    active_faces: NDArray[np.bool_],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
) -> bool:
    """Veto collapses that change manifold topology or cross a boundary."""
    incident = _edge_faces(u, v, vertex_faces, active_faces)
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
    endpoint_faces = {
        index for index in vertex_faces[u] | vertex_faces[v] if active_faces[index]
    }
    if endpoint_faces == incident:
        return False
    if any(
        {index for index in vertex_faces[vertex] if active_faces[index]} <= incident
        for vertex in opposite
    ):
        return False

    boundary_u = _boundary_vertex(u, neighbors, vertex_faces, active_faces)
    boundary_v = _boundary_vertex(v, neighbors, vertex_faces, active_faces)
    if boundary_u or boundary_v:
        return len(incident) == 1 and boundary_u and boundary_v
    if len(neighbors[u] | neighbors[v] | {u, v}) == 4:
        return False
    return len(incident) == 2


def _rebuild_neighbors(
    vertices: set[int],
    faces: NDArray[np.int64],
    active_faces: NDArray[np.bool_],
    neighbors: list[set[int]],
    vertex_faces: list[set[int]],
) -> None:
    """Recompute local one-rings after replacing one endpoint in nearby faces."""
    for vertex in vertices:
        adjacent: set[int] = set()
        for face_index in vertex_faces[vertex]:
            if active_faces[face_index]:
                adjacent.update(int(value) for value in faces[face_index])
        adjacent.discard(vertex)
        neighbors[vertex] = adjacent


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


def _build_topology(
    faces: NDArray[np.int64], n_vertices: int
) -> tuple[list[set[int]], list[set[int]]]:
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
    return vertex_faces, neighbors


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
            _edge_cost(u, v, positions, quadrics),
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
) -> None:
    """Apply one accepted edge collapse and rebuild its local topology."""
    affected = neighbors[u] | neighbors[v] | {u, v}
    changed_faces = vertex_faces[u] | vertex_faces[v]
    for face_index in changed_faces:
        if not active_faces[face_index]:
            continue
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
    alive[v] = False
    parent[v] = u
    _rebuild_neighbors(affected, work_faces, active_faces, neighbors, vertex_faces)


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
) -> NDArray[np.int64]:
    """Collapse valid edges until the referenced surface reaches its target."""
    heap, serial = _build_heap(
        neighbors,
        alive=alive,
        barrier_dims=barrier_dims,
        vertices=vertices,
        positions=positions,
        quadrics=quadrics,
        versions=versions,
    )
    parent = np.arange(len(vertices), dtype=np.int64)
    remaining = int(np.unique(work_faces).size)
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
        if not _link_condition(u, v, work_faces, active_faces, neighbors, vertex_faces):
            continue
        _, target = _edge_target(u, v, positions, quadrics)
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
            )
    return parent


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
) -> DecimatedMesh:
    """Reduce a mesh by quadric edge collapse without violating the link condition.

    Args:
        vertices: ``(N, D)`` vertex coordinates.
        faces: ``(F, 3)`` triangle indices.
        target_vertices: Approximate maximum vertex count for the result.
        normals: Optional per-vertex normals.
        normal_dims: Three coordinate columns defining the normal frame.
        colors: Optional per-vertex colours.
        scalars: Optional per-vertex scalars or a uniform scalar value.
        spatial_dims: Coordinate columns QEM may coarsen across. At least three
            are required; use :func:`decimate` for automatic clustering fallback.

    Raises:
        ValueError: If the inputs are invalid or the surface collapses completely.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    input_faces = np.ascontiguousarray(faces, dtype=np.uint32)
    spatial_dims = _validate_decimate_inputs(
        vertices, input_faces, target_vertices, normals, normal_dims, spatial_dims
    )
    if len(spatial_dims) < 3:
        raise ValueError(
            "QEM mesh decimation requires at least 3 coarsening dimensions; "
            f"got {len(spatial_dims)}"
        )
    if len(vertices) <= target_vertices:
        return DecimatedMesh(vertices, input_faces, normals, colors, scalars)

    positions = vertices[:, spatial_dims].astype(np.float64)
    _require_nondegenerate_surface(positions, input_faces)
    barrier_dims = tuple(
        index for index in range(vertices.shape[1]) if index not in spatial_dims
    )
    work_faces = input_faces.astype(np.int64)
    active_faces = np.ones(len(work_faces), dtype=bool)
    alive = np.ones(len(vertices), dtype=bool)
    versions = np.zeros(len(vertices), dtype=np.int64)
    quadrics = _vertex_quadrics(positions, work_faces)
    vertex_faces, neighbors = _build_topology(work_faces, len(vertices))
    parent = _collapse_to_target(
        target_vertices,
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
    )
    return _compact_output(
        vertices,
        input_faces,
        positions=positions,
        spatial_dims=spatial_dims,
        work_faces=work_faces,
        active_faces=active_faces,
        parent=parent,
        normals=normals,
        normal_dims=normal_dims,
        colors=colors,
        scalars=scalars,
    )


__all__ = ["decimate_qem"]
