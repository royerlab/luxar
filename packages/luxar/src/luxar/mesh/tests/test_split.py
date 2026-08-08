"""Tests for :mod:`luxar.mesh.split`.

The load-bearing test here is :func:`test_split_preserves_geometry_exactly` —
everything else is a property of the bookkeeping, but that one pins the thing a
partition must never get wrong: the re-indexed parts describe the *same
triangles in the same places* as the input. A split that renumbered
inconsistently would still produce plausible-looking arrays of the right shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..split import (
    MeshPart,
    duplication_factor,
    face_centroids,
    split_mesh_by_faces,
)


def _grid_mesh(n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """An ``n x n`` vertex grid triangulated into ``2*(n-1)^2`` faces.

    Fully connected, so every interior BSP cut is forced to duplicate vertices —
    which is exactly the case the split exists to handle.
    """
    xs, ys = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n), indexing="ij")
    vertices = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n, np.float32)], axis=1)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = (
                i * n + j,
                i * n + j + 1,
                (i + 1) * n + j,
                (i + 1) * n + j + 1,
            )
            faces.append([a, b, c])
            faces.append([b, d, c])
    return vertices, np.asarray(faces, dtype=np.uint32)


def _halves(faces: np.ndarray, vertices: np.ndarray) -> list[np.ndarray]:
    """Split faces into two parts by centroid x, a cut that crosses the surface."""
    cx = face_centroids(vertices, faces)[:, 0]
    mid = float(np.median(cx))
    return [np.nonzero(cx < mid)[0], np.nonzero(cx >= mid)[0]]


def test_split_preserves_geometry_exactly() -> None:
    """Re-indexed parts describe the identical triangles, in identical positions.

    This is the invariant. For every part and every face, the three CORNER
    POSITIONS reached through the part's own gathered table must equal the corner
    positions the original face had. Renumbering is only allowed to change the
    integers, never the geometry they address.
    """
    vertices, faces = _grid_mesh()
    parts = split_mesh_by_faces(faces, _halves(faces, vertices))

    for part in parts:
        part_vertices = vertices[part.vertex_index]
        rebuilt = part_vertices[part.faces]  # (Fi, 3, D) corner positions
        original = vertices[faces[part.face_index]]  # same faces, original table
        np.testing.assert_array_equal(rebuilt, original)


def test_split_is_a_partition_of_the_faces() -> None:
    """Every input triangle lands in exactly one part — no holes, no double-draw."""
    vertices, faces = _grid_mesh()
    parts = split_mesh_by_faces(faces, _halves(faces, vertices))

    assert sum(int(p.faces.shape[0]) for p in parts) == faces.shape[0]
    seen = np.concatenate([p.face_index for p in parts])
    np.testing.assert_array_equal(np.sort(seen), np.arange(faces.shape[0]))


def test_boundary_vertices_are_duplicated_not_dropped() -> None:
    """A cut through a connected surface duplicates its boundary vertices.

    The mechanism the whole module exists for, asserted directly rather than
    inferred from a count: some vertex must appear in BOTH parts' gather maps.
    """
    vertices, faces = _grid_mesh()
    parts = split_mesh_by_faces(faces, _halves(faces, vertices))

    left, right = (set(p.vertex_index.tolist()) for p in parts)
    shared = left & right
    assert shared, "a cut across a connected grid must share boundary vertices"
    # Total gathered exceeds the original exactly by the shared count.
    total = sum(int(p.vertex_index.size) for p in parts)
    assert total == len(left | right) + len(shared)
    assert duplication_factor(parts, vertices.shape[0]) > 1.0


def test_indices_are_local_and_in_range() -> None:
    """Each part's faces index its own table, not the original one."""
    vertices, faces = _grid_mesh()
    parts = split_mesh_by_faces(faces, _halves(faces, vertices))

    for part in parts:
        assert part.faces.dtype == np.uint32
        assert part.faces.min() >= 0
        assert part.faces.max() < part.vertex_index.size
        # Gather maps are ascending and duplicate-free WITHIN a part.
        assert np.all(np.diff(part.vertex_index) > 0)


def test_no_vertex_is_gathered_that_no_face_uses() -> None:
    """Parts carry only the vertices they reference — no dead weight."""
    vertices, faces = _grid_mesh()
    parts = split_mesh_by_faces(faces, _halves(faces, vertices))

    for part in parts:
        used = np.unique(part.faces)
        np.testing.assert_array_equal(used, np.arange(part.vertex_index.size))


def test_disjoint_components_split_without_duplication() -> None:
    """A cut that falls BETWEEN components duplicates nothing (factor 1.0)."""
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint32)
    parts = split_mesh_by_faces(faces, [np.array([0]), np.array([1])])

    assert duplication_factor(parts, vertices.shape[0]) == 1.0
    for part in parts:
        assert part.vertex_index.size == 3


def test_single_part_is_a_pure_renumbering() -> None:
    """One part covering everything still returns a valid, complete mesh."""
    vertices, faces = _grid_mesh(4)
    (part,) = split_mesh_by_faces(faces, [np.arange(faces.shape[0])])

    np.testing.assert_array_equal(
        vertices[part.vertex_index][part.faces], vertices[faces]
    )
    assert part.vertex_index.size == vertices.shape[0]


def test_face_centroids_are_the_corner_mean() -> None:
    vertices = np.array([[0, 0, 0], [3, 0, 0], [0, 3, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    np.testing.assert_allclose(face_centroids(vertices, faces), [[1.0, 1.0, 0.0]])


def test_centroids_work_in_more_than_three_dimensions() -> None:
    """nD vertices are supported — the mean is over corners, not over 3 axes."""
    vertices = np.array(
        [[0, 0, 0, 6], [3, 0, 0, 6], [0, 3, 0, 6]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.uint32)
    np.testing.assert_allclose(face_centroids(vertices, faces), [[1.0, 1.0, 0.0, 6.0]])


@pytest.mark.parametrize(
    "bad_parts, reason",
    [
        ([np.array([0])], "dropped"),
        ([np.array([0, 1]), np.array([1])], "duplicated"),
        ([np.array([0]), np.array([1]), np.array([1])], "duplicated"),
    ],
)
def test_non_partition_is_rejected(bad_parts: list, reason: str) -> None:
    """A dropped or double-counted face raises instead of rendering wrong.

    Neither failure is loud downstream — a dropped face is a hole, a duplicated
    one is an invisible double-draw — so the check has to be here.
    """
    _, faces = _grid_mesh(2)  # 2 faces
    with pytest.raises(ValueError, match="must be a partition"):
        split_mesh_by_faces(faces, bad_parts)


@pytest.mark.parametrize(
    "bad_parts", [[np.array([-1]), np.array([1])], [np.array([0, 2])]]
)
def test_out_of_range_face_indices_are_rejected(bad_parts: list) -> None:
    """An index outside ``range(F)`` is rejected, not silently wrapped.

    Counting distinct assignments is not enough on its own: numpy accepts a
    NEGATIVE index and wraps it, so ``[[-1], [1]]`` over two faces has the right
    count and the right uniqueness while dropping face 0 and drawing face 1
    twice — the exact corruption the partition check exists to catch.
    """
    _, faces = _grid_mesh(2)  # 2 faces
    with pytest.raises(ValueError, match="must be a partition"):
        split_mesh_by_faces(faces, bad_parts)


def test_bad_face_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"shape \(F, 3\)"):
        split_mesh_by_faces(np.zeros((4, 4), dtype=np.uint32), [np.arange(4)])


def test_duplication_factor_handles_empty_input() -> None:
    assert duplication_factor([], 0) == 1.0


def test_mesh_part_is_frozen() -> None:
    """Parts are immutable — a caller cannot renumber one half of the pair."""
    part = MeshPart(
        faces=np.zeros((1, 3), np.uint32),
        vertex_index=np.zeros(3, np.intp),
        face_index=np.zeros(1, np.intp),
    )
    with pytest.raises(AttributeError):
        part.faces = np.zeros((2, 3), np.uint32)  # type: ignore[misc]
