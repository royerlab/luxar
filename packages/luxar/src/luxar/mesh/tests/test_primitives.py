"""Tests for :mod:`luxar.mesh.primitives`.

The properties pinned here are the ones a *material* demo leans on and a
plausible-looking sphere could still get wrong: that the mesh is welded (shared
vertices, not a soup), closed (every edge has exactly two faces), that its
normals are the outward unit positions, and that the winding agrees with them.
"""

from __future__ import annotations

import numpy as np
import pytest

from ..primitives import icosphere


@pytest.mark.parametrize("subdivisions", [0, 1, 3])
def test_counts_follow_the_closed_form(subdivisions: int) -> None:
    vertices, faces, normals = icosphere(subdivisions)
    assert vertices.shape == (10 * 4**subdivisions + 2, 3)
    assert faces.shape == (20 * 4**subdivisions, 3)
    assert normals.shape == vertices.shape
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32
    assert normals.dtype == np.float32


def test_is_welded_and_closed() -> None:
    vertices, faces, _ = icosphere(2)
    # Welded: every vertex row is referenced by at least one face, and the face
    # count is far below three-per-face (a soup would have exactly 3F rows).
    assert set(np.unique(faces)) == set(range(len(vertices)))
    # Closed: each undirected edge is shared by exactly two faces.
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)


def test_normals_are_outward_unit_positions_and_winding_agrees() -> None:
    radius = 2.5
    vertices, faces, normals = icosphere(2, radius=radius)
    assert np.allclose(np.linalg.norm(vertices, axis=1), radius, atol=1e-5)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    assert np.allclose(vertices / radius, normals, atol=1e-6)
    # Counter-clockwise from outside: the geometric face normal points along the
    # mean vertex normal for every face.
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    face_normal = np.cross(b - a, c - a)
    outward = normals[faces].mean(axis=1)
    assert np.all(np.einsum("ij,ij->i", face_normal, outward) > 0)


def test_refuses_degenerate_arguments() -> None:
    with pytest.raises(ValueError, match="subdivisions"):
        icosphere(-1)
    with pytest.raises(ValueError, match="radius"):
        icosphere(1, radius=0.0)
