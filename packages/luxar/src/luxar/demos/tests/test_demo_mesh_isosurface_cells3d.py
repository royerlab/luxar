"""Guards for the mesh isosurface demo's extraction step.

Runs against a SYNTHETIC volume rather than ``cells3d``, so the test needs no
download and — more usefully — has a known right answer: a Gaussian ball has an
analytic isosurface, so the extracted mesh can be checked against geometry
instead of against a recorded snapshot of itself.

What is worth pinning here is not "marching cubes works" (that is scikit-image's
job) but the three things the DEMO is responsible for, each of which would render
a plausible-looking but wrong surface:

* vertices come out in **physical units**, via marching_cubes' ``spacing`` — a
  demo that forgot it would emit a mesh in voxel indices, silently anisotropic by
  the 4.5x Z/XY ratio of this dataset;
* the mesh is **welded**, so it is a surface rather than a triangle soup;
* the normals are **unit-length and outward**, which is what the shading model
  reads and what ``normal_dims`` promises.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage.measure", reason="mesh isosurface demo needs scikit-image")
pytest.importorskip("scipy.ndimage", reason="mesh isosurface demo needs scipy")

from luxar.demos.demo_mesh_isosurface_cells3d import (  # noqa: E402
    ISOLEVEL_FRACTION,
    SMOOTH_SIGMA,
    VOXEL_SIZE_ZYX,
    extract_isosurface,
)


def _gaussian_ball(shape=(40, 48, 48), sigma=8.0) -> np.ndarray:
    """A centred Gaussian blob — bright core, smooth falloff, no noise."""
    zz, yy, xx = np.meshgrid(
        *(np.arange(n) - (n - 1) / 2.0 for n in shape),
        indexing="ij",
    )
    r2 = zz**2 + yy**2 + xx**2
    return (10000.0 * np.exp(-r2 / (2.0 * sigma**2))).astype(np.float32)


@pytest.fixture(scope="module")
def ball_surface():
    return extract_isosurface(_gaussian_ball(), "ball")


def test_extracts_a_non_trivial_closed_surface(ball_surface) -> None:
    vertices, faces, normals = ball_surface
    assert len(vertices) > 500, "isosurface is implausibly small for a 40x48x48 ball"
    assert len(faces) > 1000
    # Shapes and dtypes the writer requires (it would raise otherwise, but a
    # failure here names the cause).
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    assert normals.shape == vertices.shape
    assert vertices.dtype == np.float32
    assert faces.dtype == np.uint32
    # Every index in range — the writer checks this too, and a violation here
    # would mean the demo reordered one array and not the other.
    assert faces.max() < len(vertices)


def test_vertices_are_in_physical_units_not_voxel_indices(ball_surface) -> None:
    """The `spacing` argument is load-bearing, and its absence is invisible.

    Dropping it yields a mesh in voxel indices: geometrically similar, and wrong
    by the dataset's 0.29 / 0.26 anisotropy — a 4.5x Z squash relative to the
    physical shape, which reads as a plausible flattened cell rather than as a bug.

    Asserted against an independent no-spacing extraction of the SAME volume, so
    the expected numbers come from geometry rather than from a recorded snapshot.
    An earlier version of this test compared only the Z/Y extent RATIO with a 15%
    tolerance, and was vacuous: the wrong answer (1.0) and the right one (1.115)
    differ by 10.3%, which fits inside that tolerance. Verified by mutation —
    removing `spacing=` from the demo now turns this red.
    """
    measure = pytest.importorskip("skimage.measure")
    ndimage = pytest.importorskip("scipy.ndimage")

    vertices, _faces, _normals = ball_surface
    extent = vertices.max(axis=0) - vertices.min(axis=0)

    # Re-extract at the same isolevel with NO spacing, i.e. in voxel indices.
    volume = ndimage.gaussian_filter(_gaussian_ball().astype(np.float32), SMOOTH_SIGMA)
    lo, hi = np.percentile(volume, [1.0, 99.5])
    level = float(lo + ISOLEVEL_FRACTION * (hi - lo))
    raw_vertices, _f, _n, _v = measure.marching_cubes(volume, level=level)
    raw_extent = raw_vertices.max(axis=0) - raw_vertices.min(axis=0)

    # Each axis must be scaled by exactly its voxel size. `rel=1e-4` is safe
    # because this is the same extraction on the same data — only the scale differs.
    for axis, spacing in enumerate(VOXEL_SIZE_ZYX):
        assert extent[axis] == pytest.approx(raw_extent[axis] * spacing, rel=1e-4), (
            f"axis {axis} extent {extent[axis]:.3f} != {raw_extent[axis]:.3f} x {spacing} "
            "— vertices are in voxel indices, not micrometres"
        )


def test_the_mesh_is_welded(ball_surface) -> None:
    """Shared vertices, not a triangle soup.

    Load-bearing beyond memory: picking reports a VERTEX ordinal (spec §6.5), so
    on a de-indexed mesh every triangle corner would be its own pick target and
    the shared-vertex semantics would be a fiction. `3 * n_faces` vertices is
    exactly the de-indexed count, so staying well under it is the check.
    """
    vertices, faces, _normals = ball_surface
    assert len(vertices) < len(faces) * 3 * 0.5, (
        "vertices look de-indexed (soup, not surface)"
    )


def test_normals_are_unit_length_and_point_outward(ball_surface) -> None:
    """The shading model renormalizes, but the ORIENTATION it cannot fix.

    marching_cubes' default ``gradient_direction="descent"`` points normals toward
    decreasing intensity — outward from a bright object. If that ever flipped,
    every surface would shade as though lit from behind: the §6.2 headlight's wrap
    term would land in [0, 0.5) and the whole mesh would collapse toward its
    ambient floor. For a centred ball, "outward" is checkable exactly: the normal
    must agree with the vertex's own direction from the centre.
    """
    vertices, _faces, normals = ball_surface

    lengths = np.linalg.norm(normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-3), "normals are not unit-length"

    centre = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    radial = vertices - centre
    radial /= np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    alignment = np.einsum("ij,ij->i", radial, normals)
    # Mean alignment, not per-vertex: marching-cubes normals on a discretized
    # sphere wobble by a few degrees, and a handful of near-centre vertices have
    # an ill-defined radial direction.
    assert alignment.mean() > 0.9, (
        f"normals point inward on average (alignment {alignment.mean():.2f}) — "
        "the surface would shade as if lit from behind"
    )
