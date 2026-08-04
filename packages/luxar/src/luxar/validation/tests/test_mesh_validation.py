"""Tests for mesh data validation.

Mirrors ``test_lines_validation.py`` for Mesh geometry, per the geometry-symmetry
rule. Where Lines' mandatory per-vertex attribute is ``widths``, mesh's mandatory
companion array is ``faces`` — topology rather than a size scalar — so the
rejection cases here are about index validity rather than positivity.

Every rejection is parametrized as a ``(factory, error_pattern, test_id)`` triple
so each gets its own named case, and each was verified to fail before the
corresponding validator existed (a test that passes against a no-op validator is
vacuous).
"""

import warnings

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar.typing_utils.constants import MAX_MESH_VERTICES
from luxar.validation import ValidationError
from luxar.validation.base import (
    validate_faces_for_writing,
    validate_normal_dims_for_writing,
    validate_normals_for_writing,
    validate_vertices_for_writing,
)

# A welded tetrahedron: the smallest closed surface, 4 vertices / 4 faces.
_TETRA_V = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)
_TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
_TETRA_N = np.array(
    [[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    dtype=np.float32,
)


# =============================================================================
# faces validation
# =============================================================================


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_faces_for_writing(
                np.array([[0.0, 1.0, 2.0]], dtype=np.float64), 4
            ),
            "must be an integer array",
            "float_dtype_truncates_on_cast",
        ),
        (
            lambda: validate_faces_for_writing(np.array([[-1, 1, 2]]), 4),
            r"Face index -1 < 0",
            "negative_index_wraps_on_cast",
        ),
        (
            lambda: validate_faces_for_writing(np.array([[0, 1, 9]]), 4),
            "out of range for 4 vertices",
            "index_past_last_vertex",
        ),
        (
            lambda: validate_faces_for_writing(np.array([[0, 1, 4]]), 4),
            "out of range for 4 vertices",
            "index_equal_to_vertex_count_offbyone",
        ),
        (
            lambda: validate_faces_for_writing(np.zeros((0, 3), dtype=np.uint32), 4),
            "no faces",
            "empty_faces_draws_nothing",
        ),
        (
            lambda: validate_faces_for_writing(np.zeros((2, 4), dtype=np.uint32), 4),
            r"\(F, 3\) triangle array",
            "quad_width_rejected",
        ),
        (
            lambda: validate_faces_for_writing(np.zeros((2, 2), dtype=np.uint32), 4),
            r"\(F, 3\) triangle array",
            "edge_width_rejected",
        ),
        (
            lambda: validate_faces_for_writing(np.zeros((2, 2, 3), dtype=np.uint32), 4),
            r"\(F, 3\) triangle array",
            "three_dimensional_rejected",
        ),
        (
            lambda: validate_faces_for_writing(
                np.array([0, 1, 2, 0], dtype=np.uint32), 4
            ),
            "divisible by 3",
            "flat_not_multiple_of_three",
        ),
    ],
)
def test_faces_rejections(factory, error_pattern, test_id) -> None:
    """Each malformed ``faces`` input is rejected with a specific message."""
    with pytest.raises(ValidationError, match=error_pattern):
        factory()


@pytest.mark.parametrize(
    "faces,test_id",
    [
        (np.array([[0, 1, 2], [1, 2, 3]], dtype=np.uint32), "pairs_2d"),
        (np.array([0, 1, 2, 1, 2, 3], dtype=np.uint32), "flat_3f"),
        (np.array([[0, 1, 2]], dtype=np.int64), "signed_dtype_in_range"),
        (np.array([[3, 3, 3]], dtype=np.uint32), "degenerate_but_in_range"),
    ],
)
def test_faces_accepted_forms(faces, test_id) -> None:
    """Both documented layouts and any in-range integer dtype are accepted.

    A degenerate triangle (repeated index) is deliberately allowed: it is
    zero-area, not out of range, and the renderer handles it — rejecting it would
    refuse legitimate data produced by decimation.
    """
    validate_faces_for_writing(faces, 4)


# =============================================================================
# vertex-count ceiling
# =============================================================================


def test_vertices_at_cap_accepted() -> None:
    """Exactly ``MAX_MESH_VERTICES`` is admissible — the bound is inclusive.

    Uses a stride trick rather than allocating 2^27 rows: the validator reads only
    ``shape[0]``, so a zero-stride broadcast view has the right shape without the
    ~1.6 GB the real array would cost.
    """
    at_cap = np.broadcast_to(np.zeros((1, 3), dtype=np.float32), (MAX_MESH_VERTICES, 3))
    validate_vertices_for_writing(at_cap)


def test_vertices_over_cap_rejected() -> None:
    """One past the cap is refused, naming the pick-aliasing reason."""
    over = np.broadcast_to(
        np.zeros((1, 3), dtype=np.float32), (MAX_MESH_VERTICES + 1, 3)
    )
    with pytest.raises(ValidationError, match="exceeds the maximum"):
        validate_vertices_for_writing(over)


def test_vertex_cap_is_the_alias_free_bound() -> None:
    """The cap must equal the pick vote-key stride, not merely be near it.

    Pins the arithmetic the bound rests on: the largest vertex ordinal is
    ``n_vertices - 1``, so admitting exactly ``2^27`` vertices keeps every ordinal
    strictly below the ``2^27`` stride. An off-by-one here reintroduces cross-node
    pick aliasing for exactly one vertex, which no rendering test would catch.
    """
    assert MAX_MESH_VERTICES == 2**27
    assert MAX_MESH_VERTICES - 1 < 2**27


# =============================================================================
# normals + normal_dims
# =============================================================================


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_normals_for_writing(np.zeros((4, 2), dtype=np.float32), 4),
            r"Expected shape \(n_vertices, 3\)",
            "two_component_normals",
        ),
        (
            lambda: validate_normals_for_writing(np.zeros((4, 4), dtype=np.float32), 4),
            r"Expected shape \(n_vertices, 3\)",
            "four_component_normals",
        ),
        (
            lambda: validate_normals_for_writing(np.zeros(4, dtype=np.float32), 4),
            r"Expected shape \(n_vertices, 3\)",
            "one_dimensional_normals",
        ),
        (
            lambda: validate_normals_for_writing(np.zeros((3, 3), dtype=np.float32), 4),
            "doesn't match vertex count",
            "normals_count_mismatch",
        ),
        (
            lambda: validate_normals_for_writing(
                np.full((4, 3), np.nan, dtype=np.float32), 4
            ),
            "NaN or Inf",
            "nan_normals",
        ),
        (
            lambda: validate_normals_for_writing(
                np.full((4, 3), np.inf, dtype=np.float32), 4
            ),
            "NaN or Inf",
            "inf_normals",
        ),
        (
            lambda: validate_normals_for_writing([[0.0, 0.0, 1.0]] * 4, 4),
            "Expected numpy array",
            "list_of_lists_rejected",
        ),
    ],
)
def test_normals_rejections(factory, error_pattern, test_id) -> None:
    with pytest.raises(ValidationError, match=error_pattern):
        factory()


def test_zero_length_normals_warn_not_reject() -> None:
    """A zero normal is warned about, never rejected.

    Degenerate triangles legitimately produce them and the renderer's epsilon
    guard handles them, so rejecting would refuse valid data. The warning exists
    because the fallback is pointwise: on a shared-vertex mesh the interpolated
    normal near a zero blends toward its neighbours, so shading is locally
    distorted rather than cleanly flat.
    """
    normals = _TETRA_N.copy()
    normals[1] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_normals_for_writing(normals, 4)
    assert len(caught) == 1
    assert "zero-length" in str(caught[0].message)


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_normal_dims_for_writing((0, 1), 4),
            "exactly 3 dimension indices",
            "two_entries",
        ),
        (
            lambda: validate_normal_dims_for_writing((0, 1, 2, 3), 4),
            "exactly 3 dimension indices",
            "four_entries",
        ),
        (
            lambda: validate_normal_dims_for_writing((0, 1, 1), 4),
            "must be distinct",
            "duplicate_entries",
        ),
        (
            lambda: validate_normal_dims_for_writing((0, 1, 9), 4),
            "out of range",
            "index_past_ndim",
        ),
        (
            lambda: validate_normal_dims_for_writing((0, 1, 4), 4),
            "out of range",
            "index_equal_to_ndim_offbyone",
        ),
        (
            lambda: validate_normal_dims_for_writing((-1, 1, 2), 4),
            "out of range",
            "negative_index",
        ),
        (
            lambda: validate_normal_dims_for_writing((0.0, 1, 2), 4),
            "must be an integer dimension index",
            "float_entry",
        ),
        (
            lambda: validate_normal_dims_for_writing("012", 4),
            "Expected a sequence",
            "string_rejected",
        ),
        (
            lambda: validate_normal_dims_for_writing(3, 4),
            "Expected a sequence",
            "bare_int_rejected",
        ),
    ],
)
def test_normal_dims_rejections(factory, error_pattern, test_id) -> None:
    with pytest.raises(ValidationError, match=error_pattern):
        factory()


@pytest.mark.parametrize(
    "dims,test_id",
    [
        ((True, 0, 2), "bool_first"),
        ((0, False, 2), "bool_middle"),
        (np.array([True, False, True]), "bool_ndarray"),
    ],
)
def test_normal_dims_rejects_bool_entries(dims, test_id) -> None:
    """A bool must not pass as a dimension index.

    `bool` is an `int` subclass, so only an explicit rejection catches it — and
    the check has to run on the entries AS GIVEN. Validating through
    ``np.asarray`` destroys the evidence: ``np.asarray((True, 0, 2))`` is an int64
    array, so `True` arrives already indistinguishable from dimension 1. This test
    exists because that is exactly the hole the first implementation had — it
    accepted ``(True, 0, 2)`` as dimensions ``(1, 0, 2)``.
    """
    with pytest.raises(ValidationError, match="must be an integer dimension index"):
        validate_normal_dims_for_writing(dims, 4)


@pytest.mark.parametrize(
    "dims,test_id",
    [
        ((0, 1, 2), "tuple_leading"),
        ([1, 2, 3], "list_trailing"),
        (np.array([2, 0, 1]), "ndarray_unordered"),
        ((np.int64(0), np.int64(2), np.int64(3)), "numpy_integers"),
    ],
)
def test_normal_dims_accepted_forms(dims, test_id) -> None:
    """Distinct in-range integer triples are accepted in any container or order.

    Order is deliberately not constrained: the ordering of ``normal_dims`` assigns
    which normal component belongs to which dimension, so a permutation is
    meaningful data rather than an error.
    """
    validate_normal_dims_for_writing(dims, 4)


# =============================================================================
# writer-level pairing (needs writer context, so tested through the writer)
# =============================================================================


def test_normals_without_normal_dims_rejected(tmp_path) -> None:
    """Normals with no ``normal_dims`` cannot be oriented, so the write fails."""
    with LuxarZarrCompiler(tmp_path / "n.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="normal_dims is required"):
            compiler.write_mesh("m", _TETRA_V, _TETRA_F, normals=_TETRA_N)


def test_normal_dims_without_normals_rejected(tmp_path) -> None:
    """``normal_dims`` alone describes nothing, so it is refused rather than kept."""
    with LuxarZarrCompiler(tmp_path / "d.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="without normals"):
            compiler.write_mesh("m", _TETRA_V, _TETRA_F, normal_dims=(0, 1, 2))


def test_bad_shading_value_rejected(tmp_path) -> None:
    """An unrecognised ``shading`` must not reach zarr.

    The viewer branches on this string; a typo would silently select the
    stored-normal path rather than erroring.
    """
    with LuxarZarrCompiler(tmp_path / "s.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="shading must be"):
            compiler.write_mesh("m", _TETRA_V, _TETRA_F, shading="smoooth")


def test_reserved_attrs_rejected(tmp_path) -> None:
    """A caller cannot supply the writer's own stamps.

    ``normal_dims`` in particular is reserved, which is why ``add_mesh`` surfaces
    it as an explicit keyword rather than letting it ride in through ``**attrs``.
    """
    with LuxarZarrCompiler(tmp_path / "r.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match="reserved"):
            compiler.write_mesh("m", _TETRA_V, _TETRA_F, **{"n_faces": 99})


def test_bad_vertices_shape(tmp_path) -> None:
    """Vertices must be 2D ``(V, D)`` — mirrors the lines/points equivalents."""
    with LuxarZarrCompiler(tmp_path / "v.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises((ValueError, ValidationError)):
            compiler.write_mesh("m", np.ones((3,), dtype=np.float32), _TETRA_F)


def test_mismatched_colors(tmp_path) -> None:
    """Colors length must match the vertex count."""
    with LuxarZarrCompiler(tmp_path / "c.luxar.zarr") as compiler:
        compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError):
            compiler.write_mesh(
                "m", _TETRA_V, _TETRA_F, colors=np.ones((2, 3), dtype=np.uint8)
            )


def test_broadcast_color_and_scalar_accepted(tmp_path) -> None:
    """A uniform RGB(A) tuple and a single scalar broadcast, as for the siblings."""
    with LuxarZarrCompiler(tmp_path / "b.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh("m", _TETRA_V, _TETRA_F, colors=(0.2, 0.4, 0.6, 0.8))
        assert mesh.has_colors
        other = scene.add_mesh(
            "m2", _TETRA_V, _TETRA_F, scalars=0.5, colormap="viridis"
        )
        assert other.has_scalars
