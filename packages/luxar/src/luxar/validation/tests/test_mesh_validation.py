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
import zarr

from luxar import Dimensions, LuxarZarrCompiler
from luxar.typing_utils.constants import (
    MAX_MESH_VERTICES,
    MESH_DECODE_BUDGET_BYTES,
    MESH_DECODED_BYTES_PER_VALUE,
)
from luxar.validation import ValidationError
from luxar.validation.base import (
    MAX_MESH_TEXTURE_SIZE,
    MESH_TEXTURE_DECODE_BUDGET_BYTES,
    validate_faces_for_writing,
    validate_mesh_decode_budget,
    validate_normal_dims_for_writing,
    validate_normals_for_writing,
    validate_texture_for_writing,
    validate_uvs_for_writing,
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

    MIRROR: ``MAX_MESH_VERTICES`` in
    ``packages/luxar-viewer/src/config/constants.ts`` must hold this value — the
    viewer's Stage-1 loader preflight is the twin of this write-time gate, and a
    viewer test pins that side. If the two drift, ``add_mesh`` can emit a store
    Luxar's own viewer then refuses.
    """
    assert MAX_MESH_VERTICES == 2**27
    assert MAX_MESH_VERTICES - 1 < 2**27


def test_decode_budget_matches_the_viewer_ceiling() -> None:
    """The write-time lower bound must track the viewer's admission ceiling.

    MIRROR: ``MESH_DECODE_BUDGET_BYTES`` in
    ``packages/luxar-viewer/src/config/constants.ts`` must hold this value. The
    Python validator charges fewer terms deliberately, but it must compare them
    against the same ceiling; a lower Python value would reject stores the viewer
    accepts, while a higher one would weaken the fail-fast guarantee.
    """
    assert MESH_DECODE_BUDGET_BYTES == 512 * 1024 * 1024
    assert MESH_DECODED_BYTES_PER_VALUE == 4


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


def test_mesh_appearance_attrs_are_written_and_validated(tmp_path) -> None:
    """All mesh lighting controls survive authoring, while invalid values fail early."""
    with LuxarZarrCompiler(tmp_path / "appearance.luxar.zarr") as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        mesh = scene.add_mesh(
            "m",
            _TETRA_V,
            _TETRA_F,
            ambient=0.3,
            shade_exponent=2.0,
            specular=0.12,
            shininess=24.0,
            alpha_cutoff=0.4,
        )
        assert mesh.attrs["ambient"] == 0.3
        assert mesh.attrs["shade_exponent"] == 2.0
        assert mesh.attrs["specular"] == 0.12
        assert mesh.attrs["shininess"] == 24.0
        assert mesh.attrs["alpha_cutoff"] == 0.4

        with pytest.raises(ValueError, match="Specular must be finite and between"):
            scene.add_mesh("bad_specular", _TETRA_V, _TETRA_F, specular=2.0)
        with pytest.raises(
            ValueError, match="Shininess must be finite and greater than 0"
        ):
            scene.add_mesh("bad_shininess", _TETRA_V, _TETRA_F, shininess=0.0)


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


# --- UVs and textures (#2175) ----------------------------------------------
#
# The two are a PAIR at the adder level (each is meaningless alone), but they
# validate independently, so they are tested independently here and the pairing
# refusal lives with the other composition gates in `test_mesh.py`.


@pytest.mark.parametrize(
    "uvs,error_pattern,test_id",
    [
        (
            np.zeros((4, 3), np.float32),
            "Expected shape",
            "three_components_is_not_a_uv",
        ),
        (
            np.zeros(8, np.float32),
            "Expected shape",
            "flat_array_rejected",
        ),
        (
            np.zeros((3, 2), np.float32),
            "3 rows but the mesh has 4",
            "count_mismatch",
        ),
        (
            np.array(
                [[0.0, 0.0], [np.nan, 0.0], [0.0, 0.0], [0.0, 0.0]],
                np.float32,
            ),
            "non-finite",
            "nan_samples_an_undefined_texel",
        ),
    ],
)
def test_uv_rejections(tmp_path, uvs, error_pattern, test_id) -> None:
    """Each malformed UV array is refused before anything reaches disk."""
    texture = np.zeros((2, 2, 3), dtype=np.uint8)
    store = tmp_path / f"{test_id}.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        with pytest.raises(ValueError, match=error_pattern):
            scene.add_mesh("m", _TETRA_V, _TETRA_F, uvs=uvs, texture=texture)
    assert "m" not in zarr.open_group(store, mode="r")


@pytest.mark.parametrize(
    "uvs,test_id",
    [
        (np.array([[0.0, 0.0], [1.0, 1.0]], np.float32), "unit_square"),
        # OUT of [0, 1] on purpose: this is the case a clamping validator would
        # break, and tiling a detail texture is the ordinary reason to author it.
        (np.array([[-2.0, 0.0], [7.5, 3.25]], np.float32), "outside_unit_square"),
        (np.array([[0.0, 0.0], [0.0, 0.0]], np.float32), "degenerate_all_zero"),
    ],
)
def test_uv_acceptances(uvs, test_id) -> None:
    """UVs outside [0, 1] are legal — they tile under texture_wrap='repeat'."""
    validate_uvs_for_writing(uvs, 2)


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_texture_for_writing(np.zeros((2, 2, 3), np.uint8), "tiff"),
            "Unknown texture_encoding",
            "unknown_encoding",
        ),
        (
            lambda: validate_texture_for_writing(np.zeros((2, 2), np.uint8), "raw"),
            "expects an .H, W, C. array",
            "greyscale_needs_an_explicit_channel_axis",
        ),
        (
            lambda: validate_texture_for_writing(np.zeros((2, 2, 2), np.uint8), "raw"),
            "Channels must be 1, 3 or 4",
            "two_channels",
        ),
        (
            # int32 is the dtype an unsuspecting `np.array([[...]])` produces on
            # Linux, and the same refusal element colours give.
            lambda: validate_texture_for_writing(np.zeros((2, 2, 3), np.int32), "raw"),
            "must be uint8 or uint16",
            "int32_refused_like_element_colours",
        ),
        (
            lambda: validate_texture_for_writing(
                np.full((2, 2, 3), np.nan, np.float32), "raw"
            ),
            "non-finite",
            "nan_texel",
        ),
        (
            # THE decompression-bomb refusal: encoded bytes with no declared size.
            lambda: validate_texture_for_writing(
                np.zeros(64, np.uint8), "png", None, None, None
            ),
            "requires explicit width, height and channels",
            "encoded_without_declared_dims",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros((2, 2, 3), np.uint8), "png", 2, 2, 3
            ),
            "expects a 1-D uint8 array",
            "raw_array_under_an_encoded_encoding",
        ),
        (
            lambda: validate_texture_for_writing(np.zeros((2, 2, 1), np.uint8), "ktx2"),
            "supports only RGB or RGBA",
            "ktx2_single_channel_refused",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros((2, 2, 3), np.float32), "ktx2"
            ),
            "only uint8 LDR input",
            "ktx2_hdr_refused",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros(16, np.uint8),
                "webp",
                2,
                2,
                3,
                ktx2_mode="bogus",
            ),
            "KTX2-only options",
            "ktx2_mode_rejected_for_webp",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros(16, np.uint8),
                "jpeg",
                2,
                2,
                3,
                ktx2_quality=2,
            ),
            "KTX2-only options",
            "ktx2_quality_rejected_for_jpeg",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros(16, np.uint8),
                "webp",
                2,
                2,
                3,
                ktx2_rdo_l=0.5,
            ),
            "texture_ktx2_rdo_l require texture_encoding='ktx2'",
            "ktx2_rdo_rejected_for_webp",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros((2, 2, 3), np.uint8),
                "ktx2",
                ktx2_mode="etc1s",
                ktx2_zcmp=9,
            ),
            "apply only to UASTC",
            "uastc_options_rejected_for_etc1s",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros(0, np.uint8), "jpeg", 2, 2, 3
            ),
            "payload is empty",
            "empty_encoded_payload",
        ),
        (
            # A declared size that disagrees is refused rather than silently
            # preferring one source — the viewer SPENDS the declared numbers.
            lambda: validate_texture_for_writing(
                np.zeros((8, 4, 3), np.uint8), "raw", 99, 8, 3
            ),
            "texture_width=99 disagrees",
            "declared_width_disagrees_with_payload",
        ),
        (
            # The shape the DECODE BUDGET cannot see. A 20000x2 texture is 120 KB
            # and passes every byte accounting on both sides, then exceeds
            # MAX_TEXTURE_SIZE on one axis and is silently clamped by the GPU at
            # upload — so the mesh renders the wrong image with no diagnostic.
            lambda: validate_texture_for_writing(
                np.zeros((2, 20_000, 3), np.uint8), "raw"
            ),
            "per-axis limit",
            "width_over_the_gpu_axis_limit",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros((20_000, 2, 3), np.uint8), "raw"
            ),
            "per-axis limit",
            "height_over_the_gpu_axis_limit",
        ),
        (
            # An ENCODED payload declares its dimensions rather than carrying
            # them, so the same ceiling has to hold on a declaration it cannot
            # cross-check against the bytes.
            lambda: validate_texture_for_writing(
                np.zeros(64, np.uint8), "png", 20_000, 2, 3
            ),
            "per-axis limit",
            "declared_encoded_width_over_the_axis_limit",
        ),
        (
            lambda: validate_texture_for_writing(
                np.zeros(64, np.uint8), "png", 16_000, 16_000, 3
            ),
            "over the 512 MiB per-node budget",
            "declared_encoded_texture_over_the_decode_budget",
        ),
        (
            lambda: validate_texture_for_writing(np.zeros((0, 4, 3), np.uint8), "raw"),
            "Dimensions must be positive",
            "zero_height",
        ),
    ],
)
def test_texture_rejections(factory, error_pattern, test_id) -> None:
    """Each malformed texture payload is refused before anything reaches disk."""
    with pytest.raises(ValidationError, match=error_pattern):
        factory()


@pytest.mark.parametrize(
    "texture", [np.zeros((4, 4), np.uint8), np.zeros(16, np.uint8)]
)
def test_ktx2_shape_error_names_pixel_input_contract(texture) -> None:
    """KTX2 authoring takes pixels, not the encoded-byte shape used by bitmap codecs."""
    with pytest.raises(ValidationError) as exc_info:
        validate_texture_for_writing(texture, "ktx2")

    message = str(exc_info.value)
    assert "Encoding 'ktx2'" in message
    assert "uint8 (H, W, 3|4) pixels" in message
    assert "pre-built KTX2 container" in message


@pytest.mark.parametrize(
    "texture,encoding,dims,expected,test_id",
    [
        (np.zeros((8, 4, 3), np.uint8), "raw", (None, None, None), (8, 4, 3), "rgb_u8"),
        (
            np.zeros((2, 2, 4), np.uint16),
            "raw",
            (None, None, None),
            (2, 2, 4),
            "rgba_u16",
        ),
        (np.zeros((2, 2, 1), np.uint8), "raw", (None, None, None), (2, 2, 1), "grey"),
        (
            np.zeros((2, 2, 4), np.uint8),
            "ktx2",
            (None, None, None),
            (2, 2, 4),
            "ktx2_rgba",
        ),
        # HDR: float of any width is writable, exactly as for element colours.
        (
            np.full((2, 2, 3), 9.0, np.float32),
            "raw",
            (None, None, None),
            (2, 2, 3),
            "hdr_f32",
        ),
        (
            np.full((2, 2, 3), 9.0, np.float16),
            "raw",
            (None, None, None),
            (2, 2, 3),
            "hdr_f16",
        ),
        (np.zeros(64, np.uint8), "png", (4, 8, 3), (8, 4, 3), "encoded_png"),
        (np.zeros(64, np.uint8), "webp", (4, 8, 4), (8, 4, 4), "encoded_webp"),
        (np.zeros(64, np.uint8), "jpeg", (4, 8, 3), (8, 4, 3), "encoded_jpeg"),
    ],
)
def test_texture_acceptances(texture, encoding, dims, expected, test_id) -> None:
    """The encoding x dtype matrix, and the resolved (h, w, c) it returns.

    The return value matters as much as the acceptance: it is what the writer
    stamps, and what the viewer's admission gate later spends.
    """
    w, h, c = dims
    color_space = "linear" if test_id.startswith("hdr_") else "srgb"
    assert (
        validate_texture_for_writing(texture, encoding, w, h, c, color_space)
        == expected
    )


# --- decode budget (#2145) --------------------------------------------------
#
# The write-time twin of the viewer's per-node admission ceiling. It charges only
# the DECODED term, which makes it a strict lower bound on the loader's accounting
# — see `validate_mesh_decode_budget`'s docstring for why under-counting is the
# only safe direction for a hard error. These tests pin both ends of that: it
# fires when it provably must, and it does NOT fire one value below.

_BUDGET_VALUES = MESH_DECODE_BUDGET_BYTES // MESH_DECODED_BYTES_PER_VALUE


def _largest_fitting_face_count(n_vertices: int, n_dims: int) -> int:
    """The most triangles that still fit, given the vertex block's cost."""
    return (_BUDGET_VALUES - n_vertices * n_dims) // 3


def test_decode_budget_accepts_the_largest_mesh_that_fits() -> None:
    """The boundary case must PASS, or the gate is a false-rejection machine.

    This is the more important half of the pair. A hard error that over-counts
    refuses stores the viewer would happily load — a worse bug than the one the
    gate fixes — so the acceptance test ships alongside the rejection test rather
    than after it.
    """
    n_vertices, n_dims = 1_000_000, 3
    validate_mesh_decode_budget(
        n_vertices, n_dims, _largest_fitting_face_count(n_vertices, n_dims)
    )


def test_texture_budget_uses_the_shared_mesh_ceiling() -> None:
    assert MESH_TEXTURE_DECODE_BUDGET_BYTES == MESH_DECODE_BUDGET_BYTES


def test_texture_budget_charges_ktx2_as_a_compressed_mip_chain() -> None:
    largest = np.broadcast_to(
        np.zeros((1, 1, 3), dtype=np.uint8),
        (MAX_MESH_TEXTURE_SIZE, MAX_MESH_TEXTURE_SIZE, 3),
    )

    assert validate_texture_for_writing(largest, "ktx2") == (
        MAX_MESH_TEXTURE_SIZE,
        MAX_MESH_TEXTURE_SIZE,
        3,
    )
    with pytest.raises(ValidationError, match="decodes to 1024 MiB"):
        validate_texture_for_writing(
            np.zeros(1, dtype=np.uint8),
            "webp",
            MAX_MESH_TEXTURE_SIZE,
            MAX_MESH_TEXTURE_SIZE,
            3,
        )


def test_decode_budget_charges_uvs_and_texture_together() -> None:
    geometry_bytes = (4 * 3 + 4 * 3 + 4 * 2) * MESH_DECODED_BYTES_PER_VALUE
    texture_bytes = MESH_DECODE_BUDGET_BYTES - geometry_bytes + 1
    with pytest.raises(ValidationError, match="over the viewer"):
        validate_mesh_decode_budget(
            4,
            3,
            4,
            uvs=np.zeros((4, 2), dtype=np.float32),
            texture_decoded_bytes=texture_bytes,
        )


@pytest.mark.parametrize(
    "factory,error_pattern,test_id",
    [
        (
            lambda: validate_mesh_decode_budget(
                1_000_000, 3, _largest_fitting_face_count(1_000_000, 3) + 1
            ),
            "over the viewer",
            "one_triangle_past_the_budget",
        ),
        (
            # Normals are a third of a vertex block on their own, so a mesh that
            # fits WITHOUT them can fail WITH them. A gate that ignored optional
            # channels would pass this.
            lambda: validate_mesh_decode_budget(
                40_000_000,
                3,
                4_000_000,
                normals=np.zeros((40_000_000, 3), dtype=np.float32),
            ),
            "over the viewer",
            "normals_push_it_over",
        ),
        (
            # A BROADCAST colour stores one row and decodes to n_vertices rows.
            # Charging the stored row would under-count by ~4 bytes per vertex,
            # which is exactly the trap the loader's decoded term exists to close.
            lambda: validate_mesh_decode_budget(
                43_000_000, 3, 1_000_000, colors=(1.0, 0.0, 0.0)
            ),
            "over the viewer",
            "broadcast_colour_charged_at_its_expansion",
        ),
    ],
)
def test_decode_budget_rejections(factory, error_pattern, test_id) -> None:
    """Each over-budget mesh is refused before anything reaches disk."""
    with pytest.raises(ValidationError, match=error_pattern):
        factory()


def test_decode_budget_message_names_the_remedy_and_the_undercount() -> None:
    """The message has to say what to do, and admit it is a lower bound.

    A user who splits to just under the reported figure and is refused again
    would reasonably call the first message a lie, so it says outright that the
    real footprint is larger.
    """
    with pytest.raises(ValidationError) as excinfo:
        validate_mesh_decode_budget(
            1_000_000, 3, _largest_fitting_face_count(1_000_000, 3) + 1
        )
    text = str(excinfo.value)
    assert "counts only decoded bytes" in text
    assert "per node" in text
