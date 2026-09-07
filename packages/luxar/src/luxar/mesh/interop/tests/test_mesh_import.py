"""Reader parity, sniffing, and the per-format traps."""

from __future__ import annotations

import base64
import dataclasses
import re
import xml.etree.ElementTree as ET  # nosec B405 - test-only, parses our own fixtures
import zlib
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from .._stl import is_binary_stl
from .._weld import prune_unreferenced_vertices, weld_vertices
from ..mesh_import import (
    MESH_FORMATS,
    TriangleMesh,
    detect_mesh_format,
    import_mesh,
    import_mesh_directory,
)
from ._synthetic import (
    OBJ_MID_COLORS,
    SUFFIXES,
    WRITERS,
    make_ground_truth,
    write_glb,
    write_glb_bad_node_index,
    write_glb_cyclic,
    write_glb_interleaved,
    write_glb_interleaved_at_buffer_end,
    write_glb_mirrored,
    write_glb_no_scenes,
    write_glb_shared_child,
    write_gltf_accessor_past_view,
    write_gltf_buffer_payload,
    write_gltf_dangling_accessor,
    write_gltf_draco,
    write_gltf_external_buffer,
    write_gltf_index_past_primitive,
    write_gsplat_ply,
    write_obj_colors_0_1,
    write_obj_colors_0_255,
    write_obj_colors_partial,
    write_obj_crease,
    write_obj_indexed_normals,
    write_obj_negative_indices,
    write_obj_out_of_range_index,
    write_obj_partial_normals,
    write_obj_quad,
    write_obj_unreferenced_normals,
    write_ply_ascii,
    write_ply_binary,
    write_ply_crease,
    write_ply_face_extras,
    write_ply_orphan_vertices,
    write_ply_quads,
    write_ply_truncated_ascii,
    write_stl_ascii,
    write_stl_binary,
    write_vtp,
    write_vtp_mixed_cells,
    write_vtp_point_data,
    write_vtp_points_only,
    write_vtp_polys,
    write_vtp_quad,
    write_vtp_two_pieces,
)

GT = make_ground_truth()


def _sorted_face_set(mesh: TriangleMesh) -> set[tuple[float, ...]]:
    """Faces as position triples, order-insensitive.

    Welding renumbers vertices, so index equality is not the invariant — the SURFACE
    is. Each face becomes its three corner positions, sorted, so a reader that welds
    to a different vertex order still compares equal while one that mangles topology
    does not.
    """
    out = set()
    for tri in mesh.faces:
        corners = tuple(sorted(tuple(np.round(mesh.vertices[i], 5)) for i in tri))
        out.add(tuple(c for corner in corners for c in corner))
    return out


EXPECTED_FACES = {
    tuple(
        c
        for corner in sorted(tuple(np.round(GT.vertices[i], 5)) for i in tri)
        for c in corner
    )
    for tri in GT.faces
}


def _signed_volume(mesh: TriangleMesh) -> float:
    """Six times the enclosed volume, summed over the triangles.

    Sign-sensitive to WINDING, which the sorted face set is blind to — the tetrahedron's
    faces are the same three positions either way round. Shared by the glTF mirroring
    test and the VTP strip test, which need exactly the same discrimination.
    """
    tri = mesh.vertices[mesh.faces]
    return float(
        np.sum(np.einsum("ij,ij->i", np.cross(tri[:, 0], tri[:, 1]), tri[:, 2]))
    )


@pytest.mark.parametrize(
    ("fmt", "unit_range"),
    [
        ("ply", True),
        ("obj", True),
        ("glb", True),
        ("vtp", True),
        ("ply", False),
        ("obj", False),
        ("vtp", False),
    ],
)
def test_float_colors_round_to_nearest_byte(
    fmt: str, unit_range: bool, tmp_path: Path
) -> None:
    authored = np.array(
        [
            [1.6, 63.6, 127.6],
            [2.6, 64.6, 128.6],
            [3.6, 65.6, 129.6],
            [4.6, 66.6, 130.6],
        ],
        dtype=np.float32,
    )
    expected = np.round(authored).astype(np.uint8)
    stored_colors = authored / 255.0 if unit_range else authored
    path = tmp_path / f"rounded.{fmt}"

    if fmt == "glb":
        write_glb(path, GT, float_colors=stored_colors)
    elif fmt == "vtp":
        write_vtp_point_data(
            path,
            GT,
            [(stored_colors, "Float32", "colors", 3)],
            pdata_attrs='Scalars="colors"',
        )
    else:
        header = []
        if fmt == "ply":
            header = [
                "ply",
                "format ascii 1.0",
                f"element vertex {len(GT.vertices)}",
                "property float x",
                "property float y",
                "property float z",
                "property float red",
                "property float green",
                "property float blue",
                f"element face {len(GT.faces)}",
                "property list uchar int vertex_indices",
                "end_header",
            ]
        prefix = "" if fmt == "ply" else "v "
        rows = [
            prefix + " ".join(str(float(value)) for value in (*vertex, *color))
            for vertex, color in zip(GT.vertices, stored_colors, strict=True)
        ]
        face_prefix = "3 " if fmt == "ply" else "f "
        offset = 0 if fmt == "ply" else 1
        faces = [
            face_prefix + " ".join(str(int(index) + offset) for index in face)
            for face in GT.faces
        ]
        path.write_text("\n".join(header + rows + faces) + "\n", encoding="ascii")

    mesh = import_mesh(path)
    assert mesh.colors is not None
    for vertex, color in zip(mesh.vertices, mesh.colors, strict=True):
        row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
        np.testing.assert_array_equal(color, expected[row])


@pytest.mark.parametrize(
    ("component_type", "authored", "expected"),
    [
        (
            5121,
            np.array(
                [[2, 64, 128], [3, 65, 129], [4, 66, 130], [5, 67, 131]],
                dtype=np.uint8,
            ),
            np.array(
                [[2, 64, 128], [3, 65, 129], [4, 66, 130], [5, 67, 131]],
                dtype=np.uint8,
            ),
        ),
        (
            5123,
            np.array(
                [
                    [129, 16_320, 32_768],
                    [643, 16_834, 33_282],
                    [1_157, 17_348, 33_796],
                    [1_671, 17_862, 34_310],
                ],
                dtype=np.uint16,
            ),
            np.array(
                [[1, 64, 128], [3, 66, 130], [5, 68, 132], [7, 70, 134]],
                dtype=np.uint8,
            ),
        ),
    ],
)
def test_gltf_integer_colors_use_normalized_component_range(
    component_type: int,
    authored: NDArray,
    expected: NDArray[np.uint8],
    tmp_path: Path,
) -> None:
    path = tmp_path / "integer-colors.glb"
    write_glb(
        path,
        GT,
        float_colors=authored,
        color_component_type=component_type,
    )

    mesh = import_mesh(path)
    assert mesh.colors is not None
    for vertex, color in zip(mesh.vertices, mesh.colors, strict=True):
        row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
        np.testing.assert_array_equal(color, expected[row])


class TestReaderParity:
    @pytest.mark.parametrize("fmt", MESH_FORMATS)
    def test_every_dialect_reproduces_the_tetrahedron(
        self, fmt: str, tmp_path: Path
    ) -> None:
        path = tmp_path / f"tetra{SUFFIXES[fmt]}"
        WRITERS[fmt](path, GT)
        mesh = import_mesh(path)

        assert mesh.source_format == fmt
        # Welding is load-bearing for STL, which arrives as a 12-corner soup and must
        # come back as the 4 shared vertices the tetrahedron actually has. The other
        # three fixtures are already indexed at 4, so this assertion is a WELD-NEUTRAL
        # parity check for them — `TestWelding` is what exercises the key itself.
        assert mesh.n_vertices == 4, f"{fmt} did not weld to 4 shared vertices"
        assert mesh.n_faces == 4
        assert _sorted_face_set(mesh) == EXPECTED_FACES

    @pytest.mark.parametrize("fmt", MESH_FORMATS)
    def test_detection_matches_the_writer(self, fmt: str, tmp_path: Path) -> None:
        path = tmp_path / f"sniff{SUFFIXES[fmt]}"
        WRITERS[fmt](path, GT)
        assert detect_mesh_format(path) == fmt

    def test_weld_false_preserves_the_files_own_vertex_list(
        self, tmp_path: Path
    ) -> None:
        # The anti-vacuity control for the welding assertions above: without it, a
        # reader that happened to emit 4 vertices for every format would look right.
        path = tmp_path / "soup.stl"
        write_stl_binary(path, GT)
        assert import_mesh(path, weld=False).n_vertices == 12  # 4 faces × 3 corners
        assert import_mesh(path, weld=True).n_vertices == 4

    def test_welding_prunes_vertices_outside_the_surviving_surface(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "orphan.ply"
        write_ply_orphan_vertices(path)

        mesh = import_mesh(path)

        assert mesh.n_vertices == 3
        assert mesh.n_faces == 1
        assert {tuple(vertex) for vertex in mesh.vertices} == {
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        }
        assert float(mesh.vertices.max()) == 1.0
        assert int(mesh.faces.max()) == 2
        assert mesh.normals is not None
        assert mesh.colors is not None
        expected = {
            (0.0, 0.0, 0.0): ((0.0, 1.0, 0.0), (40, 50, 60)),
            (1.0, 0.0, 0.0): ((0.0, 0.0, 1.0), (70, 80, 90)),
            (0.0, 1.0, 0.0): ((-1.0, 0.0, 0.0), (100, 110, 120)),
        }
        for vertex, normal, color in zip(mesh.vertices, mesh.normals, mesh.colors):
            expected_normal, expected_color = expected[tuple(vertex)]
            np.testing.assert_array_equal(normal, expected_normal)
            np.testing.assert_array_equal(color, expected_color)

    def test_weld_false_keeps_vertices_outside_the_surface(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "orphan.ply"
        write_ply_orphan_vertices(path)

        mesh = import_mesh(path, weld=False)

        assert mesh.n_vertices == 5
        assert mesh.n_faces == 1
        assert float(mesh.vertices.max()) == 200.0
        assert mesh.normals is not None and mesh.normals.shape == (5, 3)
        assert mesh.colors is not None and mesh.colors.shape == (5, 3)


class TestPly:
    def test_ascii_and_binary_agree(self, tmp_path: Path) -> None:
        # ASCII and big-endian are exactly what the gsplat PLY parser refuses, which
        # is why this reader exists separately.
        a, b = tmp_path / "a.ply", tmp_path / "b.ply"
        write_ply_ascii(a, GT)
        write_ply_binary(b, GT)
        assert _sorted_face_set(import_mesh(a)) == _sorted_face_set(import_mesh(b))

    def test_big_endian_body_decodes(self, tmp_path: Path) -> None:
        p = tmp_path / "be.ply"
        write_ply_binary(p, GT, big_endian=True)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_quads_are_fan_triangulated(self, tmp_path: Path) -> None:
        p = tmp_path / "quad.ply"
        write_ply_quads(p, GT)
        mesh = import_mesh(p)
        assert mesh.n_faces == 2, "one quad must become two triangles, not one"
        assert mesh.n_vertices == 4

    @pytest.mark.parametrize("binary", [True, False])
    @pytest.mark.parametrize("texcoord_first", [False, True])
    def test_face_properties_are_read_in_declaration_order(
        self, binary: bool, texcoord_first: bool, tmp_path: Path
    ) -> None:
        """A `face` element may carry a scalar BEFORE its list, and a second list.

        All legal and all common in the wild — a per-face flag ahead of
        `vertex_indices`, and the `texcoord` list an exporter that carries UVs writes
        beside it, in either order. Declaration order IS the row layout, so a reader
        that assumes list-first-scalars-after consumes the wrong bytes from row two
        onward; the binary arm decoded silently-wrong faces. The ascii arm walks tokens
        instead of bytes but has the same rule, and the `texcoord_first` arm is why the
        index list is chosen by NAME rather than by position — otherwise UV floats
        import as topology.
        """
        p = tmp_path / f"extras{'bin' if binary else 'asc'}{int(texcoord_first)}.ply"
        write_ply_face_extras(p, GT, binary=binary, texcoord_first=texcoord_first)
        mesh = import_mesh(p)
        assert mesh.n_faces == 4, "the extra properties desynchronized the face rows"
        assert _sorted_face_set(mesh) == EXPECTED_FACES

    def test_a_truncated_ascii_body_is_a_clean_error(self, tmp_path: Path) -> None:
        # A ValueError, not an IndexError: only the former is what the CLI's error
        # funnel catches, so anything else surfaces as a raw traceback.
        p = tmp_path / "short.ply"
        write_ply_truncated_ascii(p, GT)
        with pytest.raises(ValueError, match="truncated"):
            import_mesh(p)

    def test_normals_and_colors_survive(self, tmp_path: Path) -> None:
        p = tmp_path / "attrs.ply"
        write_ply_binary(p, GT)
        mesh = import_mesh(p)
        assert mesh.normals is not None and mesh.normals.shape == (4, 3)
        assert mesh.colors is not None and mesh.colors.shape == (4, 3)
        # Unit length: a reader that read the wrong three columns would not be.
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)


class TestObj:
    def test_indices_are_one_based(self, tmp_path: Path) -> None:
        # The trap: reading OBJ's 1-based indices as 0-based shifts every face by one
        # vertex, which still produces a valid-looking mesh.
        p = tmp_path / "t.obj"
        WRITERS["obj"](p, GT)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_negative_indices_resolve_from_the_end(self, tmp_path: Path) -> None:
        p = tmp_path / "neg.obj"
        write_obj_negative_indices(p, GT)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_vertex_colors_in_both_conventions_agree(self, tmp_path: Path) -> None:
        """`v x y z r g b` has no agreed range: MeshLab writes 0..1, scanners 0..255.

        Assuming 0..1 and scaling unconditionally clips every nonzero channel of a
        0..255 file to 255 — a coloured mesh imports white. Detect by observed peak,
        the same rule the PLY reader already applies to `red/green/blue`.
        """
        a, b = tmp_path / "unit.obj", tmp_path / "byte.obj"
        write_obj_colors_0_1(a, GT)
        write_obj_colors_0_255(b, GT)
        ca, cb = import_mesh(a).colors, import_mesh(b).colors
        assert ca is not None and cb is not None
        np.testing.assert_allclose(np.sort(ca, axis=0), np.sort(cb, axis=0), atol=1)
        # Pin the VALUES, not merely that the two agree: the palette is mid-range on
        # purpose, so a 255x-and-clip would drive every channel to 255 here. Asserting
        # only agreement would still pass if BOTH readers were wrong the same way.
        np.testing.assert_allclose(
            np.sort(cb, axis=0), np.sort(OBJ_MID_COLORS, axis=0), atol=1
        )
        assert int(cb.max()) < 255, "a 0..255 file must not clip to solid white"

    def test_a_colourless_vertex_is_filled_white_in_the_file_s_convention(
        self, tmp_path: Path
    ) -> None:
        """`v x y z` and `v x y z r g b` may be mixed in one file.

        The fill for the colourless rows has to be white in whichever convention the file
        uses. A fixed 1.0 is white only at 0..1: in a 0..255 file it is not scaled, so it
        arrives as RGB(1, 1, 1) — black, the most visible possible wrong answer.
        """
        p = tmp_path / "partial-colour.obj"
        write_obj_colors_partial(p, GT)
        mesh = import_mesh(p)
        assert mesh.colors is not None
        row = int(np.argmin(np.linalg.norm(mesh.vertices - GT.vertices[0], axis=1)))
        np.testing.assert_array_equal(mesh.colors[row], [255, 255, 255])

    def test_an_unreferenced_normal_pool_is_dropped(self, tmp_path: Path) -> None:
        """`vn` applies only where a face names it.

        A pool whose count happens to equal the vertex count but which no `f` corner
        references says nothing about which normal belongs to which vertex. Attaching it
        anyway invents per-vertex normals the exporter never bound, and the surface is
        then shaded by them — here every normal is +z, so the tetrahedron would light as
        if it were flat. The `v//vn` fixture in `test_indices_are_one_based` is this
        test's control: there the references exist and the normals are kept.
        """
        p = tmp_path / "unbound.obj"
        write_obj_unreferenced_normals(p, GT)
        assert import_mesh(p).normals is None
        WRITERS["obj"](tmp_path / "bound.obj", GT)
        assert import_mesh(tmp_path / "bound.obj").normals is not None

    def test_independently_indexed_normals_are_kept(self, tmp_path: Path) -> None:
        """`vn` is indexed per CORNER, and every real exporter deduplicates the pool.

        So the pool is generally neither the same length as the positions nor parallel
        to them, and a reader that requires parallel indexing drops the normals of
        essentially every smooth-shaded export. The fixture reverses the pool — counts
        still match, indexing does not — so requiring only equal counts reads every
        normal onto the wrong vertex, and requiring parallelism throws them all away.
        Neither is acceptable: split the vertices per (position, normal) pair instead.
        """
        p = tmp_path / "dedup.obj"
        write_obj_indexed_normals(p, GT)
        mesh = import_mesh(p)
        assert mesh.normals is not None, "a deduplicated `vn` pool must still bind"
        # Same surface, and each vertex carries the normal the file bound to it.
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        for vertex, normal in zip(mesh.vertices, mesh.normals):
            row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
            np.testing.assert_allclose(normal, GT.normals[row], atol=1e-5)

    def test_a_partial_normal_binding_is_dropped(self, tmp_path: Path) -> None:
        # One face written as bare `f a b c`, the rest as `f a//a`. There is no normal
        # for the unbound corners, and borrowing the positionally-matching pool entry
        # would shade them by data the exporter never bound — so the pool goes whole.
        p = tmp_path / "partial.obj"
        write_obj_partial_normals(p, GT)
        assert import_mesh(p).normals is None

    def test_a_split_preserves_a_crease_and_still_welds_a_smooth_join(
        self, tmp_path: Path
    ) -> None:
        """The pair that pins what the (position, normal) split is FOR.

        Both files hold two triangles over four positions with one `vn` per FACE — the
        shared corners carry a different normal in each face, which no per-vertex array
        can express without duplicating them. `hard`: the crease must survive, so the
        shared corners stay split (6 vertices) and each triangle is shaded by one normal.
        `smooth`: both faces agree, so the split must weld all the way back down to 4.
        Either half alone is passable by a broken reader — the pair is not.
        """
        hard, smooth = tmp_path / "hard.obj", tmp_path / "smooth.obj"
        write_obj_crease(hard, hard=True)
        write_obj_crease(smooth, hard=False)

        mh, ms = import_mesh(hard), import_mesh(smooth)
        assert mh.normals is not None and ms.normals is not None
        assert mh.n_faces == 2 and ms.n_faces == 2
        assert ms.n_vertices == 4, "an agreeing split must weld back to the shared list"
        assert mh.n_vertices == 6, "a crease must keep its corners split"
        for tri in mh.faces:
            per_corner = mh.normals[np.asarray(tri)]
            assert np.allclose(per_corner, per_corner[0]), (
                "a flat-shaded face must be shaded by exactly one normal"
            )
        assert not np.allclose(
            mh.normals[mh.faces[0][0]], mh.normals[mh.faces[1][0]]
        ), "the two faces' normals must stay distinct"

    @pytest.mark.parametrize("normal", [False, True])
    def test_an_out_of_range_index_is_a_clean_error(
        self, normal: bool, tmp_path: Path
    ) -> None:
        # A ValueError, not the OverflowError/IndexError a bare cast would raise: only
        # the former is what the CLI's error funnel catches. Both the `v` and the `vn`
        # reference are checked — the `vn` one would otherwise index the normal pool
        # out of bounds inside the split.
        p = tmp_path / "bad.obj"
        write_obj_out_of_range_index(p, GT, normal=normal)
        with pytest.raises(ValueError, match="malformed"):
            import_mesh(p)

    def test_quad_face_is_triangulated(self, tmp_path: Path) -> None:
        p = tmp_path / "q.obj"
        write_obj_quad(p)
        assert import_mesh(p).n_faces == 2


class TestStl:
    def test_binary_header_may_begin_with_solid(self, tmp_path: Path) -> None:
        # The classic misclassification: the synthetic binary STL's 80-byte header
        # starts with the word "solid", so magic-word sniffing calls it ASCII.
        p = tmp_path / "trap.stl"
        write_stl_binary(p, GT)
        raw = p.read_bytes()
        assert raw[:5] == b"solid"
        assert is_binary_stl(raw), "size arithmetic must win over the magic word"
        assert import_mesh(p).n_faces == 4

    def test_ascii_and_binary_agree(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.stl", tmp_path / "b.stl"
        write_stl_ascii(a, GT)
        write_stl_binary(b, GT)
        assert _sorted_face_set(import_mesh(a)) == _sorted_face_set(import_mesh(b))

    def test_per_facet_normals_are_dropped(self, tmp_path: Path) -> None:
        # Deliberate: STL normals are per-FACET, add_mesh takes per-VERTEX, and after
        # welding one vertex has three conflicting facet normals. Dropping them puts
        # the mesh on the derivative flat-normal path — the same faceted picture,
        # without inventing data.
        p = tmp_path / "n.stl"
        write_stl_binary(p, GT)
        assert import_mesh(p).normals is None


class TestGltf:
    def test_node_translation_is_applied(self, tmp_path: Path) -> None:
        # Skip the node graph and every part of a multi-part model stacks at the
        # origin — a plausible-looking, entirely wrong import.
        plain, moved = tmp_path / "a.glb", tmp_path / "b.glb"
        write_glb(plain, GT)
        write_glb(moved, GT, translation=[10.0, 0.0, 0.0])
        delta = import_mesh(moved).vertices.mean(axis=0) - import_mesh(
            plain
        ).vertices.mean(axis=0)
        np.testing.assert_allclose(delta, [10.0, 0.0, 0.0], atol=1e-5)

    def test_interleaved_accessors_decode(self, tmp_path: Path) -> None:
        # byteStride: ignore it and POSITION reads normal bytes as coordinates, which
        # decodes to garbage rather than raising.
        packed, interleaved = tmp_path / "p.glb", tmp_path / "i.glb"
        write_glb(packed, GT)
        write_glb_interleaved(interleaved, GT)
        np.testing.assert_allclose(
            np.sort(import_mesh(interleaved).vertices, axis=0),
            np.sort(import_mesh(packed).vertices, axis=0),
            atol=1e-6,
        )

    def test_a_node_cycle_is_refused_rather_than_recursing(
        self, tmp_path: Path
    ) -> None:
        """A malformed graph must not reach Python's recursion limit.

        glTF node graphs are a strict forest, so a child edge back into an ancestor is
        malformed. Before the ancestor check this recursed until `RecursionError` —
        which is not a `ValueError`, so it escaped the CLI's error funnel entirely and
        surfaced as a raw traceback instead of `Error: ...`.
        """
        p = tmp_path / "cycle.glb"
        write_glb_cyclic(p, GT)
        with pytest.raises(ValueError, match="cycle"):
            import_mesh(p)

    def test_a_shared_child_still_imports(self, tmp_path: Path) -> None:
        """The anti-overcorrection control: a DAG is not a cycle.

        Two parents pointing at one leaf is technically invalid glTF (nodes have at
        most one parent) but is harmless here — it just emits the mesh twice, which is
        what the file asks for. A global visited-set would have refused it; the check is
        ancestor-scoped precisely so it does not.
        """
        p = tmp_path / "dag.glb"
        write_glb_shared_child(p, GT)
        mesh = import_mesh(p)
        assert mesh.n_faces == 2 * GT.faces.shape[0]

    def test_out_of_range_node_index_is_named(self, tmp_path: Path) -> None:
        p = tmp_path / "badnode.glb"
        write_glb_bad_node_index(p, GT)
        with pytest.raises(ValueError, match="out of range"):
            import_mesh(p)

    def test_an_index_past_its_own_primitive_is_refused(self, tmp_path: Path) -> None:
        """Indices must be bounded per PRIMITIVE, before the concatenation offset.

        Primitives are merged into one vertex array, so an index past the end of its own
        primitive still lands inside the merged array and reads a LATER primitive's
        vertices. The final range check on the assembled mesh therefore passes and the
        triangle silently attaches to the wrong geometry.
        """
        p = tmp_path / "idx.gltf"
        write_gltf_index_past_primitive(p, GT)
        # Matched on the per-primitive wording, not merely "out of range": the assembled
        # mesh's own range check carries that phrase too, and this fixture is built so
        # that check PASSES (index 4 is valid against the 8 concatenated vertices).
        with pytest.raises(ValueError, match="its POSITION accessor declares"):
            import_mesh(p)

    def test_an_accessor_past_its_bufferView_is_refused(self, tmp_path: Path) -> None:
        """An accessor is bounded by its bufferView, not by the whole buffer.

        Views sit back to back in one buffer, so an accessor that overruns its own view
        reads the next view's bytes — here index data decoded as a coordinate — and a
        read bounded only by the buffer never notices.
        """
        p = tmp_path / "over.gltf"
        write_gltf_accessor_past_view(p, GT)
        # "outside its bufferView", not just "bufferView": the dangling-reference message
        # names `bufferViews[N]` and would otherwise match a fixture broken differently.
        with pytest.raises(ValueError, match="outside its bufferView"):
            import_mesh(p)

    def test_interleaved_view_at_the_END_of_the_buffer_decodes(
        self, tmp_path: Path
    ) -> None:
        """The over-read `test_interleaved_accessors_decode` cannot see.

        A valid accessor spans `(count - 1) * stride + element`, not `count * stride` —
        the last element occupies only its own width and the padding after it need not
        exist. Requesting the larger span reads past the buffer, which is invisible
        while anything follows the view (index data, in the other fixture) and raises
        only when the interleaved view is last. That is a tightly-packed exporter's
        normal output.
        """
        p = tmp_path / "tail.glb"
        write_glb_interleaved_at_buffer_end(p, GT)
        mesh = import_mesh(p)
        assert mesh.n_vertices == 4
        assert mesh.normals is not None
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)

    def test_a_reflecting_transform_reverses_winding(self, tmp_path: Path) -> None:
        """A negative-determinant node must flip the index order.

        `scale: [-1, 1, 1]` is routine for mirrored parts. Positions and normals get
        transformed correctly, but leaving the index ORDER alone makes the face's
        geometric winding disagree with its own normal — so the mirrored part faces away
        and disappears under single-sided rendering.
        """
        plain, mirrored = tmp_path / "p.glb", tmp_path / "m.glb"
        write_glb(plain, GT)
        write_glb_mirrored(mirrored, GT)
        a, b = import_mesh(plain), import_mesh(mirrored)

        # The mirror really was applied: x is negated.
        np.testing.assert_allclose(
            np.sort(b.vertices[:, 0]), np.sort(-a.vertices[:, 0]), atol=1e-6
        )

        # And every face still winds consistently with its own geometry. Compare the
        # signed volume contribution of each triangle: mirroring negates it, so a mesh
        # whose winding was NOT flipped would keep the original sign.
        va, vb = _signed_volume(a), _signed_volume(b)
        assert abs(va) > 1e-6, "the ground truth must enclose volume for this to bite"
        assert np.sign(vb) == np.sign(va), (
            "reflecting the geometry without flipping the winding inverts the surface "
            f"orientation (got {vb:.6f} against {va:.6f})"
        )

    def test_a_scene_less_node_graph_is_walked_from_its_own_roots(
        self, tmp_path: Path
    ) -> None:
        """`scenes` is optional, and its absence must not duplicate descendants.

        Walking every node as a root emits a child once through its parent (with the
        parent's transform) and again on its own (without it). Here that is the
        tetrahedron twice, one copy of it 10 units from where the file put it.
        """
        p = tmp_path / "library.glb"
        write_glb_no_scenes(p, GT)
        mesh = import_mesh(p)
        assert mesh.n_faces == GT.faces.shape[0], "the child was emitted twice"
        # And through the parent, so the translation was applied.
        np.testing.assert_allclose(
            mesh.vertices.min(axis=0), [10.0, 0.0, 0.0], atol=1e-6
        )

    def test_a_dangling_accessor_reference_is_named(self, tmp_path: Path) -> None:
        # Same reason the node walk range-checks its child edges: an IndexError is not
        # a ValueError, so it escapes the CLI's error funnel as a raw traceback.
        p = tmp_path / "dangling.gltf"
        write_gltf_dangling_accessor(p)
        with pytest.raises(ValueError, match="accessors"):
            import_mesh(p)

    def test_an_external_buffer_beside_the_gltf_loads(self, tmp_path: Path) -> None:
        p = tmp_path / "external.gltf"
        write_gltf_external_buffer(p, GT, "payload.bin")
        write_gltf_buffer_payload(tmp_path / "payload.bin", GT)
        assert import_mesh(p).n_faces == 4

    @pytest.mark.parametrize("uri", ["../outside.bin", "/etc/passwd"])
    def test_a_buffer_uri_may_not_escape_the_gltf_directory(
        self, uri: str, tmp_path: Path
    ) -> None:
        """The anti-traversal control for the test above.

        `uri` is data from the file, so a `..` climb or an absolute path would let a
        crafted glTF read any file this process can and reinterpret its bytes as
        geometry. The escape is refused before the read, not merely reported missing —
        the `..` target here EXISTS.
        """
        write_gltf_buffer_payload(tmp_path / "outside.bin", GT)
        nested = tmp_path / "model"
        nested.mkdir()
        p = nested / "escape.gltf"
        write_gltf_external_buffer(p, GT, uri)
        with pytest.raises(ValueError, match="outside the directory"):
            import_mesh(p)

    def test_draco_is_refused_by_name(self, tmp_path: Path) -> None:
        p = tmp_path / "draco.gltf"
        write_gltf_draco(p)
        with pytest.raises(ValueError, match="Draco"):
            import_mesh(p)


def _replace_array_text(raw: str, name: str, body: str) -> str:
    """Swap the ascii payload of the `<DataArray Name="name">` in `raw` for `body`."""
    patched, hits = re.subn(
        rf'(<DataArray[^>]*Name="{name}"[^>]*>)[^<]*', rf"\g<1>{body}", raw, count=1
    )
    assert hits == 1, f"no ascii DataArray named {name!r} in the fixture"
    return patched


def _appended_tail_start(raw: bytes) -> int:
    """The index of the first payload byte after the `<AppendedData ...>_` marker."""
    return raw.index(b"_", raw.index(b"<AppendedData")) + 1


def _patch_appended_header_word(raw: bytes, name: str, value: int) -> bytes:
    """Overwrite the leading UInt32 header word of the appended DataArray `name`.

    Targeted by NAME rather than by "the array at offset 0": the writer emits the
    PointData arrays in evaluation order, so offset 0 is a `UInt8` colour block whose
    one-byte item size cannot express a ragged payload at all.
    """
    match = re.search(
        rf'<DataArray[^>]*Name="{name}"[^>]*offset="(\d+)"', raw.decode("latin-1")
    )
    assert match is not None, f"no appended DataArray named {name!r} in the fixture"
    start = _appended_tail_start(raw) + int(match.group(1))
    out = bytearray(raw)
    out[start : start + 4] = value.to_bytes(4, "little")
    return bytes(out)


#: The tetrahedron's faces after a `shift` in x — the second `<Piece>`'s expected surface.
def _shifted_faces(shift: float) -> set[tuple[float, ...]]:
    delta = np.array([shift, 0.0, 0.0], dtype=np.float32)
    return {
        tuple(
            c
            for corner in sorted(
                tuple(np.round(GT.vertices[i] + delta, 5)) for i in tri
            )
            for c in corner
        )
        for tri in GT.faces
    }


#: A namespace URI to hang the VTP namespace fixtures on. Any URI does; this is VTK's.
_VTK_NS = "http://www.kitware.com/vtk"

#: (mode, compressed, header_type, big_endian) — the arms of the VTP encoding matrix.
#:
#: Every combination is a file a real writer emits: ParaView defaults to appended-raw,
#: PyVista writes inline base64 (`format="binary"`), VTK ≥ 9 defaults to a
#: UInt64 header, and legacy files are UInt32. Big-endian is synthetic — no common
#: writer emits it today — but the format allows it and `byte_order=` is a one-line
#: thing to ignore.
_VTP_MATRIX = [
    ("appended-raw", False, "UInt32", False),
    ("appended-raw", True, "UInt32", False),
    ("appended-raw", True, "UInt64", False),
    ("appended-base64", False, "UInt32", False),
    ("appended-base64", True, "UInt32", False),
    ("appended-base64", True, "UInt64", False),
    ("inline-base64", False, "UInt32", False),
    ("inline-base64", True, "UInt32", False),
    ("inline-base64", True, "UInt64", False),
    ("ascii", False, "UInt32", False),
    ("appended-raw", True, "UInt32", True),
    ("inline-base64", True, "UInt64", True),
]


class TestVtp:
    @pytest.mark.parametrize(
        "mode,compressed,header_type,big_endian",
        _VTP_MATRIX,
        ids=[
            f"{m}-{'zlib' if c else 'plain'}-{h}-{'be' if b else 'le'}"
            for m, c, h, b in _VTP_MATRIX
        ],
    )
    def test_every_encoding_decodes_the_same_surface(
        self,
        mode: str,
        compressed: bool,
        header_type: str,
        big_endian: bool,
        tmp_path: Path,
    ) -> None:
        p = tmp_path / "tetra.vtp"
        write_vtp(
            p,
            GT,
            mode=mode,
            compressed=compressed,
            header_type=header_type,
            big_endian=big_endian,
        )
        mesh = import_mesh(p)
        assert mesh.n_vertices == 4 and mesh.n_faces == 4
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        # Attributes too: the encoding matrix is per-DataArray, so a header width read
        # wrong on the SECOND array would leave the positions intact and corrupt these.
        assert mesh.normals is not None and mesh.colors is not None
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)
        for vertex, color in zip(mesh.vertices, mesh.colors):
            row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
            np.testing.assert_array_equal(color, GT.colors[row])

    def test_a_compressed_inline_block_is_two_base64_streams(
        self, tmp_path: Path
    ) -> None:
        """Trap 2, pinned directly.

        With `vtkZLibDataCompressor` VTK base64-encodes the BLOCK HEADER and the
        compressed payload separately and concatenates the two encodings in the element
        text. `b64decode` of the whole text succeeds and yields bytes that are not the
        file's data — the misparse never raises, it just produces a different mesh. So
        the guard is equality with the uncompressed encoding of the same geometry, plus
        a direct check that the naive single-stream decode really would differ.
        """
        plain, zipped = tmp_path / "plain.vtp", tmp_path / "zlib.vtp"
        write_vtp(plain, GT, mode="inline-base64", compressed=False)
        write_vtp(zipped, GT, mode="inline-base64", compressed=True)
        a, b = import_mesh(plain), import_mesh(zipped)
        assert _sorted_face_set(a) == _sorted_face_set(b) == EXPECTED_FACES
        np.testing.assert_allclose(
            np.sort(a.vertices, axis=0), np.sort(b.vertices, axis=0), atol=1e-6
        )
        assert a.colors is not None and b.colors is not None
        np.testing.assert_array_equal(
            np.sort(a.colors, axis=0), np.sort(b.colors, axis=0)
        )

        # The anti-vacuity half: reconstruct the positions from the raw element text by
        # hand, so the layout the reader relies on is pinned independently of it.
        root = ET.fromstring(zipped.read_bytes())  # nosec B314 - our own fixture
        points_el = next(e for e in root.iter("DataArray") if e.get("Name") == "Points")
        text = "".join((points_el.text or "").split())

        # Stream one: four UInt32 header words (nblocks, block size, last partial size,
        # compressed size) = 16 bytes = 24 base64 characters, trailing padding included.
        nblocks, _block, _last, csize = (
            int(w) for w in np.frombuffer(base64.b64decode(text[:24]), dtype="<u4")
        )
        assert nblocks == 1
        # Stream two starts at character 24 — a SECOND base64 stream, not a continuation.
        payload = base64.b64decode(text[24 : 24 + ((csize + 2) // 3) * 4])
        np.testing.assert_array_equal(
            np.frombuffer(zlib.decompress(payload[:csize]), dtype="<f4").reshape(-1, 3),
            GT.vertices,
        )
        # And the naive whole-text decode does NOT deliver those compressed bytes: the
        # header stream's own '=' padding terminates the decode, so the entire payload
        # silently vanishes instead of raising.
        assert len(base64.b64decode(text)) - 16 != csize, (
            "the fixture must actually concatenate two base64 streams, or this test "
            "cannot distinguish the right decode from the wrong one"
        )

    def test_non_base64_characters_cannot_shorten_an_appended_block(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "short-base64.vtp"
        write_vtp(p, GT, mode="appended-base64")
        raw = p.read_bytes()
        tag = raw.index(b"<AppendedData")
        start = raw.index(b"_", tag) + 1
        p.write_bytes(raw[:start] + b"<<<<" + raw[start + 4 :])
        with pytest.raises(
            ValueError, match=r"short-base64\.vtp.*base64 stream decoded to 3 bytes"
        ):
            import_mesh(p)

    def test_multi_block_compression_decodes(self, tmp_path: Path) -> None:
        """`nblocks > 1` — the case that makes the block header's LENGTH variable.

        With one block the header is always four words wide, so a reader that hardcoded
        that width would pass every other compressed arm. A tiny block size forces
        several compressed sizes into the header and the payload stream to start
        further along.
        """
        p = tmp_path / "blocks.vtp"
        write_vtp(p, GT, mode="inline-base64", compressed=True, block_size=16)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_appended_raw_is_not_well_formed_xml(self, tmp_path: Path) -> None:
        """Trap 1: the whole document has to be split before it is parsed.

        `encoding="raw"` puts arbitrary binary — NUL bytes here, and `<`/`&` in general —
        inside the document, so `ElementTree` refuses the file outright, header and all.
        """
        p = tmp_path / "raw.vtp"
        write_vtp(p, GT, mode="appended-raw")
        raw = p.read_bytes()
        assert b"\x00" in raw.split(b"<AppendedData", 1)[1]
        with pytest.raises(ET.ParseError):
            ET.fromstring(raw)  # nosec B314 - asserting that this FAILS
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_offsets_are_cumulative_ends_not_starts(self, tmp_path: Path) -> None:
        """A quad and a triangle in one `<Polys>`, so cell LENGTHS differ.

        VTK XML's `offsets` are cumulative END offsets with no leading zero. Read as
        start offsets, uniform triangles merely rotate by one cell; mixed lengths also
        mis-size every cell, so the face count and the surface both change.
        """
        p = tmp_path / "mixed.vtp"
        write_vtp_mixed_cells(p, GT)
        mesh = import_mesh(p)
        # Quad → 2 triangles, plus the standalone triangle.
        assert mesh.n_faces == 3
        assert mesh.n_vertices == 5
        expected = {
            tuple(
                c
                for corner in sorted(tuple(np.round(v, 5)) for v in tri)
                for c in corner
            )
            for tri in (
                [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
                [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
                [[1, 0, 0], [2, 0.5, 0], [1, 1, 0]],
            )
        }
        assert _sorted_face_set(mesh) == expected

    def test_quad_is_fan_triangulated(self, tmp_path: Path) -> None:
        p = tmp_path / "quad.vtp"
        write_vtp_quad(p, GT)
        mesh = import_mesh(p)
        assert mesh.n_faces == 2, "one quad must become two triangles"
        assert mesh.n_vertices == 4

    def test_strips_triangulate_with_a_consistent_winding(self, tmp_path: Path) -> None:
        """A triangle strip alternates winding: `i,i+1,i+2` then `i+1,i,i+2`.

        Drop the flip and every second triangle of a strip faces backwards, which
        single-sided rendering shows as holes. The face SETS are identical either way —
        reversing a triangle does not change which three vertices it has — so this is
        asserted on signed volume against the same surface written as polygons.
        """
        polys, strips = tmp_path / "polys.vtp", tmp_path / "strips.vtp"
        write_vtp(polys, GT)
        write_vtp(strips, GT, strips=True)
        a, b = import_mesh(polys), import_mesh(strips)

        assert b.n_faces == 4
        assert _sorted_face_set(b) == EXPECTED_FACES
        va, vb = _signed_volume(a), _signed_volume(b)
        assert abs(va) > 1e-6, "the ground truth must enclose volume for this to bite"
        np.testing.assert_allclose(vb, va, atol=1e-6)

    def test_verts_and_lines_beside_polys_are_dropped(self, tmp_path: Path) -> None:
        """`<Verts>`/`<Lines>` carry no surface, so they are dropped without comment.

        Folding them into the topology would add spurious "faces"; refusing the file
        would reject perfectly ordinary PolyData output.

        The fixture's cells are deliberately LONG ENOUGH to triangulate: a 3-point
        polyline (what `vtkFeatureEdges` and every contour filter emits) and a 3-point
        poly-vertex cell each fan into one real triangle over tetrahedron corners. With
        the 1-index verts and 2-point line this fixture used to carry, fan triangulation
        yielded nothing from either and a reader that folded them straight in still
        returned 4 faces — the test could not fail.
        """
        plain, mixed = tmp_path / "plain.vtp", tmp_path / "mixed.vtp"
        write_vtp(plain, GT)
        write_vtp(mixed, GT, with_verts_and_lines=True)
        plain_mesh = import_mesh(plain)
        mixed_mesh = import_mesh(mixed)
        assert mixed_mesh.n_faces == plain_mesh.n_faces == 4
        assert mixed_mesh.n_vertices == plain_mesh.n_vertices == len(GT.vertices)
        assert _sorted_face_set(mixed_mesh) == EXPECTED_FACES
        np.testing.assert_array_equal(mixed_mesh.vertices.min(axis=0), [0, 0, 0])
        np.testing.assert_array_equal(mixed_mesh.vertices.max(axis=0), [1, 1, 1])

    def test_a_surface_less_polydata_is_a_clean_error(self, tmp_path: Path) -> None:
        p = tmp_path / "cloud.vtp"
        write_vtp_points_only(p, GT)
        with pytest.raises(ValueError, match="no <Polys> or <Strips>"):
            import_mesh(p)

    @pytest.mark.parametrize("convention", ["float01", "float255"])
    def test_float_colours_are_read_by_range(
        self, convention: str, tmp_path: Path
    ) -> None:
        """`PointData` colours may be UInt8 0..255 or Float32 in either convention.

        The palette is MID-range on purpose: with an all-0/255 palette a reader that
        scaled a 0..255 file by 255 and clipped would still produce the right answer.

        Exact equality, not `atol=1`: the float32 round-trip of these twelve channels IS
        exact in both conventions, and a one-count tolerance is blind to exactly the
        error class a truncate-vs-round or a 254-vs-255 scale factor produces.
        """
        gt = dataclasses.replace(GT, colors=OBJ_MID_COLORS)
        byte_file, float_file = tmp_path / "u8.vtp", tmp_path / "f32.vtp"
        write_vtp(byte_file, gt, colors="uint8")
        write_vtp(float_file, gt, colors=convention)
        a, b = import_mesh(byte_file).colors, import_mesh(float_file).colors
        assert a is not None and b is not None
        np.testing.assert_array_equal(np.sort(a, axis=0), np.sort(b, axis=0))
        np.testing.assert_array_equal(
            np.sort(a, axis=0), np.sort(OBJ_MID_COLORS, axis=0)
        )
        assert int(b.max()) < 255, "a mid-range palette must not clip to solid white"

    def test_float_colours_outside_the_byte_range_are_clipped(
        self, tmp_path: Path
    ) -> None:
        values = np.array(
            [[-5, 10, 300], [260, -1, 128], [64, 255, 256], [1, 2, 3]],
            dtype=np.float32,
        )
        expected = np.clip(values, 0, 255).astype(np.uint8)
        p = tmp_path / "clipped.vtp"
        write_vtp_point_data(p, GT, [(values, "Float32", "colors", 3)])
        mesh = import_mesh(p)
        assert mesh.colors is not None
        for vertex, color in zip(mesh.vertices, mesh.colors):
            row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
            np.testing.assert_array_equal(color, expected[row])

    @pytest.mark.parametrize(
        "compressor", ["vtkLZ4DataCompressor", "vtkLZMADataCompressor"]
    )
    def test_an_unsupported_compressor_is_refused_by_name(
        self, compressor: str, tmp_path: Path
    ) -> None:
        """Never a silent misparse.

        The fixture's data is written UNCOMPRESSED, so a reader that ignored the
        `compressor=` attribute would import it perfectly and only fail on a real LZ4
        file — where it would inflate garbage instead. The refusal has to key on the
        declared name.
        """
        p = tmp_path / "lz4.vtp"
        write_vtp(p, GT, compressor=compressor)
        with pytest.raises(ValueError, match=compressor):
            import_mesh(p)
        # And the same bytes under the supported name do import, so the refusal is not
        # accidentally rejecting the fixture for some other reason.
        ok = tmp_path / "ok.vtp"
        write_vtp(ok, GT)
        assert import_mesh(ok).n_faces == 4

    @pytest.mark.parametrize("suffix", [".vtp", ".vtu"])
    def test_a_non_polydata_vtk_file_names_its_actual_type(
        self, suffix: str, tmp_path: Path
    ) -> None:
        """A `.vtu` is a volume mesh, not a surface — say so at the sniffer.

        Both spellings land on one message: the extension a user typed and the type the
        file actually declares are independent, and a renamed `.vtu` is the more
        confusing of the two.
        """
        p = tmp_path / f"volume{suffix}"
        write_vtp(p, GT, root_type="UnstructuredGrid")
        with pytest.raises(ValueError, match="UnstructuredGrid"):
            import_mesh(p)
        with pytest.raises(ValueError, match="PolyData"):
            detect_mesh_format(p)

    def test_a_vtp_that_is_not_vtk_xml_at_all_is_named(self, tmp_path: Path) -> None:
        p = tmp_path / "junk.vtp"
        p.write_bytes(b"<html><body>not a mesh</body></html>")
        with pytest.raises(ValueError, match="not a VTK XML file"):
            import_mesh(p)

    @pytest.mark.parametrize("declared", ["7", "3"])
    def test_a_declared_point_count_that_disagrees_is_named(
        self, declared: str, tmp_path: Path
    ) -> None:
        """`NumberOfPoints` is a free cross-check on the whole decode chain.

        A block header read at the wrong width, or an offset off by a stream, yields a
        differently-sized array rather than an exception — this is what turns that into
        a named error.

        BOTH directions, because the check is `!=`: an over-declaring file (7 against 4)
        alone leaves `>` indistinguishable from `!=`, and an under-declaring one is the
        commoner corruption — a truncated write, or a piece whose arrays were appended to
        without its header being updated.
        """
        p = tmp_path / "count.vtp"
        write_vtp(p, GT, mode="ascii")
        text = p.read_text(encoding="ascii").replace(
            'NumberOfPoints="4"', f'NumberOfPoints="{declared}"'
        )
        p.write_text(text, encoding="ascii")
        with pytest.raises(ValueError, match="NumberOfPoints"):
            import_mesh(p)

    # ---------------------------------------------------------------- PointData rules

    def test_a_nameless_float_triple_is_neither_normals_nor_colour(
        self, tmp_path: Path
    ) -> None:
        """`<PointData>` need not designate normals, and a `<DataArray>` need not be named.

        With neither present, matching `e.get("Name") == pdata.get("Normals")` compares
        `None` with `None` and the first NAMELESS array is adopted as normals. Here that
        is a velocity field — the module's documented reason for leaving a nameless float
        3-vector alone — and adopting it shades the surface with vectors that are not
        normals at all.
        """
        p = tmp_path / "velocity.vtp"
        velocity = np.array(
            [[1, 0, 0], [0, 2, 0], [0, 0, 3], [4, 4, 4]], dtype=np.float32
        )
        write_vtp_point_data(p, GT, [(velocity, "Float32", None, 3)])
        mesh = import_mesh(p)
        assert mesh.normals is None, "a nameless float 3-vector is not a normal field"
        assert mesh.colors is None, "...and it is not colour either"
        assert _sorted_face_set(mesh) == EXPECTED_FACES

    def test_a_nameless_uint8_triple_is_colour(self, tmp_path: Path) -> None:
        """VTK's own convention: unsigned-char 3-component point data IS colour.

        Nameless is the normal spelling for it, so the fallback at the end of the colour
        search has to be REACHABLE — which it only is once a nameless array stops being
        swallowed as normals first.
        """
        p = tmp_path / "vtkcolors.vtp"
        write_vtp_point_data(p, GT, [(GT.colors, "UInt8", None, 3)])
        mesh = import_mesh(p)
        assert mesh.normals is None, "a colour array was adopted as normals"
        assert mesh.colors is not None
        for vertex, color in zip(mesh.vertices, mesh.colors):
            row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
            np.testing.assert_array_equal(color, GT.colors[row])

    def test_a_nameless_rgba_array_keeps_its_fourth_channel(
        self, tmp_path: Path
    ) -> None:
        """4-component colour, which the rest of the fixtures never emit.

        Under the `None == None` match this was not merely mis-attributed but a hard
        error: the normals check demands `(N, 3)` and blamed the colour array for it.
        """
        p = tmp_path / "rgba.vtp"
        rgba = np.hstack([GT.colors, np.full((len(GT.colors), 1), 128, dtype=np.uint8)])
        write_vtp_point_data(p, GT, [(rgba, "UInt8", None, 4)])
        mesh = import_mesh(p)
        assert mesh.normals is None
        assert mesh.colors is not None and mesh.colors.shape == (4, 4)
        np.testing.assert_array_equal(mesh.colors[:, 3], 128)

    def test_the_normals_and_scalars_designations_are_honoured(
        self, tmp_path: Path
    ) -> None:
        """`Normals=` / `Scalars=` may name an array whose `Name=` is anything.

        The default fixture calls them "Normals" and "colors", which the reader's
        literal-name and colour-name fallbacks find anyway — so ignoring the designations
        entirely is invisible there. These two names match no fallback, and the colour
        array is Float32 so the UInt8 convention cannot rescue it either.
        """
        p = tmp_path / "designated.vtp"
        write_vtp(
            p,
            GT,
            normals_name="SurfaceNormals",
            colors_name="CellTint",
            colors="float255",
        )
        mesh = import_mesh(p)
        assert mesh.normals is not None, '<PointData Normals="SurfaceNormals"> ignored'
        assert mesh.colors is not None, '<PointData Scalars="CellTint"> ignored'
        np.testing.assert_allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-5)

    def test_a_normals_array_is_not_reused_as_colour(self, tmp_path: Path) -> None:
        p = tmp_path / "same-designation.vtp"
        write_vtp_point_data(
            p,
            GT,
            [(GT.normals, "Float32", "N", 3)],
            pdata_attrs='Normals="N" Scalars="N"',
        )
        mesh = import_mesh(p)
        assert mesh.normals is not None
        assert mesh.colors is None, "the normals array was reused as point colour"

    def test_a_conventional_colour_NAME_is_enough_on_its_own(
        self, tmp_path: Path
    ) -> None:
        """The middle of the three colour rules, which nothing else can reach.

        `<PointData>` need not designate `Scalars=`, and VTK's UInt8 convention only
        covers unsigned-char arrays. A **Float32** array called "Colors" with no
        designation is left to the conventional-spelling rule alone — and every other
        fixture either designates its colour array or is UInt8, so without this one the
        whole branch could be deleted and nothing would notice.
        """
        p = tmp_path / "namedcolor.vtp"
        write_vtp_point_data(
            p, GT, [(GT.colors.astype(np.float32), "Float32", "Colors", 3)]
        )
        assert b'Scalars="' not in p.read_bytes(), "the fixture must NOT designate one"
        mesh = import_mesh(p)
        assert mesh.normals is None
        assert mesh.colors is not None, '<DataArray Name="Colors"> was not taken'
        for vertex, color in zip(mesh.vertices, mesh.colors):
            row = int(np.argmin(np.linalg.norm(GT.vertices - vertex, axis=1)))
            np.testing.assert_array_equal(color, GT.colors[row])

    def test_a_saturated_0_to_1_palette_is_still_the_0_to_1_convention(
        self, tmp_path: Path
    ) -> None:
        """`peak <= 1.0`, inclusive — the boundary the mid-range palette cannot reach.

        Any 0..1 colour array containing one fully saturated channel (pure white, pure
        red, a highlight) peaks at exactly 1.0. Under a `peak < 1.0` test that file falls
        through to the 0..255 branch, where every channel clips to 0 or 1 and the surface
        renders black. The palette is otherwise mid-range so the arm stays honest about
        scaling rather than being trivially right at both ends.
        """
        palette = np.array(
            [[255, 64, 32], [10, 200, 90], [77, 77, 77], [3, 250, 128]], dtype=np.uint8
        )
        gt = dataclasses.replace(GT, colors=palette)
        p = tmp_path / "saturated.vtp"
        write_vtp(p, gt, colors="float01")
        mesh = import_mesh(p)
        assert mesh.colors is not None
        assert int(mesh.colors.max()) == 255, "the fixture must saturate a channel"
        np.testing.assert_array_equal(
            np.sort(mesh.colors, axis=0), np.sort(palette, axis=0)
        )

    @pytest.mark.parametrize(
        "attr,name,vtk_type,message",
        [
            ("Normals", "Normals", "Float32", "normals decoded to"),
            ("colours", "colors", "UInt8", "colours decoded to"),
        ],
    )
    def test_a_point_data_array_shorter_than_the_point_list_is_named(
        self, attr: str, name: str, vtk_type: str, message: str, tmp_path: Path
    ) -> None:
        """A `PointData` array must have one row per point, and both guards say so.

        Three rows against four points is what a truncated block, a wrong-width header or
        a piece assembled from mismatched arrays produces — and it does not raise on its
        own: `add_mesh` would be handed an attribute shorter than the vertex list, or (in
        a multi-piece file) a stack whose rows no longer line up with any piece.
        """
        values = (GT.normals if attr == "Normals" else GT.colors)[:3]
        p = tmp_path / "short.vtp"
        write_vtp_point_data(p, GT, [(values, vtk_type, name, 3)])
        with pytest.raises(ValueError, match=rf"short\.vtp.*{message}"):
            import_mesh(p)

    # -------------------------------------------------------------- multiple <Piece>s

    def test_two_pieces_are_rebased_into_one_vertex_array(self, tmp_path: Path) -> None:
        """Every `<Piece>` numbers its own points from 0.

        They are concatenated into one array, so each piece's indices shift by the
        running base. Drop the shift and the second tetrahedron's faces silently
        re-describe the first — a plausible surface, not an error. The two pieces are
        disjoint in space so welding cannot merge them and hide it.
        """
        p = tmp_path / "pieces.vtp"
        write_vtp_two_pieces(p, GT, shift=10.0)
        mesh = import_mesh(p)
        assert mesh.n_vertices == 8 and mesh.n_faces == 8
        assert _sorted_face_set(mesh) == EXPECTED_FACES | _shifted_faces(10.0)
        assert mesh.normals is not None and mesh.normals.shape == (8, 3)
        assert mesh.colors is not None and mesh.colors.shape == (8, 3)

    def test_normals_on_only_one_piece_are_dropped_whole(self, tmp_path: Path) -> None:
        """All or nothing: the pieces without normals have no per-vertex value.

        Keeping the partial set means either an array shorter than the vertex list or
        invented rows shading half the model with data the file never gave. Colours are
        present on both pieces, so this is not a blanket "drop everything".
        """
        p = tmp_path / "half.vtp"
        write_vtp_two_pieces(p, GT, normals_on=(True, False))
        mesh = import_mesh(p)
        assert mesh.normals is None, "normals were invented for the piece without them"
        assert mesh.colors is not None and mesh.colors.shape[0] == mesh.n_vertices

    def test_pieces_that_disagree_on_colour_width_are_named(
        self, tmp_path: Path
    ) -> None:
        """RGB in one piece, RGBA in the next — not stackable, and `np.concatenate`'s
        own complaint names neither the file nor the attribute."""
        p = tmp_path / "widths.vtp"
        write_vtp_two_pieces(p, GT, color_ncomps=(3, 4))
        with pytest.raises(ValueError, match=r"widths\.vtp.*different widths"):
            import_mesh(p)

    # ------------------------------------------------------- markup the format allows

    def test_a_canonicalized_root_tag_still_sniffs_as_polydata(
        self, tmp_path: Path
    ) -> None:
        """XML attribute order is not semantic, and C14N sorts it alphabetically.

        That puts `header_type=` ahead of `type=`, so an unanchored `type\\s*=` regex
        reads the header WIDTH as the dataset type and refuses a valid file as
        "type is 'UInt32'". Purely a sniffer defect: `read_vtp` asks the parsed tree, so
        the explicit-format path reads the same bytes fine.
        """
        p = tmp_path / "canonical.vtp"
        write_vtp(p, GT, mode="ascii")
        canonical = (
            '<VTKFile byte_order="LittleEndian" header_type="UInt32" '
            'type="PolyData" version="1.0">'
        )
        text, hits = re.subn(
            r"<VTKFile [^>]*>", canonical, p.read_text(encoding="ascii"), count=1
        )
        assert hits == 1
        p.write_text(text, encoding="ascii")
        assert detect_mesh_format(p) == "vtp"
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_a_present_but_empty_polys_group_is_zero_cells(
        self, tmp_path: Path
    ) -> None:
        """A surface written entirely as `<Strips>` may still carry `<Polys></Polys>`.

        An ABSENT cell group is already read as zero cells; a present-but-empty one was
        refused for a missing 'connectivity', which rejects the whole file.
        """
        p = tmp_path / "empty-polys.vtp"
        write_vtp(p, GT, strips=True, empty_polys=True)
        assert b"<Polys>" in p.read_bytes(), "the fixture must emit the empty group"
        mesh = import_mesh(p)
        assert mesh.n_faces == 4
        assert _sorted_face_set(mesh) == EXPECTED_FACES

    @pytest.mark.parametrize("mode", ["inline-base64", "appended-raw"])
    def test_a_default_namespace_is_read_on_both_paths(
        self, mode: str, tmp_path: Path
    ) -> None:
        """A namespaced `<VTKFile>` — legal, and what `_localname` exists for.

        Both paths are exercised because they reach the tree by different routes and a
        namespace can break either one. The inline path parses a complete document; the
        appended path parses only the cut prefix through a pull parser and takes the root
        off its first `start` event. Every tag arrives in Clark notation (`{uri}Piece`)
        on both, so a single literal tag comparison anywhere in the reader turns a
        perfectly ordinary ParaView-with-a-namespace file into "no `<Points>`" — and the
        two paths would not fail together.
        """
        p = tmp_path / "ns.vtp"
        write_vtp(p, GT, mode=mode, xmlns="http://www.kitware.com/vtk")
        assert b'xmlns="http://www.kitware.com/vtk"' in p.read_bytes()
        assert detect_mesh_format(p) == "vtp"
        mesh = import_mesh(p)
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        assert mesh.normals is not None and mesh.colors is not None

    def test_a_single_quoted_type_attribute_sniffs_the_same(
        self, tmp_path: Path
    ) -> None:
        """`AttValue ::= '"' … '"' | "'" … "'"` — both quotes are XML.

        The sniffer is a regex (the appended-raw arm is not parseable XML at all), so it
        has to accept what the parser does. Matching only double quotes splits the two
        apart: `detect_mesh_format` refuses the file as "no `<VTKFile type=…>` root
        element" while `read_vtp`, which asks the parsed tree, reads the very same bytes
        without complaint.
        """
        p = tmp_path / "singlequote.vtp"
        write_vtp(p, GT, mode="ascii")
        text, hits = re.subn(
            r'type="PolyData"',
            "type='PolyData'",
            p.read_text(encoding="ascii"),
            count=1,
        )
        assert hits == 1
        p.write_text(text, encoding="ascii")
        assert detect_mesh_format(p) == "vtp"
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_a_single_quoted_appended_encoding_is_honoured(
        self, tmp_path: Path
    ) -> None:
        """The same widening, on the arm where it corrupts DATA rather than a sniff.

        `<AppendedData encoding='base64'>` unmatched means the base64 tail is decoded as
        raw bytes, and the first four ASCII characters of a base64 stream read as a
        UInt32 byte count in the billions — so a valid file is reported as truncated.
        """
        p = tmp_path / "sqenc.vtp"
        write_vtp(p, GT, mode="appended-base64")
        raw = p.read_bytes()
        patched = raw.replace(b'encoding="base64"', b"encoding='base64'", 1)
        assert patched != raw
        p.write_bytes(patched)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    @pytest.mark.parametrize("mode", ["appended-raw", "appended-base64"])
    def test_an_appended_block_with_no_encoding_is_refused_by_name(
        self, mode: str, tmp_path: Path
    ) -> None:
        """A deliberate refusal, not a default.

        Raw bytes and base64 text cannot be told apart from the payload, and a wrong
        guess does not fail honestly — base64 read as raw reports a byte count in the
        billions and blames the data, on a file that is perfectly valid. Every VTK writer
        emits the attribute, so naming its absence costs nothing real.
        """
        p = tmp_path / "noencoding.vtp"
        write_vtp(p, GT, mode=mode)
        raw = p.read_bytes()
        patched = re.sub(rb' encoding="[^"]*"', b"", raw, count=1)
        assert patched != raw
        p.write_bytes(patched)
        with pytest.raises(ValueError, match=r"noencoding\.vtp.*no encoding="):
            import_mesh(p)

    def test_a_file_that_is_nothing_but_an_appended_section_is_named(
        self, tmp_path: Path
    ) -> None:
        """The cut can leave NO header at all, and there is then no root to return.

        Reachable through the explicit-format path, the same way `read_vtp`'s own type
        check is. Without the guard the header parser hands back `None` and the very next
        line asks it for `.tag`, so an `AttributeError` escapes the ValueError contract.
        """
        p = tmp_path / "headless.vtp"
        p.write_bytes(
            b'<AppendedData encoding="raw">\n_\x00\x01\x02\n</AppendedData>\n'
        )
        with pytest.raises(
            ValueError, match=r"headless\.vtp: malformed VTK XML header"
        ):
            import_mesh(p, format="vtp")

    def test_a_document_with_no_appended_section_must_still_be_complete(
        self, tmp_path: Path
    ) -> None:
        """Only the appended arm is allowed to be missing its closing tags.

        The header parser tolerates an unclosed document because the stream was CUT at
        `<AppendedData` on purpose. A file with no appended section was not cut, so the
        same tolerance would swallow a genuinely truncated write — here one that stops
        after the last `</Piece>`, which still carries a complete-looking surface.
        """
        p = tmp_path / "cut.vtp"
        write_vtp(p, GT, mode="inline-base64")
        raw = p.read_bytes()
        p.write_bytes(raw[: raw.index(b"</PolyData>")])
        with pytest.raises(ValueError, match=r"cut\.vtp: malformed VTK XML header"):
            import_mesh(p)

    @pytest.mark.parametrize(
        "find,replace",
        [
            pytest.param(
                b'Name="offsets"', b'Name="off&sets"', id="unescaped-ampersand"
            ),
            pytest.param(b"<Polys>", b"</Points><Polys>", id="stray-closing-tag"),
            pytest.param(b'Name="offsets"', b'Name="&nope;"', id="undefined-entity"),
        ],
    )
    def test_markup_broken_after_the_root_tag_is_refused(
        self, find: bytes, replace: bytes, tmp_path: Path
    ) -> None:
        """`XMLPullParser.feed` STORES a markup error; only iteration re-raises it.

        `feed` catches the parser's `SyntaxError` and appends it to the event queue, so
        the only thing that can surface an error past the root start tag is draining that
        queue — `close()`, which would also raise it, is deliberately not called on the
        appended arm because the prefix is legitimately unterminated. Taking the root off
        the FIRST `start` event and breaking therefore buried it, and the reader went on
        to import whatever elements the parser had already handed over: here the first
        `<Piece>` of two, i.e. half the surface, reported as a successful import.

        Each corruption is applied inside the SECOND `<Piece>` so the swallowed-error
        outcome is a plausible smaller mesh rather than an obvious empty one.
        """
        p = tmp_path / "broken.vtp"
        write_vtp_two_pieces(p, GT, mode="appended-raw")
        assert import_mesh(p).n_faces == 2 * len(GT.faces), (
            "the intact fixture must carry BOTH pieces, or a swallowed error is invisible"
        )

        raw = p.read_bytes()
        marker = raw.index(b"<AppendedData")
        head, tail = raw[:marker], raw[marker:]
        second = head.rindex(b"<Polys>")  # inside the second <Piece>
        patched = head[:second] + head[second:].replace(find, replace, 1)
        assert patched != head, "the corruption must actually apply"
        p.write_bytes(patched + tail)

        with pytest.raises(ValueError, match=r"broken\.vtp: malformed VTK XML header"):
            import_mesh(p)

    def test_an_entity_bomb_is_refused_as_a_malformed_header(
        self, tmp_path: Path
    ) -> None:
        """What the module's bandit waiver actually rests on, on the appended arm.

        `defusedxml` is not a dependency, and the waiver's argument is that libexpat's
        own input-amplification cap contains a billion-laughs expansion and that
        `_parse_header_document` then reports it as a malformed header. The second half
        of that is not free: the cap fires DURING `feed`, which stores the error instead
        of raising it, so a reader that stopped at the first `start` event reported this
        file as a decode failure somewhere in the data — a true-but-useless diagnosis
        that also stops attesting the waiver.
        """
        p = tmp_path / "bomb.vtp"
        write_vtp(p, GT, mode="appended-raw")
        raw = p.read_bytes()
        levels = ['<!ENTITY a0 "aaaaaaaaaa">']
        levels += [
            f'<!ENTITY a{i} "{"".join(f"&a{i - 1};" for _ in range(10))}">'
            for i in range(1, 7)
        ]
        dtd = ("<!DOCTYPE VTKFile [" + "".join(levels) + "]>\n").encode("ascii")
        root = raw.index(b"<VTKFile")
        p.write_bytes(
            raw[:root] + dtd + raw[root:].replace(b'Name="colors"', b'Name="&a6;"', 1)
        )
        with pytest.raises(ValueError, match=r"bomb\.vtp: malformed VTK XML header"):
            import_mesh(p)

    @pytest.mark.parametrize("encoding", ["x-mac-roman", "utf-42"])
    def test_an_unknown_declared_encoding_is_named(
        self, encoding: str, tmp_path: Path
    ) -> None:
        """An encoding expat does not know natively goes to Python's codec registry.

        Which raises `LookupError`, not `ParseError` — so on a file that is otherwise
        perfectly well-formed the failure walks straight past a `ParseError`-only handler
        and out of `import_mesh`, whose contract is `ValueError`, and out of the CLI's
        error funnel with it.
        """
        p = tmp_path / "badenc.vtp"
        write_vtp(p, GT, mode="ascii")
        raw = p.read_bytes()
        patched = raw.replace(
            b'<?xml version="1.0"?>',
            f'<?xml version="1.0" encoding="{encoding}"?>'.encode("ascii"),
            1,
        )
        assert patched != raw
        p.write_bytes(patched)
        with pytest.raises(ValueError, match=r"badenc\.vtp: malformed VTK XML header"):
            import_mesh(p)

    def test_a_declared_encoding_the_registry_knows_is_still_read(
        self, tmp_path: Path
    ) -> None:
        """The other half of the previous test: `cp1252` is a real codec, not an error.

        `LookupError` is caught, not encoding declarations in general — refusing every
        declared encoding would trade an escaped exception for a refused valid file.
        """
        p = tmp_path / "cp1252.vtp"
        write_vtp(p, GT, mode="ascii")
        raw = p.read_bytes()
        patched = raw.replace(
            b'<?xml version="1.0"?>', b'<?xml version="1.0" encoding="cp1252"?>', 1
        )
        assert patched != raw
        p.write_bytes(patched)
        assert _sorted_face_set(import_mesh(p)) == EXPECTED_FACES

    def test_a_prefixed_appended_data_tag_is_still_split_off(
        self, tmp_path: Path
    ) -> None:
        """`<vtk:AppendedData>` — the appended counterpart of a prefixed root.

        The sniffer accepts `<vtk:VTKFile>` and the reader matches every tag by local
        name, so this document is one it can decode. Searching for the LITERAL
        `<AppendedData` never cuts it: the whole file — binary payload included — reaches
        the XML parser, and what comes back is "no `<AppendedData>` section" about a file
        that plainly has one.
        """
        p = tmp_path / "pfxapp.vtp"
        write_vtp(
            p,
            GT,
            mode="appended-raw",
            root_decls=f'xmlns:vtk="{_VTK_NS}"',
            root_prefix="vtk:",
        )
        raw = p.read_bytes()
        patched = raw.replace(b"<AppendedData", b"<vtk:AppendedData", 1)
        close = patched.rindex(b"</AppendedData>")
        patched = (
            patched[:close]
            + b"</vtk:AppendedData>"
            + patched[close + len(b"</AppendedData>") :]
        )
        assert b"<vtk:AppendedData" in patched
        p.write_bytes(patched)
        assert detect_mesh_format(p) == "vtp"
        mesh = import_mesh(p)
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        assert mesh.normals is not None and mesh.colors is not None

    @pytest.mark.parametrize("attr", ["byte_order", "header_type"])
    def test_an_omitted_root_attribute_takes_the_documented_default(
        self, attr: str, tmp_path: Path
    ) -> None:
        """`byte_order` defaults to LittleEndian and `header_type` to UInt32.

        Every fixture emits both, so the two `or` defaults were never exercised — and
        either one flipped decodes a valid little-endian UInt32 file into garbage
        positions or a nonsense byte count, silently for the first and loudly for the
        second. The fixture is inline base64 so both actually matter: an ascii payload
        has neither a byte order nor a block header.
        """
        p = tmp_path / "defaults.vtp"
        write_vtp(p, GT, mode="inline-base64")
        text, hits = re.subn(
            rf' {attr}="[^"]*"', "", p.read_text(encoding="ascii"), count=1
        )
        assert hits == 1
        p.write_text(text, encoding="ascii")
        mesh = import_mesh(p)
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        assert mesh.colors is not None and mesh.normals is not None

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {
                    "xmlns": _VTK_NS,
                    "root_decls": f'xmlns:vtk="{_VTK_NS}"',
                    "root_prefix": "vtk:",
                },
                id="default-then-prefixed-root",
            ),
            pytest.param(
                {"root_decls": f'xmlns:vtk="{_VTK_NS}" xmlns="{_VTK_NS}"'},
                id="prefixed-then-default-root",
            ),
            pytest.param(
                {
                    "root_decls": f'xmlns:a="{_VTK_NS}"',
                    "root_prefix": "a:",
                    "piece_decls": f'xmlns:b="{_VTK_NS}"',
                    "piece_prefix": "b:",
                },
                id="two-prefixes-root-and-piece",
            ),
            pytest.param(
                {
                    "xmlns": _VTK_NS,
                    "piece_decls": f'xmlns:vtk="{_VTK_NS}"',
                    "piece_prefix": "vtk:",
                },
                id="default-root-prefixed-piece",
            ),
        ],
    )
    def test_two_prefixes_for_one_namespace_uri_still_read(
        self, kwargs: dict[str, str], tmp_path: Path
    ) -> None:
        """A URI may legally have more than one in-scope prefix, in any order.

        The reader takes the root straight off the pull parser's first `start` event, so
        the source's own tag SPELLINGS never have to be reconstructed — which is the
        whole point: they are not recoverable from `element.tag` (Clark notation drops
        the prefix) and not recoverable from a uri → prefix map either, because that map
        inverts a relation which is not one-to-one.

        Historical note, since it is why the cases are spelled this way. An earlier
        implementation SYNTHESISED the missing closing tags from a uri → prefix map and
        re-parsed; the first two cases were refused outright as "mismatched tag", because
        the root's own prefix was the one `</…VTKFile>` had to reproduce and a first-wins
        map picks wrong in one order or the other. The last two only escaped because a
        real writer closes `</Piece></PolyData>` before `<AppendedData>`, so those tags
        never needed synthesising — an accident of where VTK puts the appended section.
        Nothing is synthesised now, and all four are here to keep it that way.
        """
        p = tmp_path / "ns.vtp"
        write_vtp(p, GT, mode="appended-raw", **kwargs)
        assert detect_mesh_format(p) == "vtp"
        mesh = import_mesh(p)
        assert _sorted_face_set(mesh) == EXPECTED_FACES
        assert mesh.normals is not None and mesh.colors is not None

    # --------------------------------------------------- every error names the file

    @pytest.mark.parametrize("value", ["abc", ""])
    def test_a_non_integer_appended_offset_is_named(
        self, value: str, tmp_path: Path
    ) -> None:
        p = tmp_path / "badoffset.vtp"
        write_vtp(p, GT, mode="appended-raw")
        raw = p.read_bytes()
        patched = raw.replace(b'offset="0"', f'offset="{value}"'.encode("ascii"), 1)
        assert patched != raw
        p.write_bytes(patched)
        with pytest.raises(ValueError, match=r"badoffset\.vtp"):
            import_mesh(p)

    def test_a_negative_appended_base64_offset_is_refused(self, tmp_path: Path) -> None:
        p = tmp_path / "negative-offset.vtp"
        write_vtp(p, GT, mode="appended-base64")
        raw = p.read_bytes()
        patched = raw.replace(b'offset="0"', b'offset="-336"', 1)
        assert patched != raw
        p.write_bytes(patched)
        with pytest.raises(ValueError, match=r"negative-offset\.vtp.*offset.*-336"):
            import_mesh(p)

    def test_a_non_integer_component_count_is_named(self, tmp_path: Path) -> None:
        p = tmp_path / "badncomp.vtp"
        write_vtp(p, GT, mode="ascii")
        p.write_text(
            p.read_text(encoding="ascii").replace(
                'NumberOfComponents="3"', 'NumberOfComponents="x"', 1
            ),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"badncomp\.vtp"):
            import_mesh(p)

    @pytest.mark.parametrize("declared", ["0", "-3"])
    def test_a_component_count_below_one_is_refused(
        self, declared: str, tmp_path: Path
    ) -> None:
        """Only `ncomp > 1` reshapes, so 0 or -3 falls through as a 1-D array — and the
        `<Points>` 1-D rescue then imports it as if it had said 3."""
        p = tmp_path / "zerocomp.vtp"
        write_vtp(p, GT, mode="ascii")
        p.write_text(
            p.read_text(encoding="ascii").replace(
                'Name="Points" NumberOfComponents="3"',
                f'Name="Points" NumberOfComponents="{declared}"',
                1,
            ),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match="NumberOfComponents"):
            import_mesh(p)

    def test_a_non_integer_point_count_is_named(self, tmp_path: Path) -> None:
        p = tmp_path / "badcount.vtp"
        write_vtp(p, GT, mode="ascii")
        p.write_text(
            p.read_text(encoding="ascii").replace(
                'NumberOfPoints="4"', 'NumberOfPoints="abc"'
            ),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"badcount\.vtp"):
            import_mesh(p)

    @pytest.mark.parametrize(
        "name,body", [("Points", "nope 1 2"), ("connectivity", "1.5 2 3")]
    )
    def test_an_unparseable_ascii_token_is_named(
        self, name: str, body: str, tmp_path: Path
    ) -> None:
        """A word in a Float32 array, and a float in an Int64 one.

        Both raise a bare `ValueError` out of numpy (`invalid literal for int()`), which
        names neither the file nor which array it came from.
        """
        p = tmp_path / "tokens.vtp"
        write_vtp(p, GT, mode="ascii")
        p.write_text(
            _replace_array_text(p.read_text(encoding="ascii"), name, body),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"tokens\.vtp"):
            import_mesh(p)

    @pytest.mark.parametrize(
        "name,body",
        [
            ("colors", "300 0 0 " + "0 " * 9),
            ("colors", "-1 0 0 " + "0 " * 9),
            ("connectivity", "99999999999999999999999 " + "0 " * 11),
        ],
    )
    def test_an_out_of_range_ascii_integer_is_a_clean_error(
        self, name: str, body: str, tmp_path: Path
    ) -> None:
        """A ValueError, not the OverflowError a bare cast raises.

        Numpy's string→int conversion raises `ValueError` for a malformed token but
        `OverflowError` for one outside the target dtype's range — and `OverflowError` is
        an `ArithmeticError`, sharing no base with `ValueError`. `300` or `-1` in a
        `UInt8` colour array (a writer that forgot to clip) and an oversized `Int64`
        connectivity index are ordinary bad files, but an uncaught `OverflowError`
        escapes both `import_mesh`'s documented contract and the CLI's
        `except (ValueError, …)` funnel, so the user sees a raw traceback.

        The same rule the OBJ reader is held to a few hundred lines up.
        """
        p = tmp_path / "overflow.vtp"
        write_vtp(p, GT, mode="ascii")
        p.write_text(
            _replace_array_text(p.read_text(encoding="ascii"), name, body),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"overflow\.vtp"):
            import_mesh(p)

    def test_points_with_no_component_count_are_rescued_as_triples(
        self, tmp_path: Path
    ) -> None:
        """`NumberOfComponents` is optional, and a flat `<Points>` array is still triples.

        A PolyData point is always 3D, so a 1-D positions array of length 3N is
        unambiguous — and reading it as N*3 separate scalars would fail the
        `points.shape[1]` check on a perfectly good file.
        """
        p = tmp_path / "flatpoints.vtp"
        write_vtp(p, GT, mode="ascii")
        text, hits = re.subn(
            r'(Name="Points") NumberOfComponents="3"',
            r"\g<1>",
            p.read_text(encoding="ascii"),
            count=1,
        )
        assert hits == 1
        p.write_text(text, encoding="ascii")
        mesh = import_mesh(p)
        assert mesh.n_vertices == 4
        assert _sorted_face_set(mesh) == EXPECTED_FACES

    def test_flat_points_that_are_not_a_multiple_of_three_are_named(
        self, tmp_path: Path
    ) -> None:
        """The other half of the rescue: 11 coordinates are not points at all.

        Without the `% 3` refusal `reshape(-1, 3)` raises numpy's own "cannot reshape
        array of size 11", which names neither the file nor `<Points>`.
        """
        p = tmp_path / "ragged-points.vtp"
        write_vtp(p, GT, mode="ascii")
        text, hits = re.subn(
            r'(Name="Points") NumberOfComponents="3"',
            r"\g<1>",
            p.read_text(encoding="ascii"),
            count=1,
        )
        assert hits == 1
        p.write_text(
            _replace_array_text(text, "Points", "0 0 0 1 0 0 0 1 0 0 0"),
            encoding="ascii",
        )
        with pytest.raises(ValueError, match=r"ragged-points\.vtp.*multiple of 3"):
            import_mesh(p)

    def test_a_ragged_raw_payload_is_named(self, tmp_path: Path) -> None:
        """A raw block header declaring a byte count the item size does not divide.

        `np.frombuffer`'s own "buffer size must be a multiple of element size" names
        neither the file nor the array, and this is precisely what a header read at the
        wrong width looks like.
        """
        p = tmp_path / "ragged.vtp"
        write_vtp(p, GT, mode="appended-raw")
        # Five bytes is not a whole number of Float32s. Points, not the UInt8 colour
        # block, because a one-byte item size divides every length there is.
        p.write_bytes(_patch_appended_header_word(p.read_bytes(), "Points", 5))
        with pytest.raises(ValueError, match=r"ragged\.vtp.*item size"):
            import_mesh(p)

    def test_an_absurd_block_count_is_named_not_a_memory_error(
        self, tmp_path: Path
    ) -> None:
        """`_guard_nblocks`, the guard the module docstring argues for at length.

        A block header read at the wrong width or offset yields a count in the billions,
        which unbounded sizes a `_words` read and, past it, an allocation.
        """
        p = tmp_path / "blocks.vtp"
        write_vtp(p, GT, mode="appended-raw", compressed=True)
        p.write_bytes(_patch_appended_header_word(p.read_bytes(), "Points", 0xFFFFFFFF))
        with pytest.raises(ValueError, match="compressed blocks"):
            import_mesh(p)

    def test_a_vertex_index_past_the_pieces_own_point_count_is_named(
        self, tmp_path: Path
    ) -> None:
        """Bounded against the piece's OWN count, before the multi-piece rebase.

        After `+ base` an index that overran its piece lands inside the assembled vertex
        array, so the surface stitches to a neighbouring piece's geometry rather than
        raising.
        """
        p = tmp_path / "badindex.vtp"
        write_vtp_polys(p, GT.vertices, [0, 1, 99], [3])
        with pytest.raises(ValueError, match="references vertex 99"):
            import_mesh(p)

    def test_offsets_that_decrease_are_named(self, tmp_path: Path) -> None:
        p = tmp_path / "backwards.vtp"
        write_vtp_polys(p, GT.vertices, [0, 1, 2, 0, 2, 3], [6, 3])
        with pytest.raises(ValueError, match="non-decreasing"):
            import_mesh(p)

    def test_offsets_past_the_connectivity_array_are_named(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "past.vtp"
        write_vtp_polys(p, GT.vertices, [0, 1, 2], [9])
        with pytest.raises(ValueError, match="connectivity array holds only 3"):
            import_mesh(p)

    def test_the_explicit_format_path_refuses_a_non_polydata_file_too(
        self, tmp_path: Path
    ) -> None:
        """`read_vtp`'s own type check is unreachable through sniffing.

        `import_mesh(path)` goes through `detect_mesh_format`, which refuses a
        non-PolyData `<VTKFile>` first, so the reader's own refusal only ever fires on
        the explicit `format="vtp"` path — which is a caller-reachable way past the
        sniffer, not dead code.
        """
        p = tmp_path / "volume.vtp"
        write_vtp(p, GT, root_type="UnstructuredGrid")
        with pytest.raises(ValueError, match="not 'PolyData'"):
            import_mesh(p, format="vtp")


class TestErrors:
    def test_a_gsplat_ply_points_at_the_other_importer(self, tmp_path: Path) -> None:
        # `.ply` is shared between the two importers. Recognising the wrong dialect and
        # naming the right command beats a parse error thirty lines deeper.
        p = tmp_path / "splats.ply"
        write_gsplat_ply(p)
        with pytest.raises(ValueError, match="luxar gsplat import"):
            import_mesh(p)

    def test_unknown_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "thing.xyz"
        p.write_bytes(b"nope")
        with pytest.raises(ValueError, match="unrecognized extension"):
            import_mesh(p)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            import_mesh(tmp_path / "absent.ply")

    def test_junk_bytes_fail_cleanly(self, tmp_path: Path) -> None:
        p = tmp_path / "junk.ply"
        p.write_bytes(b"\x00" * 400)
        with pytest.raises(ValueError):
            import_mesh(p)

    def test_explicit_format_overrides_sniffing(self, tmp_path: Path) -> None:
        p = tmp_path / "mislabelled.dat"
        write_stl_binary(p, GT)
        with pytest.raises(ValueError, match="unrecognized extension"):
            import_mesh(p)
        assert import_mesh(p, format="stl").n_faces == 4


class TestTriangleMeshInvariants:
    def test_dimension_names_must_match_vertex_columns(self) -> None:
        with pytest.raises(ValueError, match="one dimension name per column"):
            TriangleMesh(
                vertices=np.zeros((3, 4), dtype=np.float32),
                faces=np.array([[0, 1, 2]], dtype=np.uint32),
            )

    def test_first_three_dimensions_must_be_spatial(self) -> None:
        with pytest.raises(ValueError, match="must be x, y, z"):
            TriangleMesh(
                vertices=np.zeros((3, 3), dtype=np.float32),
                faces=np.array([[0, 1, 2]], dtype=np.uint32),
                dimension_names=("a", "b", "c"),
            )

    def test_out_of_range_face_index_is_named(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            TriangleMesh(
                vertices=np.zeros((3, 3), dtype=np.float32),
                faces=np.array([[0, 1, 9]], dtype=np.uint32),
            )

    def test_attribute_length_must_match_vertex_count(self) -> None:
        with pytest.raises(ValueError, match="normals"):
            TriangleMesh(
                vertices=np.zeros((3, 3), dtype=np.float32),
                faces=np.array([[0, 1, 2]], dtype=np.uint32),
                normals=np.zeros((2, 3), dtype=np.float32),
            )


class TestMeshDirectoryImport:
    def test_stacks_numeric_time_and_channel_coordinates(self, tmp_path: Path) -> None:
        write_vtp(tmp_path / "a_Ch1-registered-T0010.vtp", GT, compressed=True)
        write_vtp(tmp_path / "z_Ch0-registered-T0002.vtp", GT, compressed=True)
        progress: list[tuple[int, int, str]] = []

        mesh = import_mesh_directory(
            tmp_path,
            progress=lambda index, total, path: progress.append(
                (index, total, path.name)
            ),
        )

        assert mesh.dimension_names == ("x", "y", "z", "t", "c")
        assert mesh.vertices.shape == (8, 5)
        assert np.array_equal(mesh.vertices[:4, 3:], [[2, 0]] * 4)
        assert np.array_equal(mesh.vertices[4:, 3:], [[10, 1]] * 4)
        assert np.array_equal(mesh.faces[4:], mesh.faces[:4] + 4)
        assert mesh.normals is not None
        assert mesh.colors is not None
        assert np.array_equal(mesh.normals[4:], mesh.normals[:4])
        assert np.array_equal(mesh.colors[4:], mesh.colors[:4])
        assert progress == [
            (1, 2, "z_Ch0-registered-T0002.vtp"),
            (2, 2, "a_Ch1-registered-T0010.vtp"),
        ]

    def test_missing_timepoints_remain_coordinate_gaps(self, tmp_path: Path) -> None:
        write_vtp(tmp_path / "z-T0001.vtp", GT)
        write_vtp(tmp_path / "a-T0003.vtp", GT)

        mesh = import_mesh_directory(tmp_path)

        assert mesh.dimension_names == ("x", "y", "z", "t")
        assert np.array_equal(np.unique(mesh.vertices[:, 3]), [1, 3])

    def test_a_named_group_regex_overrides_filename_index_parsing(
        self, tmp_path: Path
    ) -> None:
        write_vtp(tmp_path / "surface.0010-channel3.vtp", GT)
        write_vtp(tmp_path / "surface.0002-channel1.vtp", GT)

        mesh = import_mesh_directory(
            tmp_path,
            index_regex=r"surface\.(?P<t>\d+)-channel(?P<c>\d+)",
        )

        assert mesh.dimension_names == ("x", "y", "z", "t", "c")
        assert np.array_equal(mesh.vertices[:4, 3:], [[2, 1]] * 4)
        assert np.array_equal(mesh.vertices[4:, 3:], [[10, 3]] * 4)

    @pytest.mark.parametrize(
        ("filename", "index_regex", "message"),
        [
            ("frame_first.vtp", r"frame_(?P<c>\d+)", "named 't' capture"),
            ("frame_first.vtp", r"frame_(?P<t>\d+", "Invalid index regex"),
            (
                "frame_first.vtp",
                r"frame_(?P<t>[^.]+)",
                "time capture 'first' is not an integer",
            ),
            (
                "frame_first.vtp",
                r"frame_(?:(?P<t>\d+)|first)",
                "did not capture a time value",
            ),
            ("frame_42.vtp", r"(?P<t>\d)", "anchor it with"),
            ("surface.vtp", r"frame_(?P<t>\d+)", "narrow --pattern"),
        ],
    )
    def test_invalid_index_regexes_fail_before_mesh_reading(
        self, tmp_path: Path, filename: str, index_regex: str, message: str
    ) -> None:
        (tmp_path / filename).write_bytes(b"not a mesh")

        with pytest.raises(ValueError, match=message):
            import_mesh_directory(tmp_path, index_regex=index_regex)

    @pytest.mark.parametrize(
        ("index_regex", "message"),
        [
            (r"frame_(?P<c>\d+)", "named 't' capture"),
            (r"frame_(?P<t>\d+", "Invalid index regex"),
        ],
    )
    def test_index_regex_is_validated_before_file_discovery(
        self, tmp_path: Path, index_regex: str, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            import_mesh_directory(tmp_path, index_regex=index_regex)

    def test_mixed_or_duplicate_coordinates_are_refused(self, tmp_path: Path) -> None:
        write_ply_binary(tmp_path / "a_Ch0-T0001.ply", GT)
        write_ply_binary(tmp_path / "b-T0002.ply", GT)
        with pytest.raises(ValueError, match="only some filenames"):
            import_mesh_directory(tmp_path, pattern="*.ply")

        (tmp_path / "b-T0002.ply").unlink()
        write_ply_binary(tmp_path / "b_Ch0-T0001.ply", GT)
        with pytest.raises(ValueError, match="duplicate time/channel"):
            import_mesh_directory(tmp_path, pattern="*.ply")

    def test_custom_indices_keep_coordinate_validation(self, tmp_path: Path) -> None:
        regex = r"frame_(?P<t>\d+)(?:-channel(?P<c>\d+))?"
        write_ply_binary(tmp_path / "frame_1-channel0.ply", GT)
        write_ply_binary(tmp_path / "frame_2.ply", GT)
        with pytest.raises(ValueError, match="only some filenames"):
            import_mesh_directory(tmp_path, pattern="*.ply", index_regex=regex)

        for path in tmp_path.iterdir():
            path.unlink()
        write_ply_binary(tmp_path / "frame_1-first.ply", GT)
        write_ply_binary(tmp_path / "frame_1-second.ply", GT)
        with pytest.raises(ValueError, match="duplicate time/channel"):
            import_mesh_directory(
                tmp_path,
                pattern="*.ply",
                index_regex=r"frame_(?P<t>\d+)-.+",
            )

        for path in tmp_path.iterdir():
            path.unlink()
        write_ply_binary(tmp_path / "frame_16777217.ply", GT)
        with pytest.raises(ValueError, match="too large to represent exactly"):
            import_mesh_directory(
                tmp_path,
                pattern="*.ply",
                index_regex=r"frame_(?P<t>\d+)",
            )

        for path in tmp_path.iterdir():
            path.unlink()
        write_ply_binary(tmp_path / "frame_-16777217.ply", GT)
        with pytest.raises(ValueError, match="too large to represent exactly"):
            import_mesh_directory(
                tmp_path,
                pattern="*.ply",
                index_regex=r"frame_(?P<t>-?\d+)",
            )

        for path in tmp_path.iterdir():
            path.unlink()
        write_ply_binary(tmp_path / "frame_-16777216.ply", GT)
        mesh = import_mesh_directory(
            tmp_path,
            pattern="*.ply",
            index_regex=r"frame_(?P<t>-?\d+)",
        )
        assert np.array_equal(mesh.vertices[:, 3], [-16777216] * GT.vertices.shape[0])

    def test_every_matched_file_requires_a_time_index(self, tmp_path: Path) -> None:
        write_ply_binary(tmp_path / "surface.ply", GT)
        with pytest.raises(ValueError, match=r"no T<number> time index"):
            import_mesh_directory(tmp_path, pattern="*.ply")

    def test_lowercase_or_multiple_time_indices_are_refused(
        self, tmp_path: Path
    ) -> None:
        write_ply_binary(tmp_path / "surface-t0001.ply", GT)
        with pytest.raises(ValueError, match=r"no T<number> time index"):
            import_mesh_directory(tmp_path, pattern="*.ply")

        (tmp_path / "surface-t0001.ply").unlink()
        write_ply_binary(tmp_path / "run-T0001-T0002.ply", GT)
        with pytest.raises(ValueError, match="more than one time index"):
            import_mesh_directory(tmp_path, pattern="*.ply")

    @pytest.mark.parametrize(
        ("missing_name", "arrays", "attrs"),
        [
            ("colors", [(GT.normals, "Float32", "Normals", 3)], 'Normals="Normals"'),
            ("normals", [(GT.colors, "UInt8", "colors", 3)], 'Scalars="colors"'),
        ],
    )
    def test_attributes_must_be_present_in_every_file(
        self,
        tmp_path: Path,
        missing_name: str,
        arrays: list[tuple[np.ndarray, str, str, int]],
        attrs: str,
    ) -> None:
        write_vtp(tmp_path / "a-T0001.vtp", GT)
        write_vtp_point_data(tmp_path / "b-T0002.vtp", GT, arrays, pdata_attrs=attrs)

        with pytest.raises(
            ValueError, match=rf"{missing_name} are present in only some"
        ):
            import_mesh_directory(tmp_path)

    def test_attribute_component_counts_must_match(self, tmp_path: Path) -> None:
        rgba = np.column_stack(
            (GT.colors, np.full(GT.colors.shape[0], 255, dtype=np.uint8))
        )
        write_vtp(tmp_path / "a-T0001.vtp", GT)
        write_vtp_point_data(
            tmp_path / "b-T0002.vtp",
            GT,
            [
                (GT.normals, "Float32", "Normals", 3),
                (rgba, "UInt8", "colors", 4),
            ],
            pdata_attrs='Normals="Normals" Scalars="colors"',
        )

        with pytest.raises(
            ValueError, match="colors have inconsistent component counts"
        ):
            import_mesh_directory(tmp_path)

    def test_indices_must_remain_exact_float32_coordinates(
        self, tmp_path: Path
    ) -> None:
        write_ply_binary(tmp_path / "surface-T16777217.ply", GT)
        with pytest.raises(ValueError, match="too large to represent exactly"):
            import_mesh_directory(tmp_path, pattern="*.ply")


class TestWelding:
    """The weld key is position PLUS every per-vertex attribute, not position alone.

    The two ``write_ply_crease`` arms are each other's sensitivity control, so neither
    can pass vacuously: keying on position alone passes ``hard=False`` and fails
    ``hard=True``; refusing to merge anything does the reverse.
    """

    def test_pruning_an_already_compact_mesh_reuses_its_arrays(self) -> None:
        vertices = np.ascontiguousarray(GT.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(GT.faces, dtype=np.uint32)
        normals = np.ascontiguousarray(GT.normals, dtype=np.float32)

        compact_vertices, compact_faces, extras = prune_unreferenced_vertices(
            vertices, faces, extras={"normals": normals, "colors": None}
        )

        assert compact_vertices is vertices
        assert compact_faces is faces
        assert extras["normals"] is normals
        assert extras["colors"] is None

    def test_a_crease_survives_welding(self, tmp_path: Path) -> None:
        path = tmp_path / "crease.ply"
        write_ply_crease(path, hard=True)
        mesh = import_mesh(path)

        # Six distinct (position, normal) pairs, so nothing merges — the two copies of
        # (1,0,0) and of (0,1,0) each keep their own face's normal.
        assert mesh.n_vertices == 6
        assert mesh.n_faces == 2
        assert mesh.normals is not None
        # And each triangle is still shaded by ONE normal, which is what a crease means.
        for tri in mesh.faces:
            face_normals = mesh.normals[np.asarray(tri)]
            assert np.allclose(face_normals, face_normals[0]), (
                "a welded-away crease leaves one triangle carrying the other's normal"
            )
        # The two faces must disagree, or the fixture is not a crease at all.
        n0 = mesh.normals[mesh.faces[0][0]]
        n1 = mesh.normals[mesh.faces[1][0]]
        assert not np.allclose(n0, n1)

    def test_a_smooth_join_still_welds(self, tmp_path: Path) -> None:
        path = tmp_path / "smooth.ply"
        write_ply_crease(path, hard=False)
        mesh = import_mesh(path)

        # Same positions, same faces — but every duplicate agrees on its normal, so the
        # duplicates are redundant and must collapse to the 4 corners of the square.
        assert mesh.n_vertices == 4
        assert mesh.n_faces == 2

    def test_colors_are_part_of_the_key_too(self) -> None:
        # Normals are the famous case; a vertex-colour seam (a segmentation boundary,
        # common in the isosurface data this importer targets) is the same bug.
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], np.float32)
        faces = np.array([[0, 1, 2], [0, 1, 3]], np.uint32)
        colors = np.array(
            [[255, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8
        )
        welded, remapped, extras = weld_vertices(
            vertices, faces, extras={"colors": colors, "normals": None}
        )
        assert welded.shape[0] == 4, "differently-coloured duplicates must not merge"
        assert extras["normals"] is None
        assert remapped.shape == (2, 3)

        # Control: with the colours in agreement the same geometry welds to 3.
        agreed = colors.copy()
        agreed[3] = agreed[2]
        welded2, _, _ = weld_vertices(vertices, faces, extras={"colors": agreed})
        assert welded2.shape[0] == 3

    def test_position_only_input_is_unaffected(self) -> None:
        # The STL path: no attributes at all, so the key degenerates to position and a
        # soup still collapses. This is why the fix needed no per-format branch.
        soup = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 0, 1]],
            np.float32,
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], np.uint32)
        welded, _, _ = weld_vertices(
            soup, faces, extras={"normals": None, "colors": None}
        )
        assert welded.shape[0] == 4

    def test_a_mismatched_attribute_length_is_named(self) -> None:
        vertices = np.zeros((3, 3), np.float32)
        faces = np.array([[0, 1, 2]], np.uint32)
        with pytest.raises(ValueError, match="rows but there are 3 vertices"):
            weld_vertices(
                vertices, faces, extras={"colors": np.zeros((2, 3), np.uint8)}
            )
