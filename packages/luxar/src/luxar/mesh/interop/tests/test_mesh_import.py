"""Reader parity, sniffing, and the per-format traps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .._stl import is_binary_stl
from ..mesh_import import MESH_FORMATS, TriangleMesh, detect_mesh_format, import_mesh
from ._synthetic import (
    SUFFIXES,
    WRITERS,
    make_ground_truth,
    write_glb,
    write_glb_interleaved,
    write_gltf_draco,
    write_gsplat_ply,
    write_obj_negative_indices,
    write_obj_quad,
    write_ply_ascii,
    write_ply_binary,
    write_ply_quads,
    write_stl_ascii,
    write_stl_binary,
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


class TestReaderParity:
    @pytest.mark.parametrize("fmt", MESH_FORMATS)
    def test_every_dialect_reproduces_the_tetrahedron(
        self, fmt: str, tmp_path: Path
    ) -> None:
        path = tmp_path / f"tetra{SUFFIXES[fmt]}"
        WRITERS[fmt](path, GT)
        mesh = import_mesh(path)

        assert mesh.source_format == fmt
        # Welding is load-bearing: STL and index-free glTF arrive as 12-corner soups
        # and must come back as the 4 shared vertices the tetrahedron actually has.
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

    def test_draco_is_refused_by_name(self, tmp_path: Path) -> None:
        p = tmp_path / "draco.gltf"
        write_gltf_draco(p)
        with pytest.raises(ValueError, match="Draco"):
            import_mesh(p)


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
