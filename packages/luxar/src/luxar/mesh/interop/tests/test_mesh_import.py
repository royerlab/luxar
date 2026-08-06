"""Reader parity, sniffing, and the per-format traps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from .._stl import is_binary_stl
from .._weld import weld_vertices
from ..mesh_import import MESH_FORMATS, TriangleMesh, detect_mesh_format, import_mesh
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
    write_ply_quads,
    write_ply_truncated_ascii,
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
        def signed_volume(m) -> float:
            t = m.vertices[m.faces]
            return float(
                np.sum(np.einsum("ij,ij->i", np.cross(t[:, 0], t[:, 1]), t[:, 2]))
            )

        va, vb = signed_volume(a), signed_volume(b)
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


class TestWelding:
    """The weld key is position PLUS every per-vertex attribute, not position alone.

    The two ``write_ply_crease`` arms are each other's sensitivity control, so neither
    can pass vacuously: keying on position alone passes ``hard=False`` and fails
    ``hard=True``; refusing to merge anything does the reverse.
    """

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
