"""``add_mesh(substitutive_lod=…)`` — the ladder, not the decimator.

``luxar/mesh/tests/test_decimate.py`` covers whether a single coarser surface is
correct. This covers whether the LADDER is one: that the levels come out in the
right order, that every one of them is independently renderable, and that the
degenerate shapes produce something sane rather than a store the viewer cannot
load. Those are different failure modes — a perfect decimator wired up backwards
produces four valid meshes in an invalid ladder, which is exactly the bug the
first end-to-end write of this feature actually had.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler


def octasphere(subdivisions: int) -> Tuple[np.ndarray, np.ndarray]:
    """A closed, welded, boundary-free sphere — the honest decimation input.

    Welded so vertices are genuinely shared (a soup would collapse trivially), and
    closed so a level that tears the surface open is detectable.
    """
    verts: List[np.ndarray] = [
        np.array(v, dtype=np.float64)
        for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    for _ in range(subdivisions):
        midpoint: Dict[Tuple[int, int], int] = {}

        def split(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in midpoint:
                p = verts[a] + verts[b]
                verts.append(p / np.linalg.norm(p))
                midpoint[key] = len(verts) - 1
            return midpoint[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = split(a, b), split(b, c), split(c, a)
            refined += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = refined
    return (
        np.asarray(verts, dtype=np.float32),
        np.asarray(faces, dtype=np.uint32),
    )


def read_nodes(store: Path) -> Dict[str, Dict[str, Any]]:
    """Every node's attrs, keyed by path — read back from what was WRITTEN.

    Deliberately reading the store rather than inspecting the returned objects:
    the ladder's correctness is a property of the bytes the viewer will load, and
    an in-memory assertion would pass against a node tree that never reached disk.
    """
    meta = json.loads((store / ".zmetadata").read_text())["metadata"]
    out: Dict[str, Dict[str, Any]] = {}
    for key, value in meta.items():
        if key.endswith(".zattrs"):
            out[key[: -len("/.zattrs")] or "/"] = value
    return out


def write_ladder(tmp_path: Path, verts, faces, **kwargs) -> Dict[str, Dict[str, Any]]:
    store = tmp_path / "ladder.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("surf", verts, faces, **kwargs)
    return read_nodes(store)


def ladder_children(nodes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The mesh children of the `surf` group, in written (coarsest→finest) order."""
    kids = [
        (path, attrs)
        for path, attrs in nodes.items()
        if path.startswith("surf/child_") and attrs.get("type") == "mesh"
    ]
    kids.sort(key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
    return [attrs for _, attrs in kids]


class TestLadderShape:
    def test_writes_a_kind_lod_group_of_progressively_coarser_meshes(self, tmp_path):
        verts, faces = octasphere(4)  # 1026 vertices, 2048 faces
        nodes = write_ladder(tmp_path, verts, faces, substitutive_lod=True)

        assert nodes["surf"]["kind"] == "lod"
        assert nodes["surf"]["display_type"] == "mesh"

        children = ladder_children(nodes)
        assert len(children) >= 2, "a ladder needs at least one coarse level"

        counts = [c["n_vertices"] for c in children]
        # STRICTLY ascending coarsest→finest. The direction is the point: the
        # first working version of this compared each level against the coarser
        # one instead of the finer, which silently dropped every level after the
        # first and still produced a valid-looking two-level ladder.
        assert counts == sorted(counts), f"levels out of order: {counts}"
        assert len(set(counts)) == len(counts), f"duplicate levels: {counts}"
        # The finest child is the original surface, untouched.
        assert counts[-1] == len(verts)
        assert children[-1]["n_faces"] == len(faces)

    def test_every_level_is_independently_renderable(self, tmp_path):
        # A kind=lod group shows exactly ONE child at a time, so a level with no
        # triangles is a frame with nothing in it, not a cheaper frame.
        verts, faces = octasphere(4)
        children = ladder_children(
            write_ladder(tmp_path, verts, faces, substitutive_lod=True)
        )
        for i, child in enumerate(children):
            assert child["n_faces"] > 0, f"level {i} has no triangles"
            assert child["n_vertices"] >= 3, f"level {i} cannot form a triangle"

    def test_coverage_fractions_are_strictly_ascending_and_bounded(self, tmp_path):
        verts, faces = octasphere(4)
        children = ladder_children(
            write_ladder(tmp_path, verts, faces, substitutive_lod=True)
        )
        fractions = [c["coverage_fraction"] for c in children]
        assert fractions == sorted(fractions)
        assert len(set(fractions)) == len(fractions)
        assert fractions[0] == pytest.approx(0.0)
        assert fractions[-1] == pytest.approx(1.0)

    def test_level_count_follows_levels_and_K(self, tmp_path):
        verts, faces = octasphere(4)
        two = ladder_children(
            write_ladder(tmp_path, verts, faces, substitutive_lod={"levels": 2, "K": 4})
        )
        assert len(two) == 3, "2 coarse levels + the original"

    def test_a_UNIFORM_color_reaches_every_level(self, tmp_path):
        # A uniform colour is not per-vertex data, so there is nothing to average
        # — it must be forwarded verbatim to every level. Discriminating on
        # `isinstance(colors, np.ndarray)` got this wrong twice: a uniform TUPLE
        # fell through to `None` and left the coarse levels colourless against a
        # coloured finest one (a visible colour pop on every LOD switch), and a
        # uniform NDARRAY of shape (3,) reached the decimator, where
        # `colors.shape[1]` raised a bare IndexError the adder does not catch.
        verts, faces = octasphere(3)
        children = ladder_children(
            write_ladder(
                tmp_path, verts, faces, colors=(200, 0, 0), substitutive_lod=True
            )
        )
        assert len(children) >= 2
        assert all(c["has_colors"] for c in children), (
            "every level keeps the uniform colour; a colourless coarse level "
            "renders default white and pops on the LOD switch"
        )

    def test_colors_survive_to_the_coarse_levels(self, tmp_path):
        # Averaged per cluster by the decimator; the point here is only that they
        # are CARRIED — a level that silently lost its colours would render black
        # against a coloured finest level, which reads as a flicker on LOD switch.
        verts, faces = octasphere(3)
        colors = np.zeros((len(verts), 3), dtype=np.uint8)
        colors[:, 0] = 200
        children = ladder_children(
            write_ladder(tmp_path, verts, faces, colors=colors, substitutive_lod=True)
        )
        assert all(c["has_colors"] for c in children)


class TestDegenerateLadders:
    def test_a_surface_too_coarse_to_reduce_falls_back_to_a_plain_leaf(self, tmp_path):
        # An octahedron has 6 vertices: every requested target is below the
        # decimator's 4-vertex floor or fails to reduce. Writing a one-level
        # "ladder" would be a kind=lod group with a single child, which is a
        # pointless indirection; writing duplicate levels would hand
        # `coverage_fractions` a repeated ratio and raise.
        verts, faces = octasphere(0)
        nodes = write_ladder(tmp_path, verts, faces, substitutive_lod=True)
        assert nodes["surf"].get("kind") != "lod"
        assert nodes["surf"]["type"] == "mesh"
        assert nodes["surf"]["n_vertices"] == len(verts)

    def test_explicit_coverage_fractions_of_the_wrong_length_says_why(self, tmp_path):
        # The count is not simply `levels + 1`: levels that cannot reduce the
        # surface are dropped, so the ladder can come out shorter than requested.
        # A bare arity error would send the caller looking for the wrong mistake.
        verts, faces = octasphere(4)
        with pytest.raises(ValueError) as excinfo:
            write_ladder(
                tmp_path,
                verts,
                faces,
                substitutive_lod={"levels": 3, "coverage_fractions": [0.5, 1.0]},
            )
        message = str(excinfo.value)
        assert "coverage_fractions" in message
        assert "dropped" in message


class TestVocabulary:
    """The refusals from `lod/mesh.py`, reached through the real adder."""

    def test_a_gaussian_mixture_method_is_refused_by_name(self, tmp_path):
        # `kmeans` is valid for Points and Lines because both LIFT to gsplats.
        # A mesh has no mixture to reduce, so accepting the word would accept
        # something the code cannot do.
        verts, faces = octasphere(3)
        with pytest.raises(ValueError) as excinfo:
            write_ladder(tmp_path, verts, faces, substitutive_lod={"method": "kmeans"})
        message = str(excinfo.value)
        assert "kmeans" in message
        assert "cluster" in message, "the message must name what IS accepted"

    @pytest.mark.parametrize(
        "key,value",
        [
            ("truncation_radius", 3.0),
            ("max_aspect", 2.0),
            ("device", "cpu"),
            ("seed", 7),
        ],
    )
    def test_each_lift_only_key_is_refused_with_its_own_reason(
        self, tmp_path, key, value
    ):
        # Every one of these is valid on Points/Lines, so a caller who passed it
        # made no typo — the generic unknown-key error would misdiagnose them.
        verts, faces = octasphere(3)
        with pytest.raises(ValueError) as excinfo:
            write_ladder(tmp_path, verts, faces, substitutive_lod={key: value})
        message = str(excinfo.value)
        assert key in message
        assert "does not apply to a mesh" in message

    def test_cluster_and_auto_both_build_a_ladder(self, tmp_path):
        verts, faces = octasphere(4)
        for method in ("auto", "cluster"):
            nodes = write_ladder(
                tmp_path / method, verts, faces, substitutive_lod={"method": method}
            )
            assert nodes["surf"]["kind"] == "lod", method


def octasphere_4d(subdivisions: int, w: float) -> Tuple[np.ndarray, np.ndarray]:
    """The sphere with a 4th column pinned at ``w`` — a categorical axis."""
    verts, faces = octasphere(subdivisions)
    padded = np.hstack([verts, np.full((len(verts), 1), w, dtype=np.float32)])
    return padded.astype(np.float32), faces


class TestCoarsenDims:
    """`coarsen_dims` must be RESOLVED against the scene, not silently dropped.

    The resolver accepts dimension NAMES and the `"display"` convention and its
    docstring says the adder resolves them. An earlier version only honoured an
    already-integer list and passed `None` for anything else, so
    `coarsen_dims=["x", "y", "z", "time"]` quietly became "coarsen the first three
    columns and treat time as a hard barrier" — the opposite of the request, with
    no warning anywhere.
    """

    def dims_4d(self):
        from luxar import Dimension, Dimensions

        return Dimensions(
            [
                Dimension(name=n, unit="um", range=(-2.0, 2.0), step=0.1, display=d)
                for n, d in (("x", True), ("y", True), ("z", True), ("t", False))
            ]
        )

    def write(self, tmp_path, verts, faces, coarsen):
        from luxar import LuxarZarrCompiler

        store = tmp_path / "cd.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=self.dims_4d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                substitutive_lod={"levels": 2, "coarsen_dims": coarsen},
            )
        return ladder_children(read_nodes(store))

    def test_named_dims_coarsen_the_axes_they_name(self, tmp_path):
        # A 2-dim coarsening makes z a hard BARRIER, and a barrier compares exact
        # values — on a sphere almost every z is distinct, so hardly anything
        # merges. Naming `["x", "y"]` must therefore produce a visibly different
        # (much less reduced) ladder than the 3-dim default.
        #
        # This is the discriminator the bug fails: with names dropped, `["x","y"]`
        # fell back to the first three columns and came out IDENTICAL to the
        # default. Measured — named 2-dim gives [144, 258], the default [26, 66,
        # 258] — so the two are not close enough to confuse.
        verts, faces = octasphere_4d(3, 0.0)
        two_dim = self.write(tmp_path / "xy", verts, faces, ["x", "y"])
        three_dim = self.write(tmp_path / "xyz", verts, faces, ["x", "y", "z"])

        assert two_dim[0]["n_vertices"] > three_dim[0]["n_vertices"] * 2, (
            "naming only x and y must barrier z and reduce far less — if the names "
            f"were dropped both would coarsen x/y/z and agree "
            f"({[c['n_vertices'] for c in two_dim]} vs "
            f"{[c['n_vertices'] for c in three_dim]})"
        )

    def test_names_and_indices_agree(self, tmp_path):
        # The same request written both ways must produce the same ladder. This is
        # what fails when names silently fall back to the default.
        verts, faces = octasphere_4d(3, 0.0)
        # x/y specifically, because that is the case whose fallback was silent:
        # a 3-dim request coincides with the default and would agree either way.
        by_name = self.write(tmp_path / "n", verts, faces, ["x", "y"])
        by_index = self.write(tmp_path / "i", verts, faces, [0, 1])
        assert [c["n_vertices"] for c in by_name] == [c["n_vertices"] for c in by_index]


class TestNormalFrame:
    """Coarse normals come from `normal_dims`, never from the coarsening axes.

    They are different quantities: `spatial_dims` is which axes the grid merges
    over and may be any number of them, while a normal always lives in exactly
    three. Deriving one from the other produced garbage rather than an error — a
    2-dim coarsening made `np.cross` return scalars, a 4-dim one made it raise.
    """

    def test_a_two_dim_coarsening_still_produces_usable_normals(self, tmp_path):
        from luxar import Dimension, Dimensions, LuxarZarrCompiler

        verts, faces = octasphere(3)
        normals = verts / np.linalg.norm(verts, axis=1, keepdims=True)
        dims = Dimensions(
            [
                Dimension(name=n, unit="um", range=(-2.0, 2.0), step=0.1, display=True)
                for n in ("x", "y", "z")
            ]
        )
        store = tmp_path / "nf.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_mesh(
                "surf",
                verts,
                faces,
                normals=normals.astype(np.float32),
                normal_dims=[0, 1, 2],
                substitutive_lod={"levels": 2, "coarsen_dims": [0, 1]},
            )
        children = ladder_children(read_nodes(store))
        assert len(children) >= 2
        assert all(c["has_normals"] for c in children), (
            "every level keeps normals; a 2-dim coarsening must not make the "
            "normal recomputation collapse"
        )

    def test_the_decimator_refuses_normals_without_a_three_axis_frame(self):
        from luxar.mesh.decimate import decimate_cluster

        verts, faces = octasphere(3)
        normals = (verts / np.linalg.norm(verts, axis=1, keepdims=True)).astype(
            np.float32
        )
        with pytest.raises(ValueError, match="normal_dims"):
            decimate_cluster(
                verts,
                faces,
                target_vertices=40,
                normals=normals,
                normal_dims=(0, 1),
            )
