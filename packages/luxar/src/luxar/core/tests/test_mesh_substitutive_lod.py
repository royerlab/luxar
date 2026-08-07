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
