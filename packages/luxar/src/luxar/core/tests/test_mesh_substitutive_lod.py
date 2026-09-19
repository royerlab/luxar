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

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from luxar import Dimensions, LuxarZarrCompiler
from luxar._zarr_compat import read_consolidated_attrs
from luxar.core.group.lod.group import (
    MESH_SUBSTITUTIVE_METHODS,
    PARTITION_FINEST_AREA,
    WHOLE_OBJECT_FINEST_ANCHOR,
)
from luxar.mesh.decimate import DECIMATION_METHODS


def test_mesh_lod_and_decimator_method_sets_stay_in_sync() -> None:
    assert MESH_SUBSTITUTIVE_METHODS == {"auto"} | DECIMATION_METHODS


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
    return dict(read_consolidated_attrs(store))


def write_ladder(tmp_path: Path, verts, faces, **kwargs) -> Dict[str, Dict[str, Any]]:
    store = tmp_path / "ladder.luxar.zarr"
    with LuxarZarrCompiler(store) as compiler:
        scene = compiler.create_scene(dimensions=Dimensions.default_3d())
        scene.add_mesh("surf", verts, faces, **kwargs)
    return read_nodes(store)


def ladder_child_paths(nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    """Node paths of the `surf` group's mesh children, coarsest→finest."""
    paths = [
        path
        for path, attrs in nodes.items()
        if path.startswith("surf/child_") and attrs.get("type") == "mesh"
    ]
    return sorted(paths, key=lambda path: int(path.rsplit("_", 1)[1]))


def level_meshes(store: Path) -> List[Any]:
    """Each level's DECODED mesh, coarsest→finest.

    Through `LuxarScene`, not the raw zarr arrays: neither the authored dtype nor
    the per-vertex count reaches the store intact. Colours are stored under an
    encoding — a float32 SDR colour as `rgb_uint8`, an HDR one quantized to uint16
    against a stamped range, a uniform colour as ONE `(1, 3)` row plus an element
    count — so the raw array reads back with a dtype and a length that are
    properties of the encoding, not of what was authored. The loader is the only
    lens that shows what the viewer will actually receive.

    Driven off the child list the writer actually produced, and deliberately NOT
    stopping at the first `get_mesh` failure: `get_mesh` raises `ValueError` for a
    level that is missing an array or otherwise will not decode, which is a defect
    these tests exist to catch. Walking `child_0, child_1, …` until something
    raises would instead hand back a SHORTER ladder that still satisfies a
    `len(levels) >= 2` assertion.
    """
    from luxar.io.reader import LuxarScene

    scene = LuxarScene.load(store)
    return [scene.get_mesh(path) for path in ladder_child_paths(read_nodes(store))]


def level_colors(store: Path) -> List[np.ndarray]:
    """Each level's decoded colours, coarsest→finest — one RGB triple per vertex.

    The per-vertex COUNT is checked here rather than left to each caller, because
    a uniform colour is stored as a single broadcast row plus an element count:
    the callers' value assertions all reduce over the rows, so a level whose count
    was wrong — or whose row was never expanded at all — holds exactly one correct
    colour and satisfies every one of them, while handing the viewer a colour
    buffer that does not match its own geometry.
    """
    out: List[np.ndarray] = []
    for index, mesh in enumerate(level_meshes(store)):
        colors = mesh.colors
        assert colors is not None, f"level {index} decodes with no colours"
        assert colors.shape == (len(mesh.vertices), 3), (
            f"level {index} decodes {colors.shape} colours for "
            f"{len(mesh.vertices)} vertices, not one RGB triple each"
        )
        out.append(colors)
    return out


def ladder_children(nodes: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The mesh children of the `surf` group, in written (coarsest→finest) order."""
    return [nodes[path] for path in ladder_child_paths(nodes)]


class TestLadderShape:
    @pytest.mark.parametrize("method", ["cluster", "qem"])
    def test_every_level_carries_separate_geometric_error(self, tmp_path, method):
        verts, faces = octasphere(3)
        children = ladder_children(
            write_ladder(tmp_path, verts, faces, substitutive_lod={"method": method})
        )

        errors = [child["level_stats"]["geometric_error"] for child in children]
        assert all(0.0 < error < 1.0 for error in errors[:-1])
        assert errors[-1] == 0.0
        assert all("reference_energy" not in child["level_stats"] for child in children)
        assert all("quality" not in child["level_stats"] for child in children)

    def test_no_reduction_notice_names_positive_attribute_weight(
        self, tmp_path, capsys
    ):
        verts, faces = octasphere(4)
        scalars = np.arange(len(verts), dtype=np.float32)

        write_ladder(
            tmp_path,
            verts,
            faces,
            scalars=scalars,
            colormap="viridis",
            substitutive_lod={"method": "cluster", "attribute_weight": 1.0},
        )

        message = capsys.readouterr().out
        assert "no level reduced the surface" in message
        assert "attribute_weight=1.0" in message

    def test_attribute_weight_is_forwarded_to_the_decimator(self, tmp_path):
        verts, faces = octasphere(3)
        side = verts[:, 0] + 0.31 * verts[:, 1] >= 0
        colors = np.zeros((len(verts), 3), np.uint8)
        colors[side, 0] = 255
        colors[~side, 2] = 255
        store = tmp_path / "ladder.luxar.zarr"
        write_ladder(
            tmp_path,
            verts,
            faces,
            colors=colors,
            substitutive_lod={"method": "qem", "attribute_weight": 1.0},
        )

        for level_colors_array in level_colors(store):
            assert not np.any(
                (level_colors_array[:, 0] > 0) & (level_colors_array[:, 2] > 0)
            )

    def test_writes_a_kind_lod_group_of_progressively_coarser_meshes(self, tmp_path):
        verts, faces = octasphere(4)  # 1026 vertices, 2048 faces
        nodes = write_ladder(
            tmp_path, verts, faces, substitutive_lod={"method": "cluster"}
        )

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
            write_ladder(tmp_path, verts, faces, substitutive_lod={"method": "cluster"})
        )
        for i, child in enumerate(children):
            assert child["n_faces"] > 0, f"level {i} has no triangles"
            assert child["n_vertices"] >= 3, f"level {i} cannot form a triangle"

    def test_coverage_fractions_are_strictly_ascending_and_bounded(self, tmp_path):
        verts, faces = octasphere(4)
        children = ladder_children(
            write_ladder(tmp_path, verts, faces, substitutive_lod={"method": "cluster"})
        )
        fractions = [c["coverage_fraction"] for c in children]
        assert fractions == sorted(fractions)
        assert len(set(fractions)) == len(fractions)
        assert fractions[0] == pytest.approx(0.0)
        # Screen-occupancy (AREA) halving: the finest whole-object level
        # anchors at half the screen area (0.5).
        assert fractions[-1] == pytest.approx(WHOLE_OBJECT_FINEST_ANCHOR)

    def test_level_count_follows_levels_and_K(self, tmp_path):
        verts, faces = octasphere(4)
        two = ladder_children(
            write_ladder(
                tmp_path,
                verts,
                faces,
                substitutive_lod={"levels": 2, "K": 4, "method": "cluster"},
            )
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
                tmp_path,
                verts,
                faces,
                colors=(200, 0, 0),
                substitutive_lod={"method": "cluster"},
            )
        )
        assert len(children) >= 2
        assert all(c["has_colors"] for c in children), (
            "every level keeps the uniform colour; a colourless coarse level "
            "renders default white and pops on the LOD switch"
        )
        # The VALUE too: `has_colors` is set just the same by a level that
        # forwarded the wrong colour, which pops on the switch exactly as visibly
        # as a colourless one.
        for index, level in enumerate(level_colors(tmp_path / "ladder.luxar.zarr")):
            unique = np.unique(level, axis=0)
            assert unique.shape[0] == 1, f"level {index} is no longer one colour"
            assert tuple(int(v) for v in unique[0]) == (200, 0, 0), (
                f"level {index} carries {unique[0]}, not the authored colour"
            )

    @pytest.mark.parametrize(
        "dtype,low,high",
        [
            (np.uint8, 0, 200),
            (np.uint16, 300, 50000),
            (np.float32, 0.1, 0.8),  # SDR
            (np.float32, 0.5, 4.0),  # HDR — must not be clipped to 1.0
        ],
        ids=["uint8", "uint16", "float32-sdr", "float32-hdr"],
    )
    def test_coarse_levels_keep_the_colour_VALUES_and_dtype(
        self, tmp_path, dtype, low, high
    ):
        # #1355: every documented `colors` shape but `uint8 (V,3)` either lost the
        # colour, corrupted it, or crashed, while the finest child kept it — so the
        # surface CHANGED COLOUR as the camera pulled back and the viewer swapped
        # levels. The three causes are fixed; this pins the fix.
        #
        # Reading the VALUES back, because `has_colors` cannot see any of it: a
        # level that came back black, clipped, or cast to another dtype sets the
        # flag just the same. That is what the previous version of this test
        # asserted, and it would have passed against every row of the issue table.
        #
        # A GRADIENT, not a uniform colour: the writer broadcasts a constant to a
        # `(1, 3)` row, which skips per-cluster averaging entirely. The old test
        # used a uniform colour and so never exercised the averaging path at all.
        #
        # The three channels are deliberately DIFFERENT — a ramp, then the two
        # ends of the range held constant. A ramp repeated in all three channels
        # cannot tell a dropped, duplicated or swapped channel from a correct one,
        # and averaging a constant channel leaves it exactly where it was, so the
        # two flat channels stay checkable at every level.
        verts, faces = octasphere(4)
        ramp = (verts[:, 2] - verts[:, 2].min()) / np.ptp(verts[:, 2])
        red = low + ramp * (high - low)
        colors = np.stack(
            [red, np.full_like(red, low), np.full_like(red, high)], axis=1
        ).astype(dtype)
        store = tmp_path / "ladder.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                colors=colors,
                substitutive_lod={
                    "levels": 3,
                    "compression_factor": 4,
                    "method": "cluster",
                },
            )

        levels = level_colors(store)
        assert len(levels) == 4, "3 coarse levels + the original"
        slack = 1.0 if np.issubdtype(np.dtype(dtype), np.integer) else 1e-4
        for index, level in enumerate(levels):
            assert level.dtype == dtype, (
                f"level {index} came back as {level.dtype}, not {dtype.__name__} — "
                "a dtype change rescales the colours the viewer shows"
            )
            gradient, flat_low, flat_high = (
                level[:, 0].astype(np.float64),
                level[:, 1].astype(np.float64),
                level[:, 2].astype(np.float64),
            )
            # The two flat channels pin the value per CHANNEL: a cluster mean of a
            # constant is that constant, so anything that drops, duplicates or
            # reorders a channel lands here — including the HDR row clipped to 1.0
            # and a level that came back black.
            assert np.allclose(flat_low, low, atol=slack), (
                f"level {index} channel 1 is {flat_low.min()}…{flat_low.max()}, "
                f"not the authored {low}"
            )
            assert np.allclose(flat_high, high, atol=slack), (
                f"level {index} channel 2 is {flat_high.min()}…{flat_high.max()}, "
                f"not the authored {high}"
            )
            # Averaging can only move values INSIDE the input range, never past it.
            assert gradient.min() >= low - slack
            assert gradient.max() <= high + slack
            assert gradient.max() > gradient.min(), (
                f"level {index} collapsed to a single colour — the gradient is gone"
            )
            # The top of the ramp must SURVIVE, not merely be within bounds. A
            # level averaged down to the middle of the range keeps a gradient and
            # stays in range while still washing the surface out on the switch.
            assert gradient.max() >= low + 0.5 * (high - low)

    def test_a_broadcast_1x3_colour_row_reaches_every_level(self, tmp_path):
        # The other shape #1355 measured: a `(1, 3)` row is the writer's own
        # broadcast form, and an ndarray-vs-not discriminator classifies it as
        # per-vertex data, so it reaches the decimator — which cannot cluster-average
        # one row against V vertices and raises a bare `ValueError` the adder's
        # funnel does not dress up. It has to be forwarded verbatim to every level
        # instead, which is what the shape-based discriminator does. Distinct from
        # the uniform TUPLE above — that never becomes an array at all, so it takes
        # a different branch and cannot cover this one.
        verts, faces = octasphere(3)
        store = tmp_path / "ladder.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                colors=np.array([[200, 30, 30]], dtype=np.uint8),
                substitutive_lod={
                    "levels": 2,
                    "compression_factor": 4,
                    "method": "cluster",
                },
            )

        levels = level_colors(store)
        assert len(levels) == 3, "2 coarse levels + the original"
        for index, level in enumerate(levels):
            assert level.dtype == np.uint8
            unique = np.unique(level, axis=0)
            assert unique.shape[0] == 1, (
                f"level {index} holds {unique.shape[0]} colours; a broadcast row is "
                "one colour for the whole surface"
            )
            assert tuple(int(v) for v in unique[0]) == (200, 30, 30), (
                f"level {index} carries {unique[0]}, not the authored colour"
            )

    def test_scalars_reach_every_level_of_a_colormapped_ladder(
        self, tmp_path: Path
    ) -> None:
        # `colormap` is a child attr, copied onto every level; the viewer maps
        # only where `has_scalars` is set. A ladder whose coarse levels lost
        # their scalars therefore renders them UNMAPPED against a mapped finest
        # level — the LOD switch changes the colours of the surface.
        from luxar.io.reader import LuxarScene

        verts, faces = octasphere(4)
        scalars = verts[:, 2].astype(np.float32)  # z, exactly [-1, 1]
        store = tmp_path / "sc.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                scalars=scalars,
                colormap="viridis",
                substitutive_lod={"levels": 2, "method": "cluster"},
            )
        children = ladder_children(read_nodes(store))
        assert len(children) >= 2
        assert all(c["has_scalars"] for c in children), (
            "a level with the colormap but no scalars renders unmapped"
        )
        assert all(c["colormap"] == "viridis" for c in children)

        loaded = LuxarScene.load(store)
        for idx, child in enumerate(children):
            values = loaded.get_mesh(f"surf/child_{idx}").scalars
            assert values is not None
            assert values.shape[0] == child["n_vertices"]
            # Cluster means of values in [-1, 1] stay in [-1, 1]: a level may
            # not invent a value the colormap would map outside the source range.
            assert float(values.min()) >= -1.0 - 1e-3
            assert float(values.max()) <= 1.0 + 1e-3
        # Averaged, not flattened to a constant: the coarsest level still spans
        # most of the range, so the colormap still shows the field.
        coarsest = loaded.get_mesh("surf/child_0").scalars
        assert float(coarsest.max() - coarsest.min()) > 1.0

    def test_every_level_stamps_the_SAME_scalar_data_range(
        self, tmp_path: Path
    ) -> None:
        """One display window for the whole ladder, not one per level.

        The viewer windows each level's colormap on that level's stamped
        `scalar_data_range`, and cluster-averaging strictly CONTRACTS the range —
        measured [0, 2.0] / [0, 1.04] / [0, 1.61] / [0, 100] down a real ladder.
        The coarsest then maps 2.0 to the top of the LUT and the finest maps the
        same 2.0 to t=0.02: the surface recolours as you zoom, which is the pop
        the ladder exists to avoid, reintroduced one layer down.

        A PEAKED field is what makes it visible: one hot vertex whose value no
        cluster mean can reach.
        """
        from luxar.io.reader import LuxarScene

        verts, faces = octasphere(4)
        scalars = np.zeros(len(verts), np.float32)
        scalars[0] = 100.0  # the peak: averaged away on every coarse level
        scalars[1:4] = 2.0
        store = tmp_path / "range.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                scalars=scalars,
                colormap="viridis",
                substitutive_lod={"levels": 3, "method": "cluster"},
            )
        children = ladder_children(read_nodes(store))
        assert len(children) >= 3
        ranges = [tuple(c["scalar_data_range"]) for c in children]
        assert len(set(ranges)) == 1, f"levels window on different ranges: {ranges}"
        assert ranges[0] == pytest.approx((0.0, 100.0))

        # The window is not free: it is also the quantization range, so a coarse
        # level's own (much smaller) values must still decode faithfully.
        loaded = LuxarScene.load(store)
        coarse = loaded.get_mesh("surf/child_0").scalars
        assert float(coarse.max()) < 100.0, "the peak must average away"
        assert float(coarse.max()) > 0.0, "…without taking the field with it"

    def test_a_window_NARROWER_than_the_field_is_still_shared(
        self, tmp_path: Path
    ) -> None:
        """An explicit window that does not contain the field must not split.

        `write_scalars` widens (never narrows) the supplied pair onto each node's
        own values, because it is also that node's quantization range. Forwarding
        a too-narrow window verbatim therefore let each level widen it against its
        OWN cluster means — different ranges again, from the one input that looks
        like it asks for the opposite. Unioning with the whole field up front
        makes every level stamp what a plain leaf over the same field would.
        """
        verts, faces = octasphere(4)
        scalars = np.zeros(len(verts), np.float32)
        scalars[0] = 100.0
        scalars[1] = -20.0
        nodes = write_ladder(
            tmp_path,
            verts,
            faces,
            scalars=scalars,
            colormap="viridis",
            substitutive_lod={"levels": 3, "method": "cluster"},
            _scalar_data_range=(0.0, 1.0),
        )
        children = ladder_children(nodes)
        assert len(children) >= 3
        ranges = [tuple(c["scalar_data_range"]) for c in children]
        assert len(set(ranges)) == 1, f"levels window on different ranges: {ranges}"
        assert ranges[0] == pytest.approx((-20.0, 100.0))

    def test_the_private_range_key_never_reaches_disk(self, tmp_path: Path) -> None:
        # `_scalar_data_range` is plumbing for the scalars writer, not an
        # attribute; a private key on a node is something every reader and the
        # viewer would have to know to ignore.
        verts, faces = octasphere(3)
        nodes = write_ladder(
            tmp_path,
            verts,
            faces,
            scalars=verts[:, 2].astype(np.float32),
            colormap="viridis",
            substitutive_lod={"levels": 2, "method": "cluster"},
        )
        assert all("_scalar_data_range" not in attrs for attrs in nodes.values())


class TestScalarDataRangeIsInternal:
    """`_scalar_data_range` is mesh plumbing, not a node attribute.

    It was briefly listed in the SHARED `_ALLOWED_NODE_ATTRS`, which is
    geometry-blind: only the mesh writer pops it, so on points / lines / gsplats
    it sailed through the gate and landed on disk as a private key sitting next
    to the `scalar_data_range` it exists to override. And nothing checked its
    shape, so a 1-tuple reached the writer and raised a bare `IndexError` — a
    type the adder's ValueError/TypeError funnel does not catch, so it escaped
    mid-write with the vertices and faces already written.
    """

    def scene_dims(self) -> Any:
        return Dimensions.default_3d()

    @pytest.mark.parametrize("geometry", ["points", "lines"])
    def test_a_sibling_geometry_refuses_the_key_outright(
        self, tmp_path: Path, geometry: str
    ) -> None:
        store = tmp_path / f"{geometry}.luxar.zarr"
        positions = np.random.default_rng(0).random((12, 3)).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown node attribute"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=self.scene_dims())
                if geometry == "points":
                    scene.add_points("node", positions, _scalar_data_range=(0.0, 5.0))
                else:
                    scene.add_lines(
                        "node",
                        positions,
                        np.full(12, 0.1, np.float32),
                        _scalar_data_range=(0.0, 5.0),
                    )
        # The gate is pre-write, so nothing of the refused node reached disk.
        assert not (store / "node").exists()

    @pytest.mark.parametrize(
        "bad", [(1.0,), "foo", 3.5, (1.0, 0.0), (float("nan"), 1.0)]
    )
    def test_a_malformed_range_is_refused_before_anything_is_written(
        self, tmp_path: Path, bad: Any
    ) -> None:
        # A 1-tuple and a string used to raise IndexError/ValueError from inside
        # the writer, past the adder's funnel and past the point of no return; a
        # reversed pair and a NaN were swallowed silently by the widening, which
        # then encoded the field over a nonsense quantization window.
        verts, faces = octasphere(3)
        store = tmp_path / "bad.luxar.zarr"
        with pytest.raises(ValueError, match="_scalar_data_range"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh(
                    "surf",
                    verts,
                    faces,
                    scalars=verts[:, 2].astype(np.float32),
                    colormap="viridis",
                    _scalar_data_range=bad,
                )
        assert not (store / "surf").exists(), "a refused mesh must write nothing"

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    @pytest.mark.parametrize("explicit_window", [None, (-1.0, 1.0)])
    def test_a_nonfinite_field_leaves_no_childless_group(
        self, tmp_path: Path, bad_value: float, explicit_window: Any
    ) -> None:
        """A ladder must refuse a NaN/Inf field the way a plain leaf does.

        The array writer refuses non-finite values regardless, so a plain
        `add_mesh` fails and writes nothing. On the ladder path that refusal used
        to arrive from inside `child_0` — with `add_lod_group` already run — so
        the store was left holding a CHILDLESS `kind=lod` node, which no viewer
        path can load: a ladder is resolved from its children. An explicit window
        does not make the values writable, so it must not buy a way past this.
        """
        verts, faces = octasphere(3)
        scalars = verts[:, 2].astype(np.float32).copy()
        scalars[3] = bad_value
        store = tmp_path / "nonfinite.luxar.zarr"
        extra = (
            {} if explicit_window is None else {"_scalar_data_range": explicit_window}
        )
        with pytest.raises(ValueError, match="NaN or Inf"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh(
                    "surf",
                    verts,
                    faces,
                    scalars=scalars,
                    colormap="viridis",
                    substitutive_lod=True,
                    **extra,
                )
        assert not (store / "surf").exists(), "no childless kind=lod may be left"


def malformed_array_cases(n_vertices: int) -> List[Any]:
    """One case per array validator: (id, add_mesh kwargs, expected message).

    Sized against `octasphere(3)`, whose vertex count the tests assert.
    """
    rgb = np.zeros((n_vertices, 3), dtype=np.float32)
    cases: List[Tuple[str, Dict[str, Any], str]] = [
        ("colors_channels", {"colors": np.zeros((n_vertices, 2), np.float32)}, "3"),
        ("colors_length", {"colors": rgb[:-1]}, "match"),
        ("colors_uniform", {"colors": (1.0, 0.0)}, "RGB"),
        (
            "normals_shape",
            {
                "normals": np.zeros((n_vertices, 2), np.float32),
                "normal_dims": (0, 1, 2),
            },
            "normals",
        ),
        (
            "normals_length",
            {
                "normals": np.zeros((n_vertices - 1, 3), np.float32),
                "normal_dims": (0, 1, 2),
            },
            "normals",
        ),
        ("normal_dims_alone", {"normal_dims": (0, 1, 2)}, "normal_dims"),
        ("shading", {"shading": "gouraud"}, "shading"),
        ("double_sided", {"double_sided": "yes"}, "double_sided"),
        ("labels", {"labels": ["a"] * (n_vertices - 1)}, "Labels length"),
        (
            "image_labels",
            # issue #1491: image_labels had no pre-split gate at all, so a
            # wrong-length list was refused only from inside the FINEST child
            # (the last one written), stranding every coarse level on disk.
            {
                "image_labels": [
                    np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(n_vertices - 1)
                ]
            },
            "Image labels length",
        ),
        (
            "scalars_length",
            {"scalars": np.zeros(n_vertices - 1, np.float32), "colormap": "viridis"},
            "scalars",
        ),
    ]
    return [pytest.param(kwargs, message, id=id_) for id_, kwargs, message in cases]


class TestMalformedArraysWriteNothing:
    """Every array a child write validates must be validated before the group exists.

    The ladder writes the coarse children first and the ORIGINAL surface last, so
    a malformed input surfaced from whichever child first carried it. Colours with
    two components or a typo'd `shading` are validated identically on EVERY level,
    so they died in whichever child's write hit them first — typically `child_0`,
    leaving a childless `kind=lod` group (or, if a later level was the one to
    fail, one missing only its finest level). `labels` and `image_labels`
    (#1491) fail a DIFFERENT way: both are forwarded ONLY to the finest child, so
    a bad one died deep inside THAT child's own write — after its vertices,
    faces, normals, colours and scalars were already on disk. That leaves every
    level, the finest included, fully written and independently loadable; only
    the finest level's own label channel is silently missing, which is harder to
    notice than a missing surface, not milder. Either shape is a store the
    plain-leaf path's "nothing written" does not have.

    Parametrized over one case per validator rather than one per channel, since
    the fix is a shared gate: what is being pinned is that the gate runs on the
    ladder path, for the whole set.
    """

    @pytest.mark.parametrize("kwargs, message", malformed_array_cases(258))
    def test_a_malformed_array_leaves_no_group_behind(
        self, tmp_path: Path, kwargs: Dict[str, Any], message: str
    ) -> None:
        verts, faces = octasphere(3)
        assert len(verts) == 258, "the parametrized shapes assume this vertex count"
        store = tmp_path / "bad.luxar.zarr"
        with pytest.raises(ValueError, match=message):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh("surf", verts, faces, substitutive_lod=True, **kwargs)
        assert not (store / "surf").exists(), (
            "a refused mesh must not leave a half-built LOD group behind"
        )

    @pytest.mark.parametrize("kwargs, message", malformed_array_cases(258))
    def test_the_plain_leaf_refuses_it_the_same_way(
        self, tmp_path: Path, kwargs: Dict[str, Any], message: str
    ) -> None:
        """The control: the ladder gate must not be stricter or laxer than a leaf.

        Both paths run the same validator now, so what this pins is that the two
        agree — a case the leaf accepts must not be refused by the ladder.
        """
        verts, faces = octasphere(3)
        store = tmp_path / "leaf.luxar.zarr"
        with pytest.raises(ValueError, match=message):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh("surf", verts, faces, **kwargs)
        assert not (store / "surf").exists()

    def test_a_wrong_dtype_faces_array_is_refused_before_the_group(
        self, tmp_path: Path
    ) -> None:
        """Float faces reached the finest child only — the coarse ones are recast.

        `decimate_cluster` casts its faces to uint32, so every coarse level came
        out valid and only the ORIGINAL array was refused, three children into the
        write.
        """
        verts, faces = octasphere(3)
        store = tmp_path / "faces.luxar.zarr"
        with pytest.raises(ValueError, match="integer array"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh(
                    "surf", verts, faces.astype(np.float32), substitutive_lod=True
                )
        assert not (store / "surf").exists()

    def test_image_labels_refusal_is_not_double_prefixed(self, tmp_path: Path) -> None:
        """Pre-#1491: ``Could not add mesh 'surf': Could not add mesh 'child_3': …``.

        The finest child (here ``child_3``, after three decimated coarse levels)
        is where a wrong-length ``image_labels`` used to be caught, and its OWN
        funnel had already produced ``Could not add mesh 'child_3': …`` before
        the outer ``add_mesh("surf", …)`` call wrapped it a second time. The gate
        now runs pre-split, so the message names only the node the caller passed.
        """
        verts, faces = octasphere(3)
        n_vertices = len(verts)
        store = tmp_path / "img_prefix.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            try:
                scene.add_mesh(
                    "surf",
                    verts,
                    faces,
                    image_labels=[b"x"] * (n_vertices - 1),
                    substitutive_lod=True,
                )
            except ValueError as exc:
                message = str(exc)
                assert message.startswith("Could not add mesh 'surf': ")
                assert "child_" not in message
                assert message.count("Could not add") == 1
            else:  # pragma: no cover
                raise AssertionError("expected a ValueError")
        assert not (store / "surf").exists()

    def test_sparse_image_labels_out_of_range_index_is_refused_before_the_group(
        self, tmp_path: Path
    ) -> None:
        """The dict (sparse) form is checked pre-split too, not only the length —
        the Mesh twin of the Points/Lines sparse-dict cases (#1491 review)."""
        verts, faces = octasphere(3)
        n_vertices = len(verts)
        store = tmp_path / "img_sparse.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="out of range"):
                scene.add_mesh(
                    "surf",
                    verts,
                    faces,
                    image_labels={n_vertices: b"x"},
                    substitutive_lod=True,
                )
        assert not (store / "surf").exists()

    def test_valid_image_labels_still_ladder_and_reach_the_finest_child(
        self, tmp_path: Path
    ) -> None:
        """The control: a correctly-sized ``image_labels`` still writes end to end.

        The new gate must not reject legal input — the coarse decimated levels
        carry no ``image_labels`` channel at all (only the finest, original
        surface does), so the ladder is complete once that child's flag is set.
        """
        verts, faces = octasphere(3)
        n_vertices = len(verts)
        nodes = write_ladder(
            tmp_path,
            verts,
            faces,
            image_labels=[b"x"] * n_vertices,
            substitutive_lod=True,
        )
        children = ladder_children(nodes)
        assert len(children) > 1
        finest = children[-1]
        assert finest["type"] == "mesh"
        assert finest["has_image_labels"] is True
        assert finest["n_vertices"] == n_vertices
        for coarse in children[:-1]:
            assert coarse.get("has_image_labels") is not True

    def test_non_finite_vertices_say_so_instead_of_blaming_the_decimator(
        self, tmp_path: Path
    ) -> None:
        """A NaN coordinate is a data problem, not a degenerate surface.

        It used to be diagnosed as "decimation collapsed every triangle … the
        input is degenerate (collinear or coincident vertices)", because a NaN
        cell key makes every triangle collapse. Nothing wrote, so the store was
        fine; the message sent the author looking for the wrong mistake.
        """
        verts, faces = octasphere(3)
        verts = verts.copy()
        verts[5, 1] = np.nan
        store = tmp_path / "nan.luxar.zarr"
        with pytest.raises(ValueError, match="vertices: Contains 1 NaN or Inf"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh("surf", verts, faces, substitutive_lod=True)


class TestExtendToAll:
    """`extend_to_all=` and `substitutive_lod=` must compose.

    They did not. The adder resolved the extension into `attrs` and then handed
    the wrapper both an `extend_to_all=` argument and `**attrs`, so any ladder on
    a scene with a non-displayed dimension died with "got multiple values for
    keyword argument 'extend_to_all'" — a TypeError funnelled into a ValueError
    about a mesh that was perfectly valid. The sibling adders dispatch their
    structural branches BEFORE that resolution, which is why they never hit it.
    """

    def write(
        self, tmp_path: Path, extend: Any, **kwargs: Any
    ) -> Dict[str, Dict[str, Any]]:
        from luxar import Dimension, Dimensions

        verts, faces = octasphere_4d(4, 0.0)
        dims = Dimensions(
            [
                Dimension(
                    name="x", unit="um", range=(-2.0, 2.0), step=0.1, display=True
                ),
                Dimension(
                    name="y", unit="um", range=(-2.0, 2.0), step=0.1, display=True
                ),
                Dimension(
                    name="z", unit="um", range=(-2.0, 2.0), step=0.1, display=True
                ),
                Dimension(
                    name="t", unit="s", range=(0.0, 9.0), step=1.0, display=False
                ),
            ]
        )
        store = tmp_path / "e.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=dims)
            scene.add_mesh(
                "surf",
                verts,
                faces,
                extend_to_all=extend,
                substitutive_lod={"levels": 2},
                **kwargs,
            )
        return read_nodes(store)

    @pytest.mark.parametrize("extend", ["all", ["t"]])
    def test_the_ladder_builds_and_every_child_carries_the_resolution(
        self, tmp_path: Path, extend: Any
    ) -> None:
        nodes = self.write(tmp_path / str(extend), extend)
        assert nodes["surf"]["kind"] == "lod"
        children = ladder_children(nodes)
        assert len(children) >= 2
        # Resolved PER CHILD: the wrapper forwards the raw value down, and each
        # child's own `add_mesh` turns it into the concrete dimension list.
        assert all(c.get("extend_to_all") == ["t"] for c in children), (
            f"levels missing the resolved extension: "
            f"{[c.get('extend_to_all') for c in children]}"
        )

    def test_an_INVALID_attr_is_also_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """The pre-write gate covers the ATTRS, not just `extend_to_all`.

        A typo'd `colormap` (or `blending`, or any unknown key) is caught by
        `validate_render_attrs`. Since #1534 that check runs at the very top of
        `add_mesh_impl`, above every structural branch (this substitutive one
        included) and above `extend_to_all` resolution — so it fires before any
        decimation or `add_lod_group` call happens at all, not merely before the
        finest child's own write. Pre-#1534 the substitutive branch already ran
        this same validator (just after resolving `extend_to_all`, and inside
        the wrapper dispatch rather than the adder entry), so this call was
        already refused before anything was written either way; what changed is
        only the precedence against `extend_to_all` — see
        `TestMeshSubstitutiveNodeAttrsGateOutranksExtendToAll` in
        `tests/group/lod/test_source_validation.py` for that half.
        """
        verts, faces = octasphere(3)
        store = tmp_path / "attr.luxar.zarr"
        with pytest.raises(ValueError, match="magmaa|Unknown|colormap"):
            with LuxarZarrCompiler(store) as compiler:
                scene = compiler.create_scene(dimensions=Dimensions.default_3d())
                scene.add_mesh(
                    "surf",
                    verts,
                    faces,
                    scalars=verts[:, 2].astype(np.float32),
                    colormap="magmaa",
                    substitutive_lod={"levels": 2},
                )
        assert not (store / "surf").exists(), (
            "a refused mesh must not leave a half-built LOD group behind"
        )

    def test_an_INVALID_value_is_refused_before_anything_is_written(
        self, tmp_path: Path
    ) -> None:
        """The fail-fast pre-write gate must survive the dispatch move.

        The children re-resolve `extend_to_all`, so a bad value is caught either
        way — but only inside `child_0`, i.e. after every level has been
        decimated and after `add_lod_group` created the zarr group. That left
        `surf` behind as a childless kind=lod group in a store flagged
        incomplete, where the plain-leaf path writes nothing at all.
        """
        with pytest.raises(ValueError, match="extend_to_all"):
            self.write(tmp_path, "everything")
        assert not (tmp_path / "e.luxar.zarr" / "surf").exists(), (
            "a refused mesh must not leave a half-built LOD group behind"
        )


class TestDegenerateLadders:
    def test_a_surface_too_coarse_to_reduce_falls_back_to_a_plain_leaf(self, tmp_path):
        # An octahedron has 6 vertices: every requested target is below the
        # decimator's 4-vertex floor or fails to reduce. Writing a one-level
        # "ladder" would be a kind=lod group with a single child, which is a
        # pointless indirection; writing duplicate levels would make the viewer
        # swap between identical surfaces.
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

    def test_every_decimation_method_builds_a_ladder(self, tmp_path):
        verts, faces = octasphere(4)
        for method in ("auto", "cluster", "qem"):
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

    def write(self, tmp_path, verts, faces, coarsen, *, dim_order=None):
        from luxar import LuxarZarrCompiler

        store = tmp_path / "cd.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=self.dims_4d())
            scene.add_mesh(
                "surf",
                verts,
                faces,
                dim_order=dim_order,
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

    def test_display_dims_after_nonidentity_dim_order(self, tmp_path):
        verts, faces = octasphere_4d(3, 0.0)
        canonical = self.write(tmp_path / "canonical", verts, faces, "display")
        permuted = self.write(
            tmp_path / "permuted",
            verts[:, [3, 0, 1, 2]],
            faces,
            "display",
            dim_order=["t", "x", "y", "z"],
        )

        canonical_counts = [child["n_vertices"] for child in canonical]
        permuted_counts = [child["n_vertices"] for child in permuted]
        assert permuted_counts == canonical_counts
        assert permuted_counts[0] < permuted_counts[-1]


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

    def test_the_decimator_refuses_a_repeated_axis_in_the_frame(self):
        """Three entries is not enough — they have to be three DIFFERENT axes.

        `(0, 0, 1)` has the right length and is in range, so it passed both other
        guards and then had the normals crossed inside a degenerate plane: values
        that are finite, unit-length and meaningless. The writer refuses it too,
        but only once the ladder's first child is being written, by which point a
        `kind=lod` group is already on disk.
        """
        from luxar.mesh.decimate import decimate_cluster

        verts, faces = octasphere(3)
        normals = (verts / np.linalg.norm(verts, axis=1, keepdims=True)).astype(
            np.float32
        )
        with pytest.raises(ValueError, match="DISTINCT"):
            decimate_cluster(
                verts,
                faces,
                target_vertices=40,
                normals=normals,
                normal_dims=(0, 0, 1),
            )


class TestPartitionBoundAnchorMesh:
    """A hand-built partition of per-tile mesh ladders gets the fills-screen anchor.

    `add_mesh` refuses `partition=` together with `substitutive_lod=`, so building
    the `kind=partition` wrapper by hand and calling the adder once per part is the
    ONLY way to get per-tile mesh ladders. Before the mesh wrapper routed through
    `derive_coverage_fractions` it always derived the WHOLE-OBJECT ladder (finest
    = the whole-object anchor), so both tiles sat on their finest decimation level
    at the opening full-frame view. Per-tile ladders anchor at
    `PARTITION_FINEST_AREA` (1.0 — the tile alone fills the screen). The
    Points/Lines peers of these tests live in
    `core/tests/group/lod/test_substitutive_{points,lines}.py`.
    """

    @staticmethod
    def write(tmp_path: Path, *, partitioned: bool = True, **spec) -> Path:
        """Two offset spheres, each its own ladder, under a partition or the root.

        `partitioned=False` places the SAME two ladders at the scene root — the
        over-trigger control, where only the insertion point differs.
        """
        verts, faces = octasphere(4)
        store = tmp_path / "tiled.luxar.zarr"
        with LuxarZarrCompiler(store) as compiler:
            scene = compiler.create_scene(dimensions=Dimensions.default_3d())
            wrapper = (
                scene.add_partition_group(
                    "tiled", display_type="mesh", max_elements=len(verts)
                )
                if partitioned
                else scene
            )
            for i, shift in enumerate((-3.0, 3.0)):
                shifted = verts.copy()
                shifted[:, 0] += shift
                wrapper.add_mesh(
                    f"part_{i}",
                    shifted,
                    faces,
                    substitutive_lod={"levels": 2, **spec},
                )
        return store

    @staticmethod
    def coverage(nodes: Dict[str, Dict[str, Any]], part: str) -> List[float]:
        kids = [
            (path, attrs)
            for path, attrs in nodes.items()
            if path.startswith(f"{part}/child_") and attrs.get("type") == "mesh"
        ]
        kids.sort(key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
        return [float(attrs["coverage_fraction"]) for _, attrs in kids]

    def test_every_part_ladder_is_partition_anchored(self, tmp_path):
        from luxar.core.group.lod.group import partitioned_coverage_fractions

        nodes = read_nodes(self.write(tmp_path))
        assert nodes["tiled"]["kind"] == "partition"
        for i in range(2):
            part = f"tiled/part_{i}"
            assert nodes[part]["kind"] == "lod"
            counts = [
                int(attrs["n_vertices"])
                for path, attrs in sorted(
                    (
                        (p, a)
                        for p, a in nodes.items()
                        if p.startswith(f"{part}/child_") and a.get("type") == "mesh"
                    ),
                    key=lambda kv: int(kv[0].rsplit("_", 1)[1]),
                )
            ]
            assert len(counts) >= 2
            assert self.coverage(nodes, part) == pytest.approx(
                partitioned_coverage_fractions(counts)
            )
            assert self.coverage(nodes, part)[-1] == pytest.approx(
                PARTITION_FINEST_AREA
            )

    def test_scene_root_still_gets_the_whole_object_anchor(self, tmp_path):
        """CONTROL: outside a partition, the half-screen-area anchor holds."""
        nodes = read_nodes(self.write(tmp_path, partitioned=False))
        for i in range(2):
            assert self.coverage(nodes, f"part_{i}")[-1] == pytest.approx(
                WHOLE_OBJECT_FINEST_ANCHOR
            )

    def test_explicit_coverage_fractions_still_win_under_a_partition(self, tmp_path):
        """CONTROL: an explicit list is used verbatim, partition or not."""
        explicit = [0.0, 0.6, 1.0]
        nodes = read_nodes(self.write(tmp_path, coverage_fractions=explicit))
        for i in range(2):
            assert self.coverage(nodes, f"tiled/part_{i}") == pytest.approx(explicit)

    def test_an_explicit_list_may_reach_the_legacy_ceiling(self, tmp_path):
        """An explicit mesh ladder may END at `MAX_COVERAGE_FRACTION`, verbatim.

        The mesh resolver (`lod/mesh.py::_validate_coverage_fractions_spec`) bounds
        an explicit list at `[0, MAX_COVERAGE_FRACTION]`, exactly like the
        Points/Lines/GSplats resolvers. It has to, because an explicit list is
        stamped `LEGACY_LOD_SELECTOR` and `MAX_COVERAGE_FRACTION` = 4.0 IS the
        legacy diagonal metric's ceiling — the range an author's values were tuned
        in. (Not a derived anchor: a derived whole-object ladder ends at
        `WHOLE_OBJECT_FINEST_ANCHOR` = 0.5 and a derived partition-bound one at
        `PARTITION_FINEST_AREA` = 1.0, both well below this bound. Capping the
        authored list at either would silently reject legal legacy values.)
        Anything above the ceiling still raises.
        """
        from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION

        explicit = [0.0, 2.0, MAX_COVERAGE_FRACTION]
        nodes = read_nodes(self.write(tmp_path, coverage_fractions=explicit))
        for i in range(2):
            assert self.coverage(nodes, f"tiled/part_{i}") == pytest.approx(explicit)

        over = tmp_path / "over"
        over.mkdir()
        with pytest.raises(ValueError) as exc:
            self.write(over, coverage_fractions=[0.0, 2.0, MAX_COVERAGE_FRACTION + 0.5])
        assert f"[0, {MAX_COVERAGE_FRACTION:g}]" in str(exc.value)
