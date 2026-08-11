"""Per-element channels are validated against the SOURCE count, pre-split (#1437).

The LOD-wrapper half of the gate (the partition half lives in
``tests/group/partition/test_source_validation.py``, which also explains why the
parity assertion is exact-message rather than a substring match).
``slice_optional_array`` passes a value whose leading length does not match the
element count through UNCHANGED — that is how a broadcast RGB triple or a scalar
radius reaches every level — so a per-element array of the WRONG length is handed
to every level whole, and a level whose own count happens to equal that array's
length ACCEPTS it. Each additive case below is sized so that coincidence holds
(explicit CUMULATIVE ``counts`` give two equal levels), which is what makes the
per-level check insufficient.

The substitutive wrappers are the shape where a downstream check does eventually
reject (their finest child carries the FULL element set), so those are pinned by
WHERE the write fails: the finest child is written LAST, after ``add_lod_group``
and every coarse gsplat level, so without the gate a wrong-length channel strands
a partial ``kind=lod`` node on disk.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import zarr

from luxar.io.reader import LuxarScene

from ..conftest import (
    assert_same_refusal,
    assert_uniform,
    open_scene,
    random_positions,
    refusal,
)

# 100 elements with cumulative counts [50, 100] gives two levels of 50, so a
# 50-long channel matches every level and only a check against the full 100 can
# reject it (the same construction the #1422 labels tests use).
_N = 100
_HALF = 50

# Polyline units for Lines: with line_type="segments" each segment is a 2-vertex
# polyline, so cumulative [25, 50] gives two levels of 50 VERTICES each.
_LINES_LADDER = {"counts": [_HALF // 2, _N // 2]}
_POINTS_LADDER = {"counts": [_HALF, _N]}

_POINTS_CASES = [
    ("colors", np.zeros((_HALF, 3), dtype=np.float32)),
    ("radii", np.full(_HALF, 0.5, dtype=np.float32)),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32)),
]

_LINES_CASES = [
    ("widths", np.full(_HALF, 0.2, dtype=np.float32)),
    ("colors", np.zeros((_HALF, 3), dtype=np.float32)),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32)),
]

# The substitutive wrappers need enough elements for the gsplat reduce to
# synthesise coarse levels (a degenerate input falls back to a flat node).
_SUB_N = 400
_SUB_HALF = 200

_POINTS_SUB_CASES = [
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32)),
    ("radii", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32)),
]

_LINES_SUB_CASES = [
    ("widths", np.full(_SUB_HALF, 0.2, dtype=np.float32)),
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32)),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32)),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32)),
]


def _points_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    # Assigned last and explicitly, never as a duplicate dict-literal key.
    kwargs[channel] = value
    return kwargs


def _lines_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"widths": 0.2}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    kwargs[channel] = value
    return kwargs


#: The uniform colour forms the flat path accepts, all four of which the
#: substitutive wrappers used to refuse downstream in the gsplat lift (#1444),
#: each paired with the row EVERY level must end up carrying — alpha included,
#: because gsplats carry per-splat alpha and all three shaders scale intensity
#: by it (dropping it would make the coarse levels 1/alpha too bright the
#: instant the ladder switches off the finest child).
_BROADCAST_RGB = (0.25, 0.5, 1.0)
_BROADCAST_ALPHA = 0.5
_BROADCAST_COLORS = [
    (_BROADCAST_RGB, _BROADCAST_RGB),  # RGB tuple
    (np.array([_BROADCAST_RGB], dtype=np.float32), _BROADCAST_RGB),  # (1, 3) row
    (  # RGBA tuple
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
    ),
    (  # (1, 4) row
        np.array([(*_BROADCAST_RGB, _BROADCAST_ALPHA)], dtype=np.float32),
        (*_BROADCAST_RGB, _BROADCAST_ALPHA),
    ),
    (  # opaque RGBA — the clamp endpoint, see _assert_coarse_levels_carry_color
        (*_BROADCAST_RGB, 1.0),
        (*_BROADCAST_RGB, 1.0),
    ),
]

#: Tolerance for the coarse-level colour check. The merge round-trips a
#: per-splat alpha through optical depth, whose ``ALPHA_CLAMP = 511/512`` caps an
#: opaque input: alpha 1.0 comes back as 0.998046875 on every coarse level
#: (measured), a 1.95e-3 step the finest child does not have. This tolerance
#: admits exactly that and nothing looser.
_ALPHA_CLAMP_ATOL = 2.5e-3

#: Colours whose dtype the leaf write refuses (COLOR arrays are floating, uint8
#: or uint16), in both the uniform-row and per-element shapes. Normalising
#: either would bake a near-black coarse level that the encoder then rejects at
#: the finest child — after the coarse levels are on disk. Nothing may be
#: written for them.
_BAD_DTYPE_COLORS = [
    ("uniform row", np.array([[255, 0, 0]], dtype=np.int64)),
    ("per-element", np.tile([255, 0, 0], (_SUB_N, 1)).astype(np.int64)),
]


def _assert_coarse_levels_carry_color(
    reader: LuxarScene, node: str, coarse: "list[str]", want: Any
) -> None:
    """Every coarse gsplat level of a substitutive group carries the authored colour.

    A uniform colour is exactly the case a coarse level can honour trivially
    (every merged representative is that same colour), so this is an equality
    check, not a "some colour was written" one — and it covers the alpha column,
    whose loss would be a brightness jump at the LOD seam rather than a refusal.
    Equality is to within :data:`_ALPHA_CLAMP_ATOL`, the merge's optical-depth
    clamp at the opaque endpoint; RGB and every alpha below 1 are exact.
    """
    assert coarse, "no coarse gsplat levels were written"
    for child in coarse:
        data = reader.get_gsplats(f"{node}/{child}")
        assert data.colors is not None, f"{child} lost its colours"
        n = int(np.asarray(data.centers).shape[0])
        assert np.asarray(data.colors).shape[1] == len(want), (
            f"{child} carries {np.asarray(data.colors).shape[1]} channels, "
            f"expected {len(want)} (a dropped alpha renders 1/alpha too bright)"
        )
        assert_uniform(data.colors, want, n, atol=_ALPHA_CLAMP_ATOL)


def _n_levels(path: str, node: str) -> int:
    store = zarr.open_group(path, mode="r")
    n_levels = int(store[node].attrs["n_additive_sublods"])
    assert n_levels > 1, "the ladder did not fire, so this tests the flat path"
    return n_levels


class TestPointsAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _POINTS_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_add_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"points_add_flat_{channel}.luxar.zarr")
        positions = random_positions(_N, seed=31)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, **_points_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                additive_lod=_POINTS_LADDER,
                **_points_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "p" not in compiler.store


class TestLinesAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _LINES_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_add_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lines_add_flat_{channel}.luxar.zarr")
        vertices = random_positions(_N, seed=32)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                **_lines_kwargs(channel, value),
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                additive_lod=_LINES_LADDER,
                **_lines_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "line" not in compiler.store


class TestPointsSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _POINTS_SUB_CASES)
    def test_wrong_length_channel_is_refused_before_anything_is_written(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_sub_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"points_sub_flat_{channel}.luxar.zarr")
        positions = random_positions(_SUB_N, seed=41)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, **_points_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                substitutive_lod=True,
                **_points_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        # The gate runs above add_lod_group, so no partial kind=lod node is left.
        assert "p" not in compiler.store


class TestLinesSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value", _LINES_SUB_CASES)
    def test_wrong_length_channel_is_refused_before_anything_is_written(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_sub_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lines_sub_flat_{channel}.luxar.zarr")
        vertices = random_positions(_SUB_N, seed=42)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                **_lines_kwargs(channel, value),
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                line_type="segments",
                substitutive_lod=True,
                **_lines_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "line" not in compiler.store


# 24 vertices as 12 edges (see the partition sibling for why this shape).
_N_IDX = 24
_N_EDGES = 12
_INDEX_CASES = [
    ("bad_layout", np.arange(_N_IDX, dtype=np.uint32).reshape(8, 3)),
    ("odd_flat", np.arange(_N_IDX - 1, dtype=np.uint32)),
]
_INDEX_SPLITS = [
    ("additive", {"additive_lod": {"n_lods": 3}}),
    ("substitutive", {"substitutive_lod": True}),
]


class TestLinesIndicesLodValidation:
    """The LOD half of the ``indices`` gate (the partition half is in partition/).

    ``make_additive_lod_lines`` checks dtype and bounds and then reshapes to
    pairs, so a malformed edge list was reinterpreted rather than refused; the
    substitutive path refused it only from the finest child, with the coarse
    levels already on disk.
    """

    @pytest.mark.parametrize("case,indices", _INDEX_CASES)
    @pytest.mark.parametrize("split,split_kwargs", _INDEX_SPLITS)
    def test_malformed_indices_refused_exactly_as_the_flat_path(
        self,
        tmp_path: Any,
        case: str,
        indices: Any,
        split: str,
        split_kwargs: Dict[str, Any],
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"idx_{split}_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"idx_flat_{split}_{case}.luxar.zarr")
        vertices = random_positions(_N_IDX, seed=15)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, indices=indices, line_type="indexed"
            )
        )
        split_exc = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                indices=indices,
                line_type="indexed",
                **split_kwargs,
            )
        )

        assert_same_refusal(flat, split_exc)
        assert "line" not in compiler.store

    @pytest.mark.parametrize(
        "layout,indices",
        [
            ("pairs", np.arange(_N_IDX, dtype=np.uint32).reshape(-1, 2)),
            ("flat", np.arange(_N_IDX, dtype=np.uint32)),
        ],
    )
    def test_legal_layouts_still_ladder_with_every_edge_intact(
        self, tmp_path: Any, layout: str, indices: Any
    ) -> None:
        """The control: both documented layouts ladder, and keep all 12 edges.

        Edge count is the assertion that matters — an ``(E, 3)`` array
        reinterpreted as ``3E/2`` edges is the bug this gate exists for, which a
        mere "the node exists" check cannot see.
        """
        compiler, scene, path = open_scene(tmp_path, f"idx_ok_add_{layout}.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N_IDX, seed=16),
            widths=0.2,
            indices=indices,
            line_type="indexed",
            additive_lod={"n_lods": 3},
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        store = zarr.open_group(path, mode="r")
        total_segments = sum(
            int(store["line"][f"additive_{i}"].attrs["n_segments"])
            for i in range(n_levels)
        )
        assert total_segments == _N_EDGES


class TestLegalBroadcastFormsStillReachEveryLevel:
    """The negative controls: a legal broadcast still reaches every LOD level.

    A case the plain leaf accepts must not be refused by the ladder, and the
    values must land on the right elements — so each control reads every level
    back through the decoder.
    """

    def test_points_ladder_broadcast_channels(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_add_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_N, seed=51),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),  # (1, 3) row
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            additive_lod=_POINTS_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "p")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_points(f"p/additive_{i}")
            n_level = data.positions.shape[0]
            total += n_level
            assert_uniform(data.colors, [1.0, 0.0, 0.0], n_level)
            assert_uniform(data.radii, 0.5, n_level)
        assert total == _N

    def test_points_ladder_rgb_triple_on_a_three_point_node(
        self, tmp_path: Any
    ) -> None:
        """The count collision, in the ladder: 3 points, a 3-component RGB.

        Without the broadcast classification the triple satisfies
        ``slice_optional_array``'s length test and each level receives a slice of
        its COMPONENTS instead of the authored colour.
        """
        compiler, scene, path = open_scene(tmp_path, "points_add_rgb3.luxar.zarr")
        positions = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_points(
            "p", positions, colors=[0.25, 0.5, 1.0], additive_lod={"counts": [1, 3]}
        )
        compiler.finalize()

        n_levels = _n_levels(path, "p")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_points(f"p/additive_{i}")
            n_level = data.positions.shape[0]
            total += n_level
            assert_uniform(data.colors, [0.25, 0.5, 1.0], n_level)
        assert total == 3

    def test_lines_ladder_rgba_quadruple_on_a_four_vertex_node(
        self, tmp_path: Any
    ) -> None:
        """The Lines ladder's count collision — the last uncovered wrapper cell.

        4 vertices as two 2-vertex segment polylines, laddered one polyline per
        level, with a 4-component uniform RGBA whose own length equals the vertex
        count. Without the classifier in ``add_lines_multi_lod_wrapper_impl`` the
        tuple is gathered and each level gets a 2-element slice of the
        COMPONENTS: measured, that raises
        ``colors: Expected shape (2, 3) or (1, 3), got (2,)``.
        """
        compiler, scene, path = open_scene(tmp_path, "lines_add_rgba4.luxar.zarr")
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [90.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        scene.add_lines(
            "line",
            vertices,
            widths=0.2,
            colors=(0.25, 0.5, 1.0, 0.5),
            line_type="segments",
            additive_lod={"counts": [1, 2]},
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert data.colors.shape[-1] == 4
            assert_uniform(data.colors, [0.25, 0.5, 1.0, 0.5], n_level)
        assert total == 4

    def test_lines_ladder_broadcast_channels(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_add_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=52),
            widths=0.3,  # scalar
            colors=(0.25, 0.5, 1.0),  # uniform RGB
            sharpness=0.7,  # scalar
            line_type="segments",
            additive_lod=_LINES_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert_uniform(data.colors, [0.25, 0.5, 1.0], n_level)
            assert_uniform(data.widths, 0.3, n_level)
        assert total == _N

    def test_lines_ladder_broadcast_width_array(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_add_width1.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=53),
            widths=np.array([0.25], dtype=np.float32),  # (1,) broadcast
            line_type="segments",
            additive_lod=_LINES_LADDER,
        )
        compiler.finalize()

        n_levels = _n_levels(path, "line")
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            n_level = data.vertices.shape[0]
            total += n_level
            assert_uniform(data.widths, 0.25, n_level)
        assert total == _N

    @pytest.mark.parametrize("colors,want", _BROADCAST_COLORS)
    def test_points_substitutive_broadcast_channels(
        self, tmp_path: Any, colors: Any, want: Any
    ) -> None:
        # Uniform ``colors`` INCLUDED: every broadcast form the flat path accepts
        # now reaches disk under ``substitutive_lod=`` too — the lift broadcasts
        # it to the coarse gsplat levels instead of refusing it (#1444), alpha
        # and all, so every level renders at the authored opacity.
        compiler, scene, path = open_scene(tmp_path, "points_sub_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_SUB_N, seed=54),
            colors=colors,  # uniform RGB(A)
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["p"].attrs["kind"] == "lod"
        children = sorted(store["p"].group_keys())
        assert len(children) > 1
        # The finest child is the original Points node, written LAST.
        reader = LuxarScene.load(path)
        finest = children[-1]
        data = reader.get_points(f"p/{finest}")
        assert data.positions.shape[0] == _SUB_N
        assert_uniform(data.radii, 0.5, _SUB_N)
        assert_uniform(data.colors, want, _SUB_N)
        _assert_coarse_levels_carry_color(reader, "p", children[:-1], want)

    @pytest.mark.parametrize("colors,want", _BROADCAST_COLORS)
    def test_lines_substitutive_broadcast_channels(
        self, tmp_path: Any, colors: Any, want: Any
    ) -> None:
        # Colours included for the same reason as the points twin above: the
        # lift broadcasts a uniform colour onto the beads instead of gathering
        # its components as vertex rows (which raised a bare IndexError) (#1444).
        compiler, scene, path = open_scene(tmp_path, "lines_sub_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_SUB_N, seed=55),
            widths=0.3,  # scalar
            colors=colors,  # uniform RGB(A)
            sharpness=0.7,  # scalar
            line_type="segments",
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["line"].attrs["kind"] == "lod"
        children = sorted(store["line"].group_keys())
        assert len(children) > 1
        reader = LuxarScene.load(path)
        finest = children[-1]
        data = reader.get_lines(f"line/{finest}")
        assert data.vertices.shape[0] == _SUB_N
        assert_uniform(data.widths, 0.3, _SUB_N)
        assert_uniform(data.colors, want, _SUB_N)
        _assert_coarse_levels_carry_color(reader, "line", children[:-1], want)

    @pytest.mark.parametrize("shape,colors", _BAD_DTYPE_COLORS)
    @pytest.mark.parametrize("geometry", ["points", "lines"])
    def test_substitutive_bad_dtype_colors_write_nothing(
        self, tmp_path: Any, geometry: str, shape: str, colors: Any
    ) -> None:
        # A colour of a dtype the leaf refuses must be refused BEFORE the lift
        # builds anything — uniform row and per-element array alike: the coarse
        # gsplat children are written first and the finest child LAST, so
        # discovering the dtype at the encoder would strand a partial kind=lod
        # node — the #1437 stranding class.
        compiler, scene, path = open_scene(
            tmp_path, f"{geometry}_sub_dtype_{shape.replace(' ', '_')}.luxar.zarr"
        )
        add = scene.add_points if geometry == "points" else scene.add_lines
        kwargs: Dict[str, Any] = (
            {"radii": 0.5} if geometry == "points" else {"widths": 0.3}
        )
        exc = refusal(
            lambda: add(
                "n",
                random_positions(_SUB_N, seed=56),
                colors=colors,
                substitutive_lod=True,
                **kwargs,
            )
        )
        assert "dtype" in str(exc)
        store = zarr.open_group(path, mode="r")
        assert "n" not in set(store.group_keys()), (
            f"a partial node was stranded on disk: {sorted(store.group_keys())}"
        )
        compiler.finalize()
