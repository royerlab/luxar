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

The second half of this module covers the SCENE-DIMENSION COUNT gate (#1446),
which is the same bug class one validator over: the count check sat below the LOD
branches, so an additive ladder wrote ``additive_<i>`` nodes whose column count
contradicted the scene's dimensions, and the substitutive / ``lod_group=``
wrappers refused only from inside ``child_0`` — after the ``kind=lod`` group was
already on disk. Each case asserts message parity with the flat call AND an empty
store; the controls assert the hoist did not multiply the per-dimension range
``UserWarning`` across levels.

The third and last section is the same bug class one step further out (#1471),
and it is where the pattern breaks: ``labels`` / ``image_labels`` on a
multi-CHILD gsplats wrapper cannot be hoisted at all, only REFUSED. Every
substitutive level is its own set of merged representative splats with its own
count, so no single list has a per-element correspondence to slice — which is
why those cases assert a new message rather than parity with a flat one, and why
the precedence tests matter more here: the refusal has no flat counterpart, so it
must sit BELOW every check that does.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict

import numpy as np
import pytest
import zarr

from luxar.io.reader import LuxarScene

from ..conftest import (
    IMAGE_LABELS,
    LABEL_KWARGS,
    LABELS,
    N_LABELLED,
    assert_same_refusal,
    assert_uniform,
    bad_ndim_positions,
    cholesky_rows_nd,
    count_range_warnings,
    finalized_group_keys,
    open_ranged_scene,
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
    (  # opaque RGBA — above the clamp, see _assert_coarse_levels_carry_color
        (*_BROADCAST_RGB, 1.0),
        (*_BROADCAST_RGB, 1.0),
    ),
    (  # just BELOW the clamp: still bit-exact, which the tolerance split pins
        (*_BROADCAST_RGB, 0.99),
        (*_BROADCAST_RGB, 0.99),
    ),
]

#: The merge round-trips a per-splat alpha through optical depth, which caps it
#: at ``ALPHA_CLAMP = 511/512``: an authored alpha ABOVE that comes back clamped
#: on every coarse level (measured: 1.0 → 0.998046875, 0.999 → 0.998046875),
#: a step the finest child does not have. Anything at or below the clamp — RGB,
#: alpha 0.5, alpha 0.99 — is bit-exact and is asserted as such, so this
#: tolerance is reserved for the clamped case and cannot absorb a future drift
#: elsewhere.
_ALPHA_CLAMP = 511.0 / 512.0
_ALPHA_CLAMP_ATOL = 2.5e-3
_EXACT_ATOL = 1e-6

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
    Equality is EXACT unless the authored alpha exceeds the merge's
    optical-depth :data:`_ALPHA_CLAMP`, the only value the round-trip changes.
    """
    alpha = want[3] if len(want) == 4 else None
    atol = (
        _ALPHA_CLAMP_ATOL if alpha is not None and alpha > _ALPHA_CLAMP else _EXACT_ATOL
    )
    assert coarse, "no coarse gsplat levels were written"
    for child in coarse:
        data = reader.get_gsplats(f"{node}/{child}")
        assert data.colors is not None, f"{child} lost its colours"
        n = int(np.asarray(data.centers).shape[0])
        assert np.asarray(data.colors).shape[1] == len(want), (
            f"{child} carries {np.asarray(data.colors).shape[1]} channels, "
            f"expected {len(want)} (a dropped alpha renders 1/alpha too bright)"
        )
        assert_uniform(data.colors, want, n, atol=atol)


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

    def test_substitutive_one_bead_line_refuses_per_element_rgba(
        self, tmp_path: Any
    ) -> None:
        # A segment far shorter than its width lifts to exactly ONE bead, so the
        # per-bead colour array is (1, 4). Re-classifying it as the uniform form
        # would admit a per-element RGBA and bake the MEAN of two different
        # alphas — a value neither vertex has — into the coarse level. The
        # vertex-level verdict is final, so this is refused, and nothing lands.
        compiler, scene, path = open_scene(tmp_path, "one_bead_rgba.luxar.zarr")
        verts = np.array([[0, 0, 0], [0.001, 0, 0]], dtype=np.float32)
        varying = np.array([[0.2, 0.4, 0.6, 0.3], [0.2, 0.4, 0.6, 0.9]], np.float32)
        exc = refusal(
            lambda: scene.add_lines(
                "l",
                verts,
                widths=10.0,
                colors=varying,
                line_type="segments",
                substitutive_lod=True,
            )
        )
        assert "per-element" in str(exc)
        assert "l" not in set(zarr.open_group(path, mode="r").group_keys())

        # ... and the uniform twin on the very same geometry still writes, alpha
        # intact on the coarse gsplat child.
        scene.add_lines(
            "ok",
            verts,
            widths=10.0,
            colors=(*_BROADCAST_RGB, _BROADCAST_ALPHA),
            line_type="segments",
            substitutive_lod=True,
        )
        compiler.finalize()
        store = zarr.open_group(path, mode="r")
        assert store["ok"].attrs["kind"] == "lod"
        children = sorted(store["ok"].group_keys())
        _assert_coarse_levels_carry_color(
            LuxarScene.load(path),
            "ok",
            children[:-1],
            (*_BROADCAST_RGB, _BROADCAST_ALPHA),
        )

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


# ---------------------------------------------------------------------------
# Scene-dimension COUNT, pre-split (#1446)
# ---------------------------------------------------------------------------

# 400 elements is enough for the substitutive reduce to synthesise coarse levels
# (a degenerate input falls back to a flat node, which would test nothing).
_DIM_N = 400
_DIM_HALF = 200

# Cumulative ladder counts. For Lines these are POLYLINE counts: with
# line_type="segments" the 400 vertices are 200 two-vertex polylines, and the
# ladder MUST end up with more than one level — a single-polyline input
# degenerates to one level and falls through to the flat write, which refuses on
# its own and would make the case vacuous.
_DIM_POINTS_LADDER = {"counts": [_DIM_HALF, _DIM_N]}
_DIM_LINES_LADDER = {"counts": [_DIM_HALF // 2, _DIM_N // 2]}


class TestPointsAdditiveLodDimensionCount:
    def test_mismatched_ndim_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """A ladder must not write levels whose column count contradicts the scene.

        Pre-fix this call was ACCEPTED — ``refusal()`` is what catches that — and
        left ``p/additive_0``, ``p/additive_1`` on disk at ndim=4 in a
        3-dimension scene, while the same call without ``additive_lod=`` refuses.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_add_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_add_ndim_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=61)

        flat = refusal(lambda: flat_scene.add_points("p", positions))
        split = refusal(
            lambda: scene.add_points("p", positions, additive_lod=_DIM_POINTS_LADDER)
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "p" not in compiler.store


class TestLinesAdditiveLodDimensionCount:
    def test_mismatched_ndim_refused_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        """The Lines twin — 200 polylines, so the ladder really does fire.

        ``refusal()`` is the bug-catching assertion: pre-fix the ladder wrote
        every level at ndim=4.
        """
        compiler, scene, _ = open_scene(tmp_path, "lines_add_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_add_ndim_flat.luxar.zarr")
        vertices = bad_ndim_positions(_DIM_N, seed=62)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                additive_lod=_DIM_LINES_LADDER,
            )
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "line" not in compiler.store


class TestPointsSubstitutiveLodDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """Pre-fix this refused, but blamed ``child_0`` and stranded ``kind=lod``.

        ``assert_same_refusal`` fails first pre-fix — the message named
        ``child_0`` and a synthesised level's own row count — and the store
        assertion after it is an independent check on the other half of the same
        bug: pre-fix ``p`` is in the store, as a childless wrapper.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_sub_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_sub_ndim_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=63)

        flat = refusal(lambda: flat_scene.add_points("p", positions))
        split = refusal(lambda: scene.add_points("p", positions, substitutive_lod=True))

        assert_same_refusal(flat, split)
        assert "p" not in compiler.store


class TestLinesSubstitutiveLodDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """The Lines twin of the Points substitutive case; same two assertions."""
        compiler, scene, _ = open_scene(tmp_path, "lines_sub_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_sub_ndim_flat.luxar.zarr")
        vertices = bad_ndim_positions(_DIM_N, seed=64)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, line_type="segments"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                line_type="segments",
                substitutive_lod=True,
            )
        )

        assert_same_refusal(flat, split)
        assert "line" not in compiler.store


def _multi_substitutive_4d_data(n_fine: int = 8, n_coarse: int = 2) -> Any:
    """A 2-level substitutive ``GSplatData`` whose centers have FOUR columns.

    Stored levels (not a compute spec) on purpose: ``lod_group=`` then needs no
    reduce, so the pre-fix path really did reach ``add_lod_group`` and strand a
    childless ``kind=lod`` group.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    def level(n: int, seed: int, compression_factor: int, level_index: int) -> Any:
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=bad_ndim_positions(n, seed=seed),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=cholesky_rows_nd(n, 4),
                )
            ],
            compression_factor=compression_factor,
            level_index=level_index,
        )

    return GSplatData.from_substitutive_levels(
        [level(n_fine, 65, 1, 0), level(n_coarse, 66, 4, 1)]
    )


class TestGSplatsLodGroupDimensionCount:
    """``lod_group=`` is the third wrapper with the same leak (#1446).

    Both cases compare against a DIRECT ``add_gsplats`` call rather than against
    ``add_gsplats_from_data(..., lod_group=False)``: the pre-dispatch gate in
    ``from_data`` formats its ``Could not add gsplats '<name>': `` prefix by hand,
    and ``lod_group=False`` would route through that same hand-written string —
    a parity assertion comparing a string with itself, which could never catch the
    prefix drifting from what ``add_gsplats_impl``'s own funnel produces. The
    direct call is the independent witness.
    """

    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """Pre-fix: refused from inside ``child_0``, ``g`` already a ``kind=lod``.

        ``assert_same_refusal`` is what fails first pre-fix (the message named
        ``child_0`` and the coarsest level's 2-row shape). The store assertion is
        an independent check on the same bug — pre-fix ``g`` really is in the
        store, as a childless wrapper.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "gsplats_lg_ndim_flat.luxar.zarr")
        data = _multi_substitutive_4d_data()

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "g" not in compiler.store

    def test_mismatched_dim_order_length_leaves_no_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        """The ``dim_order=`` half of the same leak — the shape the docstring shows.

        With a ``dim_order`` the post-transform width is the scene's by
        construction, so the column-count check can never fire downstream; what
        raises from inside ``child_0`` is ``dim_order``'s own length-vs-columns
        check. Measured pre-fix: ``Could not add gsplats 'child_0': dim_order has
        3 names but data has 4 columns``, with ``g`` on disk as a childless
        ``kind=lod`` group. ``assert_same_refusal`` fails first pre-fix (the
        sentence is the same, the node it blames is not); the store assertion is
        an independent check on the wrapper that got written.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_dimorder.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "gsplats_lg_dimorder_flat.luxar.zarr")
        data = _multi_substitutive_4d_data()
        dim_order = ["X", "Y", "Z"]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                dim_order=dim_order,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, dim_order=dim_order
            )
        )

        assert_same_refusal(flat, split)
        assert "dim_order has 3 names but data has 4 columns" in str(split)
        assert "g" not in compiler.store

    def test_a_structural_kwarg_fault_still_outranks_the_width_fault(
        self, tmp_path: Any
    ) -> None:
        """Precedence control: ``coverage_fraction`` is refused before the width.

        The gate sits BELOW the multi-substitutive ``coverage_fraction`` refusal
        on purpose, mirroring how the three leaf adders keep their
        colours/colormap gate above their count check. A call that trips both must
        still hear about the kwarg.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_lg_cov.luxar.zarr")
        data = _multi_substitutive_4d_data()

        with pytest.raises(ValueError, match="coverage_fraction must not be passed"):
            scene.add_gsplats_from_data(
                "g", data, lod_group=True, coverage_fraction=0.5
            )

        assert "g" not in compiler.store


def _coloured_multi_substitutive_data(ndim: int) -> Any:
    """A 2-level substitutive ``GSplatData`` carrying per-splat colours."""
    return _multi_substitutive_data(
        lambda n, seed: (
            bad_ndim_positions(n, seed=seed)
            if ndim == 4
            else random_positions(n, seed=seed)
        ),
        lambda n: cholesky_rows_nd(n, ndim),
        colors_for=lambda n, _level: np.zeros((n, 3), dtype=np.float32),
    )


class TestGSplatsLodGroupColoursGate:
    """The colours/colormap exclusion is judged before the width, on both paths.

    The three leaf adders put that gate above their count check; the ``from_data``
    gate has to answer the same way or ``lod_group=`` disagrees with its own flat
    path about which fault a call that trips both is told about.
    """

    def test_a_colours_fault_outranks_a_width_fault(self, tmp_path: Any) -> None:
        """Both faults at once: the colours one wins, flat and split alike.

        Measured with the width check first (the shape this landed in briefly):
        the split path answered ``Dimension mismatch …`` where the flat path
        answered ``Cannot specify both …``. ``assert_same_refusal`` is what
        catches that.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_width.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_width_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(4)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert_same_refusal(flat, split)
        assert "both 'colors' and 'colormap'" in str(split)
        assert "g" not in compiler.store

    def test_a_colours_fault_alone_leaves_no_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        """Width perfectly fine — the same stranding class, a different fault.

        Pre-fix: ``Could not add gsplats 'child_0': Cannot specify both …`` with
        ``g`` on disk as a childless ``kind=lod`` group that survives
        ``finalize()``. The store assertion is the one that catches it here (the
        sentence is the same either way, only the node it blames differs, which
        ``assert_same_refusal`` also sees).
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_only.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_only_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(3)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert_same_refusal(flat, split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store

    def test_a_dim_order_fault_outranks_the_colours_fault(self, tmp_path: Any) -> None:
        """Three faults deep: the ``dim_order`` spec wins, flat and split alike.

        The flat path applies ``dim_order`` (spec + ``fill``, then ``fill_sigma``)
        while transforming the arrays, which is ABOVE its colours/colormap gate —
        so a call carrying colours, a ``colormap`` and a bad ``dim_order`` hears
        about the ``dim_order``. Measured with the colours gate first (the shape
        this landed in briefly): the split path answered ``Cannot specify both
        'colors' and 'colormap'`` where the flat path answered ``dim_order has 3
        names but data has 4 columns``. Together with
        :meth:`test_a_colours_fault_outranks_a_width_fault` this pins the whole
        order — dim_order spec, then colours, then width.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_dimorder.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_colour_dimorder_flat.luxar.zarr")
        data = _coloured_multi_substitutive_data(4)
        dim_order = ["X", "Y", "Z"]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                colormap="viridis",
                dim_order=dim_order,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis", dim_order=dim_order
            )
        )

        assert_same_refusal(flat, split)
        assert "dim_order has 3 names but data has 4 columns" in str(split)
        assert "g" not in compiler.store

    def test_a_colour_on_a_coarse_level_alone_is_seen_by_the_gate(
        self, tmp_path: Any
    ) -> None:
        """Only the COARSE level carries colours — the finest carries none.

        ``GSplatData.colors`` is the FINEST level's ladder merged, so a gate that
        asked only that question passed this call, and the coarsest child (written
        first) then raised ``Could not add gsplats 'child_0': Cannot specify both
        …`` with ``g`` already on disk as a childless ``kind=lod`` group. Measured
        in exactly that shape before the gate was widened to every level. No flat
        twin: a single leaf cannot express per-level colours, so the assertions
        are the blamed node and the empty store.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_colour_coarse.luxar.zarr")
        data = _multi_substitutive_data(
            lambda n, seed: random_positions(n, seed=seed),
            lambda n: cholesky_rows_nd(n, 3),
            colors_for=lambda n, level_index: (
                None if level_index == 0 else np.zeros((n, 3), dtype=np.float32)
            ),
        )
        assert data.colors is None, "the finest level must be the uncoloured one"

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, colormap="viridis"
            )
        )

        assert "both 'colors' and 'colormap'" in str(split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store


class TestRangeWarningsAreNotMultipliedByTheHoist:
    """The control: only the COUNT half was hoisted above the branches.

    ``validate_data_dimensions`` also warns once per dimension whose values fall
    outside its declared ``range``. Hoisting the WHOLE validator above the
    branches would have added one such warning per dimension for the SOURCE array
    on top of whatever each path already emits — three extra here. Both counts
    below are the measured pre-hoist numbers, so either kind of drift fails.

    ``warnings.catch_warnings`` rather than ``pytest.warns``, because the additive
    count is ZERO and ``pytest.warns`` cannot express "no warnings". That zero is
    a KNOWN GAP, not the desired end state: the multi-LOD writer never runs the
    range half at all, so a laddered node silently loses the three warnings the
    same data gets on the flat path. Closing it means hoisting the range half too
    and suppressing it per child — cross-cutting, and out of scope for #1446,
    which changed only the count half. Pinned at 0 here so the gap is visible and
    cannot widen unnoticed.
    """

    def test_points_ladder_does_not_gain_a_source_range_warning(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_ranged_scene(tmp_path, "points_add_warn.luxar.zarr")
        positions = random_positions(_DIM_N, seed=67)  # spans [0, 100), range (0, 10)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_points("p", positions, additive_lod=_DIM_POINTS_LADDER)
        compiler.finalize()

        assert count_range_warnings(records) == 0

    def test_points_substitutive_warns_once_per_dimension_per_leaf(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_ranged_scene(
            tmp_path, "points_sub_warn.luxar.zarr"
        )
        positions = random_positions(_DIM_N, seed=68)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_points("p", positions, substitutive_lod=True)
        compiler.finalize()

        # Every child is written through a leaf adder (the finest is the original
        # Points node, the coarse ones are synthesised gsplats), so each
        # contributes one warning per dimension. Asserted against the children
        # actually written rather than a bare literal, so a change in the number
        # of synthesised levels cannot silently weaken the count.
        store = zarr.open_group(path, mode="r")
        n_leaves = len(list(store["p"].group_keys()))
        assert n_leaves > 1
        assert count_range_warnings(records) == 3 * n_leaves


# The documented precedence (CHANGELOG #1446, core/group/README.md): the
# scene-dimension count is checked ABOVE the #1437 per-element channel gate, so a
# call that trips both hears about the column count — on every path. Parametrized
# over both wrapper families rather than split across the two files, because the
# claim is precisely that they all answer the same way.
_PRECEDENCE_SPLITS = [
    ("partition", {"partition": {"max_elements": _DIM_HALF}}),
    ("additive", {"additive_lod": _DIM_POINTS_LADDER}),
    ("substitutive", {"substitutive_lod": True}),
]


class TestDimensionCountPrecedesTheChannelGate:
    @pytest.mark.parametrize("case,split_kwargs", _PRECEDENCE_SPLITS)
    def test_a_wrong_length_colour_does_not_mask_the_column_count(
        self, tmp_path: Any, case: str, split_kwargs: Dict[str, Any]
    ) -> None:
        """Both faults at once: the column count is reported, flat and split alike.

        The #1437 channel gate sits at the top of each WRAPPER impl, the count
        check at the top of the leaf ADDER — so the count is reached first on the
        split paths too, and every path gives the caller the same sentence. The
        first assertion pins which fault wins on the flat path (the reference the
        other three are compared against), the second that the split paths agree.
        """
        compiler, scene, _ = open_scene(tmp_path, f"prec_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"prec_{case}_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=69)
        # Half-length per-point colours: on their own these are what the #1437
        # gate refuses.
        colors = np.zeros((_DIM_HALF, 3), dtype=np.float32)

        flat = refusal(lambda: flat_scene.add_points("p", positions, colors=colors))
        split = refusal(
            lambda: scene.add_points("p", positions, colors=colors, **split_kwargs)
        )

        assert "Dimension mismatch" in str(flat)
        assert_same_refusal(flat, split)
        assert "p" not in compiler.store


def _multi_substitutive_data(
    centers_for: Any,
    chol_for: Any,
    n_fine: int = 8,
    n_coarse: int = 2,
    colors_for: Any = None,
) -> Any:
    """A 2-level substitutive ``GSplatData`` built from two per-level factories.

    Stored levels rather than a compute spec, for the same reason as the 4-column
    twin above: ``lod_group=`` then needs no reduce, so the pre-fix path really
    did reach ``add_lod_group`` and strand a childless ``kind=lod`` group.
    """
    from luxar.gsplats.gsplat_data import (
        AdditiveSubLOD,
        GSplatData,
        SubstitutiveLevel,
    )

    def level(n: int, seed: int, compression_factor: int, level_index: int) -> Any:
        return SubstitutiveLevel(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=centers_for(n, seed),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=chol_for(n),
                    # ``colors_for`` takes the level index as well as the count so
                    # a case can colour ONE level (see the coarse-only case).
                    colors=None if colors_for is None else colors_for(n, level_index),
                )
            ],
            compression_factor=compression_factor,
            level_index=level_index,
        )

    return GSplatData.from_substitutive_levels(
        [level(n_fine, 81, 1, 0), level(n_coarse, 82, 4, 1)]
    )


def _multi_substitutive_3d_data() -> Any:
    """The well-formed 3-column twin: only the ``dim_order`` spec is at fault."""
    return _multi_substitutive_data(
        lambda n, seed: random_positions(n, seed=seed),
        lambda n: cholesky_rows_nd(n, 3),
    )


# Every refusal ``apply_dim_order`` / ``apply_dim_order_cholesky`` can reach from
# the SPEC alone — no transformed array needed — so every one of them is
# checkable before the wrapper group is created. Measured pre-fix: each named
# ``child_0`` with ``g`` already on disk as a childless ``kind=lod`` group.
_DIM_ORDER_SPEC_CASES = [
    ("duplicate_names", {"dim_order": ["X", "Y", "X"]}),
    ("name_not_in_scene", {"dim_order": ["X", "Y", "W"]}),
    ("fill_key_not_in_scene", {"dim_order": ["X", "Y", "Z"], "fill": {"Q": 0.0}}),
    ("fill_key_in_dim_order", {"dim_order": ["X", "Y", "Z"], "fill": {"X": 0.0}}),
    (
        "fill_sigma_key_not_in_scene",
        {"dim_order": ["X", "Y", "Z"], "fill_sigma": {"Q": 1.0}},
    ),
    (
        "fill_sigma_key_in_dim_order",
        {"dim_order": ["X", "Y", "Z"], "fill_sigma": {"X": 1.0}},
    ),
    # Not a sequence at all: the validator raises TypeError, which the leaf
    # adders' funnel converts to a ValueError. The pre-dispatch gate must catch
    # both or the two paths differ in exception TYPE, not just wording.
    ("dim_order_not_a_sequence", {"dim_order": 3}),
]


class TestGSplatsLodGroupDimOrderSpec:
    @pytest.mark.parametrize("case,kwargs", _DIM_ORDER_SPEC_CASES)
    def test_a_bad_spec_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, case: str, kwargs: Dict[str, Any]
    ) -> None:
        """Data is well-formed; only the ``dim_order``/``fill``/``fill_sigma`` spec is not.

        The flat reference is a direct ``add_gsplats`` with the same kwargs — the
        independent witness for the hand-written prefix (see the sibling class).
        ``assert_same_refusal`` is what fails pre-fix, on the blamed node name;
        the ``child_0`` and store assertions after it pin the two halves of the
        symptom separately.
        """
        compiler, scene, _ = open_scene(tmp_path, f"lg_spec_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lg_spec_{case}_flat.luxar.zarr")
        data = _multi_substitutive_3d_data()

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                **kwargs,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data("g", data, lod_group=True, **kwargs)
        )

        assert_same_refusal(flat, split)
        assert "child_0" not in str(split)
        assert "g" not in compiler.store


class TestGSplatsLodGroupRankGuard:
    """1-D centers: ``AdditiveSubLOD`` accepts them, so they reach the gate.

    Without the rank guard in ``validate_dimension_count`` this was a bare
    ``IndexError`` from ``shape[1]`` — which escapes the adders'
    ``except (ValueError, TypeError)`` funnels entirely.
    """

    def test_lod_group_refuses_1d_centers_exactly_as_the_flat_path(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, "lg_rank.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lg_rank_flat.luxar.zarr")
        data = _multi_substitutive_data(
            lambda n, seed: np.zeros((n,), dtype=np.float32),
            lambda n: cholesky_rows_nd(n, 3),
        )

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_data("g", data, lod_group=True))

        assert_same_refusal(flat, split)
        assert "Centers must have shape (N, D)" in str(split)
        assert "g" not in compiler.store

    def test_additive_path_refuses_1d_centers_with_a_value_error(
        self, tmp_path: Any
    ) -> None:
        """The other ``GSplatData`` path through the guard, reached per sub-LOD.

        No exact-message parity here on purpose: ``add_gsplats_multi_lod_impl``
        validates each sub-LOD in its own right, so the shape it reports is that
        sub-LOD's ``(4,)``, not the concatenated ``(8,)`` a flat call would see.
        What matters is the exception TYPE — a ``ValueError`` through the funnel,
        not the raw ``IndexError`` this used to be.
        """
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        compiler, scene, _ = open_scene(tmp_path, "add_rank.luxar.zarr")
        sublods = [
            AdditiveSubLOD(
                centers=np.zeros((4,), dtype=np.float32),
                amplitudes=np.zeros((4,), dtype=np.float32),
                cholesky_factors=cholesky_rows_nd(4, 3),
            )
            for _ in range(2)
        ]
        data = GSplatData(additive_sublods=sublods)

        with pytest.raises(ValueError, match=r"Centers must have shape \(N, D\)"):
            scene.add_gsplats_from_data("g", data)

        assert "g" not in compiler.store


# ---------------------------------------------------------------------------
# labels / image_labels on a multi-child gsplats wrapper (#1471)
# ---------------------------------------------------------------------------

# 8 finest / 2 coarsest splats — the issue's own shape. Neither channel is a
# named kwarg of ``add_gsplats_from_data``: both are named params of the LEAF
# adder only, so they arrive inside ``**attrs`` and rode into every child through
# ``child_attrs``, unsliced. The fixtures themselves live in ``../conftest`` —
# the graft half of this gate is in the partition/ sibling and states them once.
_N_FINE = N_LABELLED

# All three doors into the multi-substitutive branch. ``default`` (the kwarg
# omitted entirely, auto-lowering a stored pyramid) is the one
# ``add_gsplats_from_file`` uses for every matrix-shaped file, so it is the most
# likely real-world door and must not be the untested one. ``compute`` needs a
# FLAT input by construction: compute kwargs on data that already has stored
# levels are refused outright ("Pass recompute=True to override the stored
# pyramid"), which would make the case test that refusal instead.
_LOD_GROUP_ROUTES = [
    ("explicit", lambda: _multi_substitutive_3d_data(), {"lod_group": True}),
    ("default", lambda: _multi_substitutive_3d_data(), {}),
    (
        "compute",
        lambda: _multi_substitutive_3d_data().at_substitutive(0),
        {"lod_group": {"levels": 2, "compression_factor": 4}},
    ),
]


class TestGSplatsLodGroupRefusesLabels:
    """A substitutive ladder cannot carry per-element labels at all (#1471).

    Unlike every other case in this file the fix is a REFUSAL, not a hoist: each
    level is its own set of merged representative splats with its own count, so
    no single list has a per-element correspondence to carry.

    The two channels stranded slightly different wreckage, and the worse one is
    ``image_labels``. Pre-fix with ``labels``: ``Could not add gsplats 'child_0':
    labels: Labels length (8) must match element count (2)``, leaving ``g`` as a
    CHILDLESS ``kind=lod`` group that survives ``finalize()``. Pre-fix with
    ``image_labels``: the same shape of message, but the labels are written after
    the geometry, so ``child_0`` is left HALF-WRITTEN — ``amplitudes``,
    ``centers``, ``cholesky_factors_diag``/``_offdiag`` and ``chunk_bounds`` all
    on disk under the stranded wrapper. Both halves are asserted separately: the
    message (which named the wrong node and the wrong fault) and the store.
    """

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    @pytest.mark.parametrize("route,make_data,route_kwargs", _LOD_GROUP_ROUTES)
    def test_refused_up_front_with_nothing_written(
        self,
        tmp_path: Any,
        channel: str,
        kwargs: Dict[str, Any],
        _attr: str,
        route: str,
        make_data: Any,
        route_kwargs: Dict[str, Any],
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"lg_{channel}_{route}.luxar.zarr")
        data = make_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data("g", data, **route_kwargs, **kwargs)
        )

        assert isinstance(split, ValueError)
        # The hand-written prefix names the CALLER's node, not an internal child.
        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "child_0" not in str(split)
        assert (
            f"{channel} is not supported on a multi-level substitutive pyramid "
            "(auto-lowered to a kind=lod group)" in str(split)
        )
        # The three halves the message must state: why, this door's own remedy,
        # and the general one. The REMEDY assertion matters as much as the
        # structure one — the two doors' constants exist so their wording cannot
        # drift, and without this a swap between them passes silently.
        assert "no single list has a per-element correspondence" in str(split)
        assert "lod_group=False" in str(split)
        assert "add_lod_group()" in str(split)
        assert "g" not in compiler.store
        # And no stranded wrapper — childless or half-written — survives finalize.
        assert "g" not in finalized_group_keys(compiler, path)

    def test_labels_outranks_image_labels_when_both_are_passed(
        self, tmp_path: Any
    ) -> None:
        """Deterministic tie-break: ``labels`` is asked first, so it is reported.

        Both are refused for the identical reason, so which one is named is
        arbitrary on the merits — but it must not be arbitrary in practice. The
        order is the leaf adder's signature order (``labels`` then
        ``image_labels``).
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_both_labels.luxar.zarr")
        data = _multi_substitutive_3d_data()

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                data,
                lod_group=True,
                labels=LABELS,
                image_labels=IMAGE_LABELS,
            )
        )

        assert "labels is not supported" in str(split)
        assert "image_labels is not supported" not in str(split)
        assert "g" not in compiler.store

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    @pytest.mark.parametrize("fault", ["dim_order", "colours", "column_count"])
    def test_a_flat_parity_fault_still_outranks_the_labels_refusal(
        self,
        tmp_path: Any,
        channel: str,
        kwargs: Dict[str, Any],
        _attr: str,
        fault: str,
    ) -> None:
        """Precedence control: the refusal is LAST in the gate, and must stay there.

        Every other check in the gate mirrors a fault the FLAT path reports at
        that position; this one has no flat counterpart at all (the flat path
        ACCEPTS ``labels`` and validates it last, in the writer sweep). Ranked any
        higher it would answer ``labels is not supported …`` where the flat path
        answers something else, silently changing what the #1446 parity assertions
        elsewhere in this file mean.

        Parametrized over all THREE preceding checks — the ``dim_order`` spec, the
        colours/colormap exclusion and the column count — because pinning only one
        of them leaves a mutant that hops the labels raise above the other two
        alive. (The rank guard is the fourth and cannot be combined: 1-D centers
        make every other fault unreachable.)
        """
        compiler, scene, _ = open_scene(tmp_path, f"lg_{channel}_{fault}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lg_{channel}_{fault}_flat.luxar.zarr")
        if fault == "dim_order":
            data = _multi_substitutive_3d_data()
            extra: Dict[str, Any] = {"dim_order": ["X", "Y", "X"]}
            expected = "dim_order has duplicate names"
        elif fault == "colours":
            data = _coloured_multi_substitutive_data(3)
            extra = {"colormap": "viridis"}
            expected = "Cannot specify both 'colors' and 'colormap'"
        else:
            data = _multi_substitutive_4d_data()
            extra = {}
            expected = "centers array has 4 columns"

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=data.centers,
                amplitudes=data.amplitudes,
                cholesky_factors=data.cholesky_factors,
                colors=data.colors,
                **extra,
                **kwargs,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, lod_group=True, **extra, **kwargs
            )
        )

        assert_same_refusal(flat, split)
        assert expected in str(split)
        assert "is not supported on a multi-level substitutive" not in str(split)
        assert "g" not in compiler.store

    def test_a_structural_kwarg_fault_outranks_the_labels_refusal(
        self, tmp_path: Any
    ) -> None:
        """``coverage_fraction`` is refused ABOVE the whole gate, labels included.

        The documented precedence in ``add_gsplats_from_data``'s docstring, with
        no test until now.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_cov_labels.luxar.zarr")
        data = _multi_substitutive_3d_data()

        with pytest.raises(ValueError, match="coverage_fraction must not be passed"):
            scene.add_gsplats_from_data(
                "g", data, lod_group=True, coverage_fraction=0.5, labels=LABELS
            )

        assert "g" not in compiler.store

    def test_an_empty_labels_list_is_refused_rather_than_read_as_absent(
        self, tmp_path: Any
    ) -> None:
        """``[]`` is not None, so it is a labels REQUEST — and an impossible one."""
        compiler, scene, _ = open_scene(tmp_path, "lg_empty_labels.luxar.zarr")

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", _multi_substitutive_3d_data(), lod_group=True, labels=[]
            )
        )

        assert "labels is not supported" in str(split)
        assert "g" not in compiler.store

    def test_a_numpy_image_labels_array_is_refused_like_a_list(
        self, tmp_path: Any
    ) -> None:
        """``image_labels`` accepts several container types; the gate is truthy-free.

        An ``is not None`` test rather than a truth test, so a numpy array — whose
        ``__bool__`` raises on more than one element — cannot slip past.
        """
        compiler, scene, _ = open_scene(tmp_path, "lg_np_image_labels.luxar.zarr")

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g",
                _multi_substitutive_3d_data(),
                lod_group=True,
                image_labels=np.zeros((_N_FINE, 2, 2, 3), dtype=np.uint8),
            )
        )

        assert "image_labels is not supported" in str(split)
        assert "g" not in compiler.store


class TestAnExplicitNoneMeansNoLabels:
    """``labels=None`` must be indistinguishable from omitting it (#1471).

    ``labels=maybe_labels`` is an idiomatic call form, and the gate correctly
    reads None as "absent" — but the KEY survived in ``**attrs`` and rode into
    ``child_attrs``. ``validate_render_attrs`` rejects an unknown key by NAME and
    never looks at its value, so any child that took the additive-ladder writer
    (``write_gsplat_leaf_subtree``) raised ``Unknown node attribute 'labels'``
    from inside ``child_0`` with the ``kind=lod`` wrapper already on disk —
    exactly the strand #1471 is about, reached by a call that asked for no labels
    at all. Stock ``gsplat lod --recipe levels`` output hits this without any
    ``additive_lod=`` of its own, since its per-level stream ladders are on by
    default.

    Only the ``with_ladder`` params are true pre-fix failures. The ``plain`` two
    are CONTROLS: a single-sublod child is written by ``Group.add_gsplats``,
    which binds ``labels`` as a named param, so the stray key never reaches an
    attr validator and they pass with the strip removed. They are here so the
    normalisation cannot be "fixed" by making the plain path refuse instead.
    """

    @pytest.mark.parametrize("channel", ["labels", "image_labels"])
    @pytest.mark.parametrize(
        "case,extra",
        [("with_ladder", {"additive_lod": {"n_lods": 2}}), ("plain", {})],
    )
    def test_none_writes_the_whole_ladder_normally(
        self, tmp_path: Any, channel: str, case: str, extra: Dict[str, Any]
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"none_{channel}_{case}.luxar.zarr"
        )

        scene.add_gsplats_from_data(
            "g",
            _multi_substitutive_3d_data(),
            lod_group=True,
            **extra,
            **{channel: None},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "lod"
        # The bug's signature was a wrapper with NO children at all.
        assert sorted(store["g"].group_keys()) == ["child_0", "child_1"]


class TestLabelsStillWorkWhereTheyAlwaysDid:
    """The negatives: only MULTI-child gsplats wrappers are closed (#1471)."""

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_lod_group_false_collapses_to_a_labelled_finest_leaf(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        """``lod_group=False`` writes ONE leaf carrying all 8 splats — still labelled.

        This is the escape hatch the refusal names, so it has to keep working.
        """
        compiler, scene, path = open_scene(tmp_path, f"lg_false_{channel}.luxar.zarr")
        data = _multi_substitutive_3d_data()

        scene.add_gsplats_from_data("g", data, lod_group=False, **kwargs)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") != "lod"
        assert store["g"].attrs[attr] is True

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_a_single_level_gsplatdata_is_untouched(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        """A one-level ``GSplatData`` never enters the multi-substitutive branch."""
        from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData

        compiler, scene, path = open_scene(tmp_path, f"single_{channel}.luxar.zarr")
        data = GSplatData(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=random_positions(_N_FINE, seed=71),
                    amplitudes=np.ones(_N_FINE, dtype=np.float32),
                    cholesky_factors=cholesky_rows_nd(_N_FINE, 3),
                )
            ]
        )

        scene.add_gsplats_from_data("g", data, **kwargs)
        compiler.finalize()

        assert zarr.open_group(path, mode="r")["g"].attrs[attr] is True


class TestTheGsplatsAdditiveLadderStillHasNoLabelsChannel:
    """Not a #1471 door: this path already refused, and for a different reason."""

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_the_additive_ladder_keeps_its_own_pre_existing_answer(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """``additive_lod=`` on gsplats has NO labels channel — and still says so.

        ``write_gsplat_leaf_subtree`` documents labels as a leaf-only scene
        feature that stays on ``write_gsplats``, so the ladder path answers with
        the unknown-attr refusal (the ladder-union label support of
        ``validate_ladder_labels`` is Points/Lines only). #1471 did not touch that
        path; this pins that it did not drift into the new wording either.
        """
        compiler, scene, _ = open_scene(tmp_path, f"add_{channel}.luxar.zarr")
        data = _multi_substitutive_3d_data().at_substitutive(0)

        split = refusal(
            lambda: scene.add_gsplats_from_data(
                "g", data, additive_lod={"n_lods": 2}, **kwargs
            )
        )

        assert f"Unknown node attribute '{channel}'" in str(split)
        assert "is not supported on a multi-level substitutive" not in str(split)
        assert "g" not in compiler.store
