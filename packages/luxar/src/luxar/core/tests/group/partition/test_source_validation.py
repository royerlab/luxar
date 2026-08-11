"""Per-element channels are validated against the SOURCE count, pre-split (#1437).

``slice_optional_array`` deliberately passes a value through UNCHANGED when its
leading length does not match the element count — that is how a broadcast RGB
triple or a scalar radius reaches every part. A per-element array of the WRONG
length takes the same branch, so every part receives the whole unsliced array;
and when a part's own element count happens to equal that array's length, the
part's writer accepts it. The write then succeeds with values paired to the wrong
elements.

Every wrong-length case below asserts PARITY in the strong sense: the same input
is run with and without the wrapper, and the split path must raise the same
exception TYPE with a byte-identical message. That is assertable because both
paths now run the same validator (see ``assert_same_refusal``), and it is what
pins the property the fix is for — a substring match on the channel name would
survive a wording divergence, and ``match="colors"`` would even be satisfied by
the unrelated colors/colormap conflict. #1422 closed this for ``labels``; these
cover the rest.

The LOD wrappers live in ``tests/group/lod/test_source_validation.py``.

The last section covers the SCENE-DIMENSION COUNT gate (#1446) for the partition
wrapper: the count check sat BELOW the partition branch, so a mismatched-ndim call
was refused only from inside ``part_0`` — blaming a child the caller never wrote
and leaving ``kind=partition`` on disk with no children, where the flat path
writes nothing at all. The LOD half of that gate is in the lod/ sibling.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import pytest
import zarr

from luxar.io.reader import LuxarScene

from ..conftest import (
    assert_same_refusal,
    assert_uniform,
    bad_ndim_positions,
    cholesky_rows,
    cholesky_rows_nd,
    count_range_warnings,
    open_ranged_scene,
    open_scene,
    random_positions,
    refusal,
)

# 200 elements split at max_elements=100 gives exactly two parts of 100
# (measured), so a 100-long channel matches every part's own count and only a
# check against the full 200 can reject it.
_N = 200
_HALF = 100

# (channel name, wrong-length value)
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

_GSPLAT_CASES = [
    ("amplitudes", np.full(_HALF, 1.0, dtype=np.float32)),
    ("cholesky_factors", cholesky_rows(_HALF)),
    ("colors", np.zeros((_HALF, 3), dtype=np.float32)),
]


def _points_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    # Assigned last and explicitly, never as a duplicate dict-literal key: a
    # reordered literal would silently stop overriding the default and the case
    # would test nothing.
    kwargs[channel] = value
    return kwargs


def _lines_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"widths": 0.2}
    if channel == "scalars":
        kwargs["colormap"] = "viridis"
    kwargs[channel] = value
    return kwargs


def _gsplat_kwargs(channel: str, value: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "amplitudes": 1.0,
        "cholesky_factors": cholesky_rows(_N),
    }
    kwargs[channel] = value
    return kwargs


def _part_names(path: str, node: str) -> List[str]:
    store = zarr.open_group(path, mode="r")
    assert store[node].attrs["kind"] == "partition"
    return sorted(store[node].group_keys())


class TestPointsPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value", _POINTS_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"points_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"points_flat_{channel}.luxar.zarr")
        positions = random_positions(_N, seed=11)

        flat = refusal(
            lambda: flat_scene.add_points(
                "p", positions, **_points_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_points(
                "p",
                positions,
                partition={"max_elements": _HALF},
                **_points_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "p" not in compiler.store


class TestLinesPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value", _LINES_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"lines_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"lines_flat_{channel}.luxar.zarr")
        vertices = random_positions(_N, seed=12)

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
                partition={"max_elements": _HALF},
                **_lines_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert channel in str(split)
        assert "line" not in compiler.store


class TestGSplatsPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value", _GSPLAT_CASES)
    def test_wrong_length_channel_is_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, channel: str, value: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"gsplats_{channel}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"gsplats_flat_{channel}.luxar.zarr")
        centers = random_positions(_N, seed=13)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g", centers=centers, **_gsplat_kwargs(channel, value)
            )
        )
        split = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                partition={"max_elements": _HALF},
                **_gsplat_kwargs(channel, value),
            )
        )

        assert_same_refusal(flat, split)
        assert "g" not in compiler.store

    def test_colors_plus_colormap_leaves_no_partial_wrapper(
        self, tmp_path: Any
    ) -> None:
        """The exclusion is checked ABOVE the partition branch, like the siblings.

        Checked after it, the refusal came from inside ``part_0`` and left the
        store holding ``g`` as a childless ``kind=partition`` group; the flat path
        refuses the same call and writes nothing.
        """
        compiler, scene, _ = open_scene(tmp_path, "gsplats_colormap_clash.luxar.zarr")

        with pytest.raises(ValueError, match="both 'colors' and 'colormap'"):
            scene.add_gsplats(
                "g",
                centers=random_positions(_N, seed=14),
                amplitudes=1.0,
                cholesky_factors=cholesky_rows(_N),
                colors=np.zeros((_N, 3), dtype=np.float32),
                colormap="viridis",
                partition={"max_elements": _HALF},
            )

        assert "g" not in compiler.store


# 24 vertices as 12 edges: an (8, 3) array has the same 24 elements but a layout
# the flat writer refuses, and the split paths' topology builder would reshape it
# into 12 edges the author never wound.
_N_IDX = 24
_N_EDGES = 12
_BAD_LAYOUT = np.arange(_N_IDX, dtype=np.uint32).reshape(8, 3)
_ODD_FLAT = np.arange(_N_IDX - 1, dtype=np.uint32)
_INDEX_CASES = [
    ("bad_layout", _BAD_LAYOUT),
    ("odd_flat", _ODD_FLAT),
]
_LEGAL_LAYOUTS = [
    ("pairs", np.arange(_N_IDX, dtype=np.uint32).reshape(-1, 2)),
    ("flat", np.arange(_N_IDX, dtype=np.uint32)),
]


class TestLinesIndicesPartitionValidation:
    """``indices`` is the same bug class one channel over (#1437 follow-up).

    ``identify_polylines`` checks dtype and bounds and then reshapes to pairs, so
    a malformed edge list was reinterpreted rather than refused. Topology is
    validated FIRST now, before the channels, exactly as mesh validates ``faces``
    before any per-vertex channel. The additive / substitutive halves of this are
    in the lod/ sibling file.
    """

    @pytest.mark.parametrize("case,indices", _INDEX_CASES)
    def test_malformed_indices_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, case: str, indices: Any
    ) -> None:
        compiler, scene, _ = open_scene(tmp_path, f"idx_part_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"idx_flat_{case}.luxar.zarr")
        vertices = random_positions(_N_IDX, seed=15)

        flat = refusal(
            lambda: flat_scene.add_lines(
                "line", vertices, widths=0.2, indices=indices, line_type="indexed"
            )
        )
        split = refusal(
            lambda: scene.add_lines(
                "line",
                vertices,
                widths=0.2,
                indices=indices,
                line_type="indexed",
                partition={"max_elements": 6},
            )
        )

        assert_same_refusal(flat, split)
        assert "line" not in compiler.store

    @pytest.mark.parametrize("layout,indices", _LEGAL_LAYOUTS)
    def test_legal_layouts_still_partition_with_every_edge_intact(
        self, tmp_path: Any, layout: str, indices: Any
    ) -> None:
        """The control: both documented layouts partition, and keep 12 edges.

        Edge count is the assertion that matters — the bug this gate exists for
        is an ``(E, 3)`` array reinterpreted as ``3E/2`` edges the author never
        wound, which a mere "the node exists" check cannot see (nor can it see
        the node collapsing to a flat leaf).
        """
        compiler, scene, path = open_scene(tmp_path, f"idx_ok_part_{layout}.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N_IDX, seed=16),
            widths=0.2,
            indices=indices,
            line_type="indexed",
            partition={"max_elements": 6},
        )
        compiler.finalize()

        parts = _part_names(path, "line")
        assert len(parts) > 1
        store = zarr.open_group(path, mode="r")
        total_segments = sum(int(store["line"][p].attrs["n_segments"]) for p in parts)
        assert total_segments == _N_EDGES


class TestLegalBroadcastFormsStillReachEveryPart:
    """The negative controls: every documented broadcast form still writes.

    The gate re-runs the flat writer's validators, which accept these; the
    collision cases additionally pin the CLASSIFICATION fix — a uniform value
    whose own length equals the element count must not be gathered as if it were
    per-element data. Each control reads the values back through the decoder, so
    it pins that they landed on the right elements rather than only "no raise".
    """

    def test_per_element_channel_stays_paired_with_its_own_element(
        self, tmp_path: Any
    ) -> None:
        """Regression control for the headline property: right-length, DISTINCT values.

        Every other assertion here uses a uniform value, where mis-pairing is
        unobservable by construction. Here each point's colour IS its position
        (scaled), so reading a part back and comparing colour against the stored
        position asserts the pairing directly — parts are spatially permuted, so
        this is a join on the data rather than on the row order. Passes on main
        too: it guards the property the gate protects, it does not reproduce the
        bug.
        """
        compiler, scene, path = open_scene(tmp_path, "points_paired.luxar.zarr")
        positions = random_positions(_N, seed=25)
        colors = (positions / 100.0).astype(np.float32)
        scene.add_points(
            "p", positions, colors=colors, partition={"max_elements": _HALF}
        )
        compiler.finalize()

        parts = _part_names(path, "p")
        assert len(parts) == 2
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_points(f"p/{part}")
            seen += data.positions.shape[0]
            # uint8 colour quantization is the only slack: 1/255 ≈ 0.004.
            np.testing.assert_allclose(
                np.asarray(data.colors, dtype=np.float64),
                np.asarray(data.positions, dtype=np.float64) / 100.0,
                atol=6e-3,
            )
        assert seen == _N

    def test_points_scalar_and_broadcast_channels(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_N, seed=21),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),  # (1, 3) row
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        parts = _part_names(path, "p")
        assert len(parts) == 2
        reader = LuxarScene.load(path)
        for part in parts:
            data = reader.get_points(f"p/{part}")
            assert data.positions.shape[0] == _HALF
            # The (1, c) rows broadcast: every point of every part is pure red.
            assert_uniform(data.colors, [1.0, 0.0, 0.0], _HALF)
            assert_uniform(data.radii, 0.5, _HALF)

    def test_points_rgb_triple_on_a_three_point_node(self, tmp_path: Any) -> None:
        """The count collision: 3 points, a 3-component uniform RGB.

        Without the broadcast classification the triple satisfies
        ``slice_optional_array``'s length test and is gathered as if its three
        COMPONENTS were three point rows: each part then receives a 1-element
        slice of them (a list RGB is refused outright by the part's writer; a
        tuple RGB reaches it as a bogus 1-element color array).
        """
        compiler, scene, path = open_scene(tmp_path, "points_rgb3.luxar.zarr")
        positions = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_points(
            "p", positions, colors=[0.25, 0.5, 1.0], partition={"max_elements": 1}
        )
        compiler.finalize()

        parts = _part_names(path, "p")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_points(f"p/{part}")
            seen += data.positions.shape[0]
            assert_uniform(data.colors, [0.25, 0.5, 1.0], data.positions.shape[0])
        assert seen == 3

    def test_points_rgba_quadruple_on_a_four_point_node(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "points_rgba4.luxar.zarr")
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [30.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        scene.add_points(
            "p",
            positions,
            colors=(0.25, 0.5, 1.0, 0.5),
            partition={"max_elements": 1},
        )
        compiler.finalize()

        parts = _part_names(path, "p")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_points(f"p/{part}")
            seen += data.positions.shape[0]
            assert data.colors.shape[-1] == 4
            assert_uniform(data.colors, [0.25, 0.5, 1.0, 0.5], data.positions.shape[0])
        assert seen == 4

    def test_points_one_element_broadcast_arrays(self, tmp_path: Any) -> None:
        """A 1-element node with ``(1,)`` broadcast arrays.

        The other side of the collision: here the broadcast length and the
        element count agree, so slicing IS the right answer and must keep
        working (one part, no split fires).
        """
        compiler, scene, path = open_scene(tmp_path, "points_one.luxar.zarr")
        scene.add_points(
            "p",
            np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
            colors=np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
            radii=np.array([0.7], dtype=np.float32),
            sharpness=np.array([0.6], dtype=np.float32),
            partition={"max_elements": 1},
        )
        compiler.finalize()

        data = LuxarScene.load(path).get_points("p")
        assert data.positions.shape[0] == 1
        assert_uniform(data.colors, [0.0, 1.0, 0.0], 1)
        assert_uniform(data.radii, 0.7, 1)

    def test_lines_scalar_width_and_uniform_color(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=22),
            widths=0.3,  # scalar
            colors=(0.25, 0.5, 1.0),  # uniform RGB
            sharpness=0.7,  # scalar
            line_type="segments",
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        parts = _part_names(path, "line")
        assert len(parts) == 2
        reader = LuxarScene.load(path)
        for part in parts:
            data = reader.get_lines(f"line/{part}")
            assert data.vertices.shape[0] == _HALF
            assert_uniform(data.colors, [0.25, 0.5, 1.0], _HALF)
            assert_uniform(data.widths, 0.3, _HALF)

    def test_lines_rgba_quadruple_on_a_four_vertex_node(self, tmp_path: Any) -> None:
        """The Lines count collision: 4 vertices, a 4-component uniform RGBA.

        Two 2-vertex segments, one per part, so the RGBA tuple's own length
        matches the vertex count and ``slice_optional_array`` gathered it.
        """
        compiler, scene, path = open_scene(tmp_path, "lines_rgba4.luxar.zarr")
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
            partition={"max_elements": 2},
        )
        compiler.finalize()

        parts = _part_names(path, "line")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_lines(f"line/{part}")
            seen += data.vertices.shape[0]
            assert data.colors.shape[-1] == 4
            assert_uniform(data.colors, [0.25, 0.5, 1.0, 0.5], data.vertices.shape[0])
        assert seen == 4

    def test_lines_broadcast_width_array(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "lines_width1.luxar.zarr")
        scene.add_lines(
            "line",
            random_positions(_N, seed=23),
            widths=np.array([0.25], dtype=np.float32),  # (1,) broadcast
            line_type="segments",
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        parts = _part_names(path, "line")
        assert len(parts) == 2
        reader = LuxarScene.load(path)
        for part in parts:
            data = reader.get_lines(f"line/{part}")
            assert data.vertices.shape[0] == _HALF
            assert_uniform(data.widths, 0.25, _HALF)

    def test_gsplats_uniform_cholesky_on_a_six_splat_node(self, tmp_path: Any) -> None:
        """The other count collision: 6 splats, a uniform ``(6,)`` Cholesky.

        For 3-D data ``k = D(D+1)/2 = 6``, so the uniform form's own length
        equals the splat count and ``slice_optional_array`` gathered it as if it
        were per-splat — each part then getting a slice of the six Cholesky
        COMPONENTS as its per-splat rows.
        """
        compiler, scene, path = open_scene(tmp_path, "gsplats_uniform_chol.luxar.zarr")
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [20.0, 0.0, 0.0],
                [40.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [80.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        uniform = np.array([2.0, 0.0, 2.0, 0.0, 0.0, 2.0], dtype=np.float32)
        scene.add_gsplats(
            "g",
            centers=centers,
            amplitudes=1.0,  # scalar broadcast
            cholesky_factors=uniform,
            partition={"max_elements": 2},
        )
        compiler.finalize()

        parts = _part_names(path, "g")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_gsplats(f"g/{part}")
            n_splats = data.centers.shape[0]
            seen += n_splats
            # Every splat in every part must carry the authored uniform factor.
            chol = np.asarray(data.cholesky_factors).reshape(-1, 6)
            assert chol.shape[0] in (1, n_splats)
            for row in chol:
                np.testing.assert_allclose(row, uniform, rtol=1e-3, atol=1e-3)
            assert_uniform(data.amplitudes, 1.0, n_splats)
        assert seen == 6

    def test_gsplats_rgb_triple_on_a_three_splat_node(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "gsplats_rgb3.luxar.zarr")
        centers = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_gsplats(
            "g",
            centers=centers,
            amplitudes=1.0,
            cholesky_factors=cholesky_rows(3),
            colors=(0.25, 0.5, 1.0),
            partition={"max_elements": 1},
        )
        compiler.finalize()

        parts = _part_names(path, "g")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        seen = 0
        for part in parts:
            data = reader.get_gsplats(f"g/{part}")
            seen += data.centers.shape[0]
            assert_uniform(data.colors, [0.25, 0.5, 1.0], data.centers.shape[0])
        assert seen == 3

    def test_points_colormap_name_with_per_point_scalars(self, tmp_path: Any) -> None:
        """A colormap NAME is an attr, not a channel — it rides to every part."""
        compiler, scene, path = open_scene(tmp_path, "points_colormap.luxar.zarr")
        scene.add_points(
            "p",
            random_positions(_N, seed=24),
            scalars=np.linspace(0.0, 1.0, _N).astype(np.float32),
            colormap="viridis",
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        for part in store["p"].group_keys():
            assert store["p"][part].attrs["colormap"] == "viridis"
            assert store["p"][part]["scalars"].shape[0] == _HALF


# ---------------------------------------------------------------------------
# Scene-dimension COUNT, pre-split (#1446)
# ---------------------------------------------------------------------------

_DIM_N = 400
_DIM_HALF = 200


class TestPointsPartitionDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """Pre-fix: refused, but as ``part_0`` and with ``p`` already on disk.

        ``assert_same_refusal`` fails first pre-fix (the message named ``part_0``
        and that part's own row count). The store assertion after it is an
        independent check on the other half of the same bug — pre-fix ``p`` is in
        the store, as a childless ``kind=partition`` group.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_part_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "points_part_ndim_flat.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=71)

        flat = refusal(lambda: flat_scene.add_points("p", positions))
        split = refusal(
            lambda: scene.add_points(
                "p", positions, partition={"max_elements": _DIM_HALF}
            )
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "p" not in compiler.store


class TestLinesPartitionDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """The Lines twin of the Points case above; same two assertions."""
        compiler, scene, _ = open_scene(tmp_path, "lines_part_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "lines_part_ndim_flat.luxar.zarr")
        vertices = bad_ndim_positions(_DIM_N, seed=72)

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
                partition={"max_elements": _DIM_HALF},
            )
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "line" not in compiler.store


class TestGSplatsPartitionDimensionCount:
    def test_mismatched_ndim_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        """The GSplats twin of the Points case above; same two assertions."""
        compiler, scene, _ = open_scene(tmp_path, "gsplats_part_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "gsplats_part_ndim_flat.luxar.zarr")
        centers = bad_ndim_positions(_DIM_N, seed=73)
        # 10-wide Cholesky rows: the packed width belongs to the CENTERS' width,
        # so a 6-wide row would make the writer complain about the wrong thing.
        chol = cholesky_rows_nd(_DIM_N, 4)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g", centers=centers, amplitudes=1.0, cholesky_factors=chol
            )
        )
        split = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                partition={"max_elements": _DIM_HALF},
            )
        )

        assert_same_refusal(flat, split)
        assert "4 columns" in str(split)
        assert "g" not in compiler.store


class TestPartitionRangeWarningsAreNotMultipliedByTheHoist:
    """The control: only the COUNT half of the validator was hoisted.

    Hoisting the whole ``validate_data_dimensions`` above the partition branch
    would have added one out-of-range ``UserWarning`` per dimension for the SOURCE
    array on top of the once-per-part it already fires — three extra here. The
    count is asserted against the parts actually written, so it stays exact
    without being a bare literal. ``warnings.catch_warnings`` rather than
    ``pytest.warns``, matching the lod/ sibling control.
    """

    def test_points_partition_warns_once_per_dimension_per_part(
        self, tmp_path: Any
    ) -> None:
        compiler, scene, path = open_ranged_scene(
            tmp_path, "points_part_warn.luxar.zarr"
        )
        positions = random_positions(_DIM_N, seed=74)  # spans [0, 100), range (0, 10)

        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            scene.add_points("p", positions, partition={"max_elements": _DIM_HALF})
        compiler.finalize()

        parts = _part_names(path, "p")
        assert len(parts) > 1
        assert count_range_warnings(records) == 3 * len(parts)
