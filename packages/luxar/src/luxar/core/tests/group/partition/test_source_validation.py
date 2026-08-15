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
from typing import Any, Dict, List, Sequence

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
    cholesky_rows,
    cholesky_rows_nd,
    count_range_warnings,
    finalized_group_keys,
    grid_mesh,
    int64_rgb,
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

    def test_width_outranks_a_malformed_partition_spec(self, tmp_path: Any) -> None:
        """The deliberate precedence change the hoist forces, pinned.

        The ``partition=`` spec is validated INSIDE the branch, so before #1446 a
        call that got both the spec and the column count wrong heard about the
        spec; the count check now sits above the branch, so the width answers
        first. Both refuse and neither writes — only the message differs — but
        the ordering follows from where the check had to go, so state it here
        rather than let a reader discover it from a surprising message.
        """
        compiler, scene, _ = open_scene(tmp_path, "points_part_two_faults.luxar.zarr")
        positions = bad_ndim_positions(_DIM_N, seed=75)

        split = refusal(
            lambda: scene.add_points("p", positions, partition={"rule": "bogus"})
        )

        assert "4 columns" in str(split)
        assert "bogus" not in str(split)
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


def _nested_partition_tree(ndim: int, counts: Sequence[int] = (8, 6)) -> Any:
    """A ``kind=partition`` root of ``counts`` leaves — deliberately NOT matrix-shaped.

    ``add_gsplats_from_file`` sends every MATRIX-shaped tree (a bare leaf, an
    additive ladder, or a ``kind=lod`` of leaves) down
    ``add_gsplats_from_data_impl``, so only a genuinely nested tree like this one
    reaches ``graft_gsplat_node`` — which is the door that used to strand.

    ``counts`` is parametrized for the #1471 labels suite below, which needs a
    part pair whose sizes COLLIDE with the label count (the silent-mis-write
    case) and a ONE-part partition (which must keep labelling normally). The
    default reproduces the original two-leaf ``(8, 6)`` shape exactly.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

    def leaf(n: int, seed: int) -> Any:
        return GSplatLeaf(
            additive_sublods=[
                AdditiveSubLOD(
                    centers=bad_ndim_positions(n, seed=seed, ndim=ndim),
                    amplitudes=np.ones(n, dtype=np.float32),
                    cholesky_factors=cholesky_rows_nd(n, ndim),
                )
            ]
        )

    return GSplatPartition(
        children=[leaf(n, 91 + i) for i, n in enumerate(counts)],
        max_elements=int(max(counts)),
    )


class TestGraftedFileDimensionCount:
    """The file door of the same gate (#1446): a stored tree, not an in-memory one.

    Measured before this check existed: ``Could not add gsplats 'part_0':
    Dimension mismatch for 'part_0': …`` with the target name already on disk as a
    childless ``kind=partition`` group. The graft applies no ``dim_order`` (it
    refuses the kwarg outright), so the stored width must already be the scene's,
    and both container node types reject mixed-``ndim`` children at construction —
    which is why one leaf's centers can answer for the whole subtree.
    """

    def test_a_nested_tree_of_the_wrong_width_leaves_no_childless_wrapper(
        self, tmp_path: Any
    ) -> None:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import is_matrix_shaped

        node = _nested_partition_tree(4)
        # The premise of the whole case: a matrix-shaped tree would take the
        # already-covered data path instead, and the test would prove nothing
        # about the graft.
        assert not is_matrix_shaped(node)
        file_path = str(tmp_path / "nested.gsplats.zarr")
        write_gsplats_tree(file_path, node)

        compiler, scene, _ = open_scene(tmp_path, "graft_ndim.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "graft_ndim_flat.luxar.zarr")
        leaf_sub = next(iter(node.children)).additive_sublods[0]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=leaf_sub.centers,
                amplitudes=leaf_sub.amplitudes,
                cholesky_factors=leaf_sub.cholesky_factors,
            )
        )
        split = refusal(lambda: scene.add_gsplats_from_file("g", file_path))

        # Byte-identical to a direct add_gsplats of the same array — the graft
        # blamed ``part_0`` before, so this is what catches a regression.
        assert_same_refusal(flat, split)
        assert "part_0" not in str(split)
        assert "g" not in compiler.store

    def test_a_matching_width_still_grafts(self, tmp_path: Any) -> None:
        """Non-vacuity control: the same shape at the scene's width still lands."""
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        node = _nested_partition_tree(3)
        file_path = str(tmp_path / "nested_ok.gsplats.zarr")
        write_gsplats_tree(file_path, node)

        compiler, scene, path = open_scene(tmp_path, "graft_ok.luxar.zarr")
        scene.add_gsplats_from_file("g", file_path)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        assert len(list(store["g"].group_keys())) == 2


#: Every shape ``partition.resolve_partition_spec`` refuses, one per branch —
#: the same list the lod/ sibling runs against the ``lod_group=`` door. The two
#: non-dict cases matter as much as the dict ones: the leaf raises ``TypeError``
#: for them and only its funnel turns that into a ``ValueError``, so a gate that
#: let the raw type escape would diverge from the flat path in exception TYPE.
_BAD_PARTITION_SPECS = [
    ("not_a_dict", "nonsense"),
    ("an_int", 3),
    ("max_elements_zero", {"max_elements": 0}),
    ("unknown_rule", {"rule": "bogus"}),
]


class TestGraftedFilePartitionSpec:
    """The graft door's half of the #1550 partition-spec gate.

    ``partition=`` is not a kwarg this door consumes: it rides in ``child_attrs``
    down to each part's own ``add_gsplats``, where it drives that leaf's BSP
    split. So the spec was judged one level down, AFTER ``graft_gsplat_node`` had
    already built the ``kind=partition`` wrapper from the on-disk tree. Measured
    through the public file door, all four shapes below: ``Could not add gsplats
    'part_0': partition must be None, True, or dict; got str`` (and its three
    siblings), with ``g`` surviving ``finalize()`` as a childless
    ``kind=partition`` group. Identical to the ``lod_group=`` door's bug in the
    lod/ sibling, and closed the same way — one call in the slot that already
    holds this door's ``strip_absent_attr_kwargs`` / ``reject_data_owned_channels``
    pair.
    """

    def _file(self, tmp_path: Any, filename: str = "nested_part.gsplats.zarr") -> str:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / filename)
        write_gsplats_tree(file_path, _nested_partition_tree(3))
        return file_path

    @pytest.mark.parametrize("case,spec", _BAD_PARTITION_SPECS)
    def test_a_bad_spec_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, case: str, spec: Any
    ) -> None:
        file_path = self._file(tmp_path, f"nested_{case}.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, f"graft_part_{case}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"graft_part_{case}_flat.luxar.zarr")
        leaf_sub = next(iter(_nested_partition_tree(3).children)).additive_sublods[0]

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=leaf_sub.centers,
                amplitudes=leaf_sub.amplitudes,
                cholesky_factors=leaf_sub.cholesky_factors,
                partition=spec,
            )
        )
        split = refusal(
            lambda: scene.add_gsplats_from_file("g", file_path, partition=spec)
        )

        assert_same_refusal(flat, split)
        assert "part_0" not in str(split)
        assert "g" not in compiler.store
        assert finalized_group_keys(compiler, path) == set()

    def test_a_valid_spec_still_grafts_and_still_splits(self, tmp_path: Any) -> None:
        """Non-vacuity control: judging the value must not become refusing it.

        The stored tree is a 2-part partition of an 8- and a 6-splat leaf, so a
        cap of 4 makes each grafted part split again — the request really is
        honoured, not merely tolerated.
        """
        file_path = self._file(tmp_path, "nested_valid.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "graft_part_valid.luxar.zarr")

        scene.add_gsplats_from_file("g", file_path, partition={"max_elements": 4})
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        parts = sorted(store["g"].group_keys())
        assert parts == ["part_0", "part_1"]
        assert [store["g"][p].attrs.get("kind") for p in parts] == [
            "partition",
            "partition",
        ]

    def test_false_is_a_bypass_the_gate_must_not_judge(self, tmp_path: Any) -> None:
        """``False`` is vocabulary (the no-partition bypass), not a bad spec."""
        file_path = self._file(tmp_path, "nested_false.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "graft_part_false.luxar.zarr")

        scene.add_gsplats_from_file("g", file_path, partition=False)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        assert [
            store["g"][p].attrs.get("kind") for p in sorted(store["g"].group_keys())
        ] == [None, None]


# ---------------------------------------------------------------------------
# labels / image_labels on a GRAFTED subtree (#1471)
# ---------------------------------------------------------------------------
#
# The graft half of the #1471 gate; its ``lod_group=`` half is in the lod/
# sibling, which is also where the shared reasoning lives. It belongs here for
# the same reason ``TestGraftedFileDimensionCount`` above does: the shape that
# reaches ``graft_gsplat_node`` at all is a non-matrix-shaped one, i.e. a
# ``kind=partition``, and it reuses this module's ``_nested_partition_tree``.


def _graft(scene: Any, **kwargs: Any) -> Any:
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    return graft_gsplat_node(scene, **kwargs)


class TestGraftedSubtreeRefusesLabels:
    """A multi-leaf graft cannot carry per-element labels either (#1471).

    ``graft_gsplat_node`` builds its wrappers by calling ``add_lod_group`` /
    ``add_partition_group`` DIRECTLY, so it never meets the ``from_data`` gate —
    and it is the door for every shape that gate cannot see (``gsplat lod
    --recipe tiles|overview|adaptive``, ``gsplat partition``, a ``batch-fit
    merge`` kind=partition). Measured pre-fix, BOTH failure modes:

    * mismatched — ``Could not add gsplats 'part_0': labels: Labels length (8)
      must match element count (5)``, with ``g`` on disk as a childless
      ``kind=partition`` that survives ``finalize()``;
    * SILENT — a list whose length happens to equal a part's own count raised
      nothing at all and was written onto EVERY part, identical lists,
      ``has_labels: True`` on both. Worse than the crash, and ``refusal()`` is
      what catches it (the call SUCCEEDED, so the store assertion never ran).
    """

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    @pytest.mark.parametrize("counts", [(8, 6), (N_LABELLED, N_LABELLED)])
    def test_a_multi_part_graft_is_refused_before_the_wrapper_exists(
        self,
        tmp_path: Any,
        channel: str,
        kwargs: Dict[str, Any],
        _attr: str,
        counts: Sequence[int],
    ) -> None:
        tag = "x".join(str(c) for c in counts)
        compiler, scene, path = open_scene(
            tmp_path, f"graft_{channel}_{tag}.luxar.zarr"
        )

        split = refusal(
            lambda: _graft(
                scene, name="g", node=_nested_partition_tree(3, counts), **kwargs
            )
        )

        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "part_0" not in str(split)
        assert f"{channel} is not supported on a grafted multi-node" in str(split)
        # The remedy half, pinned as tightly as the structure half: the two doors'
        # constants exist so their wording cannot drift, and asserting only the
        # structure lets a swap between them pass.
        assert "gsplat flatten" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    def test_a_multi_leaf_lod_graft_is_refused_too(self, tmp_path: Any) -> None:
        """The other container kind, refused by the same leaf-count statement."""
        from luxar.gsplats.tree import GSplatLodGroup

        compiler, scene, _ = open_scene(tmp_path, "graft_lod_labels.luxar.zarr")
        node = GSplatLodGroup(children=list(_nested_partition_tree(3).children))

        split = refusal(lambda: _graft(scene, name="g", node=node, labels=LABELS))

        assert "labels is not supported on a grafted multi-node" in str(split)
        assert "g" not in compiler.store

    def test_a_one_child_wrapper_around_a_multi_leaf_one_is_refused_at_the_top(
        self, tmp_path: Any
    ) -> None:
        """Nesting smuggles nothing past the gate, and does not move the blame.

        ``iter_leaves`` recurses, so a wrapper with a single CHILD that is itself
        a multi-leaf wrapper is counted by its leaves and refused at the ENTRY
        call — naming ``g``, the caller's node, not the inner ``part_0``. Without
        this the leaf-count rule could just as well be read as a child count, and
        the recursion would be doing the refusing one level down (with the outer
        wrapper already on disk, i.e. the strand this gate exists to prevent).
        """
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        compiler, scene, path = open_scene(tmp_path, "graft_nested_labels.luxar.zarr")
        node = GSplatPartition(
            children=[
                GSplatLodGroup(children=list(_nested_partition_tree(3).children))
            ],
            max_elements=8,
        )

        split = refusal(lambda: _graft(scene, name="g", node=node, labels=LABELS))

        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "part_0" not in str(split)
        assert "labels is not supported on a grafted multi-node" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)


def _laddered_leaf(n_per_sublod: int = 4, n_sublods: int = 2) -> Any:
    """One leaf carrying an ADDITIVE LADDER — what ``gsplat lod`` writes.

    (A plain ``gsplat fit`` writes a FLAT leaf: its ``--recipe`` defaults to
    None. The ladder arrives with ``lod``, on every recipe, unless
    ``--no-additive``.)

    ``_nested_partition_tree``'s leaves are single-sub-LOD on purpose (the #1446
    cases they were built for care only about column count), so the laddered
    variant is stated here, where the difference is the point.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatLeaf

    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=random_positions(n_per_sublod, seed=95 + i),
                amplitudes=np.ones(n_per_sublod, dtype=np.float32),
                cholesky_factors=cholesky_rows_nd(n_per_sublod, 3),
            )
            for i in range(n_sublods)
        ]
    )


class TestALadderedLeafIsRefusedForItsOwnReason:
    """One leaf is not enough — it must also be FLAT (#1471).

    The gate's contract is "nothing is written when the call is refused", and a
    one-leaf exemption breaks it for a LADDERED leaf: the write goes to
    ``write_gsplat_leaf_subtree``, which has no labels channel at all, so the
    refusal used to arrive from inside ``part_0`` with the one-part wrapper
    already on disk (measured: ``Could not add gsplats 'part_0': Unknown node
    attribute 'labels'``, ``g`` a childless ``kind=partition`` surviving
    ``finalize()``). Not a corner case — ``--recipe tiles`` carries a stream
    ladder by DEFAULT, so a one-part tiles file plus ``labels=`` lands here.

    The message is deliberately NOT the shared wrapper template: there is one
    leaf holding every splat, so its "no per-element correspondence, cannot be
    sliced" argument would read as nonsense. The fault is that the writer has
    nowhere to put them.
    """

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_one_part_partition_of_a_laddered_leaf_is_refused(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        from luxar.gsplats.tree import GSplatPartition

        compiler, scene, path = open_scene(
            tmp_path, f"graft_ladder_{channel}.luxar.zarr"
        )
        node = GSplatPartition(children=[_laddered_leaf()], max_elements=8)

        split = refusal(lambda: _graft(scene, name="g", node=node, **kwargs))

        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "part_0" not in str(split)
        assert f"{channel} is not supported on a gsplats additive ladder" in str(split)
        assert "no labels channel" in str(split)
        # The wrapper template's argument must NOT be borrowed here: there is one
        # leaf holding every splat, so nothing is being sliced or mis-paired.
        assert "per-element correspondence" not in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_bare_laddered_leaf_graft_gets_the_same_answer(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """Reachable only by calling the graft directly, and its answer CHANGED.

        ``add_gsplats_from_file`` never sends a bare laddered leaf here (it is
        matrix-shaped, so it takes the data path), but ``graft_gsplat_node`` is
        called directly by tests and by the recursion. Before #1471 it answered
        ``Unknown node attribute 'labels'`` from the leaf's own write; it now
        answers with the ladder reason, from the gate. Nothing was written either
        way — a bare leaf creates no wrapper — so this is a message improvement,
        not a strand fix. Pinned so the two one-leaf shapes cannot drift apart.
        """
        compiler, scene, _ = open_scene(
            tmp_path, f"graft_bare_ladder_{channel}.luxar.zarr"
        )

        split = refusal(
            lambda: _graft(scene, name="g", node=_laddered_leaf(), **kwargs)
        )

        assert f"{channel} is not supported on a gsplats additive ladder" in str(split)
        assert "g" not in compiler.store

    def test_a_laddered_leaf_without_labels_still_grafts(self, tmp_path: Any) -> None:
        """Non-vacuity: the gate did not break the ordinary laddered graft."""
        from luxar.gsplats.tree import GSplatPartition

        compiler, scene, path = open_scene(tmp_path, "graft_ladder_ok.luxar.zarr")
        node = GSplatPartition(children=[_laddered_leaf()], max_elements=8)

        _graft(scene, name="g", node=node)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "partition"
        assert sorted(store["g"].group_keys()) == ["part_0"]
        assert int(store["g"]["part_0"].attrs["n_additive_sublods"]) == 2


class TestASingleFlatLeafGraftStillLabels:
    """The gate exempts a single FLAT leaf — leaf count AND sub-LOD count (#1471).

    A wrapper resolving to a single leaf has an exact per-element
    correspondence, because that one leaf holds every splat. ``luxar gsplat
    partition in out --parts 1`` emits exactly that shape (a one-part
    ``kind=partition`` of a flat leaf, which the graft's own partition branch
    already special-cases as "not a tiling"), and it labelled correctly before
    #1471. (``gsplat lod --recipe tiles`` on a small dataset emits the LADDERED
    variant, refused below.)
    Refusing it would be a regression, and every sentence of the wrapper message
    would be false there. The laddered variant of the same shape is refused for
    its own reason — see the sibling class.
    """

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_a_one_part_partition_still_labels(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"graft_1part_{channel}.luxar.zarr"
        )

        _graft(
            scene,
            name="g",
            node=_nested_partition_tree(3, (N_LABELLED,)),
            **kwargs,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "partition"
        assert sorted(store["g"].group_keys()) == ["part_0"]
        assert store["g"]["part_0"].attrs[attr] is True

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_a_bare_leaf_graft_still_labels(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        compiler, scene, path = open_scene(tmp_path, f"graft_leaf_{channel}.luxar.zarr")
        leaf = _nested_partition_tree(3, (N_LABELLED,)).children[0]

        _graft(scene, name="g", node=leaf, **kwargs)
        compiler.finalize()

        assert zarr.open_group(path, mode="r")["g"].attrs[attr] is True

    @pytest.mark.parametrize("channel,kwargs,attr", LABEL_KWARGS)
    def test_a_nest_that_still_resolves_to_one_leaf_labels_that_leaf(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], attr: str
    ) -> None:
        """Leaf count is counted through the nesting, in BOTH directions.

        The refusal side of that is pinned in the sibling class; this is the
        exemption side. Two wrappers deep, but still one leaf holding every splat,
        so the correspondence is exact and the labels ride down to it.
        """
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        compiler, scene, path = open_scene(
            tmp_path, f"graft_nest1_{channel}.luxar.zarr"
        )
        leaf = _nested_partition_tree(3, (N_LABELLED,)).children[0]
        node = GSplatPartition(
            children=[GSplatLodGroup(children=[leaf])], max_elements=N_LABELLED
        )

        _graft(scene, name="g", node=node, **kwargs)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"]["part_0"].attrs["kind"] == "lod"
        assert store["g"]["part_0"]["child_0"].attrs[attr] is True

    @pytest.mark.parametrize("channel", ["labels", "image_labels"])
    def test_an_explicit_none_still_grafts_a_multi_part_partition(
        self, tmp_path: Any, channel: str
    ) -> None:
        """The graft door's half of the None normalisation (see the lod/ sibling)."""
        compiler, scene, path = open_scene(tmp_path, f"graft_none_{channel}.luxar.zarr")

        _graft(scene, name="g", node=_nested_partition_tree(3), **{channel: None})
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "partition"
        assert sorted(store["g"].group_keys()) == ["part_0", "part_1"]

    def test_a_legal_sparse_image_labels_dict_still_labels_a_one_part_partition(
        self, tmp_path: Any
    ) -> None:
        """The positive control for the dict form (#1505's length check).

        The fix's only ``image_labels`` dict coverage on this exempt branch is
        a refusal (an out-of-range key, in the sibling class below) — so a
        future change that refused every dict at this gate would pass the
        whole suite without this control. Measured: ``{0: img, 4: img}`` on a
        one-part graft of an ``N_LABELLED``-splat leaf still writes normally.
        Not parametrized over ``LABEL_KWARGS``: the sparse dict form is
        ``image_labels``-only, ``labels`` has no dict spelling.

        ``has_image_labels is True`` alone cannot tell "both images landed"
        apart from "an empty CSR with the flag set", so this also reads back
        ``part_0``'s ``image_label_offsets`` (the CSR byte-offset array
        ``write_image_labels_csr`` writes — see
        ``io/_compiler/labels/image_labels.py``) and asserts exactly TWO
        non-empty entries. Measured on this exact call: ``[0 0 44 44 44 44 88
        88 88]``. The assertion is on the COUNT of non-empty entries, not
        their SLOTS (here 1 and 5), because the writer's spatial
        ``sort_order`` permutes stored position relative to original element
        index — asserting fixed slots would pin an ordering detail this test
        does not care about and could break on an unrelated ordering change.
        """
        compiler, scene, path = open_scene(tmp_path, "graft_sparse_dict_ok.luxar.zarr")

        _graft(
            scene,
            name="g",
            node=_nested_partition_tree(3, (N_LABELLED,)),
            image_labels={0: IMAGE_LABELS[0], 4: IMAGE_LABELS[0]},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs["kind"] == "partition"
        part_0 = store["g"]["part_0"]
        assert part_0.attrs["has_image_labels"] is True
        offsets = part_0["image_label_offsets"][:]
        assert np.count_nonzero(np.diff(offsets)) == 2


#: Channel-specific wording for the 8-long-list-on-a-5-splat-leaf case this
#: whole class is built around — measured: ``labels: Labels length (8) must
#: match element count (5)`` / ``Image labels length (8) must match element
#: count (5)``. Pinning the channel-specific form (not just the shared
#: ``must match element count (5)`` suffix) is what catches the fix
#: accidentally raising the WRONG refusal template (the multi-leaf
#: ``labels_on_wrapper_reason``, or ``labels_on_a_laddered_leaf_reason``) —
#: every one of which would still satisfy a bare ``'g'``-prefix /
#: no-``part_0`` / nothing-written assertion.
_WRONG_LENGTH_WORDING = {
    "labels": "labels: Labels length (8)",
    "image_labels": "Image labels length (8)",
}


def _flat_refusal_for_leaf(
    tmp_path: Any, filename: str, leaf: Any, kwargs: Dict[str, Any]
) -> Exception:
    """Refuse ``leaf``'s own three arrays via the TRUE flat path (``add_gsplats``).

    The flat control every parity assertion in the class below compares
    against — the leaf's own sub-LOD centers/amplitudes/cholesky_factors,
    added directly with the same label kwargs, exactly as
    ``test_parity_with_the_true_flat_path`` builds it. Factored to one place
    so the FOUR call sites (two direct one-part partitions —
    ``test_parity_with_the_true_flat_path``'s wrong-LENGTH case and
    ``test_a_wrongly_typed_label_entry_is_refused_too``'s wrong-CONTENT one —
    plus a bare-leaf graft and a two-wrapper-deep nest) cannot state the
    construction slightly differently and drift apart from each other.
    """
    _, flat_scene, _ = open_scene(tmp_path, filename)
    sub = leaf.additive_sublods[0]
    return refusal(
        lambda: flat_scene.add_gsplats(
            "g",
            centers=sub.centers,
            amplitudes=sub.amplitudes,
            cholesky_factors=sub.cholesky_factors,
            **kwargs,
        )
    )


class TestAWrongLengthLabelOnTheExemptLeafIsRefused:
    """The LENGTH half of the one-flat-leaf exemption (#1505).

    Measured on ``main`` (before this fix): an 8-long ``labels`` /
    ``image_labels`` grafted onto a one-part ``kind=partition`` of a 5-splat
    leaf sailed past ``_reject_labels_on_a_grafted_wrapper`` — leaf count == 1
    and sub-LOD count == 1, the exact shape it exempts — and refused one level
    down instead, from inside ``part_0``'s own leaf write: ``Could not add
    gsplats 'part_0': labels: Labels length (8) must match element count
    (5)``, with the childless ``kind=partition`` named ``g`` already on disk,
    surviving ``finalize()``. Both channels stranded identically, and the flat
    control (the same three arrays via ``scene.add_gsplats(...)``) refused
    with the byte-identical message and wrote nothing at all — that asymmetry
    is exactly what this class closes.
    """

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_one_part_partition_graft_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"graft_1part_wronglen_{channel}.luxar.zarr"
        )
        node = _nested_partition_tree(3, (5,))

        split = refusal(lambda: _graft(scene, name="g", node=node, **kwargs))

        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "part_0" not in str(split)
        # Pins the WORDING, not just the structure: all three assertions above
        # are also satisfied if the fix had raised the wrong refusal entirely
        # (the multi-leaf or laddered-leaf template).
        assert "must match element count (5)" in str(split)
        assert _WRONG_LENGTH_WORDING[channel] in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_parity_with_the_true_flat_path(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """Exact-message parity against the true flat write, not a bare-leaf graft.

        A bare-leaf graft also runs through this same gate (see
        ``TestASingleFlatLeafGraftStillLabels``'s ``test_a_bare_leaf_graft_still_labels``),
        so it is not an independent control for this assertion — see
        ``assert_same_refusal``'s own docstring on why the comparison has to be
        exact-message, not a substring.
        """
        node = _nested_partition_tree(3, (5,))
        compiler, scene, path = open_scene(
            tmp_path, f"graft_1part_wronglen_parity_{channel}.luxar.zarr"
        )
        flat = _flat_refusal_for_leaf(
            tmp_path,
            f"flat_wronglen_parity_{channel}.luxar.zarr",
            node.children[0],
            kwargs,
        )

        split = refusal(lambda: _graft(scene, name="g", node=node, **kwargs))

        assert_same_refusal(flat, split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_nest_that_still_resolves_to_one_leaf_is_refused_too(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """Two wrappers deep, but still one leaf — the length check is not
        accidentally scoped to a direct one-part partition only.

        Mirrors ``test_a_nest_that_still_resolves_to_one_leaf_labels_that_leaf``'s
        shape, at the wrong length instead of the right one. Asserted against
        the true flat control (measured byte-identical to it, both channels),
        not just the ``'g'`` prefix: a bare prefix/no-``part_0`` check passes
        whether or not this shape's length check was ever wired up, since the
        two-wrapper nest strands in the same visible way either answer arrives.
        """
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        leaf = _nested_partition_tree(3, (5,)).children[0]
        node = GSplatPartition(
            children=[GSplatLodGroup(children=[leaf])], max_elements=5
        )
        compiler, scene, path = open_scene(
            tmp_path, f"graft_nest1_wronglen_{channel}.luxar.zarr"
        )
        flat = _flat_refusal_for_leaf(
            tmp_path, f"flat_nest1_wronglen_{channel}.luxar.zarr", leaf, kwargs
        )

        split = refusal(lambda: _graft(scene, name="g", node=node, **kwargs))

        assert_same_refusal(flat, split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_bare_leaf_graft_is_refused_too(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """The other one-leaf shape cannot drift from the partitioned one.

        Nothing strands here either way (a bare leaf creates no wrapper), but
        pinning it keeps the two one-leaf shapes from disagreeing, exactly as
        ``test_a_bare_laddered_leaf_graft_gets_the_same_answer`` does for the
        ladder. A bare-prefix assertion alone would pass whether this shape's
        length check were scoped to only the direct one-part case or shared
        with it, so the message is compared exact-message against the true
        flat control (measured byte-identical, both channels) instead.
        """
        leaf = _nested_partition_tree(3, (5,)).children[0]
        compiler, scene, _ = open_scene(
            tmp_path, f"graft_bareleaf_wronglen_{channel}.luxar.zarr"
        )
        flat = _flat_refusal_for_leaf(
            tmp_path, f"flat_bareleaf_wronglen_{channel}.luxar.zarr", leaf, kwargs
        )

        split = refusal(lambda: _graft(scene, name="g", node=leaf, **kwargs))

        assert_same_refusal(flat, split)
        assert "g" not in compiler.store

    def test_a_multi_leaf_graft_still_answers_with_its_own_structural_reason(
        self, tmp_path: Any
    ) -> None:
        """Precedence: the length check sits on the EXEMPT branch, strictly
        below the multi-leaf refusal, so a wrong-length channel on a shape
        that is ALSO structurally disqualified gets the structural message,
        not a length one.
        """
        compiler, scene, _ = open_scene(tmp_path, "graft_multi_wronglen.luxar.zarr")
        node = _nested_partition_tree(3, (5, 6))  # two leaves, neither length 8

        split = refusal(lambda: _graft(scene, name="g", node=node, labels=LABELS))

        assert "labels is not supported on a grafted multi-node" in str(split)
        assert "Labels length" not in str(split)
        assert "g" not in compiler.store

    def test_a_laddered_leaf_still_answers_with_its_own_structural_reason(
        self, tmp_path: Any
    ) -> None:
        """Same precedence, for the OTHER structural refusal (the sub-LOD one).

        ``_laddered_leaf(3, 2)`` totals 6 splats, wrong-length for an 8-long
        ``labels`` — but the ladder branch is checked before the length
        helper ever runs, so its own message must win regardless.
        """
        from luxar.gsplats.tree import GSplatPartition

        compiler, scene, _ = open_scene(tmp_path, "graft_ladder_wronglen.luxar.zarr")
        node = GSplatPartition(children=[_laddered_leaf(3, 2)], max_elements=6)

        split = refusal(lambda: _graft(scene, name="g", node=node, labels=LABELS))

        assert "labels is not supported on a gsplats additive ladder" in str(split)
        assert "Labels length" not in str(split)
        assert "g" not in compiler.store

    def test_an_out_of_range_sparse_image_label_key_is_refused_too(
        self, tmp_path: Any
    ) -> None:
        """Proves the fix calls the WHOLE validator, not a bare ``len()`` check.

        A sparse ``image_labels`` dict has its OWN length (1 key) that would
        pass a naive comparison against nothing in particular; it is the KEY
        being out of ``[0, 5)`` that ``validate_image_labels_for_writing``
        catches, and only calling the real validator (not reimplementing a
        length rule) catches it here too.
        """
        node = _nested_partition_tree(3, (5,))
        compiler, scene, _ = open_scene(tmp_path, "graft_sparse_oob.luxar.zarr")

        split = refusal(
            lambda: _graft(
                scene, name="g", node=node, image_labels={7: IMAGE_LABELS[0]}
            )
        )

        assert str(split).startswith("Could not add gsplats 'g': ")
        # The index wording, not just the prefix: a naive
        # ``len(image_labels) != n_splats`` reimplementation ALSO refuses this
        # call (1 key vs 5 splats), just with a length message instead of this
        # index one — so only this assertion tells the two apart.
        assert "Image label index 7 out of range [0, 5)" in str(split)
        assert "g" not in compiler.store

    def test_a_non_integer_sparse_key_keeps_the_exception_TYPE_in_parity(
        self, tmp_path: Any
    ) -> None:
        """The ``TypeError`` half of the helper's funnel, which nothing else pins.

        ``validate_image_labels_for_writing`` raises ``TypeError`` — not
        ``ValueError`` — for a key with no integer-index semantics, so the
        helper's ``except (ValueError, TypeError)`` is what keeps the graft's
        answer the same exception TYPE the flat path's own funnel produces.
        Measured: narrowing that tuple to ``ValueError`` alone leaves every
        other test in this file green while this call escapes as a bare,
        unprefixed ``TypeError``. ``assert_same_refusal`` compares the type as
        well as the message, so it is the assertion that catches it.
        """
        node = _nested_partition_tree(3, (5,))
        kwargs = {"image_labels": {"1": IMAGE_LABELS[0]}}
        compiler, scene, _ = open_scene(tmp_path, "graft_sparse_badkey.luxar.zarr")
        flat = _flat_refusal_for_leaf(
            tmp_path, "flat_sparse_badkey.luxar.zarr", node.children[0], kwargs
        )

        split = refusal(lambda: _graft(scene, name="g", node=node, **kwargs))

        assert isinstance(split, ValueError)
        assert_same_refusal(flat, split)
        assert "Image label index must be an integer, got str" in str(split)
        assert "g" not in compiler.store

    def test_labels_is_answered_before_image_labels_when_both_are_wrong(
        self, tmp_path: Any
    ) -> None:
        """The two-channel tie-break, in the flat writer's own order.

        Every other case here passes ONE channel, so the order the helper runs
        them in — ``labels`` then ``image_labels``, matching ``write_gsplats``
        steps 0d then 0e — is otherwise unobservable and could be swapped
        silently. A call passing both wrong hears about ``labels``.
        """
        node = _nested_partition_tree(3, (5,))
        compiler, scene, _ = open_scene(tmp_path, "graft_both_wronglen.luxar.zarr")

        split = refusal(
            lambda: _graft(
                scene, name="g", node=node, labels=LABELS, image_labels=IMAGE_LABELS
            )
        )

        assert _WRONG_LENGTH_WORDING["labels"] in str(split)
        assert "Image labels length" not in str(split)
        assert "g" not in compiler.store

    def test_a_wrongly_typed_label_entry_is_refused_too(self, tmp_path: Any) -> None:
        """The ``labels`` analogue of the sparse-key content case above.

        A RIGHT-length list with a non-``str`` entry — the ``labels`` channel
        has no sparse form, so its own content fault is a bad entry TYPE
        rather than an out-of-range key. Measured: ``Could not add gsplats
        'g': labels: Label at index 2 is int, expected str or None (got
        3)``, byte-identical to the flat control (the same rule
        ``validate_labels_before_split`` enforces on every other path).
        """
        node = _nested_partition_tree(3, (5,))
        bad_labels = ["a", "b", 3, "d", "e"]
        compiler, scene, _ = open_scene(tmp_path, "graft_label_content.luxar.zarr")
        flat = _flat_refusal_for_leaf(
            tmp_path,
            "flat_label_content.luxar.zarr",
            node.children[0],
            {"labels": bad_labels},
        )

        split = refusal(lambda: _graft(scene, name="g", node=node, labels=bad_labels))

        assert_same_refusal(flat, split)
        assert "Label at index 2 is int, expected str or None (got 3)" in str(split)
        assert "g" not in compiler.store

    def test_a_wrong_length_label_outranks_an_unknown_attr_here(
        self, tmp_path: Any
    ) -> None:
        """The deliberate ordering this check's placement forces, pinned (#1505).

        This length check is the FIRST statement of ``graft_gsplat_node``, so
        for a multi-fault call it now outranks faults the flat path would
        report first. Measured: this exact call (an 8-long ``labels`` PLUS an
        unknown ``foo`` attr, on a one-part partition of a 5-splat leaf)
        answers the label-length fault here, where the true flat
        ``add_gsplats(..., labels=<8-long>, foo=1)`` answers ``Unknown node
        attribute 'foo'`` instead — both refuse, and neither writes. This is
        the SANCTIONED divergence, not a bug: see the ONE LEAF bullet of
        ``_reject_labels_on_a_grafted_wrapper``'s docstring, which names
        ``compositing.validate_points_channels_before_split``'s identical
        documented trade for "a NaN position, an unknown attr", and the
        ``lod_group=`` door (``from_data._reject_before_wrapper``) already
        making the same choice.
        """
        node = _nested_partition_tree(3, (5,))
        compiler, scene, _ = open_scene(tmp_path, "graft_precedence_attr.luxar.zarr")

        split = refusal(
            lambda: _graft(scene, name="g", node=node, labels=LABELS, foo=1)
        )

        assert "must match element count (5)" in str(split)
        assert "foo" not in str(split)
        assert "g" not in compiler.store


class TestTheGraftGateIsReachedThroughThePublicFileDoor:
    """Everything above calls ``graft_gsplat_node``; this proves the door reaches it.

    ``TestGraftedFileDimensionCount`` sets the precedent — a real
    ``.gsplats.zarr`` on disk, embedded with ``scene.add_gsplats_from_file``.
    Without a case at that level nothing pins that the PUBLIC entry point
    actually arrives at the gate, only that the private function refuses.
    """

    @staticmethod
    def _write(tmp_path: Any, ndim: int = 3, counts: Sequence[int] = (8, 6)) -> str:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import is_matrix_shaped

        node = _nested_partition_tree(ndim, counts)
        # The premise: a matrix-shaped tree would take the data path instead.
        assert not is_matrix_shaped(node)
        tag = "x".join(str(c) for c in counts)
        file_path = str(tmp_path / f"nested_{ndim}d_{tag}.gsplats.zarr")
        write_gsplats_tree(file_path, node)
        return file_path

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_add_gsplats_from_file_refuses_and_writes_nothing(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        file_path = self._write(tmp_path)
        compiler, scene, path = open_scene(tmp_path, f"file_{channel}.luxar.zarr")

        split = refusal(lambda: scene.add_gsplats_from_file("g", file_path, **kwargs))

        assert f"{channel} is not supported on a grafted multi-node" in str(split)
        assert "part_0" not in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("channel,kwargs,_attr", LABEL_KWARGS)
    def test_a_one_part_file_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, channel: str, kwargs: Dict[str, Any], _attr: str
    ) -> None:
        """The public-door counterpart of ``TestAWrongLengthLabelOnTheExemptLeafIsRefused``.

        That class calls ``graft_gsplat_node`` directly; ``luxar gsplat
        partition in out --parts 1`` is the real producer of the exact
        one-part ``kind=partition`` store the exemption covers, and its
        public door is ``add_gsplats_from_file`` — not exercised anywhere else
        in this class, which otherwise only ever writes the two-leaf
        (structurally-refused) shape. Measured with the fix neutralised:
        ``Could not add gsplats 'part_0': labels: Labels length (8) must
        match element count (5)`` (``image_labels`` the same, worded for its
        own channel), with ``g`` on disk as a childless ``kind=partition``
        surviving ``finalize()``. With the fix, the ``'g'``-prefixed form and
        an empty store.
        """
        file_path = self._write(tmp_path, counts=(5,))
        compiler, scene, path = open_scene(
            tmp_path, f"file_onepart_wronglen_{channel}.luxar.zarr"
        )

        split = refusal(lambda: scene.add_gsplats_from_file("g", file_path, **kwargs))

        assert str(split).startswith("Could not add gsplats 'g': ")
        assert "part_0" not in str(split)
        assert _WRONG_LENGTH_WORDING[channel] in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize(
        "fault,extra,expected",
        [
            ("column_count", {}, "centers array has 4 columns"),
            (
                "dim_order",
                {"dim_order": ["X", "Y", "Z"]},
                "dim_order / fill / fill_sigma are not supported",
            ),
        ],
    )
    def test_the_pre_graft_checks_still_outrank_the_labels_refusal(
        self, tmp_path: Any, fault: str, extra: Dict[str, Any], expected: str
    ) -> None:
        """Precedence at the file door: both #1446-era checks sit above the gate.

        The labels refusal lives INSIDE ``graft_gsplat_node``, below the
        ``dim_order`` / ``fill`` / ``fill_sigma`` refusal and below the stored
        column-count check — the same "no flat counterpart, so rank it last"
        placement the lod/ half documents.
        """
        # 4 columns in a 3-dimension scene for the count fault; the dim_order one
        # is refused before the width is ever looked at, so its file is fine.
        file_path = self._write(tmp_path, ndim=4 if fault == "column_count" else 3)
        compiler, scene, _ = open_scene(tmp_path, f"file_prec_{fault}.luxar.zarr")

        split = refusal(
            lambda: scene.add_gsplats_from_file("g", file_path, labels=LABELS, **extra)
        )

        assert expected in str(split)
        assert "is not supported on a grafted multi-node" not in str(split)
        assert "g" not in compiler.store


# ---------------------------------------------------------------------------
# Unwritable colour DTYPE, pre-split (#1489)
# ---------------------------------------------------------------------------

#: Vertices per side of the dtype cases' mesh. 15x15 = 225 vertices / 392
#: triangles: comfortably above ``_HALF``, so ``max_elements=_HALF`` is a real
#: cut rather than a fall-through to a plain leaf.
_MESH_SIDE = 15
_MESH_N = _MESH_SIDE * _MESH_SIDE


def _add_coloured(scene: Any, geometry: str, colors: Any, **extra: Any) -> Any:
    """Add one node named ``n`` of ``geometry`` carrying ``colors``.

    One call site for all four geometry types so the dtype cases below differ
    only in the structural kwarg — the property under test is that the verdict
    does NOT depend on the geometry.
    """
    if geometry == "points":
        return scene.add_points(
            "n", random_positions(_N, seed=71), colors=colors, **extra
        )
    if geometry == "lines":
        return scene.add_lines(
            "n",
            random_positions(_N, seed=72),
            widths=0.3,
            line_type="segments",
            colors=colors,
            **extra,
        )
    if geometry == "gsplats":
        return scene.add_gsplats(
            "n",
            centers=random_positions(_N, seed=73),
            amplitudes=np.full(_N, 1.0, dtype=np.float32),
            cholesky_factors=cholesky_rows(_N),
            colors=colors,
            **extra,
        )
    vertices, faces = grid_mesh(_MESH_SIDE)
    return scene.add_mesh("n", vertices, faces, colors=colors, **extra)


def _coloured_element_count(geometry: str) -> int:
    return _MESH_N if geometry == "mesh" else _N


#: ``(id, partition spec)`` for the dtype cases. Two of these REALLY split at
#: these counts — measured, uint8 colours, parts per geometry:
#: ``max_elements=100`` → 2/2/2/4 (points/lines/gsplats/mesh) and
#: ``max_elements=25`` → 8/12/8/20. ``partition=True`` does NOT: the default
#: ``max_elements`` is 1,000,000, so it writes a plain leaf, and it is listed
#: under a name that says so rather than dropped, because the refusal must hold
#: on the fall-through path too. ``{"parts": N}`` is deliberately absent — no
#: adder reads that key, so it was never a partition spec at all, only a plain
#: leaf wearing the label.
_PARTITION_SPECS = [
    ("default_no_split", True),
    ("max_elements_half", {"max_elements": _HALF}),
    ("max_elements_quarter", {"max_elements": _HALF // 4}),
]
_PARTITION_IDS = [spec_id for spec_id, _ in _PARTITION_SPECS]
#: The ids from :data:`_PARTITION_SPECS` that must produce a real wrapper.
_SPLITTING_IDS = {"max_elements_half", "max_elements_quarter"}


class TestPartitionRefusesAnUnwritableColorDtype:
    """A colour DTYPE the leaf cannot store must not strand a partial node (#1489).

    The same bug class as everything above, one validator hole over: the encoder
    refuses an integer COLOR array that is not ``uint8``/``uint16``, but the
    shared pre-write validator checked only type, shape and finiteness — so the
    refusal landed from inside ``write_colors``, one dataset AFTER ``positions``
    (measured: ``n/`` on disk holding ``positions`` and nothing else). ``int64``
    is not a contrived dtype here; it is what ``np.array([[255, 0, 0]])`` gives
    you on Linux.

    Parity with the flat path is asserted in the strong sense (same type,
    byte-identical message) for the same reason the wrong-length cases do it: the
    two paths now run the ONE validator, and a substring match would survive a
    divergence. Mesh joins these cases because ``kind=partition`` is one of its
    two structural paths — the geometry is irrelevant to the verdict, which is
    exactly the claim.

    :data:`_PARTITION_SPECS` is the parametrization, and
    ``test_the_same_colours_as_uint8_still_partition`` runs over the SAME list —
    which is what keeps the class honest. An earlier cut parametrized over
    ``partition=True`` and ``{"parts": 4}``: measured at these counts, NEITHER
    splits (both give ``kind=leaf``, 0 children — no adder reads a ``"parts"``
    key at all, and the ``max_elements`` default is 1,000,000), so two thirds of
    the class silently re-ran the flat path while its control only ever exercised
    the one spec that did split. ``default_no_split`` is kept deliberately and
    named for what it is.
    """

    @pytest.mark.parametrize("geometry", ["points", "lines", "gsplats", "mesh"])
    @pytest.mark.parametrize("spec_id,spec", _PARTITION_SPECS, ids=_PARTITION_IDS)
    def test_int64_colors_are_refused_exactly_as_the_flat_path(
        self, tmp_path: Any, geometry: str, spec_id: str, spec: Any
    ) -> None:
        tag = f"{geometry}_{spec_id}"
        compiler, scene, path = open_scene(tmp_path, f"dtype_{tag}.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, f"dtype_flat_{tag}.luxar.zarr")
        colors = int64_rgb(_coloured_element_count(geometry))

        flat = refusal(lambda: _add_coloured(flat_scene, geometry, colors))
        split = refusal(lambda: _add_coloured(scene, geometry, colors, partition=spec))

        assert_same_refusal(flat, split)
        assert "Integer COLOR arrays must use dtype uint8 or uint16" in str(split)
        assert "int64" in str(split)
        assert "n" not in compiler.store
        # Not merely absent from the live store: absent from the DELIVERED scene.
        assert "n" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize(
        "spec", [None, {"max_elements": _HALF}], ids=["flat", "split"]
    )
    def test_a_complex_colour_array_is_refused_too(
        self, tmp_path: Any, spec: Any
    ) -> None:
        """The same stranding, one dtype KIND over — which is why the rule is a
        whitelist (#1489).

        ``complex64`` is ``np.number``, so the finiteness check passes it and the
        integer rule never sees it; measured with the rule neutered, a VARYING
        complex array reached the per-channel encoder, raised ``color_mode
        required for float COLOR arrays`` and left ``n/positions`` on disk.
        Varying, not uniform, on purpose: a uniform one takes the broadcast
        encoding and used to be WRITTEN (as a complex64 zarr array), which is the
        one behaviour this change deliberately takes away.
        """
        tag = "flat" if spec is None else "split"
        compiler, scene, path = open_scene(tmp_path, f"dtype_cplx_{tag}.luxar.zarr")
        rng = np.random.default_rng(74)
        colors = (rng.random((_N, 3)) + 0j).astype(np.complex64)
        extra = {} if spec is None else {"partition": spec}

        split = refusal(
            lambda: scene.add_points(
                "n", random_positions(_N, seed=75), colors=colors, **extra
            )
        )

        assert "COLOR arrays must be floating point, or integer uint8 or uint16" in str(
            split
        )
        assert "complex64" in str(split)
        assert "n" not in compiler.store
        assert "n" not in finalized_group_keys(compiler, path)

    @pytest.mark.parametrize("geometry", ["points", "lines", "gsplats", "mesh"])
    @pytest.mark.parametrize("spec_id,spec", _PARTITION_SPECS, ids=_PARTITION_IDS)
    def test_the_same_colours_as_uint8_still_partition(
        self, tmp_path: Any, geometry: str, spec_id: str, spec: Any
    ) -> None:
        """The negative control, over the SAME specs the refusal is parametrized on.

        Two jobs. It pins that only the DTYPE was ever wrong — the identical call
        with ``.astype(np.uint8)`` writes. And, because it runs the same list, it
        pins WHICH specs actually reach the wrapper: without that, a spec that
        quietly falls through to a plain leaf makes its refusal case a re-run of
        the flat path, asserting nothing about the partition gate. That is exactly
        what ``{"parts": 4}`` was doing. ``default_no_split`` is asserted to be a
        plain leaf, so the two claims cannot swap places unnoticed.
        """
        compiler, scene, path = open_scene(
            tmp_path, f"dtype_ok_{geometry}_{spec_id}.luxar.zarr"
        )
        colors = int64_rgb(_coloured_element_count(geometry)).astype(np.uint8)

        _add_coloured(scene, geometry, colors, partition=spec)
        compiler.finalize()

        if spec_id not in _SPLITTING_IDS:
            leaf = zarr.open_group(path, mode="r")["n"]
            assert leaf.attrs.get("kind") != "partition"
            assert sorted(leaf.group_keys()) == []
            return

        store = zarr.open_group(path, mode="r")
        assert store["n"].attrs["kind"] == "partition"
        assert len(sorted(store["n"].group_keys())) > 1


# ---------------------------------------------------------------------------
# Node-attrs gate, pre-split, on the Mesh/GSplats PARTITION path (#1534)
# ---------------------------------------------------------------------------
#
# add_mesh_impl / add_gsplats_impl never ran validate_render_attrs at the
# adder entry (unlike Points/Lines since #1529): a bad attr under partition=
# was refused only from inside the first synthesised child (Mesh's
# ``part_0``, GSplats' ``part_0``), by which point ``add_partition_group`` had
# already created the wrapper's childless ``kind=partition`` node on disk.
# #1534 hoists the same validator to each adder's entry, above every
# structural branch, closing this for both geometry types the way #1529 did
# for Points/Lines. The LOD-wrapper half of the #1534 gate (Mesh's
# ``additive_lod=``, plus the placement/precedence controls shared by both
# geometry types) lives in ``tests/group/lod/test_source_validation.py``.

_MESH_PART_SIDE = 4  # grid_mesh(4): 16 vertices, 18 faces
_MESH_PART_MAX_ELEMENTS = 8  # -> 4 face-count-capped parts (measured)


class TestMeshPartitionNodeAttrsGate:
    def test_unknown_attr_typo_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(tmp_path, "mesh_part_attrs_typo.luxar.zarr")
        _, flat_scene, _ = open_scene(tmp_path, "mesh_part_attrs_typo_flat.luxar.zarr")
        vertices, faces = grid_mesh(_MESH_PART_SIDE)

        flat = refusal(
            lambda: flat_scene.add_mesh("m", vertices, faces, blending="max")
        )
        split = refusal(
            lambda: scene.add_mesh(
                "m",
                vertices,
                faces,
                partition={"max_elements": _MESH_PART_MAX_ELEMENTS},
                blending="max",
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "m" not in compiler.store
        # Pre-fix this raised the SAME message but still left a childless
        # kind=partition "m" on disk, surviving finalize() — the #1534
        # stranding this closes (the Mesh/GSplats peer of #1529).
        assert "m" not in finalized_group_keys(compiler, path)


_GSPLATS_PART_N = 200
_GSPLATS_PART_HALF = 100


class TestGSplatsPartitionNodeAttrsGate:
    def test_unknown_attr_typo_leaves_no_childless_wrapper(self, tmp_path: Any) -> None:
        compiler, scene, path = open_scene(
            tmp_path, "gsplats_part_attrs_typo.luxar.zarr"
        )
        _, flat_scene, _ = open_scene(
            tmp_path, "gsplats_part_attrs_typo_flat.luxar.zarr"
        )
        centers = random_positions(_GSPLATS_PART_N, seed=201)
        chol = cholesky_rows_nd(_GSPLATS_PART_N, 3)

        flat = refusal(
            lambda: flat_scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                blending="max",
            )
        )
        split = refusal(
            lambda: scene.add_gsplats(
                "g",
                centers=centers,
                amplitudes=1.0,
                cholesky_factors=chol,
                partition={"max_elements": _GSPLATS_PART_HALF},
                blending="max",
            )
        )

        assert_same_refusal(flat, split)
        assert "Did you mean 'blending_mode'?" in str(split)
        assert "g" not in compiler.store
        assert "g" not in finalized_group_keys(compiler, path)
