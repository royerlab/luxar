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
    ("unknown_key", {"parts": 4}),
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

    def test_the_stored_ndim_is_what_the_sub_two_dimension_skip_reads(
        self, tmp_path: Any
    ) -> None:
        """The gate's ``ndim`` argument is the STORED tree's, not a constant.

        Below 2 spatial dims the leaf DROPS the partition request with a warning
        (``warn_if_partition_needs_more_dims``) BEFORE it resolves the spec, so a
        1-dimension scene accepts even a nonsense spec — which is why
        ``reject_bad_partition_spec`` skips ``ndim < 2``. The lod/ sibling pins
        that for the ``lod_group=`` door
        (``TestTheGSplatsPartitionSpecCheckSkipsASubTwoDimensionScene``); this is
        the graft counterpart, and the only test that reads the ``node_ndim(node)``
        argument at all. Measured with that argument replaced by a constant ``3``:
        the six tests above all stay green and this one answers ``Could not add
        gsplats 'g': partition must be None, True, or dict; got str`` — a refusal
        the flat path does not make.
        """
        from luxar import Dimension, Dimensions, LuxarZarrCompiler
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / "nested_1d.gsplats.zarr")
        write_gsplats_tree(file_path, _nested_partition_tree(1))

        scene_path = str(tmp_path / "graft_part_1d.luxar.zarr")
        compiler = LuxarZarrCompiler(scene_path)
        scene = compiler.create_scene(
            dimensions=Dimensions([Dimension("X", display=True)])
        )
        scene.add_gsplats_from_file("g", file_path, partition="nonsense")
        compiler.finalize()

        store = zarr.open_group(scene_path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        # The request was DROPPED, not honoured and not refused.
        assert [
            store["g"][p].attrs.get("kind") for p in sorted(store["g"].group_keys())
        ] == [None, None]


def _laddered_partition_tree(ndim: int = 3) -> Any:
    """A ``kind=partition`` of two 2-sublod leaves — the pipeline's usual output.

    What ``gsplat lod --recipe tiles|overview|adaptive`` and ``batch-fit merge
    --recipe stream`` emit: a spatial partition whose every part carries its own
    additive ladder (stream ladders are on by default). The ``(8, 6)`` shape and
    the ndim parameter match :func:`_nested_partition_tree`; the only difference
    is the second sub-LOD per leaf.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

    def sub(n: int, seed: int) -> Any:
        return AdditiveSubLOD(
            centers=bad_ndim_positions(n, seed=seed, ndim=ndim),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=cholesky_rows_nd(n, ndim),
        )

    def leaf(n: int, seed: int) -> Any:
        return GSplatLeaf(additive_sublods=[sub(n // 2, seed), sub(n, seed + 50)])

    return GSplatPartition(children=[leaf(8, 91), leaf(6, 92)], max_elements=8)


class TestGraftedFilePartitionBesideAStoredLadder:
    """A VALID spec strands on the graft door too, when the leaves are laddered.

    ``partition=`` and an additive ladder are mutually exclusive — the multi-LOD
    writer has no ``partition`` parameter — but that conflict was judged only
    inside ``add_gsplats_from_data_impl``, which the graft reaches PER LEAF, after
    ``graft_gsplat_node`` had already built the ``kind=partition`` wrapper.
    Measured pre-fix against a stored 2-part partition of 2-sublod leaves:
    ``Could not add gsplats 'part_0': partition= is not supported alongside
    additive_lod= …``, with ``g`` surviving ``finalize()`` as a childless
    ``kind=partition``. And with ``partition=False``, the identical strand from
    ``Unknown node attribute 'partition'``.

    The wording matters as much as the placement here: there is no
    ``additive_lod=`` in an ``add_gsplats_from_file`` call, so "drop one of the
    two" names a kwarg the caller never passed. This door names the STORE and
    gives its own remedy, the convention ``labels_on_wrapper_reason`` /
    ``GRAFT_REMEDY`` already keep next door.

    Every call here passes no ``additive_lod=`` (or the ``False`` spelling), so
    the ladder is always the store's and this wording always applies.
    ``TestGraftedFilePartitionBesideAResolvedLadder`` below covers the other
    half (#1632): a ladder the kwarg itself would BUILD, which gets the
    ``additive_lod=`` pair instead.
    """

    def _file(self, tmp_path: Any, filename: str) -> str:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / filename)
        write_gsplats_tree(file_path, _laddered_partition_tree())
        return file_path

    def test_a_real_spec_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any
    ) -> None:
        file_path = self._file(tmp_path, "laddered_spec.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "graft_ladder_spec.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}
            )
        )

        assert "partition= is not supported alongside" in str(exc)
        assert "part_0" not in str(exc)
        assert "g" not in compiler.store
        assert finalized_group_keys(compiler, path) == set()

    def test_the_message_names_the_store_and_this_doors_own_remedy(
        self, tmp_path: Any
    ) -> None:
        """Not ``additive_lod=``: the caller never passed one and cannot drop it."""
        file_path = self._file(tmp_path, "laddered_wording.gsplats.zarr")
        _, scene, _ = open_scene(tmp_path, "graft_ladder_wording.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}
            )
        )

        assert "this .gsplats.zarr already carries" in str(exc)
        assert "gsplat flatten" in str(exc)
        assert "additive_lod=" not in str(exc)
        assert "Drop one of the two" not in str(exc)

    def test_a_laddered_leaf_file_answers_the_same_way(self, tmp_path: Any) -> None:
        """The matrix-shaped branch never reaches the graft — same door, same words.

        ``add_gsplats_from_file`` sends a bare laddered leaf down
        ``add_gsplats_from_data_impl``, whose own refusal names ``additive_lod=``.
        Nothing strands there, but the caller is told to drop a kwarg that is not
        in the call, so this branch gets the stored-ladder wording too.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / "laddered_leaf.gsplats.zarr")
        write_gsplats_tree(file_path, next(iter(_laddered_partition_tree().children)))
        _, scene, _ = open_scene(tmp_path, "graft_ladder_leaf.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 2}
            )
        )

        assert "this .gsplats.zarr already carries" in str(exc)
        assert "additive_lod=" not in str(exc)

    def test_false_grafts_the_whole_ladder_instead_of_stranding(
        self, tmp_path: Any
    ) -> None:
        """``False`` is the bypass, so the laddered parts are simply written."""
        file_path = self._file(tmp_path, "laddered_false.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "graft_ladder_false.luxar.zarr")

        scene.add_gsplats_from_file("g", file_path, partition=False)
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        parts = sorted(store["g"].group_keys())
        assert parts == ["part_0", "part_1"]
        # Each part is its own laddered leaf: nothing was partitioned, nothing
        # was dropped.
        assert sorted(store["g"]["part_0"].group_keys()) == [
            "additive_0",
            "additive_1",
        ]

    @pytest.mark.parametrize("shape", ["leaf", "partition"])
    def test_additive_lod_false_collapses_the_ladder_and_still_splits(
        self, tmp_path: Any, shape: str
    ) -> None:
        """The gate judges the RESOLVED ladder, so the kwarg that empties it wins.

        ``additive_lod=False`` is the documented "collapse the ladder" spelling:
        ``resolve_additive_axis_gsplats`` flattens every level to a single sub-LOD
        BEFORE the data door asks this same question, so the store's ladder is not
        the one that would be written and the conflict does not exist. Since
        #1632 that needs no hand-written exemption — ``resolve_additive_rungs``
        counts ``False`` as one rung like any other spec, and the gate skips a
        leaf on the count alone. Judging the store instead refuses two calls that
        work: measured on the pre-#1632 store-only gate with its hand-written
        ``False`` skip removed, both parametrisations answer ``Could not add
        gsplats 'g': partition= is not supported alongside the additive ladder
        this .gsplats.zarr already carries``, where without the gate entirely they
        split into 4 and 2 real parts respectively.

        Both doors, because the gate has two call sites: the matrix-shaped branch
        of ``add_gsplats_from_file`` (the bare leaf) and ``graft_gsplat_node``
        (the nested ``kind=partition``).
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        tree = _laddered_partition_tree()
        file_path = str(tmp_path / f"ladder_off_{shape}.gsplats.zarr")
        write_gsplats_tree(
            file_path, tree if shape == "partition" else next(iter(tree.children))
        )
        compiler, scene, path = open_scene(
            tmp_path, f"graft_ladder_off_{shape}.luxar.zarr"
        )

        scene.add_gsplats_from_file(
            "g", file_path, partition={"max_elements": 4}, additive_lod=False
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        parts = sorted(store["g"].group_keys())
        # The split really ran: the bare leaf holds 12 splats (its two rungs, 4
        # and 8, unioned into one leaf first), so a cap of 4 gives 4 parts, and
        # the 2-part store gives one re-split part each.
        assert parts == (
            ["part_0", "part_1"]
            if shape == "partition"
            else ["part_0", "part_1", "part_2", "part_3"]
        )
        # No ladder survived anywhere — that is what ``False`` asked for.
        assert not any(
            key.startswith("additive_")
            for part in parts
            for key in store["g"][part].group_keys()
        )

    def test_an_unladdered_partition_still_splits(self, tmp_path: Any) -> None:
        """Non-vacuity: the gate keys on the LADDER, not on grafting a partition.

        ``TestGraftedFilePartitionSpec.test_a_valid_spec_still_grafts_and_still_splits``
        covers the same shape without ladders; repeated here as this class's own
        control so a gate that refused every grafted ``partition=`` would go red.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / "unladdered.gsplats.zarr")
        write_gsplats_tree(file_path, _nested_partition_tree(3))
        compiler, scene, path = open_scene(tmp_path, "graft_ladder_control.luxar.zarr")

        scene.add_gsplats_from_file("g", file_path, partition={"max_elements": 4})
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert [
            store["g"][p].attrs.get("kind") for p in sorted(store["g"].group_keys())
        ] == ["partition", "partition"]


def _descendant_group_keys(group: Any) -> List[str]:
    """Every group name anywhere under ``group``, recursively.

    The ladder assertions below have to look deeper than one level: a split
    part is itself a ``kind=partition``, so an ``additive_0`` left behind by a
    ladder that should not exist hides two levels down.
    """
    out: List[str] = []
    for key in group.group_keys():
        out.append(key)
        out.extend(_descendant_group_keys(group[key]))
    return out


class TestGraftedFilePartitionBesideAResolvedLadder:
    """The gate must judge the RESOLVED ladder, not the stored one (#1632).

    ``additive_lod=`` is not a parameter of ``graft_gsplat_node``: it rides in
    ``**attrs`` down to each part's own ``add_gsplats_from_data_impl``, which
    BUILDS the ladder there — one level below the ``kind=partition`` wrapper the
    graft has already created. So a store-only gate discovers the conflict too
    late. Measured pre-fix on an UNLADDERED nested partition with
    ``partition={"max_elements": 4}, additive_lod={"n_lods": 2}``: ``Could not
    add gsplats 'part_0': partition= is not supported alongside an additive_lod=
    ladder …``, with ``g`` surviving ``finalize()`` as a childless
    ``kind=partition`` (``finalized keys: ['g']; g -> kind=partition,
    children=[]``).

    And the mirror image, from the issue's second comment: a store-only gate
    also refused a call that COLLAPSES the store's ladder
    (``{"n_lods": 1, "recompute": True}`` — one rung, flat route, partitions
    fine), because only the ``additive_lod=False`` spelling was skipped by hand.

    Presence of the kwarg is emphatically NOT the question. Measured rung counts
    for the specs below, on this class's 8-and-6-splat leaves: ``{"n_lods": 1}``
    → 1 (legitimate), ``{"n_lods": 2}`` → 2, ``{"method": "radial"}`` → 4
    (``n_lods`` defaults to 4 and there is no ``n_lods`` key to read; on a leaf
    of fewer than 4 splats the equal-count cuts would clamp to ``n`` instead, so
    "four" is a property of these fixtures, not of the spec).

    But the resolved ladder is only ever the answer when the kwarg CAN be read.
    An UNKNOWN count falls back to the store, because ``None`` is a statement
    about the kwarg, not about the store — see
    ``test_an_unknown_count_falls_back_to_the_stored_ladder``, the regression
    that the first draft of this gate (which skipped the leaf instead) failed.
    And WHICH REMEDY is chosen by the OFFENDING leaf's store alone, never by
    whether the kwarg also rebuilds the ladder or by what the rest of the tree
    stores: a stored ladder is the obstacle that survives dropping the kwarg.

    A WELL-FORMED energy-fraction spec on an UNLADDERED store
    (``partition={"max_elements": 4}, additive_lod={"breakpoints": [0.5, 1.0]}``)
    still reaches the child: the count is UNKNOWN (energy cuts need the ordering
    and the
    energy curve), the fallback is the store's single rung, the leaf skips, and
    this same conflict fires from inside ``part_0``. The graft transaction now
    removes ``g`` and every descendant before re-raising that builder verdict.
    Refusing on UNKNOWN instead remains rejected deliberately:
    ``{"breakpoints": [1.0]}`` resolves to a single rung and partitions fine (a
    gate refusing what the flat path accepts is its own regression), so closing
    the transaction preserves that accepted case without leaving partial output.
    """

    def _unladdered(self, tmp_path: Any, filename: str) -> str:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / filename)
        write_gsplats_tree(file_path, _nested_partition_tree(3))
        return file_path

    @pytest.mark.parametrize(
        "spec,tag",
        [({"n_lods": 2}, "n2"), ({"method": "radial"}, "radial")],
    )
    def test_a_kwarg_built_ladder_is_refused_before_the_wrapper_exists(
        self, tmp_path: Any, spec: Dict[str, Any], tag: str
    ) -> None:
        """The issue's own repro, and the spec that carries no ``n_lods`` at all."""
        file_path = self._unladdered(tmp_path, f"kwarg_ladder_{tag}.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, f"kwarg_{tag}.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}, additive_lod=spec
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "partition= is not supported alongside" in str(exc)
        assert "part_0" not in str(exc)
        assert "g" not in compiler.store
        assert finalized_group_keys(compiler, path) == set()

    def test_the_kwarg_built_refusal_uses_the_additive_lod_wording(
        self, tmp_path: Any
    ) -> None:
        """There IS an ``additive_lod=`` in this call, and dropping it is the fix.

        The stored-ladder pair ('gsplat flatten') would send the caller to
        rewrite a file that carries no ladder at all. Using the ``additive_lod=``
        pair also makes this refusal byte-identical to the one the data door
        raises for the same call one step later — for a call whose ONLY fault is
        this conflict.

        The four wording assertions alone are VACUOUS: the pre-fix refusal came
        from inside ``part_0``, and its body is byte-identical to this one by
        design, so all four held while the wrapper stranded. The name and the
        empty-store assertions are what discriminate, exactly as in the sibling
        above.
        """
        file_path = self._unladdered(tmp_path, "kwarg_wording.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "kwarg_wording.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod={"n_lods": 2},
            )
        )

        assert "an additive_lod= ladder" in str(exc)
        assert "Drop one of the two" in str(exc)
        assert "gsplat flatten" not in str(exc)
        assert "already carries" not in str(exc)
        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert finalized_group_keys(compiler, path) == set()

    def test_the_matrix_shaped_door_refuses_a_kwarg_built_ladder_identically(
        self, tmp_path: Any
    ) -> None:
        """The OTHER call site of this gate: a bare leaf, not a nested tree.

        ``add_gsplats_from_file`` sends every matrix-shaped tree straight to
        ``add_gsplats_from_data_impl`` instead of grafting, and calls this gate
        there too. A bare UNLADDERED leaf plus ``partition=`` plus
        ``additive_lod={"n_lods": 2}`` is the one shape where the store carries
        nothing and the kwarg supplies the whole ladder, on the branch whose data
        door names the caller's own node — so this is where the refusal is
        byte-identical to the data door's, message and name alike, rather than
        merely sharing its body (the graft branch says ``'g'`` where the data
        door would have said ``part_0``, which is the point of hoisting it).
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / "matrix_kwarg.gsplats.zarr")
        write_gsplats_tree(file_path, _nested_partition_tree(3).children[0])
        compiler, scene, path = open_scene(tmp_path, "matrix_kwarg.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod={"n_lods": 2},
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "an additive_lod= ladder" in str(exc)
        assert "Drop one of the two" in str(exc)
        assert "gsplat flatten" not in str(exc)
        assert "part_0" not in str(exc)
        assert "g" not in compiler.store
        assert finalized_group_keys(compiler, path) == set()

    def test_a_single_rung_spec_is_not_refused_and_the_split_really_runs(
        self, tmp_path: Any
    ) -> None:
        """``{"n_lods": 1}`` takes the flat route, so ``partition=`` is legitimate.

        A presence-only check (``or is_requested(attrs.get("additive_lod"))``)
        would refuse this, which is why the gate resolves a COUNT instead.
        """
        file_path = self._unladdered(tmp_path, "one_rung.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "one_rung.luxar.zarr")

        scene.add_gsplats_from_file(
            "g", file_path, partition={"max_elements": 4}, additive_lod={"n_lods": 1}
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        # Each stored part was re-split, exactly as the un-laddered control does.
        assert [
            store["g"][p].attrs.get("kind") for p in sorted(store["g"].group_keys())
        ] == ["partition", "partition"]
        assert not any(
            key.startswith("additive_") for key in _descendant_group_keys(store["g"])
        )

    @pytest.mark.parametrize("shape", ["leaf", "partition"])
    def test_a_recompute_spec_that_collapses_the_stored_ladder_still_splits(
        self, tmp_path: Any, shape: str
    ) -> None:
        """The issue's second comment: one rung is one rung, however it is spelled.

        ``{"n_lods": 1, "recompute": True}`` resolves the store's ladder down to
        a single sub-LOD before the data door asks anything — the same end state
        ``additive_lod=False`` produces, which this gate has always skipped.
        Measured pre-fix, both parametrisations answered ``Could not add gsplats
        'g': partition= is not supported alongside the additive ladder this
        .gsplats.zarr already carries``.

        Both doors, mirroring
        ``test_additive_lod_false_collapses_the_ladder_and_still_splits``: the
        matrix-shaped branch of ``add_gsplats_from_file`` (the bare leaf) and
        ``graft_gsplat_node`` (the nested ``kind=partition``).
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        tree = _laddered_partition_tree()
        file_path = str(tmp_path / f"recollapse_{shape}.gsplats.zarr")
        write_gsplats_tree(
            file_path, tree if shape == "partition" else next(iter(tree.children))
        )
        compiler, scene, path = open_scene(tmp_path, f"recollapse_{shape}.luxar.zarr")

        scene.add_gsplats_from_file(
            "g",
            file_path,
            partition={"max_elements": 4},
            additive_lod={"n_lods": 1, "recompute": True},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        parts = sorted(store["g"].group_keys())
        # Same split the ``additive_lod=False`` twin produces: the bare leaf
        # holds 12 splats (its two rungs, 4 and 8, unioned), so a cap of 4 gives
        # 4 parts, and the 2-part store one re-split part each.
        assert parts == (
            ["part_0", "part_1"]
            if shape == "partition"
            else ["part_0", "part_1", "part_2", "part_3"]
        )
        assert not any(
            key.startswith("additive_") for key in _descendant_group_keys(store["g"])
        )

    @pytest.mark.parametrize("partition", [None, False])
    def test_a_ladder_without_a_real_partition_request_still_grafts(
        self, tmp_path: Any, partition: Any
    ) -> None:
        """Non-vacuity: the gate keys on ``partition=``, not on ``additive_lod=``.

        ``additive_lod={"n_lods": 2}`` alone must still ladder every grafted
        part, and ``partition=False`` is the bypass — not a request — so it must
        behave identically (the leaf adder DELETES that key once the multi-LOD
        writer is the destination). A gate that refused on the kwarg alone would
        break both.
        """
        file_path = self._unladdered(tmp_path, f"ladder_only_{partition}.gsplats.zarr")
        compiler, scene, path = open_scene(
            tmp_path, f"ladder_only_{partition}.luxar.zarr"
        )

        kwargs = {} if partition is None else {"partition": partition}
        scene.add_gsplats_from_file(
            "g", file_path, additive_lod={"n_lods": 2}, **kwargs
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["g"].attrs.get("kind") == "partition"
        assert sorted(store["g"].group_keys()) == ["part_0", "part_1"]
        # A ladder per part — built by the kwarg, since the store carried none.
        for part in ("part_0", "part_1"):
            assert sorted(store["g"][part].group_keys()) == [
                "additive_0",
                "additive_1",
            ]

    @pytest.mark.parametrize(
        "spec,tag",
        [
            ({"n_lods": 2}, "n2"),
            # NOT one rung: a dict without ``recompute`` never touches a level
            # that already has a ladder, so ``n_lods`` is not even read here.
            ({"n_lods": 1}, "n1"),
            # FINDING 2's case: the kwarg DOES rebuild this ladder, and the
            # remedy is still the store's, because dropping ``additive_lod=``
            # leaves the stored ladder — and a second refusal — behind.
            ({"n_lods": 4, "recompute": True}, "rebuilt"),
        ],
    )
    def test_a_stored_ladder_keeps_its_own_wording_under_a_pass_through_spec(
        self, tmp_path: Any, spec: Dict[str, Any], tag: str
    ) -> None:
        """An ALREADY-laddered store keeps the STORED wording, whatever the kwarg.

        The resolver passes a stored multi-rung ladder through untouched (it only
        computes where ``recompute`` is set or the level has <= 1 rung), so
        ``{"n_lods": 1}`` is emphatically not one rung here. And even where the
        kwarg does recompute, the obstacle is still the store's: 'gsplat flatten'
        is the step that has to happen either way, whereas "drop one of the two"
        would send the caller straight into a second refusal.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / f"stored_passthrough_{tag}.gsplats.zarr")
        write_gsplats_tree(file_path, _laddered_partition_tree())
        compiler, scene, path = open_scene(
            tmp_path, f"stored_passthrough_{tag}.luxar.zarr"
        )

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}, additive_lod=spec
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "this .gsplats.zarr already carries" in str(exc)
        assert "gsplat flatten" in str(exc)
        assert "Drop one of the two" not in str(exc)
        assert finalized_group_keys(compiler, path) == set()

    @pytest.mark.parametrize(
        "spec,tag",
        [
            ({"n_lods": 2}, "n2"),
            # The same store where the kwarg ALSO rebuilds the stored leaf's
            # ladder, so no leaf is left whose ladder is "purely the store's" in
            # the build sense — and the answer must still be the stored pair.
            ({"n_lods": 2, "recompute": True}, "rebuilt"),
        ],
    )
    def test_a_mixed_store_is_answered_by_the_leaf_the_store_laddered(
        self, tmp_path: Any, spec: Dict[str, Any], tag: str
    ) -> None:
        """One laddered leaf, one flat, and a kwarg that ladders the flat one.

        Both leaves offend, for different reasons — the flat one only because
        the kwarg builds it a ladder, the laddered one on its own — so the walk
        must not answer with whichever it meets first. It reports the STORED
        pair, because that ladder is the obstacle that survives dropping the
        kwarg; the flat leaf is deliberately FIRST so a first-leaf-wins gate
        would go red here.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatPartition

        # ``_nested_partition_tree``'s leaves are single-sub-LOD by construction
        # (8 splats); ``_laddered_leaf`` carries two rungs of 4.
        mixed = GSplatPartition(
            children=[_nested_partition_tree(3).children[0], _laddered_leaf()],
            max_elements=8,
        )
        file_path = str(tmp_path / f"mixed_ladder_{tag}.gsplats.zarr")
        write_gsplats_tree(file_path, mixed)
        compiler, scene, path = open_scene(tmp_path, f"mixed_ladder_{tag}.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}, additive_lod=spec
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "this .gsplats.zarr already carries" in str(exc)
        assert "gsplat flatten" in str(exc)
        assert "Drop one of the two" not in str(exc)
        assert finalized_group_keys(compiler, path) == set()

    def test_a_mixed_collapse_names_the_kwarg_the_flat_leaf_being_the_blocker(
        self, tmp_path: Any
    ) -> None:
        """The discriminator is the OFFENDING leaf, not "a ladder exists somewhere".

        ``{"recompute": True, "breakpoints": [4]}`` COLLAPSES the laddered
        4-splat leaf (its cuts clamp to a single ``[4]``) and LADDERS the flat
        8-splat one (``[4, 8]``), so the only leaf blocking the call is the flat
        one — and the message is the ``additive_lod=`` pair even though a stored
        ladder sits untouched next door. That is the right answer, not a miss:
        'gsplat flatten' would send the caller to rewrite a ladder that is not in
        the way, whereas dropping either half of "drop one of the two" really
        does resolve it. Pinned because the WHICH REMEDY paragraph in
        ``from_io`` claims exactly this.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree
        from luxar.gsplats.tree import GSplatPartition

        collapsing = _laddered_leaf(n_per_sublod=2, n_sublods=2)  # 4 splats, 2 rungs
        flat = _nested_partition_tree(3).children[0]  # 8 splats, 1 rung
        assert (collapsing.n_splats, flat.n_splats) == (4, 8)
        file_path = str(tmp_path / "mixed_collapse.gsplats.zarr")
        write_gsplats_tree(
            file_path, GSplatPartition(children=[collapsing, flat], max_elements=8)
        )
        compiler, scene, path = open_scene(tmp_path, "mixed_collapse.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod={"recompute": True, "breakpoints": [4]},
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "an additive_lod= ladder" in str(exc)
        assert "Drop one of the two" in str(exc)
        assert "gsplat flatten" not in str(exc)
        assert finalized_group_keys(compiler, path) == set()

    @pytest.mark.parametrize(
        "spec,tag",
        [
            # Energy fractions: uncountable without the ordering and the energy
            # curve, which is the whole expense this query avoids.
            ({"recompute": True, "breakpoints": [0.3, 1.0]}, "energy"),
            # Malformed: ``_resolve_breakpoints`` would reject ``n_lods=0``.
            ({"recompute": True, "n_lods": 0}, "malformed"),
            # Junk type: not None, not a bool, not a dict.
            ("stream", "junk"),
        ],
    )
    def test_an_unknown_count_falls_back_to_the_stored_ladder(
        self, tmp_path: Any, spec: Any, tag: str
    ) -> None:
        """REGRESSION: an unreadable kwarg does not make the store's ladder vanish.

        The first draft of this gate skipped a leaf whose resolved count came
        back ``None``, on the rule "a query must not pre-empt the builder's own
        fault report". But UNKNOWN is a statement about the KWARG, and on a
        LADDERED store the conflict is visible in ``iter_leaves`` without
        counting anything — so skipping threw it away and re-stranded exactly the
        wrapper this gate exists to prevent. Measured with the skip, on this
        stored ``kind=partition`` of 2-sublod leaves: ``Could not add gsplats
        'part_0': …`` for the energy case, ``ValueError: n_lods must be positive``
        for the malformed one and ``TypeError: additive_lod must be None, bool,
        or dict; got str`` for the junk one — each with ``g`` surviving
        ``finalize()`` as a childless ``kind=partition``. Falling back to
        ``stored_rungs`` restores the pre-#1632 verdict for all three, byte for
        byte.

        The rule itself is untouched where it belongs: on an UNLADDERED store the
        fallback is 1 rung, so the leaf still skips and the builder still reports
        the malformed spec — see
        ``test_an_unknown_count_on_an_unladdered_store_still_reaches_the_builder``.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / f"unknown_{tag}.gsplats.zarr")
        write_gsplats_tree(file_path, _laddered_partition_tree())
        compiler, scene, path = open_scene(tmp_path, f"unknown_{tag}.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g", file_path, partition={"max_elements": 4}, additive_lod=spec
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "partition= is not supported alongside" in str(exc)
        assert "this .gsplats.zarr already carries" in str(exc)
        assert "gsplat flatten" in str(exc)
        assert "part_0" not in str(exc)
        assert finalized_group_keys(compiler, path) == set()

    def test_a_counts_list_the_resolver_would_reject_is_unknown_not_one_rung(
        self, tmp_path: Any
    ) -> None:
        """REGRESSION: the wrapper must VALIDATE a ``counts:`` list, not clamp it.

        ``resolve_additive_axis_gsplats`` runs the strict
        ``validate_counts_breakpoints`` against ``substitutive_levels[0]`` BEFORE
        its per-level clamp loop, and a single grafted leaf IS that finest level
        — so an out-of-range list ABORTS the call, it never shrinks. A wrapper
        that only clamped answered a confident ONE rung instead: measured on this
        store, ``[100]`` clamped to the 12-splat leaf's own size, the gate
        skipped, ``graft_gsplat_node`` wrote the wrapper, and ``largest
        breakpoint 100 exceeds N=12`` then fired from inside ``part_0`` with
        ``g`` surviving ``finalize()`` as a childless ``kind=partition``. Round
        1's unknown→stored fallback does not save it, because the count is not
        unknown, it is WRONG — which is why the validation has to happen here.
        """
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / "oversized_counts.gsplats.zarr")
        write_gsplats_tree(file_path, _laddered_partition_tree())
        compiler, scene, path = open_scene(tmp_path, "oversized_counts.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod={"recompute": True, "breakpoints": [100]},
            )
        )

        assert str(exc).startswith("Could not add gsplats 'g': ")
        assert "partition= is not supported alongside" in str(exc)
        # UNKNOWN falls back to the store, which IS laddered here.
        assert "this .gsplats.zarr already carries" in str(exc)
        assert "part_0" not in str(exc)
        assert "exceeds N" not in str(exc)
        assert finalized_group_keys(compiler, path) == set()

    def test_an_unknown_count_on_an_unladdered_store_still_reaches_the_builder(
        self, tmp_path: Any
    ) -> None:
        """The other half of the fallback: no stored ladder, no conflict to judge.

        ``stored_rungs`` is 1 here, so the leaf skips and the malformed spec is
        answered by the builder itself, one level down, with its own message —
        which is where "a query must not pre-empt the builder's own fault report"
        applies. The wrapper strand this leaves behind is the documented residual
        (an invalid CALL, not a partition conflict), so what is pinned is the
        AUTHOR of the verdict, not the strand.
        """
        file_path = self._unladdered(tmp_path, "unknown_unladdered.gsplats.zarr")
        _, scene, _ = open_scene(tmp_path, "unknown_unladdered.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod={"recompute": True, "n_lods": 0},
            )
        )

        assert "n_lods must be positive" in str(exc)
        assert "partition= is not supported alongside" not in str(exc)


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


class TestPartitionSpecKeys:
    @pytest.mark.parametrize("geometry", ["points", "lines", "gsplats", "mesh"])
    def test_an_unknown_key_is_refused_before_any_node_is_written(
        self, tmp_path: Any, geometry: str
    ) -> None:
        compiler, scene, path = open_scene(
            tmp_path, f"unknown_partition_key_{geometry}.luxar.zarr"
        )
        colors = np.zeros((_coloured_element_count(geometry), 3), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"unknown partition key: 'parts'"):
            _add_coloured(scene, geometry, colors, partition={"parts": 4})

        assert "n" not in compiler.store
        assert finalized_group_keys(compiler, path) == set()


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


class TestFailedGraftRollsBackItsWrapper:
    """A child refusal must not leave the graft's wrapper in either tree."""

    def _late_failure_tree(self, *, invalid_selector: bool) -> Any:
        from luxar.gsplats.tree import GSplatLodGroup, GSplatPartition

        leaves = list(_nested_partition_tree(3, counts=(8, 6, 5)).children)
        second_meta = {"selector": "bogus"} if invalid_selector else {}
        return GSplatPartition(
            children=[
                GSplatLodGroup(children=[leaves[0]]),
                GSplatLodGroup(children=[leaves[1], leaves[2]], meta=second_meta),
            ],
            max_elements=8,
        )

    def _file(self, tmp_path: Any, filename: str) -> str:
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        file_path = str(tmp_path / filename)
        write_gsplats_tree(file_path, _nested_partition_tree(3))
        return file_path

    @pytest.mark.parametrize(
        "additive_lod,message",
        [
            ({"breakpoints": [0.5, 1.0]}, "partition= is not supported alongside"),
            ({"n_lods": 0, "recompute": True}, "n_lods must be positive"),
        ],
    )
    def test_child_failure_removes_the_whole_new_graft(
        self, tmp_path: Any, additive_lod: Any, message: str
    ) -> None:
        file_path = self._file(tmp_path, "unladdered_failed.gsplats.zarr")
        compiler, scene, path = open_scene(tmp_path, "failed_graft.luxar.zarr")

        exc = refusal(
            lambda: scene.add_gsplats_from_file(
                "g",
                file_path,
                partition={"max_elements": 4},
                additive_lod=additive_lod,
            )
        )

        assert message in str(exc)
        if additive_lod == {"breakpoints": [0.5, 1.0]}:
            assert "part_0" in str(exc)
        assert "g" not in compiler.store
        assert all(child.name != "g" for child in scene.children)
        assert finalized_group_keys(compiler, path) == set()

    def test_late_child_failure_removes_written_descendants_and_bounds(
        self, tmp_path: Any
    ) -> None:
        from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

        node = self._late_failure_tree(invalid_selector=True)
        compiler, scene, path = open_scene(tmp_path, "late_failed_graft.luxar.zarr")
        scene.add_points(
            "anchor",
            np.array([[100.0, 101.0, 102.0], [103.0, 104.0, 105.0]], dtype=np.float32),
        )
        assert compiler._scene_bounds is not None
        bounds_before = {
            key: list(values) for key, values in compiler._scene_bounds.items()
        }
        container = scene.add_group("container")

        exc = refusal(
            lambda: graft_gsplat_node(scene, name="g", node=node, parent=container)
        )

        assert "bogus" in str(exc)
        assert "container/g" not in compiler.store
        assert container.children == []
        assert compiler._scene_bounds == bounds_before
        compiler.finalize()
        store = zarr.open_group(path, mode="r")
        assert "g" not in store["container"]
        assert store.attrs["position_bounds"] == bounds_before

    def test_retry_after_late_failure_writes_decodable_arrays(
        self, tmp_path: Any
    ) -> None:
        from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node
        from luxar.encoding.decoder import ArrayDecoder

        compiler, scene, _ = open_scene(tmp_path, "retry_failed_graft.luxar.zarr")
        failed = self._late_failure_tree(invalid_selector=True)
        expected = failed.children[0].children[0].additive_sublods[0].centers

        refusal(lambda: graft_gsplat_node(scene, name="g", node=failed))
        graft_gsplat_node(
            scene, name="g", node=self._late_failure_tree(invalid_selector=False)
        )

        centers = compiler.store["g/part_0/child_0/centers"]
        assert centers.attrs["encoding"]["name"] != "array_ref"
        decoded = ArrayDecoder().decode(centers, zarr_root=compiler.store)
        assert decoded.shape == expected.shape

    def test_late_failure_does_not_poison_later_deduplication(
        self, tmp_path: Any
    ) -> None:
        from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node
        from luxar.encoding.decoder import ArrayDecoder

        compiler, scene, _ = open_scene(tmp_path, "dedup_after_failed_graft.luxar.zarr")
        failed = self._late_failure_tree(invalid_selector=True)
        data = failed.children[0].children[0].additive_sublods[0]

        refusal(lambda: graft_gsplat_node(scene, name="g", node=failed))
        scene.add_gsplats(
            "h",
            centers=data.centers,
            amplitudes=data.amplitudes,
            cholesky_factors=data.cholesky_factors,
        )

        centers = compiler.store["h/centers"]
        assert centers.attrs["encoding"]["name"] != "array_ref"
        decoded = ArrayDecoder().decode(centers, zarr_root=compiler.store)
        assert decoded.shape == data.centers.shape

    def test_finalized_writer_keeps_the_builder_error(self, tmp_path: Any) -> None:
        from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

        node = self._late_failure_tree(invalid_selector=False)
        expected_compiler, expected_scene, _ = open_scene(
            tmp_path, "finalized_expected.luxar.zarr"
        )
        expected_compiler.finalize()
        expected = refusal(
            lambda: graft_gsplat_node(
                expected_scene,
                name="g",
                node=node,
                _under_partition=False,
            )
        )
        actual_compiler, actual_scene, _ = open_scene(
            tmp_path, "finalized_actual.luxar.zarr"
        )
        actual_compiler.finalize()

        actual = refusal(lambda: graft_gsplat_node(actual_scene, name="g", node=node))

        assert type(actual) is type(expected)
        assert str(actual) == str(expected)
        assert "Cannot write_group after the writer has been finalized" in str(actual)

    def test_invalid_name_keeps_the_builder_validation_error(
        self, tmp_path: Any
    ) -> None:
        from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

        file_path = self._file(tmp_path, "invalid_name.gsplats.zarr")
        _, scene, _ = open_scene(tmp_path, "invalid_name_scene.luxar.zarr")

        expected = refusal(
            lambda: graft_gsplat_node(
                scene,
                name="../victim",
                node=_nested_partition_tree(3),
                _under_partition=False,
            )
        )
        actual = refusal(lambda: scene.add_gsplats_from_file("../victim", file_path))

        assert type(actual) is type(expected)
        assert str(actual) == str(expected)
        assert "node name: Name cannot contain '/'" in str(actual)

    def test_duplicate_name_failure_keeps_the_existing_node(
        self, tmp_path: Any
    ) -> None:
        file_path = self._file(tmp_path, "duplicate.gsplats.zarr")
        compiler, scene, _ = open_scene(tmp_path, "duplicate_graft.luxar.zarr")
        existing = scene.add_group("g", opacity=0.25)

        exc = refusal(lambda: scene.add_gsplats_from_file("g", file_path))

        assert "Duplicate child name 'g'" in str(exc)
        assert "g" in compiler.store
        assert compiler.store["g"].attrs["opacity"] == 0.25
        assert scene.children == [existing]
