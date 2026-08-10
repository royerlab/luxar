"""Per-element channels are validated against the SOURCE count, pre-split (#1437).

``slice_optional_array`` passes a value through UNCHANGED when its leading length
does not match the element count — that is how a broadcast RGB triple or a scalar
radius reaches every part. A per-element array of the WRONG length takes the same
branch, so every part receives the whole unsliced array; and when a part's own
element count happens to equal that array's length, the part's writer accepts it.
The write then succeeds with values paired to the wrong elements.

Every wrong-length case below is paired with a plain-leaf control asserting the
flat path rejects the identical input, so the tests pin PARITY (a given input
fails the same way with and without ``partition=``) rather than merely "it
raises". #1422 closed this for ``labels``; these cover the rest.
"""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.io.reader import LuxarScene


def _make_3d_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("X", display=True),
            Dimension("Y", display=True),
            Dimension("Z", display=True),
        ]
    )


def _positions(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3)) * 100.0).astype(np.float32)


def _cholesky(n: int) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1))


def _scene(tmp_path, filename: str):
    path = str(tmp_path / filename)
    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=_make_3d_dims())
    return compiler, scene, path


# 200 elements split at max_elements=100 gives exactly two parts of 100
# (measured), so a 100-long channel matches every part's own count and only a
# check against the full 200 can reject it.
_N = 200
_HALF = 100

# (channel name, wrong-length value, error-message fragment)
_POINTS_CASES = [
    ("colors", np.zeros((_HALF, 3), dtype=np.float32), "colors"),
    ("radii", np.full(_HALF, 0.5, dtype=np.float32), "radii"),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32), "sharpness"),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32), "scalars"),
]

_LINES_CASES = [
    ("widths", np.full(_HALF, 0.2, dtype=np.float32), "widths"),
    ("colors", np.zeros((_HALF, 3), dtype=np.float32), "colors"),
    ("sharpness", np.full(_HALF, 0.5, dtype=np.float32), "sharpness"),
    ("scalars", np.linspace(0, 1, _HALF).astype(np.float32), "scalars"),
]


class TestPointsPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _POINTS_CASES)
    def test_wrong_length_channel_raises(self, tmp_path, channel, value, message):
        compiler, scene, _ = _scene(tmp_path, f"points_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points(
                "p",
                _positions(_N, seed=11),
                partition={"max_elements": _HALF},
                **kwargs,
            )

        assert "p" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _POINTS_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        """The parity control: identical input, no ``partition=``."""
        _, scene, _ = _scene(tmp_path, f"points_flat_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points("p", _positions(_N, seed=11), **kwargs)


class TestLinesPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _LINES_CASES)
    def test_wrong_length_channel_raises(self, tmp_path, channel, value, message):
        compiler, scene, _ = _scene(tmp_path, f"lines_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line",
                _positions(_N, seed=12),
                line_type="segments",
                partition={"max_elements": _HALF},
                **kwargs,
            )

        assert "line" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _LINES_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        _, scene, _ = _scene(tmp_path, f"lines_flat_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line", _positions(_N, seed=12), line_type="segments", **kwargs
            )


_GSPLAT_CASES = [
    ("amplitudes", np.full(_HALF, 1.0, dtype=np.float32), "[Aa]mplitudes"),
    ("cholesky_factors", _cholesky(_HALF), "Cholesky"),
    ("colors", np.zeros((_HALF, 3), dtype=np.float32), "colors"),
]


class TestGSplatsPartitionSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _GSPLAT_CASES)
    def test_wrong_length_channel_raises(self, tmp_path, channel, value, message):
        compiler, scene, _ = _scene(tmp_path, f"gsplats_{channel}.luxar.zarr")
        kwargs = {
            "amplitudes": 1.0,
            "cholesky_factors": _cholesky(_N),
            channel: value,
        }

        with pytest.raises(ValueError, match=message):
            scene.add_gsplats(
                "g",
                centers=_positions(_N, seed=13),
                partition={"max_elements": _HALF},
                **kwargs,
            )

        assert "g" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _GSPLAT_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        _, scene, _ = _scene(tmp_path, f"gsplats_flat_{channel}.luxar.zarr")
        kwargs = {
            "amplitudes": 1.0,
            "cholesky_factors": _cholesky(_N),
            channel: value,
        }

        with pytest.raises(ValueError, match=message):
            scene.add_gsplats("g", centers=_positions(_N, seed=13), **kwargs)


def _part_names(path: str, node: str) -> list[str]:
    store = zarr.open_group(path, mode="r")
    assert store[node].attrs["kind"] == "partition"
    return sorted(store[node].group_keys())


def _assert_uniform(actual, expected, atol: float = 5e-3) -> None:
    """Every row/value of a decoded broadcast channel equals ``expected``.

    The decoder expands a stored broadcast to full length on some channels and
    keeps the ``(1, c)`` / ``(1,)`` row on others, so the expectation is stated
    per element rather than by shape.
    """
    arr = np.asarray(actual, dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(arr, np.broadcast_to(want, arr.shape), atol=atol)


class TestLegalBroadcastFormsStillReachEveryPart:
    """The negative controls: every documented broadcast form still writes.

    The gate re-runs the flat writer's validators, which accept these; the
    collision cases additionally pin the CLASSIFICATION fix — a uniform value
    whose own length equals the element count must not be gathered as if it were
    per-element data. Each control reads the values back through the decoder, so
    it pins that they landed on the right elements rather than only "no raise".
    """

    def test_points_scalar_and_broadcast_channels(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "points_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            _positions(_N, seed=21),
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
            _assert_uniform(data.colors, [1.0, 0.0, 0.0])
            _assert_uniform(data.radii, 0.5)

    def test_points_rgb_triple_on_a_three_point_node(self, tmp_path):
        """The count collision: 3 points, a 3-component uniform RGB.

        Without the broadcast classification the triple satisfies
        ``slice_optional_array``'s length test and is gathered as if its three
        COMPONENTS were three point rows: each part then receives a 1-element
        slice of them (a list RGB is refused outright by the part's writer; a
        tuple RGB reaches it as a bogus 1-element color array).
        """
        compiler, scene, path = _scene(tmp_path, "points_rgb3.luxar.zarr")
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
        for part in parts:
            data = reader.get_points(f"p/{part}")
            _assert_uniform(data.colors, [0.25, 0.5, 1.0])

    def test_points_rgba_quadruple_on_a_four_point_node(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "points_rgba4.luxar.zarr")
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
        for part in parts:
            data = reader.get_points(f"p/{part}")
            assert data.colors.shape[-1] == 4
            _assert_uniform(data.colors, [0.25, 0.5, 1.0, 0.5])

    def test_points_one_element_broadcast_arrays(self, tmp_path):
        """A 1-element node with ``(1,)`` broadcast arrays.

        The other side of the collision: here the broadcast length and the
        element count agree, so slicing IS the right answer and must keep
        working (one part, no split fires).
        """
        compiler, scene, path = _scene(tmp_path, "points_one.luxar.zarr")
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
        _assert_uniform(data.colors, [0.0, 1.0, 0.0])
        _assert_uniform(data.radii, 0.7)

    def test_lines_scalar_width_and_uniform_color(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "lines_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            _positions(_N, seed=22),
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
            _assert_uniform(data.colors, [0.25, 0.5, 1.0])
            _assert_uniform(data.widths, 0.3)

    def test_lines_rgba_quadruple_on_a_four_vertex_node(self, tmp_path):
        """The Lines count collision: 4 vertices, a 4-component uniform RGBA.

        Two 2-vertex segments, one per part, so the RGBA tuple's own length
        matches the vertex count and ``slice_optional_array`` gathered it.
        """
        compiler, scene, path = _scene(tmp_path, "lines_rgba4.luxar.zarr")
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
        for part in parts:
            colors = reader.get_lines(f"line/{part}").colors
            assert colors.shape[-1] == 4
            _assert_uniform(colors, [0.25, 0.5, 1.0, 0.5])

    def test_lines_broadcast_width_array(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "lines_width1.luxar.zarr")
        scene.add_lines(
            "line",
            _positions(_N, seed=23),
            widths=np.array([0.25], dtype=np.float32),  # (1,) broadcast
            line_type="segments",
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        parts = _part_names(path, "line")
        assert len(parts) == 2
        reader = LuxarScene.load(path)
        for part in parts:
            _assert_uniform(reader.get_lines(f"line/{part}").widths, 0.25)

    def test_gsplats_uniform_cholesky_on_a_six_splat_node(self, tmp_path):
        """The other count collision: 6 splats, a uniform ``(6,)`` Cholesky.

        For 3-D data ``k = D(D+1)/2 = 6``, so the uniform form's own length
        equals the splat count and ``slice_optional_array`` gathered it as if it
        were per-splat — each part then getting a slice of the six Cholesky
        COMPONENTS as its per-splat rows.
        """
        compiler, scene, path = _scene(tmp_path, "gsplats_uniform_chol.luxar.zarr")
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
            seen += data.centers.shape[0]
            # Every splat in every part carries the authored uniform factor.
            for row in np.asarray(data.cholesky_factors).reshape(-1, 6):
                np.testing.assert_allclose(row, uniform, rtol=1e-3, atol=1e-3)
            _assert_uniform(data.amplitudes, 1.0)
        assert seen == 6

    def test_gsplats_rgb_triple_on_a_three_splat_node(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "gsplats_rgb3.luxar.zarr")
        centers = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_gsplats(
            "g",
            centers=centers,
            amplitudes=1.0,
            cholesky_factors=_cholesky(3),
            colors=(0.25, 0.5, 1.0),
            partition={"max_elements": 1},
        )
        compiler.finalize()

        parts = _part_names(path, "g")
        assert len(parts) > 1
        reader = LuxarScene.load(path)
        for part in parts:
            colors = reader.get_gsplats(f"g/{part}").colors
            _assert_uniform(colors, [0.25, 0.5, 1.0])

    def test_points_colormap_name_with_per_point_scalars(self, tmp_path):
        """A colormap NAME is an attr, not a channel — it rides to every part."""
        compiler, scene, path = _scene(tmp_path, "points_colormap.luxar.zarr")
        scene.add_points(
            "p",
            _positions(_N, seed=24),
            scalars=np.linspace(0.0, 1.0, _N).astype(np.float32),
            colormap="viridis",
            partition={"max_elements": _HALF},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        for part in store["p"].group_keys():
            assert store["p"][part].attrs["colormap"] == "viridis"
            assert store["p"][part]["scalars"].shape[0] == _HALF
