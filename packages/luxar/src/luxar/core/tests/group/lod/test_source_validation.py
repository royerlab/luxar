"""Per-element channels are validated against the SOURCE count, pre-split (#1437).

The LOD-wrapper half of the gate (the partition half lives in
``tests/group/partition/test_source_validation.py``). ``slice_optional_array``
passes a value whose leading length does not match the element count through
UNCHANGED — that is how a broadcast RGB triple or a scalar radius reaches every
level — so a per-element array of the WRONG length is handed to every level
whole, and a level whose own count happens to equal that array's length ACCEPTS
it. Each additive case below is sized so that coincidence holds (explicit
CUMULATIVE ``counts`` give two equal levels), which is what makes the per-level
check insufficient.

The substitutive wrappers are the shape where a downstream check does eventually
reject (their finest child carries the FULL element set), so those are pinned by
WHERE the write fails: the finest child is written LAST, after ``add_lod_group``
and every coarse gsplat level, so without the gate a wrong-length channel strands
a partial ``kind=lod`` node on disk.
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


def _scene(tmp_path, filename: str):
    path = str(tmp_path / filename)
    compiler = LuxarZarrCompiler(path)
    scene = compiler.create_scene(dimensions=_make_3d_dims())
    return compiler, scene, path


def _assert_uniform(actual, expected, atol: float = 5e-3) -> None:
    """Every row/value of a decoded broadcast channel equals ``expected``."""
    arr = np.asarray(actual, dtype=np.float64)
    want = np.asarray(expected, dtype=np.float64)
    np.testing.assert_allclose(arr, np.broadcast_to(want, arr.shape), atol=atol)


# 100 elements with cumulative counts [50, 100] gives two levels of 50, so a
# 50-long channel matches every level and only a check against the full 100 can
# reject it (the same construction the #1422 labels tests use).
_N = 100
_HALF = 50

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


class TestPointsAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _POINTS_CASES)
    def test_wrong_length_channel_raises(self, tmp_path, channel, value, message):
        compiler, scene, _ = _scene(tmp_path, f"points_add_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points(
                "p",
                _positions(_N, seed=31),
                additive_lod={"counts": [_HALF, _N]},
                **kwargs,
            )

        assert "p" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _POINTS_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        """The parity control: identical input, no ``additive_lod=``."""
        _, scene, _ = _scene(tmp_path, f"points_add_flat_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points("p", _positions(_N, seed=31), **kwargs)


class TestLinesAdditiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _LINES_CASES)
    def test_wrong_length_channel_raises(self, tmp_path, channel, value, message):
        compiler, scene, _ = _scene(tmp_path, f"lines_add_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line",
                _positions(_N, seed=32),
                line_type="segments",
                # Polyline units: each segment is a 2-vertex polyline, so the
                # cumulative [25, 50] gives two levels of 50 VERTICES each.
                additive_lod={"counts": [_HALF // 2, _N // 2]},
                **kwargs,
            )

        assert "line" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _LINES_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        _, scene, _ = _scene(tmp_path, f"lines_add_flat_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line", _positions(_N, seed=32), line_type="segments", **kwargs
            )


# The substitutive wrappers need enough elements for the gsplat reduce to
# synthesise coarse levels (a degenerate input falls back to a flat node).
_SUB_N = 400
_SUB_HALF = 200

_POINTS_SUB_CASES = [
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32), "colors"),
    ("radii", np.full(_SUB_HALF, 0.5, dtype=np.float32), "radii"),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32), "sharpness"),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32), "scalars"),
]

_LINES_SUB_CASES = [
    ("widths", np.full(_SUB_HALF, 0.2, dtype=np.float32), "widths"),
    ("colors", np.zeros((_SUB_HALF, 3), dtype=np.float32), "colors"),
    ("sharpness", np.full(_SUB_HALF, 0.5, dtype=np.float32), "sharpness"),
    ("scalars", np.linspace(0, 1, _SUB_HALF).astype(np.float32), "scalars"),
]


class TestPointsSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _POINTS_SUB_CASES)
    def test_wrong_length_channel_raises_before_anything_is_written(
        self, tmp_path, channel, value, message
    ):
        compiler, scene, _ = _scene(tmp_path, f"points_sub_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points(
                "p", _positions(_SUB_N, seed=41), substitutive_lod=True, **kwargs
            )

        # The gate runs above add_lod_group, so no partial kind=lod node is left.
        assert "p" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _POINTS_SUB_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        _, scene, _ = _scene(tmp_path, f"points_sub_flat_{channel}.luxar.zarr")
        kwargs = {channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_points("p", _positions(_SUB_N, seed=41), **kwargs)


class TestLinesSubstitutiveLodSourceValidation:
    @pytest.mark.parametrize("channel,value,message", _LINES_SUB_CASES)
    def test_wrong_length_channel_raises_before_anything_is_written(
        self, tmp_path, channel, value, message
    ):
        compiler, scene, _ = _scene(tmp_path, f"lines_sub_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line",
                _positions(_SUB_N, seed=42),
                line_type="segments",
                substitutive_lod=True,
                **kwargs,
            )

        assert "line" not in compiler.store

    @pytest.mark.parametrize("channel,value,message", _LINES_SUB_CASES)
    def test_flat_path_rejects_the_same_input(self, tmp_path, channel, value, message):
        _, scene, _ = _scene(tmp_path, f"lines_sub_flat_{channel}.luxar.zarr")
        kwargs = {"widths": 0.2, channel: value}
        if channel == "scalars":
            kwargs["colormap"] = "viridis"

        with pytest.raises(ValueError, match=message):
            scene.add_lines(
                "line", _positions(_SUB_N, seed=42), line_type="segments", **kwargs
            )


class TestLegalBroadcastFormsStillReachEveryLevel:
    """The negative controls: a legal broadcast still reaches every LOD level.

    A case the plain leaf accepts must not be refused by the ladder, and the
    values must land on the right elements — so each control reads every level
    back through the decoder.
    """

    def test_points_ladder_broadcast_channels(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "points_add_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            _positions(_N, seed=51),
            colors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),  # (1, 3) row
            radii=0.5,  # scalar
            sharpness=0.8,  # scalar
            additive_lod={"counts": [_HALF, _N]},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        n_levels = int(store["p"].attrs["n_additive_sublods"])
        assert n_levels > 1
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_points(f"p/additive_{i}")
            total += data.positions.shape[0]
            _assert_uniform(data.colors, [1.0, 0.0, 0.0])
            _assert_uniform(data.radii, 0.5)
        assert total == _N

    def test_points_ladder_rgb_triple_on_a_three_point_node(self, tmp_path):
        """The count collision, in the ladder: 3 points, a 3-component RGB.

        Without the broadcast classification the triple satisfies
        ``slice_optional_array``'s length test and each level receives a slice of
        its COMPONENTS instead of the authored colour.
        """
        compiler, scene, path = _scene(tmp_path, "points_add_rgb3.luxar.zarr")
        positions = np.array(
            [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]], dtype=np.float32
        )
        scene.add_points(
            "p", positions, colors=[0.25, 0.5, 1.0], additive_lod={"counts": [1, 3]}
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        n_levels = int(store["p"].attrs["n_additive_sublods"])
        assert n_levels > 1
        reader = LuxarScene.load(path)
        for i in range(n_levels):
            _assert_uniform(
                reader.get_points(f"p/additive_{i}").colors, [0.25, 0.5, 1.0]
            )

    def test_lines_ladder_broadcast_channels(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "lines_add_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            _positions(_N, seed=52),
            widths=0.3,  # scalar
            colors=(0.25, 0.5, 1.0),  # uniform RGB
            sharpness=0.7,  # scalar
            line_type="segments",
            additive_lod={"counts": [_HALF // 2, _N // 2]},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        n_levels = int(store["line"].attrs["n_additive_sublods"])
        assert n_levels > 1
        reader = LuxarScene.load(path)
        total = 0
        for i in range(n_levels):
            data = reader.get_lines(f"line/additive_{i}")
            total += data.vertices.shape[0]
            _assert_uniform(data.colors, [0.25, 0.5, 1.0])
            _assert_uniform(data.widths, 0.3)
        assert total == _N

    def test_lines_ladder_broadcast_width_array(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "lines_add_width1.luxar.zarr")
        scene.add_lines(
            "line",
            _positions(_N, seed=53),
            widths=np.array([0.25], dtype=np.float32),  # (1,) broadcast
            line_type="segments",
            additive_lod={"counts": [_HALF // 2, _N // 2]},
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        n_levels = int(store["line"].attrs["n_additive_sublods"])
        reader = LuxarScene.load(path)
        for i in range(n_levels):
            _assert_uniform(reader.get_lines(f"line/additive_{i}").widths, 0.25)

    def test_points_substitutive_broadcast_channels(self, tmp_path):
        # No uniform ``colors`` here: a broadcast RGB(A) under
        # ``substitutive_lod=`` is refused by the LIFT itself (it needs (N, 3)
        # per-element RGB to bake the coarse gsplat levels) — a separate,
        # pre-existing limitation that this gate neither creates nor changes.
        compiler, scene, path = _scene(tmp_path, "points_sub_broadcast.luxar.zarr")
        scene.add_points(
            "p",
            _positions(_SUB_N, seed=54),
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
        finest = children[-1]
        data = LuxarScene.load(path).get_points(f"p/{finest}")
        assert data.positions.shape[0] == _SUB_N
        _assert_uniform(data.radii, 0.5)

    def test_lines_substitutive_broadcast_channels(self, tmp_path):
        compiler, scene, path = _scene(tmp_path, "lines_sub_broadcast.luxar.zarr")
        scene.add_lines(
            "line",
            _positions(_SUB_N, seed=55),
            widths=0.3,  # scalar
            sharpness=0.7,  # scalar
            line_type="segments",
            substitutive_lod=True,
        )
        compiler.finalize()

        store = zarr.open_group(path, mode="r")
        assert store["line"].attrs["kind"] == "lod"
        children = sorted(store["line"].group_keys())
        assert len(children) > 1
        finest = children[-1]
        data = LuxarScene.load(path).get_lines(f"line/{finest}")
        assert data.vertices.shape[0] == _SUB_N
        _assert_uniform(data.widths, 0.3)
