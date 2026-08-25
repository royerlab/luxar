"""Tests for shared demo color helpers."""

import numpy as np
import pytest

from luxar.utils import colors as color_utils


def test_stack_colorings_shapes_and_alignment():
    n = 5
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    red = np.tile([1.0, 0.0, 0.0], (n, 1)).astype(np.float32)
    blue = np.tile([0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    out = color_utils.stack_colorings(
        coords,
        [
            {"label": "A", "colors": red, "labels": [f"a{i}" for i in range(n)]},
            {"label": "B", "colors": blue, "labels": [f"b{i}" for i in range(n)]},
        ],
    )
    assert out.positions.shape == (2 * n, 4)  # leading coloring-index column
    assert out.colors.shape == (2 * n, 3)
    assert out.categories == ["A", "B"]
    # Block 0 = coloring index 0 (red), block 1 = index 1 (blue).
    assert (out.positions[:n, 0] == 0).all() and (out.positions[n:, 0] == 1).all()
    np.testing.assert_array_equal(out.positions[:n, 1:], coords)
    np.testing.assert_array_equal(out.colors[:n], red)
    np.testing.assert_array_equal(out.colors[n:], blue)
    assert out.labels == [f"a{i}" for i in range(n)] + [f"b{i}" for i in range(n)]


def test_stack_colorings_keys_tile_with_the_blocks():
    """`keys` (#1917) is one list for the whole cloud, tiled once per coloring.

    Labels are per-view because the hover text changes with the colour scheme;
    a point's machine-readable identity does not, so passing keys per view would
    invite two views to disagree about what a point IS. Tiled here so it stays
    aligned with the stacked positions — the alignment this helper owns.
    """
    n = 4
    coords = np.arange(n * 3, dtype=np.float32).reshape(n, 3)
    c = np.zeros((n, 3), dtype=np.float32)
    keys = [f"P{i}" for i in range(n)]
    out = color_utils.stack_colorings(
        coords,
        [
            {"label": "A", "colors": c, "labels": [f"a{i}" for i in range(n)]},
            {"label": "B", "colors": c, "labels": [f"b{i}" for i in range(n)]},
        ],
        keys=keys,
    )
    assert out.keys == keys + keys
    assert len(out.keys) == len(out.positions)
    # Same point, same key, whichever block it is in.
    for i in range(n):
        assert out.keys[i] == out.keys[i + n]


def test_stack_colorings_keys_default_to_none():
    """Every existing caller omits `keys`, and must keep getting a node with no
    keys channel rather than an empty one."""
    n = 3
    coords = np.zeros((n, 3), dtype=np.float32)
    c = np.zeros((n, 3), dtype=np.float32)
    out = color_utils.stack_colorings(coords, [{"label": "A", "colors": c}])
    assert out.keys is None


def test_stack_colorings_rejects_a_wrong_length_keys():
    """A short keys list would tile into a shape that still looks plausible, so
    it is refused here rather than misaligned downstream."""
    n = 4
    coords = np.zeros((n, 3), dtype=np.float32)
    c = np.zeros((n, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="keys has 2 entries"):
        color_utils.stack_colorings(
            coords, [{"label": "A", "colors": c}], keys=["a", "b"]
        )


def test_stack_colorings_labels_none_if_any_view_missing():
    n = 3
    coords = np.zeros((n, 3), dtype=np.float32)
    c = np.zeros((n, 3), dtype=np.float32)
    out = color_utils.stack_colorings(
        coords,
        [
            {"label": "A", "colors": c, "labels": ["x", "y", "z"]},
            {"label": "B", "colors": c},  # no labels → combined labels is None
        ],
    )
    assert out.labels is None


def test_stack_colorings_rejects_bad_shape():
    coords = np.zeros((4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        color_utils.stack_colorings(
            coords, [{"label": "A", "colors": np.zeros((3, 3), dtype=np.float32)}]
        )


def test_stack_colorings_builds_a_real_scene(tmp_path):
    """End-to-end: a stacked coloring cloud writes a valid scene with a
    categorical `coloring` dimension (the pattern all 4 landscape demos use)."""
    from luxar import Dimension, Dimensions, LuxarZarrCompiler

    n = 40
    rng = np.random.default_rng(0)
    coords = rng.standard_normal((n, 3)).astype(np.float32)
    out = color_utils.stack_colorings(
        coords,
        [
            {
                "label": "View A",
                "colors": rng.random((n, 3)).astype(np.float32),
                "labels": [f"a{i}" for i in range(n)],
            },
            {
                "label": "View B",
                "colors": rng.random((n, 3)).astype(np.float32),
                "labels": [f"b{i}" for i in range(n)],
            },
        ],
    )
    scene_path = tmp_path / "coloring.luxar.zarr"
    dims = Dimensions(
        [
            Dimension("coloring", unit="", categories=out.categories, display=False),
            Dimension("x", unit="u", display=True),
            Dimension("y", unit="u", display=True),
            Dimension("z", unit="u", display=True),
        ]
    )
    with LuxarZarrCompiler(scene_path) as compiler:
        scene = compiler.create_scene(dimensions=dims)
        scene.add_points(
            "points",
            out.positions,
            colors=out.colors,
            radii=np.full(len(out.positions), 0.02, np.float32),
            labels=out.labels,
        )
    assert scene_path.exists()


def test_hsv_to_rgb_primaries_and_shape():
    # Red at hue 0, green at 1/3, blue at 2/3.
    hues = np.array([0.0, 1 / 3, 2 / 3], dtype=np.float32)
    rgb = color_utils.hsv_to_rgb(hues)
    assert rgb.shape == (3, 3)
    np.testing.assert_allclose(rgb[0], [1, 0, 0], atol=1e-5)
    np.testing.assert_allclose(rgb[1], [0, 1, 0], atol=1e-5)
    np.testing.assert_allclose(rgb[2], [0, 0, 1], atol=1e-5)
    # Value/saturation scaling.
    grey = color_utils.hsv_to_rgb(np.array([0.0]), s=0.0, v=0.5)
    np.testing.assert_allclose(grey[0], [0.5, 0.5, 0.5], atol=1e-5)
