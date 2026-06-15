"""Tests for ``scalars_to_colors`` (authoring-time colormap application)."""

import numpy as np
import pytest

from luxar.colormaps import resolve_colormap, scalars_to_colors


def test_endpoints_match_lut():
    # min scalar -> LUT[0], max scalar -> LUT[255] (viridis: dark purple -> yellow).
    s = np.array([0.0, 1.0], dtype=np.float32)
    c = scalars_to_colors(s, "viridis")
    lut = resolve_colormap("viridis").astype(np.float64) / 255.0
    np.testing.assert_allclose(c[0], lut[0], atol=1e-6)
    np.testing.assert_allclose(c[1], lut[255], atol=1e-6)


def test_shape_dtype_and_range():
    s = np.linspace(0, 10, 50).astype(np.float32)
    c = scalars_to_colors(s, "viridis")
    assert c.shape == (50, 3)
    assert c.dtype == np.float32
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_vmin_vmax_clamping():
    # Values outside [vmin, vmax] clamp to the LUT ends.
    s = np.array([-5.0, 0.0, 5.0, 10.0, 99.0], dtype=np.float32)
    c = scalars_to_colors(s, "viridis", vmin=0.0, vmax=10.0)
    lut = resolve_colormap("viridis").astype(np.float64) / 255.0
    np.testing.assert_allclose(c[0], lut[0], atol=1e-6)   # -5 clamps to min
    np.testing.assert_allclose(c[-1], lut[255], atol=1e-6)  # 99 clamps to max


def test_degenerate_constant_scalars_maps_to_lut_centre():
    s = np.full(8, 3.0, dtype=np.float32)
    c = scalars_to_colors(s, "viridis")
    lut = resolve_colormap("viridis").astype(np.float64) / 255.0
    # all-equal -> 0-width range -> LUT centre, all rows identical
    assert np.allclose(c, c[0])
    np.testing.assert_allclose(c[0], lut[round(0.5 * 255)], atol=1e-6)


def test_default_vmin_vmax_from_data():
    s = np.array([2.0, 4.0, 6.0], dtype=np.float32)  # range [2,6]
    c = scalars_to_colors(s, "viridis")
    lut = resolve_colormap("viridis").astype(np.float64) / 255.0
    np.testing.assert_allclose(c[0], lut[0], atol=1e-6)   # 2 -> min
    np.testing.assert_allclose(c[-1], lut[255], atol=1e-6)  # 6 -> max


def test_accepts_explicit_lut_array():
    s = np.array([0.0, 1.0], dtype=np.float32)
    lut_arr = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)  # resampled to 256
    c = scalars_to_colors(s, lut_arr)
    np.testing.assert_allclose(c[0], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(c[1], [1, 1, 1], atol=1e-6)


def test_unknown_colormap_raises():
    with pytest.raises(ValueError):
        scalars_to_colors(np.array([0.0, 1.0]), "definitely-not-a-colormap")
