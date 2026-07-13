"""Smoke tests for the pure image helpers in demo_gsplats_3d_visible_human_head.

Only the deterministic array helpers are exercised (no network, no PNG IO, no
GPU fit). The demo is loaded by file path (see test_demo_ppi_flow_field).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_visible_human_head.py"
)


def _load_demo_module():
    name = "_luxar_demo_vh_head_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
luminance = _demo.luminance
tissue_mask = _demo.tissue_mask
mask_background = _demo.mask_background
crop_to_content = _demo.crop_to_content
sample_colors = _demo.sample_colors
_save_colors_u8 = _demo._save_colors_u8
_load_colors_f32 = _demo._load_colors_f32


class TestColorRoundtrip:
    def test_uint8_roundtrip_within_quantization(self, tmp_path) -> None:
        colors = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
        p = tmp_path / "c.npz"
        _save_colors_u8(colors, p)
        assert p.stat().st_size > 0
        loaded = _load_colors_f32(p)
        assert loaded.dtype == np.float32
        assert loaded.min() >= 0.0 and loaded.max() <= 1.0
        np.testing.assert_allclose(loaded, colors, atol=1.0 / 255 + 1e-6)

    def test_loads_legacy_float_npz(self, tmp_path) -> None:
        p = tmp_path / "cf.npz"
        np.savez_compressed(p, colors=np.array([[0.2, 0.4, 0.6]], dtype=np.float32))
        loaded = _load_colors_f32(p)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, [[0.2, 0.4, 0.6]], atol=1e-6)


class TestLuminance:
    def test_white_and_black(self) -> None:
        rgb = np.array([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]], dtype=np.float32)
        lum = luminance(rgb)
        assert lum.shape == (1, 2)
        np.testing.assert_allclose(lum[0, 0], 1.0, atol=1e-6)
        np.testing.assert_allclose(lum[0, 1], 0.0, atol=1e-6)

    def test_rec601_weights(self) -> None:
        assert abs(float(luminance(np.array([[0.0, 1.0, 0.0]]))[0]) - 0.587) < 1e-6


class TestTissueMask:
    def test_warm_kept_blue_and_dark_rejected(self) -> None:
        rgb = np.array(
            [
                [0.8, 0.6, 0.4],  # warm tissue → keep
                [0.1, 0.2, 0.9],  # blue gel → reject
                [0.02, 0.02, 0.02],  # near-black → reject
            ],
            dtype=np.float32,
        )
        m = tissue_mask(rgb)
        assert list(m) == [True, False, False]

    def test_mask_background_zeros_nontissue(self) -> None:
        rgb = np.array([[[0.8, 0.6, 0.4], [0.1, 0.2, 0.9]]], dtype=np.float32)
        out = mask_background(rgb)
        np.testing.assert_allclose(out[0, 0], [0.8, 0.6, 0.4], atol=1e-6)
        np.testing.assert_array_equal(out[0, 1], [0.0, 0.0, 0.0])


class TestCropToContent:
    def test_crops_to_bounding_box(self) -> None:
        vol = np.zeros((3, 8, 8, 3), dtype=np.float32)
        vol[1, 2:5, 3:6] = [0.5, 0.3, 0.2]  # a small warm block
        cropped, box = crop_to_content(vol, pad=0)
        z0, z1, y0, y1, x0, x1 = box
        assert (z0, z1) == (1, 2)
        assert (y0, y1) == (2, 5)
        assert (x0, x1) == (3, 6)
        assert cropped.shape == (1, 3, 3, 3)

    def test_empty_volume_returns_unchanged(self) -> None:
        vol = np.zeros((2, 4, 4, 3), dtype=np.float32)
        cropped, box = crop_to_content(vol)
        assert cropped.shape == vol.shape


class TestSampleColors:
    def test_nearest_sample_and_clamp(self) -> None:
        vol = np.zeros((2, 2, 2, 3), dtype=np.float32)
        vol[0, 0, 0] = [1.0, 0.0, 0.0]
        vol[1, 1, 1] = [0.0, 0.0, 1.0]
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.4, 1.4, 1.4], [99.0, 99.0, 99.0]], dtype=np.float32
        )
        cols = sample_colors(vol, centers)
        assert cols.shape == (3, 3)
        np.testing.assert_allclose(cols[0], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(cols[1], [0.0, 0.0, 1.0])  # rounds to (1,1,1)
        np.testing.assert_allclose(cols[2], [0.0, 0.0, 1.0])  # clamped to (1,1,1)
