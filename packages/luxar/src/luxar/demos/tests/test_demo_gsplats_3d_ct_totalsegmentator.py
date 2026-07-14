"""Smoke tests for the pure helpers in demo_gsplats_3d_ct_totalsegmentator.

Only deterministic array helpers are exercised (no network, no nibabel IO, no
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
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_ct_totalsegmentator.py"
)


def _load_demo_module():
    name = "_luxar_demo_ct_totalseg_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
tissue_group = _demo.tissue_group
organ_palette = _demo.organ_palette
window_ct = _demo.window_ct
sample_labels = _demo.sample_labels
label_colors = _demo.label_colors
crop_to_content = _demo.crop_to_content
_save_colors_u8 = _demo._save_colors_u8
_load_colors_f32 = _demo._load_colors_f32
CLASS_MAP = _demo.CLASS_MAP


class TestTissueGroup:
    def test_known_structures(self) -> None:
        assert tissue_group("liver") == "abdominal_organ"
        assert tissue_group("aorta") == "vessel"
        assert tissue_group("iliac_vena_left") == "vessel"
        assert tissue_group("lung_upper_lobe_left") == "lung"
        assert tissue_group("vertebrae_L3") == "bone"
        assert tissue_group("rib_right_5") == "bone"
        assert tissue_group("heart") == "heart"
        assert tissue_group("gluteus_maximus_left") == "muscle"
        assert tissue_group("brain") == "brain"
        assert tissue_group("spinal_cord") == "spinal"
        assert tissue_group("colon") == "gi"
        assert tissue_group("kidney_left") == "urinary"

    def test_every_class_maps_to_a_known_group(self) -> None:
        groups = {tissue_group(n) for n in CLASS_MAP.values()}
        assert groups <= set(_demo.GROUP_COLORS.keys())


class TestOrganPalette:
    def test_shape_and_gamut(self) -> None:
        pal = organ_palette()
        assert pal.shape == (118, 3)
        assert pal.dtype == np.float32
        assert pal.min() >= 0.0 and pal.max() <= 1.0
        # background distinct from any organ
        assert not any(np.allclose(pal[0], pal[i]) for i in range(1, 118))

    def test_deterministic(self) -> None:
        np.testing.assert_array_equal(organ_palette(), organ_palette())


class TestWindowCt:
    def test_clip_and_normalize(self) -> None:
        ct = np.array(
            [-1000.0, _demo.HU_LO, 0.0, _demo.HU_HI, 3000.0], dtype=np.float32
        )
        out = window_ct(ct)
        assert out.min() == 0.0 and out.max() == 1.0
        assert out[0] == 0.0 and out[1] == 0.0  # below/at lo → 0
        assert out[3] == 1.0 and out[4] == 1.0  # at/above hi → 1
        assert 0.0 < out[2] < 1.0


class TestSampleLabelsAndColors:
    def test_nearest_label_and_clamp(self) -> None:
        vol = np.zeros((2, 2, 2), dtype=np.int32)
        vol[0, 0, 0] = 5  # liver
        vol[1, 1, 1] = 52  # aorta
        centers = np.array(
            [[0.0, 0.0, 0.0], [1.4, 1.4, 1.4], [99.0, 99.0, 99.0]], dtype=np.float32
        )
        ids = sample_labels(vol, centers)
        assert ids.tolist() == [5, 52, 52]  # 3rd clamps into (1,1,1)

    def test_label_colors_via_palette(self) -> None:
        pal = organ_palette()
        cols = label_colors(np.array([0, 5, 52], dtype=np.int32), pal)
        assert cols.shape == (3, 3) and cols.dtype == np.float32
        np.testing.assert_allclose(cols[1], pal[5])
        np.testing.assert_allclose(cols[2], pal[52])


class TestCropToContent:
    def test_bbox(self) -> None:
        m = np.zeros((10, 10, 10), dtype=bool)
        m[3:6, 4:5, 2:8] = True
        z0, z1, y0, y1, x0, x1 = crop_to_content(m, pad=0)
        assert (z0, z1, y0, y1, x0, x1) == (3, 6, 4, 5, 2, 8)

    def test_empty(self) -> None:
        m = np.zeros((4, 4, 4), dtype=bool)
        assert crop_to_content(m) == (0, 4, 0, 4, 0, 4)


class TestColorRoundtrip:
    def test_uint8_roundtrip(self, tmp_path) -> None:
        colors = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.1]], dtype=np.float32)
        p = tmp_path / "c.npz"
        _save_colors_u8(colors, p)
        loaded = _load_colors_f32(p)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded, colors, atol=1.0 / 255 + 1e-6)
