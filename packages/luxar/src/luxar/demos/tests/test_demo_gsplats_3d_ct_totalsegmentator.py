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
import zarr

from luxar.gsplats.gsplat_data import GSplatData

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
splat_layer_indices = _demo.splat_layer_indices
organ_label_text = _demo.organ_label_text
crop_to_content = _demo.crop_to_content
_save_labels_u8 = _demo._save_labels_u8
_load_labels = _demo._load_labels
CLASS_MAP = _demo.CLASS_MAP
SUPERGROUPS = _demo.SUPERGROUPS
create_luxar_scene = _demo.create_luxar_scene


def _tiny_gsplat_data(n: int, seed: int = 0) -> GSplatData:
    """A handful of valid splats — no GPU fit, enough to build the scene."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def _gsplat_layers(scene_path: Path) -> list:
    """Attr dicts of the scene's top-level gsplats layer nodes."""
    root = zarr.open_group(str(scene_path), mode="r")
    return [
        dict(group.attrs)
        for _name, group in root.groups()
        if dict(group.attrs).get("type") == "gsplats"
    ]


class TestSceneBlending:
    def test_scene_bakes_volumetric_blending_on_every_layer(self, tmp_path) -> None:
        # Every per-tissue toggle layer must composite volumetrically
        # (emission-absorption) so organs read through one another instead of
        # summing to additive glow; pin it so a silent revert is caught (the
        # helper smoke tests never build the scene). Use two tissue groups so
        # more than one layer is exercised.
        bone = next(
            lid for lid, name in CLASS_MAP.items() if tissue_group(name) == "bone"
        )
        liver = next(
            lid
            for lid, name in CLASS_MAP.items()
            if tissue_group(name) == "abdominal_organ"
        )
        labels = np.array([bone, liver] * 4, dtype=np.int32)
        fit = _tiny_gsplat_data(len(labels))
        out = create_luxar_scene(fit, labels, tmp_path / "ct.luxar.zarr")
        layers = _gsplat_layers(out)
        assert len(layers) >= 2
        for attrs in layers:
            assert attrs.get("blending_mode") == "volumetric"


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


class TestLayerSplit:
    def test_every_label_maps_to_a_layer(self) -> None:
        # every one of the 117 structures lands in exactly one supergroup
        idx = splat_layer_indices(np.arange(1, 118, dtype=np.int32))
        assert idx.min() >= 0 and idx.max() < len(SUPERGROUPS)
        # background label 0 → -1 (no layer)
        assert splat_layer_indices(np.array([0]))[0] == -1

    def test_known_label_layers(self) -> None:
        names = [g[0] for g in SUPERGROUPS]
        skeleton = names.index("Skeleton")
        organs = names.index("Organs")
        muscles = names.index("Muscles")
        # 5=liver→Organs, 27=vertebrae_L5→Skeleton, 80=gluteus→Muscles
        got = splat_layer_indices(np.array([5, 27, 80], dtype=np.int32))
        assert got.tolist() == [organs, skeleton, muscles]

    def test_muscle_layer_semi_transparent_and_boosted(self) -> None:
        muscle = next(g for g in SUPERGROUPS if g[0] == "Muscles")
        assert 0.0 < muscle[2] < 1.0  # semi-transparent so organs read through
        assert muscle[3] >= 2.0  # amplitude boost so low-HU muscle is visible

    def test_layer_tuple_shape(self) -> None:
        # (name, tissue groups, opacity, amplitude boost)
        for name, groups, opacity, boost in SUPERGROUPS:
            assert isinstance(name, str) and len(groups) >= 1
            assert 0.0 < opacity <= 1.0 and boost >= 1.0


class TestOrganLabelText:
    def test_pretty_names(self) -> None:
        assert organ_label_text(5) == "Liver"
        assert organ_label_text(2) == "Kidney right"
        assert organ_label_text(52) == "Aorta"
        assert organ_label_text(0) == ""


class TestLabelRoundtrip:
    def test_uint8_roundtrip(self, tmp_path) -> None:
        labels = np.array([0, 5, 52, 117, 80], dtype=np.int32)
        p = tmp_path / "labels.npz"
        _save_labels_u8(labels, p)
        loaded = _load_labels(p)
        assert loaded.dtype == np.int32
        np.testing.assert_array_equal(loaded, labels)
