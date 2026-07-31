"""Tests for the pure helpers in demo_gsplats_3d_milky_way_dust.

``normalize_dust_volume`` (no network, no h5py, no GPU fit) and the scene
builder's baked appearance, which was tuned against the shipped fit and would
otherwise regress silently. The demo is loaded by file path (see
test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_milky_way_dust.py"


def _load_demo_module():
    name = "_luxar_demo_milkyway_dust_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
normalize_dust_volume = _demo.normalize_dust_volume


class TestNormalizeDustVolume:
    def test_downsamples_to_target_and_normalizes(self) -> None:
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.0, 5.0, size=(16, 16, 12)).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_log_density_input_exponentiated(self) -> None:
        # Negative values ⇒ treated as log-density ⇒ exp() ⇒ all finite, non-negative.
        raw = np.linspace(-5.0, 2.0, 8 * 8 * 8).reshape(8, 8, 8).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_no_resize_when_target_matches(self) -> None:
        raw = np.abs(np.random.default_rng(1).normal(size=(8, 8, 8))).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)

    def test_native_resolution_when_target_nonpositive(self) -> None:
        # target_size <= 0 ⇒ keep the native (non-cubic) shape, no downscale.
        raw = np.abs(np.random.default_rng(3).normal(size=(9, 7, 5))).astype(np.float32)
        out = normalize_dust_volume(raw, target_size=0)
        assert out.shape == (9, 7, 5)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_deterministic(self) -> None:
        raw = np.abs(np.random.default_rng(2).normal(size=(10, 10, 10))).astype(
            np.float32
        )
        a = normalize_dust_volume(raw, target_size=6)
        b = normalize_dust_volume(raw, target_size=6)
        np.testing.assert_array_equal(a, b)


def test_scene_bakes_tuned_dust_appearance(tmp_path: Path) -> None:
    """The dust node ships the appearance tuned against the shipped fit."""
    import zarr

    from luxar.gsplats.gsplat_data import GSplatData

    rng = np.random.default_rng(0)
    n = 32
    data = GSplatData(
        centers=rng.uniform(-10.0, 10.0, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=np.tile(
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32), (n, 1)
        ),
    )

    out = _demo.create_luxar_scene(data, tmp_path / "dust.luxar.zarr")

    scene = zarr.open(str(out), mode="r")
    viewer_config = scene.attrs["viewer_config"]
    assert viewer_config["tone_mapping"] == "ACES"
    assert viewer_config["exposure"] == pytest.approx(-0.17)

    dust = dict(scene["interstellar_dust"].attrs)
    assert dust["colormap"] == "inferno"
    assert dust["blending_mode"] == "volumetric"
    assert dust["absorption"] == pytest.approx(0.3)
    # Display window [0, 0.095] — the Layers-panel gain/offset encoding.
    assert dust["intensity"] == pytest.approx(1.0 / 0.095)
