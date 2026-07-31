"""Smoke tests for demo_gsplats_3d_cryoem_virus.

Covers ``normalize_map_volume`` and the scene builder's authored blending — no
network, no mrcfile IO, no GPU fit. The demo is loaded by file path (see
test_demo_ppi_flow_field for the rationale).
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

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_cryoem_virus.py"


def _load_demo_module():
    name = "_luxar_demo_cryoem_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
normalize_map_volume = _demo.normalize_map_volume
create_luxar_scene = _demo.create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats — no GPU fit, enough to build the scene."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


class TestSceneBlending:
    def test_scene_bakes_volumetric_blending(self, tmp_path) -> None:
        # The capsid reads as a hollow shell only under volumetric
        # (emission-absorption) compositing with strong absorption; pin both
        # so a silent revert to additive glow is caught (the helper smoke
        # test never builds the scene). See the interop demos' blending test.
        out = create_luxar_scene(_tiny_gsplat_data(), tmp_path / "virus.luxar.zarr")
        node = zarr.open_group(str(out), mode="r")["virus_capsid"]
        assert dict(node.attrs).get("blending_mode") == "volumetric"
        assert dict(node.attrs).get("absorption") == 5.0


class TestNormalizeMapVolume:
    def test_clips_negative_solvent_and_normalizes(self) -> None:
        raw = np.linspace(-2.0, 8.0, 8 * 8 * 8).reshape(8, 8, 8).astype(np.float32)
        out = normalize_map_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)
        assert out.dtype == np.float32
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_downsamples_to_target(self) -> None:
        raw = np.abs(np.random.default_rng(0).normal(size=(16, 16, 16))).astype(
            np.float32
        )
        out = normalize_map_volume(raw, target_size=8)
        assert out.shape == (8, 8, 8)

    def test_native_when_target_nonpositive(self) -> None:
        raw = np.abs(np.random.default_rng(1).normal(size=(9, 7, 5))).astype(np.float32)
        out = normalize_map_volume(raw, target_size=0)
        assert out.shape == (9, 7, 5)

    def test_deterministic(self) -> None:
        raw = np.abs(np.random.default_rng(2).normal(size=(10, 10, 10))).astype(
            np.float32
        )
        np.testing.assert_array_equal(
            normalize_map_volume(raw, 6), normalize_map_volume(raw, 6)
        )
