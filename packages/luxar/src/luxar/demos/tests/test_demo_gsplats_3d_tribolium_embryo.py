"""Smoke test for the Tribolium embryo demo's authored appearance."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_tribolium_embryo.py"


def _load_demo_module():
    name = "_luxar_demo_tribolium_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


create_luxar_scene = _load_demo_module().create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats, sufficient to build the scene offline."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def test_scene_bakes_volumetric_appearance(tmp_path) -> None:
    out = create_luxar_scene(_tiny_gsplat_data(), tmp_path / "tribolium.luxar.zarr")
    node = zarr.open_group(str(out), mode="r")["tribolium_embryo"]
    attrs = dict(node.attrs)

    assert attrs["blending_mode"] == "volumetric"
    assert attrs["absorption"] == pytest.approx(3.13)
    assert attrs["opacity"] == pytest.approx(0.06)
    assert attrs["intensity"] == pytest.approx(1.0 / 1.085)
