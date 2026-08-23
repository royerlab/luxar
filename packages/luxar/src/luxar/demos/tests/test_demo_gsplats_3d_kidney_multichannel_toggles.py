"""Smoke test for the kidney toggle demo's authored opening state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData

_DEMO_PATH = (
    Path(__file__).resolve().parents[1]
    / "demo_gsplats_3d_kidney_multichannel_toggles.py"
)


def _load_demo_module():
    name = "_luxar_demo_kidney_toggles_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_DEMO_MODULE = _load_demo_module()
create_luxar_scene = _DEMO_MODULE.create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats, sufficient to build the scene offline."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


def test_scene_opens_with_every_channel_on_and_rotating(tmp_path) -> None:
    gsplats = _tiny_gsplat_data()
    out = create_luxar_scene([gsplats] * 3, tmp_path / "kidney-toggles.luxar.zarr")
    attrs = dict(zarr.open_group(str(out), mode="r").attrs)

    dimensions = attrs["scene_dimensions"]["dimensions"]
    viewer_config = attrs["viewer_config"]
    current_step = viewer_config["dimensions"]["current_step"]

    assert len(current_step) == len(dimensions)
    dimensions_by_name = {
        dimension["name"]: (index, dimension)
        for index, dimension in enumerate(dimensions)
    }
    toggle_names = {channel["name"].lower() for channel in _DEMO_MODULE.CHANNELS}
    assert toggle_names <= dimensions_by_name.keys()

    for name in toggle_names:
        index, dimension = dimensions_by_name[name]
        categories = dimension["categories"]
        category_index = round(
            (current_step[index] - dimension["range"][0]) / dimension["step"]
        )
        assert categories[category_index] == "On"

    assert viewer_config["auto_rotate"] is True
