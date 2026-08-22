"""Scene-appearance regressions for the FlyWire connectome demo."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest
import zarr

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_flywire_connectome.py"


def _load_demo_module():
    name = "_luxar_demo_flywire_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def test_neuron_layer_bakes_dim_additive_appearance(tmp_path: Path) -> None:
    neurons = pd.DataFrame(
        {
            "x_um": [1.0],
            "y_um": [2.0],
            "z_um": [3.0],
            "super_class": ["optic"],
            "cell_type": ["test neuron"],
            "top_nt": ["ACh"],
            "side": ["left"],
        }
    )
    edges = pd.DataFrame({"syn_count": [], "nt": []})
    out = tmp_path / "flywire.luxar.zarr"

    assert _demo.build_scene(out, neurons, edges) == (1, 0)

    attrs = dict(zarr.open_group(str(out), mode="r")["Neurons — optic"].attrs)
    assert attrs["intensity"] == pytest.approx(1.0 / 270.91)
    assert attrs["opacity"] == pytest.approx(0.79)
    assert attrs["blending_mode"] == "additive"
