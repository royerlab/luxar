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
            # `load_neurons` reads root_id as the frame's key (Int64), so the
            # fixture carries it too — without it this test exercised a shape
            # the demo never actually receives.
            "root_id": pd.array([720575940621039145], dtype="Int64"),
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
    # Clicking a soma opens its Codex cell page (#1917). The root id is a
    # 19-digit segment id that appears nowhere in the hover label, so it rides
    # the keys channel.
    assert attrs["link"] == (
        "https://codex.flywire.ai/app/cell_details?root_id={hover_key}"
    )
    assert attrs["copy"] == "{hover_key}"
    assert attrs["has_keys"] is True


def test_neuron_layer_without_root_id_still_builds(tmp_path: Path) -> None:
    """A frame with no ``root_id`` loses the Codex links, not the whole demo.

    The column is the frame's key upstream, so its absence means the source
    schema changed — worth degrading for rather than aborting a scene build over
    a link.
    """
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
    assert "link" not in attrs
    assert not attrs.get("has_keys")
    # The layer itself is intact.
    assert attrs["blending_mode"] == "additive"


def test_connection_lines_stay_a_faint_glow() -> None:
    """Pin the connection layers' gain, read off the `add_lines` call.

    300K luminous lines accumulate along every ray; at `intensity=0.08` the
    hosted demo rendered the whole brain as one blown-out white blob that hid
    the neuron layers (2026-09-10). 0.003 was chosen from a live sweep: 0.008
    still washed every super-class colour, 0.004 began to pale the optic lobes.
    Parsed from the source because the scene-building test above runs with an
    empty edge frame and never reaches the lines call.
    """
    import ast

    tree = ast.parse(_DEMO_PATH.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_lines"
        ):
            kwargs = {
                kw.arg: kw.value.value
                for kw in node.keywords
                if kw.arg is not None and isinstance(kw.value, ast.Constant)
            }
            assert kwargs["blending_mode"] == "luminous"
            assert kwargs["opacity"] == pytest.approx(0.08)
            assert kwargs["intensity"] == pytest.approx(0.003)
            return
    raise AssertionError("no add_lines call found in the demo")
