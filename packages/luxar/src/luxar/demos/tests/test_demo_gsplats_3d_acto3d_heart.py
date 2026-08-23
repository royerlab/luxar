"""Blending tests for demo_gsplats_3d_acto3d_heart.

Builds the scene from three tiny synthetic channels — no network, no 1.65 GB
TIFF, no GPU fit. The demo is loaded by file path (see
test_demo_gsplats_3d_cryoem_virus for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_3d_acto3d_heart.py"


def _load_demo_module():
    name = "_luxar_demo_acto3d_heart_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
CHANNELS = _demo.CHANNELS
create_luxar_scene = _demo.create_luxar_scene


def _tiny_gsplat_data(n: int = 8, seed: int = 0) -> GSplatData:
    """A handful of valid splats — no GPU fit, enough to build the scene."""
    rng = np.random.default_rng(seed)
    return GSplatData(
        centers=rng.uniform(-4.0, 4.0, (n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.2, 1.0, n).astype(np.float32),
        cholesky_factors=np.tile([1, 0, 1, 0, 0, 1], (n, 1)).astype(np.float32),
    )


@pytest.fixture(scope="module")
def channel_nodes(tmp_path_factory) -> dict[str, dict]:
    """Attributes of the three per-channel layer nodes of a built scene."""
    out = create_luxar_scene(
        [_tiny_gsplat_data(seed=i) for i in range(len(CHANNELS))],
        tmp_path_factory.mktemp("acto3d") / "heart.luxar.zarr",
    )
    root = zarr.open_group(str(out), mode="r")
    names = [
        "gsplats_"
        + ch["name"].lower().replace(" ", "_").replace("(", "").replace(")", "")
        for ch in CHANNELS
    ]
    return {name: dict(root[name].attrs) for name in names}


def test_every_channel_is_additive(channel_nodes: dict[str, dict]) -> None:
    """All three stains composite additively — none may occlude another.

    The channels are independent fluorescence emitters of one specimen, so a
    volumetric revert (which lets the nuclear stain in front attenuate the
    vasculature and cardiac tissue behind it) would invent a depth cue the
    data does not carry.
    """
    assert len(channel_nodes) == 3
    for name, attrs in channel_nodes.items():
        assert attrs.get("blending_mode") == "additive", name


def test_every_channel_keeps_its_partial_opacity(
    channel_nodes: dict[str, dict],
) -> None:
    """An unbounded sum of three channels clips without partial opacity."""
    for name, attrs in channel_nodes.items():
        assert attrs.get("opacity") == pytest.approx(0.48), name
