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


def test_every_channel_is_volumetric(channel_nodes: dict[str, dict]) -> None:
    """All three stains composite volumetrically (emission-absorption).

    This replaces the earlier all-``additive`` rule (#1964). Additive was chosen
    then because the three co-located channels had no authored draw order and the
    viewer had to infer one; being commutative, additive sidestepped the question
    rather than answering it. ``depth_level`` answers it (see
    ``test_every_channel_states_its_depth_level``), so the specimen can occlude
    itself the way tissue does instead of every stain summing into a flat glow.
    """
    assert len(channel_nodes) == 3
    for name, attrs in channel_nodes.items():
        assert attrs.get("blending_mode") == "volumetric", name


def test_every_channel_states_its_depth_level(channel_nodes: dict[str, dict]) -> None:
    """The order is AUTHORED, not inferred — which is what makes volumetric safe.

    Higher draws nearer the camera, matching the order the Layers panel lists the
    channels in (top of the list is in front). Pinned as exact values because the
    whole point is that they cannot drift: without them the order comes from
    bounding-sphere containment, and the vasculature/cardiac-tissue spheres differ
    by 2.2 units out of 1130 (0.19%), so a refit could silently swap that pair.
    """
    expected = {
        "gsplats_sytox_green_nuclei": 3,
        "gsplats_tomato_lectin_vasculature": 2,
        "gsplats_tnni3_cardiac_tissue": 1,
    }
    actual = {name: attrs.get("depth_level") for name, attrs in channel_nodes.items()}
    assert actual == expected

    # Distinct levels, or two channels share a band and the inference decides
    # between them again — the exact thing the levels exist to prevent.
    assert len(set(actual.values())) == 3


def test_every_channel_keeps_its_authored_opacity(
    channel_nodes: dict[str, dict],
) -> None:
    """The dense nuclear stain stays below the two structural channels."""
    expected = {
        "gsplats_sytox_green_nuclei": 0.12,
        "gsplats_tomato_lectin_vasculature": 0.30,
        "gsplats_tnni3_cardiac_tissue": 0.16,
    }
    assert set(channel_nodes) == set(expected)
    for name, opacity in expected.items():
        assert channel_nodes[name].get("opacity") == pytest.approx(opacity)
