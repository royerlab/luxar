"""Regression tests for the Drosophila embryogenesis timelapse demo."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatLeaf

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_4d_drosophila_embryogenesis.py"
)
_MANIFEST_PATH = _DEMO_PATH.parent / "data_manifest.json"
_ARCHIVE_SHA256 = "ee3193babb074e16958665bb96e880dea7d9af695b4521c1c2fbb81991f5b393"
_ARCHIVE_BYTES = 1_158_979_910


def _load_demo_module(name: str = "_luxar_demo_drosophila_embryogenesis_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def _sub_lod(times: list[float], offset: float = 0.0) -> AdditiveSubLOD:
    n_splats = len(times)
    centers = np.column_stack(
        (
            np.linspace(offset, offset + 1.0, n_splats),
            np.linspace(0.0, 2.0, n_splats),
            np.linspace(0.0, 1.0, n_splats),
            np.asarray(times),
        )
    ).astype(np.float32)
    cholesky = np.zeros((n_splats, 10), dtype=np.float32)
    cholesky[:, [0, 2, 5, 9]] = 1.0
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=np.linspace(0.2, 1.0, n_splats, dtype=np.float32),
        cholesky_factors=cholesky,
    )


def _node(times: list[float], *, rungs: int = 1) -> GSplatLeaf:
    return GSplatLeaf(
        additive_sublods=[_sub_lod(times, float(rung)) for rung in range(rungs)]
    )


def test_manifest_and_download_size_pin_the_500tp_archive() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text())
    pin = manifest["datasets"][_demo.DATASET]["files"][0]

    assert (pin["sha256"], pin["bytes"]) == (_ARCHIVE_SHA256, _ARCHIVE_BYTES)
    assert _demo.DEMO_META["requirements"]["download_mb"] == round(
        _ARCHIVE_BYTES / 1024**2
    )


def test_normalise_time_axis_scales_frame_indices_across_every_rung() -> None:
    node = _node([0.0, 1.0, 2.0], rungs=2)
    node.additive_sublods[1].centers.flags.writeable = False

    assert _demo.normalise_time_axis(node) == (0.0, 1.0, 3)

    for sub_lod in node.additive_sublods:
        assert np.array_equal(sub_lod.centers[:, 3], [0.0, 0.5, 1.0])


def test_normalise_time_axis_leaves_minutes_unchanged() -> None:
    node = _node([0.0, 0.5, 1.0], rungs=2)
    before = [sub_lod.centers.copy() for sub_lod in node.additive_sublods]

    assert _demo.normalise_time_axis(node) == (0.0, 1.0, 3)

    for sub_lod, expected in zip(node.additive_sublods, before, strict=True):
        assert np.array_equal(sub_lod.centers, expected)


def test_normalise_time_axis_rejects_irregular_coordinates() -> None:
    node = _node([0.0, 0.75, 1.0])

    with pytest.raises(RuntimeError, match="neither consecutive frame indices"):
        _demo.normalise_time_axis(node)


def test_scene_bakes_the_tuned_display_window(tmp_path: Path) -> None:
    source = tmp_path / "source.gsplats.zarr"
    write_gsplats_tree(source, _node([0.0, 1.0, 2.0]), ordering="none")

    output = _demo.create_luxar_scene(source, tmp_path / "scene.luxar.zarr")

    attrs = dict(zarr.open_group(str(output), mode="r")["drosophila_nuclei"].attrs)
    lo, hi = _demo.DISPLAY_WINDOW
    assert attrs["intensity"] == pytest.approx(1.0 / (hi - lo))
    assert attrs["offset"] == pytest.approx(-lo / (hi - lo))
    assert attrs["opacity"] == pytest.approx(_demo.GSPLAT_OPACITY)
    assert attrs["absorption"] == pytest.approx(_demo.GSPLAT_ABSORPTION)
