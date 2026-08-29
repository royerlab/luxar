"""Regression tests for the h2afva timelapse demo."""

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
from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_gsplats_4d_h2afva_timelapse.py"
_MANIFEST_PATH = _DEMO_PATH.parent / "data_manifest.json"
#: The RESTRUCTURED generation: substitutive levels and the 44-part partition
#: dropped, the progressive ladder rebuilt, then re-chunked for archive reads.
#: Was c5e14be9… / 1,873,559,527 (44 parts x 4 levels) before that pass, at
#: identical finest-level content.
_ARCHIVE_SHA256 = "037806639a787ac1270b07bfaa6918a5144e45f5d3198cf2165381316c29afae"
_ARCHIVE_BYTES = 1_115_714_088


def _load_demo_module(name: str = "_luxar_demo_h2afva_timelapse_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def _leaf(timepoint: float, offset: float) -> GSplatLeaf:
    cholesky = np.zeros((2, 10), dtype=np.float32)
    cholesky[:, [0, 2, 5, 9]] = 1.0
    centers = np.array(
        [[offset, 0.0, 0.0, timepoint], [offset + 1.0, 1.0, 1.0, timepoint]],
        dtype=np.float32,
    )
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=np.ones(2, dtype=np.float32),
                cholesky_factors=cholesky,
            )
        ]
    )


def _write_source(
    path: Path,
    *,
    source_stride: int = 5,
    source_timepoints: list[int] | None = None,
) -> Path:
    compress = "zip" if path.name.endswith(".zarr.zip") else None
    write_gsplats_tree(
        path,
        GSplatPartition(children=[_leaf(0.0, 0.0), _leaf(1.0, 2.0)]),
        ordering="none",
        compress=compress,
        root_attrs={
            "source_stride": source_stride,
            "source_timepoints": (
                [0, 5] if source_timepoints is None else source_timepoints
            ),
            "blending_mode": "volumetric",
            "absorption": 1.34,
            "layer": True,
        },
    )
    return path


def test_manifest_and_download_size_pin_the_51tp_archive() -> None:
    manifest = json.loads(_MANIFEST_PATH.read_text())
    variant = manifest["datasets"][_demo.DATASET]["variants"]["51tp"]
    pin = variant["files"][0]

    assert (pin["sha256"], pin["bytes"]) == (_ARCHIVE_SHA256, _ARCHIVE_BYTES)
    assert _demo.DEMO_META["requirements"]["download_mb"] == round(
        _ARCHIVE_BYTES / 1024**2
    )


def test_resolve_data_requests_the_51tp_variant(monkeypatch, tmp_path) -> None:
    source = tmp_path / "h2afva_51tp.gsplats.zarr.zip"
    calls: list[tuple[str, str | None]] = []

    def ensure_dataset(name: str, *, variant: str | None = None):
        calls.append((name, variant))
        return [source]

    monkeypatch.setattr(_demo, "ensure_dataset", ensure_dataset)

    assert _demo.resolve_data() == source
    assert calls == [(_demo.DATASET, "51tp")]


def test_scene_uses_order_independent_blending_and_display_window(tmp_path) -> None:
    source = _write_source(tmp_path / "source.gsplats.zarr")
    output = _demo.create_luxar_scene(source, tmp_path / "scene.luxar.zarr")

    root = zarr.open_group(str(output), mode="r")
    attrs = dict(root["zebrafish_nuclei_4d"].attrs)
    assert attrs["blending_mode"] == "additive"
    assert "absorption" not in attrs
    assert attrs["intensity"] == pytest.approx(17.57)
    assert attrs["offset"] == pytest.approx(-0.00879)
    assert -attrs["offset"] / attrs["intensity"] == pytest.approx(0.00050, abs=1e-6)
    assert 1.0 / attrs["intensity"] == pytest.approx(0.057, abs=1e-4)


def test_scene_rejects_an_archive_with_a_different_source_stride(tmp_path) -> None:
    source = _write_source(tmp_path / "wrong-stride.gsplats.zarr.zip", source_stride=10)

    with pytest.raises(ValueError, match="source_stride"):
        _demo.create_luxar_scene(source, tmp_path / "scene.luxar.zarr")


def test_scene_rejects_an_archive_with_a_mismatched_frame_list(tmp_path) -> None:
    source = _write_source(
        tmp_path / "wrong-frames.gsplats.zarr", source_timepoints=[0]
    )

    with pytest.raises(ValueError, match="source_timepoints"):
        _demo.create_luxar_scene(source, tmp_path / "scene.luxar.zarr")
