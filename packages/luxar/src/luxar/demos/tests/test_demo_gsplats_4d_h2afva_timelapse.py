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


def test_scene_grafts_time_parts_with_the_reviewed_layer_settings(
    tmp_path, monkeypatch
) -> None:
    source = _write_source(tmp_path / "source.gsplats.zarr")
    monkeypatch.setattr(
        _demo, "time_parts_cache_dir", lambda _p: tmp_path / "timeparts-cache"
    )
    output = _demo.create_luxar_scene(source, tmp_path / "scene.luxar.zarr")

    root = zarr.open_group(str(output), mode="r")
    node = root["zebrafish_nuclei_4d"]
    attrs = dict(node.attrs)
    # One part per timepoint: the fake archive has two timepoints.
    assert attrs["kind"] == "partition"
    parts = sorted(k for k in node.group_keys() if k.startswith("part_"))
    assert len(parts) == 2
    for part in parts:
        # A two-splat part fits its first rung, so the graft writes it flat;
        # a real part carries `additive_<k>/centers` rungs instead.
        grp = root[f"zebrafish_nuclei_4d/{part}"]
        key = "centers" if "centers" in grp.array_keys() else "additive_0/centers"
        centers = np.asarray(grp[key][:])
        assert np.unique(centers[:, _demo.TIME_COL]).size == 1
    # Loic's Layers-panel values (2026-09-10): volumetric, absorption 1.09,
    # opacity 0.55, gamma 2.2, plasma, window 0.001 - 0.025.
    assert attrs["blending_mode"] == "volumetric"
    assert attrs["absorption"] == pytest.approx(1.09)
    assert attrs["opacity"] == pytest.approx(0.55)
    assert attrs["gamma"] == pytest.approx(2.2)
    assert attrs["intensity"] == pytest.approx(1.0 / 0.024)
    assert attrs["offset"] == pytest.approx(-0.001 / 0.024)
    lo = -attrs["offset"] / attrs["intensity"]
    hi = (1.0 - attrs["offset"]) / attrs["intensity"]
    assert (lo, hi) == pytest.approx((0.001, 0.025))
    assert attrs["layer"] is True


def test_time_part_ladders_apply_the_share_floor_across_multiple_parts() -> None:
    assert _demo.FIRST_RUNG_SPLATS == 20_833
    ladders = _demo.time_part_ladders([2_400_000, 20_000, 3])
    big, small, tiny = ladders
    assert big[0] == 300_000
    assert big[-1] == 2_400_000
    increments = [b - a for a, b in zip([0, *big[:-1]], big, strict=True)]
    assert len(big) == _demo.TIME_PART_LADDER_MAX_DEPTH
    assert max(increments) - min(increments) <= 1
    assert max(increments) <= _demo.DEFAULT_MAX_ADDITIVE_COMMIT
    assert small == [2_500, 5_000, 7_500, 10_000, 12_500, 15_000, 17_500, 20_000]
    assert tiny == [1, 2, 3]
    assert _demo.time_part_ladders([2_217_045, 1])[0][0] == 277_131


def test_a_single_time_part_keeps_the_download_budget_ladder() -> None:
    assert _demo.time_part_ladders([2_400_000])[0][0] == _demo.FIRST_RUNG_SPLATS


def test_time_part_share_floor_cannot_exceed_the_per_part_commit_cap() -> None:
    with pytest.raises(ValueError, match="per-part commit cap"):
        _demo.time_part_ladders([7_200_001, 1])


def test_time_parts_preserve_every_splat_and_split_on_time(tmp_path) -> None:
    source = _write_source(tmp_path / "source.gsplats.zarr")
    out = _demo.build_time_parts(source, tmp_path / "tp.gsplats.zarr")

    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import total_splats

    node, _ = load_gsplat_node(str(out))
    assert type(node).__name__ == "GSplatPartition"
    assert len(node.children) == 2
    assert [child.n_additive_sublods for child in node.children] == [2, 2]
    assert total_splats(node) == 4
    root = zarr.open_group(str(out), mode="r")
    assert dict(root.attrs)["source_stride"] == 5
    assert dict(root.attrs)["blending_mode"] == "volumetric"


def test_time_parts_cache_key_tracks_the_recipe(tmp_path) -> None:
    source = _write_source(tmp_path / "source.gsplats.zarr")
    a = _demo.time_parts_cache_dir(source)
    assert a.parent == source.parent
    assert f"_v{_demo.TIME_PARTS_VERSION}_" in a.name


def test_resolve_time_parts_prunes_old_versioned_builds(tmp_path, monkeypatch) -> None:
    source = _write_source(tmp_path / "source.gsplats.zarr")
    current = _demo.time_parts_cache_dir(source)
    stale = tmp_path / "51tp_timeparts_v0_deadbeef00"
    stale.mkdir()
    (stale / "orphan").write_text("old")
    monkeypatch.setattr(_demo, "build_time_parts", lambda _src, out: out)

    _demo.resolve_time_parts(source)

    assert not stale.exists()
    assert current.exists()


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
