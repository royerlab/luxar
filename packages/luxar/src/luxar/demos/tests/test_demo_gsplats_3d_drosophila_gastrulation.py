"""Regression tests for the Drosophila gastrulation demo's gsplat authoring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr
from zarr.storage import ZipStore

from luxar.encoding import ArrayDecoder, EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatLeaf, GSplatPartition

_DEMO_PATH = (
    Path(__file__).resolve().parents[1] / "demo_gsplats_3d_drosophila_gastrulation.py"
)


def _load_demo_module(name: str = "_luxar_demo_drosophila_gastrulation_for_tests"):
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()


def _write_source(path: Path, shape: tuple[int, ...]) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.create_array("data", shape=shape, chunks=(1,) * len(shape), dtype="u1")


def test_validate_source_accepts_directory_and_zip_without_writing(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "recording.zarr"
    _write_source(directory, (500, 108, 1352, 532))
    directory_files = sorted(
        path.relative_to(directory) for path in directory.rglob("*")
    )

    archive = tmp_path / "recording.zarr.zip"
    with ZipStore(str(archive), mode="w") as store:
        root = zarr.group(store=store)
        root.create_array(
            "data", shape=(500, 108, 1352, 532), chunks=(1, 1, 1, 1), dtype="u1"
        )
    archive_bytes = archive.read_bytes()

    _demo._validate_source(directory)
    _demo._validate_source(archive)

    assert (
        sorted(path.relative_to(directory) for path in directory.rglob("*"))
        == directory_files
    )
    assert archive.read_bytes() == archive_bytes


def test_validate_source_rejects_sibling_and_bare_store(tmp_path: Path) -> None:
    sibling = tmp_path / "sibling.zarr"
    _write_source(sibling, (1507, 108, 1352, 532))
    with pytest.raises(ValueError, match="1507 timepoints, expected 500"):
        _demo._validate_source(sibling)

    bare = tmp_path / "bare.zarr"
    zarr.open_array(str(bare), mode="w", shape=(500, 108, 1352, 532), dtype="u1")
    with pytest.raises(ValueError, match="bare array, not a group"):
        _demo._validate_source(bare)


def test_validate_source_rejects_non_zarr_directory_without_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "not-a-recording"
    source.mkdir()
    readme = source / "README.txt"
    readme.write_text("not zarr", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected the DrosophilaHistone recording"):
        _demo._validate_source(source)

    assert list(source.iterdir()) == [readme]


def test_recompute_requires_source_and_rebuilds_every_stage(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "recording.zarr"
    _write_source(source, (500, 108, 1352, 532))
    work_dir = tmp_path / "recompute"
    commands: list[tuple[str, ...]] = []

    def fake_luxar(*args: str) -> None:
        commands.append(args)
        Path(args[3]).mkdir(parents=True)

    monkeypatch.setattr(_demo, "SOURCE_ARG", None)
    with pytest.raises(SystemExit, match="--recompute needs --source PATH"):
        _demo.recompute_archive(work_dir)

    monkeypatch.setattr(_demo, "SOURCE_ARG", source)
    monkeypatch.setattr(_demo, "run_luxar_cli", fake_luxar)
    _demo.recompute_archive(work_dir)
    commands.clear()

    _demo.recompute_archive(work_dir)

    assert [command[1] for command in commands] == ["fit", "transform", "lod"]


def _sublod(amplitudes: np.ndarray, *, marker: str) -> AdditiveSubLOD:
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    n_splats = amplitudes.size
    centers = np.zeros((n_splats, 3), dtype=np.float32)
    coordinate = np.arange(n_splats, dtype=np.float32)
    centers[:] = coordinate[:, None]
    cholesky = np.zeros((n_splats, 6), dtype=np.float32)
    cholesky[:, [0, 2, 5]] = 1.0
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=cholesky,
        stats={"marker": marker},
    )


def test_normalize_amplitudes_uses_one_robust_in_place_scale() -> None:
    pooled = np.concatenate(
        [np.linspace(10.0, 110.0, 1000, dtype=np.float32), [10_000.0]]
    ).astype(np.float32)
    pooled_reference = pooled.astype(np.float64, copy=True)
    coarse = _sublod(pooled[:500], marker="coarse")
    fine = _sublod(pooled[500:], marker="fine")
    leaf = GSplatLeaf([coarse, fine], meta={"lod_stats": {"method": "stream"}})
    originals = [coarse.amplitudes.copy(), fine.amplitudes.copy()]
    array_ids = [id(coarse.amplitudes), id(fine.amplitudes)]

    lo, hi = _demo.normalize_amplitudes(leaf)

    expected_hi = float(np.percentile(pooled_reference, 99.9))
    assert lo == 10.0
    assert hi == expected_hi
    for sublod, original, array_id in zip(
        leaf.additive_sublods, originals, array_ids, strict=True
    ):
        np.testing.assert_allclose(
            sublod.amplitudes,
            (original - lo) / (expected_hi - lo),
            rtol=2e-6,
            atol=2e-6,
        )
        assert np.all(sublod.amplitudes >= 0.0)
        assert id(sublod.amplitudes) == array_id
    assert fine.amplitudes[-1] > 1.0, "the hot outlier must not set scene exposure"
    assert coarse.stats == {"marker": "coarse"}
    assert fine.stats == {"marker": "fine"}
    assert leaf.meta == {"lod_stats": {"method": "stream"}}


def test_normalize_amplitudes_maps_constant_signal_into_range() -> None:
    sublod = _sublod(np.full(4, 7.0, dtype=np.float32), marker="constant")
    sublod.amplitudes.flags.writeable = False
    leaf = GSplatLeaf([sublod])

    assert _demo.normalize_amplitudes(leaf) == (7.0, 7.0)

    np.testing.assert_array_equal(sublod.amplitudes, np.ones(4, dtype=np.float32))


def test_robust_scale_compensation_preserves_approved_appearance() -> None:
    raw = np.array([5.0, 70.0, 512.0, 798.0], dtype=np.float64)
    old_normalized = (raw - 5.0) / (798.0 - 5.0)
    robust_normalized = (raw - 5.0) / (512.0 - 5.0)

    np.testing.assert_allclose(
        robust_normalized / _demo.DISPLAY_WINDOW_TOP,
        old_normalized / 0.737,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        robust_normalized * _demo.GSPLAT_OPACITY,
        old_normalized * 0.41,
        rtol=1e-12,
        atol=1e-12,
    )


def test_create_luxar_scene_preserves_partition_structure(tmp_path: Path) -> None:
    raw_parts = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    partition = GSplatPartition(
        children=[
            GSplatLeaf([_sublod(raw_parts[0], marker="left")]),
            GSplatLeaf([_sublod(raw_parts[1], marker="right")]),
        ]
    )
    source = tmp_path / "input.gsplats.zarr"
    output = tmp_path / "scene.luxar.zarr"
    write_gsplats_tree(
        source,
        partition,
        ordering="none",
        encoding_mode=EncodingMode.PRECISION,
    )

    _demo.create_luxar_scene(source, output)

    root = zarr.open_group(str(output), mode="r")
    stored = root["drosophila_nuclei"]
    assert stored.attrs["kind"] == "partition"
    assert stored.attrs["blending_mode"] == "volumetric"
    assert stored.attrs["opacity"] == float(f"{_demo.GSPLAT_OPACITY:.12g}")
    assert stored.attrs["intensity"] == pytest.approx(1.0 / _demo.DISPLAY_WINDOW_TOP)
    assert stored.attrs["layer"] is True
    assert set(stored.group_keys()) == {"part_0", "part_1"}

    pooled = np.concatenate(raw_parts).astype(np.float64)
    lo = float(pooled.min())
    hi = float(np.percentile(pooled, _demo.AMPLITUDE_REFERENCE_PERCENTILE))
    decoder = ArrayDecoder()
    for index, raw in enumerate(raw_parts):
        decoded = decoder.decode(stored[f"part_{index}"]["amplitudes"], root)
        expected = (raw - lo) / (hi - lo)
        np.testing.assert_allclose(
            np.sort(decoded),
            np.sort(expected),
            rtol=2e-3,
            atol=2e-3,
        )


def test_create_luxar_scene_keeps_matrix_amplitudes_non_negative(
    tmp_path: Path,
) -> None:
    raw = np.array([5.0] + [480.0] * 1000, dtype=np.float32)
    source = tmp_path / "input.gsplats.zarr"
    output = tmp_path / "scene.luxar.zarr"
    write_gsplats_tree(
        source,
        GSplatLeaf([_sublod(raw, marker="matrix")]),
        ordering="none",
        encoding_mode=EncodingMode.PRECISION,
    )

    _demo.create_luxar_scene(source, output)

    root = zarr.open_group(str(output), mode="r")
    stored = root["drosophila_nuclei"]
    decoded = ArrayDecoder().decode(stored["amplitudes"], root)
    assert np.all(decoded >= 0.0)
    assert decoded.min() == 0.0
