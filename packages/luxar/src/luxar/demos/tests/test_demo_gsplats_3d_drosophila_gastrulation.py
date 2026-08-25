"""Regression tests for the Drosophila gastrulation demo's gsplat authoring."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.gsplats.gsplat_data import AdditiveSubLOD
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


def _sublod(amplitudes: np.ndarray, *, marker: str) -> AdditiveSubLOD:
    amplitudes = np.asarray(amplitudes, dtype=np.float32)
    n_splats = amplitudes.size
    centers = np.zeros((n_splats, 3), dtype=np.float32)
    centers[:, 1] = np.arange(n_splats, dtype=np.float32)
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


def test_add_gsplat_node_preserves_partition_structure(tmp_path: Path) -> None:
    partition = GSplatPartition(
        children=[
            GSplatLeaf([_sublod(np.array([1.0, 2.0]), marker="left")]),
            GSplatLeaf([_sublod(np.array([3.0, 4.0]), marker="right")]),
        ]
    )
    output = tmp_path / "scene.luxar.zarr"
    dimensions = Dimensions(
        [
            Dimension("Z", unit="µm", display=True),
            Dimension("Y", unit="µm", display=True),
            Dimension("X", unit="µm", display=True),
        ]
    )

    with LuxarZarrCompiler(output) as compiler:
        scene = compiler.create_scene(dimensions=dimensions)
        _demo.add_gsplat_node(
            scene,
            name="drosophila_nuclei",
            node=partition,
            blending_mode="volumetric",
            opacity=0.41,
            layer=True,
        )

    stored = zarr.open_group(str(output), mode="r")["drosophila_nuclei"]
    assert stored.attrs["kind"] == "partition"
    assert stored.attrs["blending_mode"] == "volumetric"
    assert stored.attrs["opacity"] == 0.41
    assert stored.attrs["layer"] is True
    assert set(stored.group_keys()) == {"part_0", "part_1"}
