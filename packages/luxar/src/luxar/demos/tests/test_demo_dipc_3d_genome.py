"""Smoke tests for the pure parsing/geometry helpers in demo_dipc_3d_genome.

Deterministic helpers only — no network, no tar extraction, no scene I/O.
The demo is loaded by file path (see test_demo_ppi_flow_field for the rationale).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_DEMO_PATH = Path(__file__).resolve().parents[1] / "demo_dipc_3d_genome.py"


def _load_demo_module():
    name = "_luxar_demo_dipc_for_tests"
    spec = importlib.util.spec_from_file_location(name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        pytest.skip(f"Could not locate demo at {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_demo = _load_demo_module()
_split_chrom_haplotype = _demo._split_chrom_haplotype
parse_3dg = _demo.parse_3dg
chromosome_color = _demo.chromosome_color
build_polylines = _demo.build_polylines
save_polylines_npz = _demo.save_polylines_npz
load_polylines_npz = _demo.load_polylines_npz
build_scene = _demo.build_scene
MIN_BEADS_PER_ARM = _demo.MIN_BEADS_PER_ARM


class TestSplitChromHaplotype:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("1(mat)", ("1", 0)),
            ("1(pat)", ("1", 1)),
            ("chrX(pat)", ("X", 1)),
            ("chr1a", ("1", 0)),
            ("2b", ("2", 1)),
            ("chr22(mat)", ("22", 0)),
        ],
    )
    def test_parses_conventions(self, raw, expected) -> None:
        assert _split_chrom_haplotype(raw) == expected


class TestParse3dg:
    def test_groups_and_sorts_by_position(self) -> None:
        lines = [
            "# comment",
            "1(mat)\t2000000\t0.0\t0.0\t2.0",
            "1(mat)\t1000000\t0.0\t0.0\t1.0",  # out of order → must be sorted first
            "1(pat)\t1000000\t5.0\t5.0\t5.0",
            "garbage line too short",
        ]
        parsed = parse_3dg(lines)
        assert set(parsed.keys()) == {("1", 0), ("1", 1)}
        mat = parsed[("1", 0)]
        assert mat.shape == (2, 4)
        # sorted by genomic position ascending
        assert mat[0, 0] < mat[1, 0]
        np.testing.assert_allclose(mat[0, 1:], [0.0, 0.0, 1.0])


class TestChromosomeColor:
    def test_shape_dtype_range(self) -> None:
        c = chromosome_color("1")
        assert c.shape == (3,)
        assert c.dtype == np.float32
        assert (c >= 0).all() and (c <= 1).all()

    def test_distinct_and_deterministic(self) -> None:
        assert not np.allclose(chromosome_color("1"), chromosome_color("2"))
        np.testing.assert_array_equal(chromosome_color("7"), chromosome_color("7"))


class TestBuildPolylines:
    def _parsed(self, n=10):
        pos = np.arange(n) * 1_000_000
        xyz = np.random.default_rng(0).normal(size=(n, 3)) * 3 + 100.0
        arr = np.column_stack([pos, xyz]).astype(np.float64)
        short = np.column_stack([np.arange(2) * 1e6, np.zeros((2, 3))]).astype(
            np.float64
        )
        return {("1", 0): arr, ("1", 1): arr.copy(), ("2", 0): short}

    def test_drops_short_arms_and_normalizes(self) -> None:
        polys = build_polylines(self._parsed())
        # ("2", 0) has 2 beads < MIN_BEADS_PER_ARM → dropped
        assert all(not (p["chrom"] == "2") for p in polys)
        assert len(polys) == 2
        for p in polys:
            assert p["vertices"].dtype == np.float32
            assert p["vertices"].shape[1] == 3
        # normalized: combined centroid near origin
        allv = np.concatenate([p["vertices"] for p in polys])
        np.testing.assert_allclose(allv.mean(axis=0), [0, 0, 0], atol=1e-4)

    def test_npz_roundtrip(self, tmp_path) -> None:
        polys = build_polylines(self._parsed())
        path = tmp_path / "roundtrip.npz"
        save_polylines_npz(polys, path)
        loaded = load_polylines_npz(path)
        assert len(loaded) == len(polys)
        for a, b in zip(polys, loaded):
            assert a["chrom"] == b["chrom"]
            assert a["haplotype"] == b["haplotype"]
            np.testing.assert_allclose(a["vertices"], b["vertices"], atol=1e-6)

    def test_build_scene_writes_store(self, tmp_path) -> None:
        # End-to-end: locks the polyline + haplotype-toggle dim_order/fill contract.
        polys = build_polylines(self._parsed())
        out = tmp_path / "genome.luxar.zarr"
        n = build_scene(out, polys)
        assert n > 0
        assert out.exists() and any(out.iterdir())
