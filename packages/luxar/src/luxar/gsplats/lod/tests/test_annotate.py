"""Tests for in-place quality annotation (``gsplats.lod.annotate``).

The core equivalence claim: on a store whose quality stamps were stripped
(simulating a legacy dataset), :func:`annotate_quality_store` reproduces the
build-time ``energy_fraction_cum`` / ``reference_energy`` / ``quality`` values
from the on-disk arrays alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import zarr

from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.lod.annotate import annotate_quality_store
from luxar.gsplats.lod.recipes import RecipeParams, build_recipe

_QUALITY_KEYS = ("energy_fraction_cum", "quality", "reference_energy")


def _make_random_gsplat(n: int = 400, ndim: int = 3, seed: int = 0) -> GSplatData:
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    return GSplatData(
        centers=rng.uniform(0, 100, size=(n, ndim)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )


def _collect_quality_attrs(path: Path) -> Dict[str, Dict[str, Any]]:
    """Snapshot every group's quality-related attr values, keyed by zarr path."""
    root = zarr.open_group(str(path), mode="r")
    out: Dict[str, Dict[str, Any]] = {}

    def walk(g: zarr.Group) -> None:
        entry: Dict[str, Any] = {}
        for attr_key in ("lod_stats", "level_stats"):
            d = g.attrs.get(attr_key)
            if isinstance(d, dict):
                for k in _QUALITY_KEYS:
                    if k in d:
                        entry[f"{attr_key}.{k}"] = d[k]
        if entry:
            out[str(g.path or "/")] = entry
        for name in g.group_keys():
            walk(g[name])

    walk(root)
    return out


def _strip_quality_attrs(path: Path) -> None:
    """Remove all quality stamps in place — turns a fresh store 'legacy'."""
    root = zarr.open_group(str(path), mode="r+")

    def walk(g: zarr.Group) -> None:
        for attr_key in ("lod_stats", "level_stats"):
            d = g.attrs.get(attr_key)
            if isinstance(d, dict) and any(k in d for k in _QUALITY_KEYS):
                cleaned = {k: v for k, v in d.items() if k not in _QUALITY_KEYS}
                g.attrs[attr_key] = cleaned
        for name in g.group_keys():
            walk(g[name])

    walk(root)
    zarr.consolidate_metadata(root.store)


@pytest.fixture
def levels_store(tmp_path: Path) -> Path:
    """A quality-stamped 'levels' store, then stripped to simulate legacy."""
    data = _make_random_gsplat(n=400)
    params = RecipeParams(
        n_lods=3, levels=1, compression_factor=8, device="cpu", seed=0
    )
    out = tmp_path / "levels.gsplats.zarr"
    build_recipe(data, "levels", params).save(out)
    return out


def test_annotate_reproduces_build_stamps(levels_store: Path) -> None:
    """Stripped store + annotate --with-quality == the build-time stamps
    (within decode/estimator tolerance)."""
    built = _collect_quality_attrs(levels_store)
    assert built, "the fresh build must carry quality stamps"

    _strip_quality_attrs(levels_store)
    assert not _collect_quality_attrs(levels_store)

    report = annotate_quality_store(levels_store, with_quality=True, device="cpu")
    assert not report.dry_run
    annotated = _collect_quality_attrs(levels_store)

    assert set(annotated) == set(built)
    for path, built_entry in built.items():
        for key, built_val in built_entry.items():
            ann_val = annotated[path].get(key)
            assert ann_val is not None, (path, key)
            if key.endswith("quality"):
                # Q is a sampled estimate on both sides — generous tolerance.
                assert ann_val == pytest.approx(built_val, abs=0.05), (path, key)
            else:
                # e(k)/w are exact O(N) sums over the decoded arrays; only
                # encode/decode quantization separates them from build time.
                assert ann_val == pytest.approx(built_val, rel=1e-2), (path, key)


def test_annotate_e_only_skips_quality(levels_store: Path) -> None:
    """Default (cheap) mode stamps e(k)+w but never quality."""
    _strip_quality_attrs(levels_store)
    report = annotate_quality_store(levels_store, device="cpu")
    assert report.leaves and not report.levels
    annotated = _collect_quality_attrs(levels_store)
    flat_keys = {k for entry in annotated.values() for k in entry}
    assert "lod_stats.energy_fraction_cum" in flat_keys
    assert "level_stats.reference_energy" in flat_keys
    assert "level_stats.quality" not in flat_keys
    # e(k) is monotone with e(last) == 1.0 on every leaf.
    for leaf in report.leaves:
        e = leaf.energy_fraction_cum
        assert e == sorted(e)
        assert e[-1] == pytest.approx(1.0)


def test_annotate_dry_run_writes_nothing(levels_store: Path) -> None:
    _strip_quality_attrs(levels_store)
    before_hash = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    report = annotate_quality_store(
        levels_store, with_quality=True, device="cpu", dry_run=True
    )
    assert report.dry_run and report.leaves and report.levels
    assert not _collect_quality_attrs(levels_store)
    after_hash = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    assert after_hash == before_hash


def test_annotate_refreshes_content_hash(levels_store: Path) -> None:
    """New attrs must invalidate the viewer's persistent cache: the root
    content_hash changes and lands in consolidated metadata too."""
    import json

    _strip_quality_attrs(levels_store)
    before = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    annotate_quality_store(levels_store, device="cpu")
    root = zarr.open_group(str(levels_store), mode="r")
    after = root.attrs["content_hash"]
    assert after != before
    zmeta = json.loads((levels_store / ".zmetadata").read_text())
    assert zmeta["metadata"][".zattrs"]["content_hash"] == after


def test_annotate_rejects_compressed_store(tmp_path: Path) -> None:
    fake = tmp_path / "x.gsplats.zarr.zip"
    fake.write_bytes(b"not a real zip")
    with pytest.raises(ValueError, match="uncompressed"):
        annotate_quality_store(fake)


def test_annotate_rejects_non_gsplats_store(tmp_path: Path) -> None:
    store = tmp_path / "other.zarr"
    root = zarr.open_group(str(store), mode="w")
    root.attrs["format_type"] = "something_else"
    with pytest.raises(ValueError, match="format_type"):
        annotate_quality_store(store)


def test_annotate_overview_partition(tmp_path: Path) -> None:
    """A lod-of-partition (overview) store: the cap gets measured Q + the
    finest content's w; every fine part leaf gets quality 1.0 and its own w;
    disjoint parts sum to the cap's w."""
    data = _make_random_gsplat(n=400)
    params = RecipeParams(
        n_lods=3, max_elements=120, compression_factor=4, device="cpu", seed=0
    )
    out = tmp_path / "overview.gsplats.zarr"
    tree = build_recipe(data, "overview", params)
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    write_gsplats_tree(out, tree)
    _strip_quality_attrs(out)

    annotate_quality_store(out, with_quality=True, device="cpu")
    annotated = _collect_quality_attrs(out)

    cap = annotated["child_0"]
    assert 0.0 <= cap["level_stats.quality"] <= 1.0
    import re

    # Exactly the part LEAF groups (their additive_<i> subgroups carry only
    # the per-sub-LOD e(k) stamps, not level_stats).
    part_entries = {
        p: e for p, e in annotated.items() if re.fullmatch(r"child_1/part_\d+", p)
    }
    assert part_entries
    for entry in part_entries.values():
        assert entry["level_stats.quality"] == 1.0
        assert entry["level_stats.reference_energy"] > 0
    parts_w = sum(e["level_stats.reference_energy"] for e in part_entries.values())
    assert cap["level_stats.reference_energy"] == pytest.approx(parts_w, rel=1e-2)


def test_annotate_is_idempotent(levels_store: Path) -> None:
    """Running annotate twice yields identical stamps (and a stable hash)."""
    _strip_quality_attrs(levels_store)
    annotate_quality_store(levels_store, device="cpu")
    first = _collect_quality_attrs(levels_store)
    hash_first = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    annotate_quality_store(levels_store, device="cpu")
    assert _collect_quality_attrs(levels_store) == first
    hash_second = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    assert hash_second == hash_first
