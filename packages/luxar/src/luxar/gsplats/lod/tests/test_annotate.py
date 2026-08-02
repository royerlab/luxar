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


def _make_rgba_gsplat(n: int = 400, ndim: int = 3, seed: int = 0) -> GSplatData:
    """Classical-import-style RGBA splats: amplitude ≡ 1, per-splat weight
    carried entirely in the (non-uniform) color alpha — the case where raw and
    alpha-effective amplitudes diverge."""
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    colors = np.ones((n, 4), dtype=np.float32)
    colors[:, 3] = rng.uniform(0.1, 1.0, size=n).astype(np.float32)
    return GSplatData(
        centers=rng.uniform(0, 100, size=(n, ndim)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
        colors=colors,
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


def test_annotate_alpha_effective_matches_build_rgba(tmp_path: Path) -> None:
    """RGBA leaf (amplitude ≡ 1, non-uniform α): the build path stamps e(k)/w
    from the ALPHA-EFFECTIVE amplitude A·α, so the annotate retrofit must too.
    Before the fix annotate used raw amplitudes and disagreed (issue #738)."""
    data = _make_rgba_gsplat(n=400, seed=1)
    params = RecipeParams(n_lods=4, device="cpu", seed=0)
    out = tmp_path / "rgba.gsplats.zarr"
    build_recipe(data, "stream", params).save(out)

    built = _collect_quality_attrs(out)
    assert built, "the fresh build must carry quality stamps"
    # α is non-uniform, so an energy-ordered ladder is genuinely graded.
    e_cum = [
        e["lod_stats.energy_fraction_cum"]
        for e in built.values()
        if "lod_stats.energy_fraction_cum" in e
    ]
    assert any(0.0 < e < 1.0 for e in e_cum)

    _strip_quality_attrs(out)
    annotate_quality_store(out, device="cpu")
    annotated = _collect_quality_attrs(out)

    assert set(annotated) == set(built)
    for path, built_entry in built.items():
        for key, built_val in built_entry.items():
            if key.endswith("quality"):
                continue  # e-only mode; no Q pass
            ann_val = annotated[path].get(key)
            assert ann_val is not None, (path, key)
            assert ann_val == pytest.approx(built_val, rel=1e-2), (path, key)


def test_annotate_non_rgba_matches_build(tmp_path: Path) -> None:
    """Regression: for non-RGBA data effective_amplitudes is a no-op, so the
    annotate stamps equal the raw-amplitude build stamps (unchanged behavior)."""
    data = _make_random_gsplat(n=400, seed=1)
    params = RecipeParams(n_lods=4, device="cpu", seed=0)
    out = tmp_path / "raw.gsplats.zarr"
    build_recipe(data, "stream", params).save(out)

    built = _collect_quality_attrs(out)
    assert built
    _strip_quality_attrs(out)
    annotate_quality_store(out, device="cpu")
    annotated = _collect_quality_attrs(out)

    assert set(annotated) == set(built)
    for path, built_entry in built.items():
        for key, built_val in built_entry.items():
            if key.endswith("quality"):
                continue
            ann_val = annotated[path].get(key)
            assert ann_val is not None, (path, key)
            assert ann_val == pytest.approx(built_val, rel=1e-2), (path, key)


def test_annotate_transparent_leaf_writes_no_energy_stamp(tmp_path: Path) -> None:
    """FIX #738: a NONEMPTY but fully transparent RGBA leaf (α≡0) has zero
    effective energy. The build path only writes energy_fraction_cum when
    energy_total > 0 (it stamps 1.0 only for the genuinely EMPTY leaf), so the
    annotate retrofit must also skip it — not stamp a spurious 1.0."""
    data = _make_rgba_gsplat(n=200, seed=3)
    data = GSplatData(
        centers=data.centers,
        amplitudes=data.amplitudes,
        cholesky_factors=data.cholesky_factors,
        colors=np.concatenate(
            [np.asarray(data.colors)[:, :3], np.zeros((200, 1), dtype=np.float32)],
            axis=1,
        ),
    )
    params = RecipeParams(n_lods=4, device="cpu", seed=0)
    out = tmp_path / "transparent.gsplats.zarr"
    build_recipe(data, "stream", params).save(out)

    # The fresh build wrote NO energy_fraction_cum (zero effective energy).
    built = _collect_quality_attrs(out)
    assert not any("lod_stats.energy_fraction_cum" in entry for entry in built.values())

    _strip_quality_attrs(out)
    report = annotate_quality_store(out, device="cpu")
    annotated = _collect_quality_attrs(out)

    # Annotate matches: no energy stamp on any (nonempty, zero-energy) leaf.
    assert not any(
        "lod_stats.energy_fraction_cum" in entry for entry in annotated.values()
    )
    # The leaf is nonempty; its e(k) report is empty (nothing stamped).
    assert report.leaves and all(
        leaf.n_splats > 0 and leaf.energy_fraction_cum == [] for leaf in report.leaves
    )


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


def test_annotate_overview_rgba_reproduces_build_stamps(tmp_path: Path) -> None:
    """FIX #738: an overview (lod-over-partition) store built from RGBA data.

    The Q pass loads the FINEST child (a partition) as the reference; before the
    fix ``_node_content``'s partition branch dropped colors, so the reference
    was scored on RAW amplitudes (α≡1) while the coarse cap folded α in — an
    internally inconsistent stamp (~2.8× off on classical RGBA imports). This
    asserts the annotate stamps on the lod-group children (the cap's measured
    ``quality`` + the group-consistent ``reference_energy`` w) reproduce the
    fresh-build stamps. Fails without the colors-carrying fix (raw w off ~2.8×).
    """
    data = _make_rgba_gsplat(n=400, seed=2)
    params = RecipeParams(
        n_lods=3, max_elements=120, compression_factor=4, device="cpu", seed=0
    )
    out = tmp_path / "overview_rgba.gsplats.zarr"
    tree = build_recipe(data, "overview", params)
    from luxar.gsplats.io.save_gsplats import write_gsplats_tree

    write_gsplats_tree(out, tree)

    built = _collect_quality_attrs(out)
    assert built, "the fresh build must carry quality stamps"
    # The lod-group children (child_0 cap, child_1 partition) carry level_stats.
    assert "level_stats.reference_energy" in built["child_0"]

    _strip_quality_attrs(out)
    annotate_quality_store(out, with_quality=True, device="cpu")
    annotated = _collect_quality_attrs(out)

    # Every build-time stamp must be reproduced (annotate additionally stamps
    # the partition node itself — a pre-existing superset, not our concern).
    assert set(built) <= set(annotated)
    for path, built_entry in built.items():
        for key, built_val in built_entry.items():
            ann_val = annotated[path].get(key)
            assert ann_val is not None, (path, key)
            if key.endswith("quality"):
                # Q is a sampled estimate on both sides — generous tolerance.
                assert ann_val == pytest.approx(built_val, abs=0.05), (path, key)
            else:
                # e(k)/w are exact O(N) sums; the alpha-effective convention
                # must match the build path within decode quantization.
                assert ann_val == pytest.approx(built_val, rel=5e-2), (path, key)


def _identity_leaf(amps: np.ndarray, colors: np.ndarray | None) -> GSplatData:
    """A leaf whose Cholesky is identity (``|Σ|^{1/2} == 1``) so the total
    self-energy reduces to ``Σ (A·α)² · π^{D/2}`` — the cleanest probe for the
    alpha-effective convention."""
    n = amps.shape[0]
    ndim = 3
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = 1.0
    return GSplatData(
        centers=np.zeros((n, ndim), dtype=np.float32),
        amplitudes=amps.astype(np.float32),
        cholesky_factors=chol,
        colors=colors,
    )


def test_node_content_partition_mixed_dtype_rgba_normalized() -> None:
    """FIX #738 (F-A): a partition whose parts have DIFFERENT color dtypes but
    the same width (float32 RGBA + uint8 RGBA). ``np.concatenate`` would promote
    uint8 alpha 255→float 255.0 (unnormalized), inflating the reference energy
    ~255x; the canonical ``_merge_lod_colors`` normalizes integer alpha to
    [0, 1] first, so the merged reference folds α exactly like a fresh build."""
    import math

    from luxar.gsplats.lod.annotate import _node_content
    from luxar.gsplats.lod.quality import total_self_energy
    from luxar.gsplats.tree import GSplatPartition

    amps = np.ones(2, dtype=np.float32)
    part_a = _identity_leaf(
        amps, np.array([[1, 1, 1, 0.5], [1, 1, 1, 1.0]], dtype=np.float32)
    )
    part_b = _identity_leaf(
        amps, np.array([[255, 255, 255, 128], [255, 255, 255, 255]], dtype=np.uint8)
    )
    merged = _node_content(GSplatPartition(children=[part_a.tree, part_b.tree]))

    # α folded, integer alpha normalized: Σ (A·α)² · π^{3/2}.
    expected = (0.5**2 + 1.0**2 + (128 / 255) ** 2 + 1.0**2) * math.pi**1.5
    energy = total_self_energy(merged)
    assert energy == pytest.approx(expected, rel=1e-5)
    # The ×255 bug would land here — orders of magnitude larger.
    wrong = (0.5**2 + 1.0**2 + 128.0**2 + 255.0**2) * math.pi**1.5
    assert energy < wrong / 100.0


def test_node_content_partition_mixed_rgb_rgba_width() -> None:
    """FIX #738 (F-B): a partition mixing an RGB part (width 3, no alpha) with an
    RGBA part (width 4). The old code set colors=None (dropping the RGBA alpha);
    ``_merge_lod_colors`` widens RGB→opaque (α=1) and keeps the RGBA alpha, so
    the merged reference preserves per-splat opacity — matching a fresh build."""
    import math

    from luxar.gsplats.lod.annotate import _node_content
    from luxar.gsplats.lod.quality import total_self_energy
    from luxar.gsplats.tree import GSplatPartition

    amps = np.ones(2, dtype=np.float32)
    part_rgb = _identity_leaf(amps, np.ones((2, 3), dtype=np.float32))
    part_rgba = _identity_leaf(
        amps, np.array([[1, 1, 1, 0.5], [1, 1, 1, 1.0]], dtype=np.float32)
    )
    merged = _node_content(GSplatPartition(children=[part_rgb.tree, part_rgba.tree]))

    assert merged.colors is not None and merged.colors.shape[1] == 4
    # RGB widened to opaque α=1; RGBA alpha [0.5, 1.0] preserved.
    expected = (1.0**2 + 1.0**2 + 0.5**2 + 1.0**2) * math.pi**1.5
    assert total_self_energy(merged) == pytest.approx(expected, rel=1e-5)
    # The old drop-colors path would have scored every splat at α≡1 (=4·π^{3/2}).
    assert total_self_energy(merged) < (4.0 * math.pi**1.5)


def test_annotate_leaf_skips_nonfinite_energy_fraction() -> None:
    """FIX #738 (F-C): a legacy store with a non-finite amplitude makes
    total_raw=inf and c/total_raw=nan. The build only writes
    ``energy_fraction_cum`` when the fraction is finite (additive.py's
    ``if np.isfinite(e_frac)``); annotate must match — a ``min(1, max(0, nan))``
    would otherwise fabricate a 0.0 stamp the build never writes.

    A real store can't carry an inf amplitude (on-disk amplitudes are quantized
    to uint8), so this drives ``_annotate_leaf`` directly with a decoder that
    yields an inf amplitude, under ``dry_run`` (no writes) — asserting the leaf
    reports NO energy fraction rather than a fabricated 0.0."""
    from luxar.gsplats.lod.annotate import AnnotateReport, _annotate_leaf

    class _FakeGroup:
        def __init__(self, arrays: Dict[str, Any]) -> None:
            self._arrays = arrays
            self.attrs: Dict[str, Any] = {}
            self.path = "leaf"

        def __contains__(self, key: str) -> bool:
            return key in self._arrays

        def __getitem__(self, key: str) -> Any:
            return self._arrays[key]

    class _FakeDecoder:
        def decode(self, arr: Any, _root: Any) -> np.ndarray:
            return np.asarray(arr)

    group = _FakeGroup(
        {
            "amplitudes": np.array([np.inf, 1.0], dtype=np.float64),
            # identity diagonal (|Σ|^{1/2} == 1); presence routes _decode_diag
            # to the direct read.
            "cholesky_factors_diag": np.ones((2, 3), dtype=np.float64),
        }
    )
    report = AnnotateReport(path="mem", dry_run=True)
    _annotate_leaf(group, group, _FakeDecoder(), report, dry_run=True)

    # total_raw=inf ⇒ fraction nan ⇒ skipped, matching the build (no 0.0 stamp).
    assert report.leaves[0].energy_fraction_cum == []


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
