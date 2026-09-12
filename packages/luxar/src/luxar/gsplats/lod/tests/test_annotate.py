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

from luxar._zarr_compat import consolidate as zc_consolidate
from luxar._zarr_compat import open_group as zc_open_group
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
    """Remove all quality stamps in place — turns a fresh store 'legacy'.

    Edited through the facade, as Luxar's own in-place editors are. Re-opening
    an already-consolidated store with plain ``zarr.open_group`` hands back
    nodes built FROM the root index, and re-consolidating then writes that
    stale tree out as a NESTED index — which later reads prefer over the
    (correct) per-node documents. See ``_zarr_compat.open_group``.
    """
    root = zc_open_group(str(path), mode="r+")

    def walk(g: zarr.Group) -> None:
        for attr_key in ("lod_stats", "level_stats"):
            d = g.attrs.get(attr_key)
            if isinstance(d, dict) and any(k in d for k in _QUALITY_KEYS):
                cleaned = {k: v for k, v in d.items() if k not in _QUALITY_KEYS}
                g.attrs[attr_key] = cleaned
        for name in g.group_keys():
            walk(g[name])

    walk(root)
    zc_consolidate(root)


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


def test_annotate_rerun_erases_stale_energy_stamp(tmp_path: Path) -> None:
    """A leaf the build leaves unstamped (zero effective energy) may still
    carry a stale ``energy_fraction_cum`` from an annotate run under the old
    raw-amplitude convention. A re-run must ERASE it (keeping the other
    lod_stats keys) and overwrite the stale ``reference_energy`` — not skip
    silently and preserve the wrong values."""
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
    out = tmp_path / "stale.gsplats.zarr"
    build_recipe(data, "stream", params).save(out)

    # Simulate the old raw-amplitude annotate: a wrong e(k) on a sub-LOD the
    # build leaves unstamped, and a wrong (raw-convention) leaf w.
    root = zarr.open_group(str(out), mode="r+")
    sub = root["additive_0"]
    sub.attrs["lod_stats"] = {
        **dict(sub.attrs.get("lod_stats", {})),
        "energy_fraction_cum": 0.7,
    }
    root.attrs["level_stats"] = {
        **dict(root.attrs.get("level_stats", {})),
        "reference_energy": 123.0,
    }

    annotate_quality_store(out, device="cpu")

    root = zarr.open_group(str(out), mode="r")
    sub_stats = root["additive_0"].attrs["lod_stats"]
    assert "energy_fraction_cum" not in sub_stats
    assert "lod_method" in sub_stats  # other keys preserved
    # The stale w is overwritten with the (zero) alpha-effective total.
    assert root.attrs["level_stats"]["reference_energy"] == pytest.approx(0.0)


def test_annotate_rerun_repairs_stale_reference_energy(tmp_path: Path) -> None:
    """A standalone leaf's stale ``reference_energy`` (the old raw-amplitude
    convention, ~3× off on classical RGBA imports) must be REPAIRED by a
    re-run — the setdefault-only semantics would have preserved it forever."""
    data = _make_rgba_gsplat(n=200, seed=5)
    params = RecipeParams(n_lods=4, device="cpu", seed=0)
    out = tmp_path / "stale_w.gsplats.zarr"
    build_recipe(data, "stream", params).save(out)

    built = _collect_quality_attrs(out)
    built_w = built["/"]["level_stats.reference_energy"]

    root = zarr.open_group(str(out), mode="r+")
    root.attrs["level_stats"] = {
        **dict(root.attrs["level_stats"]),
        "reference_energy": built_w * 3.0,
    }

    annotate_quality_store(out, device="cpu")
    annotated = _collect_quality_attrs(out)
    assert annotated["/"]["level_stats.reference_energy"] == pytest.approx(
        built_w, rel=5e-2
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


def test_annotate_retrofits_spatial_footprint(levels_store: Path) -> None:
    root = zc_open_group(str(levels_store), mode="r+")
    for name in (key for key in root.group_keys() if key.startswith("child_")):
        child = root[name]
        stats = dict(child.attrs.get("level_stats", {}))
        stats.pop("median_footprint", None)
        stats.pop("footprint_dims", None)
        child.attrs["level_stats"] = stats
    zc_consolidate(root)

    annotate_quality_store(levels_store, device="cpu")

    root = zarr.open_group(str(levels_store), mode="r")
    for name in (key for key in root.group_keys() if key.startswith("child_")):
        stats = root[name].attrs["level_stats"]
        assert stats["median_footprint"] > 0
        assert stats["footprint_dims"] == [0, 1, 2]


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


@pytest.mark.parametrize("dry_run", [False, True])
def test_cheap_annotation_does_not_materialize_full_levels(
    levels_store: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    def fail_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("default annotation must not load full splat levels")

    monkeypatch.setattr("luxar.gsplats.lod.annotate._load_flat", fail_load)
    annotate_quality_store(levels_store, device="cpu", dry_run=dry_run)

    if not dry_run:
        root = zarr.open_group(str(levels_store), mode="r")
        for name in (key for key in root.group_keys() if key.startswith("child_")):
            stats = root[name].attrs["level_stats"]
            assert stats["median_footprint"] > 0
            assert stats["footprint_dims"] == [0, 1, 2]


def test_annotate_refreshes_content_hash(levels_store: Path) -> None:
    """New attrs must invalidate the viewer's persistent cache: the root
    content_hash changes and lands in consolidated metadata too."""
    from luxar._zarr_compat import read_consolidated_attrs

    _strip_quality_attrs(levels_store)
    before = zarr.open_group(str(levels_store), mode="r").attrs["content_hash"]
    annotate_quality_store(levels_store, device="cpu")
    root = zarr.open_group(str(levels_store), mode="r")
    after = root.attrs["content_hash"]
    assert after != before
    # The CONSOLIDATED copy is the one that matters: readers trust it over the
    # per-node attrs, so a hash written after consolidation would be invisible
    # to the viewer no matter how correct the per-node document looked. Read it
    # through the facade — the two formats key the index differently (v2 by
    # metadata document, v3 by node) and the root's attrs live outside the
    # index entirely at v3.
    assert read_consolidated_attrs(levels_store)["/"]["content_hash"] == after


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


def test_non_lod_annotation_never_decodes_cholesky_offdiag() -> None:
    """Ordinary leaves keep the cheap diagonal-only annotation path."""
    from luxar.gsplats.lod.annotate import AnnotateReport, _annotate_leaf

    offdiag = object()

    class _FakeGroup:
        def __init__(self) -> None:
            self._arrays: Dict[str, Any] = {
                "amplitudes": np.ones(2, dtype=np.float64),
                "cholesky_factors_diag": np.ones((2, 3), dtype=np.float64),
                "cholesky_factors_offdiag": offdiag,
            }
            self.attrs: Dict[str, Any] = {}
            self.path = "leaf"

        def __contains__(self, key: str) -> bool:
            return key in self._arrays

        def __getitem__(self, key: str) -> Any:
            return self._arrays[key]

    class _FakeDecoder:
        def decode(self, arr: Any, _root: Any) -> np.ndarray:
            if arr is offdiag:
                raise AssertionError("non-LOD annotation decoded off-diagonal factors")
            return np.asarray(arr)

    group = _FakeGroup()
    report = AnnotateReport(path="mem", dry_run=True)
    _annotate_leaf(group, group, _FakeDecoder(), report, dry_run=True)

    assert report.leaves[0].n_splats == 2


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


# ── reveal ladders must never be energy-stamped, by ANY writer ──────────────


def _radial_store(tmp_path: Path, method: str = "radial") -> Path:
    """A `stream` ladder ordered by ``method``, on a ball so radial is meaningful."""
    rng = np.random.default_rng(0)
    n = 300
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 50.0 * rng.random(n) ** (1 / 3)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, 0] = chol[:, 2] = chol[:, 5] = 1.5
    data = GSplatData(
        centers=(d * r[:, None]).astype(np.float32),
        amplitudes=(0.2 + rng.random(n)).astype(np.float32),
        cholesky_factors=chol,
    )
    out = tmp_path / f"{method}.gsplats.zarr"
    build_recipe(
        data, "stream", RecipeParams(n_lods=4, additive_method=method, seed=0)
    ).save(out)
    return out


def _stamp_counts(path: Path) -> "tuple[int, int, bool]":
    root = zarr.open(str(path), mode="r")
    n = int(root.attrs.get("n_additive_sublods", 1))
    stamped = sum(
        "energy_fraction_cum"
        in dict(root[f"additive_{i}"].attrs.get("lod_stats", {}) or {})
        for i in range(n)
    )
    has_w = "reference_energy" in dict(root.attrs.get("level_stats", {}) or {})
    return stamped, n, has_w


def test_annotate_does_not_energy_stamp_a_reveal_ladder(tmp_path: Path) -> None:
    """`annotate-quality` must not re-arm the 1/e(k) brightening on a reveal.

    This module's contract is to mirror the build path exactly, and the build path
    omits energy stamps for a reveal ordering — a radial prefix is a partial object
    at FULL brightness, so `1/e(k)` would blow out the innermost shell and then dim
    it as the object completes. Before this was guarded, annotating a radial store
    stamped every sub-LOD and added `reference_energy`, silently undoing the whole
    point of the ordering.
    """
    store = _radial_store(tmp_path)
    assert _stamp_counts(store) == (0, 4, False), "the BUILD must leave it unstamped"

    annotate_quality_store(store, device="cpu")
    stamped, n, has_w = _stamp_counts(store)
    assert stamped == 0, f"annotate stamped {stamped}/{n} sub-LODs of a reveal ladder"
    assert not has_w, "annotate added the paired reference_energy to a reveal ladder"


def test_annotate_report_does_not_claim_a_weight_it_erased(tmp_path: Path) -> None:
    """The report is what the CLI prints, so it must not name a w the store lacks.

    `annotate-quality` erases `reference_energy` from a reveal leaf; reporting the
    figure it computed anyway printed `w=<number>` under a "Leaves stamped" heading
    for a leaf that carries no weight at all.
    """
    store = _radial_store(tmp_path)
    report = annotate_quality_store(store, device="cpu")

    assert [leaf.reference_energy for leaf in report.leaves] == [None]
    assert [leaf.energy_fraction_cum for leaf in report.leaves] == [[]]

    # Sensitivity control: the same builder under an energy ordering DOES report
    # a weight, so the assertion above cannot pass by the field always being None.
    energy_store = _radial_store(tmp_path, method="self_energy")
    _strip_quality_attrs(energy_store)
    control = annotate_quality_store(energy_store, device="cpu")
    assert all(leaf.reference_energy is not None for leaf in control.leaves)


def test_annotate_repairs_a_wrongly_stamped_reveal_ladder(tmp_path: Path) -> None:
    """Erase, don't merely skip — a store stamped by an older build gets repaired.

    Mirrors the posture the zero-energy path already takes.
    """
    store = _radial_store(tmp_path)
    root = zarr.open(str(store), mode="a")
    n = int(root.attrs["n_additive_sublods"])
    for i in range(n):
        sub = root[f"additive_{i}"]
        stats = dict(sub.attrs.get("lod_stats", {}) or {})
        stats["energy_fraction_cum"] = 0.5
        sub.attrs["lod_stats"] = stats
    root.attrs["level_stats"] = {"reference_energy": 123.0}
    assert _stamp_counts(store) == (n, n, True), "fixture must start corrupted"

    annotate_quality_store(store, device="cpu")
    assert _stamp_counts(store) == (0, n, False), "annotate did not repair the store"


def test_annotate_still_stamps_a_non_reveal_ladder(tmp_path: Path) -> None:
    """SENSITIVITY CONTROL for the two tests above.

    Same builder, same shape, an energy-ordered method — which MUST be stamped.
    Without this, both tests above would pass if annotate stopped stamping at all.
    """
    store = _radial_store(tmp_path, method="self_energy")
    _strip_quality_attrs(store)
    annotate_quality_store(store, device="cpu")

    stamped, n, has_w = _stamp_counts(store)
    assert stamped == n, f"only {stamped}/{n} sub-LODs stamped on an energy ladder"
    assert has_w


def test_partitioned_radial_centres_each_part_on_itself_by_default(
    tmp_path: Path,
) -> None:
    """A DECISION, pinned: `tiles -m radial` self-centres each part by default.

    The ladder is built per part, so without an explicit centre each part reveals
    from its OWN bbox middle — N independent local reveals, not one object growing
    from its centre. That is useful (each visible, frustum-culled tile paints its
    own middle first) and surprising, so it is documented in the module README and
    pinned here: if someone changes the default, this test should make them do it
    deliberately.

    Passing `reveal_centre` switches to one coherent global reveal, which the
    second half asserts.
    """
    from luxar.gsplats.io.load_gsplats import load_gsplat_node
    from luxar.gsplats.tree import iter_leaves

    rng = np.random.default_rng(0)
    n = 800
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = 50.0 * rng.random(n) ** (1 / 3)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, 0] = chol[:, 2] = chol[:, 5] = 1.5
    data = GSplatData(
        centers=(d * r[:, None]).astype(np.float32),
        amplitudes=(0.2 + rng.random(n)).astype(np.float32),
        cholesky_factors=chol,
    )

    def first_shell_gap(centre: "list[float] | None") -> "tuple[float, float]":
        """(mean dist of each part's first shell to its OWN centre, to the GLOBAL)."""
        out = tmp_path / f"tiles_{centre is not None}.gsplats.zarr"
        params = RecipeParams(
            n_lods=3,
            additive_method="radial",
            max_elements=250,
            seed=0,
            reveal_centre=centre,
        )
        from luxar.gsplats.io.save_gsplats import write_gsplats_tree

        write_gsplats_tree(out, build_recipe(data, "tiles", params))
        node, _ = load_gsplat_node(out, include_stats=False)
        own, glob = [], []
        parts = [
            [np.asarray(s.centers, dtype=np.float64) for s in leaf.additive_sublods]
            for leaf in iter_leaves(node)
        ]
        assert len(parts) > 1, "fixture must actually partition"
        allc = np.concatenate([a for p in parts for a in p])
        gc = (allc.min(axis=0) + allc.max(axis=0)) / 2.0
        for p in parts:
            pall = np.concatenate(p)
            pc = (pall.min(axis=0) + pall.max(axis=0)) / 2.0
            own.append(float(np.linalg.norm(p[0] - pc, axis=1).mean()))
            glob.append(float(np.linalg.norm(p[0] - gc, axis=1).mean()))
        return sum(own) / len(own), sum(glob) / len(glob)

    own_default, glob_default = first_shell_gap(None)
    own_pinned, glob_pinned = first_shell_gap([0.0, 0.0, 0.0])

    # Default: first shells hug their OWN part centre, not the global one.
    assert own_default < glob_default, (own_default, glob_default)
    # Pinned: the global centre becomes the closer one — the ordering really moved.
    assert glob_pinned < glob_default, (glob_pinned, glob_default)


def test_with_quality_does_not_re_add_the_weight_to_a_reveal_level(
    tmp_path: Path,
) -> None:
    """The Q pass writes `reference_energy` per lod-group child — but not on a reveal.

    Otherwise a single `annotate-quality --with-quality` run contradicts itself: the
    e-pass erases the weight from a radial-laddered level and the Q pass immediately
    puts it back, leaving exactly the half-written pair the reveal exists to avoid.
    `quality` still goes on — it is a standalone readout, not half of the e pair.
    """
    data = _make_random_gsplat(n=400)
    out = tmp_path / "levels_radial.gsplats.zarr"
    build_recipe(
        data,
        "levels",
        RecipeParams(
            n_lods=3,
            levels=1,
            compression_factor=8,
            additive_method="radial",
            device="cpu",
            seed=0,
        ),
    ).save(out)

    annotate_quality_store(out, with_quality=True, device="cpu")

    root = zarr.open(str(out), mode="r")
    n_children = sum(1 for k in root.group_keys() if str(k).startswith("child_"))
    assert n_children >= 2, "the levels recipe must produce a kind=lod ladder"
    reveal_children = 0
    for i in range(n_children):
        child = root[f"child_{i}"]
        stats = dict(child.attrs.get("level_stats", {}) or {})
        n_sub = int(child.attrs.get("n_additive_sublods", 1))
        subs = [child[f"additive_{j}"] for j in range(n_sub)] if n_sub > 1 else [child]
        is_reveal = any(
            dict(s.attrs.get("lod_stats", {}) or {}).get("lod_method") == "radial"
            for s in subs
        )
        assert "quality" in stats, f"child_{i} lost its quality stamp"
        if is_reveal:
            reveal_children += 1
            assert "reference_energy" not in stats, (
                f"child_{i} is a reveal ladder but the Q pass re-added its weight"
            )
        else:
            # Control, in the same run: a level whose ladder collapsed to one
            # sub-LOD is not identifiable as a reveal, and keeps its weight.
            assert "reference_energy" in stats
    assert reveal_children, "no child was a radial ladder — the assertion was vacuous"
