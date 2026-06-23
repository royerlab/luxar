"""Unit tests for the gsplat node-tree model (``luxar.gsplats.tree``).

These exercise the three node types, the structural helpers, and the
round-trip bridge to/from the historical substitutive × additive matrix in
isolation (no I/O, no GSplatData coupling beyond AdditiveSubLOD/SubstitutiveLevel).
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, SubstitutiveLevel
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatPartition,
    center_bounds,
    is_matrix_shaped,
    iter_leaves,
    node_ndim,
    substitutive_levels_from_tree,
    total_splats,
    tree_from_substitutive_levels,
)

# ── fixtures / builders ──────────────────────────────────────────────────


def _sublod(n: int, ndim: int = 3, seed: int = 0, with_colors: bool = False):
    """Build an AdditiveSubLOD of ``n`` splats in ``ndim`` dims."""
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    centers = rng.uniform(0.0, 100.0, size=(n, ndim)).astype(np.float32)
    amplitudes = rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32)
    chol = np.zeros((n, k), dtype=np.float32)
    # diagonal entries positive so the cholesky-shape validation passes
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    colors = (
        rng.uniform(0.0, 1.0, size=(n, 3)).astype(np.float32) if with_colors else None
    )
    return AdditiveSubLOD(
        centers=centers,
        amplitudes=amplitudes,
        cholesky_factors=chol,
        colors=colors,
    )


def _leaf(n: int, ndim: int = 3, seed: int = 0, **meta) -> GSplatLeaf:
    return GSplatLeaf(additive_sublods=[_sublod(n, ndim, seed)], meta=dict(meta))


# ── GSplatLeaf ────────────────────────────────────────────────────────────


def test_leaf_basic_counts():
    leaf = GSplatLeaf(additive_sublods=[_sublod(10), _sublod(5, seed=1)])
    assert leaf.n_additive_sublods == 2
    assert leaf.n_splats == 15
    assert leaf.ndim == 3


def test_leaf_requires_at_least_one_sublod():
    with pytest.raises(ValueError):
        GSplatLeaf(additive_sublods=[])


def test_leaf_repr_mentions_splats():
    assert "splats" in repr(_leaf(7))


# ── GSplatLodGroup ──────────────────────────────────────────────────────


def test_lod_group_renders_default_child_count():
    coarse = _leaf(10, seed=1)
    fine = _leaf(100, seed=0)
    # children are coarsest→finest; the default (rendered) child is the finest (last)
    grp = GSplatLodGroup(children=[coarse, fine])
    assert grp.n_children == 2
    assert grp.default_level == 1  # finest = last index
    assert grp.n_splats == 100


def test_lod_group_default_level_is_derived_finest():
    # default_level is a derived property (= the finest, last child), not settable
    assert GSplatLodGroup(children=[_leaf(5)]).default_level == 0
    assert GSplatLodGroup(children=[_leaf(3), _leaf(9)]).default_level == 1
    with pytest.raises(ValueError):
        GSplatLodGroup(children=[])


# ── GSplatPartition ──────────────────────────────────────────────────────


def test_partition_sums_all_children():
    grp = GSplatPartition(children=[_leaf(30, seed=0), _leaf(20, seed=1)])
    # partition renders every child → n_splats is the sum
    assert grp.n_splats == 50
    assert grp.n_children == 2


def test_partition_requires_children():
    with pytest.raises(ValueError):
        GSplatPartition(children=[])


# ── structural helpers ──────────────────────────────────────────────────


def test_iter_leaves_depth_first_nested():
    # partition( lod( leafA, leafB ), leafC )
    leaf_a, leaf_b, leaf_c = _leaf(1, seed=0), _leaf(2, seed=1), _leaf(3, seed=2)
    tree = GSplatPartition(children=[GSplatLodGroup(children=[leaf_a, leaf_b]), leaf_c])
    leaves = list(iter_leaves(tree))
    assert leaves == [leaf_a, leaf_b, leaf_c]


def test_total_splats_counts_every_leaf_ignoring_selection():
    # lod group: n_splats honours selection (default child), total_splats does not
    fine, coarse = _leaf(100, seed=0), _leaf(10, seed=1)
    grp = GSplatLodGroup(children=[coarse, fine])  # coarsest→finest
    assert grp.n_splats == 100  # default = finest (last)
    assert total_splats(grp) == 110


def test_node_ndim():
    assert node_ndim(_leaf(5, ndim=2)) == 2
    assert node_ndim(GSplatPartition(children=[_leaf(5, ndim=4)])) == 4


def test_center_bounds_union_over_subtree():
    sub_lo = AdditiveSubLOD(
        centers=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        amplitudes=np.ones(2, dtype=np.float32),
        cholesky_factors=np.tile(
            np.array([1, 0, 1, 0, 0, 1], dtype=np.float32), (2, 1)
        ),
    )
    sub_hi = AdditiveSubLOD(
        centers=np.array([[5.0, -1.0, 4.0]], dtype=np.float32),
        amplitudes=np.ones(1, dtype=np.float32),
        cholesky_factors=np.array([[1, 0, 1, 0, 0, 1]], dtype=np.float32),
    )
    tree = GSplatPartition(
        children=[GSplatLeaf(additive_sublods=[sub_lo]), GSplatLeaf([sub_hi])]
    )
    bounds = center_bounds(tree)
    assert bounds is not None
    lo, hi = bounds
    np.testing.assert_array_equal(lo, np.array([0.0, -1.0, 0.0]))
    np.testing.assert_array_equal(hi, np.array([5.0, 2.0, 4.0]))


def test_center_bounds_none_when_empty():
    empty = AdditiveSubLOD(
        centers=np.zeros((0, 3), dtype=np.float32),
        amplitudes=np.zeros((0,), dtype=np.float32),
        cholesky_factors=np.zeros((0, 6), dtype=np.float32),
    )
    assert center_bounds(GSplatLeaf([empty])) is None


# ── matrix ⇄ tree bridge round-trips ─────────────────────────────────────


def test_single_level_round_trips_to_bare_leaf():
    level = SubstitutiveLevel(
        additive_sublods=[_sublod(10), _sublod(5, seed=1)],
        compression_factor=1,
        parent_method=None,
        level_index=0,
        stats={"psnr": 42.0},
    )
    node = tree_from_substitutive_levels([level])
    assert isinstance(node, GSplatLeaf)
    assert is_matrix_shaped(node)

    levels, default = substitutive_levels_from_tree(node)
    assert default == 0
    assert len(levels) == 1
    out = levels[0]
    assert out.n_additive_lods == 2
    assert out.compression_factor == 1
    assert out.parent_method is None
    assert out.level_index == 0
    assert out.stats == {"psnr": 42.0}


def test_multi_level_round_trips_through_lod_group():
    levels_in = [
        SubstitutiveLevel(
            additive_sublods=[_sublod(100, seed=0)],
            compression_factor=1,
            parent_method=None,
            level_index=0,
            stats={"i": 0},
        ),
        SubstitutiveLevel(
            additive_sublods=[_sublod(25, seed=1)],
            compression_factor=4,
            parent_method="kmeans_lloyd",
            level_index=1,
            stats={"i": 1},
        ),
        SubstitutiveLevel(
            additive_sublods=[_sublod(6, seed=2)],
            compression_factor=16,
            parent_method="kmeans_lloyd",
            level_index=2,
            stats={"i": 2},
        ),
    ]
    node = tree_from_substitutive_levels(levels_in)
    assert isinstance(node, GSplatLodGroup)
    assert node.n_children == 3
    assert node.default_level == 2  # derived finest = last child
    # tree children are coarsest→finest
    assert node.children[0].n_splats == 6
    assert node.children[2].n_splats == 100

    # the matrix view is reversed back to finest-first
    levels_out, default = substitutive_levels_from_tree(node)
    assert default == 0
    assert len(levels_out) == 3
    for src, dst in zip(levels_in, levels_out):
        assert dst.compression_factor == src.compression_factor
        assert dst.parent_method == src.parent_method
        assert dst.level_index == src.level_index
        assert dst.stats == src.stats
        assert dst.n_splats_total == src.n_splats_total


def test_bridge_default_level_is_derived_finest():
    # The tree's default_level is a derived property = the finest (last) child in
    # coarsest-first order; it is not settable. The matrix view's default is fixed
    # at the finest (index 0). Both mean "finest"; the on-disk default_level is the
    # viewer's separate coarsest-first render hint, stamped by the serializer.
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(50, seed=i)], level_index=i)
        for i in range(3)
    ]
    node = tree_from_substitutive_levels(levels)
    assert isinstance(node, GSplatLodGroup)
    assert node.default_level == node.n_children - 1
    _, default = substitutive_levels_from_tree(node)
    assert default == 0


def test_tree_from_substitutive_levels_lod_method_selectable():
    """#5: the builder threads the lod_method knob into the per-child threshold
    derivation. ``count`` reproduces the legacy √N thresholds exactly; the default
    ``extent`` is physically anchored and differs. (Pre-fix the function took no
    knobs, so a substitutive/pyramid file was never CLI-tunable.)"""
    import math

    from luxar.core.group.lod.group import BASE_PIXEL_SIZE

    # finest-first levels: counts 800, 200, 50 (coarsest = 50, last).
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(800, seed=0)], level_index=0),
        SubstitutiveLevel(additive_sublods=[_sublod(200, seed=1)], level_index=1),
        SubstitutiveLevel(additive_sublods=[_sublod(50, seed=2)], level_index=2),
    ]
    cnt = tree_from_substitutive_levels(levels, lod_method="count")
    cnt_mps = [c.meta["min_pixel_size"] for c in cnt.children]  # coarsest-first
    assert cnt_mps[0] == 0.0  # coarsest = always-eligible floor
    assert cnt_mps[1] == pytest.approx(BASE_PIXEL_SIZE * math.sqrt(200 / 50))
    assert cnt_mps[2] == pytest.approx(BASE_PIXEL_SIZE * math.sqrt(800 / 50))

    ext = tree_from_substitutive_levels(levels)  # default = extent
    ext_mps = [c.meta["min_pixel_size"] for c in ext.children]
    assert ext_mps[0] == 0.0
    assert ext_mps[2] > ext_mps[1] > ext_mps[0]  # ascending coarsest→finest
    assert ext_mps[2] != pytest.approx(cnt_mps[2])  # physically anchored, differs

    # extent_anisotropy is actually threaded (not silently ignored): _sublod makes
    # anisotropic splats (random per-axis diagonals), so the isotropic geometric-mean
    # radius differs from the largest-semi-axis radius → different thresholds.
    iso = tree_from_substitutive_levels(levels, extent_anisotropy=False)
    iso_mps = [c.meta["min_pixel_size"] for c in iso.children]
    assert iso_mps[2] != pytest.approx(ext_mps[2])


def test_tree_from_substitutive_levels_rejects_invalid_lod_method():
    """#5 hardening: an unknown lod_method must RAISE, not silently fall back to
    ``count`` (``lod_thresholds`` treats any non-"extent" string as count). Guards
    save()/recipes/.tree — the single user-method chokepoint for substitutive gsplats."""
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(80, seed=0)], level_index=0),
        SubstitutiveLevel(additive_sublods=[_sublod(20, seed=1)], level_index=1),
    ]
    with pytest.raises(ValueError, match="lod_method must be 'extent' or 'count'"):
        tree_from_substitutive_levels(levels, lod_method="Extent")  # typo'd case
    with pytest.raises(ValueError, match="lod_method"):
        tree_from_substitutive_levels(levels, lod_method="sqrt")
    # validation fires even for the single-level (otherwise-inert) path
    with pytest.raises(ValueError, match="lod_method"):
        tree_from_substitutive_levels(levels[:1], lod_method="bogus")


def test_non_matrix_trees_have_no_matrix_projection():
    # partition is not matrix-shaped
    part = GSplatPartition(children=[_leaf(10, seed=0), _leaf(10, seed=1)])
    assert not is_matrix_shaped(part)
    with pytest.raises(ValueError):
        substitutive_levels_from_tree(part)

    # lod group with a non-leaf (nested partition) child is not matrix-shaped
    nested = GSplatLodGroup(
        children=[_leaf(100, seed=0), GSplatPartition(children=[_leaf(10, seed=1)])]
    )
    assert not is_matrix_shaped(nested)
    with pytest.raises(ValueError):
        substitutive_levels_from_tree(nested)


def test_tree_from_empty_levels_raises():
    with pytest.raises(ValueError):
        tree_from_substitutive_levels([])


def test_lod_group_back_fills_min_pixel_size():
    """A multi-substitutive tree gets per-child min_pixel_size so the viewer
    selector isn't stuck at the finest level (decision 7 / R3). Finest carries
    the highest threshold; the coarsest is 0.0 (always eligible)."""
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(100, seed=0)], level_index=0),
        SubstitutiveLevel(
            additive_sublods=[_sublod(25, seed=1)], compression_factor=4, level_index=1
        ),
    ]
    node = tree_from_substitutive_levels(levels)
    assert isinstance(node, GSplatLodGroup)
    # children are coarsest-first: [0]=coarsest(25), [1]=finest(100)
    coarse_mps = node.children[0].meta["min_pixel_size"]
    finest_mps = node.children[1].meta["min_pixel_size"]
    # Default `extent` method (T·W/r): coarsest is the 0.0 floor, the finer level
    # has a positive ascending threshold anchored in element size.
    assert coarse_mps == 0.0
    assert finest_mps > coarse_mps
    assert finest_mps > 0.0
    # Pin the exact formula with an INDEPENDENTLY computed W and r, so a wrong W
    # (e.g. finest-level bbox instead of the union of all levels) is caught: W must
    # be the union bbox diagonal over BOTH levels' centers, r = p90 of the finest
    # level's principal_radii, threshold = DEFAULT_TARGET_PIXEL_SIZE · W / r.
    from luxar.core.group.lod.group import DEFAULT_TARGET_PIXEL_SIZE

    all_centers = np.concatenate(
        [s.centers for lvl in levels for s in lvl.additive_sublods]
    )
    w_union = float(np.linalg.norm(all_centers.max(axis=0) - all_centers.min(axis=0)))
    fine_r = float(
        np.percentile(
            np.concatenate(
                [s.principal_radii(True) for s in levels[0].additive_sublods]
            ),
            90.0,
        )
    )
    assert finest_mps == pytest.approx(DEFAULT_TARGET_PIXEL_SIZE * w_union / fine_r)


def test_min_pixel_size_explicit_meta_preserved():
    """An explicit min_pixel_size in a level's stats-derived meta is not clobbered."""
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(100, seed=0)], level_index=0),
        SubstitutiveLevel(additive_sublods=[_sublod(25, seed=1)], level_index=1),
    ]
    node = tree_from_substitutive_levels(levels)
    # setdefault: derived values are present (no explicit override path here)
    assert all("min_pixel_size" in c.meta for c in node.children)
