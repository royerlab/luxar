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


# ── mixed-dimensionality rejection ───────────────────────────────────────
# ndim is read from the first sub-LOD/child; a mix would silently mis-describe
# the rest (and serialize/recombine inconsistently), so it must be rejected.


def test_leaf_rejects_mixed_ndim_sublods():
    with pytest.raises(ValueError, match="share one dimensionality"):
        GSplatLeaf(additive_sublods=[_sublod(10, ndim=3), _sublod(5, ndim=2)])


def test_lod_group_rejects_mixed_ndim_children():
    with pytest.raises(ValueError, match="share one dimensionality"):
        GSplatLodGroup(children=[_leaf(10, ndim=3), _leaf(5, ndim=4)])


def test_partition_rejects_mixed_ndim_children():
    with pytest.raises(ValueError, match="share one dimensionality"):
        GSplatPartition(children=[_leaf(30, ndim=3), _leaf(20, ndim=2)])


def test_nested_mixed_ndim_rejected_at_inner_node():
    # The inner lod group is built first and raises before the partition sees it.
    with pytest.raises(ValueError, match="share one dimensionality"):
        GSplatPartition(
            children=[GSplatLodGroup(children=[_leaf(10, ndim=3), _leaf(5, ndim=2)])]
        )


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


def test_tree_from_substitutive_levels_stamps_coverage_fractions():
    """The builder back-fills per-child ``coverage_fraction`` = ``sqrt(N_i/N_finest)``
    (coarsest 0.0, finest 1.0). ``levels`` is finest-first; the tree stores children
    coarsest-first, so the stamped fractions are ascending coarsest→finest."""
    import math

    # finest-first levels: counts 800, 200, 50 (coarsest = 50, last).
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(800, seed=0)], level_index=0),
        SubstitutiveLevel(additive_sublods=[_sublod(200, seed=1)], level_index=1),
        SubstitutiveLevel(additive_sublods=[_sublod(50, seed=2)], level_index=2),
    ]
    node = tree_from_substitutive_levels(levels)
    cov = [c.meta["coverage_fraction"] for c in node.children]  # coarsest-first
    assert cov[0] == 0.0  # coarsest = always-eligible floor
    # coarsest-first counts are [50, 200, 800]; N_finest = 800.
    assert cov[1] == pytest.approx(math.sqrt(200 / 800))
    assert cov[2] == pytest.approx(1.0)  # finest fills the screen
    assert cov[2] > cov[1] > cov[0]  # ascending coarsest→finest


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


def test_lod_group_back_fills_coverage_fraction():
    """A multi-substitutive tree gets per-child coverage_fraction so the viewer
    selector isn't stuck at the finest level (decision 7 / R3). Finest carries the
    highest fraction (1.0); the coarsest is 0.0 (always eligible)."""

    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(100, seed=0)], level_index=0),
        SubstitutiveLevel(
            additive_sublods=[_sublod(25, seed=1)], compression_factor=4, level_index=1
        ),
    ]
    node = tree_from_substitutive_levels(levels)
    assert isinstance(node, GSplatLodGroup)
    # children are coarsest-first: [0]=coarsest(25), [1]=finest(100)
    coarse_cov = node.children[0].meta["coverage_fraction"]
    finest_cov = node.children[1].meta["coverage_fraction"]
    # Coverage fractions = sqrt(N_i/N_finest): coarsest is the 0.0 floor, finest 1.0.
    assert coarse_cov == 0.0
    assert finest_cov == pytest.approx(1.0)
    assert finest_cov > coarse_cov


def test_coverage_fraction_explicit_meta_preserved():
    """The derived coverage_fraction is present on every child (setdefault path)."""
    levels = [
        SubstitutiveLevel(additive_sublods=[_sublod(100, seed=0)], level_index=0),
        SubstitutiveLevel(additive_sublods=[_sublod(25, seed=1)], level_index=1),
    ]
    node = tree_from_substitutive_levels(levels)
    # setdefault: derived values are present (no explicit override path here)
    assert all("coverage_fraction" in c.meta for c in node.children)


# ── map_leaves / default-selection global stats (PR-4 tree-aware transform) ──


def _leaf_xyz(centers: np.ndarray, amps) -> GSplatLeaf:
    """Build a single-sublod leaf with explicit centers + amplitudes (3D)."""
    centers = np.asarray(centers, dtype=np.float32)
    n, ndim = centers.shape
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = 1.0
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=np.asarray(amps, dtype=np.float32),
                cholesky_factors=chol,
            )
        ]
    )


def test_map_leaves_preserves_partition_shape_and_meta():
    from luxar.gsplats.tree import map_leaves

    leaf_a, leaf_b = _leaf(4, seed=0), _leaf(6, seed=1)
    tree = GSplatPartition(
        children=[leaf_a, leaf_b], max_elements=7, meta={"foo": "bar"}
    )
    out = map_leaves(
        tree,
        lambda lf: GSplatLeaf(additive_sublods=lf.additive_sublods, meta={"tagged": 1}),
    )
    assert isinstance(out, GSplatPartition)
    assert out.max_elements == 7
    assert out.meta == {"foo": "bar"}  # group meta copied, not mutated
    assert out.n_children == 2
    assert all(c.meta == {"tagged": 1} for c in out.children)
    # original tree is untouched (immutable rebuild)
    assert tree.children[0].meta == {} and tree.meta == {"foo": "bar"}


def test_map_leaves_nested_lod_inside_partition_visits_every_leaf():
    from luxar.gsplats.tree import map_leaves

    inner = GSplatLodGroup(children=[_leaf(2, seed=1), _leaf(8, seed=0)], meta={"m": 1})
    tree = GSplatPartition(children=[inner, _leaf(3, seed=2)])
    seen: list[int] = []

    def fn(lf: GSplatLeaf) -> GSplatLeaf:
        seen.append(lf.n_splats)
        return lf

    out = map_leaves(tree, fn)
    assert isinstance(out, GSplatPartition)
    assert isinstance(out.children[0], GSplatLodGroup)
    assert out.children[0].meta == {"m": 1}
    # fn applied to every leaf (both lod-group children + the standalone leaf)
    assert sorted(seen) == [2, 3, 8]


def test_amplitude_weighted_centroid_partition():
    from luxar.gsplats.tree import amplitude_weighted_centroid

    a = _leaf_xyz([[0.0, 0.0, 0.0]], [1.0])
    b = _leaf_xyz([[10.0, 0.0, 0.0]], [3.0])
    c = amplitude_weighted_centroid(GSplatPartition(children=[a, b]))
    # weighted mean: (0·1 + 10·3) / 4 = 7.5
    assert np.allclose(c, [7.5, 0.0, 0.0])


def test_global_stats_ignore_coarse_substitutive_levels():
    """A lod group contributes only its default (finest) child to global stats."""
    from luxar.gsplats.tree import amplitude_weighted_centroid, global_amplitude_max

    coarse = _leaf_xyz([[100.0, 0.0, 0.0]], [50.0])  # would dominate if counted
    fine = _leaf_xyz([[2.0, 0.0, 0.0]], [1.0])
    grp = GSplatLodGroup(children=[coarse, fine])  # coarsest→finest; default=fine
    assert np.allclose(amplitude_weighted_centroid(grp), [2.0, 0.0, 0.0])
    assert global_amplitude_max(grp) == pytest.approx(1.0)


def test_global_amplitude_max_partition():
    from luxar.gsplats.tree import global_amplitude_max

    a = _leaf_xyz([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [0.2, 0.9])
    b = _leaf_xyz([[5.0, 0.0, 0.0]], [0.5])
    assert global_amplitude_max(GSplatPartition(children=[a, b])) == pytest.approx(0.9)


def test_amplitude_weighted_centroid_zero_amplitude_falls_back_to_mean():
    from luxar.gsplats.tree import amplitude_weighted_centroid

    leaf = _leaf_xyz([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]], [0.0, 0.0])
    assert np.allclose(amplitude_weighted_centroid(leaf), [2.0, 0.0, 0.0])


def _leaf_4d_zero_time(centers, amps) -> GSplatLeaf:
    """4D leaf with isotropic spatial sigma=1 and a zero-variance time axis."""
    centers = np.asarray(centers, dtype=np.float32)
    n = centers.shape[0]
    chol = np.zeros((n, 10), dtype=np.float32)  # 4D packed lower-triangular
    chol[:, [0, 2, 5]] = 1.0  # spatial diagonals; dim3 diagonal (idx 9) stays 0
    return GSplatLeaf(
        additive_sublods=[
            AdditiveSubLOD(
                centers=centers,
                amplitudes=np.asarray(amps, dtype=np.float32),
                cholesky_factors=chol,
            )
        ]
    )


def test_nondegenerate_axes_all_spatial_3d():
    from luxar.gsplats.tree import nondegenerate_axes

    leaf = _leaf_xyz([[0, 0, 0], [1, 2, 3]], [1.0, 1.0])
    assert list(nondegenerate_axes(leaf)) == [0, 1, 2]


def test_nondegenerate_axes_excludes_zero_variance_time():
    from luxar.gsplats.tree import nondegenerate_axes

    leaf = _leaf_4d_zero_time([[1, 2, 3, 0], [4, 5, 6, 1]], [1.0, 1.0])
    assert list(nondegenerate_axes(leaf)) == [0, 1, 2]
    # Also over a partition of such leaves (the transform --center graft path).
    part = GSplatPartition(children=[leaf, leaf])
    assert list(nondegenerate_axes(part)) == [0, 1, 2]


def test_without_meta_key_scrubs_group_and_leaf_meta():
    """``without_meta_key`` drops the key from EVERY node — including group
    nodes that ``map_leaves`` copies verbatim — while preserving other meta.

    This is the mechanism that fixes the stale-``coverage_fraction`` bug on the
    transform tree path for multiscale / mosaic topologies.
    """
    from luxar.gsplats.tree import map_leaves, without_meta_key

    leaf_a = GSplatLeaf(
        additive_sublods=[_sublod(4, seed=0)],
        meta={"coverage_fraction": 0.1, "compression_factor": 4},
    )
    # multiscale-like: a lod group whose finest child is a partition that itself
    # carries a coverage_fraction on its GROUP meta.
    fine = GSplatPartition(
        children=[leaf_a, GSplatLeaf(additive_sublods=[_sublod(6, seed=1)])],
        meta={"coverage_fraction": 0.99},
    )
    root = GSplatLodGroup(
        children=[_leaf(3, seed=2), fine], meta={"coverage_fraction": 0.5}
    )

    scrubbed = without_meta_key(root, "coverage_fraction")
    assert "coverage_fraction" not in scrubbed.meta  # root group
    assert "coverage_fraction" not in scrubbed.children[1].meta  # partition group
    scrubbed_leaf = scrubbed.children[1].children[0]
    assert "coverage_fraction" not in scrubbed_leaf.meta
    assert scrubbed_leaf.meta["compression_factor"] == 4  # other provenance kept
    # original tree untouched (immutable rebuild)
    assert (
        root.meta["coverage_fraction"] == 0.5 and fine.meta["coverage_fraction"] == 0.99
    )

    # Contrast: map_leaves copies GROUP meta verbatim — it does NOT scrub the
    # partition's stale coverage_fraction (the bug this helper exists to close).
    mapped = map_leaves(root, lambda lf: lf)
    assert mapped.children[1].meta["coverage_fraction"] == 0.99
