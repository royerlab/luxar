"""Round-trip tests for the GSplatData ⇄ node-tree bridge (.tree / from_tree).

These guard the Phase-1 invariant that the tree is a faithful, lossless
re-expression of the historical substitutive × additive matrix: arrays,
per-level provenance metadata, default_substitutive, and stats all survive a
GSplatData → tree → GSplatData round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup, GSplatPartition


def _sublod(n: int, ndim: int = 3, seed: int = 0) -> AdditiveSubLOD:
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    return AdditiveSubLOD(
        centers=rng.uniform(0, 100, size=(n, ndim)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    )


def test_additive_sublods_property_is_a_defensive_copy():
    """Mutating the list returned by ``.additive_sublods`` must NOT corrupt the
    ground-truth node or desync the cached ``centers``/``n_splats``.

    Regression: after the tree-backing refactor the property returned the node's
    internal list by reference (it was a defensive copy before), so
    ``data.additive_sublods.append(...)`` leaked into ``data._node`` and left
    ``n_additive_sublods`` (=len of the mutated list) inconsistent with the
    stale cached ``n_splats``.
    """
    data = GSplatData.from_additive_sublods([_sublod(10, seed=0)])
    assert data.n_additive_sublods == 1

    returned = data.additive_sublods
    returned.append(_sublod(7, seed=1))  # mutate the returned list in place

    # The node, the derived view, and the cached arrays must all be untouched.
    assert data.n_additive_sublods == 1
    assert len(data.additive_sublods) == 1
    assert isinstance(data.tree, GSplatLeaf)
    assert len(data.tree.additive_sublods) == 1  # node not corrupted
    assert data.n_splats == 10  # cached centers still consistent


def test_readonly_view_stats_are_detached():
    """A read-only single-level view (at_substitutive / flattened / additive_prefix)
    must be immutable through-and-through: mutating the view's per-sublod ``stats``
    must NOT leak into the source node's lod_stats.

    Regression: ``_readonly_sublod`` made arrays read-only but aliased the stats
    dict (``stats=lod.stats``), so a "read-only" view could still corrupt the
    node's on-disk-bound lod_stats via ``view.additive_sublod(0).stats[k] = v``.
    """
    sub = AdditiveSubLOD(
        centers=_sublod(8, seed=0).centers,
        amplitudes=_sublod(8, seed=0).amplitudes,
        cholesky_factors=_sublod(8, seed=0).cholesky_factors,
        stats={"psnr": 40.0},
    )
    data = GSplatData.from_additive_sublods([sub])

    view = data.at_substitutive(0)
    view.additive_sublod(0).stats["psnr"] = 999.0  # mutate the read-only view's stats

    # The source node's lod stats must be untouched.
    assert data.additive_sublod(0).stats["psnr"] == 40.0
    assert data.tree.additive_sublods[0].stats["psnr"] == 40.0


def test_single_substitutive_tree_is_a_leaf():
    data = GSplatData(
        centers=_sublod(20).centers,
        amplitudes=_sublod(20).amplitudes,
        cholesky_factors=_sublod(20).cholesky_factors,
    )
    node = data.tree
    assert isinstance(node, GSplatLeaf)
    assert node.n_splats == data.n_splats


def test_additive_ladder_preserved_in_leaf():
    data = GSplatData.from_additive_sublods([_sublod(30, seed=0), _sublod(10, seed=1)])
    node = data.tree
    assert isinstance(node, GSplatLeaf)
    assert node.n_additive_sublods == 2
    assert node.n_splats == 40


def test_multi_substitutive_tree_is_a_lod_group():
    levels = [
        SubstitutiveLevel(
            additive_sublods=[_sublod(100, seed=0)],
            compression_factor=1,
            parent_method=None,
            level_index=0,
        ),
        SubstitutiveLevel(
            additive_sublods=[_sublod(25, seed=1)],
            compression_factor=4,
            parent_method="kmeans_lloyd",
            level_index=1,
        ),
    ]
    data = GSplatData.from_substitutive_levels(levels)
    node = data.tree
    assert isinstance(node, GSplatLodGroup)
    assert node.n_children == 2
    # tree default_level is the derived finest = last child (coarsest-first storage)
    assert node.default_level == node.n_children - 1
    # tree children are coarsest-first; the matrix view stays finest-first
    assert node.children[0].n_splats == 25
    assert node.children[-1].n_splats == 100
    assert [lvl.n_splats_total for lvl in data.substitutive_levels] == [100, 25]


def test_round_trip_preserves_arrays_metadata_and_stats():
    levels = [
        SubstitutiveLevel(
            additive_sublods=[_sublod(100, seed=0), _sublod(40, seed=10)],
            compression_factor=1,
            parent_method=None,
            level_index=0,
            stats={"psnr": 41.0},
        ),
        SubstitutiveLevel(
            additive_sublods=[_sublod(25, seed=1)],
            compression_factor=4,
            parent_method="greedy",
            level_index=1,
            stats={"psnr": 33.0},
        ),
    ]
    data = GSplatData.from_substitutive_levels(levels, stats={"source": "test"})

    rebuilt = GSplatData.from_tree(data.tree, stats=dict(data.stats))

    assert rebuilt.n_substitutive == data.n_substitutive == 2
    # The data-model default is fixed at the finest level (index 0).
    assert rebuilt.default_substitutive == data.default_substitutive == 0
    assert rebuilt.stats == {"source": "test"}
    for s in range(data.n_substitutive):
        src = data.substitutive_levels[s]
        dst = rebuilt.substitutive_levels[s]
        assert dst.compression_factor == src.compression_factor
        assert dst.parent_method == src.parent_method
        assert dst.level_index == src.level_index
        assert dst.stats == src.stats
        assert dst.n_additive_lods == src.n_additive_lods
        for a in range(src.n_additive_lods):
            np.testing.assert_array_equal(
                dst.additive_sublods[a].centers, src.additive_sublods[a].centers
            )
            np.testing.assert_array_equal(
                dst.additive_sublods[a].amplitudes, src.additive_sublods[a].amplitudes
            )


def test_from_tree_rejects_partition():
    leaf = GSplatLeaf(additive_sublods=[_sublod(10)])
    part = GSplatPartition(children=[leaf, GSplatLeaf([_sublod(5, seed=1)])])
    with pytest.raises(ValueError):
        GSplatData.from_tree(part)
