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
    data = GSplatData.from_substitutive_levels(levels, default_substitutive=0)
    node = data.tree
    assert isinstance(node, GSplatLodGroup)
    assert node.n_children == 2
    assert node.default_level == 0


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
    data = GSplatData.from_substitutive_levels(
        levels, stats={"source": "test"}, default_substitutive=1
    )

    rebuilt = GSplatData.from_tree(data.tree, stats=dict(data.stats))

    assert rebuilt.n_substitutive == data.n_substitutive == 2
    assert rebuilt.default_substitutive == data.default_substitutive == 1
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
