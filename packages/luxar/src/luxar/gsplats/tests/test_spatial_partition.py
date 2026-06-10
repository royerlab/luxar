"""GSplatData.to_spatial_partition — spatial BSP into a kind=partition tree (decision 5).

`gsplat partition` now produces ONE kind=partition .gsplats.zarr via a spatial
BSP (not N index-split files). Each part holds <= max_elements splats and carries
its own position_bounds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD, GSplatData, SubstitutiveLevel
from luxar.gsplats.io.save_gsplats import write_gsplats_tree
from luxar.gsplats.tree import GSplatPartition, iter_leaves
from luxar.io._compiler.gsplat_tree import read_gsplat_node


def _clustered(n_per: int = 30) -> GSplatData:
    """Two well-separated 3D clusters so a spatial split is unambiguous."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 10, size=(n_per, 3)).astype(np.float32)
    b = (np.array([100.0, 100.0, 100.0], dtype=np.float32)
         + rng.uniform(0, 10, size=(n_per, 3))).astype(np.float32)
    centers = np.concatenate([a, b], axis=0)
    n = centers.shape[0]
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )


def test_spatial_partition_respects_max_elements_and_preserves_splats():
    data = _clustered(30)  # 60 splats
    part = data.to_spatial_partition(max_elements=30)
    assert isinstance(part, GSplatPartition)
    leaves = list(iter_leaves(part))
    assert len(leaves) >= 2
    assert all(leaf.n_splats <= 30 for leaf in leaves)
    assert sum(leaf.n_splats for leaf in leaves) == 60


def test_spatial_partition_is_spatially_coherent():
    """The two separated clusters land in different parts (BSP splits on space)."""
    data = _clustered(30)
    part = data.to_spatial_partition(max_elements=30)
    # Each part should be tight (a single cluster's spatial extent < the gap).
    for leaf in iter_leaves(part):
        c = leaf.additive_sublods[0].centers
        extent = c.max(axis=0) - c.min(axis=0)
        assert extent.max() < 50.0  # << the ~100-unit inter-cluster gap


def test_spatial_partition_round_trips_as_single_file():
    data = _clustered(40)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "part.gsplats.zarr"
        write_gsplats_tree(p, data.to_spatial_partition(max_elements=40),
                           ordering="none", encoding_mode=EncodingMode.PRECISION)
        root = zarr.open_group(str(p), mode="r")
        assert root.attrs["kind"] == "partition"
        assert root.attrs["display_type"] == "gsplats"
        assert "position_bounds" in root.attrs  # union, for framing
        # round-trip the tree
        node = read_gsplat_node(root, root)
        assert isinstance(node, GSplatPartition)
        assert sum(leaf.n_splats for leaf in iter_leaves(node)) == 80
        # each part carries its own position_bounds
        n_parts = sum(1 for k in root if str(k).startswith("part_"))
        for i in range(n_parts):
            assert "position_bounds" in root[f"part_{i}"].attrs


def test_spatial_partition_validates_max_elements():
    with pytest.raises(ValueError, match="max_elements must be"):
        _clustered(5).to_spatial_partition(max_elements=0)


def test_spatial_partition_warns_on_multi_substitutive():
    def _sub(n, seed):
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        return AdditiveSubLOD(
            centers=rng.uniform(0, 100, (n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )

    pyr = GSplatData.from_substitutive_levels(
        [SubstitutiveLevel(additive_sublods=[_sub(60, 0)], level_index=0),
         SubstitutiveLevel(additive_sublods=[_sub(15, 1)], compression_factor=4, level_index=1)]
    )
    with pytest.warns(UserWarning, match="flattens LOD structure"):
        part = pyr.to_spatial_partition(max_elements=30)
    # flattened to the default (finest) level → 60 splats partitioned
    assert sum(leaf.n_splats for leaf in iter_leaves(part)) == 60
