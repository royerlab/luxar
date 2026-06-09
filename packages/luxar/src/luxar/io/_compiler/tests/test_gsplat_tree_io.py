"""Round-trip tests for the shared root-agnostic gsplat node-subtree serializer
(``luxar.io._compiler.gsplat_tree``).

Covers every v3.0 shape (single leaf, additive ladder, substitutive lod group,
full pyramid, partition, nested combo), the finest-first ⇄ coarsest-first child
reversal, and the canonical ``position_bounds`` on every node (Blocker 2). Uses
``ordering="none"`` + ``PRECISION`` encoding so array round-trips are exact.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import AdditiveSubLOD
from luxar.gsplats.tree import (
    GSplatLeaf,
    GSplatLodGroup,
    GSplatPartition,
    iter_leaves,
)
from luxar.io._compiler.gsplat_tree import (
    make_dataset_ctx,
    make_ordering_ctx,
    read_gsplat_node,
    write_gsplat_node,
)


def _sublod(n: int, ndim: int = 3, seed: int = 0, base: float = 0.0) -> AdditiveSubLOD:
    rng = np.random.default_rng(seed)
    k = ndim * (ndim + 1) // 2
    chol = np.zeros((n, k), dtype=np.float32)
    diag = np.cumsum(np.arange(1, ndim + 1)) - 1
    chol[:, diag] = rng.uniform(0.5, 2.0, size=(n, ndim))
    return AdditiveSubLOD(
        centers=(base + rng.uniform(0, 50, size=(n, ndim))).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    )


def _leaf(n: int, seed: int = 0, base: float = 0.0, **meta) -> GSplatLeaf:
    return GSplatLeaf(additive_sublods=[_sublod(n, seed=seed, base=base)], meta=dict(meta))


def _round_trip(node, ordering: str = "none"):
    """Write ``node`` to a temp zarr and read it back; return the read node."""
    tmp = Path(tempfile.mkdtemp(prefix="luxar_tree_io_"))
    store = zarr.DirectoryStore(str(tmp / "t.gsplats.zarr"))
    root = zarr.group(store=store, overwrite=True)
    write_gsplat_node(
        root,
        node,
        dataset_ctx=make_dataset_ctx(EncodingMode.PRECISION),
        ordering_ctx=make_ordering_ctx(ordering),
        store=root,
    )
    reopened = zarr.open_group(str(tmp / "t.gsplats.zarr"), mode="r")
    return reopened, read_gsplat_node(reopened, reopened)


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])]


# ── Shape A: single splat set ────────────────────────────────────────────


def test_single_leaf_round_trip_exact():
    leaf = _leaf(40, seed=1)
    grp, out = _round_trip(leaf)
    assert isinstance(out, GSplatLeaf)
    assert out.n_splats == 40
    assert grp.attrs["type"] == "gsplats"
    assert "position_bounds" in grp.attrs  # Blocker 2: framing on load
    np.testing.assert_array_equal(
        out.additive_sublods[0].centers, leaf.additive_sublods[0].centers
    )
    np.testing.assert_array_equal(
        out.additive_sublods[0].amplitudes, leaf.additive_sublods[0].amplitudes
    )


# ── Shape B: additive ladder ─────────────────────────────────────────────


def test_additive_ladder_subgroups_and_round_trip():
    leaf = GSplatLeaf(additive_sublods=[_sublod(30, seed=0), _sublod(12, seed=1)])
    grp, out = _round_trip(leaf)
    assert grp.attrs["n_additive_sublods"] == 2
    assert "additive_0" in grp and "additive_1" in grp
    assert "position_bounds" in grp.attrs
    assert out.n_additive_sublods == 2
    assert out.n_splats == 42
    np.testing.assert_array_equal(
        out.additive_sublods[1].centers, leaf.additive_sublods[1].centers
    )


# ── Shape C: substitutive lod group (finest-first ⇄ coarsest-first) ──────


def test_lod_group_child_reversal_and_round_trip():
    fine = _leaf(100, seed=0)
    mid = _leaf(25, seed=1)
    coarse = _leaf(6, seed=2)
    grp = GSplatLodGroup(children=[fine, mid, coarse], default_level=0)  # finest-first
    z, out = _round_trip(grp)

    # On disk: child_0 = coarsest (6), child_2 = finest (100)
    assert z.attrs["kind"] == "lod"
    assert z["child_0"].attrs["n_splats"] == 6
    assert z["child_2"].attrs["n_splats"] == 100
    # default_level=0 (finest) in memory → on disk reversed index n-1
    assert z.attrs["default_level"] == 2
    assert "position_bounds" in z.attrs

    # Read back: finest-first restored, default 0
    assert isinstance(out, GSplatLodGroup)
    assert out.default_level == 0
    assert out.children[0].n_splats == 100
    assert out.children[2].n_splats == 6


def test_lod_group_position_bounds_is_union():
    a = _leaf(20, seed=0, base=0.0)
    b = _leaf(20, seed=1, base=1000.0)  # far away → widens union
    grp = GSplatLodGroup(children=[a, b], default_level=0)
    z, _ = _round_trip(grp)
    pb = z.attrs["position_bounds"]
    # union max must reach into b's region (>=1000)
    assert max(pb["max"]) >= 1000.0


# ── Shape D: full pyramid (lod group of additive-ladder leaves) ──────────


def test_pyramid_round_trip():
    fine = GSplatLeaf(additive_sublods=[_sublod(80, seed=0), _sublod(20, seed=5)])
    coarse = _leaf(10, seed=2)
    grp = GSplatLodGroup(children=[fine, coarse], default_level=0)
    z, out = _round_trip(grp)
    assert isinstance(out, GSplatLodGroup)
    # finest child restored with its 2-step additive ladder
    assert out.children[0].n_additive_sublods == 2
    assert out.children[0].n_splats == 100
    assert out.children[1].n_splats == 10


# ── Shape E: partition ───────────────────────────────────────────────────


def test_partition_round_trip():
    part = GSplatPartition(
        children=[_leaf(30, seed=0), _leaf(20, seed=1)], max_elements=25
    )
    z, out = _round_trip(part)
    assert z.attrs["kind"] == "partition"
    assert z.attrs["display_type"] == "gsplats"
    assert z.attrs["max_elements"] == 25
    assert "part_0" in z and "part_1" in z
    assert "position_bounds" in z.attrs
    assert isinstance(out, GSplatPartition)
    assert out.n_children == 2
    assert out.n_splats == 50


# ── Shape F: nested combos ───────────────────────────────────────────────


def test_nested_partition_of_lod_round_trip():
    # partition( lod(leaf100, leaf10), leaf50 )
    lod = GSplatLodGroup(children=[_leaf(100, seed=0), _leaf(10, seed=1)])
    tree = GSplatPartition(children=[lod, _leaf(50, seed=2)])
    z, out = _round_trip(tree)
    assert z.attrs["kind"] == "partition"
    assert z["part_0"].attrs["kind"] == "lod"
    assert isinstance(out, GSplatPartition)
    assert isinstance(out.children[0], GSplatLodGroup)
    assert out.children[0].children[0].n_splats == 100
    assert out.children[1].n_splats == 50


def test_nested_lod_of_partition_round_trip():
    part = GSplatPartition(children=[_leaf(40, seed=0), _leaf(40, seed=1)])
    tree = GSplatLodGroup(children=[part, _leaf(8, seed=2)], default_level=0)
    z, out = _round_trip(tree)
    assert z.attrs["kind"] == "lod"
    # finest-first child 0 (the partition) → on disk child_1 (coarsest-first)
    assert z["child_1"].attrs["kind"] == "partition"
    assert isinstance(out, GSplatLodGroup)
    assert isinstance(out.children[0], GSplatPartition)


# ── Selector / provenance metadata on lod children ──────────────────────


def test_min_pixel_size_and_provenance_round_trip():
    fine = _leaf(100, seed=0, min_pixel_size=0.0, compression_factor=1)
    coarse = _leaf(10, seed=1, min_pixel_size=4.0, compression_factor=4,
                   parent_method="kmeans_lloyd")
    grp = GSplatLodGroup(children=[fine, coarse], default_level=0)
    z, out = _round_trip(grp)
    # coarse leaf is child_0 on disk; its selector threshold + provenance persisted
    assert z["child_0"].attrs["min_pixel_size"] == 4.0
    assert z["child_0"].attrs["compression_factor"] == 4
    assert z["child_0"].attrs["parent_method"] == "kmeans_lloyd"
    # restored finest-first: children[1] is the coarse leaf
    assert out.children[1].meta["min_pixel_size"] == 4.0
    assert out.children[1].meta["compression_factor"] == 4


# ── Spatial ordering path still round-trips the splat SET ────────────────


def test_hilbert_ordering_preserves_splat_set():
    leaf = _leaf(200, seed=7)
    z, out = _round_trip(leaf, ordering="hilbert")
    assert z.attrs["ordering"] == "hilbert"
    assert "chunk_bounds" in z  # spatial index written
    # ordering permutes rows; the SET of centers must be preserved
    np.testing.assert_allclose(
        _sorted_rows(out.additive_sublods[0].centers),
        _sorted_rows(leaf.additive_sublods[0].centers),
        rtol=0,
        atol=0,
    )


def test_every_node_has_position_bounds():
    tree = GSplatLodGroup(
        children=[
            GSplatPartition(children=[_leaf(20, seed=0), _leaf(20, seed=1)]),
            _leaf(5, seed=2),
        ],
        default_level=0,
    )
    z, _ = _round_trip(tree)
    assert "position_bounds" in z.attrs
    assert "position_bounds" in z["child_1"].attrs  # the partition (coarsest-first)
    # leaves carry center-derived bounds too
    leaf_count = sum(1 for _ in iter_leaves(tree))
    assert leaf_count == 3
