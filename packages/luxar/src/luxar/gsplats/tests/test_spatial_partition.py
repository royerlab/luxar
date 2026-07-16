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
    b = (
        np.array([100.0, 100.0, 100.0], dtype=np.float32)
        + rng.uniform(0, 10, size=(n_per, 3))
    ).astype(np.float32)
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
        write_gsplats_tree(
            p,
            data.to_spatial_partition(max_elements=40),
            ordering="none",
            encoding_mode=EncodingMode.PRECISION,
        )
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


def _bsp_leaf_order(tree: dict) -> list[int]:
    """Leaf `part` refs of a serialized bsp_tree in left-first DFS order."""
    out: list[int] = []

    def walk(node: dict) -> None:
        if "part" in node:
            out.append(node["part"])
        else:
            walk(node["left"])
            walk(node["right"])

    walk(tree)
    return out


def test_spatial_partition_emits_bsp_tree_and_is_plane_consistent():
    """`to_spatial_partition` records the BSP split planes (`bsp_tree`): its
    leaves reference every part exactly once (in child_index order), and each
    internal split cleanly separates its subtrees (left coords < split <=
    right coords) — the invariant the viewer's exact back-to-front order relies
    on."""
    rng = np.random.default_rng(3)
    n = 400
    centers = (rng.random((n, 3)) * 100).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=centers,
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )
    part = data.to_spatial_partition(max_elements=60, rule="median")
    tree = part.bsp_tree
    assert tree is not None, "bsp_tree must be recorded"

    # Leaves reference each part once, in child_index (DFS) order.
    leaves = list(iter_leaves(part))
    assert _bsp_leaf_order(tree) == list(range(len(leaves)))

    # Split axes are spatial (0/1/2) — comparable in the viewer's 3D space.
    part_centers = {i: leaf.additive_sublods[0].centers for i, leaf in enumerate(leaves)}

    def parts_under(node: dict) -> list[int]:
        if "part" in node:
            return [node["part"]]
        return parts_under(node["left"]) + parts_under(node["right"])

    def check(node: dict) -> None:
        if "part" in node:
            return
        ax = node["axis"]
        assert ax in (0, 1, 2)
        sp = node["split"]
        left_coords = np.concatenate([part_centers[p][:, ax] for p in parts_under(node["left"])])
        right_coords = np.concatenate([part_centers[p][:, ax] for p in parts_under(node["right"])])
        # left holds coord < split; right holds coord >= split (ties → left).
        assert left_coords.max() <= sp + 1e-4
        assert right_coords.min() >= sp - 1e-4
        check(node["left"])
        check(node["right"])

    check(tree)


def test_spatial_partition_bsp_tree_round_trips_on_disk():
    data = _clustered(40)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "part.gsplats.zarr"
        src = data.to_spatial_partition(max_elements=40)
        write_gsplats_tree(p, src, ordering="none", encoding_mode=EncodingMode.PRECISION)
        root = zarr.open_group(str(p), mode="r")
        assert "bsp_tree" in root.attrs
        # Same leaf→part structure survives the write.
        assert _bsp_leaf_order(dict(root.attrs["bsp_tree"])) == _bsp_leaf_order(src.bsp_tree)
        # And read_gsplat_node restores it onto the in-memory node (disk→node),
        # the path the scene graft depends on.
        node = read_gsplat_node(root, root)
        assert isinstance(node, GSplatPartition)
        assert node.bsp_tree is not None
        assert _bsp_leaf_order(node.bsp_tree) == _bsp_leaf_order(src.bsp_tree)


def test_gsplat_info_handles_partition_file():
    """`gsplat info` must report a partition file's tree shape, not crash
    (GSplatData.load raises on a non-matrix tree — decision 5 gap)."""
    from luxar.cli.gsplat_ops.inspect_commands import info_dataset

    data = _clustered(40)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "part.gsplats.zarr"
        write_gsplats_tree(
            p, data.to_spatial_partition(max_elements=40), ordering="none"
        )
        # Must not raise (previously GSplatData.load → ValueError crashed info).
        info_dataset(p, show_histograms=False, bins=40)


def test_partition_file_grafts_into_a_scene():
    """A kind=partition .gsplats.zarr embeds into a scene as the identical
    kind=partition subtree. Regression: convert/add_gsplats_from_file used to
    crash on a partition root because GSplatData.load can't represent it. The
    graft composes the scene's own builders."""
    from luxar import Dimensions, LuxarZarrCompiler

    data = _clustered(40)  # 80 splats, 2 clusters
    with tempfile.TemporaryDirectory() as tmp:
        part = Path(tmp) / "part.gsplats.zarr"
        write_gsplats_tree(
            part, data.to_spatial_partition(max_elements=40), ordering="hilbert"
        )

        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_file(name="g", path=part)

        root = zarr.open_group(str(scene_path), mode="r")["g"]
        assert root.attrs["kind"] == "partition"
        assert root.attrs["display_type"] == "gsplats"
        n_parts = sum(1 for k in root if str(k).startswith("part_"))
        assert n_parts >= 2
        total = sum(root[f"part_{i}"]["centers"].shape[0] for i in range(n_parts))
        assert total == 80
        # The grafted partition WRAPPER must carry position_bounds (back-filled at
        # scene finalization) — add_partition_group doesn't compute the union the
        # standalone writer stamps, so without the back-fill the wrapper reaches the
        # viewer bounds-less, losing partition-unit culling / standalone parity.
        wrapper_pb = root.attrs.get("position_bounds")
        assert wrapper_pb is not None, "grafted partition wrapper lost position_bounds"
        # the wrapper union must contain every part's bounds.
        for i in range(n_parts):
            part_pb = root[f"part_{i}"].attrs["position_bounds"]
            for d in range(len(part_pb["min"])):
                assert wrapper_pb["min"][d] <= part_pb["min"][d]
                assert wrapper_pb["max"][d] >= part_pb["max"][d]
        # The BSP split planes must survive the graft (disk→node→scene) so the
        # viewer keeps exact back-to-front part ordering — the graft used to
        # drop everything but max_elements/position_bounds.
        assert "bsp_tree" in root.attrs, "grafted partition wrapper lost bsp_tree"
        assert _bsp_leaf_order(dict(root.attrs["bsp_tree"])) == list(range(n_parts))


def test_grafting_a_partition_rejects_dim_order():
    """A graft preserves the file's own coordinates — dim_order/fill cannot be
    applied to a partition/nested file, and must raise clearly (not silently
    no-op or crash deep in the writer)."""
    from luxar import Dimensions, LuxarZarrCompiler

    data = _clustered(40)
    with tempfile.TemporaryDirectory() as tmp:
        part = Path(tmp) / "part.gsplats.zarr"
        write_gsplats_tree(
            part, data.to_spatial_partition(max_elements=40), ordering="hilbert"
        )
        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            with pytest.raises(ValueError, match="dim_order / fill"):
                scene.add_gsplats_from_file(
                    name="g", path=part, dim_order=["z", "y", "x"]
                )


def test_grafted_lod_uses_coarsest_default_level():
    """Grafting a standalone kind=lod .gsplats.zarr into a scene must set the
    viewer default_level to the COARSEST child (0) — a progressive-load hint —
    NOT the finest. (Regression: graft_gsplat_node briefly carried the same
    finest-default bug fixed in the scene + standalone writers.)"""
    from luxar import Dimensions, LuxarZarrCompiler

    def _sub(n, seed):
        rng = np.random.default_rng(seed)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        return AdditiveSubLOD(
            centers=rng.uniform(0, 50, (n, 3)).astype(np.float32),
            amplitudes=np.ones(n, dtype=np.float32),
            cholesky_factors=chol,
        )

    pyr = GSplatData.from_substitutive_levels(
        [
            SubstitutiveLevel(additive_sublods=[_sub(100, 0)], level_index=0),
            SubstitutiveLevel(
                additive_sublods=[_sub(10, 1)], compression_factor=4, level_index=1
            ),
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        std = Path(tmp) / "x.gsplats.zarr"
        write_gsplats_tree(std, pyr.tree, ordering="none")
        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_file(name="g", path=std)
        g = zarr.open_group(str(scene_path), mode="r")["g"]
        assert g.attrs["kind"] == "lod"
        assert g.attrs["default_level"] == 0  # coarsest, NOT finest
        # child_0 is the coarsest (10 splats), not the finest (100).
        assert g["child_0"].attrs["n_splats"] == 10


def test_grafted_multiscale_stamps_coverage_fraction_on_partition_child():
    """Grafting a ``multiscale`` recipe (kind=lod over a kind=partition fine
    branch) must stamp the per-child ``coverage_fraction`` selector threshold on the
    PARTITION child — not only on the coarse leaf. Without it the viewer has no
    threshold to gate the fine branch on, so the lod can never switch off it and
    is stuck rendering the fine partition ("multiscale stuck at level 1" bug).

    Regression: ``graft_gsplat_node`` popped coverage_fraction for all non-leaf
    nodes but only re-applied it to the lod/partition WRAPPER for leaves; the
    partition wrapper got None. The standalone writer always stamped it, so the
    bug only bit the scene-graft path (add_gsplats_from_file / gsplat convert)."""
    from luxar import Dimensions, LuxarZarrCompiler
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
    from luxar.gsplats.tree import GSplatLodGroup

    rng = np.random.default_rng(0)
    n = 400
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = rng.uniform(0.5, 2.0, size=(n, 3))
    data = GSplatData(
        centers=rng.uniform(0, 100, size=(n, 3)).astype(np.float32),
        amplitudes=rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        cholesky_factors=chol,
    )
    ms = build_recipe(
        data,
        "overview",
        RecipeParams(max_elements=120, compression_factor=4, n_lods=3, device="cpu"),
    )
    assert isinstance(ms, GSplatLodGroup)

    with tempfile.TemporaryDirectory() as tmp:
        std = Path(tmp) / "ms.gsplats.zarr"
        write_gsplats_tree(std, ms, ordering="none")
        # The standalone writer must stamp the partition child too (the contract
        # the graft has to match).
        std_g = zarr.open_group(str(std), mode="r")
        assert std_g["child_1"].attrs["kind"] == "partition"
        assert std_g["child_1"].attrs["coverage_fraction"] is not None

        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(scene_path) as c:
            scene = c.create_scene(dimensions=Dimensions.default_3d())
            scene.add_gsplats_from_file(name="g", path=std)
        g = zarr.open_group(str(scene_path), mode="r")["g"]
        assert g.attrs["kind"] == "lod"
        # child_0 = coarse leaf cap (threshold 0 = always-eligible coarsest);
        # child_1 = fine partition branch, which MUST carry a positive threshold.
        # (A gsplats leaf writes no ``kind`` attr; only group nodes do.)
        assert g["child_0"].attrs.get("kind", "leaf") == "leaf"
        assert g["child_0"].attrs["coverage_fraction"] == 0.0
        assert g["child_1"].attrs["kind"] == "partition"
        fine_cov = g["child_1"].attrs["coverage_fraction"]
        assert fine_cov is not None, "partition child lost its coverage_fraction"
        # fractions must be strictly ascending coarsest -> finest for the
        # selector to switch (0 -> positive).
        assert fine_cov > g["child_0"].attrs["coverage_fraction"]
        # the graft must match the standalone writer's fraction exactly.
        assert fine_cov == std_g["child_1"].attrs["coverage_fraction"]
        # the fraction rides on the partition WRAPPER, not its parts.
        assert "coverage_fraction" not in g["child_1"]["part_0"].attrs


def test_gsplat_info_legacy_file_shows_migrate_hint_not_traceback():
    """`gsplat info` on a legacy (non-v3.0) file must surface the migrate-format
    hint and exit cleanly — NOT route into the tree summary and crash (review
    finding #5: the v3.0 rejection message contains 'node-tree', so the old
    substring dispatch mis-routed legacy files)."""
    import typer

    from luxar.cli.gsplat_ops.inspect_commands import info_dataset

    with tempfile.TemporaryDirectory() as tmp:
        legacy = Path(tmp) / "legacy.gsplats.zarr"
        root = zarr.open_group(str(legacy), mode="w")
        root.attrs["format_type"] = "gsplats_zarr"
        root.attrs["format_version"] = "2.0"  # legacy matrix format
        with pytest.raises(typer.Exit):
            info_dataset(legacy, show_histograms=False, bins=40)


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
        [
            SubstitutiveLevel(additive_sublods=[_sub(60, 0)], level_index=0),
            SubstitutiveLevel(
                additive_sublods=[_sub(15, 1)], compression_factor=4, level_index=1
            ),
        ]
    )
    with pytest.warns(UserWarning, match="flattens LOD structure"):
        part = pyr.to_spatial_partition(max_elements=30)
    # flattened to the default (finest) level → 60 splats partitioned
    assert sum(leaf.n_splats for leaf in iter_leaves(part)) == 60


# ── partition_from_regions per-part LOD recipe (fit --recipe seam) ──────────


def _region(n: int, offset: float, seed: int) -> GSplatData:
    """A small 3D region of ``n`` splats, spatially offset so regions are disjoint."""
    rng = np.random.default_rng(seed)
    centers = (offset + rng.uniform(0, 10, size=(n, 3))).astype(np.float32)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    return GSplatData(
        centers=centers,
        amplitudes=rng.uniform(0.2, 1.0, size=(n,)).astype(np.float32),
        cholesky_factors=chol,
    )


def test_partition_from_regions_no_recipe_keeps_bare_leaves():
    from luxar.gsplats.tree import GSplatLeaf

    regions = [_region(40, 0.0, 0), _region(40, 100.0, 1)]
    node = GSplatData.partition_from_regions(regions)
    assert isinstance(node, GSplatPartition)
    leaves = list(iter_leaves(node))
    assert len(leaves) == 2
    # bare leaves: a single (trivial) additive sub-LOD each
    assert all(isinstance(leaf, GSplatLeaf) for leaf in leaves)
    assert all(leaf.n_additive_sublods == 1 for leaf in leaves)
    assert sum(leaf.n_splats for leaf in leaves) == 80


def test_partition_from_regions_additive_recipe_gives_each_part_a_ladder():
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatLeaf

    regions = [_region(40, 0.0, 0), _region(40, 100.0, 1)]
    params = RecipeParams(n_lods=3, additive_method="greedy")
    node = GSplatData.partition_from_regions(
        regions, recipe="additive", recipe_params=params
    )
    assert isinstance(node, GSplatPartition)
    parts = node.children
    assert len(parts) == 2
    # additive → each part is a leaf carrying a multi-entry additive ladder
    assert all(isinstance(p, GSplatLeaf) for p in parts)
    assert all(p.n_additive_sublods == 3 for p in parts)
    # the ladder preserves the full splat set per part (prefix-sum, no loss)
    assert all(p.n_splats == 40 for p in parts)


def test_partition_from_regions_substitutive_recipe_gives_lod_groups():
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatLodGroup

    regions = [_region(40, 0.0, 0), _region(40, 100.0, 1)]
    params = RecipeParams(
        compression_factor=2, levels=2, substitutive_method="auto", device="cpu"
    )
    node = GSplatData.partition_from_regions(
        regions, recipe="substitutive", recipe_params=params
    )
    assert isinstance(node, GSplatPartition)
    # substitutive → each part is its own coarse↔fine lod group (mosaic)
    assert all(isinstance(p, GSplatLodGroup) for p in node.children)
    # the finest (default) level of each part keeps all 40 splats
    assert all(p.n_splats == 40 for p in node.children)


def test_partition_from_regions_single_region_recipe_returns_bare_part_node():
    """A single non-empty region returns the part node directly (no 1-part wrapper)
    — but still LOD-built when a recipe is given."""
    from luxar.gsplats.lod.recipes import RecipeParams
    from luxar.gsplats.tree import GSplatLeaf

    node = GSplatData.partition_from_regions(
        [_region(40, 0.0, 0)], recipe="additive", recipe_params=RecipeParams(n_lods=3)
    )
    assert isinstance(node, GSplatLeaf)  # not wrapped in a GSplatPartition
    assert node.n_additive_sublods == 3
