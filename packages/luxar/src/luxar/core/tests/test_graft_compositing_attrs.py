"""Attr routing when a nested `.gsplats.zarr` is GRAFTED into a scene.

``graft_gsplat_node`` (``core/group/gsplats_pipeline/from_io.py``) is the third
writer of a multi-node gsplats subtree, next to
``lod_dispatch.add_gsplats_as_lod_group_impl`` and the ``adders/``
partition-wrapping path. All three must route
:data:`~luxar.core.group.compositing.COMPOSITING_ATTRS` the same way — onto the
WRAPPER only — because the viewer resolves them down the ancestry.

``blending_mode`` is the one that bites: it is *nearest-setter-wins*, not
multiplicative. Re-stamping it on each grafted part (which the graft path used
to do "belt-and-suspenders") makes every part SHADOW the wrapper, so the layers
panel's single Blend control becomes inert on exactly the part-based recipes
(``tiles`` / ``overview`` / ``adaptive``) while flat/stream/levels layers keep
responding.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import zarr

from luxar import Dimension, Dimensions, LuxarZarrCompiler
from luxar.encoding import EncodingMode
from luxar.gsplats.gsplat_data import GSplatData
from luxar.gsplats.io.save_gsplats import write_gsplats_tree


def _two_clusters(n_per: int = 30) -> GSplatData:
    """Two well-separated clusters so the BSP split is unambiguous."""
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


def _partition_file(dirpath: Path) -> Path:
    """A standalone kind=partition ``.gsplats.zarr`` (what ``fit --tiling`` writes)."""
    path = dirpath / "part.gsplats.zarr"
    write_gsplats_tree(
        path,
        _two_clusters(30).to_spatial_partition(max_elements=30),
        ordering="none",
        encoding_mode=EncodingMode.PRECISION,
    )
    return path


def _scene_dims() -> Dimensions:
    return Dimensions(
        [
            Dimension("x", unit="px", display=True),
            Dimension("y", unit="px", display=True),
            Dimension("z", unit="px", display=True),
        ]
    )


def test_grafted_partition_keeps_blending_mode_on_the_wrapper_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = _partition_file(tmpdir)
        scene_path = tmpdir / "scene.luxar.zarr"

        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            scene.add_gsplats_from_file(
                "tiles",
                str(src),
                opacity=1.0,
                absorption=1.0,
                blending_mode="volumetric",
                layer=True,
            )

        root = zarr.open_group(str(scene_path), mode="r")
        wrapper = root["tiles"]
        assert wrapper.attrs["kind"] == "partition"
        # The layer — and the ONLY node that may set the mode.
        assert wrapper.attrs["blending_mode"] == "volumetric"
        assert wrapper.attrs["layer"] is True

        part_names = [k for k in wrapper.group_keys() if str(k).startswith("part_")]
        assert len(part_names) >= 2, "expected a real BSP split"
        for name in part_names:
            part_attrs = dict(wrapper[name].attrs)
            # A stamped copy here is nearest-setter-wins → it shadows the
            # wrapper and the layer's Blend control stops doing anything.
            assert "blending_mode" not in part_attrs, (
                f"part {name} shadows the wrapper's blending_mode"
            )
            # `layer` likewise stays on the wrapper: the parts are internal
            # structure, not user-facing layers.
            assert "layer" not in part_attrs


def test_grafted_partition_parts_are_not_exposed_as_layers() -> None:
    """Only the wrapper is a layer, so the panel shows ONE row for the graft."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        src = _partition_file(tmpdir)
        scene_path = tmpdir / "scene.luxar.zarr"

        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            scene.add_gsplats_from_file("tiles", str(src), layer=True)

        root = zarr.open_group(str(scene_path), mode="r")

        n_layers = 0

        def _visit(group: zarr.Group) -> None:
            nonlocal n_layers
            if group.attrs.get("layer") is True:
                n_layers += 1
            for name in group.group_keys():
                _visit(group[name])

        _visit(root)
        assert n_layers == 1


# ────────────────────────────────────────────────────────────────────────
# Topology-aware coverage_fraction fallback (parity with the standalone writer)
# ────────────────────────────────────────────────────────────────────────


def _one_part_adaptive():
    """The shape ``gsplat lod --recipe adaptive`` emits below ``max_elements``:
    a ``kind=partition`` holding a SINGLE per-part lod group."""
    from luxar.gsplats.lod.recipes import RecipeParams, build_recipe
    from luxar.gsplats.tree import GSplatPartition, without_meta_key

    node = build_recipe(
        _two_clusters(60),
        "adaptive",
        # max_elements unset → the default 1,000,000, so the BSP never splits.
        RecipeParams(compression_factor=4, levels=2, device="cpu", seed=0),
    )
    assert isinstance(node, GSplatPartition) and len(node.children) == 1
    # Scrubbed, so the graft's FALLBACK derivation is what gets exercised.
    return without_meta_key(node, "coverage_fraction")


def test_grafted_one_part_partition_uses_the_whole_object_anchor() -> None:
    """A one-part partition is not a tiling, so the ladder under it keeps the
    whole-object anchor (finest 1.0) — the same rule
    ``gsplat_tree.write_gsplat_node`` applies, which is what makes a
    file → scene graft agree with a standalone rewrite."""
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            graft_gsplat_node(scene, name="adaptive", node=_one_part_adaptive())

        root = zarr.open_group(str(scene_path), mode="r")
        lod = root["adaptive"]["part_0"]
        covs = [
            float(lod[k].attrs["coverage_fraction"])
            for k in sorted(lod.group_keys(), key=lambda s: int(s.split("_")[1]))
        ]
        assert covs[0] == 0.0
        assert covs[-1] == 1.0, f"expected the whole-object anchor; got {covs}"


def test_graft_into_a_multi_part_partition_keeps_the_tile_anchor() -> None:
    """The mirror case: the SCENE insertion point is a REAL tiling (>= 2 parts),
    so the ladder is per-tile and keeps the fills-screen anchor (finest 4.0).

    Note what the grafted subtree itself is: the same ONE-part
    ``kind=partition`` as the test above, whose own part count says "not a
    tiling". An outer binding must survive that — a one-part partition nested
    inside a tile is still inside that tile — so the two rules compose rather
    than the inner one cancelling the outer.
    """
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node
    from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION

    with tempfile.TemporaryDirectory() as tmp:
        scene_path = Path(tmp) / "scene.luxar.zarr"
        with LuxarZarrCompiler(
            scene_path, encoding_mode=EncodingMode.PRECISION
        ) as compiler:
            scene = compiler.create_scene(dimensions=_scene_dims())
            tiled = scene.add_partition_group(
                "tiled", display_type="gsplats", max_elements=120
            )
            # A sibling tile FIRST, so the insertion point is a genuine tiling
            # and not the degenerate one-part shape at graft time.
            tiled.add_gsplats_from_data("part_0", _two_clusters(30))
            graft_gsplat_node(tiled, name="part_1", node=_one_part_adaptive())

        root = zarr.open_group(str(scene_path), mode="r")
        lod = root["tiled"]["part_1"]["part_0"]
        covs = [
            float(lod[k].attrs["coverage_fraction"])
            for k in sorted(lod.group_keys(), key=lambda s: int(s.split("_")[1]))
        ]
        assert covs[0] == 0.0
        assert covs[-1] == MAX_COVERAGE_FRACTION, (
            f"expected the per-tile anchor; got {covs}"
        )


# ────────────────────────────────────────────────────────────────────────
# coverage_fraction anchor for a ladder added INTO a hand-built partition
# ────────────────────────────────────────────────────────────────────────
#
# Two paths, and it matters which one a test exercises:
#
#  * ``add_gsplats_from_file`` of an ORDINARY ladder store (``levels`` / ``stream``
#    / a plain fit) is **matrix-shaped**, so ``add_gsplats_from_file_impl`` routes
#    it to ``add_gsplats_from_data_impl`` →
#    ``add_gsplats_as_lod_group_impl``, which anchors via
#    ``derive_coverage_fractions``. That is the real public per-part path, pinned
#    by ``TestAddGsplatsFromFileAnchor`` below.
#  * ``graft_gsplat_node`` only ever sees a NON-matrix-shaped subtree, and its own
#    fallback derivation is reached with no partition flag set only for a nested
#    lod-of-lods — a shape no library producer writes today. The two
#    ``test_graft_*`` functions call the internal function directly to cover that
#    defensive term; they do NOT pin the public per-part path.


def _meta_less_ladder():
    """A ``kind=lod`` ``GSplatNode`` tree whose children carry NO ``meta``.

    That is what makes a fallback derivation live at all: with an authored
    ``coverage_fraction`` in each child's ``meta`` the graft just copies it.
    Counts are ``[16, 64, 256]``, so the whole-object ladder is ``[0, 0.5, 1.0]``
    and the partition-bound one ``[0, 2.0, 4.0]``.
    """
    from luxar.gsplats.gsplat_data import AdditiveSubLOD
    from luxar.gsplats.tree import GSplatLeaf, GSplatLodGroup

    children = []
    for n in (16, 64, 256):
        rng = np.random.default_rng(n)
        chol = np.zeros((n, 6), dtype=np.float32)
        chol[:, [0, 2, 5]] = 1.0
        children.append(
            GSplatLeaf(
                additive_sublods=[
                    AdditiveSubLOD(
                        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
                        amplitudes=np.ones(n, dtype=np.float32),
                        cholesky_factors=chol,
                    )
                ]
            )
        )
    for leaf in children:
        assert not leaf.meta, "the fixture must carry no authored coverage_fraction"
    return GSplatLodGroup(children=children)


def _ladder_child_coverage(lod_group) -> list:
    """Per-child ``coverage_fraction`` off a written ``kind=lod`` group."""
    assert lod_group.attrs["kind"] == "lod"
    return [float(lod_group[f"child_{i}"].attrs["coverage_fraction"]) for i in range(3)]


def _graft_ladder_coverage(scene_path: Path, *, partitioned: bool) -> list:
    """Graft the meta-less ladder at the scene root or inside a partition wrapper.

    Calls ``graft_gsplat_node`` DIRECTLY: this tree is matrix-shaped, so the
    public ``add_gsplats_from_file`` would never route it here.
    """
    from luxar.core.group.gsplats_pipeline.from_io import graft_gsplat_node

    with LuxarZarrCompiler(
        scene_path, encoding_mode=EncodingMode.PRECISION
    ) as compiler:
        scene = compiler.create_scene(dimensions=_scene_dims())
        if partitioned:
            target = scene.add_partition_group(
                "tiled", display_type="gsplats", max_elements=256
            )
        else:
            target = scene
        graft_gsplat_node(target, name="part_0", node=_meta_less_ladder())

    root = zarr.open_group(str(scene_path), mode="r")
    return _ladder_child_coverage((root["tiled"] if partitioned else root)["part_0"])


def test_graft_fallback_sees_a_scene_side_partition() -> None:
    """The graft's fallback anchor also consults the SCENE-graph insertion point.

    ``_under_partition`` is threaded down the ``GSplatNode`` tree being walked, so
    a hand-built ``kind=partition`` wrapper in the SCENE would be invisible to it.
    The entry call therefore SEEDS the flag from `is_partition_bound(parent or
    group)` — once, before any wrapper of its own exists.

    SCOPE: this is a DEFENSIVE term, and this test reaches it only by calling
    ``graft_gsplat_node`` directly. The everyday
    ``wrapper.add_gsplats_from_file(...)``-per-part case is matrix-shaped and is
    anchored by ``add_gsplats_as_lod_group_impl`` instead — see
    ``TestAddGsplatsFromFileAnchor``. The shape that genuinely lands here with no
    flag set is a nested lod-of-lods, which no library producer emits today.
    """
    from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION

    with tempfile.TemporaryDirectory() as tmp:
        cov = _graft_ladder_coverage(Path(tmp) / "tiled.luxar.zarr", partitioned=True)
    assert cov == [0.0, 2.0, MAX_COVERAGE_FRACTION]


def test_graft_fallback_at_the_scene_root_keeps_the_whole_object_anchor() -> None:
    """CONTROL (passes pre-fix): no partition ancestor → the finest stays 1.0."""
    with tempfile.TemporaryDirectory() as tmp:
        cov = _graft_ladder_coverage(Path(tmp) / "root.luxar.zarr", partitioned=False)
    assert cov == [0.0, 0.5, 1.0]


def _ladder_file(dirpath: Path, n: int, seed: int) -> Path:
    """A standalone matrix-shaped ``kind=lod`` ``.gsplats.zarr`` (``levels`` output).

    Tiny on purpose (<= 64 splats, 3 levels): the assertions read per-level counts
    only, so reduction quality is irrelevant.
    """
    from luxar.gsplats.lod import make_substitutive_lod
    from luxar.gsplats.tree import is_matrix_shaped

    rng = np.random.default_rng(seed)
    chol = np.zeros((n, 6), dtype=np.float32)
    chol[:, [0, 2, 5]] = 1.0
    data = GSplatData(
        centers=rng.uniform(0, 10, (n, 3)).astype(np.float32),
        amplitudes=np.ones(n, dtype=np.float32),
        cholesky_factors=chol,
    )
    pyramid = make_substitutive_lod(data, compression_factor=4, levels=2, device="cpu")
    assert is_matrix_shaped(pyramid.tree), (
        "this fixture must stay matrix-shaped — that is what makes it exercise the "
        "add_gsplats_as_lod_group_impl path rather than graft_gsplat_node"
    )
    path = dirpath / f"ladder_{seed}.gsplats.zarr"
    write_gsplats_tree(
        path, pyramid.tree, ordering="none", encoding_mode=EncodingMode.PRECISION
    )
    return path


#: Two per-tile ladder files, so the fixture is a REAL 2-tile partition rather than
#: the degenerate 1-part shape ``partitioned_coverage_fractions`` documents as 4x
#: too coarse. Both are (n, seed) pairs whose counts give a 3-level ladder.
_TILE_SPECS = ((64, 0), (32, 1))


def _from_file_coverage(dirpath: Path, store_name: str, *, partitioned: bool) -> list:
    """One ``add_gsplats_from_file`` per tile; returns a coverage list per part."""
    srcs = [_ladder_file(dirpath, n, seed) for n, seed in _TILE_SPECS]
    scene_path = dirpath / store_name
    with LuxarZarrCompiler(
        scene_path, encoding_mode=EncodingMode.PRECISION
    ) as compiler:
        scene = compiler.create_scene(dimensions=_scene_dims())
        target = (
            scene.add_partition_group("tiled", display_type="gsplats", max_elements=64)
            if partitioned
            else scene
        )
        for i, src in enumerate(srcs):
            target.add_gsplats_from_file(f"part_{i}", str(src))

    root = zarr.open_group(str(scene_path), mode="r")
    node = root["tiled"] if partitioned else root
    return [_ladder_child_coverage(node[f"part_{i}"]) for i in range(len(srcs))]


class TestAddGsplatsFromFileAnchor:
    """The REAL public per-part path: one ``add_gsplats_from_file`` per tile.

    A ladder store is matrix-shaped, so this goes through
    ``add_gsplats_from_data_impl`` → ``add_gsplats_as_lod_group_impl`` and is
    anchored by ``derive_coverage_fractions`` — the round-1 fix. This is what a
    caller actually writes when hand-building a ``kind=partition`` of per-tile
    ladders from files, so the fixture is a real TWO-tile partition and both tiles
    are asserted.
    """

    def test_under_a_hand_built_partition_uses_the_tile_anchor(self) -> None:
        """REGRESSION (#1411): fails pre-fix with the finest at 1.0."""
        from luxar.core.group.lod.group import MAX_COVERAGE_FRACTION

        with tempfile.TemporaryDirectory() as tmp:
            parts = _from_file_coverage(Path(tmp), "tiled.luxar.zarr", partitioned=True)
        assert len(parts) == 2, "must be a real 2-tile partition"
        for i, cov in enumerate(parts):
            assert cov == [0.0, 2.0, MAX_COVERAGE_FRACTION], f"part_{i}"

    def test_at_the_scene_root_keeps_the_whole_object_anchor(self) -> None:
        """CONTROL (passes pre-fix): the over-trigger guard."""
        with tempfile.TemporaryDirectory() as tmp:
            parts = _from_file_coverage(Path(tmp), "root.luxar.zarr", partitioned=False)
        assert len(parts) == 2
        for i, cov in enumerate(parts):
            assert cov == [0.0, 0.5, 1.0], f"part_{i}"
